from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from simpipe.config.hardware import HardwareConfig
from simpipe.config.model import ModelConfig
from simpipe.config.parallel import ParallelConfig
from simpipe.graph.operator import OpType, Operator
from simpipe.graph.model_graph import EMBEDDING_LAYER_IDX, ModelGraph, head_layer_idx
from simpipe.graph.tensor import TensorKind, tensor_bytes
from simpipe.memory.zero import ZeroMemoryShard, zero_model_state_bytes
from simpipe.models.pattern import ATTN, MAMBA, MLP, MOE, TRANSFORMER, stack_layer_symbols
from simpipe.pipeline.types import WorkloadPlan


_FLASH_ATTENTION_SAVED_ACTIVATION_SCALE = 0.5


@dataclass(frozen=True)
class LayerParameterCount:
    non_expert: int = 0
    expert: int = 0

    @property
    def total(self) -> int:
        return self.non_expert + self.expert


@dataclass(frozen=True)
class ModelParameterSpec:
    embedding: LayerParameterCount
    layers: tuple[LayerParameterCount, ...]
    head: LayerParameterCount
    weight_dtype_bytes: float = 2.0
    expert_weight_dtype_bytes: float = 2.0


@dataclass(frozen=True)
class StageMemoryEstimate:
    stage_id: int
    parameter_count: int
    parameter_bytes: int
    gradient_bytes: int
    master_parameter_bytes: int
    optimizer_moment_bytes: int
    optimizer_bytes: int
    activation_per_microbatch_bytes: int
    p2p_buffer_bytes: int

    @property
    def model_state_bytes(self) -> int:
        return (
            self.parameter_bytes
            + self.gradient_bytes
            + self.master_parameter_bytes
            + self.optimizer_moment_bytes
        )


@dataclass(frozen=True)
class DeviceMemoryEstimate:
    did: int
    stage_ids: tuple[int, ...]
    parameter_bytes: int
    gradient_bytes: int
    master_parameter_bytes: int
    optimizer_moment_bytes: int
    optimizer_bytes: int
    activation_peak_bytes: int
    p2p_buffer_bytes: int
    hbm_bytes: int
    stages: tuple[StageMemoryEstimate, ...] = field(default_factory=tuple)

    @property
    def model_state_bytes(self) -> int:
        return (
            self.parameter_bytes
            + self.gradient_bytes
            + self.master_parameter_bytes
            + self.optimizer_moment_bytes
        )

    @property
    def peak_bytes(self) -> int:
        return self.model_state_bytes + self.activation_peak_bytes + self.p2p_buffer_bytes

    @property
    def feasible(self) -> bool:
        return self.peak_bytes <= self.hbm_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "did": self.did,
            "pp_rank": self.did,
            "stage_ids": list(self.stage_ids),
            "parameter_bytes": self.parameter_bytes,
            "gradient_bytes": self.gradient_bytes,
            "master_parameter_bytes": self.master_parameter_bytes,
            "optimizer_moment_bytes": self.optimizer_moment_bytes,
            "optimizer_bytes": self.optimizer_bytes,
            "model_state_bytes": self.model_state_bytes,
            "activation_peak_bytes": self.activation_peak_bytes,
            "p2p_buffer_bytes": self.p2p_buffer_bytes,
            "peak_bytes": self.peak_bytes,
            "hbm_bytes": self.hbm_bytes,
            "feasible": self.feasible,
            "parameter_gb": _gb(self.parameter_bytes),
            "gradient_gb": _gb(self.gradient_bytes),
            "master_parameter_gb": _gb(self.master_parameter_bytes),
            "optimizer_moment_gb": _gb(self.optimizer_moment_bytes),
            "optimizer_gb": _gb(self.optimizer_bytes),
            "activation_peak_gb": _gb(self.activation_peak_bytes),
            "p2p_buffer_gb": _gb(self.p2p_buffer_bytes),
            "peak_gb": _gb(self.peak_bytes),
            "hbm_gb": _gb(self.hbm_bytes),
        }


@dataclass(frozen=True)
class PipelineMemoryEstimate:
    per_device: tuple[DeviceMemoryEstimate, ...]
    zero_stage: int
    total_parameter_count: int

    @property
    def peak_bytes(self) -> int:
        return max((device.peak_bytes for device in self.per_device), default=0)

    @property
    def feasible(self) -> bool:
        return all(device.feasible for device in self.per_device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "zero_stage": self.zero_stage,
            "total_parameter_count": self.total_parameter_count,
            "peak_bytes": self.peak_bytes,
            "peak_gb": _gb(self.peak_bytes),
            "feasible": self.feasible,
            "per_device": [device.to_dict() for device in self.per_device],
        }


def estimate_pipeline_memory(
    graph: ModelGraph,
    plan: WorkloadPlan,
    parallel: ParallelConfig,
    hardware: HardwareConfig,
    records: list[dict] | None = None,
) -> PipelineMemoryEstimate:
    spec = model_parameter_spec(graph.model, graph)
    stage_estimates = _stage_memory_estimates(graph, plan, parallel, spec)
    activation_peak = _activation_peak_by_device(graph, plan, parallel, records, stage_estimates)
    p2p_by_stage = {stage.stage_id: stage.p2p_buffer_bytes for stage in stage_estimates}
    stage_by_id = {stage.stage_id: stage for stage in stage_estimates}
    devices: list[DeviceMemoryEstimate] = []

    hbm_bytes = hardware.gpu_hbm_bytes
    for did, stage_ids in enumerate(plan.placement.device_stages):
        stages = tuple(stage_by_id[sid] for sid in stage_ids)
        devices.append(
            DeviceMemoryEstimate(
                did=did,
                stage_ids=tuple(stage_ids),
                parameter_bytes=sum(stage.parameter_bytes for stage in stages),
                gradient_bytes=sum(stage.gradient_bytes for stage in stages),
                master_parameter_bytes=sum(stage.master_parameter_bytes for stage in stages),
                optimizer_moment_bytes=sum(stage.optimizer_moment_bytes for stage in stages),
                optimizer_bytes=sum(stage.optimizer_bytes for stage in stages),
                activation_peak_bytes=activation_peak.get(did, 0),
                p2p_buffer_bytes=sum(p2p_by_stage.get(sid, 0) for sid in stage_ids),
                hbm_bytes=hbm_bytes,
                stages=stages,
            )
        )

    return PipelineMemoryEstimate(
        per_device=tuple(devices),
        zero_stage=parallel.zero_stage,
        total_parameter_count=(
            spec.embedding.total + spec.head.total + sum(layer.total for layer in spec.layers)
        ),
    )


def model_parameter_spec(model: ModelConfig, graph: ModelGraph | None = None) -> ModelParameterSpec:
    if model.hf_config_path:
        return _hf_parameter_spec(model.hf_config_path, model)
    if graph is None:
        graph = ModelGraph.from_config(model)
    by_layer: dict[int, LayerParameterCount] = {}
    for op in graph.operators:
        if op.layer_idx is None:
            continue
        current = by_layer.get(op.layer_idx, LayerParameterCount())
        by_layer[op.layer_idx] = LayerParameterCount(
            current.non_expert + op.parameter_count,
            current.expert + op.expert_parameter_count,
        )
    layers = tuple(by_layer.get(idx, LayerParameterCount()) for idx in range(model.num_layers))
    return ModelParameterSpec(
        embedding=by_layer.get(EMBEDDING_LAYER_IDX, LayerParameterCount()),
        layers=layers,
        head=by_layer.get(head_layer_idx(model.num_layers), LayerParameterCount()),
    )


def _stage_memory_estimates(
    graph: ModelGraph,
    plan: WorkloadPlan,
    parallel: ParallelConfig,
    spec: ModelParameterSpec,
) -> tuple[StageMemoryEstimate, ...]:
    estimates: list[StageMemoryEstimate] = []
    for stage in plan.partition.stages:
        param_count = _stage_parameter_count(graph, stage.operator_ids, spec)
        sharded = _local_model_state_bytes(graph, stage.operator_ids, spec, parallel)
        estimates.append(
            StageMemoryEstimate(
                stage_id=stage.stage_id,
                parameter_count=param_count,
                parameter_bytes=sharded.parameter_bytes,
                gradient_bytes=sharded.gradient_bytes,
                master_parameter_bytes=sharded.master_parameter_bytes,
                optimizer_moment_bytes=sharded.optimizer_bytes,
                optimizer_bytes=sharded.master_parameter_bytes + sharded.optimizer_bytes,
                activation_per_microbatch_bytes=_stage_activation_bytes(graph, stage.operator_ids, parallel),
                p2p_buffer_bytes=_stage_p2p_buffer_bytes(graph, stage.p2p_output_id, parallel),
            )
        )
    return tuple(estimates)


def _stage_parameter_count(
    graph: ModelGraph,
    operator_ids: list[str],
    spec: ModelParameterSpec,
) -> int:
    return sum(layer.total for layer in _stage_layer_counts(graph, operator_ids, spec))


def _local_model_state_bytes(
    graph: ModelGraph,
    operator_ids: list[str],
    spec: ModelParameterSpec,
    parallel: ParallelConfig,
) -> ZeroMemoryShard:
    tp = max(1, parallel.tp_size)
    ep = max(1, parallel.ep_size)
    non_expert = 0
    expert = 0
    for layer in _stage_layer_counts(graph, operator_ids, spec):
        non_expert += layer.non_expert
        expert += layer.expert
    local_non_expert = non_expert / tp
    local_expert = expert / (tp * ep)
    dense_state = _state_bytes_for_param_count(
        local_non_expert,
        spec.weight_dtype_bytes,
        parallel,
    )
    expert_parallel = replace(parallel, dp_size=_expert_data_parallel_size(parallel))
    expert_state = _state_bytes_for_param_count(
        local_expert,
        spec.expert_weight_dtype_bytes,
        expert_parallel,
    )
    return ZeroMemoryShard(
        parameter_bytes=dense_state.parameter_bytes + expert_state.parameter_bytes,
        gradient_bytes=dense_state.gradient_bytes + expert_state.gradient_bytes,
        master_parameter_bytes=dense_state.master_parameter_bytes + expert_state.master_parameter_bytes,
        optimizer_bytes=dense_state.optimizer_bytes + expert_state.optimizer_bytes,
    )


def _state_bytes_for_param_count(
    param_count: float,
    weight_dtype_bytes: float,
    parallel: ParallelConfig,
) -> ZeroMemoryShard:
    parameter_bytes = int(param_count * weight_dtype_bytes)
    gradient_bytes = int(param_count * 4.0) if parallel.grad_reduce_in_fp32 else parameter_bytes
    master_parameter_bytes = int(param_count * 4.0)
    optimizer_bytes = int(param_count * 8.0)
    return zero_model_state_bytes(
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        master_parameter_bytes=master_parameter_bytes,
        optimizer_bytes=optimizer_bytes,
        parallel=parallel,
    )


def _expert_data_parallel_size(parallel: ParallelConfig) -> int:
    dp = max(1, parallel.dp_size)
    ep = max(1, parallel.ep_size)
    if ep == 1:
        return dp
    if dp % ep != 0:
        raise ValueError(f"dp_size must be divisible by ep_size for EP, got dp={dp}, ep={ep}")
    return max(1, dp // ep)


def _stage_layer_counts(
    graph: ModelGraph,
    operator_ids: list[str],
    spec: ModelParameterSpec,
) -> list[LayerParameterCount]:
    layers: list[LayerParameterCount] = []
    seen: set[int] = set()
    for oid in operator_ids:
        layer_idx = graph.op_by_id(oid).layer_idx
        if layer_idx is None or layer_idx in seen:
            continue
        seen.add(layer_idx)
        if layer_idx == EMBEDDING_LAYER_IDX:
            layers.append(spec.embedding)
        elif layer_idx == head_layer_idx(graph.model.num_layers):
            layers.append(spec.head)
        elif 0 <= layer_idx < len(spec.layers):
            layers.append(spec.layers[layer_idx])
    return layers


def _stage_activation_bytes(
    graph: ModelGraph,
    operator_ids: list[str],
    parallel: ParallelConfig,
) -> int:
    tp = max(1, parallel.tp_size)
    total = 0
    for oid in operator_ids:
        op = graph.op_by_id(oid)
        op_bytes = sum(tensor_bytes(t) for t in op.outputs if t.kind == TensorKind.ACTIVATION)
        if graph.model.flash_attention and _is_attention_saved_activation(op):
            op_bytes = int(op_bytes * _FLASH_ATTENTION_SAVED_ACTIVATION_SCALE)
        if op.op_type == OpType.HEAD:
            op_bytes += _cross_entropy_temp_buffer_bytes(graph.model, parallel)
        total += op_bytes
    return total // tp


def _is_attention_saved_activation(op: Operator) -> bool:
    if op.op_type == OpType.ATTENTION:
        return True
    return op.op_type == OpType.MATMUL and op.id.endswith("_qkv")


def _cross_entropy_temp_buffer_bytes(model: ModelConfig, parallel: ParallelConfig) -> int:
    """Unsharded FP32 logits temp in Megatron tensor-parallel cross entropy."""
    del parallel
    return int(model.micro_batch_size * model.seq_len * model.vocab_size * 4)


def _stage_p2p_buffer_bytes(
    graph: ModelGraph,
    op_id: str | None,
    parallel: ParallelConfig,
) -> int:
    if op_id is None:
        return 0
    op = graph.op_by_id(op_id)
    tp = max(1, parallel.tp_size)
    return sum(tensor_bytes(t) for t in op.outputs if t.kind == TensorKind.ACTIVATION) // tp


def _activation_peak_by_device(
    graph: ModelGraph,
    plan: WorkloadPlan,
    parallel: ParallelConfig,
    records: list[dict] | None,
    stages: tuple[StageMemoryEstimate, ...],
) -> dict[int, int]:
    stage_bytes = {stage.stage_id: stage.activation_per_microbatch_bytes for stage in stages}
    if not records:
        # Static fallback (one in-flight microbatch per stage): size it for
        # the largest microbatch when a variable-length batch spec is set.
        worst_ratio = (
            max(lin for lin, _quad in plan.mid_scales) if plan.mid_scales else 1.0
        )
        return {
            did: int(sum(stage_bytes.get(sid, 0) for sid in stage_ids) * worst_ratio)
            for did, stage_ids in enumerate(plan.placement.device_stages)
        }

    sid_to_did = {
        sid: did
        for did, stage_ids in enumerate(plan.placement.device_stages)
        for sid in stage_ids
    }
    events: dict[int, list[tuple[float, float]]] = {}
    for record in records:
        wtype = str(record.get("wtype", "")).upper()
        sid = int(record.get("sid", 0))
        did = sid_to_did.get(sid, int(record.get("did", 0)))
        # Activation bytes are token-linear: weight each microbatch by its
        # token count relative to the profiled shape.
        size = stage_bytes.get(sid, 0) * plan.token_ratio_for_mid(int(record.get("mid", 0)))
        if size <= 0:
            continue
        if wtype == "F":
            t = float(record.get("end") or record.get("start") or 0)
            events.setdefault(did, []).append((t, size))
        elif wtype == "B":
            t = float(record.get("end") or record.get("start") or 0)
            events.setdefault(did, []).append((t, -size))

    peaks: dict[int, int] = {did: 0 for did in range(plan.device_num)}
    for did, device_events in events.items():
        live = 0.0
        for _time, delta in sorted(device_events, key=lambda item: (item[0], -item[1])):
            live = max(0.0, live + delta)
            peaks[did] = max(peaks[did], int(live))
    return peaks


def _hf_parameter_spec(path: str, model: ModelConfig) -> ModelParameterSpec:
    with Path(path).expanduser().open() as f:
        data = json.load(f)
    hidden = int(data.get("hidden_size", model.hidden_size))
    layers = int(data.get("num_hidden_layers", model.num_layers))
    vocab = int(data.get("vocab_size", model.vocab_size))
    heads = int(data.get("num_attention_heads", model.num_attention_heads))
    head_dim = int(data.get("head_dim", max(1, hidden // max(1, heads))))
    kv_heads = int(data.get("num_key_value_heads", heads))
    rope_dim = int(data.get("qk_rope_head_dim", 0))
    inter = int(data.get("intermediate_size") or data.get("moe_intermediate_size") or hidden * 4)
    routed_experts = int(data.get("n_routed_experts", 0))
    shared_experts = int(data.get("n_shared_experts", 0))

    if data.get("hybrid_override_pattern"):
        symbols = stack_layer_symbols(data["hybrid_override_pattern"])
        if len(symbols) != layers:
            raise ValueError(
                f"hybrid_override_pattern has {len(symbols)} layers, expected {layers}"
            )
        layer_counts = tuple(
            _hybrid_layer_parameter_count(symbol, data, hidden, heads, head_dim, kv_heads, rope_dim, inter)
            for symbol in symbols
        )
    elif routed_experts:
        router_params = hidden * routed_experts
        shared_params = shared_experts * 3 * hidden * inter
        expert_params = routed_experts * 3 * hidden * inter
        per_layer = LayerParameterCount(
            non_expert=(
                _attention_parameter_count(data, hidden, heads, head_dim, kv_heads, rope_dim)
                + _norm_parameter_count(hidden, 2)
                + router_params
                + shared_params
            ),
            expert=expert_params,
        )
        layer_counts = tuple(per_layer for _ in range(layers))
    else:
        per_layer = LayerParameterCount(
            non_expert=(
                _attention_parameter_count(data, hidden, heads, head_dim, kv_heads, rope_dim)
                + _norm_parameter_count(hidden, 2)
                + 3 * hidden * inter
            )
        )
        layer_counts = tuple(per_layer for _ in range(layers))

    weight_dtype = _dtype_bytes(
        ((data.get("quantization_config") or {}).get("fmt"))
        or ((data.get("quantization_config") or {}).get("weight_dtype"))
        or data.get("torch_dtype")
    )
    expert_dtype = _dtype_bytes(data.get("expert_dtype"), default=weight_dtype)
    return ModelParameterSpec(
        embedding=LayerParameterCount(non_expert=vocab * hidden),
        layers=layer_counts,
        head=LayerParameterCount(non_expert=0 if data.get("tie_word_embeddings") else vocab * hidden),
        weight_dtype_bytes=weight_dtype,
        expert_weight_dtype_bytes=expert_dtype,
    )


def _hybrid_layer_parameter_count(
    symbol: str,
    data: dict,
    hidden: int,
    heads: int,
    head_dim: int,
    kv_heads: int,
    rope_dim: int,
    inter: int,
) -> LayerParameterCount:
    if symbol == MAMBA:
        return LayerParameterCount(
            non_expert=_mamba_parameter_count(data, hidden) + _norm_parameter_count(hidden, 1)
        )
    if symbol == MLP:
        # Nemotron-H uses relu2-style dense MLP layers in the hybrid stack: up + down.
        return LayerParameterCount(non_expert=2 * hidden * inter + _norm_parameter_count(hidden, 1))
    if symbol == ATTN:
        return LayerParameterCount(
            non_expert=(
                _attention_parameter_count(data, hidden, heads, head_dim, kv_heads, rope_dim)
                + _norm_parameter_count(hidden, 1)
            )
        )
    if symbol == TRANSFORMER:
        return LayerParameterCount(
            non_expert=(
                _attention_parameter_count(data, hidden, heads, head_dim, kv_heads, rope_dim)
                + _norm_parameter_count(hidden, 2)
                + 3 * hidden * inter
            )
        )
    if symbol == MOE:
        routed_experts = int(
            data.get("n_routed_experts")
            or data.get("num_experts")
            or data.get("n_experts")
            or 0
        )
        shared_experts = int(data.get("n_shared_experts", 0) or 0)
        return LayerParameterCount(
            non_expert=(
                _attention_parameter_count(data, hidden, heads, head_dim, kv_heads, rope_dim)
                + _norm_parameter_count(hidden, 2)
                + hidden * routed_experts
                + shared_experts * 3 * hidden * inter
            ),
            expert=routed_experts * 3 * hidden * inter,
        )
    raise ValueError(f"Unsupported hybrid layer symbol: {symbol!r}")


def _attention_parameter_count(
    data: dict,
    hidden: int,
    heads: int,
    head_dim: int,
    kv_heads: int,
    rope_dim: int,
) -> int:
    q_lora_rank = int(data.get("q_lora_rank", 0) or 0)
    if q_lora_rank:
        q_params = hidden * q_lora_rank + q_lora_rank * heads * (head_dim + rope_dim)
    else:
        q_params = hidden * heads * head_dim
    kv_params = hidden * kv_heads * (head_dim + rope_dim) * 2
    o_params = heads * head_dim * hidden
    return q_params + kv_params + o_params


def _mamba_parameter_count(data: dict, hidden: int) -> int:
    mamba_heads = int(data.get("mamba_num_heads") or data.get("num_attention_heads") or 1)
    mamba_head_dim = int(data.get("mamba_head_dim") or data.get("head_dim") or max(1, hidden // mamba_heads))
    inner = int(data.get("mamba_inner_size") or mamba_heads * mamba_head_dim)
    groups = int(data.get("mamba_num_groups") or data.get("n_groups") or 1)
    state = int(data.get("mamba_state_dim") or data.get("ssm_state_size") or 16)
    conv_kernel = int(data.get("conv_kernel", 4) or 4)
    conv_channels = inner + 2 * groups * state
    in_proj_out = 2 * inner + 2 * groups * state + mamba_heads
    params = hidden * in_proj_out + inner * hidden + conv_channels * conv_kernel
    if data.get("use_conv_bias", False):
        params += conv_channels
    params += mamba_heads  # dt bias / recurrent scalars
    return params


def _norm_parameter_count(hidden: int, num_norms: int) -> int:
    return hidden * num_norms


def _dtype_bytes(name: object, default: float = 2.0) -> float:
    if name is None:
        return default
    key = str(name).lower()
    if key in {"float32", "fp32", "torch.float32"}:
        return 4.0
    if key in {"float16", "fp16", "bfloat16", "bf16", "torch.float16", "torch.bfloat16"}:
        return 2.0
    if key in {"float8", "fp8", "e4m3", "e5m2"}:
        return 1.0
    if key in {"float4", "fp4", "e2m1"}:
        return 0.5
    return default


def _gb(value: int) -> float:
    return round(value / 1024**3, 3)
