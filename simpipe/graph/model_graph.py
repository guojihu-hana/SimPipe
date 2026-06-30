from __future__ import annotations

from dataclasses import dataclass, field

from simpipe.config.model import ModelConfig
from simpipe.graph.layer_template import LayerTemplate, MoELayerTemplate, TransformerBlockTemplate
from simpipe.graph.operator import OpType, Operator
from simpipe.graph.tensor import TensorKind, TensorSpec

EMBEDDING_LAYER_IDX = -1


def head_layer_idx(num_layers: int) -> int:
    return num_layers


def build_embedding_operators(model: ModelConfig) -> list[Operator]:
    b, s, h, v = model.micro_batch_size, model.seq_len, model.hidden_size, model.vocab_size
    token_ids = TensorSpec("token_ids", (b, s), kind=TensorKind.ACTIVATION)
    embedded = TensorSpec("embedded", (b, s, h), kind=TensorKind.ACTIVATION)
    return [
        Operator(
            "embedding",
            OpType.EMBEDDING,
            [token_ids],
            [embedded],
            flops=2 * b * s * v * h,
            layer_idx=EMBEDDING_LAYER_IDX,
        )
    ]


def build_head_operators(model: ModelConfig) -> list[Operator]:
    b, s, h, v = model.micro_batch_size, model.seq_len, model.hidden_size, model.vocab_size
    hidden = TensorSpec("head_hidden", (b, s, h), kind=TensorKind.ACTIVATION)
    logits = TensorSpec("logits", (b, s, v), kind=TensorKind.ACTIVATION)
    return [
        Operator(
            "head",
            OpType.HEAD,
            [hidden],
            [logits],
            flops=2 * b * s * h * v,
            layer_idx=head_layer_idx(model.num_layers),
        )
    ]


@dataclass
class ModelGraph:
    model: ModelConfig
    operators: list[Operator] = field(default_factory=list)
    layer_template: LayerTemplate | None = None

    @classmethod
    def from_config(cls, model: ModelConfig) -> ModelGraph:
        if model.use_moe:
            template: LayerTemplate = MoELayerTemplate()
        else:
            template = TransformerBlockTemplate()
        ops: list[Operator] = []
        ops.extend(build_embedding_operators(model))
        for layer_idx in range(model.num_layers):
            ops.extend(template.build_operators(layer_idx, model))
        ops.extend(build_head_operators(model))
        return cls(model=model, operators=ops, layer_template=template)

    def op_by_id(self, op_id: str) -> Operator:
        for op in self.operators:
            if op.id == op_id:
                return op
        raise KeyError(op_id)

    def op_ids(self) -> list[str]:
        return [op.id for op in self.operators]

    def ops_for_layers(self, layer_indices: list[int]) -> list[Operator]:
        return [op for op in self.operators if op.layer_idx in layer_indices]

    def distribute_profile_times(
        self,
        layer_f_times: list[float],
        layer_b_times: list[float] | None = None,
        layer_w_times: list[float] | None = None,
        *,
        embedding_f_time: float | None = None,
        embedding_b_time: float | None = None,
        embedding_w_time: float | None = None,
        head_f_time: float | None = None,
        head_b_time: float | None = None,
        head_w_time: float | None = None,
    ) -> None:
        """Assign per-layer profile times to operators proportional to flops."""
        layer_b_times = layer_b_times or layer_f_times
        layer_w_times = layer_w_times or [0.0] * len(layer_f_times)
        by_layer: dict[int, list[Operator]] = {}
        for op in self.operators:
            if op.layer_idx is not None:
                by_layer.setdefault(op.layer_idx, []).append(op)
        for layer_idx, ops in by_layer.items():
            if 0 <= layer_idx < len(layer_f_times):
                total_flops = sum(op.flops or 1 for op in ops)
                for op in ops:
                    share = (op.flops or 1) / total_flops
                    op.profiled_time_us = layer_f_times[layer_idx] * share
                    op.profiled_bwd_time_us = layer_b_times[layer_idx] * share
                    op.profiled_w_time_us = layer_w_times[layer_idx] * share

        special_times = {
            EMBEDDING_LAYER_IDX: (embedding_f_time, embedding_b_time, embedding_w_time),
            head_layer_idx(self.model.num_layers): (head_f_time, head_b_time, head_w_time),
        }
        for layer_idx, (f_time, b_time, w_time) in special_times.items():
            if f_time is None:
                continue
            for op in by_layer.get(layer_idx, []):
                op.profiled_time_us = f_time
                op.profiled_bwd_time_us = b_time if b_time is not None else f_time
                op.profiled_w_time_us = w_time if w_time is not None else 0.0

    def stage_forward_time(self, op_ids: list[str], **kwargs) -> float:
        return sum(self.op_by_id(oid).forward_time(**kwargs) for oid in op_ids)

    def stage_backward_time(self, op_ids: list[str], **kwargs) -> float:
        return sum(self.op_by_id(oid).backward_time(**kwargs) for oid in op_ids)

    def stage_weight_time(self, op_ids: list[str], **kwargs) -> float:
        return sum(self.op_by_id(oid).weight_time(**kwargs) for oid in op_ids)
