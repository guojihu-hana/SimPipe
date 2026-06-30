from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from simpipe.config.model import ModelConfig
from simpipe.graph.operator import OpType, Operator
from simpipe.graph.tensor import DType, TensorKind, TensorSpec


@dataclass
class LayerTemplate(ABC):
    name: str

    @abstractmethod
    def build_operators(self, layer_idx: int, model: ModelConfig) -> list[Operator]:
        ...


class TransformerBlockTemplate(LayerTemplate):
    def __init__(self, name: str = "transformer") -> None:
        self.name = name

    def build_operators(self, layer_idx: int, model: ModelConfig) -> list[Operator]:
        b, s, h = model.micro_batch_size, model.seq_len, model.hidden_size
        inter = model.intermediate_size
        nh, hd = model.num_attention_heads, model.head_dim
        prefix = f"L{layer_idx}"
        hidden = TensorSpec(f"{prefix}_hidden", (b, s, h), kind=TensorKind.ACTIVATION)
        qkv_out = TensorSpec(f"{prefix}_qkv", (b, s, 3 * h), kind=TensorKind.ACTIVATION)
        attn_out = TensorSpec(f"{prefix}_attn", (b, s, h), kind=TensorKind.ACTIVATION)
        mlp_out = TensorSpec(f"{prefix}_mlp", (b, s, h), kind=TensorKind.ACTIVATION)
        qkv_flops = 2 * b * s * h * (3 * h)
        proj_flops = 2 * b * s * h * h
        attn_flops = 2 * b * nh * s * s * hd * 2
        mlp_flops = 2 * b * s * h * inter * 3
        return [
            Operator(f"{prefix}_ln0", OpType.LAYERNORM, [hidden], [hidden], flops=b * s * h, layer_idx=layer_idx),
            Operator(
                f"{prefix}_qkv",
                OpType.MATMUL,
                [hidden],
                [qkv_out],
                flops=qkv_flops,
                layer_idx=layer_idx,
            ),
            Operator(
                f"{prefix}_attn",
                OpType.ATTENTION,
                [qkv_out],
                [attn_out],
                flops=attn_flops,
                layer_idx=layer_idx,
            ),
            Operator(
                f"{prefix}_proj",
                OpType.MATMUL,
                [attn_out],
                [hidden],
                flops=proj_flops,
                layer_idx=layer_idx,
            ),
            Operator(f"{prefix}_ln1", OpType.LAYERNORM, [hidden], [hidden], flops=b * s * h, layer_idx=layer_idx),
            Operator(
                f"{prefix}_mlp_gate",
                OpType.MLP_GATE,
                [hidden],
                [TensorSpec(f"{prefix}_gate", (b, s, inter))],
                flops=2 * b * s * h * inter,
                layer_idx=layer_idx,
            ),
            Operator(
                f"{prefix}_mlp_up",
                OpType.MLP_UP,
                [hidden],
                [TensorSpec(f"{prefix}_up", (b, s, inter))],
                flops=2 * b * s * h * inter,
                layer_idx=layer_idx,
            ),
            Operator(
                f"{prefix}_mlp_down",
                OpType.MLP_DOWN,
                [TensorSpec(f"{prefix}_gate", (b, s, inter))],
                [mlp_out],
                flops=2 * b * s * inter * h,
                layer_idx=layer_idx,
            ),
        ]


class MoELayerTemplate(LayerTemplate):
    def __init__(self, name: str = "moe") -> None:
        self.name = name

    def build_operators(self, layer_idx: int, model: ModelConfig) -> list[Operator]:
        base_ops = TransformerBlockTemplate("moe_base").build_operators(layer_idx, model)
        b, s, h = model.micro_batch_size, model.seq_len, model.hidden_size
        prefix = f"L{layer_idx}"
        hidden = TensorSpec(f"{prefix}_hidden", (b, s, h))
        router_out = TensorSpec(f"{prefix}_router", (b, s, model.num_experts))
        ops: list[Operator] = base_ops[:5]  # LN + attn block
        ops.append(
            Operator(
                f"{prefix}_router",
                OpType.ROUTER,
                [hidden],
                [router_out],
                flops=b * s * h * model.num_experts,
                layer_idx=layer_idx,
            )
        )
        ops.append(
            Operator(
                f"{prefix}_dispatch_a2a",
                OpType.COMM_A2A,
                [hidden],
                [TensorSpec(f"{prefix}_dispatched", (b * model.top_k, s, h))],
                layer_idx=layer_idx,
            )
        )
        for e in range(min(model.num_experts, 4)):  # cap for simulation
            ops.append(
                Operator(
                    f"{prefix}_expert_{e}",
                    OpType.EXPERT,
                    [TensorSpec(f"{prefix}_dispatched", (b * model.top_k, s, h))],
                    [TensorSpec(f"{prefix}_expert_out_{e}", (b * model.top_k, s, h))],
                    flops=2 * b * s * h * model.intermediate_size,
                    layer_idx=layer_idx,
                )
            )
        ops.append(
            Operator(
                f"{prefix}_combine_a2a",
                OpType.COMM_A2A,
                [TensorSpec(f"{prefix}_expert_out_0", (b * model.top_k, s, h))],
                [hidden],
                layer_idx=layer_idx,
            )
        )
        return ops
