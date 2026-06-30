from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from simpipe.graph.tensor import TensorSpec


class OpType(str, Enum):
    LAYERNORM = "layernorm"
    MATMUL = "matmul"
    ATTENTION = "attention"
    MLP = "mlp"
    MLP_GATE = "mlp_gate"
    MLP_UP = "mlp_up"
    MLP_DOWN = "mlp_down"
    ROUTER = "router"
    EXPERT = "expert"
    EMBEDDING = "embedding"
    HEAD = "head"
    LOSS = "loss"
    COMM_ALLGATHER = "comm_allgather"
    COMM_ALLREDUCE = "comm_allreduce"
    COMM_A2A = "comm_a2a"
    COMM_PP = "comm_pp"


@dataclass
class Operator:
    id: str
    op_type: OpType
    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)
    profiled_time_us: float | None = None
    profiled_bwd_time_us: float | None = None
    profiled_w_time_us: float | None = None
    flops: int | None = None
    layer_idx: int | None = None
    is_checkpoint_boundary: bool = False

    def forward_time(self, peak_tflops: float = 300.0, mem_bw_gbps: float = 2000.0) -> float:
        if self.profiled_time_us is not None:
            return self.profiled_time_us
        if self.flops is None:
            return 1.0
        compute_s = self.flops / (peak_tflops * 1e12)
        bytes_moved = sum(t.nbytes() for t in self.inputs + self.outputs)
        mem_s = bytes_moved / (mem_bw_gbps * 1e9)
        return max(compute_s, mem_s) * 1e6  # microseconds

    def backward_time(self, peak_tflops: float = 300.0, mem_bw_gbps: float = 2000.0) -> float:
        if self.profiled_bwd_time_us is not None:
            return self.profiled_bwd_time_us
        return self.forward_time(peak_tflops, mem_bw_gbps)

    def weight_time(self, peak_tflops: float = 300.0, mem_bw_gbps: float = 2000.0) -> float:
        if self.profiled_w_time_us is not None:
            return self.profiled_w_time_us
        return self.forward_time(peak_tflops, mem_bw_gbps)
