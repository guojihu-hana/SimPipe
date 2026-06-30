from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from simpipe.graph.model_graph import ModelGraph
from simpipe.graph.tensor import TensorKind, tensor_bytes


@dataclass
class MemoryBreakdown:
    activation_bytes: int = 0
    parameter_bytes: int = 0
    gradient_bytes: int = 0
    optimizer_bytes: int = 0
    p2p_buffer_bytes: int = 0

    @property
    def total(self) -> int:
        return (
            self.activation_bytes
            + self.parameter_bytes
            + self.gradient_bytes
            + self.optimizer_bytes
            + self.p2p_buffer_bytes
        )


def analyze_liveness(graph: ModelGraph) -> tuple[int, MemoryBreakdown]:
    """Forward liveness analysis over operator DAG; returns peak activation bytes."""
    live: dict[str, int] = {}
    peak = 0
    breakdown = MemoryBreakdown()

    for op in graph.operators:
        for inp in op.inputs:
            if inp.kind == TensorKind.ACTIVATION:
                live[inp.name] = live.get(inp.name, 0) + 1
        for out in op.outputs:
            if out.kind == TensorKind.ACTIVATION:
                live[out.name] = live.get(out.name, 0) + 1
            elif out.kind == TensorKind.PARAMETER:
                breakdown.parameter_bytes += tensor_bytes(out)
        # release inputs after op (simplified: last consumer)
        for inp in op.inputs:
            if inp.kind == TensorKind.ACTIVATION and inp.name in live:
                live[inp.name] -= 1
                if live[inp.name] <= 0:
                    del live[inp.name]
        act = sum(tensor_bytes(graph_op.outputs[0]) for graph_op in [op] if graph_op.outputs)
        current = sum(
            tensor_bytes(next(t for t in op.inputs + op.outputs if t.name == n))
            for n in live
            if any(t.name == n for t in op.inputs + op.outputs)
        )
        peak = max(peak, current)

    breakdown.activation_bytes = peak
    breakdown.gradient_bytes = breakdown.parameter_bytes
    breakdown.optimizer_bytes = breakdown.parameter_bytes * 4
    return peak, breakdown


def check_memory_feasible(peak_bytes: int, hbm_bytes: int) -> bool:
    return peak_bytes <= hbm_bytes
