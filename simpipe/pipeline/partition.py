from __future__ import annotations

from dataclasses import dataclass, field

from simpipe.graph.model_graph import EMBEDDING_LAYER_IDX, ModelGraph, head_layer_idx


@dataclass
class StageSpec:
    """One PP stage: contiguous slice of operators (may span layers)."""

    stage_id: int
    operator_ids: list[str] = field(default_factory=list)

    @property
    def p2p_output_id(self) -> str | None:
        return self.operator_ids[-1] if self.operator_ids else None


@dataclass
class OperatorPartition:
    stages: list[StageSpec]

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def validate(self, graph: ModelGraph) -> None:
        all_ids = graph.op_ids()
        seen: list[str] = []
        for stage in self.stages:
            for oid in stage.operator_ids:
                if oid not in all_ids:
                    raise ValueError(f"Unknown operator {oid}")
                seen.append(oid)
        if set(seen) != set(all_ids):
            raise ValueError(f"Partition must cover all operators (missing={set(all_ids)-set(seen)})")
        if seen != all_ids:
            raise ValueError("Partition operators must follow global topological order")

    def layer_counts(self, graph: ModelGraph) -> list[int]:
        """Transformer layers per stage (excludes embedding/head)."""
        counts: list[int] = []
        for stage in self.stages:
            layers = {
                graph.op_by_id(oid).layer_idx
                for oid in stage.operator_ids
                if graph.op_by_id(oid).layer_idx is not None
                and 0 <= graph.op_by_id(oid).layer_idx < graph.model.num_layers
            }
            counts.append(len(layers) if layers else 1)
        return counts

    @property
    def p2p_boundaries(self) -> list[tuple[int, int, str]]:
        """(from_stage, to_stage, tensor_op_id) for each adjacent stage pair."""
        out: list[tuple[int, int, str]] = []
        for i in range(len(self.stages) - 1):
            out_id = self.stages[i].p2p_output_id
            if out_id:
                out.append((i, i + 1, out_id))
        return out


def layer_partition_to_stage_specs(
    graph: ModelGraph,
    layer_counts: list[int],
) -> OperatorPartition:
    """Convert legacy layer-level partition [n0, n1, ...] to operator StageSpecs."""
    ops_by_layer: dict[int, list[str]] = {}
    for op in graph.operators:
        if op.layer_idx is not None:
            ops_by_layer.setdefault(op.layer_idx, []).append(op.id)
    embedding_ops = ops_by_layer.get(EMBEDDING_LAYER_IDX, [])
    head_ops = ops_by_layer.get(head_layer_idx(graph.model.num_layers), [])
    stages: list[StageSpec] = []
    layer_cursor = 0
    for sid, count in enumerate(layer_counts):
        op_ids: list[str] = []
        if sid == 0:
            op_ids.extend(embedding_ops)
        for _ in range(count):
            if layer_cursor >= graph.model.num_layers:
                break
            op_ids.extend(ops_by_layer.get(layer_cursor, []))
            layer_cursor += 1
        if sid == len(layer_counts) - 1:
            op_ids.extend(head_ops)
        stages.append(StageSpec(stage_id=sid, operator_ids=op_ids))
    return OperatorPartition(stages=stages)


def split_layer_at_fraction(
    graph: ModelGraph,
    layer_idx: int,
    fraction: float,
) -> tuple[list[str], list[str]]:
    """Split one layer's operators at fraction (0..1)."""
    ops = [op.id for op in graph.operators if op.layer_idx == layer_idx]
    if not ops:
        return [], []
    cut = max(1, min(len(ops) - 1, int(len(ops) * fraction)))
    return ops[:cut], ops[cut:]
