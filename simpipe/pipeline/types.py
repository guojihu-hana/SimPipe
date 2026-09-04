from __future__ import annotations

from dataclasses import dataclass, field, replace

from simpipe.core.types import Schedule, WorkloadType
from simpipe.pipeline.partition import OperatorPartition
from simpipe.pipeline.placement import Placement


@dataclass
class StageTiming:
    stage_id: int
    operator_ids: list[str]
    f_time: float
    b_time: float
    w_time: float
    # Attention-score portion of f_time/b_time: scales quadratically with
    # sequence length while the remainder (GEMMs, mamba, norms) and w_time
    # scale linearly with token count.
    f_quad: float = 0.0
    b_quad: float = 0.0


@dataclass
class WorkloadPlan:
    partition: OperatorPartition
    placement: Placement
    schedule: Schedule
    stage_timings: list[StageTiming]
    static_schedule: list[list[tuple[WorkloadType, int, int]]] | None = None
    layer_f_times: list[float] = field(default_factory=list)
    layer_b_times: list[float] = field(default_factory=list)
    layer_w_times: list[float] = field(default_factory=list)
    # Transformer-layer count per stage (activation-memory weights); stage
    # operator_ids are finer grained (~8 ops/layer) so cannot be used here.
    layers_per_stage: list[int] | None = None
    # Per-microbatch (linear_scale, quadratic_scale) vs the profiled shape,
    # indexed by local microbatch id (mid % len).  None = uniform microbatches.
    mid_scales: list[tuple[float, float]] | None = None
    # Tuned execution order: slot mid k runs input microbatch mid_order[k]
    # (mid_scales is already permuted accordingly).  None = input order.
    mid_order: list[int] | None = None

    @property
    def stage_num(self) -> int:
        return self.partition.num_stages

    @property
    def device_num(self) -> int:
        return len(self.placement.device_stages)

    def scale_for_mid(self, mid: int) -> tuple[float, float]:
        if not self.mid_scales:
            return (1.0, 1.0)
        return self.mid_scales[mid % len(self.mid_scales)]

    def token_ratio_for_mid(self, mid: int) -> float:
        """Token count of a microbatch relative to the profiled shape."""
        return self.scale_for_mid(mid)[0]

    def timing_for_stage(self, stage_id: int, mid: int | None = None) -> StageTiming:
        timing = self.stage_timings[stage_id]
        if mid is None or not self.mid_scales:
            return timing
        lin, quad = self.scale_for_mid(mid)
        return replace(
            timing,
            f_time=(timing.f_time - timing.f_quad) * lin + timing.f_quad * quad,
            b_time=(timing.b_time - timing.b_quad) * lin + timing.b_quad * quad,
            w_time=timing.w_time * lin,
            f_quad=timing.f_quad * quad,
            b_quad=timing.b_quad * quad,
        )
