from __future__ import annotations

from dataclasses import dataclass, field

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

    @property
    def stage_num(self) -> int:
        return self.partition.num_stages

    @property
    def device_num(self) -> int:
        return len(self.placement.device_stages)

    def timing_for_stage(self, stage_id: int) -> StageTiming:
        return self.stage_timings[stage_id]
