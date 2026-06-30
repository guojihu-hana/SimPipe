from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from simpipe.core.types import Schedule, WorkloadType
from simpipe.pipeline.placement import Placement


@dataclass
class ScheduleContext:
    device_num: int
    stage_num: int
    micro_batch_num: int
    mid_offset: int
    bwd_split: bool
    chunk_num: int
    placement: Placement
    recomp: bool = False
    split_recomp: bool = False
    max_act: int = 1


class ScheduleStrategy(ABC):
    name: Schedule

    @abstractmethod
    def generate(self, ctx: ScheduleContext) -> list[list[tuple[WorkloadType, int, int]]]:
        """Per device: list of (wtype, microbatch_id, stage_id)."""
        ...


def get_strategy(schedule: Schedule) -> ScheduleStrategy:
    from simpipe.pipeline.scheduling.one_f_one_b import OneFOneBStrategy
    from simpipe.pipeline.scheduling.interleaved import InterleavedStrategy
    from simpipe.pipeline.scheduling.zbh import ZBHStrategy
    from simpipe.pipeline.scheduling.afab import AFABStrategy

    mapping = {
        Schedule.S1F1B: OneFOneBStrategy(),
        Schedule.BAPAR: OneFOneBStrategy(),
        Schedule.INTERLEAVED: InterleavedStrategy(),
        Schedule.ZBH: ZBHStrategy(),
        Schedule.AFAB: AFABStrategy(),
        Schedule.OctoPipe: OneFOneBStrategy(),  # dynamic at runtime
    }
    if schedule not in mapping:
        raise ValueError(f"Unsupported schedule {schedule}")
    return mapping[schedule]
