from __future__ import annotations

from simpipe.core.types import Schedule, WorkloadType
from simpipe.pipeline.scheduling.base import ScheduleContext, ScheduleStrategy


class AFABStrategy(ScheduleStrategy):
    name = Schedule.AFAB

    def generate(self, ctx: ScheduleContext) -> list[list[tuple[WorkloadType, int, int]]]:
        schedule: list[list[tuple[WorkloadType, int, int]]] = [[] for _ in range(ctx.device_num)]
        for did in range(ctx.device_num):
            sids = ctx.placement.device_stages[did]
            for mid in range(ctx.micro_batch_num):
                schedule[did].append((WorkloadType.F, mid + ctx.mid_offset, sids[0]))
            for mid in range(ctx.micro_batch_num):
                schedule[did].append((WorkloadType.B, mid + ctx.mid_offset, sids[0]))
            if ctx.bwd_split:
                for mid in range(ctx.micro_batch_num):
                    schedule[did].append((WorkloadType.W, mid + ctx.mid_offset, sids[0]))
        return schedule
