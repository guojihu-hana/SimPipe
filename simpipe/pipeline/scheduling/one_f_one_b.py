from __future__ import annotations

from simpipe.core.types import Schedule, WorkloadType
from simpipe.pipeline.scheduling.base import ScheduleContext, ScheduleStrategy


class OneFOneBStrategy(ScheduleStrategy):
    """Match PipelineSimulator.generate_1f1b_schedule: warmup F then alternate B,F,B,F,..."""

    name = Schedule.S1F1B

    def generate(self, ctx: ScheduleContext) -> list[list[tuple[WorkloadType, int, int]]]:
        if ctx.bwd_split:
            workload_type_order = [WorkloadType.B, WorkloadType.W, WorkloadType.F]
        else:
            workload_type_order = [WorkloadType.B, WorkloadType.F]
        workload_type_num = len(workload_type_order)
        idx_map = {WorkloadType.F: 0, WorkloadType.B: 1, WorkloadType.W: 2}

        schedule: list[list[tuple[WorkloadType, int, int]]] = [[] for _ in range(ctx.device_num)]
        mid_offset = ctx.mid_offset

        for did in range(ctx.device_num):
            mids = [0] * workload_type_num
            # Warmup: inject forward passes
            while mids[0] < min(ctx.device_num - did, ctx.micro_batch_num):
                schedule[did].append((WorkloadType.F, mids[0] + mid_offset, did))
                mids[0] += 1

            it = 0
            finish_flag = [0] * workload_type_num
            while sum(finish_flag) < workload_type_num:
                next_wtype = workload_type_order[it % workload_type_num]
                slot = idx_map[next_wtype]
                next_mid = mids[slot]
                if next_mid < ctx.micro_batch_num:
                    schedule[did].append((next_wtype, next_mid + mid_offset, did))
                    mids[slot] += 1
                else:
                    finish_flag[slot] = 1
                it += 1

        return schedule
