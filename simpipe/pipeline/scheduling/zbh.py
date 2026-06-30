from __future__ import annotations

from simpipe.core.types import Schedule, WorkloadType
from simpipe.pipeline.scheduling.base import ScheduleContext, ScheduleStrategy


class ZBHStrategy(ScheduleStrategy):
    """Match PipelineSimulator.generate_zbh_schedule (requires bwd_split)."""

    name = Schedule.ZBH

    def generate(self, ctx: ScheduleContext) -> list[list[tuple[WorkloadType, int, int]]]:
        if not ctx.bwd_split:
            raise ValueError("ZBH schedule requires bwd_split=True")

        workload_type_order = [WorkloadType.B, WorkloadType.W, WorkloadType.F]
        workload_type_num = len(workload_type_order)
        idx_map = {WorkloadType.F: 0, WorkloadType.B: 1, WorkloadType.W: 2}
        max_act = ctx.max_act
        mid_offset = ctx.mid_offset

        schedule: list[list[tuple[WorkloadType, int, int]]] = [
            [] for _ in range(ctx.device_num)
        ]

        for did in range(ctx.device_num):
            accumulated_act_num = min(
                ctx.micro_batch_num,
                (ctx.device_num - did - 1) * max_act + 1,
            )
            mids = [0] * workload_type_num

            while mids[0] < min(accumulated_act_num, ctx.micro_batch_num):
                schedule[did].append((WorkloadType.F, mids[0] + mid_offset, did))
                mids[0] += 1

            it = 0
            finish_flag = [0] * workload_type_num
            act_limit = min(ctx.micro_batch_num, ctx.stage_num * max_act)
            while sum(finish_flag) < workload_type_num:
                next_wtype = workload_type_order[it % workload_type_num]
                slot = idx_map[next_wtype]
                next_mid = mids[slot]
                if mids[0] < act_limit and next_wtype == WorkloadType.W:
                    it += 1
                    continue
                if next_mid < ctx.micro_batch_num:
                    schedule[did].append((next_wtype, next_mid + mid_offset, did))
                    mids[slot] += 1
                else:
                    finish_flag[slot] = 1
                it += 1

        return schedule
