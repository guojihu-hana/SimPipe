from __future__ import annotations

from simpipe.core.types import Schedule, WorkloadType
from simpipe.pipeline.scheduling.base import ScheduleContext, ScheduleStrategy


class InterleavedStrategy(ScheduleStrategy):
    """Match PipelineSimulator.generate_interleaved_1f1b_schedule."""

    name = Schedule.INTERLEAVED

    def generate(self, ctx: ScheduleContext) -> list[list[tuple[WorkloadType, int, int]]]:
        workload_type_num = 2
        mid_offset = ctx.mid_offset
        schedule: list[list[tuple[WorkloadType, int, int]]] = [
            [] for _ in range(ctx.device_num)
        ]

        for did in range(ctx.device_num):
            sids = list(ctx.placement.device_stages[did])
            chunk_num = len(sids)

            mids = [0] * (workload_type_num * chunk_num)
            f_mid_count = 0

            f_next_sid_idx = 0
            f_next_sid = sids[f_next_sid_idx]
            idx_in_f_mids = f_next_sid_idx * workload_type_num

            warmup_f_num = (chunk_num - 1) * ctx.device_num + (ctx.device_num - did - 1) * 2
            while mids[idx_in_f_mids] < ctx.micro_batch_num and f_mid_count < warmup_f_num:
                schedule[did].append(
                    (WorkloadType.F, mids[idx_in_f_mids] + mid_offset, f_next_sid)
                )
                mids[idx_in_f_mids] += 1
                f_mid_count += 1
                if f_mid_count % ctx.device_num == 0:
                    f_next_sid_idx = (f_next_sid_idx + 1) % len(sids)
                    f_next_sid = sids[f_next_sid_idx]
                    idx_in_f_mids = f_next_sid_idx * workload_type_num

            b_mid_count = 0
            bsids = list(reversed(sids))
            b_next_sid_idx = 0
            b_next_sid = bsids[b_next_sid_idx]
            idx_in_b_mids = 1 + b_next_sid_idx * workload_type_num

            operation_flag = "f"
            total_ops = ctx.micro_batch_num * chunk_num * workload_type_num
            while b_mid_count + f_mid_count < total_ops:
                if operation_flag == "f":
                    if mids[idx_in_f_mids] < ctx.micro_batch_num:
                        schedule[did].append(
                            (WorkloadType.F, mids[idx_in_f_mids] + mid_offset, f_next_sid)
                        )
                        mids[idx_in_f_mids] += 1
                        f_mid_count += 1
                        if f_mid_count % ctx.device_num == 0:
                            f_next_sid_idx = (f_next_sid_idx + 1) % len(sids)
                            f_next_sid = sids[f_next_sid_idx]
                            idx_in_f_mids = f_next_sid_idx * workload_type_num
                    operation_flag = "b"
                elif operation_flag == "b":
                    if mids[idx_in_b_mids] < ctx.micro_batch_num:
                        if ctx.recomp and ctx.split_recomp:
                            schedule[did].append(
                                (WorkloadType.R, mids[idx_in_b_mids] + mid_offset, b_next_sid)
                            )
                        schedule[did].append(
                            (WorkloadType.B, mids[idx_in_b_mids] + mid_offset, b_next_sid)
                        )
                        if ctx.bwd_split:
                            schedule[did].append(
                                (WorkloadType.W, mids[idx_in_b_mids] + mid_offset, b_next_sid)
                            )
                        mids[idx_in_b_mids] += 1
                        b_mid_count += 1
                        if b_mid_count % ctx.device_num == 0:
                            b_next_sid_idx = (b_next_sid_idx + 1) % len(bsids)
                            b_next_sid = bsids[b_next_sid_idx]
                            idx_in_b_mids = 1 + b_next_sid_idx * workload_type_num
                    operation_flag = "f"

        return schedule
