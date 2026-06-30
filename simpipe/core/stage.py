from __future__ import annotations

from simpipe.core.types import WorkloadConstraint, WorkloadType
from simpipe.core.workload import Workload


class Stage:
    def __init__(
        self,
        stage_idx: int,
        device_idx: int,
        timing,
        schedule_method,
        total_stage_num: int,
        micro_batch_num: int,
        mid_offset: int,
        bwd_split: bool,
        placement: list[list[int]],
        recomp: bool = False,
        split_recomp: bool = False,
        comp_power: float = 1.0,
        vocab_parallel: bool = False,
        comm_time: float = 0.0,
    ):
        self.sid = stage_idx
        self.did = device_idx
        self.timing = timing
        self.total_stage_num = total_stage_num
        self.bwd_split = bwd_split
        self.workloads: dict[int, dict[WorkloadType, Workload]] = {}
        self._build_workloads(
            schedule_method,
            micro_batch_num,
            mid_offset,
            bwd_split,
            placement,
            recomp,
            split_recomp,
            comp_power,
            vocab_parallel,
            comm_time,
        )

    def _build_workloads(
        self,
        schedule_method,
        nmb,
        mid_offset,
        bwd_split,
        placement,
        recomp,
        split_recomp,
        comp_power,
        vocab_parallel,
        comm_time,
    ) -> None:
        for mid in range(mid_offset, mid_offset + nmb):
            self.workloads[mid] = {}
            for wtype, duration in (
                (WorkloadType.F, self.timing.f_time),
                (WorkloadType.B, self.timing.b_time),
                (WorkloadType.W, self.timing.w_time),
            ):
                if wtype == WorkloadType.W and not bwd_split and duration <= 0:
                    continue
                w = Workload(
                    schedule_method=schedule_method,
                    device_idx=self.did,
                    microbatch_idx=mid,
                    stage_idx=self.sid,
                    bwd_split=bwd_split,
                    duration=duration,
                    total_stage_num=self.total_stage_num,
                    wtype=wtype,
                    recomp=recomp,
                    split_recomp=split_recomp,
                    comp_power=comp_power,
                    vocab_parallel=vocab_parallel,
                    placement=placement,
                    comm_time=comm_time,
                )
                self.workloads[mid][wtype] = w

    def get_workload(self, mid: int, wtype: WorkloadType) -> Workload | None:
        return self.workloads.get(mid, {}).get(wtype)

    def update_constraints_within_stage(self, time: float, completed: Workload) -> Workload | None:
        """Targeted constraint release matching PipelineSimulator Stage logic."""
        c_did = completed.did
        c_sid = completed.sid
        c_mid = completed.mid
        c_wlt = completed.wtype

        if c_wlt == WorkloadType.F:
            if self.sid == c_sid + 1 and c_mid in self.workloads:
                cstr = WorkloadConstraint(c_did, c_mid, c_sid, c_wlt)
                self.workloads[c_mid][WorkloadType.F].update_constraints(time, cstr)
                return self.workloads[c_mid][WorkloadType.F]
            if self.sid == c_sid and self.sid == completed.total_stage_num - 1:
                cstr = WorkloadConstraint(c_did, c_mid, c_sid, c_wlt)
                self.workloads[c_mid][WorkloadType.B].update_constraints(time, cstr)
                return self.workloads[c_mid][WorkloadType.B]
        elif c_wlt == WorkloadType.B:
            if self.sid == c_sid - 1 and c_mid in self.workloads:
                cstr = WorkloadConstraint(c_did, c_mid, c_sid, c_wlt)
                self.workloads[c_mid][WorkloadType.B].update_constraints(time, cstr)
                return self.workloads[c_mid][WorkloadType.B]
            if self.sid == c_sid and self.bwd_split and c_mid in self.workloads:
                cstr = WorkloadConstraint(c_did, c_mid, c_sid, c_wlt)
                self.workloads[c_mid][WorkloadType.W].update_constraints(time, cstr)
                return self.workloads[c_mid][WorkloadType.W]
        elif c_wlt == WorkloadType.R:
            if self.sid == c_sid and c_mid in self.workloads:
                cstr = WorkloadConstraint(c_did, c_mid, c_sid, c_wlt)
                self.workloads[c_mid][WorkloadType.B].update_constraints(time, cstr)
                return self.workloads[c_mid][WorkloadType.B]
        elif c_wlt == WorkloadType.W:
            if self.sid == c_sid - 1 and c_mid in self.workloads:
                cstr = WorkloadConstraint(c_did, c_mid, c_sid, c_wlt)
                self.workloads[c_mid][WorkloadType.B].update_constraints(time, cstr)
                return self.workloads[c_mid][WorkloadType.B]
        return None
