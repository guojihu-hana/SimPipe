from __future__ import annotations

from simpipe.core.types import Schedule, WorkloadConstraint, WorkloadType


class Workload:
    not_started = 1
    in_progress = 2
    finished = 3

    def __init__(
        self,
        schedule_method: Schedule,
        device_idx: int,
        microbatch_idx: int,
        stage_idx: int,
        bwd_split: bool,
        duration: float,
        total_stage_num: int,
        wtype: WorkloadType,
        recomp: bool,
        split_recomp: bool,
        comp_power: float,
        vocab_parallel: bool,
        placement: list[list[int]],
        comm_time: float = 0.0,
    ):
        self.schedule_method = schedule_method
        self.bwd_split = bwd_split
        self.did = device_idx
        self.mid = microbatch_idx
        self.sid = stage_idx
        self.duration = max(1, int(duration / comp_power))
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.state = Workload.not_started
        self.ready_time = 0 if microbatch_idx == 0 and stage_idx == 0 else -1
        self.total_stage_num = total_stage_num
        self.recomp = recomp
        self.split_recomp = split_recomp
        self.comp_power = comp_power
        self.vocab_parallel = vocab_parallel
        self.placement = placement
        self.comm_time = comm_time
        self.wtype = wtype
        self.wtype_str = wtype.name.lower()
        self.constraints: set[WorkloadConstraint] = set()
        self._generate_constraints()

    def sid2did(self, sid: int) -> list[int]:
        dids = []
        for did, sids in enumerate(self.placement):
            if sid in sids:
                dids.append(did)
            if self.vocab_parallel and self.wtype == WorkloadType.B and sid == self.total_stage_num - 1:
                dids.append(did)
        return dids

    def _generate_constraints(self) -> None:
        if self.wtype == WorkloadType.F:
            if self.sid > 0:
                for did in self.sid2did(self.sid - 1):
                    self.constraints.add(
                        WorkloadConstraint(did, self.mid, self.sid - 1, WorkloadType.F)
                    )
        elif self.wtype == WorkloadType.R:
            for did in self.sid2did(self.sid):
                self.constraints.add(
                    WorkloadConstraint(did, self.mid, self.sid, WorkloadType.F)
                )
        elif self.wtype == WorkloadType.B:
            if self.sid + 1 < self.total_stage_num:
                dep_type = (
                    WorkloadType.W
                    if self.bwd_split
                    and self.schedule_method
                    in (Schedule.S1F1B, Schedule.BAPAR, Schedule.INTERLEAVED)
                    else WorkloadType.B
                )
                for did in self.sid2did(self.sid + 1):
                    self.constraints.add(
                        WorkloadConstraint(did, self.mid, self.sid + 1, dep_type)
                    )
            else:
                for did in self.sid2did(self.total_stage_num - 1):
                    self.constraints.add(
                        WorkloadConstraint(did, self.mid, self.total_stage_num - 1, WorkloadType.F)
                    )
            if self.recomp and self.split_recomp:
                for did in self.sid2did(self.sid):
                    self.constraints.add(
                        WorkloadConstraint(did, self.mid, self.sid, WorkloadType.R)
                    )
        elif self.wtype == WorkloadType.W:
            self.constraints.add(
                WorkloadConstraint(self.did, self.mid, self.sid, WorkloadType.B)
            )

    def update_constraints(self, time: float, constraint: WorkloadConstraint) -> None:
        before = len(self.constraints)
        self.constraints.discard(constraint)
        if len(self.constraints) < before:
            if constraint.device_id != self.did:
                self.ready_time = max(self.ready_time, time + self.comm_time)
            else:
                self.ready_time = max(self.ready_time, time)

    def is_executable(self, time: float) -> bool:
        return (
            len(self.constraints) == 0
            and self.ready_time <= time
            and self.state == Workload.not_started
        )

    def execute(self, time: float) -> bool:
        if self.state == Workload.not_started and self.is_executable(time):
            self.state = Workload.in_progress
            self.start_time = time
            self.end_time = self.start_time + self.duration
            return True
        return False

    def complete(self, time: float) -> None:
        if self.state == Workload.in_progress and self.end_time is not None and self.end_time <= time:
            self.state = Workload.finished

    def __repr__(self) -> str:
        return (
            f"Workload(did={self.did}, mid={self.mid}, sid={self.sid}, "
            f"wtype={self.wtype.name}, dur={self.duration})"
        )
