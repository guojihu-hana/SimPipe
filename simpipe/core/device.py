from __future__ import annotations

from simpipe.core.stage import Stage
from simpipe.core.types import OrderedQueue, Schedule, WorkloadType
from simpipe.core.workload import Workload
from simpipe.pipeline.types import WorkloadPlan


class Device:
    BUSY = 1
    IDLE = 2

    def __init__(
        self,
        device_idx: int,
        plan: WorkloadPlan,
        mid_offset: int,
        comp_power: float = 1.0,
        max_mem: float = 80.0,
        comm_time: float = 0.0,
        workload_overhead: float = 0.0,
        runtime=None,
    ):
        self.did = device_idx
        self.plan = plan
        self.schedule_method = plan.schedule
        self.bwd_split = runtime.bwd_split if runtime else False
        self.nmb = runtime.micro_batch_num if runtime else 8
        self.device_num = plan.device_num
        self.placement = plan.placement.device_stages
        self.mid_offset = mid_offset
        self.comp_power = comp_power
        self.max_mem = max_mem
        self.comm_time = comm_time
        self.workload_overhead = workload_overhead
        self.state = Device.IDLE
        self.current_workload: Workload | None = None
        self.stages: dict[int, Stage] = {}
        self.static_schedule = (
            plan.static_schedule[device_idx] if plan.static_schedule else None
        )
        self.next_workload_idx = 0
        self.exe_num_f = 0
        self.exe_num_b = 0
        self.exe_num_w = 0
        self.idle_time = 0
        self.peak_memory_usage = 0.0
        self.current_mem_usage = 0.0
        self.executable_workloads = OrderedQueue(
            [WorkloadType.B, WorkloadType.F, WorkloadType.W]
        )
        self.last_wtype: WorkloadType | None = None
        self.runtime = runtime
        self.workload_execute_record: list[Workload] = []

    def add_stage(self, stage_id: int, recomp: bool = False) -> None:
        self.stages[stage_id] = Stage(
            stage_idx=stage_id,
            device_idx=self.did,
            plan=self.plan,
            schedule_method=self.schedule_method,
            total_stage_num=self.plan.stage_num,
            micro_batch_num=self.nmb,
            mid_offset=self.mid_offset,
            bwd_split=self.bwd_split,
            placement=self.placement,
            recomp=recomp,
            comp_power=self.comp_power,
            comm_time=self.comm_time,
            overhead=self.workload_overhead,
        )

    def get_initial_executable_workload(self, time: float) -> list[Workload]:
        ready = []
        for stage in self.stages.values():
            for wmap in stage.workloads.values():
                for w in wmap.values():
                    if w.is_executable(time):
                        ready.append(w)
        return ready

    def check_workload_status(self, time: float) -> list[Workload]:
        """Deprecated: completion handled in PipelineRuntime.check_workload_status."""
        return []

    def execute_workload(self, time: float) -> Workload | None:
        if self.state == Device.BUSY:
            return None
        w = self._next_workload(time)
        if w and w.execute(time):
            if self.runtime is not None:
                self.runtime.on_workload_started(w)
            self.state = Device.BUSY
            self.current_workload = w
            self.last_wtype = w.wtype
            self.workload_execute_record.append(w)
            if w.wtype == WorkloadType.F:
                self.exe_num_f += 1
            elif w.wtype == WorkloadType.B:
                self.exe_num_b += 1
            elif w.wtype == WorkloadType.W:
                self.exe_num_w += 1
            return w
        return None

    def _next_workload(self, time: float) -> Workload | None:
        if self.schedule_method == Schedule.OctoPipe:
            self._set_octopipe_type_order()
            overlap_deferred: list[Workload] = []
            not_ready_deferred: list[Workload] = []
            while self.executable_workloads:
                w = self.executable_workloads.pop()
                if w and w.is_executable(time):
                    if (
                        w.wtype == WorkloadType.F
                        and self.runtime is not None
                        and self.runtime.f_admission_blocked(self.did, w.sid, w.mid)
                    ):
                        not_ready_deferred.append(w)
                        continue
                    if self._should_delay_for_overlap(time, w):
                        overlap_deferred.append(w)
                        continue
                    for item in overlap_deferred + not_ready_deferred:
                        self.executable_workloads.push(item)
                    return w
                if w and w.state == Workload.not_started and len(w.constraints) == 0:
                    not_ready_deferred.append(w)
            if overlap_deferred:
                selected = overlap_deferred.pop(0)
                for item in overlap_deferred + not_ready_deferred:
                    self.executable_workloads.push(item)
                return selected
            for item in not_ready_deferred:
                self.executable_workloads.push(item)
            return None
        if self.static_schedule and self.next_workload_idx < len(self.static_schedule):
            wtype, mid, sid = self.static_schedule[self.next_workload_idx]
            # Static schedules are generated with mids 0..nmb-1; replica
            # pipelines (dp_idx > 0) number their workloads from mid_offset.
            mid += self.mid_offset
            stage = self.stages.get(sid)
            if stage is None:
                # A schedule/placement mismatch would otherwise stall silently
                # (this entry never executes and the index never advances).
                raise RuntimeError(
                    f"static schedule references stage {sid} which is not placed "
                    f"on device {self.did} (placement: {self.placement})"
                )
            w = stage.get_workload(mid, wtype)
            if w and w.is_executable(time):
                self.next_workload_idx += 1
                return w
            return None
        for stage in self.stages.values():
            for wmap in stage.workloads.values():
                for w in wmap.values():
                    if w.is_executable(time):
                        return w
        return None

    def _should_delay_for_overlap(self, time: float, workload: Workload) -> bool:
        if not self.runtime or not self.runtime.parallel.overlap_aware:
            return False
        if self.runtime.is_overlap_exempt(workload):
            return False
        if (
            self.runtime.parallel.skip_overlap_until_first_backward
            and not self.runtime.has_started_backward()
        ):
            return False
        return self.has_direct_dependency(time, workload)

    def has_direct_dependency(self, time: float, workload: Workload) -> bool:
        if not self.workload_execute_record:
            return False
        last_local = self.workload_execute_record[-1]
        if last_local.end_time is not None and last_local.end_time < time:
            return False
        if not self.runtime:
            return False
        for device in self.runtime.devices:
            if device.did == self.did or not device.workload_execute_record:
                continue
            pivot = device.workload_execute_record[-1]
            if pivot.mid != workload.mid:
                continue
            if (
                pivot.sid == workload.sid - 1
                and pivot.wtype == workload.wtype == WorkloadType.F
                and pivot.end_time is not None
                and last_local.start_time is not None
                and pivot.end_time > last_local.start_time
            ):
                return True
            if (
                pivot.sid == workload.sid + 1
                and pivot.wtype == workload.wtype == WorkloadType.B
                and pivot.end_time is not None
                and last_local.start_time is not None
                and pivot.end_time > last_local.start_time
            ):
                return True
        return False

    def _set_octopipe_type_order(self) -> None:
        type_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
        if self.runtime and self.runtime.parallel.switch_workload_type:
            if self.last_wtype == WorkloadType.B:
                type_order = [WorkloadType.F, WorkloadType.B, WorkloadType.W]
            elif self.last_wtype == WorkloadType.F:
                type_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
        self.executable_workloads.set_type_order(type_order)

    def push_executable_workload(self, workload: Workload, time: float) -> None:
        if workload.state == Workload.not_started and len(workload.constraints) == 0:
            self.executable_workloads.push(workload)

