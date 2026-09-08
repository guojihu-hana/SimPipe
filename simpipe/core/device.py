from __future__ import annotations

import heapq

from simpipe.core.stage import Stage
from simpipe.core.types import OrderedQueue, Schedule, WorkloadType
from simpipe.core.workload import Workload
from simpipe.pipeline.types import WorkloadPlan

# Aux (encoder/decoder) pick order when several are ready: earliest
# microbatch first, B before F (a ready decoder B gates the backbone's
# backward chain), W last.
_AUX_TYPE_RANK = {WorkloadType.B: 0, WorkloadType.F: 1, WorkloadType.W: 2}


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
        # Dedicated aux devices sit past the backbone's device ids and have
        # no row in the static schedule (they only run aux workloads).
        self.static_schedule = (
            plan.static_schedule[device_idx]
            if plan.static_schedule and device_idx < len(plan.static_schedule)
            else None
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
        # Encoder/decoder workloads placed on this device.  Two regimes:
        # - static schedules (1f1b/interleaved/zbh/...): copies anchored to
        #   their boundary stage are spliced into static_schedule itself
        #   (runtime._init_aux), so each microbatch runs enc F -> bb F ...
        #   bb B -> enc B in order; aux_static resolves those entries.
        # - everything else (OctoPipe, dedicated devices, off-anchor shards)
        #   backfills idle slots: the backbone keeps priority, and whenever
        #   its next entry is blocked the device runs a ready aux instead --
        #   but only one short enough to finish before the blocked entry
        #   could start (_backbone_eta), so aux compute interleaves with the
        #   pipeline without pushing it back.
        self.aux_ready: list = []  # heap of ((mid, type_rank, seq), workload)
        self.aux_static: dict = {}  # (sid, mid, wtype) -> Workload
        self._aux_seq = 0
        self.aux_remaining = 0

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
            admission_deferred: list[Workload] = []
            while self.executable_workloads:
                w = self.executable_workloads.pop()
                if w and w.is_executable(time):
                    if (
                        w.wtype == WorkloadType.F
                        and self.runtime is not None
                        and self.runtime.f_admission_blocked(self.did, w.sid, w.mid)
                    ):
                        # release time unknown (waits on activation frees)
                        admission_deferred.append(w)
                        continue
                    if self._should_delay_for_overlap(time, w):
                        overlap_deferred.append(w)
                        continue
                    for item in overlap_deferred + not_ready_deferred + admission_deferred:
                        self.executable_workloads.push(item)
                    return w
                if w and w.state == Workload.not_started and len(w.constraints) == 0:
                    not_ready_deferred.append(w)
            if overlap_deferred:
                selected = overlap_deferred.pop(0)
                for item in overlap_deferred + not_ready_deferred + admission_deferred:
                    self.executable_workloads.push(item)
                return selected
            # backfill budget: the nearest queued backbone entry only waits
            # on its arrival time (P2P), so aux work must fit before it
            eta = min(
                (it.ready_time for it in not_ready_deferred), default=float("inf")
            )
            for item in not_ready_deferred + admission_deferred:
                self.executable_workloads.push(item)
            return self._next_aux(time, until=eta)
        if self.static_schedule and self.next_workload_idx < len(self.static_schedule):
            wtype, mid, sid = self.static_schedule[self.next_workload_idx]
            # Static schedules are generated with mids 0..nmb-1; replica
            # pipelines (dp_idx > 0) number their workloads from mid_offset.
            mid += self.mid_offset
            aux_w = self.aux_static.get((sid, mid, wtype))
            if aux_w is not None:
                # spliced encoder/decoder entry: runs exactly here so each
                # microbatch keeps enc F -> bb F ... bb B -> enc B order
                # (exempt from the activation throttle -- the slot is already
                # the latest useful moment, see aux_f_blocked)
                if aux_w.is_executable(time):
                    self.next_workload_idx += 1
                    return aux_w
                return self._next_aux(time, until=self._backbone_eta(aux_w))
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
            # backfill: only run aux work that fits before the blocked
            # backbone entry could start, so aux never squeezes it out
            return self._next_aux(time, until=self._backbone_eta(w))
        for stage in self.stages.values():
            for wmap in stage.workloads.values():
                for w in wmap.values():
                    if w.is_executable(time):
                        return w
        return self._next_aux(time)

    def _backbone_eta(self, w: Workload | None) -> float:
        """Earliest start of the blocked schedule entry, when knowable.

        Arrivals already in flight are w.ready_time; for pending upstream
        producers that are currently executing, their fixed end_time (plus
        P2P latency when crossing devices) bounds the start.  Any producer
        not even running yet makes the estimate open-ended (inf), which
        lets aux work run freely (the backbone entry is far away).
        """
        if w is None or w.state != Workload.not_started:
            return float("inf")
        if self.runtime is None:
            return float("inf")
        eta = w.ready_time
        for c in w.constraints:
            finish = self.runtime.constraint_finish_eta(c)
            if finish is None:
                return float("inf")
            arrive = finish + (self.comm_time if c.device_id != w.did else 0)
            eta = max(eta, arrive)
        return eta

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

    def add_aux_workload(self, workload: Workload, static_entry: bool = False) -> None:
        self.aux_remaining += 1
        if static_entry:
            # runs at its spliced position in static_schedule, never backfills
            self.aux_static[(workload.sid, workload.mid, workload.wtype)] = workload
            return
        if len(workload.constraints) == 0:
            self.push_aux_ready(workload)

    def push_aux_ready(self, workload: Workload) -> None:
        heapq.heappush(
            self.aux_ready,
            ((workload.mid, _AUX_TYPE_RANK.get(workload.wtype, 3), self._aux_seq), workload),
        )
        self._aux_seq += 1

    def _next_aux(self, time: float, until: float = float("inf")) -> Workload | None:
        """Best ready aux workload, skipping entries whose ready_time (P2P
        latency) has not arrived yet, that the encoder-activation throttle
        blocks, or that would not finish by `until` (the blocked backbone
        entry's earliest start -- aux backfills gaps, it must not push the
        backbone schedule back); skipped entries go back on the heap."""
        heap = self.aux_ready
        if not heap:
            return None
        skipped = []
        picked: Workload | None = None
        while heap:
            key, w = heapq.heappop(heap)
            if w.state != Workload.not_started:
                continue
            if (
                w.is_executable(time)
                and time + w.duration <= until
                and not (self.runtime is not None and self.runtime.aux_f_blocked(w))
            ):
                picked = w
                break
            skipped.append((key, w))
        for item in skipped:
            heapq.heappush(heap, item)
        return picked

