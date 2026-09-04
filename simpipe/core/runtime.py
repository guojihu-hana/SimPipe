from __future__ import annotations

import math

from simpipe.config.hardware import HardwareConfig
from simpipe.config.parallel import ParallelConfig
from simpipe.core.device import Device
from simpipe.core.types import Schedule, WorkloadConstraint, WorkloadType
from simpipe.core.workload import Workload
from simpipe.pipeline.placement import Placement, sort_placement_by_first_stage
from simpipe.pipeline.types import WorkloadPlan


class PipelineRuntime:
    """Discrete-event PP simulation runtime (execution only, no planning)."""

    def __init__(
        self,
        plan: WorkloadPlan,
        parallel: ParallelConfig,
        hardware: HardwareConfig,
        pipeline_idx: int = 0,
        mid_offset: int | None = None,
        executor=None,
        overlap_exempt_workloads: set[tuple] | None = None,
        overlap_exempt_group_by: str = "mid_type",
    ):
        self.plan = plan
        self.parallel = parallel
        self.hardware = hardware
        self.pipeline_idx = pipeline_idx
        self.micro_batch_num = parallel.micro_batch_num
        self.bwd_split = parallel.bwd_split
        self.device_num = plan.device_num
        self.stage_num = plan.stage_num
        self.time = 0
        self.finish_flag = False
        self.mid_offset = mid_offset if mid_offset is not None else pipeline_idx * parallel.micro_batch_num
        self.executor = executor
        self.overlap_exempt_workloads = overlap_exempt_workloads or set()
        self.overlap_exempt_group_by = overlap_exempt_group_by
        self.devices: list[Device] = []
        self.workload_execute_record: list[list[Workload]] = [
            [] for _ in range(self.device_num)
        ]
        # Activation-memory admission control: block F launches that would
        # push a device's in-flight activations (layer*microbatch; allocated
        # at F start, freed when B and W both finish) over the cap.  The
        # emitted schedule then keeps the same bound on real hardware.
        tuning = getattr(getattr(executor, "config", None), "tuning", None)
        self.max_inflight_layers = getattr(tuning, "max_inflight_layers", None) or 0
        if plan.layers_per_stage:
            self._stage_layer_weight = {
                i: max(1, n) for i, n in enumerate(plan.layers_per_stage)
            }
        else:
            self._stage_layer_weight = {
                i: max(1, len(stage.operator_ids))
                for i, stage in enumerate(plan.partition.stages)
            }
        self._inflight_layers = [0] * self.device_num
        self._act_pending: dict[tuple, list] = {}
        # Deadlock-free admission with reservations: a bare device-level cap
        # can deadlock (early stages hog the budget, blocking the F chain the
        # first B depends on).  Instead, one microbatch slot per stage is
        # reserved so the mid-0 F chain always completes; the remaining
        # budget is shared freely across the device's stages (uneven
        # distribution allowed, unlike a uniform per-stage quota).
        self._stage_inflight_mb: dict[int, int] = {}
        self.stall_flag = False
        self._last_progress_time = self.time
        self.peak_inflight_layers = 0
        self._init_devices()
        self._init_dynamic_ready_queues()
        self.num_finished = 0
        self.total_workload = self._estimate_total_workloads()

    def _estimate_total_workloads(self) -> int:
        if self.plan.static_schedule:
            return sum(len(row) for row in self.plan.static_schedule)
        count = 0
        for device in self.devices:
            for stage in device.stages.values():
                for wmap in stage.workloads.values():
                    count += len(wmap)
        return max(count, 1)

    def check_device_status(self, time: float) -> None:
        if self.plan.static_schedule:
            done = all(
                d.next_workload_idx >= len(self.plan.static_schedule[d.did])
                and d.state == Device.IDLE
                for d in self.devices
            )
            if done:
                self.finish_flag = True
            return
        if self.num_finished >= self.total_workload:
            all_idle = all(d.state == Device.IDLE for d in self.devices)
            if all_idle:
                self.finish_flag = True

    def _init_devices(self) -> None:
        placement = self.plan.placement.device_stages
        # Profiled timings are in 0.01 ms ticks; the empirical overheads below
        # are given in ms and converted here.
        overhead = self.hardware.workload_overhead_ms * 100.0
        comm_time = self.hardware.comm_alpha_us / 1000.0 + self.hardware.p2p_latency_ms * 100.0
        for did in range(self.device_num):
            dev = Device(
                device_idx=did,
                plan=self.plan,
                mid_offset=self.mid_offset,
                comp_power=self.hardware.comp_power,
                max_mem=self.hardware.gpu_hbm_gb,
                comm_time=comm_time,
                workload_overhead=overhead,
                runtime=self,
            )
            for sid in placement[did]:
                dev.add_stage(sid)
            self.devices.append(dev)

    def _init_dynamic_ready_queues(self) -> None:
        if self.plan.schedule != Schedule.OctoPipe:
            return
        for device in self.devices:
            for workload in device.get_initial_executable_workload(self.time):
                device.push_executable_workload(workload, self.time)

    def _act_weight(self, sid: int, mid: int) -> float:
        """In-flight activation weight of one microbatch on a stage.

        Base unit is one reference-shape transformer layer; variable-length
        microbatches scale it by their token count ratio (activation bytes
        are token-linear).
        """
        return self._stage_layer_weight.get(sid, 1) * self.plan.token_ratio_for_mid(mid)

    def f_admission_blocked(self, did: int, sid: int, mid: int) -> bool:
        if not self.max_inflight_layers:
            return False
        weight = self._act_weight(sid, mid)
        # First in-flight microbatch of a stage draws from its reservation:
        # admit whenever the raw cap allows, so the mid-0 chain never stalls.
        if self._stage_inflight_mb.get(sid, 0) == 0:
            return self._inflight_layers[did] + weight > self.max_inflight_layers
        # Additional microbatches must leave the reservations of this
        # device's still-empty stages untouched (reservations use the
        # reference-shape weight; the incoming microbatch is unknown).
        reserved = sum(
            self._stage_layer_weight.get(s, 1)
            for s in self.plan.placement.device_stages[did]
            if self._stage_inflight_mb.get(s, 0) == 0
        )
        return (
            self._inflight_layers[did] + weight + reserved
            > self.max_inflight_layers
        )

    def on_workload_started(self, workload: Workload) -> None:
        if not self.max_inflight_layers or workload.wtype != WorkloadType.F:
            return
        sid = workload.sid
        did = self.sid_to_did(sid)
        weight = self._act_weight(sid, workload.mid)
        self._inflight_layers[did] += weight
        self._stage_inflight_mb[sid] = self._stage_inflight_mb.get(sid, 0) + 1
        if self._inflight_layers[did] > self.peak_inflight_layers:
            self.peak_inflight_layers = self._inflight_layers[did]
        stage = self.devices[did].stages.get(sid)
        expected = 1
        if stage is not None:
            wmap = stage.workloads.get(WorkloadType.W)
            if wmap and workload.mid in wmap:
                expected = 2
        self._act_pending[(sid, workload.mid)] = [expected, did, weight]

    def _on_activation_consumer_finished(self, workload: Workload) -> None:
        if not self.max_inflight_layers or workload.wtype not in (
            WorkloadType.B,
            WorkloadType.W,
        ):
            return
        key = (workload.sid, workload.mid)
        entry = self._act_pending.get(key)
        if entry is None:
            return
        entry[0] -= 1
        if entry[0] <= 0:
            del self._act_pending[key]
            self._inflight_layers[entry[1]] -= entry[2]
            self._stage_inflight_mb[workload.sid] = max(
                0, self._stage_inflight_mb.get(workload.sid, 0) - 1
            )

    def sid_to_did(self, sid: int) -> int:
        for did, sids in enumerate(self.plan.placement.device_stages):
            if sid in sids:
                return did
        return 0

    def has_started_backward(self) -> bool:
        for device in self.devices:
            if device.current_workload and device.current_workload.wtype == WorkloadType.B:
                return True
            for workload in device.workload_execute_record:
                if workload.wtype == WorkloadType.B:
                    return True
        return False

    def is_overlap_exempt(self, workload: Workload) -> bool:
        mode = self.overlap_exempt_group_by.lower().replace("+", "_").replace("-", "_")
        if mode == "mid":
            key = (workload.mid,)
        elif mode in ("mid_type", "mid_wtype"):
            key = (workload.mid, workload.wtype)
        else:
            key = (workload.mid, workload.sid, workload.wtype)
        return key in self.overlap_exempt_workloads

    def check_workload_status(self, time: float) -> None:
        for device in self.devices:
            if device.state != Device.BUSY or not device.current_workload:
                continue
            w = device.current_workload
            if time < (w.end_time or 0):
                continue
            w.complete(time)
            if w.state == Workload.finished:
                device.state = Device.IDLE
                self.num_finished += 1
                self._on_activation_consumer_finished(w)
                self._propagate_constraints(w, time)
                device.current_workload = None

    def _propagate_constraints(self, completed: Workload, time: float) -> None:
        for device in self.devices:
            device.update_constraints_within_device(time, completed)

    def execute_workload(self, time: float) -> None:
        for device in self.devices:
            device.execute_workload(time)

    # No single workload takes anywhere near this long (ticks are 0.01 ms,
    # so this is 2 s of simulated time); exceeding it without finishing any
    # workload means the schedule is stuck, not slow.
    STALL_WINDOW = 200_000

    def _stalled(self) -> bool:
        if self.time - self._last_progress_time <= self.STALL_WINDOW:
            return False
        self.stall_flag = True
        return True

    def run(self, time_limit: int) -> int:
        last_finished = self.num_finished
        while self.time <= time_limit and not self.finish_flag:
            self.check_workload_status(self.time)
            self.execute_workload(self.time)
            self.check_device_status(self.time)
            if self.num_finished != last_finished:
                last_finished = self.num_finished
                self._last_progress_time = self.time
            elif self._stalled():
                break
            self.time += 1
        return self.time

    def run_discrete(self, time_limit: int) -> int:
        last_finished = self.num_finished
        while self.time <= time_limit and not self.finish_flag:
            self.check_workload_status(self.time)
            self.execute_workload(self.time)
            self.check_device_status(self.time)
            if self.num_finished != last_finished:
                last_finished = self.num_finished
                self._last_progress_time = self.time
            elif self._stalled():
                break
            next_t = self._next_tick(time_limit)
            if next_t <= self.time:
                self.time += 1
            else:
                self.time = next_t
        return self.time

    def _next_tick(self, time_limit: int) -> int:
        t0 = self.time
        ticks: set[int] = set()
        for device in self.devices:
            if device.current_workload and device.current_workload.end_time:
                et = int(math.ceil(device.current_workload.end_time))
                if et > t0:
                    ticks.add(et)
            for stage in device.stages.values():
                for wmap in stage.workloads.values():
                    for w in wmap.values():
                        if w.ready_time > t0:
                            ticks.add(int(w.ready_time))
        if not ticks:
            return min(t0 + 1, time_limit + 1)
        return min(min(ticks), time_limit + 1)

    def collect_results(self) -> dict:
        records = []
        for device in self.devices:
            for w in device.workload_execute_record:
                records.append(
                    {
                        "did": w.did,
                        "mid": w.mid,
                        "sid": w.sid,
                        "wtype": w.wtype.name,
                        "start": w.start_time,
                        "end": w.end_time,
                        "duration": w.duration,
                    }
                )
        makespan = max((r["end"] or 0 for r in records), default=0)
        return {"makespan": makespan, "records": records, "time": self.time}
