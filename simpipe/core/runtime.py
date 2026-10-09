from __future__ import annotations

import heapq
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
        # Future wake-up ticks for run_discrete, registered when a workload
        # starts: its end (completion, same-device release) and end+comm
        # (cross-device release lands after the P2P latency).  Together these
        # cover every instant the pipeline state can change, so _next_tick
        # pops the heap instead of rescanning every workload's ready_time.
        self._event_heap: list[int] = []
        # set once the first B starts; replaces scanning execution records
        self._backward_started = False
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
        # Aux (encoder/decoder) dependency edges: completed-workload key ->
        # workloads waiting on it.  Backbone-internal edges keep using the
        # stage-adjacency propagation; every edge that touches an aux
        # workload is registered here instead.
        self._release_index: dict[WorkloadConstraint, list[tuple[Device, Workload]]] = {}
        # Encoder activation throttle: in-flight microbatches (F started,
        # B not finished) per aux stage id, capped by the aux plan's
        # encoder_inflight_limit so encoders don't stockpile activations
        # long before the backbone consumes them.
        self._aux_inflight: dict[int, int] = {}
        aux_plan = getattr(plan, "aux_plan", None)
        self._aux_inflight_limit = (
            aux_plan.encoder_inflight_limit if aux_plan is not None else None
        )
        self._init_devices()
        self._init_interleaved_comm_gates()
        self._init_aux()
        self._init_dynamic_ready_queues()
        self.num_finished = 0
        self.total_workload = self._estimate_total_workloads()

    def _estimate_total_workloads(self) -> int:
        aux_count = sum(d.aux_remaining for d in self.devices)
        if self.plan.static_schedule:
            return sum(len(row) for row in self.plan.static_schedule) + aux_count
        count = aux_count
        for device in self.devices:
            for stage in device.stages.values():
                for wmap in stage.workloads.values():
                    count += len(wmap)
        return max(count, 1)

    def check_device_status(self, time: float) -> None:
        if self.plan.static_schedule:
            done = all(
                (not d.static_schedule or d.next_workload_idx >= len(d.static_schedule))
                and d.state == Device.IDLE
                and d.aux_remaining <= 0
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
        # Profiled timings are in 0.01 ms (= 10 us) ticks; the empirical
        # overheads below are given in ms / us and converted to ticks here.
        # comm_time delays every cross-device dependency edge; with a whole
        # simulation tick clock any fractional arrival rounds up, so e.g. the
        # default 5 us alpha shows up as one full tick between an F's end and
        # the next stage's F start.  Set both to 0 for back-to-back edges.
        overhead = self.hardware.workload_overhead_ms * 100.0
        comm_time = self.hardware.comm_alpha_us / 10.0 + self.hardware.p2p_latency_ms * 100.0
        self._comm_time = comm_time
        self._workload_overhead = overhead
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
        # Constraint edges only connect adjacent stages, so completions are
        # delivered to sid-1/sid/sid+1 directly instead of broadcast to every
        # stage of every device (each sid lives on exactly one device).
        self._stage_index = {
            sid: (device, stage)
            for device in self.devices
            for sid, stage in device.stages.items()
        }

    def _init_interleaved_comm_gates(self) -> None:
        """Model interleaved 1F1B's coupled steady-state communication.

        Megatron's interleaved schedule exchanges activations with batched
        send-recv calls: after each steady-state (F, B) compute pair the rank
        posts one p2p that sends both results and receives the next F's input
        together with the next B's gradient.  A steady F therefore cannot
        start before the gradient consumed by its step-partner B exists
        downstream -- adjacent ranks stay staggered by one full (F, B) step.
        Plain per-op ASAP timing instead lets the F float early (its input
        arrived long ago) and packs each B head-to-tail after the downstream
        B, which is not how the synchronous schedule behaves on hardware.

        Encode the coupling as an extra dependency: every post-warmup F also
        waits for the downstream-gradient producers of the B that immediately
        follows it in the device's schedule.  Warmup Fs (no B partner before
        them) and drain Bs (no F left to pair) keep plain dependencies, and
        a last-stage B contributes nothing (its "gradient" is its own F, a
        same-device edge).  Releases go through _release_index because the
        stage-adjacency propagation only walks sid +/- 1 of the completed
        workload, which does not reach the F's stage when the paired B
        belongs to a different chunk.
        """
        if self.plan.schedule != Schedule.INTERLEAVED or not self.plan.static_schedule:
            return
        for did, row in enumerate(self.plan.static_schedule):
            if did >= len(self.devices):
                break
            device = self.devices[did]
            # The generator's warmup count (InterleavedStrategy.generate).
            # When micro_batch_num is small every F fits in the warmup quota
            # and the device has no steady phase at all: its Fs then all run
            # up front and its Bs drain afterwards.  Pairing the last such F
            # with the first drain B would gate an *earlier* list entry on a
            # *later* neighbour step and deadlock the fill (each rank's last
            # warmup F waiting on the next rank's first B, which sits behind
            # that rank's own gated F).  Only steady-phase Fs are paired.
            chunk_num = len(self.plan.placement.device_stages[did])
            warmup = (chunk_num - 1) * self.device_num + (
                self.device_num - did - 1
            ) * 2
            f_seen = 0
            pending_f: Workload | None = None
            for wtype, local_mid, sid in row:
                stage = device.stages.get(sid)
                if stage is None:
                    continue
                mid = local_mid + self.mid_offset
                if wtype == WorkloadType.F:
                    # overwrite: only the F directly preceding a B is its
                    # step partner; warmup Fs stay ungated
                    f_seen += 1
                    pending_f = (
                        stage.get_workload(mid, WorkloadType.F)
                        if f_seen > warmup
                        else None
                    )
                elif wtype == WorkloadType.B:
                    b_w = stage.get_workload(mid, WorkloadType.B)
                    if pending_f is not None and b_w is not None:
                        for cstr in b_w.constraints:
                            if cstr.stage_id == b_w.sid + 1:
                                pending_f.constraints.add(cstr)
                                self._release_index.setdefault(cstr, []).append(
                                    (device, pending_f)
                                )
                    pending_f = None

    def _init_aux(self) -> None:
        """Materialize encoder/decoder workloads and wire their dependencies.

        Data flow per microbatch: every encoder F -> backbone stage-0 F ->
        ... -> last-stage F -> every decoder F -> decoder B -> last-stage B
        -> ... -> stage-0 B -> every encoder B (aux W follows its own B when
        bwd_split is on).  Multi-copy placements shard microbatches over the
        copies, so each edge binds the one copy owning that microbatch to the
        pipeline boundary stage.  All these edges are plain
        WorkloadConstraints: cross-device ones pick up the P2P latency in
        Workload.update_constraints, and completions are delivered through
        self._release_index (see _propagate_constraints).
        """
        aux_plan = getattr(self.plan, "aux_plan", None)
        if aux_plan is None:
            return
        from simpipe.pipeline.aux_modules import DECODER, ENCODER

        # Dedicated aux devices continue the backbone's device ids.
        for extra in range(aux_plan.extra_device_num):
            self.devices.append(
                Device(
                    device_idx=self.device_num + extra,
                    plan=self.plan,
                    mid_offset=self.mid_offset,
                    comp_power=self.hardware.comp_power,
                    max_mem=self.hardware.gpu_hbm_gb,
                    comm_time=self._comm_time,
                    workload_overhead=self._workload_overhead,
                    runtime=self,
                )
            )
        dev_by_id = {d.did: d for d in self.devices}
        stage_num = self.plan.stage_num
        last_sid = stage_num - 1
        placement = self.plan.placement.device_stages

        def register(cstr: WorkloadConstraint, device: Device, workload: Workload) -> None:
            self._release_index.setdefault(cstr, []).append((device, workload))

        def gate(backbone_w: Workload, dev: Device, cstr: WorkloadConstraint) -> None:
            """Make a backbone workload wait for an aux completion."""
            backbone_w.constraints.add(cstr)
            register(cstr, dev, backbone_w)

        # Static schedules (everything but OctoPipe) splice anchored copies
        # into the device's entry list so every microbatch runs
        # enc F -> bb F ... bb B -> enc B in that order: encoder F right
        # before the stage-0 F (no early warmup runs), encoder B right after
        # the stage-0 B (later backbone Fs cannot push it back), decoder F+B
        # right after the last-stage F.  A copy is anchored when it sits on
        # the same device as its boundary stage; dedicated devices and
        # off-anchor shards (replicated / explicit lists) keep backfilling.
        splice_before: dict[int, dict[tuple, list[tuple]]] = {}
        splice_after: dict[int, dict[tuple, list[tuple]]] = {}

        for inst in aux_plan.instances:
            dev = dev_by_id[inst.device_id]
            b_ticks = inst.b_ticks if self.bwd_split else inst.b_ticks + inst.w_ticks
            w_ticks = inst.w_ticks if self.bwd_split else 0.0
            # encoders feed backbone stage 0, decoders sit behind the last
            # stage; the copy only wires the microbatches in its shard
            attach_sids = [0] if inst.role == ENCODER else [last_sid]
            anchor_sid = attach_sids[0]
            anchored = (
                dev.static_schedule is not None
                and self._stage_index[anchor_sid][0].did == inst.device_id
            )

            for local_mid in inst.mids:
                mid = local_mid + self.mid_offset

                def make(wtype: WorkloadType, duration: float) -> Workload:
                    w = Workload(
                        schedule_method=self.plan.schedule,
                        device_idx=inst.device_id,
                        microbatch_idx=mid,
                        stage_idx=inst.aux_sid,
                        bwd_split=self.bwd_split,
                        duration=duration,
                        total_stage_num=stage_num,
                        wtype=wtype,
                        recomp=False,
                        split_recomp=False,
                        comp_power=self.hardware.comp_power,
                        vocab_parallel=False,
                        placement=placement,
                        comm_time=self._comm_time,
                    )
                    # aux dependencies are wired explicitly below
                    w.constraints.clear()
                    w.is_aux = True
                    w.aux_name = inst.name
                    w.aux_role = inst.role
                    return w

                # Variable batches scale aux blocks with the microbatch's
                # token count (linear): the module runs as one opaque block,
                # so no per-op attention split applies (that quadratic term
                # only exists inside the backbone's stage timings).
                lin = self.plan.token_ratio_for_mid(mid)
                f_w = make(WorkloadType.F, inst.f_ticks * lin)
                b_w = make(WorkloadType.B, b_ticks * lin)
                w_w = make(WorkloadType.W, w_ticks * lin) if w_ticks > 0 else None
                f_done = WorkloadConstraint(inst.device_id, mid, inst.aux_sid, WorkloadType.F)
                b_done = WorkloadConstraint(inst.device_id, mid, inst.aux_sid, WorkloadType.B)

                if inst.role == ENCODER:
                    # encoder F has no upstream; backbone F waits for it
                    for sid in attach_sids:
                        s_dev, s_stage = self._stage_index[sid]
                        bf = s_stage.get_workload(mid, WorkloadType.F)
                        if bf is not None:
                            gate(bf, s_dev, f_done)
                        # encoder B waits for the attached stage's B (input
                        # gradients flow back over the same link)
                        bb = s_stage.get_workload(mid, WorkloadType.B)
                        if bb is not None:
                            cstr = WorkloadConstraint(s_dev.did, mid, sid, WorkloadType.B)
                            b_w.constraints.add(cstr)
                            register(cstr, dev, b_w)
                else:  # DECODER
                    for sid in attach_sids:
                        s_dev, s_stage = self._stage_index[sid]
                        # decoder F waits for the attached stage's F
                        bf = s_stage.get_workload(mid, WorkloadType.F)
                        if bf is not None:
                            cstr = WorkloadConstraint(s_dev.did, mid, sid, WorkloadType.F)
                            f_w.constraints.add(cstr)
                            register(cstr, dev, f_w)
                        # the attached stage's B waits for decoder B (the
                        # loss sits behind the decoders)
                        bb = s_stage.get_workload(mid, WorkloadType.B)
                        if bb is not None:
                            gate(bb, s_dev, b_done)
                    # loss turnaround inside the decoder: B after own F
                    b_w.constraints.add(f_done)
                    register(f_done, dev, b_w)

                if inst.role == ENCODER:
                    # encoder B additionally needs its own F to have run
                    b_w.constraints.add(f_done)
                    register(f_done, dev, b_w)
                if w_w is not None:
                    w_w.constraints.add(b_done)
                    register(b_done, dev, w_w)

                if anchored:
                    before = splice_before.setdefault(inst.device_id, {})
                    after = splice_after.setdefault(inst.device_id, {})
                    f_e = (WorkloadType.F, local_mid, inst.aux_sid)
                    b_e = (WorkloadType.B, local_mid, inst.aux_sid)
                    if inst.role == ENCODER:
                        before.setdefault(
                            (WorkloadType.F, local_mid, anchor_sid), []
                        ).append(f_e)
                        after.setdefault(
                            (WorkloadType.B, local_mid, anchor_sid), []
                        ).append(b_e)
                    else:
                        # decoder F right after bb F; decoder B right before
                        # bb B, so it runs in the backward phase (on AFAB:
                        # AF does the Fs, AB the Bs) instead of eagerly
                        # after its own F
                        after.setdefault(
                            (WorkloadType.F, local_mid, anchor_sid), []
                        ).append(f_e)
                        before.setdefault(
                            (WorkloadType.B, local_mid, anchor_sid), []
                        ).append(b_e)
                    for w in (f_w, b_w):
                        w.aux_static_entry = True
                        dev.add_aux_workload(w, static_entry=True)
                    # aux W has no downstream: it backfills idle slots
                    # instead of taking a spliced slot on the critical chain
                    if w_w is not None:
                        dev.add_aux_workload(w_w)
                else:
                    dev.add_aux_workload(f_w)
                    dev.add_aux_workload(b_w)
                    if w_w is not None:
                        dev.add_aux_workload(w_w)

        # rebuild the anchored devices' entry lists with the splices in place
        for did in set(splice_before) | set(splice_after):
            dev = dev_by_id[did]
            before = splice_before.get(did, {})
            after = splice_after.get(did, {})
            rebuilt: list[tuple] = []
            for entry in dev.static_schedule:
                rebuilt.extend(before.pop(entry, ()))
                rebuilt.append(entry)
                rebuilt.extend(after.pop(entry, ()))
            if before or after:  # anchor entries missing from the schedule
                raise RuntimeError(
                    f"aux splice anchors not found in device {did}'s static "
                    f"schedule: {list(before) + list(after)}"
                )
            dev.static_schedule = rebuilt

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
        end = workload.end_time or 0.0
        heapq.heappush(self._event_heap, int(math.ceil(end)))
        if workload.comm_time:
            heapq.heappush(self._event_heap, int(math.ceil(end + workload.comm_time)))
        if workload.wtype == WorkloadType.B:
            self._backward_started = True
        if getattr(workload, "is_aux", False):
            # encoder activations become resident at F start (throttled via
            # aux_f_blocked); aux modules skip the backbone's admission
            if (
                workload.wtype == WorkloadType.F
                and getattr(workload, "aux_role", None) == "encoder"
            ):
                sid = workload.sid
                self._aux_inflight[sid] = self._aux_inflight.get(sid, 0) + 1
            return
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
        return self._backward_started

    def constraint_finish_eta(self, constraint: WorkloadConstraint) -> float | None:
        """Finish time of the workload a constraint waits on, when already
        known: the producer is currently executing (its end_time is fixed).
        None = unknown (not started yet, or not the producer's turn)."""
        if constraint.device_id >= len(self.devices):
            return None
        w = self.devices[constraint.device_id].current_workload
        if (
            w is not None
            and w.state == Workload.in_progress
            and w.mid == constraint.microbatch_id
            and w.sid == constraint.stage_id
            and w.wtype == constraint.workload_type
        ):
            return w.end_time
        return None

    def aux_f_blocked(self, workload: Workload) -> bool:
        """Encoder-F throttle for backfilling copies: block once the copy has
        limit activations alive.  A blocked F waits for an encoder B, which
        follows the backbone's backward; _tune_aux_inflight only keeps caps
        that reproduce the uncapped makespan (a too-small cap deadlocks and
        reports stalled).  Statically spliced copies are exempt: their Fs
        already run at the latest useful moment (right before the backbone F
        they feed), so their in-flight peak is fixed by the schedule shape
        and a cap could only deadlock it (e.g. AFAB legitimately keeps every
        microbatch's encoder activation alive through the AF phase)."""
        if self._aux_inflight_limit is None:
            return False
        if getattr(workload, "aux_static_entry", False):
            return False
        if getattr(workload, "aux_role", None) != "encoder":
            return False
        if workload.wtype != WorkloadType.F:
            return False
        return self._aux_inflight.get(workload.sid, 0) >= self._aux_inflight_limit

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
                if getattr(w, "is_aux", False):
                    device.aux_remaining -= 1
                    # encoder B frees the microbatch's encoder activation
                    if (
                        w.wtype == WorkloadType.B
                        and getattr(w, "aux_role", None) == "encoder"
                    ):
                        self._aux_inflight[w.sid] = self._aux_inflight.get(w.sid, 1) - 1
                self._on_activation_consumer_finished(w)
                self._propagate_constraints(w, time)
                device.current_workload = None

    def _propagate_constraints(self, completed: Workload, time: float) -> None:
        for sid in (completed.sid - 1, completed.sid, completed.sid + 1):
            entry = self._stage_index.get(sid)
            if entry is None:
                continue
            device, stage = entry
            for w in stage.update_constraints_within_stage(time, completed):
                if (
                    device.schedule_method == Schedule.OctoPipe
                    and w.state == Workload.not_started
                    and len(w.constraints) == 0
                ):
                    device.executable_workloads.push(w)
        if self._release_index:
            cstr = WorkloadConstraint(
                completed.did, completed.mid, completed.sid, completed.wtype
            )
            waiters = self._release_index.get(cstr)
            if waiters:
                for device, w in waiters:
                    w.update_constraints(time, cstr)
                    if w.state != Workload.not_started or w.constraints:
                        continue
                    if getattr(w, "is_aux", False):
                        # statically spliced aux entries run at their slot in
                        # static_schedule; only backfilling copies queue here
                        if not getattr(w, "aux_static_entry", False):
                            device.push_aux_ready(w)
                    elif device.schedule_method == Schedule.OctoPipe:
                        device.executable_workloads.push(w)

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
        heap = self._event_heap
        t0 = self.time
        while heap and heap[0] <= t0:
            heapq.heappop(heap)
        if not heap:
            return min(t0 + 1, time_limit + 1)
        return min(heap[0], time_limit + 1)

    def collect_results(self) -> dict:
        records = []
        for device in self.devices:
            for w in device.workload_execute_record:
                rec = {
                    "did": w.did,
                    "mid": w.mid,
                    "sid": w.sid,
                    "wtype": w.wtype.name,
                    "start": w.start_time,
                    "end": w.end_time,
                    "duration": w.duration,
                }
                aux_name = getattr(w, "aux_name", None)
                if aux_name is not None:
                    rec["aux"] = aux_name
                    rec["role"] = getattr(w, "aux_role", "")
                records.append(rec)
        makespan = max((r["end"] or 0 for r in records), default=0)
        return {"makespan": makespan, "records": records, "time": self.time}
