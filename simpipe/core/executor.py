from __future__ import annotations

from dataclasses import dataclass, replace

from simpipe.config.sim_config import SimConfig
from simpipe.core.runtime import PipelineRuntime
from simpipe.core.types import Schedule
from simpipe.graph.model_graph import ModelGraph
from simpipe.memory.estimate import (PipelineMemoryEstimate,
                                     estimate_pipeline_memory)
from simpipe.pipeline.partition import layer_partition_to_stage_specs
from simpipe.pipeline.schedule_config import (apply_schedule_config,
                                              resolve_partition_layers,
                                              resolve_placement,
                                              resolve_schedule,
                                              split_layer_times_for_zbh)
from simpipe.pipeline.types import WorkloadPlan
from simpipe.pipeline.workload_gen import build_workload_plan


@dataclass
class SimulationResult:
    makespan: float
    records: list[dict]
    idle_per_device: list[float]
    memory: PipelineMemoryEstimate | None = None
    stalled: bool = False
    peak_inflight_layers: float = 0.0


class Executor:
    def __init__(
        self,
        config: SimConfig,
        graph: ModelGraph,
        plan: WorkloadPlan,
        discrete_event_time: bool | None = None,
        overlap_exempt_workloads: set[tuple] | None = None,
        overlap_exempt_group_by: str = "mid_type",
        bubble_overlap_trials: list | None = None,
    ):
        self.config = config
        self.graph = graph
        self.plan = plan
        self.discrete_event_time = (
            discrete_event_time
            if discrete_event_time is not None
            else config.discrete_event_time
        )
        self.pipelines: list[PipelineRuntime] = []
        self.overlap_exempt_workloads = overlap_exempt_workloads or set()
        self.overlap_exempt_group_by = overlap_exempt_group_by
        self.bubble_overlap_trials = bubble_overlap_trials or []
        self.tune_top_results: list = []
        self.batch_order_result = None
        self.time = 0
        self._init_pipelines()

    def _init_pipelines(self) -> None:
        for dp_idx in range(self.config.parallel.dp_size):
            mid_offset = dp_idx * self.config.parallel.micro_batch_num
            self.pipelines.append(
                PipelineRuntime(
                    plan=self.plan,
                    parallel=self.config.parallel,
                    hardware=self.config.hardware,
                    pipeline_idx=dp_idx,
                    mid_offset=mid_offset,
                    executor=self,
                    overlap_exempt_workloads=self.overlap_exempt_workloads,
                    overlap_exempt_group_by=self.overlap_exempt_group_by,
                )
            )

    def run(self, time_limit: int | None = None) -> SimulationResult:
        limit = time_limit or self.config.time_limit
        max_time = 0
        all_records: list[dict] = []
        aux_plan = getattr(self.plan, "aux_plan", None)
        extra_devices = aux_plan.extra_device_num if aux_plan else 0
        idle = [0.0] * (self.plan.device_num + extra_devices)
        for pipeline in self.pipelines:
            if self.discrete_event_time:
                t = pipeline.run_discrete(limit)
            else:
                t = pipeline.run(limit)
            max_time = max(max_time, t)
            res = pipeline.collect_results()
            all_records.extend(res["records"])
            for d in pipeline.devices:
                idle[d.did] += d.idle_time
        makespan = max((r.get("end") or 0 for r in all_records), default=0)
        memory = estimate_pipeline_memory(
            self.graph,
            self.plan,
            self.config.parallel,
            self.config.hardware,
            all_records,
        )
        return SimulationResult(
            makespan=makespan,
            records=all_records,
            idle_per_device=idle,
            memory=memory,
            stalled=any(getattr(p, "stall_flag", False) for p in self.pipelines),
            peak_inflight_layers=max(
                (getattr(p, "peak_inflight_layers", 0) for p in self.pipelines),
                default=0,
            ),
        )


def first_replica_records(records: list[dict], micro_batch_num: int) -> list[dict]:
    """Records of the first data-parallel replica (mids 0..nmb-1).

    DP replicas repeat the same schedule with mids offset by dp_idx * nmb
    and their records share device ids, so plots and bubble stats would
    show every block dp_size times; one copy carries all the information.
    """
    return [r for r in records if int(r["mid"]) < micro_batch_num]


def build_simulation(
    config: SimConfig,
    layer_f_times: list[float] | None = None,
    layer_b_times: list[float] | None = None,
    layer_w_times: list[float] | None = None,
    embedding_f_time: float | None = None,
    embedding_b_time: float | None = None,
    embedding_w_time: float | None = None,
    head_f_time: float | None = None,
    head_b_time: float | None = None,
    head_w_time: float | None = None,
    partition_layers: list[int] | None = None,
    placement: list[list[int]] | None = None,
    schedule: Schedule | None = None,
    overlap_exempt_workloads: set[tuple] | None = None,
    overlap_exempt_group_by: str = "mid_type",
    bubble_overlap_trials: list | None = None,
) -> Executor:
    sched = resolve_schedule(config, schedule)
    raw_config = replace(config, schedule="octopipe") if sched == Schedule.OctoPipe else config
    config = apply_schedule_config(config, sched)
    layer_f_times, layer_b_times, layer_w_times = split_layer_times_for_zbh(
        sched, layer_f_times, layer_b_times, layer_w_times
    )

    from simpipe.models.registry import layer_symbols_for_model_config

    layer_symbols = layer_symbols_for_model_config(config.model)
    graph = ModelGraph.from_config(config.model)
    pp = config.parallel
    if layer_f_times:
        graph.distribute_profile_times(
            layer_f_times,
            layer_b_times,
            layer_w_times,
            embedding_f_time=embedding_f_time,
            embedding_b_time=embedding_b_time,
            embedding_w_time=embedding_w_time,
            head_f_time=head_f_time,
            head_b_time=head_b_time,
            head_w_time=head_w_time,
        )

    # Aux modules pinned to the first/last-stage device compete with that
    # boundary stage for compute, so the partition search must account for
    # them; fold their per-microbatch cost into the embedding/head knobs the
    # search already understands.  Tune-only: build_workload_plan below keeps
    # the original values, aux compute stays a separate simulated workload.
    # Replicated/dedicated/explicit placements don't skew any single stage
    # (uniform or off-pipeline load), so nothing is folded for them.
    tune_emb = [embedding_f_time, embedding_b_time, embedding_w_time]
    tune_head = [head_f_time, head_b_time, head_w_time]
    if (config.encoders or config.decoders) and layer_f_times:
        from simpipe.config.multimodal import FIRST_STAGE, LAST_STAGE

        aux_all = list(config.encoders) + list(config.decoders)
        bwd_split = config.parallel.bwd_split
        for target, place in ((tune_emb, FIRST_STAGE), (tune_head, LAST_STAGE)):
            mods = [m for m in aux_all if m.placement == place]
            if not mods:
                continue
            f_x = sum(m.f_ticks for m in mods)
            b_x = sum(m.b_ticks for m in mods)
            w_x = sum(m.w_ticks for m in mods)
            if not bwd_split:
                b_x += w_x
                w_x = 0.0
            target[0] = (target[0] or 0.0) + f_x
            target[1] = (target[1] or 0.0) + b_x
            target[2] = (target[2] or 0.0) + w_x

    tune_top_results: list = []
    if (
        sched == Schedule.OctoPipe
        and partition_layers is None
        and placement is None
        and config.tuning.auto_tune
    ):
        from simpipe.tuning.octopipe_tune import tune_octopipe

        # Without per-layer profile times the partition search would index
        # an empty list; uniform times make the search fall back to an even
        # split (tune-only: the workload plan keeps its analytic timings).
        tune_f = layer_f_times or [1.0] * config.model.num_layers
        tune_b = layer_b_times or [1.0] * config.model.num_layers
        tuned = tune_octopipe(
            raw_config,
            tune_f,
            tune_b,
            layer_w_times,
            config.tuning,
            embedding_f_time=tune_emb[0],
            embedding_b_time=tune_emb[1],
            embedding_w_time=tune_emb[2],
            head_f_time=tune_head[0],
            head_b_time=tune_head[1],
            head_w_time=tune_head[2],
        )
        partition_layers = tuned.partition_layers
        placement = tuned.placement
        overlap_exempt_workloads = tuned.overlap_exempt_workloads
        bubble_overlap_trials = tuned.bubble_overlap_trials
        tune_top_results = tuned.top_results
    else:
        partition_layers = resolve_partition_layers(
            config,
            sched,
            partition_layers,
            layer_f_times=layer_f_times,
            layer_b_times=layer_b_times,
            layer_w_times=layer_w_times,
            embedding_f_time=tune_emb[0],
            embedding_b_time=tune_emb[1],
            embedding_w_time=tune_emb[2],
            head_f_time=tune_head[0],
            head_b_time=tune_head[1],
            head_w_time=tune_head[2],
        )
        tune_top_results = []

    partition = layer_partition_to_stage_specs(graph, partition_layers)
    pl = resolve_placement(sched, pp.device_num, pp.chunk_num, placement)
    plan = build_workload_plan(
        graph,
        partition,
        pl,
        pp,
        config.hardware,
        sched,
        layer_f_times,
        layer_b_times,
        layer_w_times,
        embedding_f_time,
        embedding_b_time,
        embedding_w_time,
        head_f_time,
        head_b_time,
        head_w_time,
        layer_symbols=layer_symbols,
    )
    plan.layers_per_stage = list(partition_layers)
    if config.encoders or config.decoders:
        from simpipe.pipeline.aux_modules import build_aux_plan

        plan.aux_plan = build_aux_plan(
            config.encoders,
            config.decoders,
            pl.device_stages,
            partition.num_stages,
            pp.micro_batch_num,
        )
    batch_order_result = None
    if config.batch is not None:
        plan.mid_scales = config.batch.scales(
            config.model.micro_batch_size, config.model.seq_len
        )
        order_tune = config.tuning.batch_order_tune
        if order_tune is None:
            order_tune = config.tuning.auto_tune
        if order_tune and len(set(plan.mid_scales)) > 1:
            from simpipe.tuning.batch_order import tune_batch_order

            scales = list(plan.mid_scales)

            def _evaluate(order: list[int]) -> float:
                trial_plan = replace(
                    plan, mid_scales=[scales[i] for i in order]
                )
                result = Executor(
                    config,
                    graph,
                    trial_plan,
                    overlap_exempt_workloads=overlap_exempt_workloads,
                    overlap_exempt_group_by=overlap_exempt_group_by,
                ).run()
                return float("inf") if result.stalled else result.makespan

            batch_order_result = tune_batch_order(
                scales, _evaluate, config.tuning.batch_order_max_sims
            )
            if not batch_order_result.is_identity:
                plan.mid_scales = [scales[i] for i in batch_order_result.order]
                plan.mid_order = list(batch_order_result.order)
    if plan.aux_plan is not None and _has_backfilling_encoder(plan):
        if config.tuning.aux_inflight_limit:
            plan.aux_plan.encoder_inflight_limit = config.tuning.aux_inflight_limit
        elif config.tuning.aux_memory_opt:
            _tune_aux_inflight(
                config,
                graph,
                plan,
                overlap_exempt_workloads,
                overlap_exempt_group_by,
            )
    executor = Executor(
        config,
        graph,
        plan,
        overlap_exempt_workloads=overlap_exempt_workloads,
        overlap_exempt_group_by=overlap_exempt_group_by,
        bubble_overlap_trials=bubble_overlap_trials,
    )
    executor.tune_top_results = tune_top_results
    executor.batch_order_result = batch_order_result
    return executor


def _has_backfilling_encoder(plan: WorkloadPlan) -> bool:
    """True when some encoder copy schedules by backfilling (so the in-flight
    throttle can shape it).  On static schedules, copies anchored to stage 0's
    device are spliced into the entry list and run at the latest useful
    moment already -- throttling them is a no-op or a deadlock (AFAB), so a
    plan whose encoders are all spliced skips the cap search entirely."""
    encoders = [i for i in plan.aux_plan.instances if i.role == "encoder"]
    if not encoders:
        return False
    if plan.schedule == Schedule.OctoPipe or not plan.static_schedule:
        return True
    stage0_did = next(
        did for did, sids in enumerate(plan.placement.device_stages) if 0 in sids
    )
    return any(inst.device_id != stage0_did for inst in encoders)


def _tune_aux_inflight(
    config: SimConfig,
    graph: ModelGraph,
    plan: WorkloadPlan,
    overlap_exempt_workloads: set[tuple] | None,
    overlap_exempt_group_by: str,
) -> None:
    """Smallest encoder in-flight cap that keeps the uncapped makespan.

    Without a cap, encoders run as early as idle slots allow and stockpile
    activations the backbone only consumes much later (a dedicated encoder
    finishes all its microbatches upfront: peak = micro_batch_num).  Capping
    the alive microbatches (F started, B unfinished) delays encoder Fs into
    later idle slots, cutting the activation peak; the search keeps only
    caps whose makespan matches the uncapped baseline, so the bubble stays
    unchanged.  Doubling search from pp_size+1 (the warmup burst wants about
    one activation per pipeline stage) costs ~log2 extra simulations; the cap
    at micro_batch_num is a no-op, so it always terminates.
    """

    def sim() -> float:
        result = Executor(
            config,
            graph,
            plan,
            overlap_exempt_workloads=overlap_exempt_workloads,
            overlap_exempt_group_by=overlap_exempt_group_by,
        ).run()
        return float("inf") if result.stalled else result.makespan

    plan.aux_plan.encoder_inflight_limit = None
    base = sim()
    nmb = config.parallel.micro_batch_num
    k = min(nmb, config.parallel.pp_size + 1)
    while True:
        plan.aux_plan.encoder_inflight_limit = k
        if sim() <= base:
            return
        if k >= nmb:  # defensive; a cap of nmb can never block anything
            plan.aux_plan.encoder_inflight_limit = None
            return
        k = min(nmb, k * 2)
