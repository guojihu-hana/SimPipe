from __future__ import annotations

from dataclasses import dataclass, replace

from simpipe.config.hardware import HardwareConfig
from simpipe.config.parallel import ParallelConfig
from simpipe.config.sim_config import SimConfig
from simpipe.core.runtime import PipelineRuntime
from simpipe.core.types import Schedule
from simpipe.core.types import WorkloadType
from simpipe.graph.model_graph import ModelGraph
from simpipe.memory.estimate import PipelineMemoryEstimate, estimate_pipeline_memory
from simpipe.pipeline.partition import OperatorPartition, layer_partition_to_stage_specs
from simpipe.pipeline.placement import Placement
from simpipe.pipeline.schedule_config import (
    apply_schedule_config,
    resolve_partition_layers,
    resolve_placement,
    resolve_schedule,
    split_layer_times_for_zbh,
)
from simpipe.pipeline.types import WorkloadPlan
from simpipe.pipeline.workload_gen import build_workload_plan


@dataclass
class SimulationResult:
    makespan: float
    records: list[dict]
    idle_per_device: list[float]
    memory: PipelineMemoryEstimate | None = None


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
        idle = [0.0] * self.plan.device_num
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
        )


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

    tune_top_results: list = []
    if (
        sched == Schedule.OctoPipe
        and partition_layers is None
        and placement is None
        and config.tuning.auto_tune
    ):
        from simpipe.tuning.octopipe_tune import tune_octopipe

        tuned = tune_octopipe(
            raw_config,
            layer_f_times or [],
            layer_b_times or [],
            layer_w_times,
            config.tuning,
            embedding_f_time=embedding_f_time,
            embedding_b_time=embedding_b_time,
            embedding_w_time=embedding_w_time,
            head_f_time=head_f_time,
            head_b_time=head_b_time,
            head_w_time=head_w_time,
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
            embedding_f_time=embedding_f_time,
            embedding_b_time=embedding_b_time,
            embedding_w_time=embedding_w_time,
            head_f_time=head_f_time,
            head_b_time=head_b_time,
            head_w_time=head_w_time,
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
    return executor
