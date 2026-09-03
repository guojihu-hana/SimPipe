from __future__ import annotations

import math
from dataclasses import dataclass, field

from simpipe.config.sim_config import SimConfig
from simpipe.config.tuning import TuningConfig
from simpipe.core.executor import build_simulation
from simpipe.core.types import Schedule, parse_schedule
from simpipe.metrics.comp_bubble import analyze_pipeline_comp_bubble, tuning_score
from simpipe.pipeline.schedule_config import resolve_chunk_num
from simpipe.tuning.bubble_overlap import tune_overlap_exemptions
from simpipe.tuning.fast_est import (
    generate_octopipe_placement_candidates,
    placement_proxy_score,
)
from simpipe.tuning.partition_search import (
    legal_chunk_values,
    legal_stage_nums,
    partition_variance,
    stage_times_for_partition,
    top_partitions_by_stage_variance,
)


@dataclass
class TuneCandidateAnalysis:
    rank: int
    chunk_num: int
    partition_layers: list[int]
    placement: list[list[int]]
    partition_variance: float
    makespan: float
    score: float
    comp_bubble: dict
    overlap_exempt_workloads: set[tuple] = field(default_factory=set)
    bubble_overlap_trials: list = field(default_factory=list)


@dataclass
class OctoPipeTuneResult:
    partition_layers: list[int]
    placement: list[list[int]]
    makespan: float
    overlap_exempt_workloads: set[tuple] = field(default_factory=set)
    bubble_overlap_trials: list = field(default_factory=list)
    top_results: list[TuneCandidateAnalysis] = field(default_factory=list)


def _embedding_head_times(
    embedding_f_time: float | None,
    embedding_b_time: float | None,
    embedding_w_time: float | None,
    head_f_time: float | None,
    head_b_time: float | None,
    head_w_time: float | None,
) -> tuple[float, float, float, float, float, float]:
    return (
        embedding_f_time or 0.0,
        embedding_b_time or 0.0,
        0.0 if embedding_w_time is None else float(embedding_w_time),
        head_f_time or 0.0,
        head_b_time or 0.0,
        0.0 if head_w_time is None else float(head_w_time),
    )


def _evaluate_candidate(
    config: SimConfig,
    partition_layers: list[int],
    placement: list[list[int]],
    layer_f_times: list[float],
    layer_b_times: list[float],
    layer_w_times: list[float] | None,
    tuning: TuningConfig,
    *,
    chunk_num: int,
    embedding_f_time: float = 0.0,
    embedding_b_time: float = 0.0,
    embedding_w_time: float = 0.0,
    head_f_time: float = 0.0,
    head_b_time: float = 0.0,
    head_w_time: float = 0.0,
) -> tuple[float, list[dict], set[tuple], list, dict, float]:
    overlap_exempt: set[tuple] = set()
    trials: list = []

    def run_simulation(
        exemptions: set[tuple] | None = None,
    ) -> tuple[float, list[dict], float]:
        executor = build_simulation(
            config,
            layer_f_times=layer_f_times,
            layer_b_times=layer_b_times,
            layer_w_times=layer_w_times,
            embedding_f_time=embedding_f_time,
            embedding_b_time=embedding_b_time,
            embedding_w_time=embedding_w_time,
            head_f_time=head_f_time,
            head_b_time=head_b_time,
            head_w_time=head_w_time,
            partition_layers=partition_layers,
            placement=placement,
            schedule=Schedule.OctoPipe,
            overlap_exempt_workloads=exemptions,
            overlap_exempt_group_by=tuning.bubble_overlap_group_by,
        )
        result = executor.run()
        if result.stalled:
            return float("inf"), result.records, result.peak_inflight_layers
        return result.makespan, result.records, result.peak_inflight_layers

    if tuning.bubble_overlap_tune:

        def run_with_exemptions(exemptions: set[tuple]) -> list[dict]:
            _makespan, records, _peak = run_simulation(exemptions)
            return records

        initial_records = run_with_exemptions(set())
        tune_result = tune_overlap_exemptions(
            initial_records,
            run_with_exemptions,
            tuning.bubble_overlap_max_iter,
            group_by=tuning.bubble_overlap_group_by,
        )
        overlap_exempt = tune_result.exemptions
        trials = tune_result.trials
        makespan, records, peak_inflight = run_simulation(overlap_exempt)
    else:
        makespan, records, peak_inflight = run_simulation(None)

    stats = analyze_pipeline_comp_bubble(records, device_num=config.parallel.pp_size)
    if math.isinf(makespan):
        # Simulation stalled (schedule cannot make progress under the
        # activation cap); treat as infeasible rather than grinding on.
        return float("inf"), records, overlap_exempt, trials, stats.to_dict(), float("inf")
    score = tuning_score(stats, makespan)
    if tuning.max_inflight_layers:
        # Runtime-tracked per-device peak (record-based accounting would
        # double count DP replicas, which share device ids in the records).
        if peak_inflight > tuning.max_inflight_layers:
            # Would OOM on real hardware: too many concurrent activations.
            return float("inf"), records, overlap_exempt, trials, stats.to_dict(), float("inf")
    return makespan, records, overlap_exempt, trials, stats.to_dict(), score


def tune_octopipe(
    config: SimConfig,
    layer_f_times: list[float],
    layer_b_times: list[float],
    layer_w_times: list[float] | None,
    tuning: TuningConfig | None = None,
    *,
    embedding_f_time: float | None = None,
    embedding_b_time: float | None = None,
    embedding_w_time: float | None = None,
    head_f_time: float | None = None,
    head_b_time: float | None = None,
    head_w_time: float | None = None,
) -> OctoPipeTuneResult:
    """Three-phase tuning: partition variance, placement bubble, scheduling overlap."""
    tuning = tuning or TuningConfig()
    schedule = parse_schedule(config.schedule)
    pp = config.parallel
    emb = _embedding_head_times(
        embedding_f_time,
        embedding_b_time,
        embedding_w_time,
        head_f_time,
        head_b_time,
        head_w_time,
    )

    fixed_chunk = config.parallel.chunk_num
    stage_nums = legal_stage_nums(
        config.model.num_layers,
        pp.device_num,
        fixed_chunk,
        include_irregular=tuning.irregular_stage_num,
        max_stage_num=tuning.max_stage_num,
        min_stage_num=tuning.min_stage_num,
    )
    pending_jobs: list[tuple[float, tuple[float, int, float], list[int], list[list[int]], int, bool]] = []

    for stage_num in stage_nums:
        partitions = top_partitions_by_stage_variance(
            layer_f_times,
            layer_b_times,
            layer_w_times,
            config.model.num_layers,
            stage_num,
            tuning.partition_top_k,
            embedding_f_time=emb[0],
            embedding_b_time=emb[1],
            embedding_w_time=emb[2],
            head_f_time=emb[3],
            head_b_time=emb[4],
            head_w_time=emb[5],
        )
        # Per-stage runtime overhead (f+b+w workloads), in ticks, so the proxy
        # ranking does not systematically favor many-stage partitions.
        stage_overhead = 3.0 * config.hardware.workload_overhead_ms * 100.0
        for part_variance, partition in partitions:
            stage_times = stage_times_for_partition(
                layer_f_times,
                layer_b_times,
                layer_w_times,
                partition,
                embedding_f_time=emb[0],
                embedding_b_time=emb[1],
                embedding_w_time=emb[2],
                head_f_time=emb[3],
                head_b_time=emb[4],
                head_w_time=emb[5],
            )
            if stage_overhead > 0:
                stage_times = [t + stage_overhead for t in stage_times]
            placements = generate_octopipe_placement_candidates(
                pp.device_num,
                stage_num,
                beam_width=tuning.beam_width,
                stage_times=stage_times,
            )
            interleaved = placements[0]
            pending_jobs.append(
                (part_variance, placement_proxy_score(interleaved, stage_times), partition, interleaved, stage_num, True)
            )
            for placement in placements[1:]:
                pending_jobs.append(
                    (
                        part_variance,
                        placement_proxy_score(placement, stage_times),
                        partition,
                        placement,
                        stage_num,
                        False,
                    )
                )

    pending_jobs.sort(key=lambda job: (job[0], job[1], 0 if job[5] else 1))
    eval_budget = max(tuning.sim_k, tuning.result_top_k)
    eval_jobs = _select_eval_jobs(pending_jobs, eval_budget)
    candidates: list[TuneCandidateAnalysis] = []
    seen: set[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = set()

    for part_variance, _proxy, partition, placement, stage_num, _is_interleaved in eval_jobs:
        key = (tuple(partition), _placement_key(placement))
        if key in seen:
            continue
        seen.add(key)
        chunk = -(-stage_num // pp.device_num)  # ceil: max stages on any device
        stage_times = stage_times_for_partition(
            layer_f_times,
            layer_b_times,
            layer_w_times,
            partition,
            embedding_f_time=emb[0],
            embedding_b_time=emb[1],
            embedding_w_time=emb[2],
            head_f_time=emb[3],
            head_b_time=emb[4],
            head_w_time=emb[5],
        )
        makespan, _records, overlap_exempt, trials, comp_bubble, score = _evaluate_candidate(
            config,
            partition,
            placement,
            layer_f_times,
            layer_b_times,
            layer_w_times,
            tuning,
            chunk_num=chunk,
            embedding_f_time=emb[0],
            embedding_b_time=emb[1],
            embedding_w_time=emb[2],
            head_f_time=emb[3],
            head_b_time=emb[4],
            head_w_time=emb[5],
        )
        candidates.append(
            TuneCandidateAnalysis(
                rank=0,
                chunk_num=chunk,
                partition_layers=partition,
                placement=placement,
                partition_variance=partition_variance(stage_times),
                makespan=makespan,
                score=score,
                comp_bubble=comp_bubble,
                overlap_exempt_workloads=overlap_exempt,
                bubble_overlap_trials=trials,
            )
        )

    if not candidates:
        raise RuntimeError("OctoPipe tuning produced no candidates")

    candidates.sort(key=lambda c: c.makespan)
    if candidates[0].makespan == float("inf"):
        raise RuntimeError(
            "OctoPipe tuning: every candidate exceeds max_inflight_layers="
            f"{tuning.max_inflight_layers}; raise the cap or reduce microbatches"
        )
    top_k = min(tuning.result_top_k, len(candidates))
    top_results: list[TuneCandidateAnalysis] = []
    for rank, candidate in enumerate(candidates[:top_k], start=1):
        top_results.append(
            TuneCandidateAnalysis(
                rank=rank,
                chunk_num=candidate.chunk_num,
                partition_layers=candidate.partition_layers,
                placement=candidate.placement,
                partition_variance=candidate.partition_variance,
                makespan=candidate.makespan,
                score=candidate.score,
                comp_bubble=candidate.comp_bubble,
                overlap_exempt_workloads=set(candidate.overlap_exempt_workloads),
                bubble_overlap_trials=list(candidate.bubble_overlap_trials),
            )
        )

    best = top_results[0]
    return OctoPipeTuneResult(
        partition_layers=best.partition_layers,
        placement=best.placement,
        makespan=best.makespan,
        overlap_exempt_workloads=set(best.overlap_exempt_workloads),
        bubble_overlap_trials=list(best.bubble_overlap_trials),
        top_results=top_results,
    )


def _placement_key(placement: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in placement)


def _select_eval_jobs(
    pending_jobs: list[tuple[float, tuple[float, int, float], list[int], list[list[int]], int, bool]],
    eval_budget: int,
    *,
    per_chunk_min: int = 2,
) -> list[tuple[float, tuple[float, int, float], list[int], list[list[int]], int, bool]]:
    """Pick eval jobs: top per_chunk_min per chunk, then fill from global ranking."""
    by_chunk: dict[int, list[tuple]] = {}
    for job in pending_jobs:
        by_chunk.setdefault(job[4], []).append(job)

    selected: list[tuple] = []
    selected_keys: set[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = set()

    def add(job: tuple) -> None:
        key = (tuple(job[2]), _placement_key(job[3]))
        if key in selected_keys:
            return
        selected_keys.add(key)
        selected.append(job)

    chunk_slots = min(per_chunk_min, max(1, eval_budget // max(len(by_chunk), 1)))
    for chunk in sorted(by_chunk):
        chunk_jobs = sorted(by_chunk[chunk], key=lambda j: (j[0], j[1], 0 if j[5] else 1))
        for job in chunk_jobs[:chunk_slots]:
            add(job)

    for job in pending_jobs:
        if len(selected) >= eval_budget:
            break
        add(job)

    return selected[:eval_budget]


def generate_octopipe_candidates(
    config: SimConfig,
    layer_f: list[float],
    layer_b: list[float],
    layer_w: list[float] | None,
    tuning: TuningConfig,
    *,
    embedding_f_time: float = 0.0,
    embedding_b_time: float = 0.0,
    embedding_w_time: float = 0.0,
    head_f_time: float = 0.0,
    head_b_time: float = 0.0,
    head_w_time: float = 0.0,
) -> list[tuple[list[int], list[list[int]], float]]:
    """Backward-compatible fast candidate listing using partition variance only."""
    del embedding_f_time, embedding_b_time, embedding_w_time, head_f_time, head_b_time, head_w_time
    pp = config.parallel
    schedule = parse_schedule(config.schedule)
    fixed_chunk = resolve_chunk_num(config, schedule) if pp.chunk_num is not None else None
    raw: list[tuple[list[int], list[list[int]], float]] = []
    for chunk in legal_chunk_values(config.model.num_layers, pp.device_num, fixed_chunk):
        stage_num = pp.device_num * chunk
        for variance, partition in top_partitions_by_stage_variance(
            layer_f, layer_b, layer_w, config.model.num_layers, stage_num, tuning.partition_top_k
        ):
            for placement in generate_octopipe_placement_candidates(
                pp.device_num, stage_num, beam_width=tuning.beam_width
            ):
                raw.append((partition, placement, variance))
    raw.sort(key=lambda x: x[2])
    return raw
