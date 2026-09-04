from __future__ import annotations

from dataclasses import replace

from simpipe.config.hardware import HardwareConfig
from simpipe.config.parallel import ParallelConfig
from simpipe.core.types import Schedule
from simpipe.graph.model_graph import ModelGraph
from simpipe.graph.operator import OpType
from simpipe.models.pattern import ATTN, MOE, TRANSFORMER
from simpipe.pipeline.partition import OperatorPartition
from simpipe.pipeline.placement import Placement
from simpipe.pipeline.scheduling.base import ScheduleContext, get_strategy
from simpipe.pipeline.types import StageTiming, WorkloadPlan


def stage_timing(
    plan: WorkloadPlan,
    sid: int,
    mid: int | None = None,
    overhead: float = 0.0,
) -> StageTiming:
    """Workload durations of one (stage, microbatch) pair.

    The plan scales the profiled stage time per microbatch (uniform when mid
    is None or no batch spec is set); the constant per-launch overhead (same
    tick unit, not token-dependent) is added on top.  W keeps zero when the
    stage has no separate W pass.
    """
    timing = plan.timing_for_stage(sid, mid)
    if overhead <= 0:
        return timing
    return replace(
        timing,
        f_time=timing.f_time + overhead,
        b_time=timing.b_time + overhead,
        w_time=timing.w_time + overhead if timing.w_time > 0 else timing.w_time,
    )

# Layer kinds whose profiled time contains an attention-score component that
# scales quadratically with sequence length (mamba / dense-MLP layers do not).
_QUAD_LAYER_SYMBOLS = frozenset({ATTN, TRANSFORMER, MOE})


def _quad_operator_ids(
    graph: ModelGraph,
    operator_ids: list[str],
    layer_symbols: list[str] | None,
) -> list[str]:
    """Operators in a stage whose time scales with seq_len^2.

    The graph models every stack layer as a transformer block, so profiled
    time distributed onto the ATTENTION op of a mamba/MLP layer is actually
    linear-cost work; layer_symbols (from the model pattern) masks those
    layers out.  Without a pattern every layer is treated as attention.
    """
    quad_ids: list[str] = []
    for oid in operator_ids:
        op = graph.op_by_id(oid)
        if op.op_type != OpType.ATTENTION:
            continue
        if layer_symbols is not None:
            layer_idx = op.layer_idx
            if (
                layer_idx is None
                or not 0 <= layer_idx < len(layer_symbols)
                or layer_symbols[layer_idx] not in _QUAD_LAYER_SYMBOLS
            ):
                continue
        quad_ids.append(oid)
    return quad_ids


def _apply_profile_times(
    graph: ModelGraph,
    layer_f_times: list[float] | None,
    layer_b_times: list[float] | None,
    layer_w_times: list[float] | None,
    *,
    embedding_f_time: float | None = None,
    embedding_b_time: float | None = None,
    embedding_w_time: float | None = None,
    head_f_time: float | None = None,
    head_b_time: float | None = None,
    head_w_time: float | None = None,
) -> None:
    if not layer_f_times:
        return
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


def build_workload_plan(
    graph: ModelGraph,
    partition: OperatorPartition,
    placement: Placement,
    parallel: ParallelConfig,
    hardware: HardwareConfig,
    schedule: Schedule,
    layer_f_times: list[float] | None = None,
    layer_b_times: list[float] | None = None,
    layer_w_times: list[float] | None = None,
    embedding_f_time: float | None = None,
    embedding_b_time: float | None = None,
    embedding_w_time: float | None = None,
    head_f_time: float | None = None,
    head_b_time: float | None = None,
    head_w_time: float | None = None,
    mid_offset: int = 0,
    layer_symbols: list[str] | None = None,
) -> WorkloadPlan:
    partition.validate(graph)
    stage_num = partition.num_stages
    device_num = parallel.device_num
    placement.validate(device_num, stage_num)

    _apply_profile_times(
        graph,
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

    peak = hardware.gpu_peak_tflops
    bw = hardware.intra_node_bw_gbps
    stage_timings: list[StageTiming] = []
    for stage in partition.stages:
        f_t = graph.stage_forward_time(stage.operator_ids, peak_tflops=peak, mem_bw_gbps=bw)
        b_t = graph.stage_backward_time(stage.operator_ids, peak_tflops=peak, mem_bw_gbps=bw)
        w_t = graph.stage_weight_time(stage.operator_ids, peak_tflops=peak, mem_bw_gbps=bw)
        quad_ids = _quad_operator_ids(graph, stage.operator_ids, layer_symbols)
        f_quad = graph.stage_forward_time(quad_ids, peak_tflops=peak, mem_bw_gbps=bw)
        b_quad = graph.stage_backward_time(quad_ids, peak_tflops=peak, mem_bw_gbps=bw)
        if not parallel.bwd_split:
            b_t += w_t
            w_t = 0.0
        stage_timings.append(
            StageTiming(
                stage.stage_id,
                list(stage.operator_ids),
                f_t,
                b_t,
                w_t,
                f_quad=f_quad,
                b_quad=b_quad,
            )
        )

    static = None
    if schedule in (
        Schedule.S1F1B,
        Schedule.BAPAR,
        Schedule.INTERLEAVED,
        Schedule.ZBH,
        Schedule.AFAB,
    ):
        ctx = ScheduleContext(
            device_num=device_num,
            stage_num=stage_num,
            micro_batch_num=parallel.micro_batch_num,
            mid_offset=mid_offset,
            bwd_split=parallel.bwd_split,
            chunk_num=parallel.chunk_num,
            placement=placement,
            max_act=1,
        )
        static = get_strategy(schedule).generate(ctx)

    return WorkloadPlan(
        partition=partition,
        placement=placement,
        schedule=schedule,
        stage_timings=stage_timings,
        static_schedule=static,
        layer_f_times=layer_f_times or [],
        layer_b_times=layer_b_times or [],
        layer_w_times=layer_w_times or [],
    )
