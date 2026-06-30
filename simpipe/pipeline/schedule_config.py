from __future__ import annotations

from dataclasses import replace

from simpipe.config.sim_config import SimConfig
from simpipe.core.types import Schedule, parse_schedule
from simpipe.pipeline.placement import Placement


def resolve_schedule(config: SimConfig, schedule: Schedule | None = None) -> Schedule:
    if schedule is not None:
        return schedule
    return parse_schedule(config.schedule)


def max_interleaved_chunk_num(num_layers: int, pp_size: int) -> int:
    """Max virtual stages per device: one layer per virtual stage."""
    if pp_size <= 0:
        raise ValueError("pp_size must be positive")
    if num_layers < pp_size:
        raise ValueError("num_layers must be >= pp_size for interleaved PP")
    return num_layers // pp_size


def _multi_chunk_schedules() -> tuple[Schedule, ...]:
    return (Schedule.INTERLEAVED, Schedule.OctoPipe)


def resolve_chunk_num(config: SimConfig, schedule: Schedule) -> int:
    parallel = config.parallel
    if schedule not in _multi_chunk_schedules():
        return 1
    max_chunk = max_interleaved_chunk_num(config.model.num_layers, parallel.pp_size)
    if parallel.chunk_num is None:
        return max_chunk
    if not 1 <= parallel.chunk_num <= max_chunk:
        raise ValueError(
            f"{schedule.name} chunk_num must be in [1, {max_chunk}], got {parallel.chunk_num}"
        )
    return parallel.chunk_num


def resolve_placement(
    schedule: Schedule,
    device_num: int,
    chunk_num: int,
    placement: list[list[int]] | None = None,
) -> Placement:
    if placement is not None:
        return Placement(placement)
    if schedule in _multi_chunk_schedules():
        return Placement.interleaved(device_num, chunk_num)
    return Placement.sequential(device_num)


def apply_schedule_config(config: SimConfig, schedule: Schedule) -> SimConfig:
    """Align parallel settings with PipelineSimulator schedule policy."""
    parallel = config.parallel
    if schedule == Schedule.ZBH:
        parallel = replace(parallel, bwd_split=True)
        config = replace(config, parallel=parallel)
    chunk_num = resolve_chunk_num(config, schedule)
    if config.parallel.chunk_num != chunk_num:
        config = replace(config, parallel=replace(config.parallel, chunk_num=chunk_num))
    return config


def balanced_layer_partition(num_layers: int, stage_num: int) -> list[int]:
    if stage_num <= 0:
        raise ValueError("stage_num must be positive")
    base, rem = divmod(num_layers, stage_num)
    return [base + (1 if i < rem else 0) for i in range(stage_num)]


def resolve_partition_layers(
    config: SimConfig,
    schedule: Schedule,
    partition_layers: list[int] | None,
    *,
    layer_f_times: list[float] | None = None,
    layer_b_times: list[float] | None = None,
    layer_w_times: list[float] | None = None,
    embedding_f_time: float | None = None,
    embedding_b_time: float | None = None,
    embedding_w_time: float | None = None,
    head_f_time: float | None = None,
    head_b_time: float | None = None,
    head_w_time: float | None = None,
) -> list[int]:
    if partition_layers is not None:
        return partition_layers
    pp = config.parallel
    if schedule == Schedule.BAPAR and layer_f_times:
        from simpipe.tuning.partition_search import best_partition_by_stage_variance

        return best_partition_by_stage_variance(
            layer_f_times,
            layer_b_times or layer_f_times,
            layer_w_times,
            config.model.num_layers,
            pp.device_num,
            embedding_f_time=embedding_f_time or 0.0,
            embedding_b_time=embedding_b_time or 0.0,
            embedding_w_time=embedding_w_time or 0.0,
            head_f_time=head_f_time or 0.0,
            head_b_time=head_b_time or 0.0,
            head_w_time=head_w_time or 0.0,
        )
    if schedule in _multi_chunk_schedules():
        stage_num = pp.device_num * pp.chunk_num
        return balanced_layer_partition(config.model.num_layers, stage_num)
    per_stage = config.model.num_layers // pp.device_num
    return [per_stage] * pp.device_num


def split_layer_times_for_zbh(
    schedule: Schedule,
    layer_f_times: list[float] | None,
    layer_b_times: list[float] | None,
    layer_w_times: list[float] | None,
) -> tuple[list[float] | None, list[float] | None, list[float] | None]:
    """Match PipelineSimulator: halve B and set W equal to the new B."""
    if schedule != Schedule.ZBH or not layer_b_times:
        return layer_f_times, layer_b_times, layer_w_times
    b_times = [t * 0.5 for t in layer_b_times]
    w_times = list(b_times)
    return layer_f_times, b_times, w_times
