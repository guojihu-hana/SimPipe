from simpipe.core.types import Schedule, WorkloadType, parse_schedule
from simpipe.models.registry import get_preset, get_layer_times, get_profile_times
from simpipe.core.executor import build_simulation
from simpipe.config.hardware import HardwareConfig
from simpipe.config.tuning import TuningConfig
from simpipe.pipeline.schedule_config import apply_schedule_config, resolve_partition_layers


def _preset_cfg(name: str):
    from dataclasses import replace

    cfg = get_preset(name)
    return replace(cfg, parallel=replace(cfg.parallel, pp_size=4, micro_batch_num=8))


def _build_with_preset(cfg, model_name: str, **kwargs):
    profile = get_profile_times(model_name).slice_layers(cfg.model.num_layers)
    return build_simulation(
        cfg,
        profile.layer_f,
        profile.layer_b,
        profile.layer_w,
        embedding_f_time=profile.embedding_f,
        embedding_b_time=profile.embedding_b,
        embedding_w_time=profile.embedding_w,
        head_f_time=profile.head_f,
        head_b_time=profile.head_b,
        head_w_time=profile.head_w,
        **kwargs,
    )


def test_zbh_executes_all_workloads_including_w():
    cfg = _preset_cfg("nemotronh-4B")
    ex = _build_with_preset(cfg, "nemotronh-4B", schedule=parse_schedule("zbh"))
    assert ex.config.parallel.bwd_split is True
    result = ex.run()
    types = [r["wtype"] for r in result.records]
    static_len = sum(len(row) for row in ex.plan.static_schedule)
    assert len(result.records) == static_len
    assert static_len == 96
    assert types.count("W") == 32
    assert types.count("F") == 32


def test_profiled_layer_times_fall_back_to_backward_for_w():
    _f, b, w = get_layer_times("nemotronh-4B")

    assert w == b
    assert all(t > 0 for t in w)


def test_test_model_preset_has_48_layers():
    cfg = _preset_cfg("test_model")
    f, b, w = get_layer_times("test_model")

    assert cfg.model.name == "test_model"
    assert cfg.model.num_layers == 48
    assert len(f) == len(b) == len(w) == 48


def test_octopipe_auto_tune_searches_partition_when_chunk_num_is_auto():
    from dataclasses import replace

    cfg = _preset_cfg("test_model")
    cfg = replace(
        cfg,
        schedule="octopipe",
        parallel=replace(cfg.parallel, bwd_split=True, chunk_num=None),
        tuning=TuningConfig(auto_tune=True, sim_k=32, beam_width=32),
    )
    ex = _build_with_preset(cfg, "test_model", schedule=parse_schedule("octopipe"))

    assert ex.plan.stage_num > cfg.parallel.pp_size
    assert sum(ex.plan.partition.layer_counts(ex.graph)) == cfg.model.num_layers


def test_octopipe_auto_tune_can_add_bubble_overlap_exemptions():
    from dataclasses import replace

    cfg = _preset_cfg("test_model")
    cfg = replace(
        cfg,
        schedule="octopipe",
        hardware=HardwareConfig(comm_alpha_us=0),
        parallel=replace(cfg.parallel, bwd_split=True, chunk_num=None),
        tuning=TuningConfig(
            auto_tune=True,
            sim_k=32,
            beam_width=32,
            bubble_overlap_tune=True,
            bubble_overlap_max_iter=8,
        ),
    )
    ex = _build_with_preset(cfg, "test_model", schedule=parse_schedule("octopipe"))

    assert ex.bubble_overlap_trials


def test_interleaved_default_chunk_is_max():
    cfg = _preset_cfg("nemotronh-4B")
    cfg = apply_schedule_config(cfg, Schedule.INTERLEAVED)
    layers = resolve_partition_layers(cfg, Schedule.INTERLEAVED, None)
    assert cfg.parallel.chunk_num == 13
    assert len(layers) == 52
    assert sum(layers) == 52
    assert layers == [1] * 52


def test_interleaved_respects_manual_chunk_num():
    from dataclasses import replace

    cfg = _preset_cfg("nemotronh-4B")
    cfg = replace(cfg, parallel=replace(cfg.parallel, chunk_num=2))
    cfg = apply_schedule_config(cfg, Schedule.INTERLEAVED)
    layers = resolve_partition_layers(cfg, Schedule.INTERLEAVED, None)
    assert cfg.parallel.chunk_num == 2
    assert len(layers) == 8
    assert sum(layers) == 52
    assert layers == [7, 7, 7, 7, 6, 6, 6, 6]


def test_octopipe_respects_chunk_num():
    from dataclasses import replace

    cfg = _preset_cfg("nemotronh-4B")
    cfg = replace(cfg, model=replace(cfg.model, num_layers=48), schedule="octopipe")
    cfg = replace(cfg, parallel=replace(cfg.parallel, chunk_num=2), tuning=TuningConfig(auto_tune=False))
    cfg = apply_schedule_config(cfg, Schedule.OctoPipe)
    assert cfg.parallel.chunk_num == 2
    layers = resolve_partition_layers(cfg, Schedule.OctoPipe, None)
    assert len(layers) == 8
    assert sum(layers) == 48
    ex = _build_with_preset(cfg, "nemotronh-4B", schedule=parse_schedule("octopipe"))
    assert ex.plan.stage_num == 8
    assert ex.plan.placement.device_stages == [[0, 4], [1, 5], [2, 6], [3, 7]]
    ex.run()


def test_octopipe_initializes_dynamic_ready_queues():
    from dataclasses import replace

    cfg = _preset_cfg("nemotronh-4B")
    cfg = replace(cfg, model=replace(cfg.model, num_layers=8), schedule="octopipe")
    cfg = replace(cfg, parallel=replace(cfg.parallel, chunk_num=2), tuning=TuningConfig(auto_tune=False))
    ex = _build_with_preset(cfg, "nemotronh-4B", schedule=parse_schedule("octopipe"))

    assert ex.plan.static_schedule is None
    assert bool(ex.pipelines[0].devices[0].executable_workloads)


def test_octopipe_completes_all_dynamic_workloads():
    from dataclasses import replace

    cfg = _preset_cfg("nemotronh-4B")
    cfg = replace(cfg, model=replace(cfg.model, num_layers=8), schedule="octopipe")
    cfg = replace(cfg, parallel=replace(cfg.parallel, chunk_num=2), tuning=TuningConfig(auto_tune=False))
    ex = _build_with_preset(cfg, "nemotronh-4B", schedule=parse_schedule("octopipe"))

    result = ex.run(time_limit=100_000)
    assert len(result.records) == 8 * cfg.parallel.micro_batch_num * 2


def test_1f1b_merges_w_into_b_when_bwd_split_false():
    from dataclasses import replace

    cfg = _preset_cfg("test_model")
    cfg = replace(cfg, schedule="1f1b", parallel=replace(cfg.parallel, bwd_split=False))
    ex = _build_with_preset(cfg, "test_model", schedule=parse_schedule("1f1b"))

    assert ex.config.parallel.bwd_split is False
    for timing in ex.plan.stage_timings:
        assert timing.w_time == 0.0
        assert timing.b_time > 0.0

    stage0 = ex.pipelines[0].devices[0].stages[0]
    mid = next(iter(stage0.workloads))
    assert WorkloadType.W not in stage0.workloads[mid]

    merged = ex.run()
    split_cfg = replace(cfg, parallel=replace(cfg.parallel, bwd_split=True))
    split_ex = _build_with_preset(split_cfg, "test_model", schedule=parse_schedule("1f1b"))
    split = split_ex.run()

    assert merged.makespan == split.makespan
    assert {r["wtype"] for r in merged.records} == {"B", "F"}


def test_bapar_uses_dp_partition_and_1f1b_scheduling():
    from dataclasses import replace

    from simpipe.core.types import parse_schedule
    from simpipe.pipeline.placement import Placement
    from simpipe.pipeline.scheduling.base import ScheduleContext
    from simpipe.pipeline.scheduling.one_f_one_b import OneFOneBStrategy

    cfg = _preset_cfg("test_model")
    cfg = replace(cfg, schedule="bapar")
    ex = _build_with_preset(cfg, "test_model", schedule=parse_schedule("bapar"))

    equal = [12, 12, 12, 12]
    actual = ex.plan.partition.layer_counts(ex.graph)
    assert sum(actual) == cfg.model.num_layers
    assert actual != equal
    assert ex.plan.placement.device_stages == Placement.sequential(cfg.parallel.pp_size).device_stages

    ref = OneFOneBStrategy().generate(
        ScheduleContext(
            device_num=cfg.parallel.pp_size,
            stage_num=cfg.parallel.pp_size,
            micro_batch_num=cfg.parallel.micro_batch_num,
            mid_offset=0,
            bwd_split=cfg.parallel.bwd_split,
            chunk_num=1,
            placement=ex.plan.placement,
        )
    )
    assert len(ex.plan.static_schedule[0]) == len(ref[0])
    assert ex.run().makespan > 0


def test_1f1b_ignores_yaml_chunk_num():
    from dataclasses import replace

    cfg = _preset_cfg("nemotronh-4B")
    cfg = replace(cfg, parallel=replace(cfg.parallel, chunk_num=2), schedule="1f1b")
    cfg = apply_schedule_config(cfg, Schedule.S1F1B)
    assert cfg.parallel.chunk_num == 1
    ex = _build_with_preset(cfg, "nemotronh-4B", schedule=parse_schedule("1f1b"))
    assert ex.plan.stage_num == 4
    assert ex.plan.placement.device_stages == [[0], [1], [2], [3]]


def test_interleaved_manual_chunk_runs():
    from dataclasses import replace

    cfg = _preset_cfg("nemotronh-4B")
    cfg = replace(cfg, parallel=replace(cfg.parallel, chunk_num=2))
    ex = _build_with_preset(cfg, "nemotronh-4B", schedule=parse_schedule("interleaved"))
    assert ex.plan.stage_num == 8
    assert ex.config.parallel.chunk_num == 2
    result = ex.run()
    static_len = sum(len(row) for row in ex.plan.static_schedule)
    assert len(result.records) == static_len


def test_interleaved_uses_layer_per_stage_partition():
    test_interleaved_default_chunk_is_max()

def test_interleaved_schedule_length_matches_ps():
    from simpipe.pipeline.placement import Placement
    from simpipe.pipeline.scheduling.base import ScheduleContext
    from simpipe.pipeline.scheduling.interleaved import InterleavedStrategy

    placement = Placement([[i + 4 * j for j in range(13)] for i in range(4)])
    ctx = ScheduleContext(
        device_num=4,
        stage_num=52,
        micro_batch_num=8,
        mid_offset=0,
        bwd_split=False,
        chunk_num=13,
        placement=placement,
    )
    sched = InterleavedStrategy().generate(ctx)
    assert len(sched[0]) == 208


def test_interleaved_runs_all_static_ops():
    cfg = _preset_cfg("nemotronh-4B")
    ex = _build_with_preset(cfg, "nemotronh-4B", schedule=parse_schedule("interleaved"))
    assert ex.plan.stage_num == 52
    static_len = sum(len(row) for row in ex.plan.static_schedule)
    result = ex.run()
    assert len(result.records) == static_len
    assert static_len == 832
