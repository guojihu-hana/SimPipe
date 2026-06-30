import yaml

from simpipe.config.pipeline_config import serialize_scheduling_records, write_pipeline_config


def test_serialize_scheduling_records():
    records = [
        {"did": 1, "mid": 0, "sid": 0, "wtype": "F", "start": 10, "end": 20, "duration": 10},
        {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 10, "duration": 10},
    ]
    serialized = serialize_scheduling_records(records)
    assert serialized == [
        "(f, 0, 0, 0, 0, 10)",
        "(f, 0, 0, 1, 10, 20)",
    ]


def test_write_pipeline_config_contains_partition_placement_scheduling(tmp_path):
    out = tmp_path / "pipeline_config.yaml"
    records = [
        {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 100, "duration": 100},
        {"did": 0, "mid": 0, "sid": 0, "wtype": "B", "start": 100, "end": 200, "duration": 100},
    ]
    write_pipeline_config(
        out,
        partition=[6, 6, 6, 6],
        placement=[[0, 3], [1, 2]],
        schedule="octopipe",
        makespan=1234.0,
        chunk_num=2,
        scheduling_records=records,
    )

    text = out.read_text()
    assert "scheduling:  # (workload_type, mid, sid, did, start_time, end_time)" in text
    assert "- (f, 0, 0, 0, 0, 100)" in text
    assert "- (b, 0, 0, 0, 100, 200)" in text

    data = yaml.safe_load(text)
    assert data["schedule"] == "octopipe"
    assert data["partition"] == [6, 6, 6, 6]
    assert data["placement"] == [[0, 3], [1, 2]]
    assert data["chunk_num"] == 2
    assert data["makespan"] == 1234.0
    assert data["scheduling"] == [
        "(f, 0, 0, 0, 0, 100)",
        "(b, 0, 0, 0, 100, 200)",
    ]


def test_build_pipeline_config_from_executor(tmp_path):
    from dataclasses import replace

    from simpipe.config.hardware import HardwareConfig
    from simpipe.config.tuning import TuningConfig
    from simpipe.core.executor import build_simulation
    from simpipe.models.registry import get_preset, get_profile_times

    cfg = get_preset("test_model")
    cfg = replace(
        cfg,
        schedule="octopipe",
        hardware=HardwareConfig(comm_alpha_us=0),
        parallel=replace(cfg.parallel, pp_size=4, bwd_split=True, chunk_num=2, micro_batch_num=2),
        tuning=TuningConfig(auto_tune=False),
        partition_layers=[6, 6, 6, 6, 6, 6, 6, 6],
        placement=[[0, 4], [1, 5], [2, 6], [3, 7]],
    )
    profile = get_profile_times("test_model").slice_layers(cfg.model.num_layers)
    executor = build_simulation(
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
        partition_layers=cfg.partition_layers,
        placement=cfg.placement,
    )
    result = executor.run()
    out = tmp_path / "pipeline_config.yaml"
    write_pipeline_config(
        out,
        executor=executor,
        makespan=result.makespan,
        scheduling_records=result.records,
    )

    data = yaml.safe_load(out.read_text())
    assert data["partition"] == [6, 6, 6, 6, 6, 6, 6, 6]
    assert data["placement"] == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert data["makespan"] == result.makespan
    assert len(data["scheduling"]) == len(result.records)
    for entry in data["scheduling"]:
        assert entry.startswith("(") and entry.endswith(")")


def test_build_pipeline_config_from_executor_includes_stage_layers(tmp_path):
    from dataclasses import replace

    from simpipe.config.hardware import HardwareConfig
    from simpipe.config.tuning import TuningConfig
    from simpipe.core.executor import build_simulation
    from simpipe.models.registry import get_preset, get_profile_times

    cfg = get_preset("nemotronh-4B")
    cfg = replace(
        cfg,
        schedule="octopipe",
        hardware=HardwareConfig(comm_alpha_us=0),
        parallel=replace(cfg.parallel, pp_size=4, bwd_split=True, chunk_num=2, micro_batch_num=2),
        tuning=TuningConfig(auto_tune=False),
        partition_layers=[13, 13, 13, 13, 13, 13, 13, 13],
        placement=[[0, 4], [1, 5], [2, 6], [3, 7]],
    )
    profile = get_profile_times("nemotronh-4B").slice_layers(cfg.model.num_layers)
    executor = build_simulation(
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
        partition_layers=cfg.partition_layers,
        placement=cfg.placement,
    )
    result = executor.run()
    out = tmp_path / "pipeline_config.yaml"
    write_pipeline_config(
        out,
        executor=executor,
        makespan=result.makespan,
        scheduling_records=result.records,
    )

    text = out.read_text()
    assert "stage_layers:" in text
    assert '"0: E' in text
    data = yaml.safe_load(text)
    assert data["stage_layers"][0].startswith("0: ")
    assert data["stage_layers"][0][3] == "E"
    assert data["stage_layers"][-1].endswith("L")
