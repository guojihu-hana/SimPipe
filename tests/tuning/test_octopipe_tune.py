from dataclasses import replace

from simpipe.config.tuning import TuningConfig
from simpipe.metrics.comp_bubble import (
    device_bubble_variance,
    device_comp_variance,
    tuning_score,
)
from simpipe.models.registry import get_profile_times


def test_tuning_config_parses_partition_and_result_top_k():
    cfg = TuningConfig.from_dict(
        {
            "partition_top_k": 3,
            "result_top_k": 7,
        }
    )
    assert cfg.partition_top_k == 3
    assert cfg.result_top_k == 7


def test_tuning_score_uses_makespan_only():
    from simpipe.metrics.comp_bubble import DeviceCompBubbleStats, PipelineCompBubbleStats

    low = PipelineCompBubbleStats(
        None,
        None,
        100.0,
        (
            DeviceCompBubbleStats(0, 50, 50, 10, 20, 20),
            DeviceCompBubbleStats(1, 50, 50, 10, 20, 20),
        ),
    )
    high = PipelineCompBubbleStats(
        None,
        None,
        200.0,
        (
            DeviceCompBubbleStats(0, 50, 50, 30, 40, 20),
            DeviceCompBubbleStats(1, 50, 50, 30, 40, 20),
        ),
    )
    assert tuning_score(low, 100.0) == 100.0
    assert tuning_score(high, 200.0) == 200.0
    assert tuning_score(low, 100.0) < tuning_score(high, 200.0)


def test_device_variance_helpers():
    from simpipe.metrics.comp_bubble import DeviceCompBubbleStats, PipelineCompBubbleStats

    balanced = PipelineCompBubbleStats(
        None,
        None,
        100.0,
        (
            DeviceCompBubbleStats(0, 50, 50, 10, 20, 20),
            DeviceCompBubbleStats(1, 50, 50, 10, 20, 20),
        ),
    )
    imbalanced = PipelineCompBubbleStats(
        None,
        None,
        100.0,
        (
            DeviceCompBubbleStats(0, 80, 20, 10, 5, 5),
            DeviceCompBubbleStats(1, 20, 80, 10, 5, 5),
        ),
    )
    assert device_comp_variance(balanced) < device_comp_variance(imbalanced)
    assert device_bubble_variance(balanced) < device_bubble_variance(imbalanced)


def test_select_eval_jobs_includes_each_chunk():
    from simpipe.tuning.octopipe_tune import _select_eval_jobs

    jobs = [
        (10.0, (1.0, 0, 1.0), [1, 1], [[0], [1]], 3, True),
        (20.0, (2.0, 0, 2.0), [1, 1, 1, 1], [[0, 2], [1, 3]], 6, True),
        (5.0, (0.5, 0, 0.5), [2, 2], [[0], [1]], 3, False),
    ]
    selected = _select_eval_jobs(jobs, eval_budget=2)
    chunks = {job[4] for job in selected}
    assert 3 in chunks
    assert 6 in chunks


def test_octopipe_tune_returns_top_k_analyses():
    from simpipe.core.executor import build_simulation
    from simpipe.core.types import parse_schedule
    from simpipe.models.registry import get_preset
    from simpipe.config.hardware import HardwareConfig

    cfg = get_preset("test_model")
    cfg = replace(
        cfg,
        schedule="octopipe",
        hardware=HardwareConfig(comm_alpha_us=0),
        parallel=replace(cfg.parallel, bwd_split=True, chunk_num=2, micro_batch_num=2),
        tuning=TuningConfig(
            auto_tune=True,
            sim_k=2,
            beam_width=2,
            partition_top_k=2,
            result_top_k=2,
            bubble_overlap_tune=False,
        ),
    )
    profile = get_profile_times("test_model").slice_layers(cfg.model.num_layers)
    ex = build_simulation(
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
        schedule=parse_schedule("octopipe"),
    )

    assert len(ex.tune_top_results) <= 2
    assert ex.tune_top_results[0].rank == 1
    assert ex.tune_top_results[0].comp_bubble["per_device"]
    assert sum(ex.tune_top_results[0].partition_layers) == cfg.model.num_layers
