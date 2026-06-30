from simpipe.metrics.comp_bubble import (
    analyze_pipeline_comp_bubble,
    total_inter_workload_bubble,
)


def test_analyze_pipeline_comp_bubble_reports_per_device_ratios():
    records = [
        {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 3, "duration": 3},
        {"did": 0, "mid": 1, "sid": 0, "wtype": "B", "start": 5, "end": 7, "duration": 2},
        {"did": 1, "mid": 0, "sid": 0, "wtype": "F", "start": 1, "end": 4, "duration": 3},
        {"did": 1, "mid": 1, "sid": 0, "wtype": "W", "start": 10, "end": 12, "duration": 2},
    ]

    stats = analyze_pipeline_comp_bubble(records)

    assert stats.first_backward_start == 5.0
    assert stats.last_forward_end == 4.0
    assert stats.makespan == 12.0
    assert stats.per_device[0].to_dict() == {
        "did": 0,
        "comp": 5.0,
        "bubble": 7.0,
        "warmup_bubble": 2.0,
        "cooldown_bubble": 5.0,
        "residual_bubble": 0.0,
        "total": 12.0,
        "comp_ratio": 5.0 / 12.0,
        "bubble_ratio": 7.0 / 12.0,
        "warmup_bubble_ratio": 2.0 / 12.0,
        "cooldown_bubble_ratio": 5.0 / 12.0,
        "residual_bubble_ratio": 0.0,
        "total_ratio": 1.0,
    }
    assert stats.per_device[1].comp == 5.0
    assert stats.per_device[1].bubble == 7.0
    assert stats.per_device[1].warmup_bubble == 2.0
    assert stats.per_device[1].cooldown_bubble == 5.0
    assert stats.per_device[1].residual_bubble == 0.0


def test_analyze_pipeline_comp_bubble_splits_residual_between_warmup_and_cooldown():
    records = [
        {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 3, "duration": 3},
        {"did": 0, "mid": 1, "sid": 0, "wtype": "B", "start": 10, "end": 12, "duration": 2},
        {"did": 0, "mid": 2, "sid": 0, "wtype": "F", "start": 15, "end": 18, "duration": 3},
        {"did": 1, "mid": 0, "sid": 0, "wtype": "F", "start": 5, "end": 8, "duration": 3},
        {"did": 1, "mid": 1, "sid": 0, "wtype": "B", "start": 20, "end": 22, "duration": 2},
    ]

    stats = analyze_pipeline_comp_bubble(records)

    assert stats.first_backward_start == 10.0
    assert stats.last_forward_end == 18.0
    assert stats.per_device[0].warmup_bubble == 7.0
    assert stats.per_device[0].residual_bubble == 3.0
    assert stats.per_device[0].cooldown_bubble == 4.0
    assert stats.per_device[1].warmup_bubble == 7.0
    assert stats.per_device[1].residual_bubble == 8.0
    assert stats.per_device[1].cooldown_bubble == 2.0


def test_analyze_pipeline_comp_bubble_includes_empty_devices():
    records = [
        {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 3, "duration": 3},
    ]

    stats = analyze_pipeline_comp_bubble(records, device_num=2)

    assert len(stats.per_device) == 2
    assert stats.per_device[1].comp == 0.0
    assert stats.per_device[1].bubble == 3.0
    assert stats.avg_bubble_ratio(device_num=2) == (0.0 + 3.0) / (2 * 3)


def test_total_inter_workload_bubble_ignores_leading_and_trailing_idle():
    records = [
        {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 3},
        {"did": 0, "mid": 1, "sid": 0, "wtype": "B", "start": 10, "end": 12},
        {"did": 1, "mid": 0, "sid": 0, "wtype": "F", "start": 1, "end": 4},
    ]

    assert total_inter_workload_bubble(records) == 7.0
