from simpipe.core.types import WorkloadType
from simpipe.tuning.bubble_overlap import (
    bubble_by_group,
    choose_next_bubble_group,
    total_inter_workload_bubble,
    tune_overlap_exemptions,
)


def test_bubble_stats_group_by_mid_and_type_by_default():
    records = [
        {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 3},
        {"did": 0, "mid": 7, "sid": 1, "wtype": "B", "start": 10, "end": 12},
        {"did": 0, "mid": 7, "sid": 2, "wtype": "B", "start": 15, "end": 18},
        {"did": 1, "mid": 1, "sid": 0, "wtype": "F", "start": 1, "end": 4},
        {"did": 1, "mid": 7, "sid": 3, "wtype": "B", "start": 9, "end": 13},
    ]

    stats = bubble_by_group(records)

    assert stats[(7, WorkloadType.B)] == 15
    assert (1, WorkloadType.F) not in stats
    assert total_inter_workload_bubble(records) == 15


def test_bubble_stats_supports_grouping_modes():
    records = [
        {"did": 0, "mid": 7, "sid": 1, "wtype": "B", "start": 0, "end": 3},
        {"did": 0, "mid": 7, "sid": 2, "wtype": "B", "start": 10, "end": 12},
        {"did": 1, "mid": 7, "sid": 2, "wtype": "B", "start": 0, "end": 2},
        {"did": 1, "mid": 7, "sid": 2, "wtype": "F", "start": 4, "end": 5},
    ]

    assert bubble_by_group(records, "mid") == {(7,): 9}
    assert bubble_by_group(records, "mid_type") == {
        (7, WorkloadType.B): 7,
        (7, WorkloadType.F): 2,
    }
    assert bubble_by_group(records, "mid_sid_type") == {
        (7, 2, WorkloadType.B): 7,
        (7, 2, WorkloadType.F): 2,
    }


def test_choose_next_bubble_group_skips_existing_and_rejected():
    records = [
        {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 3},
        {"did": 0, "mid": 7, "sid": 1, "wtype": "B", "start": 10, "end": 12},
        {"did": 1, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 2},
        {"did": 1, "mid": 3, "sid": 1, "wtype": "W", "start": 6, "end": 7},
    ]

    assert choose_next_bubble_group(records, set(), set()) == (7, WorkloadType.B)
    assert choose_next_bubble_group(records, {(7, WorkloadType.B)}, set()) == (3, WorkloadType.W)
    assert choose_next_bubble_group(records, set(), {(7, WorkloadType.B)}) == (3, WorkloadType.W)


def test_tune_overlap_exemptions_reports_accepted_iteration():
    initial = [
        {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 1},
        {"did": 0, "mid": 7, "sid": 0, "wtype": "B", "start": 10, "end": 11},
        {"did": 1, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 1},
        {"did": 1, "mid": 3, "sid": 0, "wtype": "W", "start": 6, "end": 7},
    ]

    def run_with_exemptions(exemptions):
        if exemptions == {(3, WorkloadType.W)}:
            return [
                {"did": 0, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 1},
                {"did": 0, "mid": 7, "sid": 0, "wtype": "B", "start": 10, "end": 11},
                {"did": 1, "mid": 0, "sid": 0, "wtype": "F", "start": 0, "end": 1},
                {"did": 1, "mid": 3, "sid": 0, "wtype": "W", "start": 2, "end": 3},
            ]
        return initial

    result = tune_overlap_exemptions(
        initial,
        run_with_exemptions,
        max_iter=2,
        group_by="mid_type",
    )

    assert result.exemptions == {(3, WorkloadType.W)}
    assert result.trials[0].accepted is False
    assert result.trials[1].accepted is True
    assert result.trials[1].iteration == 2
