import pytest

from simpipe.pipeline.placement import Placement, sort_placement_by_first_stage, validate_placement


def test_interleaved_placement():
    p = Placement.interleaved(4, 1)
    p.validate(4, 4)
    assert len(p.device_stages) == 4


def test_sort_placement():
    raw = [[2], [0], [3], [1]]
    sorted_p = sort_placement_by_first_stage(raw)
    assert sorted_p[0][0] == 0


def test_validate_duplicate_stage():
    with pytest.raises(ValueError):
        Placement([[0, 1], [0]]).validate(2, 2)
