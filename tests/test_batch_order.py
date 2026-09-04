"""Microbatch execution-order tuning under variable-length batch specs."""

from __future__ import annotations

import pytest

from simpipe.config.sim_config import SimConfig
from simpipe.core.executor import build_simulation
from simpipe.tuning.batch_order import heuristic_orders, tune_batch_order


def test_heuristic_orders_shapes() -> None:
    scales = [(1.0, 1.0), (3.0, 3.0), (2.0, 2.0), (0.5, 0.5)]
    orders = heuristic_orders(scales)
    assert [0, 1, 2, 3] in orders  # identity
    assert [3, 0, 2, 1] in orders  # ascending by scale
    assert [1, 2, 0, 3] in orders  # descending
    # Valley: shortest at both ends, longest in the middle.
    valley = [3, 2, 1, 0]
    assert valley in orders
    for order in orders:
        assert sorted(order) == [0, 1, 2, 3]


def test_tune_batch_order_finds_known_optimum() -> None:
    # Synthetic cost: strongly prefers ascending scale sequence.
    scales = [(2.0, 2.0), (1.0, 1.0), (3.0, 3.0)]

    def evaluate(order):
        seq = [scales[i][0] for i in order]
        return sum(v * (pos + 1) for pos, v in enumerate(seq))

    result = tune_batch_order(scales, evaluate)
    assert [scales[i][0] for i in result.order] == [3.0, 2.0, 1.0]
    assert result.makespan <= result.baseline_makespan
    assert result.trials <= 10  # dedup keeps the budget small


def test_tune_batch_order_dedups_equal_scales() -> None:
    scales = [(1.0, 1.0)] * 3 + [(2.0, 2.0)]
    calls = []

    def evaluate(order):
        calls.append(order)
        return 1.0

    result = tune_batch_order(scales, evaluate)
    # Only 4 distinct scale sequences exist (position of the long mb).
    assert len(calls) <= 4
    assert result.is_identity


def _config(order_tune: bool | None) -> SimConfig:
    data = {
        "profiled_data": False,
        "model": {
            "name": "varlen-test",
            "hidden_size": 512,
            "num_layers": 8,
            "num_attention_heads": 8,
            "seq_len": 4096,
            "vocab_size": 32000,
        },
        "parallel": {"pp_size": 2, "micro_batch_num": 4},
        "schedule": "1f1b",
        "batch": {"time_scales": [3, 1, 1, 1]},
    }
    if order_tune is not None:
        data["tuning"] = {"batch_order_tune": order_tune}
    return SimConfig.from_dict(data)


def test_batch_order_tune_end_to_end() -> None:
    baseline_exec = build_simulation(_config(False))
    baseline = baseline_exec.run().makespan
    assert baseline_exec.batch_order_result is None

    tuned_exec = build_simulation(_config(True))
    bo = tuned_exec.batch_order_result
    assert bo is not None
    assert bo.baseline_makespan == pytest.approx(baseline)
    assert bo.makespan <= bo.baseline_makespan
    result = tuned_exec.run()
    assert result.makespan == pytest.approx(bo.makespan)
    if not bo.is_identity:
        assert tuned_exec.plan.mid_order == bo.order
        scales = _config(False).batch.scales(1, 4096)
        assert tuned_exec.plan.mid_scales == [scales[i] for i in bo.order]


def test_batch_order_tune_defaults_follow_auto_tune() -> None:
    # batch_order_tune unset + auto_tune off (1f1b) -> no order search.
    executor = build_simulation(_config(None))
    assert executor.batch_order_result is None
