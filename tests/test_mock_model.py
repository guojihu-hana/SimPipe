"""Inline mock model timings (model.layer_time / layer_f/b/w_time)."""

from __future__ import annotations

import pytest

from simpipe.cli import _load_run_inputs
from simpipe.config.model import ModelConfig
from simpipe.core.executor import build_simulation
from simpipe.models.registry import mock_profile_times, uses_mock_times


def test_mock_profile_times_default_ratio() -> None:
    model = ModelConfig(name="mock_model", num_layers=4, layer_time=500)
    pt = mock_profile_times(model)
    assert pt.layer_f == [500.0] * 4
    assert pt.layer_b == [500.0] * 4  # 1:1:1 default
    assert pt.layer_w == [500.0] * 4
    assert pt.embedding_f == 0.0 and pt.head_f == 0.0


def test_mock_profile_times_overrides() -> None:
    model = ModelConfig(
        name="mock_model", num_layers=2, layer_time=100, layer_b_time=300
    )
    pt = mock_profile_times(model)
    assert pt.layer_f == [100.0, 100.0]
    assert pt.layer_b == [300.0, 300.0]
    assert pt.layer_w == [100.0, 100.0]  # W still defaults to F


def test_mock_times_apply_to_any_model_name() -> None:
    assert uses_mock_times(ModelConfig(name="mock_model"))
    assert uses_mock_times(ModelConfig(name="whatever", layer_time=10))
    assert not uses_mock_times(ModelConfig(name="nemotron-h-4B"))
    with pytest.raises(ValueError):
        mock_profile_times(ModelConfig(name="mock_model", layer_f_time=0))


def test_mock_pattern_per_type_times() -> None:
    # "ET*3ML": run-length T*3 expands; times are ms -> 0.01 ms ticks (x100).
    model = ModelConfig(
        name="mock_model",
        num_layers=4,
        pattern="ET*3ML",
        forward_ms={"T": 1.0, "M": 0.5},
        backward_ms={"T": 2.0, "M": 1.0},
        weight_ms={"T": 0.25, "M": 0.125},
    )
    pt = mock_profile_times(model)
    assert pt.layer_f == [100.0, 100.0, 100.0, 50.0]
    assert pt.layer_b == [200.0, 200.0, 200.0, 100.0]
    assert pt.layer_w == [25.0, 25.0, 25.0, 12.5]
    assert pt.embedding_f == 0.0 and pt.head_f == 0.0  # E/L default to 0


def test_mock_pattern_defaults_and_validation() -> None:
    # backward defaults to forward, weight to backward
    pt = mock_profile_times(
        ModelConfig(name="mock_model", num_layers=2, pattern="ETT",
                    forward_ms={"T": 1.0})
    )
    assert pt.layer_b == pt.layer_f == pt.layer_w == [100.0, 100.0]
    with pytest.raises(ValueError, match="num_layers"):
        mock_profile_times(
            ModelConfig(name="mock_model", num_layers=5, pattern="ET*8L",
                        forward_ms={"T": 1.0})
        )
    with pytest.raises(ValueError, match="missing pattern type"):
        mock_profile_times(
            ModelConfig(name="mock_model", num_layers=2, pattern="EMML",
                        forward_ms={"T": 1.0})
        )


def _write_config(tmp_path, extra_model: str = "", extra: str = "") -> str:
    path = tmp_path / "mock.yaml"
    path.write_text(
        f"""
profiled_data: true
model:
  name: mock_model
  num_layers: 8
  layer_time: 100
{extra_model}
parallel:
  pp_size: 2
  micro_batch_num: 4
  bwd_split: True
schedule: 1f1b
{extra}
"""
    )
    return str(path)


def test_mock_model_end_to_end(tmp_path) -> None:
    cfg, pt = _load_run_inputs(_write_config(tmp_path), "mock_model", None)
    assert pt.layer_f == [100.0] * 8
    executor = build_simulation(
        cfg,
        layer_f_times=pt.layer_f,
        layer_b_times=pt.layer_b,
        layer_w_times=pt.layer_w,
        embedding_f_time=pt.embedding_f,
        embedding_b_time=pt.embedding_b,
        embedding_w_time=pt.embedding_w,
        head_f_time=pt.head_f,
        head_b_time=pt.head_b,
        head_w_time=pt.head_w,
    )
    result = executor.run()
    # 8 layers / pp2 -> 4 layers per stage, 100 ticks each, emb/head free.
    for wtype in ("F", "B", "W"):
        durations = {r["duration"] for r in result.records if r["wtype"] == wtype}
        assert durations == {400}


def test_mock_model_with_time_scales(tmp_path) -> None:
    extra = """
batch:
  time_scales: [1, 2, 0.5, 1]
"""
    cfg, pt = _load_run_inputs(_write_config(tmp_path, extra=extra), "mock_model", None)
    executor = build_simulation(
        cfg,
        layer_f_times=pt.layer_f,
        layer_b_times=pt.layer_b,
        layer_w_times=pt.layer_w,
        embedding_f_time=pt.embedding_f,
        embedding_b_time=pt.embedding_b,
        embedding_w_time=pt.embedding_w,
        head_f_time=pt.head_f,
        head_b_time=pt.head_b,
        head_w_time=pt.head_w,
    )
    result = executor.run()
    f_by_mid = {
        r["mid"]: r["duration"]
        for r in result.records
        if r["wtype"] == "F" and r["sid"] == 0
    }
    assert f_by_mid == {0: 400, 1: 800, 2: 200, 3: 400}
