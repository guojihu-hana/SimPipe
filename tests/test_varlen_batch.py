"""Variable-length microbatch (pack/pad) time and memory accounting."""

from __future__ import annotations

import pytest

from simpipe.config.batch import BatchConfig
from simpipe.config.sim_config import SimConfig
from simpipe.core.executor import build_simulation


def test_batch_config_pack_scales() -> None:
    batch = BatchConfig(mode="pack", microbatches=[[2048, 2048], [4096], [8192]])
    # [2048, 2048]: linear 4096/4096 = 1, quadratic 2*2048^2 / 4096^2 = 0.5.
    assert batch.scales(1, 4096) == [(1.0, 0.5), (1.0, 1.0), (2.0, 4.0)]


def test_batch_config_pad_scales() -> None:
    batch = BatchConfig(mode="pad", microbatches=[[2048, 1024], [4096]])
    # Pads to 2 x 2048: linear 4096/4096, quadratic 2*2048^2 / 4096^2.
    assert batch.scales(1, 4096) == [(1.0, 0.5), (1.0, 1.0)]


def test_batch_config_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        BatchConfig(mode="truncate", microbatches=[[1024]])
    with pytest.raises(ValueError):
        BatchConfig(mode="pack", microbatches=[])
    with pytest.raises(ValueError):
        BatchConfig(mode="pack", microbatches=[[]])
    with pytest.raises(ValueError):
        BatchConfig(mode="pack", microbatches=[[0]])
    with pytest.raises(ValueError):  # both specs at once
        BatchConfig(microbatches=[[1024]], time_scales=[1.0])
    with pytest.raises(ValueError):
        BatchConfig(time_scales=[1.0, 0.0])
    with pytest.raises(ValueError):
        BatchConfig(time_scales=[1.0], time_ref=0)


def test_batch_config_time_scales() -> None:
    batch = BatchConfig(time_scales=[1, 1.2, 1.8])
    assert batch.num_microbatches == 3
    # Linear and quadratic components carry the same factor.
    assert batch.scales(1, 4096) == [(1.0, 1.0), (1.2, 1.2), (1.8, 1.8)]


def test_batch_config_time_scales_with_ref() -> None:
    batch = BatchConfig(time_scales=[2000, 3000, 4000], time_ref=2000)
    assert batch.scales(1, 4096) == [(1.0, 1.0), (1.5, 1.5), (2.0, 2.0)]


def _config_dict(batch: dict | None, **parallel_overrides) -> dict:
    parallel = {"pp_size": 2, "micro_batch_num": 3, **parallel_overrides}
    data = {
        "model": {
            "name": "varlen-test",
            "hidden_size": 512,
            "num_layers": 8,
            "num_attention_heads": 8,
            "seq_len": 4096,
            "vocab_size": 32000,
        },
        "parallel": parallel,
        "schedule": "1f1b",
    }
    if batch is not None:
        data["batch"] = batch
    return data


def test_sim_config_batch_sets_micro_batch_num() -> None:
    data = _config_dict({"mode": "pack", "microbatches": [[4096], [2048], [1024]]})
    del data["parallel"]["micro_batch_num"]
    cfg = SimConfig.from_dict(data)
    assert cfg.parallel.micro_batch_num == 3
    assert cfg.batch is not None and cfg.batch.mode == "pack"


def test_sim_config_batch_micro_batch_num_mismatch_raises() -> None:
    data = _config_dict(
        {"mode": "pack", "microbatches": [[4096], [2048]]}, micro_batch_num=3
    )
    with pytest.raises(ValueError, match="micro_batch_num"):
        SimConfig.from_dict(data)


def _run(batch: dict | None):
    cfg = SimConfig.from_dict(_config_dict(batch))
    executor = build_simulation(cfg)
    result = executor.run()
    return executor, result


def _f_durations_by_mid(records: list[dict], sid: int) -> dict[int, float]:
    return {
        r["mid"]: r["duration"]
        for r in records
        if r["wtype"] == "F" and r["sid"] == sid
    }


def test_varlen_durations_scale_per_microbatch() -> None:
    executor, result = _run(
        {"mode": "pack", "microbatches": [[4096], [2048, 2048], [8192]]}
    )
    plan = executor.plan
    assert plan.mid_scales == [(1.0, 1.0), (1.0, 0.5), (2.0, 4.0)]

    for sid in range(plan.stage_num):
        base = plan.timing_for_stage(sid)
        assert base.f_quad > 0  # attention ops present in every stage
        durations = _f_durations_by_mid(result.records, sid)
        for mid, (lin, quad) in enumerate(plan.mid_scales):
            expected = (base.f_time - base.f_quad) * lin + base.f_quad * quad
            # Workload truncates durations to integer ticks.
            assert durations[mid] == max(1, int(expected))
        # Same token count packed into two docs beats one 4096 doc
        # (quadratic attention shrinks); a 8192-token doc costs the most.
        assert durations[1] < durations[0] < durations[2]


def test_uniform_batch_matches_unbatched_run() -> None:
    _, uniform = _run({"mode": "pack", "microbatches": [[4096]] * 3})
    _, plain = _run(None)
    assert uniform.makespan == pytest.approx(plain.makespan)


def test_pad_costs_at_least_pack() -> None:
    seqs = [[4096], [2048, 512], [3072, 1024, 256]]
    _, packed = _run({"mode": "pack", "microbatches": seqs})
    _, padded = _run({"mode": "pad", "microbatches": seqs})
    assert padded.makespan > packed.makespan

    pack_act = max(d.activation_peak_bytes for d in packed.memory.per_device)
    pad_act = max(d.activation_peak_bytes for d in padded.memory.per_device)
    assert pad_act >= pack_act


def test_memory_peak_weights_microbatch_tokens() -> None:
    _, small = _run({"mode": "pack", "microbatches": [[4096], [1024], [1024]]})
    _, full = _run({"mode": "pack", "microbatches": [[4096], [4096], [4096]]})
    # Device 0 overlaps mid 0 and mid 1 during 1F1B warmup: the small run
    # holds 1.25x one reference microbatch, the full run 2x.  (The global
    # peak sits on the last device, dominated by the head/CE buffers of a
    # single 4096 microbatch, identical in both runs.)
    small_act = small.memory.per_device[0].activation_peak_bytes
    full_act = full.memory.per_device[0].activation_peak_bytes
    assert small_act < full_act
    assert small_act == pytest.approx(full_act * 1.25 / 2.0, rel=1e-6)


def test_time_scales_multiply_all_durations() -> None:
    executor, result = _run({"time_scales": [2000, 3000, 4000], "time_ref": 2000})
    plan = executor.plan
    assert plan.mid_scales == [(1.0, 1.0), (1.5, 1.5), (2.0, 2.0)]
    for sid in range(plan.stage_num):
        base = plan.timing_for_stage(sid)
        durations = _f_durations_by_mid(result.records, sid)
        for mid, (scale, _quad) in enumerate(plan.mid_scales):
            assert durations[mid] == max(1, int(base.f_time * scale))


def test_admission_weight_scales_with_tokens() -> None:
    executor, _ = _run({"mode": "pack", "microbatches": [[4096], [1024], [8192]]})
    runtime = executor.pipelines[0]
    base = runtime._stage_layer_weight[0]
    assert runtime._act_weight(0, 0) == pytest.approx(base)
    assert runtime._act_weight(0, 1) == pytest.approx(base * 0.25)
    assert runtime._act_weight(0, 2) == pytest.approx(base * 2.0)
