"""Multimodal encoder/decoder simulation: data dependencies and placements."""
from __future__ import annotations

from collections import defaultdict

import pytest

from simpipe.cli import (_build_executor, _config_from_data,
                         _profile_times_for_config)

COMM = 40  # 0.4 ms p2p latency in 0.01 ms ticks
NMB = 8


def base_data(schedule: str = "1f1b") -> dict:
    return {
        "schedule": schedule,
        "model": {
            "name": "mock_model",
            "num_layers": 16,
            "pattern": "ET*16L",
            "forward_ms": {"T": 1.0},
        },
        "parallel": {"pp_size": 4, "micro_batch_num": NMB},
        "hardware": {"p2p_latency_ms": 0.4},
    }


def run(data: dict):
    cfg = _config_from_data(data)
    ex = _build_executor(cfg, _profile_times_for_config(cfg))
    res = ex.run()
    assert not res.stalled
    return ex, res


def split_records(res):
    backbone, aux = [], defaultdict(list)
    for r in res.records:
        if "aux" in r:
            aux[r["aux"]].append(r)
        else:
            backbone.append(r)
    return backbone, aux


def by_key(recs):
    return {(r["mid"], r["wtype"]): r for r in recs}


def stage_recs(backbone, sid):
    return {(r["mid"], r["wtype"]): r for r in backbone if r["sid"] == sid}


def assert_no_overlap(res):
    per_dev = defaultdict(list)
    for r in res.records:
        per_dev[r["did"]].append((r["start"], r["end"]))
    for did, spans in per_dev.items():
        spans.sort()
        for (_, e1), (s2, _) in zip(spans, spans[1:]):
            assert s2 >= e1, f"device {did} runs two workloads at once"


def test_encoder_decoder_dependencies_first_last_stage():
    data = base_data()
    data["encoders"] = [{"name": "enc", "forward_ms": 2.0, "placement": "first_stage"}]
    data["decoders"] = [{"name": "dec", "forward_ms": 2.0, "placement": "last_stage"}]
    _, res = run(data)
    assert_no_overlap(res)
    backbone, aux = split_records(res)
    enc, dec = by_key(aux["enc"]), by_key(aux["dec"])
    s0, sl = stage_recs(backbone, 0), stage_recs(backbone, 3)
    # bwd_split off: W folds into B, so each module runs F+B per microbatch
    assert len(aux["enc"]) == NMB * 2
    for m in range(NMB):
        assert s0[(m, "F")]["start"] >= enc[(m, "F")]["end"]
        assert enc[(m, "B")]["start"] >= s0[(m, "B")]["end"]
        assert dec[(m, "F")]["start"] >= sl[(m, "F")]["end"]
        assert dec[(m, "B")]["start"] >= dec[(m, "F")]["end"]
        assert sl[(m, "B")]["start"] >= dec[(m, "B")]["end"]


def test_dedicated_encoder_adds_device_and_p2p_latency():
    data = base_data()
    data["encoders"] = [{"name": "enc", "forward_ms": 2.0, "placement": "dedicated"}]
    _, res = run(data)
    backbone, aux = split_records(res)
    assert {r["did"] for r in aux["enc"]} == {4}  # appended after 4 backbone devices
    enc = by_key(aux["enc"])
    s0 = stage_recs(backbone, 0)
    for m in range(NMB):
        assert s0[(m, "F")]["start"] >= enc[(m, "F")]["end"] + COMM
        assert enc[(m, "B")]["start"] >= s0[(m, "B")]["end"] + COMM


def test_replicated_encoder_shards_microbatches_dp():
    """replicated = module DP over all backbone devices: one weight copy per
    rank, microbatches round-robin, each runs its encoder exactly once."""
    data = base_data()
    data["encoders"] = [{"name": "enc", "forward_ms": 1.0, "placement": "replicated"}]
    _, res = run(data)
    assert_no_overlap(res)
    backbone, aux = split_records(res)
    assert sorted(aux) == [f"enc@d{d}" for d in range(4)]
    s0 = stage_recs(backbone, 0)
    seen_mids = []
    for d in range(4):
        copies = aux[f"enc@d{d}"]
        assert all(r["did"] == d for r in copies)
        shard = {r["mid"] for r in copies}
        assert shard == {m for m in range(NMB) if m % 4 == d}  # round-robin
        seen_mids += [r["mid"] for r in copies if r["wtype"] == "F"]
        ck = by_key(copies)
        for m in sorted(shard):
            gap = 0 if d == 0 else COMM  # non-local shards feed stage 0 via P2P
            assert s0[(m, "F")]["start"] >= ck[(m, "F")]["end"] + gap
            assert ck[(m, "B")]["start"] >= s0[(m, "B")]["end"] + gap
    # each microbatch ran its encoder F exactly once across all copies
    assert sorted(seen_mids) == list(range(NMB))
    # copies on different ranks run in parallel: the first F of every shard
    # starts at t=0 (no cross-copy serialization)
    for d in range(4):
        first = min(r["start"] for r in aux[f"enc@d{d}"] if r["wtype"] == "F")
        assert first == 0


def test_explicit_device_list_shards_microbatches():
    data = base_data()
    data["encoders"] = [{"name": "enc", "forward_ms": 2.0, "placement": [0, 1]}]
    _, res = run(data)
    backbone, aux = split_records(res)
    assert {r["mid"] for r in aux["enc@d0"]} == {0, 2, 4, 6}
    assert {r["mid"] for r in aux["enc@d1"]} == {1, 3, 5, 7}
    s0 = stage_recs(backbone, 0)
    shards = {0: by_key(aux["enc@d0"]), 1: by_key(aux["enc@d1"])}
    for m in range(NMB):
        gap = 0 if m % 2 == 0 else COMM  # the d1 shard feeds stage 0 across devices
        assert s0[(m, "F")]["start"] >= shards[m % 2][(m, "F")]["end"] + gap


def test_multiple_encoders_run_in_parallel():
    data = base_data()
    data["encoders"] = [
        {"name": "e1", "forward_ms": 2.0, "placement": "dedicated"},
        {"name": "e2", "forward_ms": 2.0, "placement": "dedicated"},
    ]
    _, res = run(data)
    backbone, aux = split_records(res)
    e1, e2 = by_key(aux["e1"]), by_key(aux["e2"])
    # no data dependency between encoders: both start at t=0 on their devices
    assert e1[(0, "F")]["start"] == 0
    assert e2[(0, "F")]["start"] == 0
    s0 = stage_recs(backbone, 0)
    for m in range(NMB):
        both = max(e1[(m, "F")]["end"], e2[(m, "F")]["end"])
        assert s0[(m, "F")]["start"] >= both + COMM  # AND-gate over all encoders


@pytest.mark.parametrize(
    "schedule,extra",
    [
        ("1f1b", {}),
        ("afab", {}),
        ("zbh", {}),
        ("interleaved", {"parallel": {"pp_size": 4, "micro_batch_num": NMB, "chunk_num": 2}}),
        ("octopipe", {}),
    ],
)
def test_all_schedules_respect_aux_edges(schedule, extra):
    data = base_data(schedule)
    data.update(extra)
    data["encoders"] = [{"name": "enc", "forward_ms": 2.0, "placement": "first_stage"}]
    data["decoders"] = [{"name": "dec", "forward_ms": 2.0, "placement": "last_stage"}]
    _, res = run(data)
    assert_no_overlap(res)
    backbone, aux = split_records(res)
    enc, dec = by_key(aux["enc"]), by_key(aux["dec"])
    last_sid = max(r["sid"] for r in backbone)
    s0, sl = stage_recs(backbone, 0), stage_recs(backbone, last_sid)
    for m in range(NMB):
        assert s0[(m, "F")]["start"] >= enc[(m, "F")]["end"]
        assert dec[(m, "F")]["start"] >= sl[(m, "F")]["end"]
        assert sl[(m, "B")]["start"] >= dec[(m, "B")]["end"]
        assert enc[(m, "B")]["start"] >= s0[(m, "B")]["end"]


def test_zbh_split_creates_aux_w_workloads():
    data = base_data("zbh")  # zbh forces bwd_split
    data["encoders"] = [{"name": "enc", "forward_ms": 2.0, "placement": "first_stage"}]
    _, res = run(data)
    _, aux = split_records(res)
    kinds = {r["wtype"] for r in aux["enc"]}
    assert kinds == {"F", "B", "W"}
    assert len(aux["enc"]) == NMB * 3


def test_dp_replicas_offset_aux_mids():
    data = base_data()
    data["parallel"] = {"pp_size": 4, "micro_batch_num": NMB, "dp_size": 2}
    data["encoders"] = [{"name": "enc", "forward_ms": 2.0, "placement": "first_stage"}]
    _, res = run(data)
    _, aux = split_records(res)
    assert sorted({r["mid"] for r in aux["enc"]}) == list(range(2 * NMB))


def test_octopipe_tuning_rebalances_for_first_stage_encoder():
    data = base_data("octopipe")
    ex_plain, _ = run(data)
    data_enc = base_data("octopipe")
    data_enc["encoders"] = [
        {"name": "vit", "forward_ms": 6.0, "backward_ms": 12.0, "placement": "first_stage"}
    ]
    ex_enc, res = run(data_enc)
    plain = ex_plain.plan.layers_per_stage
    tuned = ex_enc.plan.layers_per_stage
    # the heavy encoder must push layers off stage 0 relative to what stage 0
    # would carry under a uniform split of the same stage count
    assert tuned[0] < 16 / len(tuned), (plain, tuned)
    assert sum(tuned) == 16


def test_memory_includes_aux_params():
    data = base_data()
    data["encoders"] = [
        {"name": "enc", "forward_ms": 1.0, "placement": "replicated", "params_gb": 1.0}
    ]
    data["decoders"] = [
        {"name": "dec", "forward_ms": 1.0, "placement": "dedicated", "params_gb": 2.0}
    ]
    _, res = run(data)
    per_dev = res.memory.per_device
    assert len(per_dev) == 5  # 4 backbone + 1 dedicated
    gib = 1024**3
    # replicated encoder: every backbone device pays params+grad+master+moments
    base_run = run(base_data())[1]
    for dev, plain in zip(per_dev[:4], base_run.memory.per_device):
        assert dev.model_state_bytes - plain.model_state_bytes == 8 * gib
    # dedicated decoder device: 2 GB params -> 16 GB model state, no backbone
    assert per_dev[4].stage_ids == ()
    assert per_dev[4].model_state_bytes == 16 * gib


def test_no_aux_keeps_baseline_makespan():
    _, plain = run(base_data())
    data = base_data()
    data["encoders"] = []
    data["decoders"] = []
    _, empty = run(data)
    assert plain.makespan == empty.makespan


@pytest.mark.parametrize("schedule", ["1f1b", "interleaved", "zbh", "afab", "bapar"])
def test_static_schedule_aux_chain_order(schedule):
    """Static schedules splice anchored aux copies into the entry list, so
    every microbatch runs enc F -> bb F ... bb B -> enc B in strict order:
    no encoder F runs early during warmup, and no encoder B is deferred by a
    later backbone F.  Decoders run right after bb F / right before bb B on
    the last rank.  (recycle has no static entry list -- it schedules by
    rule, so its aux copies backfill and only the dependency chain holds.)
    """
    data = base_data(schedule)
    if schedule == "interleaved":
        data["parallel"]["chunk_num"] = 2
    if schedule == "zbh":
        data["parallel"]["bwd_split"] = True
    data["encoders"] = [{"name": "enc", "forward_ms": 0.5, "placement": "first_stage"}]
    data["decoders"] = [{"name": "dec", "forward_ms": 0.5, "placement": "last_stage"}]
    _, res = run(data)
    assert_no_overlap(res)

    def device_seq(did, keep):
        recs = [r for r in res.records if r["did"] == did and keep(r)]
        recs.sort(key=lambda r: r["start"])
        return [(r.get("aux", "bb"), r["wtype"], r["mid"]) for r in recs]

    # encoder device: only stage-0 backbone events + encoder events
    s0 = device_seq(0, lambda r: r.get("aux") == "enc" or ("aux" not in r and r["sid"] == 0))
    for k, t in enumerate(s0):
        if t[:2] == ("enc", "F"):
            assert s0[k + 1] == ("bb", "F", t[2]), f"enc F{t[2]} then {s0[k + 1]}"
        if t[:2] == ("bb", "B"):
            assert s0[k + 1] == ("enc", "B", t[2]), f"bb B{t[2]} then {s0[k + 1]}"
    # decoder device: last-stage backbone events + decoder events
    last_sid = max(r["sid"] for r in res.records if "aux" not in r)
    sl = device_seq(3, lambda r: r.get("aux") == "dec"
                    or ("aux" not in r and r["sid"] == last_sid))
    for k, t in enumerate(sl):
        if t[:2] == ("bb", "F"):
            assert sl[k + 1] == ("dec", "F", t[2]), f"bb F{t[2]} then {sl[k + 1]}"
        if t[:2] == ("bb", "B"):
            assert sl[k - 1] == ("dec", "B", t[2]), f"{sl[k - 1]} then bb B{t[2]}"


def test_afab_decoder_backward_in_ab_phase():
    """AFAB: spliced decoder Bs sit right before their backbone B, so they
    all run in the AB phase -- none may fire eagerly after its own F while
    the AF phase is still going.  Encoders are exempt from the in-flight
    throttle there (the AF phase legitimately keeps all activations alive),
    so no cap search runs and the makespan stays clean."""
    data = base_data("afab")
    data["encoders"] = [{"name": "enc", "forward_ms": 0.5, "placement": "first_stage"}]
    data["decoders"] = [{"name": "dec", "forward_ms": 0.5, "placement": "last_stage"}]
    ex, res = run(data)
    assert ex.plan.aux_plan.encoder_inflight_limit is None  # tune skipped
    backbone, aux = split_records(res)
    last_sid = max(r["sid"] for r in backbone)
    af_end = max(r["end"] for r in backbone
                 if r["sid"] == last_sid and r["wtype"] == "F")
    for r in aux["dec"]:
        if r["wtype"] == "B":
            assert r["start"] >= af_end, f"dec B{r['mid']} ran in the AF phase"


def _dedicated_encoder_data(**tuning) -> dict:
    data = base_data()
    data["encoders"] = [
        {"name": "enc", "forward_ms": 1.0, "placement": "dedicated", "act_gb": 0.5}
    ]
    if tuning:
        data["tuning"] = tuning
    return data


def test_encoder_inflight_throttle_keeps_makespan():
    """aux_memory_opt (default on) caps encoder in-flight activations at the
    smallest value that keeps the uncapped makespan.  Without it a dedicated
    encoder front-loads every microbatch (peak == NMB)."""
    from simpipe.memory.estimate import aux_inflight_peaks

    ex_eager, eager = run(_dedicated_encoder_data(aux_memory_opt=False))
    ex_lazy, lazy = run(_dedicated_encoder_data())
    assert ex_eager.plan.aux_plan.encoder_inflight_limit is None
    limit = ex_lazy.plan.aux_plan.encoder_inflight_limit
    assert limit is not None and limit < NMB
    assert lazy.makespan == eager.makespan  # zero bubble cost
    sid = ex_lazy.plan.aux_plan.instances[0].aux_sid
    assert aux_inflight_peaks(eager.records)[sid] == NMB
    assert aux_inflight_peaks(lazy.records)[sid] <= limit


def test_explicit_inflight_limit_respected():
    from simpipe.cli import (_build_executor, _config_from_data,
                             _profile_times_for_config)
    from simpipe.memory.estimate import aux_inflight_peaks

    ex, res = run(_dedicated_encoder_data(aux_inflight_limit=8))
    assert ex.plan.aux_plan.encoder_inflight_limit == 8
    sid = ex.plan.aux_plan.instances[0].aux_sid
    assert aux_inflight_peaks(res.records)[sid] <= 8

    # a cap below the warmup burst (pp+1 = 5) deadlocks the static order;
    # the simulation must detect it as stalled, not hang
    cfg = _config_from_data(_dedicated_encoder_data(aux_inflight_limit=3))
    stalled = _build_executor(cfg, _profile_times_for_config(cfg)).run()
    assert stalled.stalled


def test_aux_act_memory_charged_by_peak():
    """act_gb enters the activation estimate as peak_inflight * act_gb."""
    from simpipe.memory.estimate import aux_inflight_peaks

    ex, res = run(_dedicated_encoder_data(aux_memory_opt=False))
    inst = ex.plan.aux_plan.instances[0]
    peak = aux_inflight_peaks(res.records)[inst.aux_sid]
    dev = res.memory.per_device[inst.device_id]
    assert dev.activation_peak_bytes == int(0.5 * peak * 1024**3)


def test_aux_recompute_extends_backward():
    """recompute mirrors the backbone's: the module's B re-runs its F first
    (B duration = F+B), W untouched."""
    plain = base_data()
    plain["encoders"] = [
        {"name": "enc", "forward_ms": 1.0, "backward_ms": 2.0, "weight_ms": 0.0,
         "placement": "dedicated"}
    ]
    rec = base_data()
    rec["encoders"] = [
        {"name": "enc", "forward_ms": 1.0, "backward_ms": 2.0, "weight_ms": 0.0,
         "placement": "dedicated", "recompute": True}
    ]
    _, r_plain = run(plain)
    _, r_rec = run(rec)
    b_plain = [x for x in split_records(r_plain)[1]["enc"] if x["wtype"] == "B"][0]
    b_rec = [x for x in split_records(r_rec)[1]["enc"] if x["wtype"] == "B"][0]
    assert b_plain["end"] - b_plain["start"] == 200
    assert b_rec["end"] - b_rec["start"] == 300  # +100 ticks of re-run F
    # model-spec path folds the same way; a top-level recompute flag lands
    # on model.recompute (an aux module IS a model)
    from simpipe.config.multimodal import AuxModuleConfig
    c = AuxModuleConfig.from_dict(
        {"model": {"pattern": "T*4", "forward_ms": {"T": 0.5}}, "recompute": True},
        default_name="e")
    assert c.model.recompute is True
    assert c.f_ticks == 200 and c.b_ticks == 400  # B = F(200) + B(200)
    # the canonical spelling puts recompute inside the model mapping
    c2 = AuxModuleConfig.from_dict(
        {"model": {"pattern": "T*4", "forward_ms": {"T": 0.5}, "recompute": True}},
        default_name="e")
    assert c2.b_ticks == 400


def test_aux_profiled_model():
    """model.name can be any profiled model: pattern/layers/times come from
    the registry profile, exactly like the backbone's model loading."""
    from simpipe.config.multimodal import AuxModuleConfig
    from simpipe.models.registry import (profile_times_for_model,
                                         profiled_model_names)

    name = sorted(profiled_model_names())[0]
    c = AuxModuleConfig.from_dict({"model": {"name": name}}, default_name="e")
    assert c.model.name == name and c.model.num_layers > 0
    pt = profile_times_for_model(c.model)
    assert c.f_ticks == sum(pt.layer_f) + (pt.embedding_f or 0) + (pt.head_f or 0)
    assert c.b_ticks > 0 and c.f_ticks > 0

    # and it simulates end to end as an encoder
    data = base_data()
    data["encoders"] = [{"name": "enc", "model": {"name": name}, "placement": "dedicated"}]
    _, result = run(data)
    assert not result.stalled
    enc_f = [x for x in split_records(result)[1]["enc"] if x["wtype"] == "F"]
    assert enc_f and enc_f[0]["end"] - enc_f[0]["start"] == c.f_ticks

    import pytest

    with pytest.raises(ValueError, match="unknown model"):
        AuxModuleConfig.from_dict({"model": {"name": "no_such_model"}}, default_name="e")
    with pytest.raises(ValueError, match="unknown model field"):
        AuxModuleConfig.from_dict({"model": {"pattern": "T", "fwd_ms": {}}}, default_name="e")


def test_aux_model_spec_sums_layers():
    """An aux 'model' spec (backbone-style mock: pattern + per-type ms)
    yields the same simulation as the equivalent flat total scalars."""
    layered = base_data()
    layered["encoders"] = [{
        "name": "vit",
        "model": {"name": "mock_model", "num_layers": 4, "pattern": "T*4",
                  "forward_ms": {"T": 0.5}, "backward_ms": {"T": 1.0}},
        "placement": "first_stage",
    }]
    flat = base_data()
    flat["encoders"] = [
        {"name": "vit", "forward_ms": 2.0, "backward_ms": 4.0, "placement": "first_stage"}
    ]
    _, r_layered = run(layered)
    _, r_flat = run(flat)
    assert r_layered.makespan == r_flat.makespan
    (_, aux_l), (_, aux_f) = split_records(r_layered), split_records(r_flat)
    assert by_key(aux_l["vit"]).keys() == by_key(aux_f["vit"]).keys()
    for k, rec in by_key(aux_l["vit"]).items():
        assert rec["start"] == by_key(aux_f["vit"])[k]["start"]
