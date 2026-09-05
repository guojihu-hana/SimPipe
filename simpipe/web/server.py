"""Web UI for SimPipe: edit a config, run, inspect and export the result.

Launch (one command, Flask required):

    python -m simpipe.cli web --port 8080

Then open http://<host>:8080/ -- edit the YAML on the left, Run (Ctrl+Enter),
and read the optimized pipeline on the right: makespan/bubble/memory summary,
Gantt SVG, and the pipeline_config.yaml produced by the tuner.  Both artifacts
can be downloaded from the page.
"""
from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import yaml

from simpipe.cli import _build_executor, _config_from_data, _profile_times_for_config
from simpipe.config.pipeline_config import write_pipeline_config
from simpipe.core.executor import first_replica_records
from simpipe.metrics.comp_bubble import analyze_pipeline_comp_bubble
from simpipe.viz.gantt import write_gantt_svg

STATIC_DIR = Path(__file__).resolve().parent / "static"


class _ConfigDumper(yaml.SafeDumper):
    """Block-style mappings; inline [..] only for lists of scalars."""


def _represent_list(dumper: yaml.SafeDumper, data: list):
    flow = all(not isinstance(item, (list, dict)) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_ConfigDumper.add_representer(list, _represent_list)
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def run_simulation(yaml_text: str) -> dict:
    """Run one simulation from YAML config text; result as a JSON-safe dict."""
    data = yaml.safe_load(yaml_text) or {}
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    cfg = _config_from_data(data)
    profile = _profile_times_for_config(cfg)
    executor = _build_executor(cfg, profile)

    partition = executor.plan.partition.layer_counts(executor.graph)
    placement = executor.plan.placement.device_stages
    tuning_lines: list[str] = []
    if cfg.schedule == "octopipe" and cfg.tuning.auto_tune and cfg.partition_layers is None:
        tuning_lines.append(f"Auto-tuned partition: {partition}")
        tuning_lines.append(f"Auto-tuned placement: {placement}")
    bo = executor.batch_order_result
    batch_order = None
    if bo is not None:
        batch_order = {
            "order": list(bo.order),
            "makespan": bo.makespan,
            "baseline_makespan": bo.baseline_makespan,
            "trials": bo.trials,
            "is_identity": bo.is_identity,
        }
        if bo.is_identity:
            tuning_lines.append(
                f"Batch order: input order kept, makespan {bo.makespan:.0f} ({bo.trials} sims)"
            )
        else:
            tuning_lines.append(
                f"Batch order tuned: {bo.order} (makespan {bo.baseline_makespan:.0f} "
                f"-> {bo.makespan:.0f}, {bo.trials} sims; slot k runs input microbatch order[k])"
            )

    result = executor.run()
    # dp replicas duplicate every block on the same device rows with mids
    # offset by dp_idx * nmb; show/export/analyze the first replica only.
    records = first_replica_records(result.records, cfg.parallel.micro_batch_num)
    stats = analyze_pipeline_comp_bubble(records, device_num=cfg.parallel.pp_size)
    title = f"{cfg.model.name} {cfg.schedule}"
    with tempfile.TemporaryDirectory() as td:
        svg_path = Path(td) / "pipeline_gantt.svg"
        write_gantt_svg(
            records,
            svg_path,
            title=title,
            partition_layers=partition,
            placement=placement,
        )
        svg = svg_path.read_text()
        cfg_path = Path(td) / "pipeline_config.yaml"
        write_pipeline_config(
            cfg_path,
            executor=executor,
            makespan=result.makespan,
            scheduling_records=records,
            memory=result.memory.to_dict() if result.memory else None,
        )
        pipeline_config = cfg_path.read_text()
    memory = result.memory.to_dict() if result.memory else None
    return {
        "ok": True,
        "model": cfg.model.name,
        "schedule": cfg.schedule,
        "makespan": result.makespan,
        "stalled": bool(result.stalled),
        "bubble_ratio": stats.avg_bubble_ratio(cfg.parallel.pp_size),
        "partition": [int(x) for x in partition],
        "placement": [list(map(int, stages)) for stages in placement],
        "batch_order": batch_order,
        "tuning_lines": tuning_lines,
        "memory": memory,
        "ranks": _per_rank_rows(partition, placement, stats, memory),
        "gantt": _gantt_data(records),
        "svg": svg,
        "pipeline_config": pipeline_config,
    }


def _gantt_data(records: list[dict]) -> dict:
    """Compact block list for client-side canvas rendering.

    One entry per device: blocks = [[start, end, wtype, mid, sid], ...]
    sorted by start time.  Times are simulator ticks (0.01 ms).
    """
    devices: dict[int, list] = {}
    max_t = 0.0
    for r in records:
        start = float(r.get("start") or 0)
        end = float(r.get("end") or start + (r.get("duration") or 1))
        if end < start:
            start, end = end, start
        devices.setdefault(int(r["did"]), []).append(
            [round(start, 3), round(end, 3), str(r.get("wtype", "F")).upper(),
             int(r["mid"]), int(r.get("sid", 0))]
        )
        max_t = max(max_t, end)
    return {
        "max_t": max_t,
        "devices": [
            {"did": did, "blocks": sorted(blks)}
            for did, blks in sorted(devices.items())
        ],
    }


def _per_rank_rows(partition, placement, stats, memory) -> list[dict]:
    """One row per PP rank: partition, placement, comp/bubble, memory."""
    comp_by_did = {d.did: d for d in stats.per_device}
    mem_by_did = {d["did"]: d for d in (memory or {}).get("per_device", [])}

    def _gb(mem: dict, key: str):
        # some sizes are exported only as bytes (e.g. model_state)
        value = mem.get(f"{key}_gb")
        if value is None and mem.get(f"{key}_bytes") is not None:
            value = mem[f"{key}_bytes"] / 1024**3
        return value

    rows = []
    for rank, stages in enumerate(placement):
        comp = comp_by_did.get(rank)
        mem = mem_by_did.get(rank, {})
        rows.append(
            {
                "rank": rank,
                "stages": list(map(int, stages)),
                "layers": [int(partition[sid]) for sid in stages],
                "comp": comp.comp if comp else 0.0,
                "bubble": comp.bubble if comp else 0.0,
                "bubble_ratio": comp.bubble_ratio if comp else 0.0,
                "warmup_bubble": comp.warmup_bubble if comp else 0.0,
                "cooldown_bubble": comp.cooldown_bubble if comp else 0.0,
                "residual_bubble": comp.residual_bubble if comp else 0.0,
                "model_state_gb": _gb(mem, "model_state"),
                "activation_peak_gb": _gb(mem, "activation_peak"),
                "peak_gb": _gb(mem, "peak"),
                "hbm_gb": _gb(mem, "hbm"),
                "feasible": mem.get("feasible"),
            }
        )
    return rows


def list_examples() -> list[dict]:
    if not EXAMPLES_DIR.is_dir():
        return []
    return [
        {"name": path.name, "content": path.read_text()}
        for path in sorted(EXAMPLES_DIR.glob("*.yaml"))
    ]


def create_app():
    try:
        from flask import Flask, jsonify, request, send_from_directory
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The web UI needs Flask: pip install -e '.[web]' (or pip install flask)"
        ) from exc

    app = Flask("simpipe_web", static_folder=str(STATIC_DIR), static_url_path="/static")

    @app.get("/")
    def index():
        return send_from_directory(str(STATIC_DIR), "index.html")

    @app.get("/api/examples")
    def examples():
        return jsonify(list_examples())

    @app.get("/api/options")
    def api_options():
        """Selectable values for form dropdowns (models, profile files)."""
        import os

        from simpipe.models.pattern import EMBEDDING, HEAD
        from simpipe.models.registry import (
            PRESETS,
            PROFILES_DIR,
            profile_data,
            profiled_model_names,
        )

        paths = []
        for p in sorted(PROFILES_DIR.glob("*.json")):
            rel = os.path.relpath(p, Path.cwd())
            paths.append(rel if not rel.startswith("..") else str(p))
        # profiled models plus the synthetic mock; analytic-only presets
        # (gpt, deepseek, test fixtures) are not exposed in the UI.
        names = ["mock_model"] + sorted(profiled_model_names())

        # per-model config values the UI fills in when the model is switched:
        # preset metadata, with num_layers derived from the profiled pattern.
        # model.num_layers counts transformer-body layers only, so the
        # pattern's embedding (E) / head (L) symbols are excluded.
        meta: dict[str, dict] = {}
        # layer detail for the partition editor: pattern incl. E/L plus the
        # per-symbol f/b/w times (ms) from the profile fit
        layers: dict[str, dict] = {}
        for n in names:
            if n == "mock_model":
                # the mock's properties are defined by the user / UI defaults,
                # not by the registry fixture preset
                continue
            m = {k: v for k, v in PRESETS.get(n, {}).get("model", {}).items()
                 if k != "name"}
            prof = profile_data(n)
            if prof and prof.get("pattern"):
                m["num_layers"] = sum(
                    1 for c in prof["pattern"] if c not in (EMBEDDING, HEAD)
                )
                layers[n] = {
                    "pattern": prof["pattern"],
                    "f": prof.get("forward_ms") or {},
                    "b": prof.get("backward_ms") or {},
                    "w": prof.get("weight_ms") or {},
                }
            meta[n] = m
        return jsonify({
            "models": names,
            "profile_paths": paths,
            "model_meta": meta,
            "model_layers": layers,
        })

    @app.post("/api/parse")
    def api_parse():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            data = yaml.safe_load(payload.get("config") or "") or {}
            if not isinstance(data, dict):
                raise ValueError("config must be a YAML mapping")
        except Exception as exc:
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify({"ok": True, "data": data})

    @app.post("/api/dump")
    def api_dump():
        payload = request.get_json(force=True, silent=True) or {}
        data = payload.get("data")
        if not isinstance(data, dict):
            return jsonify({"ok": False, "error": "data must be a mapping"}), 400
        text = yaml.dump(
            data, Dumper=_ConfigDumper, sort_keys=False, allow_unicode=True
        )
        return jsonify({"ok": True, "text": text})

    @app.post("/api/run")
    def api_run():
        payload = request.get_json(force=True, silent=True) or {}
        text = payload.get("config") or ""
        try:
            return jsonify(run_simulation(text))
        except Exception as exc:  # user config errors surface in the UI
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                ),
                400,
            )

    return app


def serve(host: str = "0.0.0.0", port: int = 8080, debug: bool = False) -> None:
    app = create_app()
    print(f"SimPipe web UI on http://{host}:{port}/  (Ctrl+C to stop)")
    app.run(host=host, port=port, debug=debug, threaded=True)
