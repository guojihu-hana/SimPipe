from __future__ import annotations

from pathlib import Path

import click
import yaml

from simpipe.config.pipeline_config import write_pipeline_config
from simpipe.config.sim_config import SimConfig
from simpipe.core.executor import build_simulation, first_replica_records
from simpipe.metrics.comp_bubble import analyze_pipeline_comp_bubble
from simpipe.models.registry import (
    MOCK_MODEL_NAME,
    get_preset,
    get_profile_times,
    mock_profile_times,
    profile_data,
    timing_model_data,
    uses_mock_times,
)
from simpipe.tuning.bubble_overlap import format_group
from simpipe.tuning.sweep import run_sweep, sweep_configs
from simpipe.viz.gantt import format_gantt_detailed_info, write_gantt_svg


_MOCK_TIME_KEYS = (
    "layer_time", "layer_f_time", "layer_b_time", "layer_w_time", "pattern",
)


def _config_from_data(data: dict) -> SimConfig:
    """SimConfig from a parsed YAML dict; profiled presets fill model fields.

    When the YAML does not set ``profiled_data`` explicitly, it is inferred:
    mock timings, an external profile path, or a registry profile for the
    model name all enable per-layer profiled times.  An explicit ``false``
    keeps the analytic timing formulas.
    """
    if "profiled_data" not in data:
        md = data.get("model") or {}
        name = md.get("name")
        inferred = (
            name == MOCK_MODEL_NAME
            or any(md.get(k) is not None for k in _MOCK_TIME_KEYS)
            or bool(md.get("profile_times_path"))
            or bool(name and profile_data(name))
        )
        data = {**data, "profiled_data": inferred}
    if data.get("profiled_data"):
        model_data = data.get("model") or {}
        model_name = model_data.get("name")
        merged = timing_model_data(model_name) if model_name else None
        if merged is not None:
            data = {**data, "model": {**merged, **model_data}}
    return SimConfig.from_dict(data)


def _load_config_with_profiled_defaults(config: str) -> SimConfig:
    return _config_from_data(yaml.safe_load(Path(config).read_text()) or {})


def _profile_times_for_config(cfg: SimConfig):
    """ProfileTimes for a config (mock, external YAML, or registry), or None."""
    if not cfg.profiled_data:
        return None
    if uses_mock_times(cfg.model):
        pt = mock_profile_times(cfg.model)
    elif cfg.model.profile_times_path:
        from simpipe.models.profile_times import profile_times_from_preset

        data = yaml.safe_load(Path(cfg.model.profile_times_path).read_text())
        pt = profile_times_from_preset(data).slice_layers(cfg.model.num_layers)
    else:
        pt = get_profile_times(cfg.model.name).slice_layers(cfg.model.num_layers)
    if cfg.model.recompute:
        pt = pt.with_full_recompute()
    return pt


def _load_run_inputs(config: str | None, model: str, schedule: str | None):
    if config:
        cfg = _load_config_with_profiled_defaults(config)
    else:
        cfg = get_preset(model)
        cfg.profiled_data = True
    if schedule is not None:
        cfg.schedule = schedule
    return cfg, _profile_times_for_config(cfg)


def _build_executor(cfg: SimConfig, profile):
    return build_simulation(
        cfg,
        layer_f_times=profile.layer_f if profile else None,
        layer_b_times=profile.layer_b if profile else None,
        layer_w_times=profile.layer_w if profile else None,
        embedding_f_time=profile.embedding_f if profile else None,
        embedding_b_time=profile.embedding_b if profile else None,
        embedding_w_time=profile.embedding_w if profile else None,
        head_f_time=profile.head_f if profile else None,
        head_b_time=profile.head_b if profile else None,
        head_w_time=profile.head_w if profile else None,
        partition_layers=cfg.partition_layers,
        placement=cfg.placement,
    )


@click.group()
def main() -> None:
    """SimPipe LLM training pipeline simulator."""


@main.command("run")
@click.option("--config", type=click.Path(exists=True), default=None)
@click.option("--model", default="nemotron-h-4B")
@click.option("--schedule", default=None, help="Override schedule from config (default: use config or 1f1b)")
@click.option("--output", type=click.Path(), default="./results")
@click.option(
    "--detailed-gantt",
    is_flag=True,
    default=False,
    help="Render detailed statistics and layout tables below the Gantt chart.",
)
def run_cmd(
    config: str | None,
    model: str,
    schedule: str | None,
    output: str,
    detailed_gantt: bool,
) -> None:
    cfg, profile = _load_run_inputs(config, model, schedule)

    executor = _build_executor(cfg, profile)
    if cfg.schedule == "octopipe" and cfg.tuning.auto_tune and cfg.partition_layers is None:
        click.echo(f"Auto-tuned partition: {executor.plan.partition.layer_counts(executor.graph)}")
        click.echo(f"Auto-tuned placement: {executor.plan.placement.device_stages}")
        accepted_trials = [trial for trial in executor.bubble_overlap_trials if trial.accepted]
        if accepted_trials:
            details = ", ".join(
                f"iter {trial.iteration}: {format_group(trial.group)}"
                for trial in accepted_trials
            )
            click.echo(f"Bubble-overlap tuned workloads: {details}")
        if executor.tune_top_results:
            click.echo(f"Top {len(executor.tune_top_results)} tuning candidates:")
            for row in executor.tune_top_results:
                per_device = row.comp_bubble.get("per_device", [])
                comps = [d["comp"] for d in per_device]
                bubbles = [d["bubble"] for d in per_device]
                dev_comp_var = (
                    sum((c - sum(comps) / len(comps)) ** 2 for c in comps) / len(comps)
                    if comps
                    else 0.0
                )
                dev_bubble_var = (
                    sum((b - sum(bubbles) / len(bubbles)) ** 2 for b in bubbles) / len(bubbles)
                    if bubbles
                    else 0.0
                )
                warmup = sum(d["warmup_bubble"] for d in per_device)
                cooldown = sum(d["cooldown_bubble"] for d in per_device)
                residual = sum(d["residual_bubble"] for d in per_device)
                click.echo(
                    f"  #{row.rank} chunk={row.chunk_num} makespan={row.makespan:.0f} "
                    f"stage_var={row.partition_variance:.2f} "
                    f"dev_comp_var={dev_comp_var:.0f} dev_bubble_var={dev_bubble_var:.0f} "
                    f"warmup={warmup:.0f} cooldown={cooldown:.0f} residual={residual:.0f}"
                )
    bo = executor.batch_order_result
    if bo is not None:
        if bo.is_identity:
            click.echo(
                f"Batch order: input order kept, makespan {bo.makespan:.0f} "
                f"({bo.trials} sims)"
            )
        else:
            click.echo(
                f"Batch order tuned: {bo.order} "
                f"(makespan {bo.baseline_makespan:.0f} -> {bo.makespan:.0f}, "
                f"{bo.trials} sims; slot k runs input microbatch order[k])"
            )
    result = executor.run()
    # dp replicas duplicate every block on the same device rows with mids
    # offset by dp_idx * nmb; show/export/analyze the first replica only.
    records = first_replica_records(result.records, cfg.parallel.micro_batch_num)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    sched_label = cfg.schedule
    partition_layers = executor.plan.partition.layer_counts(executor.graph)
    placement = executor.plan.placement.device_stages
    write_gantt_svg(
        records,
        out / "pipeline_gantt.svg",
        title=f"{cfg.model.name} {sched_label}",
        partition_layers=partition_layers,
        placement=placement,
        detailed=detailed_gantt,
    )
    (out / "detailed_info.md").write_text(
        format_gantt_detailed_info(
            records,
            partition_layers=partition_layers,
            placement=placement,
        )
    )
    write_pipeline_config(
        out / "pipeline_config.yaml",
        executor=executor,
        makespan=result.makespan,
        scheduling_records=records,
        memory=result.memory.to_dict() if result.memory else None,
    )
    stats = analyze_pipeline_comp_bubble(records, device_num=cfg.parallel.pp_size)
    click.echo(f"Makespan: {result.makespan}")
    click.echo(f"Bubble ratio: {stats.avg_bubble_ratio(cfg.parallel.pp_size):.2%}")
    if result.memory:
        click.echo(f"Peak memory: {result.memory.peak_bytes / 1024**3:.2f} GiB")
        for device in result.memory.per_device:
            status = "OK" if device.feasible else "OOM"
            click.echo(
                f"  D{device.did}: peak={device.peak_bytes / 1024**3:.2f} GiB "
                f"(model={device.model_state_bytes / 1024**3:.2f}, "
                f"master={device.master_parameter_bytes / 1024**3:.2f}, "
                f"moments={device.optimizer_moment_bytes / 1024**3:.2f}, "
                f"act={device.activation_peak_bytes / 1024**3:.2f}, "
                f"p2p={device.p2p_buffer_bytes / 1024**3:.2f}) {status}"
            )
    click.echo(f"Results written to {out}")


@main.command("web")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8080, show_default=True, type=int)
def web_cmd(host: str, port: int) -> None:
    """Serve the interactive web UI (edit config, run, export SVG/config)."""
    from simpipe.web.server import serve

    serve(host=host, port=port)


@main.command("sweep")
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path(), default="./sweep_results")
def sweep_cmd(config: str, output: str) -> None:
    with open(config) as f:
        data = yaml.safe_load(f)
    from simpipe.config.sim_config import SimConfig

    base = SimConfig.from_dict(data.get("base", data))
    grid = data.get("grid", {"pp_size": [2, 4], "micro_batch_num": [4, 8]})
    configs = sweep_configs(base, grid)
    rows = run_sweep(configs, Path(output))
    click.echo(f"Sweep complete: {len(rows)} configs, output={output}")


if __name__ == "__main__":
    main()
