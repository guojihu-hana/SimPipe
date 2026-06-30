from __future__ import annotations

from pathlib import Path

import click
import yaml

from simpipe.config.pipeline_config import write_pipeline_config
from simpipe.config.sim_config import SimConfig, load_config
from simpipe.core.executor import build_simulation
from simpipe.metrics.comp_bubble import analyze_pipeline_comp_bubble
from simpipe.models.registry import PRESETS, get_preset, get_profile_times, preset_model_data
from simpipe.tuning.bubble_overlap import format_group
from simpipe.tuning.sweep import run_sweep, sweep_configs
from simpipe.viz.gantt import write_gantt_svg


def _load_config_with_profiled_defaults(config: str) -> SimConfig:
    data = yaml.safe_load(Path(config).read_text()) or {}
    if data.get("profiled_data"):
        model_data = data.get("model") or {}
        model_name = model_data.get("name")
        if model_name in PRESETS:
            data = {
                **data,
                "model": {**preset_model_data(model_name), **model_data},
            }
            return SimConfig.from_dict(data)
    return load_config(config)


def _load_run_inputs(config: str | None, model: str, schedule: str | None):
    if config:
        cfg = _load_config_with_profiled_defaults(config)
    else:
        cfg = get_preset(model)
        cfg.profiled_data = True
    if schedule is not None:
        cfg.schedule = schedule
    if not cfg.profiled_data:
        return cfg, None
    return cfg, get_profile_times(cfg.model.name).slice_layers(cfg.model.num_layers)


@click.group()
def main() -> None:
    """SimPipe LLM training pipeline simulator."""


@main.command("run")
@click.option("--config", type=click.Path(exists=True), default=None)
@click.option("--model", default="nemotronh-4B")
@click.option("--schedule", default=None, help="Override schedule from config (default: use config or 1f1b)")
@click.option("--output", type=click.Path(), default="./results")
def run_cmd(config: str | None, model: str, schedule: str | None, output: str) -> None:
    cfg, profile = _load_run_inputs(config, model, schedule)

    executor = build_simulation(
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
    result = executor.run()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    sched_label = cfg.schedule
    write_gantt_svg(
        result.records,
        out / "pipeline_gantt.svg",
        title=f"{cfg.model.name} {sched_label}",
        partition_layers=executor.plan.partition.layer_counts(executor.graph),
        placement=executor.plan.placement.device_stages,
    )
    write_pipeline_config(
        out / "pipeline_config.yaml",
        executor=executor,
        makespan=result.makespan,
        scheduling_records=result.records,
    )
    stats = analyze_pipeline_comp_bubble(result.records, device_num=cfg.parallel.pp_size)
    click.echo(f"Makespan: {result.makespan}")
    click.echo(f"Bubble ratio: {stats.avg_bubble_ratio(cfg.parallel.pp_size):.2%}")
    click.echo(f"Results written to {out}")


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
