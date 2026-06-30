from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from simpipe.models.pattern import stage_layer_pattern_strings
from simpipe.models.registry import stack_layer_symbols_for_model

if TYPE_CHECKING:
    from simpipe.core.executor import Executor


def serialize_scheduling_records(records: list[dict]) -> list[str]:
    sorted_records = sorted(
        records,
        key=lambda r: (
            r.get("start") or 0,
            r.get("did", 0),
            r.get("mid", 0),
            r.get("sid", 0),
        ),
    )
    return [
        f"({r['wtype'].lower()}, {r['mid']}, {r['sid']}, {r['did']}, {r['start']}, {r['end']})"
        for r in sorted_records
    ]


def _format_inline_list(values: list[int]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def _format_inline_nested_list(values: list[list[int]]) -> str:
    inner = ", ".join("[" + ", ".join(str(stage) for stage in stages) + "]" for stages in values)
    return f"[{inner}]"


def _format_stage_layers(stage_layers: list[str]) -> list[str]:
    return [f"{idx}: {pattern}" for idx, pattern in enumerate(stage_layers)]


def format_pipeline_config_yaml(
    *,
    schedule: str,
    partition: list[int],
    placement: list[list[int]],
    scheduling: list[str],
    chunk_num: int | None = None,
    makespan: float | None = None,
    stage_layers: list[str] | None = None,
) -> str:
    lines = [
        f"schedule: {schedule}",
        f"partition: {_format_inline_list(partition)}",
        f"placement: {_format_inline_nested_list(placement)}",
    ]
    if stage_layers is not None:
        lines.append("stage_layers:  # stage_idx: layer pattern (E=embedding, M=mamba, -=mlp, *=attn, L=head)")
        lines.extend(f'- "{entry}"' for entry in _format_stage_layers(stage_layers))
    lines.extend(
        [
            "scheduling:  # (workload_type, mid, sid, did, start_time, end_time)",
            *[f"- {entry}" for entry in scheduling],
        ]
    )
    if chunk_num is not None:
        lines.append(f"chunk_num: {chunk_num}")
    if makespan is not None:
        lines.append(f"makespan: {makespan}")
    return "\n".join(lines) + "\n"


def build_pipeline_config(
    *,
    partition: list[int],
    placement: list[list[int]],
    schedule: str,
    scheduling_records: list[dict],
    makespan: float | None = None,
    chunk_num: int | None = None,
    stage_layers: list[str] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schedule": schedule,
        "partition": partition,
        "placement": placement,
        "scheduling": serialize_scheduling_records(scheduling_records),
    }
    if stage_layers is not None:
        data["stage_layers"] = stage_layers
    if chunk_num is not None:
        data["chunk_num"] = chunk_num
    if makespan is not None:
        data["makespan"] = makespan
    return data


def build_pipeline_config_from_executor(
    executor: Executor,
    *,
    scheduling_records: list[dict],
    makespan: float | None = None,
) -> dict[str, Any]:
    partition = executor.plan.partition.layer_counts(executor.graph)
    stack_symbols = stack_layer_symbols_for_model(
        executor.config.model.name,
        executor.graph.model.num_layers,
    )
    stage_layers = (
        stage_layer_pattern_strings(partition, stack_symbols)
        if stack_symbols is not None
        else None
    )
    return build_pipeline_config(
        partition=partition,
        placement=executor.plan.placement.device_stages,
        schedule=executor.config.schedule,
        scheduling_records=scheduling_records,
        makespan=makespan,
        chunk_num=executor.config.parallel.chunk_num,
        stage_layers=stage_layers,
    )


def write_pipeline_config(
    output: Path,
    *,
    partition: list[int] | None = None,
    placement: list[list[int]] | None = None,
    schedule: str | None = None,
    scheduling_records: list[dict] | None = None,
    makespan: float | None = None,
    chunk_num: int | None = None,
    executor: Executor | None = None,
) -> None:
    if executor is not None:
        if scheduling_records is None:
            raise ValueError("scheduling_records is required when writing from executor")
        data = build_pipeline_config_from_executor(
            executor,
            scheduling_records=scheduling_records,
            makespan=makespan,
        )
    else:
        if partition is None or placement is None or schedule is None:
            raise ValueError("partition, placement, and schedule are required without executor")
        data = build_pipeline_config(
            partition=partition,
            placement=placement,
            schedule=schedule,
            scheduling_records=scheduling_records or [],
            makespan=makespan,
            chunk_num=chunk_num,
        )
    output.write_text(
        format_pipeline_config_yaml(
            schedule=data["schedule"],
            partition=data["partition"],
            placement=data["placement"],
            scheduling=data["scheduling"],
            chunk_num=data.get("chunk_num"),
            makespan=data.get("makespan"),
            stage_layers=data.get("stage_layers"),
        )
    )
