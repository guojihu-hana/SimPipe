from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from simpipe.models.pattern import stage_layer_pattern_strings
from simpipe.models.registry import stack_layer_symbols_for_model
from simpipe.memory.estimate import estimate_pipeline_memory

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
    memory: dict[str, Any] | None = None,
    batch_order: list[int] | None = None,
) -> str:
    lines = [
        f"schedule: {schedule}",
        f"partition: {_format_inline_list(partition)}",
        f"placement: {_format_inline_nested_list(placement)}",
    ]
    if batch_order is not None:
        lines.append(
            f"batch_order: {_format_inline_list(batch_order)}"
            "  # slot mid k runs input microbatch batch_order[k]"
        )
    if stage_layers is not None:
        lines.append(
            "stage_layers:  # stage_idx: layer pattern "
            "(E=embedding, M=mamba, -=mlp, *=attn, T=transformer, #=moe, L=head)"
        )
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
    if memory is not None:
        lines.extend(_format_memory_yaml(memory))
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
    memory: dict[str, Any] | None = None,
    batch_order: list[int] | None = None,
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
    if memory is not None:
        data["memory"] = memory
    if batch_order is not None:
        data["batch_order"] = batch_order
    return data


def build_pipeline_config_from_executor(
    executor: Executor,
    *,
    scheduling_records: list[dict],
    makespan: float | None = None,
    memory: dict[str, Any] | None = None,
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
    if memory is None:
        memory = estimate_pipeline_memory(
            executor.graph,
            executor.plan,
            executor.config.parallel,
            executor.config.hardware,
            scheduling_records,
        ).to_dict()
    return build_pipeline_config(
        partition=partition,
        placement=executor.plan.placement.device_stages,
        schedule=executor.config.schedule,
        scheduling_records=scheduling_records,
        makespan=makespan,
        chunk_num=executor.config.parallel.chunk_num,
        stage_layers=stage_layers,
        memory=memory,
        batch_order=executor.plan.mid_order,
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
    memory: dict[str, Any] | None = None,
) -> None:
    if executor is not None:
        if scheduling_records is None:
            raise ValueError("scheduling_records is required when writing from executor")
        data = build_pipeline_config_from_executor(
            executor,
            scheduling_records=scheduling_records,
            makespan=makespan,
            memory=memory,
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
            memory=memory,
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
            memory=data.get("memory"),
            batch_order=data.get("batch_order"),
        )
    )


def _format_memory_yaml(memory: dict[str, Any]) -> list[str]:
    lines = [
        "memory:",
        f"  zero_stage: {memory.get('zero_stage', 0)}",
        f"  peak_bytes: {memory.get('peak_bytes', 0)}",
        f"  peak_gb: {memory.get('peak_gb', 0)}",
        f"  feasible: {_format_scalar(memory.get('feasible', True))}",
        f"  total_parameter_count: {memory.get('total_parameter_count', 0)}",
        "  per_device:",
    ]
    for device in memory.get("per_device", []):
        lines.extend(
            [
                f"  - did: {device.get('did', 0)}",
                f"    pp_rank: {device.get('pp_rank', device.get('did', 0))}",
                f"    stage_ids: {_format_inline_list(device.get('stage_ids', []))}",
                f"    peak_bytes: {device.get('peak_bytes', 0)}",
                f"    peak_gb: {device.get('peak_gb', 0)}",
                f"    hbm_gb: {device.get('hbm_gb', 0)}",
                f"    feasible: {_format_scalar(device.get('feasible', True))}",
                f"    parameter_gb: {device.get('parameter_gb', 0)}",
                f"    gradient_gb: {device.get('gradient_gb', 0)}",
                f"    master_parameter_gb: {device.get('master_parameter_gb', 0)}",
                f"    optimizer_moment_gb: {device.get('optimizer_moment_gb', 0)}",
                f"    optimizer_gb: {device.get('optimizer_gb', 0)}",
                f"    activation_peak_gb: {device.get('activation_peak_gb', 0)}",
                f"    p2p_buffer_gb: {device.get('p2p_buffer_gb', 0)}",
            ]
        )
    return lines


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
