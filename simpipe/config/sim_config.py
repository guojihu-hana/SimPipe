from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from simpipe.config.batch import BatchConfig
from simpipe.config.hardware import HardwareConfig
from simpipe.config.model import ModelConfig
from simpipe.config.parallel import ParallelConfig
from simpipe.config.tuning import TuningConfig


@dataclass
class SimConfig:
    model: ModelConfig
    parallel: ParallelConfig
    hardware: HardwareConfig
    schedule: str = "1f1b"
    profiled_data: bool = False
    time_limit: int = 1_000_000
    discrete_event_time: bool = True
    # Layer counts per stage (sum must equal model.num_layers).
    partition_layers: list[int] | None = None
    # Device -> ordered stage ids on that device (length must equal parallel.pp_size).
    placement: list[list[int]] | None = None
    tuning: TuningConfig = field(default_factory=TuningConfig)
    # Variable-length microbatch spec; when set it defines micro_batch_num.
    batch: BatchConfig | None = None

    @classmethod
    def from_dict(cls, data: dict) -> SimConfig:
        schedule = data.get("schedule", "1f1b")
        has_pp = data.get("partition_layers") is not None or data.get("placement") is not None
        tuning_data = data.get("tuning") or {}
        # OctoPipe without explicit partition/placement: auto-search by default
        if "auto_tune" not in tuning_data and schedule == "octopipe" and not has_pp:
            tuning_data = {**tuning_data, "auto_tune": True}
        parallel_data = dict(data.get("parallel", {}))
        batch = BatchConfig.from_dict(data["batch"]) if data.get("batch") else None
        if batch is not None:
            explicit_mb = parallel_data.get("micro_batch_num")
            if explicit_mb is not None and int(explicit_mb) != batch.num_microbatches:
                raise ValueError(
                    f"parallel.micro_batch_num={explicit_mb} does not match "
                    f"batch.microbatches length {batch.num_microbatches}"
                )
            parallel_data["micro_batch_num"] = batch.num_microbatches
        return cls(
            model=ModelConfig(**data.get("model", {})),
            parallel=ParallelConfig(**parallel_data),
            hardware=HardwareConfig(**data.get("hardware", {})),
            schedule=schedule,
            profiled_data=bool(data.get("profiled_data", False)),
            time_limit=data.get("time_limit", 1_000_000),
            discrete_event_time=data.get("discrete_event_time", True),
            partition_layers=data.get("partition_layers"),
            placement=data.get("placement"),
            tuning=TuningConfig.from_dict(tuning_data),
            batch=batch,
        )


def load_config(path: str | Path) -> SimConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return SimConfig.from_dict(data)
