from __future__ import annotations

import csv
import itertools
from pathlib import Path

from simpipe.config.sim_config import SimConfig
from simpipe.core.executor import build_simulation
from simpipe.core.types import parse_schedule


def sweep_configs(base: SimConfig, grid: dict) -> list[SimConfig]:
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    configs: list[SimConfig] = []
    for combo in itertools.product(*values):
        import copy

        cfg = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            if k == "pp_size":
                cfg.parallel.pp_size = v
            elif k == "tp_size":
                cfg.parallel.tp_size = v
            elif k == "dp_size":
                cfg.parallel.dp_size = v
            elif k == "micro_batch_num":
                cfg.parallel.micro_batch_num = v
            elif k == "schedule":
                cfg.schedule = v
        configs.append(cfg)
    return configs


def run_sweep(configs: list[SimConfig], output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, cfg in enumerate(configs):
        ex = build_simulation(cfg)
        result = ex.run()
        row = {
            "id": i,
            "pp": cfg.parallel.pp_size,
            "tp": cfg.parallel.tp_size,
            "dp": cfg.parallel.dp_size,
            "micro_batch_num": cfg.parallel.micro_batch_num,
            "schedule": cfg.schedule,
            "makespan": result.makespan,
        }
        rows.append(row)
    csv_path = output / "sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows
