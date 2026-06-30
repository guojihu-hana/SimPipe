# SimPipe

LLM training pipeline parallelism simulator with operator-level graph IR, PP/TP/DP modeling, memory/ZeRO analysis, and MoE support.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```bash
simpipe run --config examples/nemotronh_4gpu.yaml --schedule 1f1b --output ./results/
simpipe sweep --config examples/sweep_3d_parallel.yaml --output ./sweep_results/
```

## Architecture

- `simpipe/graph/` — Operator-level compute graph IR (TensorSpec, Operator, ModelGraph)
- `simpipe/pipeline/` — Partition, placement, scheduling (PP planning)
- `simpipe/core/` — Discrete-event simulation runtime
- `simpipe/comm/` — TP/DP/PP collective communication models
- `simpipe/memory/` — Tensor liveness and ZeRO memory analysis
- `simpipe/tuning/` — Partition/placement search and parameter sweeps
- `simpipe/viz/` — Gantt charts and bubble analysis

## Tests

```bash
pytest
```
