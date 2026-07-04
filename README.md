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

## OctoPipe Auto-Tune

OctoPipe can search pipeline partitions and stage placements automatically. Use `schedule: octopipe` and omit `partition_layers` / `placement`, or set `tuning.auto_tune: true`.

```bash
simpipe run --config examples/octopipe_auto_tune.yaml --output ./results/octopipe/
```

The run writes:

- `pipeline_gantt.svg` — per-rank schedule visualization.
- `detailed_info.md` — detailed Gantt time statistics and layout tables.
- `pipeline_config.yaml` — selected partition, placement, schedule records, stage layer pattern, makespan, and memory estimate.

By default, `pipeline_gantt.svg` contains only the schedule plot. Pass `--detailed-gantt` to also render the detailed tables below the chart.

### Example Config

```yaml
profiled_data: true
time_limit: 20000000

model:
  name: nemotron-nano-v2-9B
  hf_config_path: simpipe/models/hf_configs/NemotronNanoV2-9B.json
  seq_len: 4096
  micro_batch_size: 1
  flash_attention: true

parallel:
  pp_size: 4
  tp_size: 1
  dp_size: 1
  ep_size: 1
  micro_batch_num: 8
  chunk_num: null
  bwd_split: true
  zero_stage: 1
  grad_reduce_in_fp32: true

schedule: octopipe

tuning:
  auto_tune: true
  sim_k: 32
  beam_width: 32
  partition_top_k: 32
  result_top_k: 10
  bubble_overlap_tune: true
  bubble_overlap_max_iter: 4
  bubble_overlap_group_by: mid_type

hardware:
  gpu_peak_tflops: 312.0
  gpu_hbm_gb: 80.0
  comm_alpha_us: 0
```

### Model And Profile Data

Model shape is configured under `model:`. The fields are defined in `simpipe/config/model.py`.

- `name`: model preset name. Built-in presets live in `simpipe/models/registry.py`.
- `hf_config_path`: optional HuggingFace-style JSON config. This fills model shape fields such as `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `vocab_size`, MoE expert counts, and hybrid layer pattern when present.
- `seq_len`, `micro_batch_size`: training shape used for activation and memory estimates.
- `flash_attention`: defaults to `true`; reduces attention saved-activation estimates.

Timing profile data is selected by `profiled_data: true` plus `model.name`.

- Preset layer timings are in `simpipe/models/registry.py`.
- Hybrid model layer patterns use symbols `M` = Mamba, `-` = MLP, `*` = Attention, `T` = dense Transformer layer, and `#` = MoE layer. Quote YAML pattern strings that contain `#`.
- For HF configs with `hybrid_override_pattern`, memory parameter estimates use layer-specific formulas by pattern. Runtime timing still comes from the preset profile keyed by `model.name`.
- To add a new profiled model, add a new entry to `PRESETS` in `simpipe/models/registry.py` with either:
  - `pattern`, `forward_ms`, `backward_ms`, and optional `weight_ms`, or
  - explicit `layer_f_times`, `layer_b_times`, `layer_w_times`.

### Parallel And Training Parameters

Parallel/training fields are under `parallel:` and defined in `simpipe/config/parallel.py`.

- `pp_size`: number of physical pipeline ranks/devices.
- `tp_size`: tensor parallel size. Dense parameters and activations are divided by TP where applicable.
- `dp_size`: data parallel size.
- `ep_size`: expert parallel size. EP only shards expert parameters/states; dense parameters are not divided by EP. `dp_size` must be divisible by `ep_size`.
- `micro_batch_num`: number of pipeline microbatches.
- `chunk_num`: virtual pipeline chunks per physical PP rank. `null` lets OctoPipe search legal chunk counts.
- `bwd_split`: if `true`, backward and weight update workloads are separate.
- `zero_stage`: ZeRO stage. Default is `1`.
- `grad_reduce_in_fp32`: default `true`; gradient buffer memory is FP32. If `false`, gradient memory follows model parameter dtype.

For BF16/FP16 parameters with `grad_reduce_in_fp32: true`, model-state memory is estimated as:

```text
weights:         2X
grad buffer:     4X
fp32 master:     4X
Adam moments:    8X
total:          18X bytes
```

where `X` is the local parameter count after PP/TP/EP sharding and ZeRO rules.

### OctoPipe Tuning Parameters

Tuning fields are under `tuning:` and defined in `simpipe/config/tuning.py`.

- `auto_tune`: enable OctoPipe partition/placement search.
- `partition_top_k`: number of layer partitions retained from the partition-balance search.
- `beam_width`: number of placement candidates generated per chunk/partition.
- `sim_k`: number of top fast-estimated candidates to fully simulate.
- `result_top_k`: number of top candidates printed and stored.
- `bubble_overlap_tune`: enable workload-level bubble-overlap exemption tuning.
- `bubble_overlap_max_iter`: max iterations for bubble-overlap tuning.
- `bubble_overlap_group_by`: grouping key for exemptions: `mid`, `mid_type`, or `mid_sid_type`.

OctoPipe tuning is implemented in `simpipe/tuning/octopipe_tune.py`. Partition search is in `simpipe/tuning/partition_search.py`; placement scoring is in `simpipe/tuning/fast_est.py`.

### Manual Partition And Placement

To disable auto-tune and run a fixed plan, set both:

```yaml
partition_layers: [2, 2, 2, 2]
placement: [[0], [1], [2], [3]]
tuning:
  auto_tune: false
```

For multi-chunk plans, `partition_layers` has one entry per virtual stage, and `placement` maps each physical device to ordered virtual stage ids:

```yaml
parallel:
  pp_size: 4
  chunk_num: 2

partition_layers: [7, 7, 7, 7, 7, 7, 7, 7]
placement: [[0, 4], [1, 5], [2, 6], [3, 7]]
```

### Memory Estimation

Memory is estimated after simulation in `simpipe/memory/estimate.py` and stored in `result.memory` plus `pipeline_config.yaml`.

The estimate includes:

- local model weights
- gradient buffer, controlled by `grad_reduce_in_fp32`
- FP32 master weights
- Adam first/second moments
- activation peak from schedule records
- P2P buffers
- FP32 tensor-parallel cross-entropy temporary logits buffer on the head stage

The CLI prints per-rank peak memory and OOM/OK against `hardware.gpu_hbm_gb`.

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
