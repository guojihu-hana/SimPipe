import json

import pytest

from simpipe.config.model import ModelConfig
from simpipe.config.hardware import HardwareConfig
from simpipe.config.parallel import ParallelConfig
from simpipe.config.sim_config import SimConfig
from simpipe.core.executor import build_simulation
from simpipe.graph.model_graph import ModelGraph
from simpipe.memory.estimate import (
    _cross_entropy_temp_buffer_bytes,
    _expert_data_parallel_size,
    estimate_pipeline_memory,
    model_parameter_spec,
)
from simpipe.memory.liveness import analyze_liveness, check_memory_feasible


def test_liveness_peak_positive():
    model = ModelConfig(num_layers=2, hidden_size=128, num_attention_heads=4, seq_len=256)
    graph = ModelGraph.from_config(model)
    peak, breakdown = analyze_liveness(graph)
    assert peak >= 0
    assert breakdown.total >= 0


def test_feasibility():
    assert check_memory_feasible(1000, 2000)
    assert not check_memory_feasible(3000, 2000)


def test_hf_config_loads_model_shape_and_parameter_spec():
    model = ModelConfig(
        hf_config_path="simpipe/models/hf_configs/DeepSeekV4Pro.json",
        seq_len=4096,
    )
    spec = model_parameter_spec(model)
    assert model.hidden_size == 7168
    assert model.num_layers == 61
    assert model.use_moe
    assert spec.embedding.total > 0
    assert spec.layers[0].expert > 0
    assert spec.expert_weight_dtype_bytes == 0.5


def test_hybrid_hf_pattern_uses_layer_specific_parameter_formulas():
    model = ModelConfig(
        name="nemotronh-nano-v2-9B",
        hf_config_path="simpipe/models/hf_configs/NemotronNanoV2-9B.json",
    )
    spec = model_parameter_spec(model)
    total_params = spec.embedding.total + spec.head.total + sum(layer.total for layer in spec.layers)

    assert len(spec.layers) == 56
    assert 8_000_000_000 < total_params < 10_000_000_000
    assert len({layer.total for layer in spec.layers}) > 1


def test_hybrid_hf_pattern_supports_transformer_and_moe_symbols(tmp_path):
    cfg_path = tmp_path / "hybrid.json"
    cfg_path.write_text(
        json.dumps(
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "intermediate_size": 256,
                "vocab_size": 128,
                "torch_dtype": "bfloat16",
                "hybrid_override_pattern": "T#",
                "n_routed_experts": 8,
                "n_shared_experts": 1,
            }
        )
    )

    model = ModelConfig(hf_config_path=str(cfg_path))
    spec = model_parameter_spec(model)

    assert len(spec.layers) == 2
    assert spec.layers[0].expert == 0
    assert spec.layers[0].non_expert > 0
    assert spec.layers[1].expert > 0
    assert spec.layers[1].non_expert > spec.layers[0].non_expert


def test_pipeline_memory_estimate_per_pp_rank_with_zero():
    cfg = SimConfig(
        model=ModelConfig(num_layers=4, hidden_size=128, num_attention_heads=4, seq_len=64),
        parallel=ParallelConfig(pp_size=2, dp_size=4, micro_batch_num=2, zero_stage=2),
        hardware=HardwareConfig(gpu_hbm_gb=1),
        schedule="1f1b",
    )
    f = [5.0] * 4
    executor = build_simulation(cfg, layer_f_times=f, layer_b_times=f)
    result = executor.run(time_limit=100000)
    memory = result.memory or estimate_pipeline_memory(
        executor.graph,
        executor.plan,
        cfg.parallel,
        cfg.hardware,
        result.records,
    )

    assert len(memory.per_device) == 2
    assert memory.zero_stage == 2
    assert memory.total_parameter_count > 0
    assert all(device.parameter_bytes > 0 for device in memory.per_device)
    assert all(device.gradient_bytes * cfg.parallel.dp_size == 2 * device.parameter_bytes for device in memory.per_device)
    assert all(device.master_parameter_bytes > 0 for device in memory.per_device)
    assert all(device.optimizer_moment_bytes > 0 for device in memory.per_device)
    assert all(
        device.optimizer_bytes == device.master_parameter_bytes + device.optimizer_moment_bytes
        for device in memory.per_device
    )
    assert all(device.activation_peak_bytes > 0 for device in memory.per_device)
    assert all(device.peak_bytes >= device.model_state_bytes for device in memory.per_device)


def test_gradient_bytes_default_to_fp32_reduce_buffer():
    cfg = SimConfig(
        model=ModelConfig(num_layers=2, hidden_size=64, num_attention_heads=4, seq_len=32),
        parallel=ParallelConfig(pp_size=1, dp_size=4, micro_batch_num=1, zero_stage=0),
        hardware=HardwareConfig(gpu_hbm_gb=1),
        schedule="1f1b",
    )
    executor = build_simulation(cfg, layer_f_times=[1.0, 1.0], layer_b_times=[1.0, 1.0])
    memory = executor.run(time_limit=100000).memory

    assert memory is not None
    device = memory.per_device[0]
    assert device.gradient_bytes == 2 * device.parameter_bytes


def test_gradient_bytes_match_parameter_dtype_when_not_reduced_in_fp32():
    cfg = SimConfig(
        model=ModelConfig(num_layers=2, hidden_size=64, num_attention_heads=4, seq_len=32),
        parallel=ParallelConfig(
            pp_size=1,
            dp_size=4,
            micro_batch_num=1,
            zero_stage=0,
            grad_reduce_in_fp32=False,
        ),
        hardware=HardwareConfig(gpu_hbm_gb=1),
        schedule="1f1b",
    )
    executor = build_simulation(cfg, layer_f_times=[1.0, 1.0], layer_b_times=[1.0, 1.0])
    memory = executor.run(time_limit=100000).memory

    assert memory is not None
    device = memory.per_device[0]
    assert device.gradient_bytes == device.parameter_bytes


def test_flash_attention_reduces_attention_activation_estimate():
    base = dict(num_layers=2, hidden_size=128, num_attention_heads=4, seq_len=64)
    cfg_flash = SimConfig(
        model=ModelConfig(**base),
        parallel=ParallelConfig(pp_size=1, micro_batch_num=1),
        hardware=HardwareConfig(gpu_hbm_gb=1),
        schedule="1f1b",
    )
    cfg_no_flash = SimConfig(
        model=ModelConfig(**base, flash_attention=False),
        parallel=ParallelConfig(pp_size=1, micro_batch_num=1),
        hardware=HardwareConfig(gpu_hbm_gb=1),
        schedule="1f1b",
    )

    flash = build_simulation(cfg_flash, layer_f_times=[1.0, 1.0], layer_b_times=[1.0, 1.0])
    no_flash = build_simulation(cfg_no_flash, layer_f_times=[1.0, 1.0], layer_b_times=[1.0, 1.0])

    flash_memory = estimate_pipeline_memory(
        flash.graph,
        flash.plan,
        cfg_flash.parallel,
        cfg_flash.hardware,
    )
    no_flash_memory = estimate_pipeline_memory(
        no_flash.graph,
        no_flash.plan,
        cfg_no_flash.parallel,
        cfg_no_flash.hardware,
    )

    assert cfg_flash.model.flash_attention
    assert flash_memory.per_device[0].activation_peak_bytes < no_flash_memory.per_device[0].activation_peak_bytes


def test_cross_entropy_temp_buffer_formula_matches_vocab_logits_fp32():
    model = ModelConfig(
        micro_batch_size=1,
        seq_len=4096,
        vocab_size=131072,
    )

    assert _cross_entropy_temp_buffer_bytes(model, ParallelConfig(tp_size=1)) == 2 * 1024**3


def test_ep_shards_only_expert_model_state():
    base_model = ModelConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        seq_len=64,
        use_moe=True,
        num_experts=8,
        top_k=2,
    )
    cfg_ep1 = SimConfig(
        model=base_model,
        parallel=ParallelConfig(pp_size=1, dp_size=4, ep_size=1, zero_stage=0, micro_batch_num=1),
        hardware=HardwareConfig(gpu_hbm_gb=1),
        schedule="1f1b",
    )
    cfg_ep2 = SimConfig(
        model=base_model,
        parallel=ParallelConfig(pp_size=1, dp_size=4, ep_size=2, zero_stage=0, micro_batch_num=1),
        hardware=HardwareConfig(gpu_hbm_gb=1),
        schedule="1f1b",
    )

    ep1 = build_simulation(cfg_ep1).run(time_limit=100000).memory
    ep2 = build_simulation(cfg_ep2).run(time_limit=100000).memory

    assert ep1 is not None
    assert ep2 is not None
    assert ep2.per_device[0].parameter_bytes < ep1.per_device[0].parameter_bytes
    assert ep2.per_device[0].parameter_bytes > ep1.per_device[0].parameter_bytes / 2


def test_ep_requires_even_dp_group_split():
    assert _expert_data_parallel_size(ParallelConfig(dp_size=8, ep_size=4)) == 2
    with pytest.raises(ValueError):
        _expert_data_parallel_size(ParallelConfig(dp_size=3, ep_size=2))
