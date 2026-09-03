from simpipe.config.hardware import HardwareConfig
from simpipe.config.model import ModelConfig
from simpipe.config.parallel import ParallelConfig
from simpipe.config.sim_config import SimConfig
from simpipe.core.executor import build_simulation


def test_unprofiled_stage_timing_uses_equal_f_b_w():
    cfg = SimConfig(
        model=ModelConfig(num_layers=4, hidden_size=256, num_attention_heads=4, seq_len=512),
        parallel=ParallelConfig(pp_size=2, micro_batch_num=2, bwd_split=True),
        hardware=HardwareConfig(),
        schedule="octopipe",
    )

    executor = build_simulation(cfg)

    for timing in executor.plan.stage_timings:
        assert timing.f_time == timing.b_time == timing.w_time


def test_yaml_configs_do_not_use_profiled_data_by_default(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
model:
  name: nemotron-h-4B
  num_layers: 4
  hidden_size: 256
  num_attention_heads: 4
  seq_len: 512
parallel:
  pp_size: 2
  micro_batch_num: 2
  bwd_split: true
schedule: octopipe
"""
    )

    from simpipe.cli import _load_run_inputs

    cfg, profile = _load_run_inputs(str(cfg_path), "nemotron-h-4B", None)

    assert cfg.profiled_data is False
    assert profile is None


def test_profiled_yaml_uses_preset_model_defaults(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
profiled_data: true
model:
  name: test_model
parallel:
  pp_size: 4
  micro_batch_num: 8
  bwd_split: true
schedule: octopipe
"""
    )

    from simpipe.cli import _load_run_inputs

    cfg, profile = _load_run_inputs(str(cfg_path), "nemotron-h-4B", None)

    assert cfg.model.name == "test_model"
    assert cfg.model.num_layers == 48
    assert profile is not None
    assert len(profile.layer_f) == len(profile.layer_b) == len(profile.layer_w) == 48
    assert profile.embedding_f == 5.0
    assert profile.head_f == 30.0


def test_tuning_config_parses_bubble_overlap_options(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
model:
  name: test_model
parallel:
  pp_size: 4
schedule: octopipe
tuning:
  auto_tune: true
  bubble_overlap_tune: true
  bubble_overlap_max_iter: 3
  bubble_overlap_group_by: mid_sid_type
"""
    )

    from simpipe.config.sim_config import load_config

    cfg = load_config(cfg_path)

    assert cfg.tuning.bubble_overlap_tune is True
    assert cfg.tuning.bubble_overlap_max_iter == 3
    assert cfg.tuning.bubble_overlap_group_by == "mid_sid_type"


def test_tuning_config_parses_partition_and_result_top_k(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
tuning:
  partition_top_k: 6
  result_top_k: 9
"""
    )
    from simpipe.config.sim_config import load_config

    cfg = load_config(cfg_path)
    assert cfg.tuning.partition_top_k == 6
    assert cfg.tuning.result_top_k == 9
