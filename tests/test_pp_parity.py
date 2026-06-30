from simpipe.config.hardware import HardwareConfig
from simpipe.config.model import ModelConfig
from simpipe.config.parallel import ParallelConfig
from simpipe.config.sim_config import SimConfig
from simpipe.core.executor import build_simulation
from simpipe.memory.zero import zero_sharded_bytes


def test_simulation_runs():
    cfg = SimConfig(
        model=ModelConfig(num_layers=8, hidden_size=256, num_attention_heads=4, seq_len=512),
        parallel=ParallelConfig(pp_size=4, micro_batch_num=4),
        hardware=HardwareConfig(),
        schedule="1f1b",
    )
    f = [5.0] * 8
    ex = build_simulation(cfg, layer_f_times=f, layer_b_times=f)
    result = ex.run(time_limit=100000)
    assert result.makespan > 0
    assert len(result.records) > 0


def test_zero_sharding():
    cfg = ParallelConfig(tp_size=2, pp_size=4, dp_size=2, zero_stage=1)
    assert zero_sharded_bytes(8000, cfg) < 8000
