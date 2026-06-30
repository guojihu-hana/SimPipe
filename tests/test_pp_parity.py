from simpipe.config.hardware import HardwareConfig
from simpipe.config.model import ModelConfig
from simpipe.config.parallel import ParallelConfig
from simpipe.config.sim_config import SimConfig
from simpipe.core.executor import build_simulation
from simpipe.memory.zero import zero_model_state_bytes, zero_sharded_bytes


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


def test_zero_model_state_bytes_by_stage():
    base = dict(
        parameter_bytes=100,
        gradient_bytes=200,
        master_parameter_bytes=400,
        optimizer_bytes=800,
    )

    z0 = zero_model_state_bytes(**base, parallel=ParallelConfig(dp_size=4, zero_stage=0))
    assert (
        z0.parameter_bytes,
        z0.gradient_bytes,
        z0.master_parameter_bytes,
        z0.optimizer_bytes,
    ) == (100, 200, 400, 800)

    z1 = zero_model_state_bytes(**base, parallel=ParallelConfig(dp_size=4, zero_stage=1))
    assert (
        z1.parameter_bytes,
        z1.gradient_bytes,
        z1.master_parameter_bytes,
        z1.optimizer_bytes,
    ) == (100, 200, 100, 200)

    z2 = zero_model_state_bytes(**base, parallel=ParallelConfig(dp_size=4, zero_stage=2))
    assert (
        z2.parameter_bytes,
        z2.gradient_bytes,
        z2.master_parameter_bytes,
        z2.optimizer_bytes,
    ) == (100, 50, 100, 200)

    z3 = zero_model_state_bytes(**base, parallel=ParallelConfig(dp_size=4, zero_stage=3))
    assert (
        z3.parameter_bytes,
        z3.gradient_bytes,
        z3.master_parameter_bytes,
        z3.optimizer_bytes,
    ) == (25, 50, 100, 200)
