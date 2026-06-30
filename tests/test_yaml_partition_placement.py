import yaml

from simpipe.config.sim_config import load_config
from simpipe.core.executor import build_simulation
from simpipe.models.registry import get_layer_times


def test_yaml_partition_and_placement(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "model": {"num_layers": 8, "hidden_size": 256, "num_attention_heads": 4},
                "parallel": {"pp_size": 4, "micro_batch_num": 4},
                "schedule": "1f1b",
                "partition_layers": [2, 2, 2, 2],
                "placement": [[0], [1], [2], [3]],
            }
        )
    )
    cfg = load_config(cfg_path)
    f = [5.0] * 8
    ex = build_simulation(cfg, f, f, None, partition_layers=cfg.partition_layers, placement=cfg.placement)
    assert ex.plan.stage_num == 4
    assert ex.plan.placement.device_stages == [[0], [1], [2], [3]]
