import pytest

from simpipe.config.model import ModelConfig
from simpipe.graph.model_graph import ModelGraph
from simpipe.pipeline.partition import (
    OperatorPartition,
    StageSpec,
    layer_partition_to_stage_specs,
    split_layer_at_fraction,
)


@pytest.fixture
def small_graph():
    model = ModelConfig(hidden_size=256, num_layers=4, num_attention_heads=4, seq_len=512)
    return ModelGraph.from_config(model)


def test_layer_partition_covers_all_ops(small_graph):
    partition = layer_partition_to_stage_specs(small_graph, [2, 2])
    partition.validate(small_graph)
    assert partition.num_stages == 2


def test_operator_split_across_layers(small_graph):
    first, second = split_layer_at_fraction(small_graph, 1, 0.5)
    assert first and second
    stages = [
        StageSpec(0, ["embedding"] + [op.id for op in small_graph.ops_for_layers([0])] + first),
        StageSpec(1, second + [op.id for op in small_graph.ops_for_layers([2, 3])] + ["head"]),
    ]
    part = OperatorPartition(stages)
    part.validate(small_graph)


def test_p2p_boundaries(small_graph):
    partition = layer_partition_to_stage_specs(small_graph, [2, 2])
    bounds = partition.p2p_boundaries
    assert len(bounds) == 1
    assert bounds[0][0] == 0 and bounds[0][1] == 1
