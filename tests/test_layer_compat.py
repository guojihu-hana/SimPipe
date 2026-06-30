from simpipe.config.model import ModelConfig
from simpipe.graph.model_graph import ModelGraph
from simpipe.pipeline.partition import layer_partition_to_stage_specs


def test_layer_compat_matches_op_count():
    model = ModelConfig(num_layers=8, hidden_size=256, num_attention_heads=4)
    graph = ModelGraph.from_config(model)
    layer_counts = [4, 4]
    part = layer_partition_to_stage_specs(graph, layer_counts)
    total_ops = sum(len(s.operator_ids) for s in part.stages)
    assert total_ops == len(graph.operators)
