from simpipe.config.model import ModelConfig
from simpipe.graph.layer_template import MoELayerTemplate
from simpipe.graph.model_graph import ModelGraph


def test_moe_template_has_a2a():
    model = ModelConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        use_moe=True,
        num_experts=8,
        top_k=2,
    )
    graph = ModelGraph.from_config(model)
    op_types = {op.op_type.value for op in graph.operators}
    assert "comm_a2a" in op_types
    assert "router" in op_types
