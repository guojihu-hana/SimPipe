from simpipe.config.model import ModelConfig
from simpipe.graph.model_graph import ModelGraph
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
