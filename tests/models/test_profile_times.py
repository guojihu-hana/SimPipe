from simpipe.config.model import ModelConfig
from simpipe.graph.model_graph import EMBEDDING_LAYER_IDX, ModelGraph, head_layer_idx
from simpipe.models.registry import get_profile_times
from simpipe.pipeline.partition import layer_partition_to_stage_specs
import pytest


def test_model_graph_includes_embedding_and_head():
    graph = ModelGraph.from_config(ModelConfig(num_layers=4, hidden_size=256, num_attention_heads=4))

    op_types = {op.op_type.value for op in graph.operators}
    layer_indices = {op.layer_idx for op in graph.operators if op.layer_idx is not None}

    assert "embedding" in op_types
    assert "head" in op_types
    assert EMBEDDING_LAYER_IDX in layer_indices
    assert head_layer_idx(4) in layer_indices


def test_distribute_profile_times_applies_embedding_and_head():
    graph = ModelGraph.from_config(ModelConfig(num_layers=2, hidden_size=128, num_attention_heads=4))
    graph.distribute_profile_times(
        [10.0, 12.0],
        [15.0, 18.0],
        [5.0, 6.0],
        embedding_f_time=20.0,
        embedding_b_time=25.0,
        embedding_w_time=8.0,
        head_f_time=30.0,
        head_b_time=35.0,
        head_w_time=10.0,
    )

    embedding = graph.op_by_id("embedding")
    head = graph.op_by_id("head")

    assert embedding.profiled_time_us == 20.0
    assert embedding.profiled_bwd_time_us == 25.0
    assert embedding.profiled_w_time_us == 8.0
    assert head.profiled_time_us == 30.0
    assert head.profiled_bwd_time_us == 35.0
    assert head.profiled_w_time_us == 10.0


def test_partition_places_embedding_on_first_stage_and_head_on_last():
    graph = ModelGraph.from_config(ModelConfig(num_layers=4, hidden_size=256, num_attention_heads=4))
    partition = layer_partition_to_stage_specs(graph, [2, 2])

    assert partition.stages[0].operator_ids[0] == "embedding"
    assert partition.stages[-1].operator_ids[-1] == "head"
    partition.validate(graph)


def test_preset_exposes_embedding_and_head_profile_times():
    profile = get_profile_times("test_model")

    assert profile.embedding_f == 5.0
    assert profile.embedding_b == 4.0
    assert profile.embedding_w == 3.0
    assert profile.head_f == 30.0
    assert profile.head_b == 35.0
    assert profile.head_w == 10.0


def test_unprofiled_preset_uses_uniform_fallback_with_warning():
    with pytest.warns(RuntimeWarning, match="No profiled layer times or pattern found"):
        profile = get_profile_times("gpt-13B")

    assert len(profile.layer_f) == 40
    assert profile.layer_f == [1.0] * 40
    assert profile.layer_b == [2.0] * 40
    assert profile.layer_w == [0.0] * 40
    assert profile.embedding_f == 0.0
    assert profile.embedding_b == 0.0
    assert profile.embedding_w == 0.0
    assert profile.head_f == 0.0
    assert profile.head_b == 0.0
    assert profile.head_w == 0.0
