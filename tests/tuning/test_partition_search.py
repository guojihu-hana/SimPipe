from simpipe.tuning.fast_est import (
    generate_octopipe_placement_candidates,
    interleaved_placement,
    placement_proxy_score,
)
from simpipe.tuning.partition_search import (
    best_partition_by_stage_variance,
    legal_chunk_values,
    partition_variance,
    stage_times_for_partition,
    top_partitions_by_stage_variance,
)


def test_legal_chunk_values_require_stage_num_le_num_layers():
    assert legal_chunk_values(48, 4, None) == list(range(12, 0, -1))
    assert legal_chunk_values(48, 4, 2) == [2]
    assert legal_chunk_values(10, 4, None) == [2, 1]


def test_top_partitions_returns_at_least_one_valid_partition():
    layer_f = [10.0, 20.0, 30.0, 40.0]
    layer_b = [5.0, 5.0, 5.0, 5.0]
    layer_w = [1.0, 1.0, 1.0, 1.0]

    results = top_partitions_by_stage_variance(layer_f, layer_b, layer_w, 4, 2, top_k=2)

    assert len(results) >= 1
    variance, partition = results[0]
    assert len(partition) == 2
    assert sum(partition) == 4
    assert all(count >= 1 for count in partition)
    stage_times = stage_times_for_partition(layer_f, layer_b, layer_w, partition)
    assert variance == partition_variance(stage_times)


def test_generate_octopipe_placement_includes_interleaved():
    stage_num = 8
    device_num = 4
    candidates = generate_octopipe_placement_candidates(device_num, stage_num, beam_width=4)
    interleaved = interleaved_placement(device_num, stage_num)

    assert candidates[0] == interleaved
    assert interleaved in candidates


def test_balanced_placement_can_use_unequal_stage_counts():
    device_num = 4
    stage_num = 8
    stage_times = [10.0, 10.0, 10.0, 10.0, 100.0, 10.0, 10.0, 10.0]
    base = interleaved_placement(device_num, stage_num)
    candidates = generate_octopipe_placement_candidates(
        device_num,
        stage_num,
        beam_width=32,
        stage_times=stage_times,
    )

    assert candidates[0] == base
    assert len(candidates) > 1
    base_score = placement_proxy_score(base, stage_times)
    best_score = min(placement_proxy_score(p, stage_times) for p in candidates)
    assert best_score <= base_score
    assert any(len(row) != len(base[i]) for p in candidates for i, row in enumerate(p))


def test_max_chunk_layerwise_partition_is_returned():
    layer_f = [float(i + 1) for i in range(8)]
    layer_b = [1.0] * 8
    layer_w = [1.0] * 8
    results = top_partitions_by_stage_variance(layer_f, layer_b, layer_w, 8, 8, top_k=1)
    assert results
    _, partition = results[0]
    assert partition == [1] * 8


def test_best_partition_by_stage_variance_balances_uneven_layers():
    layer_f = [10.0, 10.0, 10.0, 100.0]
    layer_b = [5.0, 5.0, 5.0, 50.0]
    layer_w = [1.0, 1.0, 1.0, 10.0]

    equal = [1, 1, 1, 1]
    optimized = best_partition_by_stage_variance(layer_f, layer_b, layer_w, 4, 2)

    assert sum(optimized) == 4
    assert optimized != equal
    assert partition_variance(stage_times_for_partition(layer_f, layer_b, layer_w, optimized)) <= (
        partition_variance(stage_times_for_partition(layer_f, layer_b, layer_w, equal))
    )
