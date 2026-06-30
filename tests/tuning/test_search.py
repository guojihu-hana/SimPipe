from simpipe.tuning.fast_est import fast_estimate_makespan, search_placements


def test_fast_est_increases_with_mb():
    stage_f = [10.0, 10.0, 10.0, 10.0]
    stage_b = [10.0, 10.0, 10.0, 10.0]
    stage_w = [5.0, 5.0, 5.0, 5.0]
    placement = [[0], [1], [2], [3]]
    t4 = fast_estimate_makespan(stage_f, stage_b, stage_w, 4, placement)
    t8 = fast_estimate_makespan(stage_f, stage_b, stage_w, 8, placement)
    assert t8 >= t4


def test_fast_est_includes_weight_time():
    stage_f = [1.0, 1.0]
    stage_b = [1.0, 1.0]
    stage_w = [20.0, 0.0]
    placement = [[0], [1]]

    assert fast_estimate_makespan(stage_f, stage_b, stage_w, 1, placement) == 22.0


def test_search_returns_ranked():
    stage_f = [10.0] * 4
    stage_b = [10.0] * 4
    stage_w = [5.0] * 4
    results = search_placements(
        stage_f,
        stage_b,
        stage_w,
        num_mb=8,
        device_num=4,
        evaluate_fn=lambda p: fast_estimate_makespan(stage_f, stage_b, stage_w, 8, p),
        top_k=3,
    )
    assert len(results) <= 3
    assert results[0][1] <= results[-1][1]
