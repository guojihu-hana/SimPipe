from __future__ import annotations

import random
from typing import Callable

from simpipe.pipeline.placement import Placement


def interleaved_placement(device_num: int, stage_num: int) -> list[list[int]]:
    """Round-robin stages over devices; handles stage_num % device_num != 0."""
    placement: list[list[int]] = [[] for _ in range(device_num)]
    for sid in range(stage_num):
        placement[sid % device_num].append(sid)
    return placement


def _placement_key(placement: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in placement)


def _stage_device_map(placement: list[list[int]], stage_num: int) -> list[int | None]:
    """sid -> device id lookup table (one O(S) pass instead of scanning the
    whole placement per stage, which made proxy scoring O(S^2 * D))."""
    owner: list[int | None] = [None] * stage_num
    for did, stages in enumerate(placement):
        for sid in stages:
            if 0 <= sid < stage_num:
                owner[sid] = did
    return owner


def _row_load(row: list[int], stage_times: list[float]) -> float:
    # summation order matches _device_loads exactly (row order)
    return sum(stage_times[sid] for sid in row)


def _device_loads(placement: list[list[int]], stage_times: list[float]) -> list[float]:
    return [sum(stage_times[sid] for sid in row) for row in placement]


def _moved_score(
    rows: list[list[int]],
    loads: list[float],
    owner: list[int | None],
    adj: int,
    stage_times: list[float],
    sid: int,
    from_d: int,
    to_d: int,
    to_row: list[int] | None = None,
) -> tuple[tuple[float, int, float], float, float, list[int]]:
    """Score of the placement after moving sid from from_d to to_d.

    Only the two affected rows are recomputed (in the same element order the
    full scorer uses), the untouched loads are reused verbatim, and the
    adjacent-pair count is updated by an integer delta, so the returned score
    is bit-identical to placement_proxy_score(_transfer_stage(...)).
    Returns (score, new_from_load, new_to_load, new_to_row).
    """
    new_from = _row_load([s for s in rows[from_d] if s != sid], stage_times)
    if to_row is None:
        to_row = sorted(rows[to_d] + [sid])
    new_to = _row_load(to_row, stage_times)
    trial = loads[:]
    trial[from_d] = new_from
    trial[to_d] = new_to
    new_adj = adj
    for nb in (sid - 1, sid + 1):
        if 0 <= nb < len(owner) and nb != sid:
            if owner[nb] == from_d:
                new_adj -= 1
            if owner[nb] == to_d:
                new_adj += 1
    score = (_load_variance(trial), new_adj, max(trial) if trial else 0.0)
    return score, new_from, new_to, to_row


def _load_variance(loads: list[float]) -> float:
    if not loads:
        return 0.0
    mean = sum(loads) / len(loads)
    return sum((load - mean) ** 2 for load in loads) / len(loads)


def _adjacent_same_device_count(placement: list[list[int]], stage_num: int) -> int:
    owner = _stage_device_map(placement, stage_num)
    return sum(
        1
        for sid in range(stage_num - 1)
        if owner[sid] is not None and owner[sid] == owner[sid + 1]
    )


def placement_proxy_score(
    placement: list[list[int]],
    stage_times: list[float],
) -> tuple[float, int, float]:
    """Lower is better: device comp variance, adjacent penalty, max device load."""
    loads = _device_loads(placement, stage_times)
    return (
        _load_variance(loads),
        _adjacent_same_device_count(placement, len(stage_times)),
        max(loads) if loads else 0.0,
    )


def _transfer_stage(
    placement: list[list[int]],
    stage_id: int,
    from_device: int,
    to_device: int,
) -> list[list[int]] | None:
    if from_device == to_device:
        return None
    variant = [row[:] for row in placement]
    if stage_id not in variant[from_device]:
        return None
    variant[from_device].remove(stage_id)
    variant[to_device].append(stage_id)
    for row in variant:
        row.sort()
    return variant


def _generate_adjacent_dispersal_variants(
    base: list[list[int]],
    device_num: int,
    stage_num: int,
    *,
    max_stage_per_device: int | None,
    add: Callable[[list[list[int]]], None],
    scorer: "_SeedScorer | None" = None,
) -> None:
    owner = _stage_device_map(base, stage_num)
    for sid in range(stage_num - 1):
        d0 = owner[sid]
        d1 = owner[sid + 1]
        if d0 is None or d1 is None or d0 != d1:
            continue
        for target in range(device_num):
            if target == d0:
                continue
            variant = _transfer_stage(base, sid + 1, d0, target)
            if variant is None:
                continue
            if max_stage_per_device is not None and any(
                len(row) > max_stage_per_device for row in variant
            ):
                continue
            if scorer is not None:
                scorer.record_move(variant, sid + 1, d0, target)
            add(variant)


def _generate_transfer_neighbors(
    seed_placements: list[list[list[int]]],
    device_num: int,
    add: Callable[[list[list[int]]], None],
    scorers: "list[_SeedScorer] | None" = None,
) -> None:
    for seed_idx, placement in enumerate(seed_placements):
        scorer = scorers[seed_idx] if scorers is not None else None
        for from_device, stages in enumerate(placement):
            for stage_id in stages:
                for target in range(device_num):
                    if target == from_device:
                        continue
                    variant = _transfer_stage(placement, stage_id, from_device, target)
                    if variant is not None:
                        if scorer is not None:
                            scorer.record_move(variant, stage_id, from_device, target)
                        add(variant)


class _SeedScorer:
    """Scores one-move neighbors of a fixed seed placement incrementally and
    collects them into a shared placement-key -> proxy-score table."""

    def __init__(
        self,
        seed: list[list[int]],
        stage_times: list[float],
        scores: dict[tuple, tuple],
    ) -> None:
        self.rows = seed
        self.stage_times = stage_times
        self.scores = scores
        self.loads = _device_loads(seed, stage_times)
        self.owner = _stage_device_map(seed, len(stage_times))
        self.adj = _adjacent_same_device_count(seed, len(stage_times))
        scores.setdefault(
            _placement_key(seed),
            (
                _load_variance(self.loads),
                self.adj,
                max(self.loads) if self.loads else 0.0,
            ),
        )

    def record_move(
        self, variant: list[list[int]], sid: int, from_d: int, to_d: int
    ) -> None:
        key = _placement_key(variant)
        if key in self.scores:
            return
        score, _new_from, _new_to, _to_row = _moved_score(
            self.rows,
            self.loads,
            self.owner,
            self.adj,
            self.stage_times,
            sid,
            from_d,
            to_d,
        )
        self.scores[key] = score


def _greedy_balance_placement(
    base: list[list[int]],
    device_num: int,
    stage_times: list[float],
    max_steps: int,
) -> list[list[int]]:
    """Hill-climb single-stage moves; each neighbor is scored incrementally
    (two rows + O(devices)) instead of rebuilding and rescoring the whole
    placement, with scores bit-identical to the full scorer.  Visit order
    and the strict < acceptance match the original, so the walk is too."""
    rows = [row[:] for row in base]
    stage_cnt = len(stage_times)
    loads = _device_loads(rows, stage_times)
    owner = _stage_device_map(rows, stage_cnt)
    adj = _adjacent_same_device_count(rows, stage_cnt)
    current_score = (_load_variance(loads), adj, max(loads) if loads else 0.0)
    for _ in range(max_steps):
        best_score = current_score
        best_move: tuple | None = None
        for from_d, stages in enumerate(rows):
            for sid in stages:
                for to_d in range(device_num):
                    if to_d == from_d:
                        continue
                    score, new_from, new_to, to_row = _moved_score(
                        rows, loads, owner, adj, stage_times, sid, from_d, to_d
                    )
                    if score < best_score:
                        best_score = score
                        best_move = (from_d, sid, to_d, new_from, new_to, to_row)
        if best_move is None:
            break
        from_d, sid, to_d, new_from, new_to, to_row = best_move
        rows[from_d] = [s for s in rows[from_d] if s != sid]
        rows[to_d] = to_row
        loads[from_d] = new_from
        loads[to_d] = new_to
        owner[sid] = to_d
        adj = best_score[1]
        current_score = best_score
    return rows


def generate_octopipe_placement_candidates(
    device_num: int,
    stage_num: int,
    beam_width: int = 32,
    stage_times: list[float] | None = None,
) -> list[list[list[int]]]:
    """Generate placement candidates; always includes interleaved."""
    candidates: list[list[list[int]]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()

    def add(placement: list[list[int]]) -> None:
        key = _placement_key(placement)
        if key in seen:
            return
        seen.add(key)
        candidates.append(placement)

    base = interleaved_placement(device_num, stage_num)
    add(base)

    chunk = max(1, stage_num // device_num)
    max_stage_per_device = chunk + 1 if stage_times is None else None

    # single-move variants of a seed are scored incrementally against it;
    # everything else (seeds, random swaps) falls back to the full scorer
    scores: dict[tuple, tuple] = {}
    base_scorer = _SeedScorer(base, stage_times, scores) if stage_times is not None else None

    _generate_adjacent_dispersal_variants(
        base,
        device_num,
        stage_num,
        max_stage_per_device=max_stage_per_device,
        add=add,
        scorer=base_scorer,
    )

    if stage_num == device_num:
        add([[i] for i in range(device_num)])

    seed_placements = [base]
    if stage_times is not None:
        balanced = _greedy_balance_placement(
            base,
            device_num,
            stage_times,
            max_steps=stage_num * device_num,
        )
        add(balanced)
        seed_placements.extend([balanced])
        balanced_scorer = _SeedScorer(balanced, stage_times, scores)
        _generate_adjacent_dispersal_variants(
            balanced,
            device_num,
            stage_num,
            max_stage_per_device=None,
            add=add,
            scorer=balanced_scorer,
        )
        _generate_transfer_neighbors(
            seed_placements,
            device_num,
            add=add,
            scorers=[base_scorer, balanced_scorer],
        )

    if stage_times is not None:
        for _ in range(max(0, beam_width - len(candidates))):
            c = [row[:] for row in base]
            if device_num >= 2 and len(c[0]) > 0 and len(c[1]) > 0:
                i = random.randrange(len(c[0]))
                j = random.randrange(len(c[1]))
                c[0][i], c[1][j] = c[1][j], c[0][i]
                for row in c:
                    row.sort()
                add(c)

        def score_of(placement: list[list[int]]) -> tuple:
            key = _placement_key(placement)
            score = scores.get(key)
            if score is None:
                score = placement_proxy_score(placement, stage_times)
                scores[key] = score
            return score

        ranked = sorted(candidates, key=score_of)
        ordered: list[list[list[int]]] = []
        seen_keys: set[tuple[tuple[int, ...], ...]] = set()
        for placement in [base] + ranked:
            key = _placement_key(placement)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            ordered.append(placement)
            if len(ordered) >= beam_width:
                break
        return ordered

    for _ in range(max(0, beam_width - len(candidates))):
        c = [row[:] for row in base]
        if device_num >= 2 and len(c[0]) > 1 and len(c[1]) > 0:
            i = random.randrange(len(c[0]))
            j = random.randrange(len(c[1]))
            c[0][i], c[1][j] = c[1][j], c[0][i]
            for row in c:
                row.sort()
            add(c)

    return candidates[:beam_width]


def fast_estimate_makespan(
    stage_f: list[float],
    stage_b: list[float],
    stage_w: list[float] | None,
    num_mb: int,
    placement: list[list[int]],
) -> float:
    """Pure-Python fast PP estimator (fallback when C ext unavailable)."""
    device_num = len(placement)
    device_load = [0.0] * device_num
    # simplified 1F1B: each microbatch visits all stages
    for mb in range(num_mb):
        for did, sids in enumerate(placement):
            for sid in sids:
                if sid < len(stage_f):
                    w_time = stage_w[sid] if stage_w is not None and sid < len(stage_w) else 0.0
                    device_load[did] += stage_f[sid] + stage_b[sid] + w_time
    return max(device_load) if device_load else 0.0


def generate_placement_candidates(
    device_num: int,
    stage_num: int,
    beam_width: int = 32,
) -> list[list[list[int]]]:
    """Generate diverse placement candidates."""
    return generate_octopipe_placement_candidates(device_num, stage_num, beam_width)


def search_placements(
    stage_f: list[float],
    stage_b: list[float],
    stage_w: list[float] | None,
    num_mb: int,
    device_num: int,
    evaluate_fn: Callable[[list[list[int]]], float],
    top_k: int = 8,
) -> list[tuple[list[list[int]], float]]:
    stage_num = len(stage_f)
    candidates = generate_placement_candidates(device_num, stage_num)
    scored: list[tuple[list[list[int]], float]] = []
    for placement in candidates:
        est = fast_estimate_makespan(stage_f, stage_b, stage_w, num_mb, placement)
        scored.append((placement, est))
    scored.sort(key=lambda x: x[1])
    # refine top-k with full simulation if provided
    refined = []
    for placement, est in scored[:top_k]:
        full = evaluate_fn(placement)
        refined.append((placement, full))
    refined.sort(key=lambda x: x[1])
    return refined
