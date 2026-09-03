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


def _stage_device(placement: list[list[int]], stage_id: int) -> int | None:
    for did, stages in enumerate(placement):
        if stage_id in stages:
            return did
    return None


def _device_loads(placement: list[list[int]], stage_times: list[float]) -> list[float]:
    return [sum(stage_times[sid] for sid in row) for row in placement]


def _load_variance(loads: list[float]) -> float:
    if not loads:
        return 0.0
    mean = sum(loads) / len(loads)
    return sum((load - mean) ** 2 for load in loads) / len(loads)


def _adjacent_same_device_count(placement: list[list[int]], stage_num: int) -> int:
    count = 0
    for sid in range(stage_num - 1):
        d0 = _stage_device(placement, sid)
        d1 = _stage_device(placement, sid + 1)
        if d0 is not None and d0 == d1:
            count += 1
    return count


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
) -> None:
    for sid in range(stage_num - 1):
        d0 = _stage_device(base, sid)
        d1 = _stage_device(base, sid + 1)
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
            add(variant)


def _generate_transfer_neighbors(
    seed_placements: list[list[list[int]]],
    device_num: int,
    add: Callable[[list[list[int]]], None],
) -> None:
    for placement in seed_placements:
        for from_device, stages in enumerate(placement):
            for stage_id in stages:
                for target in range(device_num):
                    if target == from_device:
                        continue
                    variant = _transfer_stage(placement, stage_id, from_device, target)
                    if variant is not None:
                        add(variant)


def _greedy_balance_placement(
    base: list[list[int]],
    device_num: int,
    stage_times: list[float],
    max_steps: int,
) -> list[list[int]]:
    current = [row[:] for row in base]
    current_score = placement_proxy_score(current, stage_times)
    for _ in range(max_steps):
        best_variant: list[list[int]] | None = None
        best_score = current_score
        for from_device, stages in enumerate(current):
            for stage_id in stages:
                for target in range(device_num):
                    if target == from_device:
                        continue
                    variant = _transfer_stage(current, stage_id, from_device, target)
                    if variant is None:
                        continue
                    score = placement_proxy_score(variant, stage_times)
                    if score < best_score:
                        best_score = score
                        best_variant = variant
        if best_variant is None:
            break
        current = best_variant
        current_score = best_score
    return current


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

    _generate_adjacent_dispersal_variants(
        base,
        device_num,
        stage_num,
        max_stage_per_device=max_stage_per_device,
        add=add,
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
        _generate_adjacent_dispersal_variants(
            balanced,
            device_num,
            stage_num,
            max_stage_per_device=None,
            add=add,
        )
        _generate_transfer_neighbors(seed_placements, device_num, add=add)

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
        ranked = sorted(
            candidates,
            key=lambda placement: placement_proxy_score(placement, stage_times),
        )
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
