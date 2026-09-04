from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# Executing microbatches in a different order is equivalent to permuting the
# per-slot (linear, quadratic) scale list: every schedule issues slot mids
# 0..n-1 in order, so slot k simply carries the workload of input microbatch
# order[k].  Candidates are evaluated with real simulations under the exact
# final configuration (same partition/placement/exemptions), so the reported
# best order is what the final run delivers.


@dataclass
class BatchOrderResult:
    # Slot k executes input microbatch order[k].
    order: list[int]
    makespan: float
    baseline_makespan: float
    trials: int

    @property
    def is_identity(self) -> bool:
        return self.order == list(range(len(self.order)))


def _dedup(orders: list[list[int]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    unique: list[list[int]] = []
    for order in orders:
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            unique.append(order)
    return unique


def heuristic_orders(scales: list[tuple[float, float]]) -> list[list[int]]:
    """Canonical starting orders: input, short-first, long-first, valley, mountain.

    Valley places the shortest microbatches at both pipeline fill and drain
    (cheap warmup and cooldown) with the longest in the middle; mountain is
    its reverse.
    """
    n = len(scales)
    identity = list(range(n))
    asc = sorted(identity, key=lambda i: (scales[i][0] + scales[i][1], i))
    desc = asc[::-1]
    left: list[int] = []
    right: list[int] = []
    for pos, idx in enumerate(asc):
        (left if pos % 2 == 0 else right).append(idx)
    valley = left + right[::-1]
    mountain = valley[::-1]
    return _dedup([identity, asc, desc, valley, mountain])


def tune_batch_order(
    scales: list[tuple[float, float]],
    evaluate: Callable[[list[int]], float],
    max_sims: int = 64,
) -> BatchOrderResult:
    """Search microbatch order: heuristics + pairwise-swap hill climb.

    evaluate(order) must return the makespan of running input microbatch
    order[k] in slot k (inf for stalled schedules).  Orders yielding the same
    scale sequence are deduplicated, so duplicate microbatch sizes do not
    burn simulation budget.
    """
    n = len(scales)
    identity = list(range(n))
    cache: dict[tuple[tuple[float, float], ...], float] = {}
    trials = 0

    def run(order: list[int]) -> float | None:
        nonlocal trials
        key = tuple(scales[i] for i in order)
        if key in cache:
            return cache[key]
        if trials >= max_sims:
            return None
        trials += 1
        makespan = evaluate(order)
        cache[key] = makespan
        return makespan

    baseline = run(identity)
    assert baseline is not None
    best_order, best = identity, baseline

    for order in heuristic_orders(scales):
        makespan = run(order)
        if makespan is not None and makespan < best:
            best_order, best = order, makespan

    # First-improvement pairwise swaps around the incumbent.
    for _ in range(2):
        improved = False
        for a in range(n - 1):
            for b in range(a + 1, n):
                candidate = list(best_order)
                candidate[a], candidate[b] = candidate[b], candidate[a]
                makespan = run(candidate)
                if makespan is not None and makespan < best - 1e-9:
                    best_order, best = candidate, makespan
                    improved = True
        if not improved:
            break

    return BatchOrderResult(
        order=best_order,
        makespan=best,
        baseline_makespan=baseline,
        trials=trials,
    )
