"""Search heuristics for partition/placement (calls pipeline/ API)."""

from simpipe.tuning.fast_est import (
    fast_estimate_makespan,
    generate_placement_candidates,
    search_placements,
)

__all__ = [
    "fast_estimate_makespan",
    "generate_placement_candidates",
    "search_placements",
]
