from __future__ import annotations


def exposed_comm_time(comm_time: float, compute_time: float, overlap_fraction: float) -> float:
    """Return comm time not hidden behind compute."""
    if compute_time >= comm_time:
        return comm_time * (1.0 - overlap_fraction)
    hidden = compute_time * overlap_fraction
    return max(0.0, comm_time - hidden)


def effective_step_time(compute_time: float, comm_time: float, overlap_fraction: float) -> float:
    exposed = exposed_comm_time(comm_time, compute_time, overlap_fraction)
    return compute_time + exposed
