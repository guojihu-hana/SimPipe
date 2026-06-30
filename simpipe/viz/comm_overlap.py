"""PP send/recv overlap analysis (post-simulation)."""

from __future__ import annotations


def analyze_comm_overlap(records: list[dict], device_num: int) -> dict:
    """Simplified overlap stats from execution records."""
    edges = 0
    overlapped = 0
    for i, r in enumerate(records):
        if r.get("wtype") != "F":
            continue
        for j, other in enumerate(records):
            if other["did"] != r["did"] and other.get("mid") == r.get("mid"):
                edges += 1
                if other.get("start", 0) <= r.get("end", 0):
                    overlapped += 1
                break
    return {
        "num_comm_edges": edges,
        "num_either_overlapped": overlapped,
        "overlap_ratio": overlapped / edges if edges else 0.0,
    }
