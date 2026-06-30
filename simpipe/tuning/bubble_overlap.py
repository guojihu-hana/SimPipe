from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from simpipe.core.types import WorkloadType
from simpipe.metrics.comp_bubble import iter_inter_workload_gaps, total_inter_workload_bubble

OverlapGroup = tuple
OverlapExemption = OverlapGroup


@dataclass
class BubbleOverlapTrial:
    iteration: int
    group: OverlapGroup
    previous_bubble: float
    trial_bubble: float
    accepted: bool


@dataclass
class BubbleOverlapTuneResult:
    exemptions: set[OverlapExemption]
    trials: list[BubbleOverlapTrial]


def _record_wtype(record: dict) -> WorkloadType:
    wtype = record["wtype"]
    if isinstance(wtype, WorkloadType):
        return wtype
    return WorkloadType[str(wtype)]


def normalize_group_by(group_by: str) -> str:
    normalized = group_by.lower().replace("+", "_").replace("-", "_")
    aliases = {
        "mid": "mid",
        "mid_type": "mid_type",
        "mid_wtype": "mid_type",
        "mid_sid_type": "mid_sid_type",
        "mid_sid_wtype": "mid_sid_type",
    }
    if normalized not in aliases:
        raise ValueError(
            "bubble_overlap_group_by must be one of: mid, mid_type, mid_sid_type"
        )
    return aliases[normalized]


def group_key_from_record(record: dict, group_by: str = "mid_type") -> OverlapGroup:
    mode = normalize_group_by(group_by)
    mid = int(record["mid"])
    if mode == "mid":
        return (mid,)
    wtype = _record_wtype(record)
    if mode == "mid_type":
        return (mid, wtype)
    return (mid, int(record["sid"]), wtype)


def group_key_from_workload(workload, group_by: str = "mid_type") -> OverlapGroup:
    mode = normalize_group_by(group_by)
    if mode == "mid":
        return (workload.mid,)
    if mode == "mid_type":
        return (workload.mid, workload.wtype)
    return (workload.mid, workload.sid, workload.wtype)


def format_group(group: OverlapGroup) -> str:
    parts = []
    for item in group:
        parts.append(item.name if isinstance(item, WorkloadType) else str(item))
    return "(" + ", ".join(parts) + ")"


def bubble_by_group(
    records: list[dict], group_by: str = "mid_type"
) -> dict[OverlapExemption, float]:
    bubbles: dict[OverlapExemption, float] = defaultdict(float)
    for record, gap in iter_inter_workload_gaps(records):
        bubbles[group_key_from_record(record, group_by)] += gap
    return dict(bubbles)


def bubble_by_mid_type(records: list[dict]) -> dict[OverlapExemption, float]:
    return bubble_by_group(records, "mid_type")


def choose_next_bubble_group(
    records: list[dict],
    exemptions: set[OverlapExemption],
    rejected: set[OverlapExemption],
    group_by: str = "mid_type",
) -> OverlapExemption | None:
    candidates = {
        key: bubble
        for key, bubble in bubble_by_group(records, group_by).items()
        if key not in exemptions and key not in rejected and bubble > 0
    }
    if not candidates:
        return None
    return max(candidates, key=candidates.get)


def tune_overlap_exemptions(
    initial_records: list[dict],
    run_with_exemptions: Callable[[set[OverlapExemption]], list[dict]],
    max_iter: int,
    group_by: str = "mid_type",
) -> BubbleOverlapTuneResult:
    exemptions: set[OverlapExemption] = set()
    rejected: set[OverlapExemption] = set()
    trials: list[BubbleOverlapTrial] = []
    best_records = initial_records
    best_bubble = total_inter_workload_bubble(best_records)

    for iteration in range(1, max_iter + 1):
        candidate = choose_next_bubble_group(best_records, exemptions, rejected, group_by)
        if candidate is None:
            break
        trial_exemptions = {*exemptions, candidate}
        trial_records = run_with_exemptions(trial_exemptions)
        trial_bubble = total_inter_workload_bubble(trial_records)
        if trial_bubble < best_bubble:
            exemptions = trial_exemptions
            best_records = trial_records
            trials.append(BubbleOverlapTrial(iteration, candidate, best_bubble, trial_bubble, True))
            best_bubble = trial_bubble
        else:
            rejected.add(candidate)
            trials.append(BubbleOverlapTrial(iteration, candidate, best_bubble, trial_bubble, False))
    return BubbleOverlapTuneResult(exemptions, trials)
