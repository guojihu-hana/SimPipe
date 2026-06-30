from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceCompBubbleStats:
    did: int
    comp: float
    bubble: float
    warmup_bubble: float
    cooldown_bubble: float
    residual_bubble: float

    @property
    def total(self) -> float:
        return self.comp + self.bubble

    @property
    def comp_ratio(self) -> float:
        return self.comp / self.total if self.total else 0.0

    @property
    def bubble_ratio(self) -> float:
        return self.bubble / self.total if self.total else 0.0

    @property
    def warmup_bubble_ratio(self) -> float:
        return self.warmup_bubble / self.total if self.total else 0.0

    @property
    def cooldown_bubble_ratio(self) -> float:
        return self.cooldown_bubble / self.total if self.total else 0.0

    @property
    def residual_bubble_ratio(self) -> float:
        return self.residual_bubble / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "did": self.did,
            "comp": self.comp,
            "bubble": self.bubble,
            "warmup_bubble": self.warmup_bubble,
            "cooldown_bubble": self.cooldown_bubble,
            "residual_bubble": self.residual_bubble,
            "total": self.total,
            "comp_ratio": self.comp_ratio,
            "bubble_ratio": self.bubble_ratio,
            "warmup_bubble_ratio": self.warmup_bubble_ratio,
            "cooldown_bubble_ratio": self.cooldown_bubble_ratio,
            "residual_bubble_ratio": self.residual_bubble_ratio,
            "total_ratio": 1.0 if self.total else 0.0,
        }


@dataclass(frozen=True)
class PipelineCompBubbleStats:
    first_backward_start: float | None
    last_forward_end: float | None
    makespan: float
    per_device: tuple[DeviceCompBubbleStats, ...]

    @property
    def total_comp(self) -> float:
        return sum(device.comp for device in self.per_device)

    @property
    def total_bubble(self) -> float:
        return sum(device.bubble for device in self.per_device)

    @property
    def total_warmup_bubble(self) -> float:
        return sum(device.warmup_bubble for device in self.per_device)

    @property
    def total_cooldown_bubble(self) -> float:
        return sum(device.cooldown_bubble for device in self.per_device)

    @property
    def total_residual_bubble(self) -> float:
        return sum(device.residual_bubble for device in self.per_device)

    def avg_bubble_ratio(self, device_num: int | None = None) -> float:
        n = device_num if device_num is not None else len(self.per_device)
        if n <= 0 or self.makespan <= 0:
            return 0.0
        if device_num is None:
            return self.total_bubble / (n * self.makespan)
        idle = 0.0
        by_did = {device.did: device for device in self.per_device}
        for did in range(device_num):
            device = by_did.get(did)
            idle += device.bubble if device is not None else self.makespan
        return idle / (n * self.makespan)

    def to_dict(self) -> dict:
        return {
            "first_backward_start": self.first_backward_start,
            "last_forward_end": self.last_forward_end,
            "makespan": self.makespan,
            "per_device": [device.to_dict() for device in self.per_device],
        }


def _record_interval(record: dict) -> tuple[float, float]:
    start = float(record.get("start") or 0)
    end = float(record.get("end") or start + (record.get("duration") or 0))
    if end < start:
        start, end = end, start
    return start, end


def _records_by_device(records: list[dict]) -> dict[int, list[dict]]:
    by_device: dict[int, list[dict]] = {}
    for record in records:
        by_device.setdefault(int(record["did"]), []).append(record)
    for device_records in by_device.values():
        device_records.sort(key=lambda r: (_record_interval(r)[0], _record_interval(r)[1]))
    return by_device


def _pipeline_bubble_boundaries(records: list[dict]) -> tuple[float | None, float | None, float]:
    first_b_start: float | None = None
    last_f_end: float | None = None
    max_t = 0.0
    for record in records:
        wtype = str(record.get("wtype", "F")).upper()
        start, end = _record_interval(record)
        max_t = max(max_t, end)
        if wtype == "B":
            first_b_start = start if first_b_start is None else min(first_b_start, start)
        if wtype == "F":
            last_f_end = end if last_f_end is None else max(last_f_end, end)
    return first_b_start, last_f_end, max_t


def _classify_bubble_interval(
    start: float,
    end: float,
    *,
    first_b_start: float | None,
    last_f_end: float | None,
) -> tuple[float, float, float]:
    """Split idle interval [start, end) into warmup, cooldown, and residual bubble."""
    if end <= start:
        return 0.0, 0.0, 0.0

    points = [start, end]
    if first_b_start is not None and start < first_b_start < end:
        points.append(first_b_start)
    if last_f_end is not None and start < last_f_end < end:
        points.append(last_f_end)
    points = sorted(set(points))

    warmup = cooldown = residual = 0.0
    for i in range(len(points) - 1):
        seg_start, seg_end = points[i], points[i + 1]
        seg_mid = (seg_start + seg_end) / 2.0
        length = seg_end - seg_start
        if first_b_start is not None and seg_mid < first_b_start:
            warmup += length
        elif last_f_end is not None and seg_mid > last_f_end:
            cooldown += length
        else:
            residual += length
    return warmup, cooldown, residual


def analyze_pipeline_comp_bubble(
    records: list[dict],
    *,
    device_num: int | None = None,
) -> PipelineCompBubbleStats:
    """Compute per-device compute and bubble metrics for pipeline tuning and viz."""
    if not records:
        return PipelineCompBubbleStats(None, None, 0.0, ())

    first_b_start, last_f_end, max_t = _pipeline_bubble_boundaries(records)
    per_device: list[DeviceCompBubbleStats] = []
    seen: set[int] = set()

    for did, device_records in sorted(_records_by_device(records).items()):
        seen.add(did)
        comp = 0.0
        warmup_bubble = 0.0
        cooldown_bubble = 0.0
        residual_bubble = 0.0
        prev_end = 0.0
        for record in device_records:
            start, end = _record_interval(record)
            comp += max(0.0, end - start)
            w, c, r = _classify_bubble_interval(
                prev_end,
                start,
                first_b_start=first_b_start,
                last_f_end=last_f_end,
            )
            warmup_bubble += w
            cooldown_bubble += c
            residual_bubble += r
            prev_end = end
        w, c, r = _classify_bubble_interval(
            prev_end,
            max_t,
            first_b_start=first_b_start,
            last_f_end=last_f_end,
        )
        warmup_bubble += w
        cooldown_bubble += c
        residual_bubble += r
        bubble = warmup_bubble + cooldown_bubble + residual_bubble
        per_device.append(
            DeviceCompBubbleStats(
                did=did,
                comp=comp,
                bubble=bubble,
                warmup_bubble=warmup_bubble,
                cooldown_bubble=cooldown_bubble,
                residual_bubble=residual_bubble,
            )
        )

    if device_num is not None:
        for did in range(device_num):
            if did in seen:
                continue
            w, c, r = _classify_bubble_interval(
                0.0,
                max_t,
                first_b_start=first_b_start,
                last_f_end=last_f_end,
            )
            per_device.append(
                DeviceCompBubbleStats(
                    did=did,
                    comp=0.0,
                    bubble=w + c + r,
                    warmup_bubble=w,
                    cooldown_bubble=c,
                    residual_bubble=r,
                )
            )
        per_device.sort(key=lambda device: device.did)

    return PipelineCompBubbleStats(
        first_backward_start=first_b_start,
        last_forward_end=last_f_end,
        makespan=max_t,
        per_device=tuple(per_device),
    )


def iter_inter_workload_gaps(records: list[dict]) -> Iterator[tuple[dict, float]]:
    """Yield (following_workload, gap_duration) for inter-workload idle gaps only."""
    for device_records in _records_by_device(records).values():
        prev_end: float | None = None
        for record in device_records:
            start, end = _record_interval(record)
            if prev_end is not None:
                gap = max(0.0, start - prev_end)
                if gap > 0:
                    yield record, gap
            prev_end = end


def total_inter_workload_bubble(records: list[dict]) -> float:
    return sum(gap for _, gap in iter_inter_workload_gaps(records))


def _per_device_variance(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def device_comp_variance(stats: PipelineCompBubbleStats) -> float:
    return _per_device_variance(tuple(device.comp for device in stats.per_device))


def device_bubble_variance(stats: PipelineCompBubbleStats) -> float:
    return _per_device_variance(tuple(device.bubble for device in stats.per_device))


def tuning_score(
    stats: PipelineCompBubbleStats,
    makespan: float,
) -> float:
    """Lower is better: makespan only."""
    del stats
    return makespan
