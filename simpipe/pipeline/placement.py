from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Placement:
    """device_id -> ordered list of stage ids on that device."""

    device_stages: list[list[int]]

    def validate(self, device_num: int, stage_num: int) -> None:
        if len(self.device_stages) != device_num:
            raise ValueError(f"Expected {device_num} devices, got {len(self.device_stages)}")
        seen: set[int] = set()
        for row in self.device_stages:
            for sid in row:
                if sid in seen:
                    raise ValueError(f"Duplicate stage {sid}")
                if sid < 0 or sid >= stage_num:
                    raise ValueError(f"Invalid stage id {sid}")
                seen.add(sid)
        if len(seen) != stage_num:
            raise ValueError(f"Expected {stage_num} stages, covered {len(seen)}")

    @classmethod
    def interleaved(cls, device_num: int, chunk_num: int) -> Placement:
        rows = [[i + device_num * j for j in range(chunk_num)] for i in range(device_num)]
        return cls(device_stages=rows)

    @classmethod
    def sequential(cls, device_num: int) -> Placement:
        return cls(device_stages=[[i] for i in range(device_num)])


def sort_placement_by_first_stage(placement: list[list[int]]) -> list[list[int]]:
    if not placement:
        return []
    return sorted((list(row) for row in placement), key=lambda row: row[0] if row else float("inf"))


def validate_placement(
    placement: list[list[int]],
    device_num: int,
    stage_num: int,
    partition_layer_counts: list[int] | None = None,
) -> None:
    Placement(placement).validate(device_num, stage_num)
    if partition_layer_counts is not None and len(partition_layer_counts) != stage_num:
        raise ValueError("partition length must match stage_num")
