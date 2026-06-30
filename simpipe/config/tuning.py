from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TuningConfig:
    """OctoPipe partition/placement search knobs."""

    auto_tune: bool = False
    sim_k: int = 8
    beam_width: int = 32
    partition_top_k: int = 4
    result_top_k: int = 5
    bubble_overlap_tune: bool = False
    bubble_overlap_max_iter: int = 8
    bubble_overlap_group_by: str = "mid_type"

    @classmethod
    def from_dict(cls, data: dict | None) -> TuningConfig:
        if not data:
            return cls()
        return cls(
            auto_tune=bool(data.get("auto_tune", False)),
            sim_k=int(data.get("sim_k", 8)),
            beam_width=int(data.get("beam_width", 32)),
            partition_top_k=int(data.get("partition_top_k", 4)),
            result_top_k=int(data.get("result_top_k", 5)),
            bubble_overlap_tune=bool(data.get("bubble_overlap_tune", False)),
            bubble_overlap_max_iter=int(data.get("bubble_overlap_max_iter", 8)),
            bubble_overlap_group_by=str(data.get("bubble_overlap_group_by", "mid_type")),
        )
