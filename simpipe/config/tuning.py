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
    # Allow total stage counts that are not multiples of pp_size (uneven
    # stages-per-device placements). max_stage_num caps the enumeration.
    irregular_stage_num: bool = False
    max_stage_num: int | None = None
    min_stage_num: int | None = None
    # Reject schedules whose peak per-device in-flight activation (in
    # layer*microbatch units; allocated at F start, freed after B and W)
    # exceeds this. Set to roughly num_layers to match 1F1B's rank-0 peak.
    max_inflight_layers: int | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> TuningConfig:
        if not data:
            return cls()
        max_stage_num = data.get("max_stage_num")
        min_stage_num = data.get("min_stage_num")
        max_inflight_layers = data.get("max_inflight_layers")
        return cls(
            auto_tune=bool(data.get("auto_tune", False)),
            sim_k=int(data.get("sim_k", 8)),
            beam_width=int(data.get("beam_width", 32)),
            partition_top_k=int(data.get("partition_top_k", 4)),
            result_top_k=int(data.get("result_top_k", 5)),
            bubble_overlap_tune=bool(data.get("bubble_overlap_tune", False)),
            bubble_overlap_max_iter=int(data.get("bubble_overlap_max_iter", 8)),
            bubble_overlap_group_by=str(data.get("bubble_overlap_group_by", "mid_type")),
            irregular_stage_num=bool(data.get("irregular_stage_num", False)),
            max_stage_num=None if max_stage_num is None else int(max_stage_num),
            min_stage_num=None if min_stage_num is None else int(min_stage_num),
            max_inflight_layers=(
                None if max_inflight_layers is None else int(max_inflight_layers)
            ),
        )
