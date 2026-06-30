from __future__ import annotations

from dataclasses import dataclass

from simpipe.config.parallel import ParallelConfig


@dataclass(frozen=True)
class ZeroMemoryShard:
    parameter_bytes: int
    gradient_bytes: int
    master_parameter_bytes: int
    optimizer_bytes: int

    @property
    def total(self) -> int:
        return (
            self.parameter_bytes
            + self.gradient_bytes
            + self.master_parameter_bytes
            + self.optimizer_bytes
        )


def zero_sharded_bytes(total_bytes: int, parallel: ParallelConfig) -> int:
    """Per-GPU bytes after ZeRO + TP + PP + DP sharding."""
    tp = max(1, parallel.tp_size)
    pp = max(1, parallel.pp_size)
    dp = max(1, parallel.dp_size)
    zero_factor = {0: 1, 1: dp, 2: dp * tp, 3: dp * tp * pp}.get(parallel.zero_stage, 1)
    return total_bytes // (tp * pp * zero_factor)


def zero_model_state_bytes(
    parameter_bytes: int,
    gradient_bytes: int,
    optimizer_bytes: int,
    parallel: ParallelConfig,
    master_parameter_bytes: int = 0,
) -> ZeroMemoryShard:
    """Per-rank ZeRO sharding for model states already local to one PP/TP/EP rank.

    ZeRO-1 shards optimizer-owned states (FP32 master params + optimizer moments).
    ZeRO-2 shards optimizer-owned states and gradients across DP ranks.
    ZeRO-3 shards optimizer-owned states, gradients, and parameters across DP ranks.
    """
    dp = max(1, parallel.dp_size)
    stage = parallel.zero_stage
    if stage <= 0:
        return ZeroMemoryShard(
            parameter_bytes,
            gradient_bytes,
            master_parameter_bytes,
            optimizer_bytes,
        )
    if stage == 1:
        return ZeroMemoryShard(
            parameter_bytes,
            gradient_bytes,
            master_parameter_bytes // dp,
            optimizer_bytes // dp,
        )
    if stage == 2:
        return ZeroMemoryShard(
            parameter_bytes,
            gradient_bytes // dp,
            master_parameter_bytes // dp,
            optimizer_bytes // dp,
        )
    if stage == 3:
        return ZeroMemoryShard(
            parameter_bytes // dp,
            gradient_bytes // dp,
            master_parameter_bytes // dp,
            optimizer_bytes // dp,
        )
    raise ValueError(f"Unsupported ZeRO stage: {stage}")
