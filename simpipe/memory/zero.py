from __future__ import annotations

from simpipe.config.parallel import ParallelConfig


def zero_sharded_bytes(total_bytes: int, parallel: ParallelConfig) -> int:
    """Per-GPU bytes after ZeRO + TP + PP + DP sharding."""
    tp = max(1, parallel.tp_size)
    pp = max(1, parallel.pp_size)
    dp = max(1, parallel.dp_size)
    zero_factor = {0: 1, 1: dp, 2: dp * tp, 3: dp * tp * pp}.get(parallel.zero_stage, 1)
    return total_bytes // (tp * pp * zero_factor)
