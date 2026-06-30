from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParallelConfig:
    pp_size: int = 1
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    micro_batch_num: int = 8
    zero_stage: int = 1
    chunk_num: int | None = None  # None = auto (interleaved: max; else 1)
    bwd_split: bool = False
    vocab_parallel: bool = False
    grad_reduce_in_fp32: bool = True
    overlap_aware: bool = True
    save_memory: bool = False
    constrain_warmup: bool = False
    switch_workload_type: bool = True
    skip_overlap_until_first_backward: bool = True

    @property
    def device_num(self) -> int:
        return self.pp_size
