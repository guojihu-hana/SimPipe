from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HardwareConfig:
    gpu_peak_tflops: float = 312.0
    gpu_hbm_gb: float = 80.0
    intra_node_bw_gbps: float = 600.0
    inter_node_bw_gbps: float = 50.0
    comm_alpha_us: float = 5.0
    # Empirical octopipe runtime overheads, in ms (profiled/tick mode only):
    # workload_overhead_ms is added to every f/b/w workload duration (python
    # dispatch, event bookkeeping, chunk switch); p2p_latency_ms delays every
    # cross-device dependency edge (NVSHMEM staged put + signal wait).
    workload_overhead_ms: float = 0.0
    p2p_latency_ms: float = 0.0
    comp_power: float = 1.0
    tp_overlap_fraction: float = 0.5
    dp_overlap_fraction: float = 0.5

    @property
    def gpu_hbm_bytes(self) -> int:
        return int(self.gpu_hbm_gb * 1024**3)
