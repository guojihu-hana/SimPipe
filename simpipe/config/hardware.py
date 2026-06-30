from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HardwareConfig:
    gpu_peak_tflops: float = 312.0
    gpu_hbm_gb: float = 80.0
    intra_node_bw_gbps: float = 600.0
    inter_node_bw_gbps: float = 50.0
    comm_alpha_us: float = 5.0
    comp_power: float = 1.0
    tp_overlap_fraction: float = 0.5
    dp_overlap_fraction: float = 0.5

    @property
    def gpu_hbm_bytes(self) -> int:
        return int(self.gpu_hbm_gb * 1024**3)
