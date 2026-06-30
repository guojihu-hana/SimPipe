from __future__ import annotations

from simpipe.config.hardware import HardwareConfig
from simpipe.graph.tensor import tensor_bytes, TensorSpec


def collective_time_us(
    nbytes: int,
    hardware: HardwareConfig,
    *,
    intra_node: bool = True,
    n_ranks: int = 1,
) -> float:
    """Alpha-beta collective latency model (microseconds)."""
    if n_ranks <= 1:
        return 0.0
    bw = hardware.intra_node_bw_gbps if intra_node else hardware.inter_node_bw_gbps
    factor = 2 * (n_ranks - 1) / n_ranks  # ring allreduce approximation
    return hardware.comm_alpha_us + (nbytes * factor) / (bw * 1e3)


def allreduce_time(spec: TensorSpec, hardware: HardwareConfig, tp_size: int) -> float:
    return collective_time_us(tensor_bytes(spec), hardware, intra_node=True, n_ranks=tp_size)


def allgather_time(spec: TensorSpec, hardware: HardwareConfig, tp_size: int) -> float:
    return collective_time_us(tensor_bytes(spec), hardware, intra_node=True, n_ranks=tp_size)


def alltoall_time(spec: TensorSpec, hardware: HardwareConfig, ep_size: int) -> float:
    return collective_time_us(tensor_bytes(spec), hardware, intra_node=False, n_ranks=ep_size)
