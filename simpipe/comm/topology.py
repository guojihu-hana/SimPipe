from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NetworkTopology:
    num_nodes: int = 1
    gpus_per_node: int = 8
    nvlink_bw_gbps: float = 600.0
    ib_bw_gbps: float = 50.0

    def is_intra_node(self, rank_a: int, rank_b: int) -> bool:
        return rank_a // self.gpus_per_node == rank_b // self.gpus_per_node
