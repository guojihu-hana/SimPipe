from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TraceNode:
    id: str
    node_type: str  # compute | comm | mem
    duration_us: float
    deps: list[str]


@dataclass
class ExecutionTrace:
    nodes: list[TraceNode]

    @classmethod
    def from_json(cls, path: Path) -> ExecutionTrace:
        data = json.loads(path.read_text())
        nodes = [
            TraceNode(
                id=n["id"],
                node_type=n["type"],
                duration_us=n["duration_us"],
                deps=n.get("deps", []),
            )
            for n in data["nodes"]
        ]
        return cls(nodes=nodes)

    def to_operator_durations(self) -> dict[str, float]:
        return {n.id: n.duration_us for n in self.nodes if n.node_type == "compute"}


def load_chakra_trace(path: Path) -> ExecutionTrace:
    """Load simplified Chakra-compatible trace JSON."""
    return ExecutionTrace.from_json(path)
