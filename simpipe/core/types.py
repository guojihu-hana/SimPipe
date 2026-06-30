from __future__ import annotations

from enum import Enum, IntEnum
import heapq


class WorkloadType(IntEnum):
    F = 1
    B = 2
    W = 3
    R = 4
    COMM_TP = 5
    COMM_DP = 6
    COMM_PP = 7


class Schedule(Enum):
    S1F1B = 10
    INTERLEAVED = 11
    ZBH = 12
    ZBV = 13
    AFAB = 14
    OctoPipe = 15
    Mist = 16
    ReCycle = 17
    BAPAR = 18


SCHEDULE_ALIASES = {
    "1f1b": Schedule.S1F1B,
    "interleaved": Schedule.INTERLEAVED,
    "zbh": Schedule.ZBH,
    "afab": Schedule.AFAB,
    "octopipe": Schedule.OctoPipe,
    "recycle": Schedule.ReCycle,
    "bapar": Schedule.BAPAR,
}


def parse_schedule(name: str) -> Schedule:
    key = name.lower().replace("-", "").replace("_", "")
    for alias, sched in SCHEDULE_ALIASES.items():
        if key == alias.replace("_", ""):
            return sched
    raise ValueError(f"Unknown schedule: {name}")


class WorkloadConstraint:
    __slots__ = ("device_id", "microbatch_id", "stage_id", "workload_type")

    def __init__(self, device_id: int, microbatch_id: int, stage_id: int, workload_type: WorkloadType):
        self.device_id = device_id
        self.microbatch_id = microbatch_id
        self.stage_id = stage_id
        self.workload_type = workload_type

    def __hash__(self) -> int:
        return hash((self.device_id, self.microbatch_id, self.stage_id, self.workload_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorkloadConstraint):
            return False
        return (
            self.device_id == other.device_id
            and self.microbatch_id == other.microbatch_id
            and self.stage_id == other.stage_id
            and self.workload_type == other.workload_type
        )


class OrderedQueue:
    def __init__(self, type_order: list[WorkloadType]):
        self._heap: list = []
        self._counter = 0
        self.last_type_order = list(type_order)
        self.type_priority = {t: i for i, t in enumerate(type_order)}

    def set_type_order(self, type_order: list[WorkloadType]) -> None:
        if type_order == self.last_type_order:
            return
        self.last_type_order = list(type_order)
        self.type_priority = {t: i for i, t in enumerate(type_order)}
        new_heap = []
        for _, workload in self._heap:
            new_heap.append((self._key(workload), workload))
        heapq.heapify(new_heap)
        self._heap = new_heap

    def _key(self, workload):
        type_rank = self.type_priority.get(workload.wtype, 999)
        if workload.wtype == WorkloadType.B:
            return (type_rank, -workload.sid, workload.mid, self._counter)
        return (type_rank, workload.mid, workload.sid, self._counter)

    def push(self, workload) -> None:
        key = self._key(workload)
        self._counter += 1
        heapq.heappush(self._heap, (key, workload))

    def pop(self):
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[1]

    def peek(self):
        if not self._heap:
            return None
        return self._heap[0][1]

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)
