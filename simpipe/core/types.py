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
    """Ready queue with one heap per workload type.

    The type order only decides which bucket pops first, so changing it
    (OctoPipe flips between B-first and F-first after every workload when
    switch_workload_type is on) is O(1) instead of rebuilding one big heap
    whose keys embed the type rank.  Within a type the order never changes:
    B drains the deepest stage first, everything else runs by (mid, sid).
    One workload per (mid, sid, type) makes keys unique; the insertion
    counter stays as a defensive tie-break.
    """

    def __init__(self, type_order: list[WorkloadType]):
        self._heaps: dict[WorkloadType, list] = {}
        self._counter = 0
        self._order: tuple[WorkloadType, ...] = tuple(type_order)

    def set_type_order(self, type_order: list[WorkloadType]) -> None:
        self._order = tuple(type_order)

    def _key(self, workload):
        if workload.wtype == WorkloadType.B:
            return (-workload.sid, workload.mid, self._counter)
        return (workload.mid, workload.sid, self._counter)

    def push(self, workload) -> None:
        heap = self._heaps.setdefault(workload.wtype, [])
        heapq.heappush(heap, (self._key(workload), workload))
        self._counter += 1

    def _next_heap(self) -> list | None:
        for t in self._order:
            heap = self._heaps.get(t)
            if heap:
                return heap
        # types outside the current order rank last (legacy rank-999 slot)
        for t, heap in self._heaps.items():
            if heap and t not in self._order:
                return heap
        return None

    def pop(self):
        heap = self._next_heap()
        return heapq.heappop(heap)[1] if heap else None

    def peek(self):
        heap = self._next_heap()
        return heap[0][1] if heap else None

    def __bool__(self) -> bool:
        return any(self._heaps.values())

    def __len__(self) -> int:
        return sum(len(heap) for heap in self._heaps.values())
