from types import SimpleNamespace

from simpipe.core.device import Device
from simpipe.core.types import OrderedQueue, Schedule, WorkloadType
from simpipe.core.workload import Workload


class FakeWorkload:
    def __init__(self, wtype: WorkloadType, sid: int, mid: int = 0):
        self.wtype = wtype
        self.sid = sid
        self.mid = mid
        self.state = Workload.not_started
        self.constraints = set()

    def is_executable(self, _time: float) -> bool:
        return True


def _device_with_queue(*workloads: FakeWorkload, started_backward: bool = True) -> Device:
    device = Device.__new__(Device)
    device.schedule_method = Schedule.OctoPipe
    device.did = 0
    device.executable_workloads = OrderedQueue([WorkloadType.B, WorkloadType.F, WorkloadType.W])
    for workload in workloads:
        device.executable_workloads.push(workload)
    device.last_wtype = None
    device.runtime = SimpleNamespace(
        parallel=SimpleNamespace(
            overlap_aware=True,
            switch_workload_type=True,
            skip_overlap_until_first_backward=True,
        ),
        has_started_backward=lambda: started_backward,
        overlap_exempt_workloads=set(),
        overlap_exempt_group_by="mid_type",
        is_overlap_exempt=lambda workload: False,
        f_admission_blocked=lambda did, sid, mid: False,
    )
    return device


def test_octopipe_ignores_direct_dependency_before_first_backward():
    workload = FakeWorkload(WorkloadType.F, sid=1)
    device = _device_with_queue(workload, started_backward=False)
    device.has_direct_dependency = lambda _time, _workload: True

    assert device._next_workload(0) is workload


def test_octopipe_picks_first_non_overlap_workload_after_type_reset():
    b = FakeWorkload(WorkloadType.B, sid=2)
    f = FakeWorkload(WorkloadType.F, sid=0)
    device = _device_with_queue(b, f, started_backward=True)
    device.last_wtype = WorkloadType.F  # reset order to B, F, W
    device.has_direct_dependency = lambda _time, workload: workload is b

    assert device._next_workload(0) is f
    assert device.executable_workloads.pop() is b


def test_octopipe_overlap_exemption_bypasses_direct_dependency_delay():
    b = FakeWorkload(WorkloadType.B, sid=2, mid=7)
    device = _device_with_queue(b, started_backward=True)
    device.runtime.is_overlap_exempt = lambda workload: (workload.mid, workload.wtype) == (
        7,
        WorkloadType.B,
    )
    device.has_direct_dependency = lambda _time, _workload: True

    assert device._next_workload(0) is b
