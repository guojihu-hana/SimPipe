from types import SimpleNamespace

from simpipe.core.types import OrderedQueue, WorkloadType


def test_ordered_queue_reorders_existing_items_when_type_order_changes():
    queue = OrderedQueue([WorkloadType.B, WorkloadType.F, WorkloadType.W])
    f = SimpleNamespace(wtype=WorkloadType.F, sid=0, mid=0)
    b = SimpleNamespace(wtype=WorkloadType.B, sid=0, mid=0)

    queue.push(f)
    queue.push(b)
    queue.set_type_order([WorkloadType.F, WorkloadType.B, WorkloadType.W])

    assert queue.pop() is f


def test_ordered_queue_prioritizes_forward_by_mid_then_sid_and_backward_by_reverse_sid():
    queue = OrderedQueue([WorkloadType.F, WorkloadType.B])
    f_late_stage = SimpleNamespace(wtype=WorkloadType.F, sid=3, mid=0)
    f_early_stage = SimpleNamespace(wtype=WorkloadType.F, sid=0, mid=0)
    b_early_stage = SimpleNamespace(wtype=WorkloadType.B, sid=0, mid=0)
    b_late_stage = SimpleNamespace(wtype=WorkloadType.B, sid=3, mid=0)

    queue.push(f_late_stage)
    queue.push(f_early_stage)
    assert queue.pop() is f_early_stage

    queue = OrderedQueue([WorkloadType.B])
    queue.push(b_early_stage)
    queue.push(b_late_stage)
    assert queue.pop() is b_late_stage
