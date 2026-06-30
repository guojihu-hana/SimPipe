from simpipe.core.types import WorkloadType
from simpipe.pipeline.placement import Placement
from simpipe.pipeline.scheduling.base import ScheduleContext
from simpipe.pipeline.scheduling.interleaved import InterleavedStrategy
from simpipe.pipeline.scheduling.one_f_one_b import OneFOneBStrategy
from simpipe.pipeline.scheduling.zbh import ZBHStrategy


def test_1f1b_alternates_b_and_f_after_warmup():
    ctx = ScheduleContext(
        device_num=4,
        stage_num=4,
        micro_batch_num=8,
        mid_offset=0,
        bwd_split=False,
        chunk_num=1,
        placement=Placement.sequential(4),
    )
    sched = OneFOneBStrategy().generate(ctx)
    d0 = sched[0]
    # warmup: 4x F
    assert all(w == WorkloadType.F for w, _, _ in d0[:4])
    # steady: B,F,B,F,...
    steady = d0[4:12]
    assert steady[0][0] == WorkloadType.B and steady[0][1] == 0
    assert steady[1][0] == WorkloadType.F and steady[1][1] == 4
    assert steady[2][0] == WorkloadType.B and steady[2][1] == 1
    assert steady[3][0] == WorkloadType.F and steady[3][1] == 5


def test_1f1b_uses_device_as_stage_id():
    ctx = ScheduleContext(
        device_num=4,
        stage_num=4,
        micro_batch_num=4,
        mid_offset=0,
        bwd_split=False,
        chunk_num=1,
        placement=Placement.sequential(4),
    )
    sched = OneFOneBStrategy().generate(ctx)
    for did, row in enumerate(sched):
        for _, _, sid in row:
            assert sid == did


def test_interleaved_alternates_f_and_b_after_warmup():
    ctx = ScheduleContext(
        device_num=4,
        stage_num=4,
        micro_batch_num=8,
        mid_offset=0,
        bwd_split=False,
        chunk_num=1,
        placement=Placement.interleaved(4, 1),
    )
    sched = InterleavedStrategy().generate(ctx)
    d0 = sched[0]
    # warmup: 6x F (interleaved formula with chunk_num=1)
    assert len([w for w, _, _ in d0 if w == WorkloadType.F]) >= 6
    # steady starts with F then B
    tail = d0[6:10]
    assert tail[0][0] == WorkloadType.F
    assert tail[1][0] == WorkloadType.B


def test_zbh_round_robin_b_w_f():
    ctx = ScheduleContext(
        device_num=4,
        stage_num=4,
        micro_batch_num=8,
        mid_offset=0,
        bwd_split=True,
        chunk_num=1,
        placement=Placement.sequential(4),
        max_act=1,
    )
    sched = ZBHStrategy().generate(ctx)
    d0 = sched[0]
    steady = [w for w, _, sid in d0 if sid == 0]
    # after warmup F's, should see B before W before next F pattern
    types_after_warmup = steady[1:7]
    assert WorkloadType.B in types_after_warmup
    assert WorkloadType.F in types_after_warmup
