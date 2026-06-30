from simpipe.comm.collective import allreduce_time
from simpipe.config.hardware import HardwareConfig
from simpipe.graph.tensor import DType, TensorKind, TensorSpec


def test_tp_comm_increases_with_size():
    hw = HardwareConfig()
    small = TensorSpec("a", (1, 1024, 1024), DType.FP16, TensorKind.ACTIVATION)
    large = TensorSpec("b", (1, 4096, 4096), DType.FP16, TensorKind.ACTIVATION)
    t1 = allreduce_time(small, hw, tp_size=8)
    t8 = allreduce_time(large, hw, tp_size=8)
    assert t8 > t1


def test_no_comm_for_tp1():
    hw = HardwareConfig()
    spec = TensorSpec("a", (1, 1024, 1024))
    assert allreduce_time(spec, hw, tp_size=1) == 0.0
