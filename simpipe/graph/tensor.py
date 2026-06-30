from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from math import prod


class DType(IntEnum):
    FP16 = 2
    BF16 = 2
    FP32 = 4


class TensorKind(str, Enum):
    ACTIVATION = "activation"
    PARAMETER = "parameter"
    GRADIENT = "gradient"


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: DType = DType.FP16
    kind: TensorKind = TensorKind.ACTIVATION

    def nbytes(self) -> int:
        return tensor_bytes(self)


def dtype_size(dtype: DType) -> int:
    return int(dtype)


def tensor_bytes(spec: TensorSpec) -> int:
    return prod(spec.shape) * dtype_size(spec.dtype)
