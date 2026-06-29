# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import qttools.kernels.operator as operator
from qttools import xp

value_types = {
    xp.float64: "double",
    xp.complex128: "complex<double>",
}

index_types = {
    xp.int32: "int",
    xp.int64: "long long",
}

__all__ = [
    "operator",
    "value_types",
    "index_types",
]
