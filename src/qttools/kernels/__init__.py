# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import os

import qttools.kernels.operator as operator
from qttools import strtobool, xp

# TODO: adapt testing suite to test both JIT and non-JIT versions
USE_CUPY_JIT = strtobool(os.getenv("USE_CUPY_JIT", "True"), default=True)

if xp.__name__ == "numpy":
    from qttools.kernels.numba import dsbcoo as dsbcoo_kernels
    from qttools.kernels.numba import dsbcsr as dsbcsr_kernels
    from qttools.kernels.numba import dsbsparse as dsbsparse_kernels

elif xp.__name__ == "cupy":
    from qttools.kernels.cuda import dsbcoo as dsbcoo_kernels
    from qttools.kernels.cuda import dsbcsr as dsbcsr_kernels
    from qttools.kernels.cuda import dsbsparse as dsbsparse_kernels

else:
    raise ValueError(f"Unrecognized ARRAY_MODULE '{xp.__name__}'")


__all__ = [
    "dsbsparse_kernels",
    "dsbcoo_kernels",
    "dsbcsr_kernels",
    "operator",
]
