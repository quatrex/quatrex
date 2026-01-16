# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.


from qttools.kernels.inplace.numba.inplace_add import iadd_OBC_CPU
from qttools.kernels.inplace.numba.inplace_sub import isub_OBC_CPU

THREADS_PER_BLOCK = 1024

__all__ = [
    "iadd_OBC_CPU",
    "isub_OBC_CPU",
]
