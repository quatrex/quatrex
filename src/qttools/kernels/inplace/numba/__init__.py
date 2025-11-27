# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.


from qttools.kernels.inplace.numba.inplace_add import add_OBC_inplace_CPU
from qttools.kernels.inplace.numba.inplace_sub import sub_OBC_inplace_CPU

THREADS_PER_BLOCK = 1024

__all__ = [
    "add_OBC_inplace_CPU",
    "sub_OBC_inplace_CPU",
]
