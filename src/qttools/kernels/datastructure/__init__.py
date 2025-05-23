# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.
from qttools import xp

if xp.__name__ == "numpy":
    from qttools.kernels.datastructure.numba import dsdbcoo as dsdbcoo_kernels
    from qttools.kernels.datastructure.numba import dsdbcsr as dsdbcsr_kernels
    from qttools.kernels.datastructure.numba import dsdbsparse as dsdbsparse_kernels

elif xp.__name__ == "cupy":
    from qttools.kernels.datastructure.cupy import dsdbcoo as dsdbcoo_kernels
    from qttools.kernels.datastructure.cupy import dsdbcsr as dsdbcsr_kernels
    from qttools.kernels.datastructure.cupy import dsdbsparse as dsdbsparse_kernels

else:
    raise ValueError(f"Unrecognized ARRAY_MODULE '{xp.__name__}'")

__all__ = [
    "dsdbsparse_kernels",
    "dsdbcoo_kernels",
    "dsdbcsr_kernels",
]
