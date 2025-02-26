# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import xp

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

import qttools.kernels.operator as operator
import qttools.kernels.eig as eig
import qttools.kernels.eigvalsh as eigvalsh
import qttools.kernels.svd as svd

__all__ = [
    "dsbsparse_kernels",
    "dsbcoo_kernels",
    "dsbcsr_kernels",
    "operator",
    "eig",
    "svd",
    "eigvalsh",
]
