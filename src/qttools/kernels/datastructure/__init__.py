# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.
from qttools import xp

if xp.__name__ == "numpy":
    from qttools.kernels.datastructure.numba import dsbcoo as dsbcoo_kernels
    from qttools.kernels.datastructure.numba import dsbcsr as dsbcsr_kernels
    from qttools.kernels.datastructure.numba import dsbsparse as dsbsparse_kernels

elif xp.__name__ == "cupy":
    from qttools.kernels.datastructure.cupy import dsbcoo as dsbcoo_kernels
    from qttools.kernels.datastructure.cupy import dsbcsr as dsbcsr_kernels
    from qttools.kernels.datastructure.cupy import dsbsparse as dsbsparse_kernels

else:
    raise ValueError(f"Unrecognized ARRAY_MODULE '{xp.__name__}'")

__all__ = [
    "dsbsparse_kernels",
    "dsbcoo_kernels",
    "dsbcsr_kernels",
]
