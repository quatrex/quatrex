# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
from qttools import xp

if xp.__name__ == "numpy":
    from qttools.kernels.inplace.numba.inplace import iadd, iadd_obc
elif xp.__name__ == "cupy":
    from qttools.kernels.inplace.cupy.inplace import iadd, iadd_obc
else:
    raise ValueError(f"Unrecognized ARRAY_MODULE '{xp.__name__}'")

__all__ = ["iadd", "iadd_obc"]
