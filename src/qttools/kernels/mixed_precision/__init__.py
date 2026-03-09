# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
from qttools import xp

if xp.__name__ == "numpy":
    from qttools.kernels.mixed_precision.numba import compress, decompress

elif xp.__name__ == "cupy":
    from qttools.kernels.mixed_precision.cupy import compress, decompress

else:
    raise ValueError(f"Unrecognized ARRAY_MODULE '{xp.__name__}'")

__all__ = [
    "compress",
    "decompress",
]
