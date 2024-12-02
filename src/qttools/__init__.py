# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import os
from typing import Any, TypeAlias, TypeVar
from warnings import warn

from numpy.typing import ArrayLike

from qttools.__about__ import __version__

# Allows user to specify the array module via an environment variable.
ARRAY_MODULE = os.environ.get("ARRAY_MODULE")
if ARRAY_MODULE is not None:
    if ARRAY_MODULE == "numpy":
        import numpy as xp
        from scipy import sparse

    elif ARRAY_MODULE == "cupy":
        # Attempt to import cupy, defaulting to numpy if it fails.
        try:
            import cupy as xp
            from cupyx.scipy import sparse

            # Check if cupy is actually working. This could still raise
            # a cudaErrorInsufficientDriver error or something.
            xp.abs(1)

        except Exception as e:
            warn(f"'cupy' is unavailable or not working, defaulting to 'numpy'. ({e})")
            import numpy as xp
            from scipy import sparse

    else:
        raise ValueError(f"Unrecognized ARRAY_MODULE '{ARRAY_MODULE}'")

else:
    # If the user does not specify the array module, prioritize cupy but
    # default to numpy if cupy is not available or not working.
    try:
        import cupy as xp
        from cupyx.scipy import sparse

        # Check if cupy is actually working. This could still raise
        # a cudaErrorInsufficientDriver error or something.
        xp.abs(1)

    except Exception as e:
        warn(f"'cupy' is unavailable or not working, defaulting to 'numpy'. ({e})")
        import numpy as xp
        from scipy import sparse

# Some type aliases for the array module.
_ScalarType = TypeVar("ScalarType", bound=xp.generic, covariant=True)
_DType = xp.dtype[_ScalarType]
NDArray: TypeAlias = xp.ndarray[Any, _DType]


__all__ = ["__version__", "xp", "sparse", "NDArray", "ArrayLike"]
