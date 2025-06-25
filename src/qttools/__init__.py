# Copyright (c) 2024-2025 ETH Zurich and the authors of the qttools package.

import os
import warnings
from typing import TypeAlias, TypeVar, Union

import numpy as np
from mpi4py.MPI import COMM_WORLD as global_comm
from numpy.typing import ArrayLike

from qttools.__about__ import __version__


def strtobool(s: str, default: bool | None = None) -> bool:
    """Convert a string to a boolean."""
    if s is None and default is not None:
        return default
    elif s.lower() in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif s.lower() in ("n", "no", "f", "false", "off", "0"):
        return False
    if default is None:
        raise ValueError(f"Invalid truth value {s=}.")

    warnings.warn(f"Invalid truth value {s=}. Defaulting to {default=}.")

    return default


# Suppress warnings from the jit module if not rank 0.
if global_comm.rank != 0:
    warnings.filterwarnings(
        action="ignore",
        category=FutureWarning,
        module=r".*jit",
    )


# Allows user to specify the array module via an environment variable.
QTX_ARRAY_MODULE = os.getenv("QTX_ARRAY_MODULE", "cupy")
if QTX_ARRAY_MODULE == "numpy":
    import numpy as xp
    from scipy import sparse

elif QTX_ARRAY_MODULE == "cupy":
    # Attempt to import cupy, defaulting to numpy if it fails.
    try:
        import cupy as xp
        from cupyx.scipy import sparse

        # Check if cupy is actually working. This could still raise
        # a cudaErrorInsufficientDriver error or something.
        xp.abs(1)

    except Exception as e:
        if global_comm.rank == 0:
            warnings.warn(
                f"'cupy' is unavailable or not working, defaulting to 'numpy'. ({e})",
            )
        import numpy as xp
        from scipy import sparse

else:
    raise ValueError(f"Unrecognized ARRAY_MODULE '{QTX_ARRAY_MODULE}'")

# TODO: adapt testing suite to test both JIT and non-JIT versions
QTX_USE_CUPY_JIT = strtobool(os.getenv("QTX_USE_CUPY_JIT"), default=True)

# Some type aliases for the array module.
# NOTE: CuPy is currently not type-annotated (see https://github.com/cupy/cupy/pull/9148), so we use numpy's types.

_ScalarType = TypeVar("_ScalarType", bound=np.generic, covariant=True)
NDArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[_ScalarType]]
IntType: TypeAlias = Union[np.int32, np.int64]
FloatType: TypeAlias = Union[
    np.float16, np.float32, np.float64, np.complex64, np.complex128
]

__all__ = [
    "__version__",
    "xp",
    "sparse",
    "ArrayLike",
    "NDArray",
    "IntType",
    "FloatType",
]
