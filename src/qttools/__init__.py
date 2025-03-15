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

        host_xp = xp
        pinned_xp = None

    elif ARRAY_MODULE == "cupy":
        # Attempt to import cupy, defaulting to numpy if it fails.
        try:
            import cupy as xp
            import cupyx as pinned_xp
            import numpy as host_xp
            from cupyx.scipy import sparse

            # Check if cupy is actually working. This could still raise
            # a cudaErrorInsufficientDriver error or something.
            xp.abs(1)

        except Exception as e:
            warn(f"'cupy' is unavailable or not working, defaulting to 'numpy'. ({e})")
            import numpy as xp
            from scipy import sparse

            host_xp = xp
            pinned_xp = None

    else:
        raise ValueError(f"Unrecognized ARRAY_MODULE '{ARRAY_MODULE}'")

else:
    # If the user does not specify the array module, prioritize cupy but
    # default to numpy if cupy is not available or not working.
    try:
        import cupy as xp
        import cupyx as pinned_xp
        import numpy as host_xp
        from cupyx.scipy import sparse

        # Check if cupy is actually working. This could still raise
        # a cudaErrorInsufficientDriver error or something.
        xp.abs(1)

    except Exception as e:
        warn(f"'cupy' is unavailable or not working, defaulting to 'numpy'. ({e})")
        import numpy as xp
        from scipy import sparse

        host_xp = xp
        pinned_xp = None

# TODO: adapt testing suite to test both JIT and non-JIT versions
# Implemented with a env variable to allow for easy switching
USE_CUPY_JIT = os.environ.get("USE_CUPY_JIT", "true").lower()
if USE_CUPY_JIT in ("y", "yes", "t", "true", "on", "1"):
    USE_CUPY_JIT = True
elif USE_CUPY_JIT in ("n", "no", "f", "false", "off", "0"):
    USE_CUPY_JIT = False
else:
    warn(f"Invalid truth value {USE_CUPY_JIT=}. Defaulting to 'true'.")
    USE_CUPY_JIT = True

# Some type aliases for the array module.
_ScalarType = TypeVar("ScalarType", bound=xp.generic, covariant=True)
_DType = xp.dtype[_ScalarType]
NDArray: TypeAlias = xp.ndarray[Any, _DType]

# Check if NCCL is available.
NCCL_AVAILABLE = False
nccl_comm = None

# TODO: This is a temporary solution. We need to find a better way to
# handle this. Very specific to mpich.
CUDA_AWARE_MPI = os.environ.get("MPICH_GPU_SUPPORT_ENABLED", "false").lower()
if CUDA_AWARE_MPI in ("y", "yes", "t", "true", "on", "1"):
    CUDA_AWARE_MPI = True
elif CUDA_AWARE_MPI in ("n", "no", "f", "false", "off", "0"):
    CUDA_AWARE_MPI = False
else:
    warn(f"Invalid truth value {CUDA_AWARE_MPI=}. Defaulting to 'false'.")
    CUDA_AWARE_MPI = False

if xp.__name__ == "cupy":

    from cupy.cuda import nccl

    if nccl.available:
        NCCL_AVAILABLE = True

        from cupyx import distributed
        from mpi4py.MPI import COMM_WORLD as mpi_comm

        # TODO: This will probably not work with communicators other than
        # MPI.COMM_WORLD. We need to fix this if we want to use other
        # communicators.
        nccl_comm = distributed.NCCLBackend(mpi_comm.size, mpi_comm.rank, use_mpi=True)


__all__ = [
    "__version__",
    "xp",
    "host_xp",
    "pinned_xp",
    "sparse",
    "NDArray",
    "ArrayLike",
    "USE_CUPY_JIT",
    "NCCL_AVAILABLE",
    "nccl_comm",
]
