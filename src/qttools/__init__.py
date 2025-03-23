# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import os
from typing import Any, TypeAlias, TypeVar
from warnings import warn

from mpi4py.MPI import COMM_WORLD as global_comm
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

            ARRAY_MODULE = "numpy"

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

        ARRAY_MODULE = "numpy"

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

if xp.__name__ == "cupy":

    from cupy.cuda import nccl

    if nccl.available:
        NCCL_AVAILABLE = True

        from cupyx import distributed

        # TODO: This will probably not work with communicators other than
        # MPI.COMM_WORLD. We need to fix this if we want to use other
        # communicators.
        nccl_comm = distributed.NCCLBackend(
            global_comm.size, global_comm.rank, use_mpi=True
        )

block_comm = None
stack_comm = None
nccl_block_comm = None
nccl_stack_comm = None
BLOCK_COMM_SIZE = os.environ.get("BLOCK_COMM_SIZE", None)
if BLOCK_COMM_SIZE is not None:
    try:
        BLOCK_COMM_SIZE = int(BLOCK_COMM_SIZE)
    except ValueError:
        warn(f"Invalid BLOCK_COMM_SIZE '{BLOCK_COMM_SIZE}', defaulting to None.")
        BLOCK_COMM_SIZE = None

    if global_comm.size % BLOCK_COMM_SIZE != 0:
        raise ValueError(
            f"Total number of ranks must be a multiple of {BLOCK_COMM_SIZE=}"
        )

    # Compute the color and key for each rank.
    color = global_comm.rank % (global_comm.size // BLOCK_COMM_SIZE)
    key = global_comm.rank // (global_comm.size // BLOCK_COMM_SIZE)

    # Split the communicator twice.
    block_comm = global_comm.Split(color=color, key=key)
    stack_comm = global_comm.Split(color=key, key=color)

    # Absolute hack to try to get two split NCCL communicators. This is
    # the way the communicators get initialized with use_mpi=True but we
    # need to force it not to use MPI.COMM_WORLD.
    if NCCL_AVAILABLE:
        # Initialize the block communicator.
        nccl_block_comm = distributed.NCCLBackend(
            global_comm.size, global_comm.rank, use_mpi=True
        )
        nccl_block_comm._n_devices = block_comm.size
        nccl_block_comm._mpi_comm = block_comm
        nccl_block_comm._mpi_rank = nccl_block_comm._mpi_comm.Get_rank()
        nccl_block_comm._mpi_comm.Barrier()
        nccl_block_id = None
        if nccl_block_comm._mpi_rank == 0:
            nccl_block_id = nccl.get_unique_id()
        nccl_block_id = nccl_block_comm._mpi_comm.bcast(nccl_block_id, root=0)

        nccl_block_comm._comm = nccl.NcclCommunicator(
            block_comm.size, nccl_block_id, block_comm.rank
        )

        # Initialize the stack communicator.
        nccl_stack_comm = distributed.NCCLBackend(
            global_comm.size, global_comm.rank, use_mpi=True
        )
        nccl_stack_comm._n_devices = stack_comm.size
        nccl_stack_comm._mpi_comm = stack_comm
        nccl_stack_comm._mpi_rank = nccl_stack_comm._mpi_comm.Get_rank()
        nccl_stack_comm._mpi_comm.Barrier()
        nccl_stack_id = None
        if nccl_stack_comm._mpi_rank == 0:
            nccl_stack_id = nccl.get_unique_id()
        nccl_stack_id = nccl_stack_comm._mpi_comm.bcast(nccl_stack_id, root=0)

        nccl_stack_comm._comm = nccl.NcclCommunicator(
            stack_comm.size, nccl_stack_id, stack_comm.rank
        )

__all__ = [
    "__version__",
    "xp",
    "host_xp",
    "pinned_xp",
    "sparse",
    "NDArray",
    "ArrayLike",
    "USE_CUPY_JIT",
    "block_comm",
    "stack_comm",
    "NCCL_AVAILABLE",
    "nccl_comm",
    "nccl_block_comm",
    "nccl_stack_comm",
]
