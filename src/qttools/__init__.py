# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import math
import os
from typing import Any, TypeAlias, TypeVar
from warnings import warn

from mpi4py.MPI import COMM_WORLD as global_comm
from numpy.typing import ArrayLike

from qttools.__about__ import __version__


def strtobool(s: str, default: bool | None = None) -> bool:
    """Convert a string to a boolean."""
    if s.lower() in ("y", "yes", "t", "true", "on", "1"):
        return True
    if s.lower() in ("n", "no", "f", "false", "off", "0"):
        return False

    if default is None:
        raise ValueError(f"Invalid truth value {s=}.")

    warn(f"Invalid truth value {s=}. Defaulting to {default=}.")
    return default


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


# Some type aliases for the array module.
_ScalarType = TypeVar("ScalarType", bound=xp.generic, covariant=True)
_DType = xp.dtype[_ScalarType]
NDArray: TypeAlias = xp.ndarray[Any, _DType]

# Check if NCCL is available.
NCCL_AVAILABLE = False
nccl_comm = None

ALLTOALL_COMM_TYPE = os.environ.get("ALLTOALL_COMM_TYPE", "nccl").lower()
if ALLTOALL_COMM_TYPE not in ("nccl", "host_mpi", "device_mpi"):
    raise ValueError(f"Unrecognized ALLTOALL_COMM_TYPE '{ALLTOALL_COMM_TYPE}'")

OTHER_COMM_TYPE = os.environ.get("OTHER_COMM_TYPE", "nccl").lower()
if OTHER_COMM_TYPE not in ("nccl", "host_mpi", "device_mpi"):
    raise ValueError(f"Unrecognized OTHER_COMM_TYPE '{OTHER_COMM_TYPE}'")

nccl_comm = None
if xp.__name__ == "cupy":

    from cupy.cuda import nccl

    if nccl.available:
        NCCL_AVAILABLE = True

        from cupy.cuda import nccl
        from cupyx import distributed
        from cupyx.distributed import _store
        from cupyx.distributed._comm import _Backend


#         # TODO: This will probably not work with communicators other than
#         # MPI.COMM_WORLD. We need to fix this if we want to use other
#         # communicators.
#         nccl_comm = distributed.NCCLBackend(
#             global_comm.size, global_comm.rank, use_mpi=True
#         )

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

    # # Compute the color and key for each rank.
    # color = global_comm.rank % (global_comm.size // BLOCK_COMM_SIZE)
    # key = global_comm.rank // (global_comm.size // BLOCK_COMM_SIZE)

    # # Split the communicator twice.
    # block_comm = global_comm.Split(color=color, key=key)
    # stack_comm = global_comm.Split(color=key, key=color)

    # Block communicator
    # BLOCK_COMM_SIZE in every block
    # GLOBAL_COMM_SIZE // BLOCK_COMM_SIZE blocks
    block_color = global_comm.rank // BLOCK_COMM_SIZE
    block_key = global_comm.rank % BLOCK_COMM_SIZE
    block_comm = global_comm.Split(color=block_color, key=block_key)
    # Stack communicator
    # BLOCK_COMM_SIZE stacks
    # GLOBAL_COMM_SIZE // BLOCK_COMM_SIZE ranks in every stack
    stack_color = global_comm.rank % BLOCK_COMM_SIZE
    stack_key = global_comm.rank // BLOCK_COMM_SIZE
    stack_comm = global_comm.Split(color=stack_color, key=stack_key)

    # block_comm = global_comm.Split(color=key, key=color)
    # stack_comm = global_comm.Split(color=color, key=key)

    # Absolute hack to try to get two split NCCL communicators. This is
    # the way the communicators get initialized with use_mpi=True but we
    # need to force it not to use MPI.COMM_WORLD.
    if NCCL_AVAILABLE:
        # Initialize the block communicator.

        nccl_block_comm = distributed.NCCLBackend.__new__(distributed.NCCLBackend)
        _Backend.__init__(
            nccl_block_comm,
            global_comm.size,
            global_comm.rank,
            _store._DEFAULT_HOST,
            port=_store._DEFAULT_PORT,
        )

        nccl_block_comm.use_mpi = True

        nccl_block_comm = distributed.NCCLBackend(
            global_comm.size, global_comm.rank, use_mpi=True
        )

        nccl_block_comm._n_devices = block_comm.size
        nccl_block_comm._mpi_comm = block_comm
        nccl_block_comm._mpi_rank = nccl_block_comm._mpi_comm.Get_rank()
        nccl_block_comm._mpi_comm.Barrier()
        nccl_block_id = None

        # round to get a multiple of BLOCK_COMM_SIZE
        group_size = max(
            BLOCK_COMM_SIZE,
            global_comm.size // 8
            + BLOCK_COMM_SIZE
            - (global_comm.size // 8) % BLOCK_COMM_SIZE,
        )

        print(
            f"Initializing nccl_block_comm with group size {group_size} for {global_comm.size=}",
            flush=True,
        )

        for i in range(math.ceil(global_comm.size / group_size)):

            if global_comm.rank // group_size == i:
                print(
                    f"Initializing nccl_block_comm {global_comm.rank} in group {i}",
                    flush=True,
                )

                if nccl_block_comm._mpi_rank == 0:
                    nccl_block_id = nccl.get_unique_id()
                nccl_block_id = nccl_block_comm._mpi_comm.bcast(nccl_block_id, root=0)

                nccl_block_comm._comm = nccl.NcclCommunicator(
                    block_comm.size, nccl_block_id, block_comm.rank
                )

            global_comm.Barrier()

        # # Initialize the stack communicator.
        # nccl_stack_comm = distributed.NCCLBackend(
        #     global_comm.size, global_comm.rank, use_mpi=True
        # )
        # nccl_stack_comm._n_devices = stack_comm.size
        # nccl_stack_comm._mpi_comm = stack_comm
        # nccl_stack_comm._mpi_rank = nccl_stack_comm._mpi_comm.Get_rank()
        # nccl_stack_comm._mpi_comm.Barrier()
        # nccl_stack_id = None
        # if nccl_stack_comm._mpi_rank == 0:
        #     nccl_stack_id = nccl.get_unique_id()
        # nccl_stack_id = nccl_stack_comm._mpi_comm.bcast(nccl_stack_id, root=0)

        # nccl_stack_comm._comm = nccl.NcclCommunicator(
        #     stack_comm.size, nccl_stack_id, stack_comm.rank
        # )

__all__ = [
    "__version__",
    "xp",
    "host_xp",
    "pinned_xp",
    "sparse",
    "NDArray",
    "ArrayLike",
    "block_comm",
    "stack_comm",
    "NCCL_AVAILABLE",
    "nccl_comm",
    "nccl_block_comm",
    "nccl_stack_comm",
]
