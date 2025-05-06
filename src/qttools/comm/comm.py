# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, xp
from qttools.utils.gpu_utils import (
    empty_like_pinned,
    get_any_location,
    get_array_module_name,
    synchronize_device,
)


def _check_gpu_aware_mpi() -> bool:
    """Checks if the MPI implementation is GPU-aware.

    This is done by inspecting the MPI info object for the presence of
    the "gpu" memory allocation kind.

    See [here](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node279.htm)
    for more info.

    On Cray systems, the check is done by inspecting the MPI library
    version string.

    Returns
    -------
    bool
        True if the MPI implementation is GPU-aware on all ranks, False
        otherwise.

    """
    info = global_comm.Get_info()
    local_gpu_aware = (
        "gpu" in info.get("mpi_memory_alloc_kinds", "")
        or "CRAY MPICH" in MPI.Get_library_version()
    )
    local_gpu_aware = np.array(local_gpu_aware, dtype=bool)
    gpu_aware = np.empty_like(local_gpu_aware, dtype=bool)
    global_comm.Allreduce(local_gpu_aware, gpu_aware, op=MPI.LAND)
    return bool(gpu_aware)


def _check_bufs_aliased(sendbuf: NDArray, recvbuf: NDArray) -> bool:
    """Checks if the send and receive buffers are aliased.

    This is done by checking if the memory addresses of the two buffers
    are the same.

    Parameters
    ----------
    sendbuf : NDArray
        The send buffer.
    recvbuf : NDArray
        The receive buffer.

    Returns
    -------
    bool
        True if the buffers are aliased, False otherwise.

    """
    if get_array_module_name(sendbuf) == "cupy":
        sendbuf_ptr = sendbuf.data.ptr
        recvbuf_ptr = recvbuf.data.ptr

    elif get_array_module_name(sendbuf) == "numpy":
        sendbuf_ptr = sendbuf.ctypes.data
        recvbuf_ptr = recvbuf.ctypes.data

    else:
        raise ValueError(f"Unsupported array module: {get_array_module_name(sendbuf)}")

    # Check if the two memory regions overlap.
    if sendbuf_ptr == recvbuf_ptr:
        return True
    if sendbuf_ptr < recvbuf_ptr:
        return sendbuf_ptr + sendbuf.nbytes > recvbuf_ptr

    return recvbuf_ptr + recvbuf.nbytes > sendbuf_ptr


GPU_AWARE_MPI = _check_gpu_aware_mpi()


_backends = ("nccl", "host_mpi", "device_mpi")

_default_config = {
    "all_to_all": "host_mpi",
    "all_gather": "host_mpi",
    "all_reduce": "host_mpi",
    "bcast": "host_mpi",
}

_mpi_ops = {
    "sum": MPI.SUM,
    "prod": MPI.PROD,
    "max": MPI.MAX,
    "min": MPI.MIN,
}


def pad_buffer(buffer: NDArray, global_size: int, comm_size: int, axis: int) -> NDArray:
    """Pads the given buffer to the given global size.
    Parameters
    ----------
    buffer : NDArray
        The buffer to pad.
    global_size : int
        The global size including padding of the buffer along the given axis.
    comm_size : int
        The size of the communicator.
    axis : int
        The axis along which to pad the buffer.
    Returns
    -------
    NDArray
        The padded buffer.
    """

    padding_width = global_size // comm_size - buffer.shape[axis]

    padding = [(0, 0) if i != axis else (0, padding_width) for i in range(buffer.ndim)]

    buffer = xp.pad(buffer, padding)
    return buffer


class _SubCommunicator:
    """A class that handles communication for a subset of ranks.

    Parameters
    ----------
    mpi_comm : MPI.Comm
        The MPI communicator to use.
    config : dict
        The configuration for the communication backend. The keys
        are the names of the communication operations and the values
        are the backends to use. The available backends are "nccl",
        "host_mpi", and "device_mpi". The default is "host_mpi".

    """

    def __init__(self, mpi_comm: MPI.Comm, config: dict):
        """Initializes the communication backend."""
        self._validate_config(config)
        self._config = config.copy()

        self.rank = mpi_comm.rank
        self.size = mpi_comm.size

        self._mpi_comm = mpi_comm

        if "nccl" in config.values():
            self._init_nccl()

    def _validate_config(self, config: dict):
        """Validate the configuration for the communication backend."""
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary.")

        for key, value in config.items():
            if key not in _default_config:
                raise ValueError(f"Invalid configuration key: {key}")

            if value not in _backends:
                raise ValueError(
                    f"Invalid backend: {value}. Must be one of {_backends}."
                )

            if value != "device_mpi" and xp.__name__ == "numpy":
                raise ValueError(
                    f"Backend '{value}' is not available with NumPy."
                    "Use 'device_mpi' instead."
                )
            if value == "device_mpi" and xp.__name__ == "cupy" and not GPU_AWARE_MPI:
                raise ValueError(
                    f"Backend '{value}' is not available with this MPI implementation."
                )

    def _init_nccl(self):
        """Initializes the NCCL backend."""
        if not xp.__name__ == "cupy":
            raise RuntimeError("NCCL is only available with CuPy.")

        from cupy.cuda import nccl

        if not nccl.available:
            raise RuntimeError("NCCL is not available.")

        from cupyx import distributed
        from cupyx.distributed import _store
        from cupyx.distributed._comm import _Backend

        # NOTE: We try to emulate the behavior of the NCCL backend in
        # the cupyx.distributed package here. Unfortunately, the NCCL
        # backend will always use the global communicator, which is not
        # what we want.
        nccl_comm = distributed.NCCLBackend.__new__(distributed.NCCLBackend)
        _Backend.__init__(
            nccl_comm,
            global_comm.size,
            global_comm.rank,
            _store._DEFAULT_HOST,
            port=_store._DEFAULT_PORT,
        )

        nccl_comm._use_mpi = True

        nccl_comm._n_devices = self._mpi_comm.size
        nccl_comm._mpi_comm = self._mpi_comm
        nccl_comm._mpi_rank = self._mpi_comm.rank
        nccl_comm._mpi_comm.barrier()

        nccl_block_id = None
        if nccl_comm._mpi_rank == 0:
            nccl_block_id = nccl.get_unique_id()
        nccl_block_id = self._mpi_comm.bcast(nccl_block_id, root=0)

        nccl_comm._comm = nccl.NcclCommunicator(
            self._mpi_comm.size, nccl_block_id, self._mpi_comm.rank
        )

        self._nccl_comm = nccl_comm

    def _check_bufs_consistent(self, sendbuf: NDArray, recvbuf: NDArray):
        """Checks that the send and receive buffers are in the correct place."""
        if (
            get_array_module_name(sendbuf) != xp.__name__
            or get_array_module_name(recvbuf) != xp.__name__
        ):
            raise ValueError(
                f"sendbuf and recvbuf must be {xp.__name__} arrays, but "
                f"got {get_array_module_name(sendbuf)} and {get_array_module_name(recvbuf)}."
            )

    def all_to_all(
        self, sendbuf: NDArray, recvbuf: NDArray, backend: str | None = None
    ):
        """Performs all-to-all communication."""
        if backend is None:
            backend = self._config["all_to_all"]
        elif backend not in _backends:
            raise ValueError(f"Invalid backend: {backend}. Must be one of {_backends}.")

        self._check_bufs_consistent(sendbuf, recvbuf)

        if _check_bufs_aliased(sendbuf, recvbuf):
            raise ValueError("sendbuf and recvbuf must not be aliased.")

        if sendbuf.size != recvbuf.size:
            raise ValueError(
                f"sendbuf and recvbuf must have the same size, but got {sendbuf.size} and {recvbuf.size}."
            )

        synchronize_device()
        if backend == "nccl":
            self._nccl_comm.all_to_all(sendbuf, recvbuf)

        elif backend == "device_mpi":
            self._mpi_comm.Alltoall(sendbuf, recvbuf)

        elif backend == "host_mpi":

            _sendbuf_host = get_any_location(
                sendbuf,
                output_module="numpy",
                use_pinned_memory=True,
            )

            synchronize_device()
            _recvbuf_host = empty_like_pinned(recvbuf)
            self._mpi_comm.Alltoall(_sendbuf_host, _recvbuf_host)

            recvbuf[:] = get_any_location(
                _recvbuf_host,
                output_module="cupy",
                use_pinned_memory=True,
            )

        synchronize_device()

    def all_gather(
        self, sendbuf: NDArray, recvbuf: NDArray, backend: str | None = None
    ):
        """Performs all-gather communication."""
        if backend is None:
            backend = self._config["all_gather"]
        elif backend not in _backends:
            raise ValueError(f"Invalid backend: {backend}. Must be one of {_backends}.")

        self._check_bufs_consistent(sendbuf, recvbuf)

        if sendbuf.size * self.size != recvbuf.size:
            raise ValueError(
                "sendbuf must be the same size as recvbuf divided by the number of ranks. "
                f"Got {sendbuf.size=} and {recvbuf.size=}."
            )

        synchronize_device()
        if backend == "nccl":
            # NOTE: The count argument is actually unused in the NCCL
            # backend but it is still a required parameter.
            self._nccl_comm.all_gather(sendbuf, recvbuf, count=None)

        elif backend == "device_mpi":
            aliased = _check_bufs_aliased(sendbuf, recvbuf)
            self._mpi_comm.Allgather(sendbuf.copy() if aliased else sendbuf, recvbuf)

        elif backend == "host_mpi":

            _sendbuf_host = get_any_location(
                sendbuf,
                output_module="numpy",
                use_pinned_memory=True,
            )

            synchronize_device()
            _recvbuf_host = empty_like_pinned(recvbuf)
            self._mpi_comm.Allgather(_sendbuf_host, _recvbuf_host)

            recvbuf[:] = get_any_location(
                _recvbuf_host,
                output_module="cupy",
                use_pinned_memory=True,
            )

        synchronize_device()

    def all_reduce(
        self,
        sendbuf: NDArray,
        recvbuf: NDArray,
        op: str = "sum",
        backend: str | None = None,
    ):
        """Performs all-reduce communication."""
        if backend is None:
            backend = self._config["all_reduce"]
        elif backend not in _backends:
            raise ValueError(f"Invalid backend: {backend}. Must be one of {_backends}.")

        self._check_bufs_consistent(sendbuf, recvbuf)

        if sendbuf.size != recvbuf.size:
            raise ValueError(
                f"sendbuf and recvbuf must have the same size, but got {sendbuf.size} and {recvbuf.size}."
            )

        if op not in _mpi_ops:
            raise ValueError(
                f"Invalid operation '{op}'. Must be one of {_mpi_ops.keys()}."
            )
        synchronize_device()
        if backend == "nccl":
            self._nccl_comm.all_reduce(sendbuf, recvbuf, op=op)
        elif backend == "device_mpi":
            aliased = _check_bufs_aliased(sendbuf, recvbuf)
            self._mpi_comm.Allreduce(
                sendbuf.copy() if aliased else sendbuf, recvbuf, op=_mpi_ops[op]
            )
        elif backend == "host_mpi":
            _sendbuf_host = get_any_location(
                sendbuf,
                output_module="numpy",
                use_pinned_memory=True,
            )

            synchronize_device()
            _recvbuf_host = empty_like_pinned(recvbuf)
            self._mpi_comm.Allreduce(_sendbuf_host, _recvbuf_host, op=_mpi_ops[op])

            recvbuf[:] = get_any_location(
                _recvbuf_host,
                output_module="cupy",
                use_pinned_memory=True,
            )

        synchronize_device()

    def bcast(self, sendrecvbuf: NDArray, root: int = 0, backend: str | None = None):
        """Perform broadcast communication."""
        if backend is None:
            backend = self._config["bcast"]
        elif backend not in _backends:
            raise ValueError(f"Invalid backend: {backend}. Must be one of {_backends}.")

        synchronize_device()
        if backend == "nccl":
            self._nccl_comm.broadcast(sendrecvbuf, root=root)
        elif backend == "device_mpi":
            self._mpi_comm.Bcast(sendrecvbuf, root=root)
        elif backend == "host_mpi":
            _sendrecvbuf_host = get_any_location(
                sendrecvbuf,
                output_module="numpy",
                use_pinned_memory=True,
            )

            synchronize_device()
            self._mpi_comm.Bcast(_sendrecvbuf_host, root=root)

            sendrecvbuf[:] = get_any_location(
                _sendrecvbuf_host,
                output_module="cupy",
                use_pinned_memory=True,
            )
        synchronize_device()

    def barrier(self):
        """Perform barrier synchronization."""
        self._mpi_comm.barrier()

    def all_gather_v(
        self,
        sendbuf: NDArray,
        axis: int,
        mask: NDArray | None = None,
    ) -> NDArray:
        """Gathers the sendbuf from all ranks and returns the result.

        Parameters
        ----------
        comm : _SubCommunicator
            The communicator to use.
        sendbuf : NDArray
            The buffer to send.
        axis : int
            The axis along which to pad the buffer.
        mask : NDArray, optional
            The mask to use for gathering the buffer. If None, the buffer will be automatically padded.

        Returns
        -------
        NDArray
            The gathered buffer.
        """

        if mask is not None:
            if mask.size // self.size < sendbuf.size:
                raise ValueError(
                    f"The mask is too small for the sendbuf: {mask.size // self.size} < {sendbuf.size}."
                )
            global_size = mask.size
        else:
            counts = np.zeros(self.size, dtype=xp.int32)
            self.all_gather(np.array(sendbuf.shape[axis]), counts, backend="device_mpi")
            global_size = np.max(counts) * self.size
            mask = xp.zeros(global_size, dtype=bool)
            for i in range(self.size):
                mask[np.max(counts) * i : np.max(counts) * i + counts[i]] = True

        if mask.ndim > 1:
            raise ValueError("mask must be 1D or None")

        sendbuf = pad_buffer(sendbuf, global_size, self.size, axis)

        sendbuf = xp.ascontiguousarray(xp.moveaxis(sendbuf, axis, 0))
        recvbuf = xp.empty((global_size, *sendbuf.shape[1:]), dtype=sendbuf.dtype)
        self.all_gather(sendbuf, recvbuf)
        recvbuf = xp.moveaxis(recvbuf, 0, axis)

        indices = xp.where(mask)[0]
        return xp.take(
            recvbuf,
            indices,
            axis=axis,
        )


class QuatrexCommunicator:
    """A communicator that handles all block and stack communications.

    This class is a singleton and should be used as such. It is
    initialized with the global communicator and can be configured
    with the block and stack communicators.

    Attributes
    ----------
    block : SubCommunicator
        The block communicator.
    stack : SubCommunicator
        The stack communicator.

    """

    _instance = None
    _is_configured = False

    size = None
    rank = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QuatrexCommunicator, cls).__new__(cls)

        return cls._instance

    def configure(
        self,
        block_comm_size: int,
        block_comm_config: dict,
        stack_comm_config: dict,
    ):
        """Configures the communicator.

        Parameters
        ----------
        block_comm_size : int
            The size of the block communicator.
        block_comm_config : dict
            The configuration for the block sub-communicator.
        stack_comm_config : dict
            The configuration for the stack sub-communicator.

        Raises
        -------
        RuntimeError
            If the communicator is already configured.
        ValueError
            If the block communicator size is not a multiple of the
            total number of ranks.

        """

        if self._is_configured:
            raise RuntimeError("Communicator is already configured.")

        if global_comm.size % block_comm_size != 0:
            raise ValueError(
                f"Total number of ranks must be a multiple of {block_comm_size=}"
            )

        if block_comm_size <= 0:
            raise ValueError("Block communicator size must be greater than 0.")

        if block_comm_size > global_comm.size:
            raise ValueError(
                f"Block communicator size {block_comm_size} cannot be greater than the total number of ranks {global_comm.size}."
            )

        self.rank = global_comm.rank
        self.size = global_comm.size

        # if block_comm_size == 1:
        #     # No domain decomposition, use the global communicator.
        #     self.stack = SubCommunicator(global_comm, stack_comm_config)
        #     self.block = None
        #     self._is_configured = True
        #     return

        color = global_comm.rank // block_comm_size
        key = global_comm.rank % block_comm_size

        block_comm = global_comm.Split(color=color, key=key)
        stack_comm = global_comm.Split(color=key, key=color)

        self.block = _SubCommunicator(block_comm, block_comm_config)
        self.stack = _SubCommunicator(stack_comm, stack_comm_config)

        self._is_configured = True

    def barrier(self):
        """Perform barrier synchronization."""
        global_comm.Barrier()
