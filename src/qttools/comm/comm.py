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


def check_gpu_aware_mpi() -> bool:
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


GPU_AWARE_MPI = check_gpu_aware_mpi()


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
        self.nccl_comm = distributed.NCCLBackend.__new__(distributed.NCCLBackend)
        _Backend.__init__(
            self.nccl_comm,
            global_comm.size,
            global_comm.rank,
            _store._DEFAULT_HOST,
            port=_store._DEFAULT_PORT,
        )

        self.nccl_comm._use_mpi = True

        self.nccl_comm._n_devices = self._mpi_comm.size
        self.nccl_comm._mpi_comm = self._mpi_comm
        self.nccl_comm._mpi_rank = self._mpi_comm.rank
        self.nccl_comm._mpi_comm.barrier()

        nccl_block_id = None
        if self.nccl_comm._mpi_rank == 0:
            nccl_block_id = nccl.get_unique_id()
        nccl_block_id = self._mpi_comm.bcast(nccl_block_id, root=0)

        self.nccl_comm._comm = nccl.NcclCommunicator(
            self._mpi_comm.size, nccl_block_id, self._mpi_comm.rank
        )

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

    def all_to_all(self, sendbuf: NDArray, recvbuf: NDArray):
        """Performs all-to-all communication."""
        backend = self._config["all_to_all"]

        self._check_bufs_consistent(sendbuf, recvbuf)

        if sendbuf.size != recvbuf.size:
            raise ValueError(
                f"sendbuf and recvbuf must have the same size, but got {sendbuf.size} and {recvbuf.size}."
            )

        synchronize_device()
        if backend == "nccl":
            self.nccl_comm.all_to_all(sendbuf, recvbuf)

        elif backend == "device_mpi":
            self._mpi_comm.Alltoall(sendbuf, recvbuf)

        elif backend == "host_mpi":

            _sendbuf_host = get_any_location(
                sendbuf,
                output_module="numpy",
                use_pinned_memory=True,
            )

            synchronize_device()
            _recbuf_host = empty_like_pinned(_sendbuf_host)
            self._mpi_comm.Alltoall(_sendbuf_host, _recbuf_host)

            recvbuf[:] = get_any_location(
                _recbuf_host,
                output_module="cupy",
                use_pinned_memory=True,
            )

        synchronize_device()

    def all_gather(self, sendbuf: NDArray, recvbuf: NDArray):
        """Performs all-gather communication."""
        backend = self._config["all_gather"]

        self._check_bufs_consistent(sendbuf, recvbuf)

        if sendbuf.size * self.size != recvbuf.size:
            raise ValueError(
                "sendbuf must be the same size as recvbuf divided by the number of ranks."
            )

        synchronize_device()
        if backend == "nccl":
            # NOTE: The count argument is actually unused in the NCCL
            # backend but it is still a required parameter.
            self.nccl_comm.all_gather(sendbuf, recvbuf, count=None)

        elif backend == "device_mpi":
            self._mpi_comm.Allgather(sendbuf, recvbuf)

        elif backend == "host_mpi":

            _sendbuf_host = get_any_location(
                sendbuf,
                output_module="numpy",
                use_pinned_memory=True,
            )

            synchronize_device()
            _recbuf_host = empty_like_pinned(_sendbuf_host)
            self._mpi_comm.Allgather(_sendbuf_host, _recbuf_host)

            recvbuf[:] = get_any_location(
                _recbuf_host,
                output_module="cupy",
                use_pinned_memory=True,
            )

        synchronize_device()

    def all_reduce(self, sendbuf: NDArray, recvbuf: NDArray, op: str = "sum"):
        """Performs all-reduce communication."""
        backend = self._config["all_reduce"]

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
            self.nccl_comm.all_reduce(sendbuf, recvbuf, op=op)
        elif backend == "device_mpi":
            self._mpi_comm.Allreduce(sendbuf, recvbuf, op=_mpi_ops[op])
        elif backend == "host_mpi":
            _sendbuf_host = get_any_location(
                sendbuf,
                output_module="numpy",
                use_pinned_memory=True,
            )

            synchronize_device()
            _recbuf_host = empty_like_pinned(_sendbuf_host)
            self._mpi_comm.Allreduce(_sendbuf_host, _recbuf_host, op=_mpi_ops[op])

            recvbuf[:] = get_any_location(
                _recbuf_host,
                output_module="cupy",
                use_pinned_memory=True,
            )

        synchronize_device()

    def bcast(self, sendrecvbuf: NDArray, root: int = 0):
        """Perform broadcast communication."""
        backend = self._config["bcast"]

        synchronize_device()
        if backend == "nccl":
            self.nccl_comm.broadcast(sendrecvbuf, root=root)
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


class Communicator:
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

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Communicator, cls).__new__(cls)

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
