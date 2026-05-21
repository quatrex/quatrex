# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

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

if xp.__name__ == "cupy":
    from cupy.cuda import nccl


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
    match or the buffers overlap.

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
    "send_recv": "host_mpi",
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
        _SubCommunicator._validate_config(config)
        self._config = _default_config.copy()
        self._config.update(config)

        self.rank = mpi_comm.rank
        self.size = mpi_comm.size

        self._mpi_comm = mpi_comm

        if "nccl" in config.values():
            self._init_nccl()

        # NOTE: One can create still very unexpected behavior by using
        # group start of both subcommunicators at once or
        # by calling externally nccl group start or end.
        # This is not something we can easily guard against.
        self._group_start_called = False

    @classmethod
    def _validate_config(cls, config: dict):
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
        if get_array_module_name(sendbuf) != get_array_module_name(recvbuf):
            raise ValueError(
                f"sendbuf and recvbuf must be of the same type, but got {sendbuf.dtype} and {recvbuf.dtype}."
            )

    def all_to_all(
        self, sendbuf: NDArray, recvbuf: NDArray, backend: str | None = None
    ) -> None:
        """Performs all-to-all communication.

        Parameters
        ----------
        sendbuf : NDArray
            The buffer to send.
        recvbuf : NDArray
            The buffer to receive.
        backend : str, optional
            The backend to use for the communication. If None, the default
            backend will be used.

        """
        if self._group_start_called:
            raise RuntimeError(
                "This cannot be called between group_start and group_end."
            )

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
    ) -> None:
        """Performs all-gather communication.

        Parameters
        ----------
        sendbuf : NDArray
            The buffer to send.
        recvbuf : NDArray
            The buffer to receive.
        backend : str, optional
            The backend to use for the communication. If None, the default
            backend will be used.

        """
        if self._group_start_called:
            raise RuntimeError(
                "This cannot be called between group_start and group_end."
            )

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
    ) -> None:
        """Performs all-reduce communication.

        Parameters
        ----------
        sendbuf : NDArray
            The buffer to send.
        recvbuf : NDArray
            The buffer to receive.
        op : str, optional
            The operation to perform. Default is "sum".
        backend : str, optional
            The backend to use for the communication. If None, the default
            backend will be used.

        """
        if self._group_start_called:
            raise RuntimeError(
                "This cannot be called between group_start and group_end."
            )

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

    def bcast(
        self, sendrecvbuf: NDArray, root: int = 0, backend: str | None = None
    ) -> None:
        """Perform broadcast communication.

        Parameters
        ----------
        sendrecvbuf : NDArray
            The buffer to send and receive.
        root : int, optional
            The rank of the root process. Default is 0.
        backend : str, optional
            The backend to use for the communication. If None, the default
            backend will be used.

        """
        if self._group_start_called:
            raise RuntimeError(
                "This cannot be called between group_start and group_end."
            )

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

    def barrier(self) -> None:
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
            The axis along which to gather the buffer.
        mask : NDArray, optional
            The mask to use for gathering the buffer. If None, the buffer will be automatically padded.

        Returns
        -------
        NDArray
            The gathered buffer.
        """
        if self._group_start_called:
            raise RuntimeError(
                "This cannot be called between group_start and group_end."
            )

        if mask is not None:
            if mask.size // self.size < sendbuf.shape[axis]:
                raise ValueError(
                    f"The mask is too small for the sendbuf: {mask.size // self.size} < {sendbuf.shape[axis]}."
                )
            global_size = mask.size
        else:
            counts = np.zeros(self.size, dtype=xp.int32)
            self.all_gather(
                np.array(sendbuf.shape[axis], dtype=xp.int32),
                counts,
                backend="device_mpi",
            )
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
        return xp.take(recvbuf, indices, axis=axis)

    def send_recv(
        self,
        sendbuf: NDArray,
        dest: int,
        recvbuf: NDArray,
        source: int,
        backend: str | None = None,
    ) -> None:
        """Performs sendrecv communication.

        Parameters
        ----------
        sendbuf : NDArray
            The buffer to send.
        dest : int
            The rank to send to.
        recvbuf : NDArray
            The buffer to receive into.
        source : int
            The rank to receive from.
        backend : str, optional
            The backend to use for the communication. If None, the default
            backend for sendrecv will be used.

        """
        if self._group_start_called:
            raise RuntimeError(
                "This cannot be called between group_start and group_end."
            )

        if backend is None:
            backend = self._config["send_recv"]
        elif backend not in _backends:
            raise ValueError(f"Invalid backend: {backend}. Must be one of {_backends}.")

        if backend == "nccl":

            nccl.groupStart()

            self._nccl_comm.send(sendbuf, dest)
            self._nccl_comm.recv(recvbuf, source)

            nccl.groupEnd()

        elif backend == "device_mpi":
            self._mpi_comm.Sendrecv(
                sendbuf=sendbuf,
                dest=dest,
                recvbuf=recvbuf,
                source=source,
            )
        elif backend == "host_mpi":
            sendbuf_host = get_any_location(
                sendbuf,
                output_module="numpy",
                use_pinned_memory=True,
            )
            recvbuf_host = empty_like_pinned(recvbuf)

            # Need to synchronize the current stream to ensure that
            # the H2D copy finished before the MPI call is made.
            synchronize_device()

            self._mpi_comm.Sendrecv(
                sendbuf=sendbuf_host,
                dest=dest,
                recvbuf=recvbuf_host,
                source=source,
            )
            recvbuf[:] = get_any_location(
                recvbuf_host,
                output_module="cupy",
                use_pinned_memory=True,
            )

    def group_start(
        self,
        backend: str,
    ) -> None:
        """Starts a group of communication operations.

        NOTE: The backend needs to specificied since it can be used
        together with any communication operation, not just sendrecv.

        Parameters
        ----------
        backend : str
            The backend to use for the communication. Must be one of "nccl",
            "device_mpi", or "host_mpi".

        """

        if self._group_start_called:
            raise RuntimeError(
                "group_start has already been called."
                "group_start and group_end must be called in pairs."
            )
        self._group_start_called = True

        if backend == "nccl":
            self._nccl_comm.groupStart()

    def group_end(
        self,
        backend: str,
        requests: list[MPI.Request] | list[None] | None = None,
    ) -> None:
        """Ends a group of communication operations.

        NOTE: The backend needs to specificied since it can be used
        together with any communication operation, not just sendrecv.

        Parameters
        ----------
        backend : str
            The backend to use for the communication. Must be one of "nccl",
            "device_mpi", or "host_mpi".
        requests : list[MPI.Request] | list[None] | None, optional
            A list of requests to wait for. In the case of the "nccl" backend,
            this must be None or a list of None, as the NCCL backend does not
            use requests. For the "device_mpi" and "host_mpi" backends, this
            must be a list of MPI.Request objects returned by non-blocking
            communication operations.

        """

        if not self._group_start_called:
            raise RuntimeError("group_start must be called before group_end.")
        self._group_start_called = False

        if backend == "nccl":
            # check that requests is None or a list of None
            if requests is not None and not all(r is None for r in requests):
                raise ValueError(
                    "requests must be None or"
                    " a list of None when using the nccl backend."
                )

            self._nccl_comm.groupStart()
        else:
            if requests is None:
                raise ValueError(
                    "requests must be provided for device_mpi and host_mpi backends."
                )

            if not all(isinstance(r, MPI.Request) for r in requests):
                raise ValueError("requests must be a list of MPI.Request objects.")

            MPI.Request.Waitall(requests)

    def send(
        self,
        buf: NDArray,
        dest: int,
        backend: str | None = None,
    ) -> None:
        """Performs non-blocking send communication.

        Parameters
        ----------
        buf : NDArray
            The buffer to send.
        dest : int
            The rank to send to.
        backend : str, optional
            The backend to use for the communication. If None, the default
            backend for sendrecv will be used.

        """

        if self._group_start_called:
            raise RuntimeError(
                "send cannot be called between group_start and group_end."
                "They should be called together with isend."
            )

        if backend is None:
            backend = self._config["send_recv"]
        elif backend not in _backends:
            raise ValueError(f"Invalid backend: {backend}. Must be one of {_backends}.")

        if backend == "nccl":
            self._nccl_comm.send(buf, dest)

        elif backend == "device_mpi":
            self._mpi_comm.Send(
                buf=buf,
                dest=dest,
            )

        elif backend == "host_mpi":
            buf_host = get_any_location(
                buf,
                output_module="numpy",
                use_pinned_memory=True,
            )

            # Need to synchronize the current stream to ensure that
            # the H2D copy finished before the MPI call is made.
            synchronize_device()

            self._mpi_comm.Send(
                buf=buf_host,
                dest=dest,
            )

    def recv(
        self,
        buf: NDArray,
        source: int,
        backend: str | None = None,
    ) -> None:
        """Performs non-blocking receive communication.

        Parameters
        ----------
        buf : NDArray
            The buffer to receive into.
        source : int
            The rank to receive from.
        backend : str, optional
            The backend to use for the communication. If None, the default
            backend for sendrecv will be used.

        """

        if self._group_start_called:
            raise RuntimeError(
                "recv cannot be called between group_start and group_end."
                "They should be called together with irecv."
            )

        if backend is None:
            backend = self._config["send_recv"]
        elif backend not in _backends:
            raise ValueError(f"Invalid backend: {backend}. Must be one of {_backends}.")

        if backend == "nccl":
            self._nccl_comm.recv(buf, source)

        elif backend == "device_mpi":
            self._mpi_comm.Recv(
                buf=buf,
                source=source,
            )

        elif backend == "host_mpi":
            buf_host = empty_like_pinned(buf)

            # Need to synchronize the current stream to ensure that
            # the alloc call is finished
            synchronize_device()

            self._mpi_comm.Recv(
                buf=buf_host,
                source=source,
            )
            buf[:] = get_any_location(
                buf_host,
                output_module="cupy",
                use_pinned_memory=True,
            )

    def isend(
        self,
        buf: NDArray,
        dest: int,
        backend: str | None = None,
    ) -> MPI.Request | None:
        """Performs non-blocking send communication.

        NOTE: For the nccl backend, the function is only non-blocking in the
        sense that it can be used in a group of communication operations between
        group_start and group_end.

        Parameters
        ----------
        buf : NDArray
            The buffer to send.
        dest : int
            The rank to send to.
        backend : str, optional
            The backend to use for the communication. If None, the default
            backend for sendrecv will be used.

        Returns
        -------
        MPI.Request | None
            The request object for the non-blocking communication operation. For
            the nccl backend, this will always be None, as the NCCL backend does
            not use requests

        """

        if not self._group_start_called:
            raise RuntimeError(
                "isend must be called between group_start and group_end."
            )

        if backend is None:
            backend = self._config["send_recv"]
        elif backend not in _backends:
            raise ValueError(f"Invalid backend: {backend}. Must be one of {_backends}.")

        if backend == "nccl":
            self._nccl_comm.send(buf, dest)
            return None

        elif backend == "device_mpi":
            return self._mpi_comm.Isend(
                buf=buf,
                dest=dest,
            )

        elif backend == "host_mpi":
            raise NotImplementedError(
                "Non-blocking send is not implemented for the host_mpi backend."
                "This is because it would need to synchronize in the irecv call,"
                "which would defeat the purpose of non-blocking communication."
            )

    def irecv(
        self,
        buf: NDArray,
        source: int,
        backend: str | None = None,
    ) -> MPI.Request | None:
        """Performs non-blocking receive communication.

        NOTE: For the nccl backend, the function is only non-blocking in the
        sense that it can be used in a group of communication operations between
        group_start and group_end.

        Parameters
        ----------
        buf : NDArray
            The buffer to receive into.
        source : int
            The rank to receive from.
        backend : str, optional
            The backend to use for the communication. If None, the default
            backend for sendrecv will be used.

        Returns
        -------
        MPI.Request | None
            The request object for the non-blocking communication operation. For
            the nccl backend, this will always be None, as the NCCL backend does
            not use requests

        """

        if not self._group_start_called:
            raise RuntimeError(
                "irecv must be called between group_start and group_end."
            )

        if backend is None:
            backend = self._config["send_recv"]
        elif backend not in _backends:
            raise ValueError(f"Invalid backend: {backend}. Must be one of {_backends}.")

        if backend == "nccl":
            self._nccl_comm.recv(buf, source)
            return None

        elif backend == "device_mpi":
            return self._mpi_comm.Irecv(
                buf=buf,
                source=source,
            )

        elif backend == "host_mpi":
            raise NotImplementedError(
                "Non-blocking receive is not implemented for the host_mpi backend."
                "This is because it would need to synchronize when copying the data back to the device,"
                "which would defeat the purpose of non-blocking communication."
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

    size = global_comm.size
    rank = global_comm.rank

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QuatrexCommunicator, cls).__new__(cls)

        return cls._instance

    def configure(
        self,
        block_comm_size: int,
        block_comm_config: dict,
        stack_comm_config: dict,
        override: bool = False,
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
        override : bool, optional
            Whether to override a previous configuration. Defaul
            is False.


        Raises
        -------
        RuntimeError
            If the communicator is already configured.
        ValueError
            If the block communicator size is not a multiple of the
            total number of ranks.

        """
        if self._is_configured and not override:
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
