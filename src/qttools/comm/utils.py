import numpy as np

from qttools import NDArray, xp
from qttools.comm.comm import _SubCommunicator


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


def all_gather_v(
    comm: _SubCommunicator,
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
        if mask.size // comm.size < sendbuf.size:
            raise ValueError(
                f"The mask is too small for the sendbuf: {mask.size // comm.size} < {sendbuf.size}."
            )
        global_size = mask.size
    else:
        counts = np.zeros(comm.size, dtype=xp.int32)
        comm.all_gather(np.array(sendbuf.shape[axis]), counts, backend="device_mpi")
        global_size = np.max(counts) * comm.size
        mask = xp.zeros(global_size, dtype=bool)
        for i in range(comm.size):
            mask[np.max(counts) * i : np.max(counts) * i + counts[i]] = True

    if mask.ndim > 1:
        raise ValueError("mask must be 1D or None")

    sendbuf = pad_buffer(sendbuf, global_size, comm.size, axis)

    sendbuf = xp.ascontiguousarray(xp.moveaxis(sendbuf, axis, 0))
    recvbuf = xp.empty((global_size, *sendbuf.shape[1:]), dtype=sendbuf.dtype)
    comm.all_gather(sendbuf, recvbuf)
    recvbuf = xp.moveaxis(recvbuf, 0, axis)

    indices = xp.where(mask)[0]
    return xp.take(
        recvbuf,
        indices,
        axis=axis,
    )
