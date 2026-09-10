# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Functions on distributed arrays."""

from qttools import NDArray, xp
from qttools.comm.comm import _SubCommunicator


def distributed_max(a: NDArray, comm: _SubCommunicator) -> NDArray:
    """Returns the maximum of the real and possibly distributed `NDArray` `a`.

    Parameters
    ----------
    a : NDArray
        The input array, which can be distributed across multiple processes.
    comm : _SubCommunicator
        The communicator that defines the distribution of the array `a`.

    Returns
    -------
    NDArray
        The maximum value of the array `a` across all processes.

    """

    if xp.iscomplexobj(a):
        raise ValueError("The maximum of a complex array is not defined")

    # NOTE: numpy would return a scalar and thus, we use `atleast_1d`.
    local_maximum = xp.atleast_1d(xp.max(a))
    maximum = xp.empty_like(local_maximum)
    comm.all_reduce(local_maximum, maximum, op="max")

    return maximum
