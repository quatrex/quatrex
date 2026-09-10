# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Functions on distributed arrays."""

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, xp
from qttools.utils.gpu_utils import get_host


def distributed_max(a: NDArray):
    """Returns the maximum of the real and possibly distributed `NDArray` `a`."""

    if xp.iscomplexobj(a):
        raise ValueError("The maximum of a complex array is not defined")

    local_maximum = get_host(xp.max(a))
    maximum = np.empty_like(local_maximum)
    global_comm.Allreduce(local_maximum, maximum, op=MPI.MAX)

    return maximum
