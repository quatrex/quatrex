# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.


import cupy as cp
import numpy as np

from qttools import QTX_USE_CUPY_JIT, NDArray
from qttools.kernels.datastructure.cuda import THREADS_PER_BLOCK
from qttools.profiling import Profiler

if QTX_USE_CUPY_JIT:
    from qttools.kernels.datastructure.cuda import _cupy_jit as cupy_backend
else:
    from qttools.kernels.datastructure.cuda import _cupy_rawkernel as cupy_backend


profiler = Profiler()


@profiler.profile(level="api")
def find_ranks(nnz_section_offsets: NDArray, inds: NDArray) -> NDArray:
    """Finds the ranks of the indices in the offsets.

    Parameters
    ----------
    nnz_section_offsets : NDArray
        The offsets of the non-zero sections.
    inds : NDArray
        The indices to find the ranks for.

    Returns
    -------
    ranks : NDArray
        The ranks of the indices in the offsets.

    """
    ranks = cp.zeros(inds.shape[0], dtype=cp.int16)

    nnz_section_offsets = nnz_section_offsets.astype(cp.int32)
    inds = inds.astype(cp.int32)

    blocks_per_grid = (inds.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cupy_backend._find_ranks(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (
            nnz_section_offsets,
            inds,
            ranks,
            np.int32(nnz_section_offsets.shape[0]),
            np.int32(inds.shape[0]),
        ),
    )
    return ranks
