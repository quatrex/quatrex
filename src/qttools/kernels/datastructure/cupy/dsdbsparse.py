# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.


import cupy as cp

from qttools import QTX_USE_CUPY_JIT, NDArray
from qttools.kernels.datastructure.cupy import THREADS_PER_BLOCK

if QTX_USE_CUPY_JIT:
    from qttools.kernels.datastructure.cupy import _cupy_jit as cupy_backend
else:
    from qttools.kernels.datastructure.cupy import _cupy_rawkernel as cupy_backend


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
    dtype = nnz_section_offsets.dtype.type
    if inds.dtype.type != dtype:
        raise TypeError(
            f"All input arrays must have the same dtype, but got {nnz_section_offsets.dtype}, {inds.dtype}."
        )

    ranks = cp.zeros_like(inds)

    blocks_per_grid = (inds.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cupy_backend._find_ranks(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (
            nnz_section_offsets,
            inds,
            ranks,
            dtype(nnz_section_offsets.shape[0]),
            dtype(inds.shape[0]),
        ),
    )
    return ranks
