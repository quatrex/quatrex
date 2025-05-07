# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.


import cupy as cp
import numpy as np
from cupyx import jit

from qttools import QTX_USE_CUPY_JIT, NDArray
from qttools.kernels.cuda import THREADS_PER_BLOCK
from qttools.profiling import Profiler

profiler = Profiler()

if QTX_USE_CUPY_JIT:

    @jit.rawkernel()
    def _find_ranks_kernel(
        nnz_section_offsets: NDArray,
        inds: NDArray,
        ranks: NDArray,
        nnz_section_offsets_len: int,
        inds_len: int,
    ):
        """Finds the ranks of the indices in the offsets.

        Parameters
        ----------
        nnz_section_offsets : NDArray
            The offsets of the non-zero sections.
        inds : NDArray
            The indices to find the ranks for.
        ranks : NDArray
            The ranks of the indices in the offsets.

        """
        i = int(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
        if i < inds_len:
            for j in range(nnz_section_offsets_len):
                cond = int(nnz_section_offsets[j] <= inds[i])
                ranks[i] = ranks[i] * (1 - cond) + j * cond

else:
    _find_ranks_kernel = cp.RawKernel(
        r"""
        extern "C" __global__
        void _find_ranks_kernel(
            int *nnz_section_offsets,
            int *inds,
            short *ranks,
            int nnz_section_offsets_len,
            int inds_len
        ) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < inds_len) {
                for (int j = 0; j < nnz_section_offsets_len; j++) {
                    int cond = nnz_section_offsets[j] <= inds[tid];
                    ranks[tid] = ranks[tid] * (1 - cond) + j * cond;
                }
            }
        }
    """,
        "_find_ranks_kernel",
    )


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
    _find_ranks_kernel(
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
