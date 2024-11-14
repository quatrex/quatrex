import cupy as cp
from cupy.typing import ArrayLike
from cupyx import jit

from qttools.kernels.cuda import THREADS_PER_BLOCK


@jit.rawkernel()
def _find_ranks_kernel(
    nnz_section_offsets: ArrayLike, inds: ArrayLike, ranks: ArrayLike
):
    """Finds the ranks of the indices in the offsets.

    Parameters
    ----------
    nnz_section_offsets : array_like
        The offsets of the non-zero sections.
    inds : array_like
        The indices to find the ranks for.
    ranks : array_like
        The ranks of the indices in the offsets.

    """
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < inds.shape[0]:
        for j in range(nnz_section_offsets.shape[0]):
            cond = int(nnz_section_offsets[j] < inds[i])
            ranks[i] = ranks[i] * (1 - cond) + j * cond


def find_ranks(nnz_section_offsets: ArrayLike, inds: ArrayLike) -> ArrayLike:
    """Finds the ranks of the indices in the offsets.

    Parameters
    ----------
    nnz_section_offsets : array_like
        The offsets of the non-zero sections.
    inds : array_like
        The indices to find the ranks for.

    Returns
    -------
    ranks : array_like
        The ranks of the indices in the offsets.

    """
    ranks = cp.zeros(inds.shape[0], dtype=cp.int16)

    blocks_per_grid = (inds.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    _find_ranks_kernel(
        (blocks_per_grid,),
        (THREADS_PER_BLOCK,),
        (nnz_section_offsets, inds, ranks),
    )
    return ranks


def compute_block_sort_index(): ...
