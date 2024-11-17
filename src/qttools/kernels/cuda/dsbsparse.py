import cupy as cp
from cupy.typing import ArrayLike
from cupyx import jit

from qttools.kernels.cuda import THREADS_PER_BLOCK
from qttools.kernels.cuda.dsbcoo import _compute_coo_block_mask_kernel


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
            cond = int(nnz_section_offsets[j] <= inds[i])
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


def compute_block_sort_index(
    coo_rows: ArrayLike, coo_cols: ArrayLike, block_sizes: ArrayLike
) -> ArrayLike:
    """Computes the block-sorting index for a sparse matrix.

    Note
    ----
    Due to the Python for loop around the kernel, this method will
    perform best for larger block sizes (>500).


    Parameters
    ----------
    coo_rows : array_like
        The row indices of the matrix in coordinate format.
    coo_cols : array_like
        The column indices of the matrix in coordinate format.
    block_sizes : array_like
        The block sizes of the block-sparse matrix we want to construct.

    Returns
    -------
    sort_index : array_like
        The indexing that sorts the data by block-row and -column.

    """
    num_blocks = block_sizes.shape[0]
    block_offsets = cp.hstack((cp.array([0]), cp.cumsum(block_sizes)))

    sort_index = cp.zeros(len(coo_cols), dtype=cp.int32)
    mask = cp.zeros(len(coo_cols), dtype=cp.bool_)

    blocks_per_grid = (len(coo_cols) + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    offset = 0
    for i, j in cp.ndindex(num_blocks, num_blocks):
        _compute_coo_block_mask_kernel(
            (blocks_per_grid,),
            (THREADS_PER_BLOCK,),
            (
                coo_rows,
                coo_cols,
                int(block_offsets[i]),
                int(block_offsets[i + 1]),
                int(block_offsets[j]),
                int(block_offsets[j + 1]),
                mask,
            ),
        )

        bnnz = cp.sum(mask)

        if bnnz != 0:
            # Sort the data by block-row and -column.
            sort_index[offset : offset + bnnz] = cp.nonzero(mask)[0]

            offset += bnnz

    return sort_index
