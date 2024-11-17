import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.njit(parallel=True, cache=True)
def find_ranks(nnz_section_offsets: NDArray, inds: NDArray) -> NDArray:
    """Find the ranks of the indices in the offsets.

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
    ranks = np.zeros(inds.shape[0], dtype=np.int16)
    for i in nb.prange(inds.shape[0]):
        for j in range(nnz_section_offsets.shape[0]):
            cond = int(nnz_section_offsets[j] <= inds[i])
            ranks[i] = ranks[i] * (1 - cond) + j * cond

    return ranks


def densify_blocks(): ...


@nb.njit(parallel=True, cache=True)
def compute_block_sort_index(
    coo_rows: NDArray, coo_cols: NDArray, block_sizes: NDArray
) -> NDArray:
    """Computes the block-sorting index for a sparse matrix.

    Note
    ----
    This method incurs a bit of memory overhead compared to a naive
    implementation. No assumptions on the sparsity pattern of the matrix
    are made here. See the source code for more details.

    Parameters
    ----------
    coo_rows : NDArray
        The row indices of the matrix in coordinate format.
    coo_cols : NDArray
        The column indices of the matrix in coordinate format.
    block_sizes : NDArray
        The block sizes of the block-sparse matrix we want to construct.

    Returns
    -------
    sort_index : NDArray
        The indexing that sorts the data by block-row and -column.

    """
    num_blocks = block_sizes.shape[0]
    block_offsets = np.hstack((np.array([0]), np.cumsum(block_sizes)))

    sort_index = np.zeros(len(coo_cols), dtype=np.int64)

    # NOTE: This is a very generous estimate of the number of
    # nonzeros in each row of blocks. No assumption on the sparsity
    # pattern of the matrix is made here.
    nnz_estimate = min(len(coo_cols), max(block_sizes) ** 2)
    inds = np.zeros((num_blocks, nnz_estimate), dtype=np.int32)

    block_nnz = np.zeros(num_blocks, dtype=np.int32)
    nnz_offset = 0
    for i in range(num_blocks):
        # Precompute the row mask.
        row_mask = (block_offsets[i] <= coo_rows) & (coo_rows < block_offsets[i + 1])

        # Process in parallel.
        for j in nb.prange(num_blocks):
            mask = (
                row_mask
                & (block_offsets[j] <= coo_cols)
                & (coo_cols < block_offsets[j + 1])
            )
            nnz = np.sum(mask)
            if nnz > 0:
                block_nnz[j] = nnz
                inds[j, :nnz] = np.nonzero(mask)[0]

        # Reduce the indices sequentially.
        for j in range(num_blocks):
            nnz = block_nnz[j]
            # Sort the data by block-row and -column.
            sort_index[nnz_offset : nnz_offset + nnz] = inds[j, :nnz]
            nnz_offset += nnz

    return sort_index
