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
            cond = int(nnz_section_offsets[j] < inds[i])
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
    This method incurs some memory overhead compared to a naive
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

    bnnz_map = np.zeros((num_blocks, num_blocks), dtype=np.int32)

    # NOTE: This is a very generous estimate of the number of nonzeros
    # in each row of blocks. No assumption on the sparsity pattern of
    # the matrix is made here.
    row_nnz_estimate = min(len(coo_cols), max(block_sizes) * sum(block_sizes))
    inds = np.zeros((num_blocks, row_nnz_estimate), dtype=np.int32)

    # Treat block rows in parallel.
    for i in nb.prange(num_blocks):
        row_nnz_offset = 0
        row_mask = (block_offsets[i] <= coo_rows) & (coo_rows < block_offsets[i + 1])
        for j in range(num_blocks):
            mask = (
                row_mask
                & (block_offsets[j] <= coo_cols)
                & (coo_cols < block_offsets[j + 1])
            )
            nnz = np.sum(mask)
            if nnz > 0:
                bnnz_map[i, j] = nnz
                inds[i, row_nnz_offset : row_nnz_offset + nnz] = np.nonzero(mask)[0]
                row_nnz_offset += nnz

    # Construct the block-sorting index sequentially.
    sort_index = np.zeros(len(coo_cols), dtype=np.int64)
    nnz_offset = 0
    for i in range(num_blocks):
        row_nnz_offset = 0
        for j in range(num_blocks):
            nnz = bnnz_map[i, j]
            # Sort the data by block-row and -column.
            sort_index[nnz_offset : nnz_offset + nnz] = inds[
                i, row_nnz_offset : row_nnz_offset + nnz
            ]
            row_nnz_offset += nnz
            nnz_offset += nnz

    return sort_index
