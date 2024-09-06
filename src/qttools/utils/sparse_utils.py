import numpy as np


def compute_block_sort_index(
    coo_rows: np.ndarray, coo_cols: np.ndarray, block_sizes: np.ndarray
) -> np.ndarray:
    """Computes the block-sorting index for a sparse matrix.

    Parameters
    ----------
    coo_rows : np.ndarray
        The row indices of the matrix in coordinate format.
    coo_cols : np.ndarray
        The column indices of the matrix in coordinate format.
    block_sizes : np.ndarray
        The block sizes of the block-sparse matrix we want to construct.

    Returns
    -------
    sort_index : np.ndarray
        The indexing that sorts the data by block-row and -column.

    """
    num_blocks = len(block_sizes)
    block_offsets = np.hstack(([0], np.cumsum(block_sizes)))

    sort_index = np.zeros(len(coo_cols), dtype=int)
    offset = 0
    for i, j in np.ndindex(num_blocks, num_blocks):
        mask = (
            (block_offsets[i] <= coo_rows)
            & (coo_rows < block_offsets[i + 1])
            & (block_offsets[j] <= coo_cols)
            & (coo_cols < block_offsets[j + 1])
        )
        if not np.any(mask):
            # Skip empty blocks.
            continue

        bnnz = np.sum(mask)

        # Sort the data by block-row and -column.
        sort_index[offset : offset + bnnz] = np.argwhere(mask).squeeze()

        offset += bnnz

    return sort_index


def compute_ptr_map(
    coo_rows: np.ndarray, coo_cols: np.ndarray, block_sizes: np.ndarray
) -> dict:
    """Computes the rowptr map for a sparse matrix.

    Parameters
    ----------
    coo_rows : np.ndarray
        The row indices of the matrix in coordinate format.
    coo_cols : np.ndarray
        The column indices of the matrix in coordinate format.
    block_sizes : np.ndarray
        The block sizes of the block-sparse matrix we want to construct.

    Returns
    -------
    rowptr_map : dict
        The row pointer map, describing the block-sparse matrix in
        blockwise column-sparse-row format.

    """
    num_blocks = len(block_sizes)
    block_offsets = np.hstack(([0], np.cumsum(block_sizes)))

    # NOTE: This is a naive implementation and can be parallelized.
    rowptr_map = {}
    offset = 0
    for i, j in np.ndindex(num_blocks, num_blocks):
        mask = (
            (block_offsets[i] <= coo_rows)
            & (coo_rows < block_offsets[i + 1])
            & (block_offsets[j] <= coo_cols)
            & (coo_cols < block_offsets[j + 1])
        )
        if not np.any(mask):
            # Skip empty blocks.
            continue

        # Compute the rowptr map.
        rowptr, __ = np.histogram(
            coo_rows[mask] - block_offsets[i],
            bins=np.arange(block_sizes[i] + 1),
        )
        rowptr = np.hstack(([0], np.cumsum(rowptr))) + offset
        rowptr_map[(i, j)] = rowptr

        bnnz = np.sum(mask)
        offset += bnnz

    return rowptr_map
