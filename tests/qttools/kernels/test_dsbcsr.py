# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, sparse, xp
from qttools.kernels import dsbcsr_kernels


def _reference_compute_rowptr_map(
    coo_rows: NDArray, coo_cols: NDArray, block_sizes: NDArray
) -> tuple[NDArray, dict]:
    """Computes the rowptr map for a sparse matrix.

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
        The block-sorting index for the sparse matrix.
    rowptr_map : dict
        The row pointer map, describing the block-sparse matrix in
        blockwise column-sparse-row format.


    """
    num_blocks = len(block_sizes)
    block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))

    sort_index = xp.zeros(len(coo_cols), dtype=int)
    rowptr_map = {}
    offset = 0
    for i, j in xp.ndindex(num_blocks, num_blocks):
        mask = (
            (block_offsets[i] <= coo_rows)
            & (coo_rows < block_offsets[i + 1])
            & (block_offsets[j] <= coo_cols)
            & (coo_cols < block_offsets[j + 1])
        )
        if not xp.any(mask):
            # Skip empty blocks.
            continue

        # Compute the rowptr map.
        rowptr, __ = xp.histogram(
            coo_rows[mask] - block_offsets[i],
            bins=xp.arange(block_sizes[i] + 1),
        )
        rowptr = xp.hstack(([0], xp.cumsum(rowptr))) + offset
        rowptr_map[(i, j)] = rowptr

        bnnz = xp.sum(mask)

        # Sort the data by block-row and -column.
        sort_index[offset : offset + bnnz] = xp.argwhere(mask).squeeze()

        offset += bnnz

    return sort_index, rowptr_map


def _reference_find_inds(
    rowptr_map: dict,
    block_offsets: NDArray,
    self_cols: NDArray,
    rows: NDArray,
    cols: NDArray,
) -> tuple[NDArray, NDArray]:
    """Computes the indices of the given rows and columns.

    Parameters
    ----------
    rowptr_map : dict
        The row pointer map of the block-sparse matrix.
    block_offsets : NDArray
        The block offsets of the block-sparse matrix.
    self_cols : NDArray
        The column indices of this matrix.
    rows : NDArray
        The requested row indices.
    cols : NDArray
        The requested column indices.

    """
    brows = (block_offsets <= rows[:, xp.newaxis]).sum(-1) - 1
    bcols = (block_offsets <= cols[:, xp.newaxis]).sum(-1) - 1

    # Get an ordered list of unique blocks.
    unique_blocks = dict.fromkeys(zip(map(int, brows), map(int, bcols))).keys()

    inds, value_inds = [], []
    for brow, bcol in unique_blocks:
        rowptr = rowptr_map.get((brow, bcol), None)
        if rowptr is None:
            continue

        mask = (brows == brow) & (bcols == bcol)
        mask_inds = xp.nonzero(mask)[0]

        # Renormalize the row indices for this block.
        rr = rows[mask] - block_offsets[brow]
        cc = cols[mask]

        for i, (r, c) in enumerate(zip(rr, cc)):
            ind = xp.nonzero(self_cols[rowptr[r] : rowptr[r + 1]] == c)[0]

            if len(ind) == 0:
                continue

            value_inds.append(mask_inds[i])
            inds.append(rowptr[r] + ind[0])

    return xp.array(inds, dtype=int), xp.array(value_inds, dtype=int)


@pytest.mark.usefixtures("shape", "num_inds", "num_blocks")
def test_find_inds(shape: tuple[int, int], num_inds: int, num_blocks: int):
    """Tests the that we find the correct indices."""
    coo = sparse.random(*shape, density=0.25, format="coo")
    rows = xp.random.choice(shape[0], size=num_inds, replace=False)
    cols = xp.random.choice(shape[1], size=num_inds, replace=False)

    coo.sum_duplicates()

    block_sizes = xp.array(
        [a.size for a in xp.array_split(xp.arange(shape[0]), num_blocks)]
    )
    block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))

    sort_index, rowptr_map = _reference_compute_rowptr_map(
        coo.row, coo.col, block_sizes
    )

    reference_inds, reference_value_inds = _reference_find_inds(
        rowptr_map, block_offsets, coo.col[sort_index], rows, cols
    )
    inds, value_inds = dsbcsr_kernels.find_inds(
        rowptr_map, block_offsets, coo.col[sort_index], rows, cols
    )

    assert xp.all(inds == reference_inds)
    assert xp.all(value_inds == reference_value_inds)


@pytest.mark.usefixtures("shape")
def test_densify_block(shape: tuple[int, int]):
    """Tests that the block is densified correctly."""
    csr = sparse.random(*shape, density=0.25, format="csr")

    reference_block = csr.toarray()

    block = xp.zeros_like(reference_block)
    dsbcsr_kernels.densify_block(block, 0, csr.indices, csr.indptr, csr.data)

    assert xp.allclose(block, reference_block)


@pytest.mark.usefixtures("shape")
def test_sparsify_block(shape: tuple[int, int]):
    """Tests that the block is sparsified correctly."""
    csr = sparse.random(*shape, density=0.25, format="csr")

    data = xp.zeros_like(csr.data)
    dsbcsr_kernels.sparsify_block(csr.toarray(), 0, csr.indices, csr.indptr, data)

    assert xp.allclose(data, csr.data)


@pytest.mark.usefixtures("shape", "num_blocks")
def test_compute_rowptr_map(shape: tuple[int, int], num_blocks: int):
    """Tests that the row pointer map is computed correctly."""
    coo = sparse.random(*shape, density=0.25, format="coo")
    coo.sum_duplicates()

    block_sizes = xp.array(
        [a.size for a in xp.array_split(xp.arange(shape[0]), num_blocks)]
    )

    reference_sort_index, reference_rowptr_map = _reference_compute_rowptr_map(
        coo.row, coo.col, block_sizes
    )
    sort_index, rowptr_map = dsbcsr_kernels.compute_rowptr_map(
        coo.row, coo.col, block_sizes
    )

    assert xp.all(sort_index == reference_sort_index)

    for key in reference_rowptr_map:
        assert xp.all(rowptr_map[key] == reference_rowptr_map[key])
