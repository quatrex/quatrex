# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, sparse, xp
from qttools.kernels import dsbcoo_kernels


def _reference_compute_block_sort_index(
    coo_rows: NDArray, coo_cols: NDArray, block_sizes: NDArray
) -> NDArray:
    """Computes the block-sorting index for a sparse matrix.

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
    num_blocks = len(block_sizes)
    block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))

    sort_index = xp.zeros(len(coo_cols), dtype=int)
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

        bnnz = xp.sum(mask)

        # Sort the data by block-row and -column.
        sort_index[offset : offset + bnnz] = xp.argwhere(mask).squeeze()

        offset += bnnz

    return sort_index


def _reference_compute_block_slice(rows, cols, block_offsets, row, col):
    """Computes the slice of a block in a sparse matrix.

    Parameters
    ----------
    rows : NDArray
        The row indices of the matrix in coordinate format.
    cols : NDArray
        The column indices of the matrix in coordinate format.
    block_offsets : NDArray
        The block offsets of the block-sparse matrix.
    row : int
        The row index of the block.
    col : int
        The column index of the block.

    Returns
    -------
    start : int
        The start index of the block.
    stop : int
        The stop index of the block.

    """

    mask = (
        (rows >= block_offsets[row])
        & (rows < block_offsets[row + 1])
        & (cols >= block_offsets[col])
        & (cols < block_offsets[col + 1])
    )
    inds = mask.nonzero()[0]
    if len(inds) == 0:
        # No data in this block, cache an empty slice.
        return None, None
    # NOTE: The data is sorted by block-row and -column, so
    # we can safely assume that the block is contiguous.

    return inds[0], inds[-1] + 1


@pytest.mark.usefixtures("shape", "num_inds")
def test_find_inds(shape: tuple[int, int], num_inds: int):
    """Tests that the indices are found correctly."""
    coo = sparse.random(*shape, density=0.25, format="coo")
    rows = xp.random.choice(shape[0], size=num_inds, replace=False)
    cols = xp.random.choice(shape[1], size=num_inds, replace=False)

    reference_inds, reference_value_inds = xp.nonzero(
        (coo.row[:, xp.newaxis] == rows) & (coo.col[:, xp.newaxis] == cols)
    )
    inds, value_inds, max_count = dsbcoo_kernels.find_inds(coo.row, coo.col, rows, cols)

    assert max_count in (0, 1)
    assert xp.all(inds == reference_inds)
    assert xp.all(value_inds == reference_value_inds)


@pytest.mark.usefixtures("shape", "num_blocks", "block_coords")
def test_compute_block_slice(
    shape: tuple[int, int], num_blocks: int, block_coords: tuple[int, int]
):
    """Tests that block slices are computed correctly."""
    coo = sparse.random(*shape, density=0.25, format="coo")
    coo.sum_duplicates()

    block_sizes = xp.array(
        [a.size for a in xp.array_split(xp.arange(shape[0]), num_blocks)]
    )
    block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))

    sort_index = _reference_compute_block_sort_index(coo.row, coo.col, block_sizes)
    rows, cols = coo.row[sort_index], coo.col[sort_index]

    reference_block_slice = _reference_compute_block_slice(
        rows, cols, block_offsets, *block_coords
    )
    block_slice = dsbcoo_kernels.compute_block_slice(
        rows, cols, block_offsets, *block_coords
    )

    assert block_slice == reference_block_slice


@pytest.mark.usefixtures("shape")
def test_densify_block(shape: tuple[int, int]):
    """Tests that the block gets densified correctly."""
    coo = sparse.random(*shape, density=0.25, format="coo")
    coo.sum_duplicates()

    reference_block = coo.toarray()

    block = xp.zeros_like(reference_block)
    dsbcoo_kernels.densify_block(block, coo.row, coo.col, coo.data)

    assert xp.allclose(block, reference_block)


@pytest.mark.usefixtures("shape")
def test_sparsify_block(shape: tuple[int, int]):
    """Tests that the block gets sparsified correctly."""
    coo = sparse.random(*shape, density=0.25, format="coo")
    coo.sum_duplicates()

    data = xp.zeros_like(coo.data)
    dsbcoo_kernels.sparsify_block(coo.toarray(), coo.row, coo.col, data)

    assert xp.allclose(data, coo.data)


@pytest.mark.usefixtures("shape", "num_blocks")
def test_compute_block_sort_index(shape: tuple[int, int], num_blocks: int):
    """Tests that the block sort is computed correctly."""
    coo = sparse.random(*shape, density=0.25, format="coo")
    coo.sum_duplicates()

    block_sizes = xp.array(
        [a.size for a in xp.array_split(xp.arange(shape[0]), num_blocks)]
    )

    reference_sort_index = _reference_compute_block_sort_index(
        coo.row, coo.col, block_sizes
    )
    sort_index = dsbcoo_kernels.compute_block_sort_index(coo.row, coo.col, block_sizes)

    assert xp.all(sort_index == reference_sort_index)
