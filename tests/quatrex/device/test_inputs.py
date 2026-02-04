# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest

from qttools import NDArray, xp
from quatrex.device.inputs import (
    _get_transport_block,
    create_coordinate_grid,
    expand_tight_binding_matrix,
    trim_tight_binding_matrix,
)


@pytest.mark.parametrize(
    "value_cutoff, neighbor_cell_cutoff",
    [
        (0.5, None),
        (None, (3, 2, 0)),
        (0.5, (3, 2, 0)),
        (None, (7, 9, 0)),
        (0.5, (7, 9, 0)),
    ],
)
def test_trim_tight_binding_matrix(
    unit_cells: NDArray, value_cutoff: float, neighbor_cell_cutoff: tuple[int, int, int]
):
    """Tests the trimming of the tight binding matrix."""

    trimmed_unit_cells = trim_tight_binding_matrix(
        tight_binding_matrix=unit_cells,
        value_cutoff=value_cutoff,
        neighbor_cell_cutoff=neighbor_cell_cutoff,
    )

    if neighbor_cell_cutoff is None:
        neighbor_cell_cutoff = [s // 2 if s > 1 else 1 for s in unit_cells.shape[:3]]

    neighbor_cell_cutoff = np.array(neighbor_cell_cutoff)

    if value_cutoff is None:
        # Only neighbor_cell_cutoff is set
        # Additional value cutoff could lead to smaller shape
        assert trimmed_unit_cells.shape[:3] == tuple(
            [
                2 * r + 1 if i > 1 else 1
                for r, i in zip(neighbor_cell_cutoff, unit_cells.shape[:3])
            ]
        )

    # Orbital dimensions should remain the same
    assert trimmed_unit_cells.shape[3:] == unit_cells.shape[3:]

    if value_cutoff is None:
        value_cutoff = -np.inf

    trimmed_cutoff = np.array(
        [s // 2 if s > 1 else 1 for s in trimmed_unit_cells.shape[:3]]
    )

    for r in np.ndindex(unit_cells.shape[:3]):

        r = np.asarray(r) - np.asarray(unit_cells.shape[:3]) // 2

        if np.all(np.abs(r) <= neighbor_cell_cutoff):
            unit_cells_ref = unit_cells[*r]

            zerod_unit_cells_ref = unit_cells_ref.copy()
            zerod_unit_cells_ref[xp.abs(zerod_unit_cells_ref) < value_cutoff] = 0

            # Some R values can have been removed because of the value_cutoff,
            # but then the corresponding unit_cells values should be zero
            # meaning the cell does not contain values above the cutoff
            if np.all(np.abs(r) <= trimmed_cutoff):
                assert xp.allclose(trimmed_unit_cells[*r], zerod_unit_cells_ref)

            else:
                assert len(unit_cells_ref[xp.abs(unit_cells[*r]) >= value_cutoff]) == 0


@pytest.mark.parametrize(
    "supercell_size, shift",
    [
        (
            (2, 2, 1),
            (0, 0, 0),
        ),
        (
            (2, 2, 1),
            (1, 1, 0),
        ),
        (
            (3, 1, 1),
            (0, 0, 0),
        ),
        (
            (3, 1, 1),
            (1, 0, 0),
        ),
    ],
)
def test_get_transport_block(
    unit_cells: NDArray,
    supercell_size: tuple[int, int, int],
    shift: tuple[int, int, int],
):
    """Tests the extraction of the transport block from the tight binding matrix."""

    bs = unit_cells.shape[-1]
    test_block = _get_transport_block(unit_cells, supercell_size, shift)

    global_shift = np.multiply(np.asarray(shift), np.asarray(supercell_size))

    for ind_r in np.ndindex(supercell_size):
        br = np.ravel_multi_index(np.array(ind_r), supercell_size)
        for ind_c in np.ndindex(supercell_size):
            bc = np.ravel_multi_index(np.array(ind_c), supercell_size)

            target_ind = tuple(np.array(ind_c) - np.array(ind_r) + global_shift)

            is_out_of_bounds = any(
                abs(val) > unit_cells.shape[dim] // 2
                for dim, val in enumerate(target_ind)
            )

            if is_out_of_bounds:
                ref_block = xp.zeros((bs, bs))
            else:
                ref_block = unit_cells[target_ind]

            assert xp.allclose(
                test_block[br * bs : (br + 1) * bs, bc * bs : (bc + 1) * bs], ref_block
            )


@pytest.mark.parametrize(
    "coords, supercell, lat_vecs",
    [
        (xp.ones((10, 3)), (2, 2, 2), xp.eye(3)),
    ],
)
def test_create_coordinate_grid(
    coords: NDArray, supercell: tuple[int, int, int], lat_vecs: NDArray
):
    """Tests the creation of a coordinate grid for a supercell."""
    grid = create_coordinate_grid(coords, supercell, lat_vecs)
    assert grid.shape == (xp.prod(xp.asarray(supercell)) * 10, 3)
    for ind in xp.ndindex(supercell):
        row_ind = xp.ravel_multi_index(xp.asarray(ind), supercell)
        assert xp.allclose(grid[row_ind * 10 : (row_ind + 1) * 10], 1 + xp.array(ind))


@pytest.mark.parametrize(
    "hr, num_transport_cells, transport_direction, block_start, block_end",
    [
        (
            xp.ones((7, 3, 3, 2, 2)),
            10,
            "x",
            None,
            None,
        ),
        (
            xp.ones((7, 3, 3, 2, 2)),
            10,
            "x",
            0,
            2,
        ),
    ],
    ids=[
        "sparse_no-block-inds",
        "sparse_with-block-inds",
    ],
)
def test_expand_tight_binding_matrix(
    hr: NDArray,
    num_transport_cells: int,
    transport_direction: str,
    block_start: int | None,
    block_end: int | None,
):
    """Tests the expansion of the tight-binding matrix into a block-tridiagonal Hamiltonian."""
    # NOTE: hr is set to ones to make it easy to check the resulting matrix
    # Using example data would require complex checks.

    sparse_hamiltonian, block_sizes = expand_tight_binding_matrix(
        tight_binding_matrix=hr,
        num_transport_cells=num_transport_cells,
        transport_direction=transport_direction,
        block_start=block_start,
        block_end=block_end,
    )
    block_start = block_start or 0
    block_end = block_end or num_transport_cells

    transport_direction = "xyz".index(transport_direction)

    # Number of unit cells in a transport cell
    transport_cell_size = hr.shape[transport_direction] // 2

    # Assumes the block sizes are the same for all blocks
    assert sparse_hamiltonian.shape[0] == block_sizes[0] * num_transport_cells
    assert sparse_hamiltonian.shape[1] == block_sizes[0] * num_transport_cells
    assert len(block_sizes) == block_end - block_start
    num_wann_per_supercell = int(block_sizes[0])
    num_wann = num_wann_per_supercell // transport_cell_size
    # Make sure the sparse hamiltonian is csr to be able to slice it
    if sparse_hamiltonian.format != "csr":
        sparse_hamiltonian = sparse_hamiltonian.tocsr()

    for i in range(block_start, block_end):
        block_slice = slice(
            i * num_wann_per_supercell, (i + 1) * num_wann_per_supercell
        )

        # Assume cut-off is large enough to include all interaction in diagonal blocks
        assert xp.allclose(sparse_hamiltonian[block_slice, block_slice].todense(), 1)

        if i < num_transport_cells - 1:
            off_diagonal_block_slice = slice(
                (i + 1) * num_wann_per_supercell, (i + 2) * num_wann_per_supercell
            )
            h_upper = sparse_hamiltonian[
                block_slice, off_diagonal_block_slice
            ].todense()
            h_lower = sparse_hamiltonian[
                off_diagonal_block_slice, block_slice
            ].todense()

            for r in range(transport_cell_size):
                row_unit_slice = slice(r * num_wann, (r + 1) * num_wann)
                for c in range(transport_cell_size):
                    col_unit_slice = slice(c * num_wann, (c + 1) * num_wann)

                    if r < c:
                        assert xp.allclose(
                            h_upper[block_slice][row_unit_slice, col_unit_slice], 0
                        )
                        assert xp.allclose(
                            h_lower[block_slice][col_unit_slice, row_unit_slice], 0
                        )
                    else:
                        assert xp.allclose(
                            h_upper[block_slice][row_unit_slice, col_unit_slice], 1
                        )
                        assert xp.allclose(
                            h_lower[block_slice][col_unit_slice, row_unit_slice], 1
                        )
