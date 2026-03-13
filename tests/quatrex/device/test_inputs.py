# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest

from qttools import NDArray, xp
from quatrex.device.inputs import (
    _construct_transport_cell,
    _expand_tight_binding_matrix,
    create_coordinate_grid,
)


@pytest.mark.parametrize(
    "transport_cell_size, shift",
    [
        (
            2,
            (0, 0, 0),
        ),
        (
            2,
            (1, 1, 0),
        ),
        (
            3,
            (0, 0, 0),
        ),
        (
            3,
            (1, 0, 0),
        ),
    ],
)
def test_construct_transport_cell(
    matrix_dict: dict,
    transport_cell_size: int,
    shift: tuple[int, int, int],
):
    """Tests the extraction of the transport block from the tight binding matrix."""

    # z-direction for transport
    transport_ind = 2
    test_block = _construct_transport_cell(
        matrix_dict, transport_cell_size, transport_ind, shift
    )

    bs = matrix_dict[(0, 0, 0)].shape[-1]
    for br in range(transport_cell_size):
        for bc in range(transport_cell_size):
            target_ind = list(shift)
            target_ind[transport_ind] = bc - br
            target_ind = tuple(target_ind)

            ref_block = matrix_dict.get(target_ind, xp.zeros((bs, bs)))

            assert xp.allclose(
                test_block[br * bs : (br + 1) * bs, bc * bs : (bc + 1) * bs], ref_block
            )


@pytest.mark.parametrize(
    "coords, transport_cell_size, transport_ind, lat_vecs",
    [
        (xp.ones((10, 3)), 2, 2, xp.eye(3)),
    ],
)
def test_create_coordinate_grid(
    coords: NDArray, transport_cell_size: int, transport_ind: int, lat_vecs: NDArray
):
    """Tests the creation of a coordinate grid for a transport cell."""
    grid = create_coordinate_grid(coords, transport_cell_size, transport_ind, lat_vecs)
    assert grid.shape == (transport_cell_size * 10, 3)

    transport_cell = [1, 1, 1]
    transport_cell[transport_ind] = transport_cell_size
    transport_cell = tuple(transport_cell)

    for ind in xp.ndindex(transport_cell):
        row_ind = xp.ravel_multi_index(xp.asarray(ind), transport_cell)
        assert xp.allclose(grid[row_ind * 10 : (row_ind + 1) * 10], 1 + xp.array(ind))


@pytest.mark.parametrize(
    "hopping_shape, num_transport_cells, transport_ind, block_start, block_end, periodic_shift",
    [
        ((7, 5, 5), 10, 0, None, None, (0, 0, 0)),
        ((7, 5, 5), 10, 0, 0, 2, (0, 0, 0)),
        ((7, 5, 5), 10, 0, None, None, (0, 0, 2)),
        ((7, 5, 5), 10, 0, 0, 2, (0, 0, 2)),
    ],
)
def test_expand_tight_binding_matrix(
    hopping_shape: tuple[int, int, int],
    num_transport_cells: int,
    transport_ind: int,
    block_start: int | None,
    block_end: int | None,
    periodic_shift: tuple,
):
    """Tests the expansion of the tight-binding matrix into a block-tridiagonal Hamiltonian."""
    # NOTE: set to ones to make it easy to check the resulting matrix
    # Using example data would require complex checks.

    block = xp.ones((2, 2))

    matrix_dict = {
        tuple(int(i) for i in ind): block for ind in np.ndindex(hopping_shape)
    }

    for ind in list(matrix_dict.keys()):
        for axis in range(3):
            neg_ind = list(ind)
            neg_ind[axis] = -ind[axis]
            matrix_dict[tuple(neg_ind)] = block

    matrix_dict[(0, 0, 0)] = xp.triu(block)

    # drop half the keys
    matrix_dict = {
        coord: matrix
        for coord, matrix in matrix_dict.items()
        if coord > (0, 0, 0) or (coord == (0, 0, 0))
    }

    sparse_hamiltonian = _expand_tight_binding_matrix(
        matrix_dict=matrix_dict,
        num_transport_cells=num_transport_cells,
        transport_ind=transport_ind,
        block_start=block_start,
        block_end=block_end,
        periodic_shift=periodic_shift,
    )
    block_start = block_start or 0
    block_end = block_end or num_transport_cells

    # Number of unit cells in a transport cell
    transport_cell_size = matrix_dict[(0, 0, 0)].shape[transport_ind] * (
        hopping_shape[transport_ind] - 1
    )
    num_unit_cells = hopping_shape[transport_ind] - 1
    unit_cell_size = matrix_dict[(0, 0, 0)].shape[0]

    assert sparse_hamiltonian.shape[0] == sparse_hamiltonian.shape[1]
    assert sparse_hamiltonian.shape[0] % transport_cell_size == 0

    # Make sure the sparse hamiltonian is csr to be able to slice it
    if sparse_hamiltonian.format != "csr":
        sparse_hamiltonian = sparse_hamiltonian.tocsr()

    for i in range(block_start, block_end):
        block_slice = slice(i * transport_cell_size, (i + 1) * transport_cell_size)

        # Assume cut-off is large enough to include all interaction in diagonal blocks
        if np.all(np.array(periodic_shift) == 0):
            assert xp.allclose(
                xp.triu(sparse_hamiltonian[block_slice, block_slice].todense()),
                xp.triu(xp.ones((transport_cell_size, transport_cell_size))),
            )
            assert xp.allclose(
                xp.tril(sparse_hamiltonian[block_slice, block_slice].todense(), k=-1),
                xp.zeros((transport_cell_size, transport_cell_size)),
            )
        else:
            assert xp.allclose(
                sparse_hamiltonian[block_slice, block_slice].todense(),
                1,
            )

        if i < num_transport_cells - 1:
            off_diagonal_block_slice = slice(
                (i + 1) * transport_cell_size, (i + 2) * transport_cell_size
            )
            h_upper = sparse_hamiltonian[
                block_slice, off_diagonal_block_slice
            ].todense()

            if not np.all(np.array(periodic_shift) == 0):
                h_lower = sparse_hamiltonian[
                    off_diagonal_block_slice, block_slice
                ].todense()

            for r in range(num_unit_cells):
                row_unit_slice = slice(r * unit_cell_size, (r + 1) * unit_cell_size)
                for c in range(num_unit_cells):
                    col_unit_slice = slice(c * unit_cell_size, (c + 1) * unit_cell_size)

                    if r < c:
                        assert xp.allclose(
                            h_upper[block_slice][row_unit_slice, col_unit_slice], 0
                        )
                    else:
                        assert xp.allclose(
                            h_upper[block_slice][row_unit_slice, col_unit_slice], 1
                        )

                    if not np.all(np.array(periodic_shift) == 0):
                        if r < c:
                            assert xp.allclose(
                                h_lower[block_slice][col_unit_slice, row_unit_slice], 0
                            )
                        else:
                            assert xp.allclose(
                                h_lower[block_slice][col_unit_slice, row_unit_slice], 1
                            )
