# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest

from qttools import NDArray, xp
from quatrex.device.inputs import (
    _expand_tight_binding_matrix,
    construct_transport_cell,
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
@pytest.mark.parametrize("key_assumption", [None, "upper", "half"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_construct_transport_cell(
    matrix_dict: dict,
    transport_cell_size: int,
    shift: tuple[int, int, int],
    key_assumption: str | None,
    batch_size: int,
):
    """Tests the extraction of the transport block from the tight binding matrix."""

    # z-direction for transport
    transport_ind = 2

    # Avoid modifying the orignal one
    if batch_size > 1:
        matrix_dict_copy = {
            key: xp.array([matrix.copy() for _ in range(batch_size)])
            for key, matrix in matrix_dict.items()
        }
    else:
        matrix_dict_copy = {key: matrix.copy() for key, matrix in matrix_dict.items()}

    if key_assumption == "upper":
        matrix_dict_copy = {
            key: xp.triu(matrix) for key, matrix in matrix_dict_copy.items()
        }
    elif key_assumption == "half":
        matrix_dict_copy = {
            key: matrix for key, matrix in matrix_dict_copy.items() if key >= (0, 0, 0)
        }

    test_block = construct_transport_cell(
        matrix_dict_copy, transport_cell_size, transport_ind, shift, key_assumption
    )

    shape = matrix_dict[(0, 0, 0)].shape
    block_size = matrix_dict[(0, 0, 0)].shape[-1]
    for r_i in range(transport_cell_size):
        for r_j in range(transport_cell_size):
            target_ind = list(shift)
            target_ind[transport_ind] = r_j - r_i
            target_ind = tuple(target_ind)

            ref_block = matrix_dict.get(target_ind, xp.zeros(shape))

            assert xp.allclose(
                test_block[
                    ...,
                    r_i * block_size : (r_i + 1) * block_size,
                    r_j * block_size : (r_j + 1) * block_size,
                ],
                ref_block,
            )


@pytest.mark.parametrize(
    "unit_cell_coords, num_unit_cells, transport_ind, lat_vecs",
    [
        (xp.ones((10, 3)), 2, 2, xp.eye(3)),
    ],
)
def test_create_coordinate_grid(
    unit_cell_coords: NDArray,
    num_unit_cells: int,
    transport_ind: int,
    lat_vecs: NDArray,
):
    """Tests the creation of a coordinate grid for a transport cell."""

    num_coords = unit_cell_coords.shape[0]
    grid = create_coordinate_grid(
        unit_cell_coords, num_unit_cells, transport_ind, lat_vecs
    )
    assert grid.shape == (num_unit_cells * num_coords, 3)

    for i in range(num_unit_cells):
        ref = unit_cell_coords.copy()
        ref[:, :] += i * lat_vecs[transport_ind][None, :]
        assert xp.allclose(grid[i * num_coords : (i + 1) * num_coords], ref)


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
        tuple(int(i) for i in ind): xp.triu(block) for ind in np.ndindex(hopping_shape)
    }

    for ind in list(matrix_dict.keys()):
        neg_ind = tuple(-i for i in ind)
        matrix_dict[tuple(neg_ind)] = block

    matrix_dict = {coord: xp.triu(matrix) for coord, matrix in matrix_dict.items()}

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
        assert xp.allclose(
            xp.triu(sparse_hamiltonian[block_slice, block_slice].todense()),
            xp.triu(xp.ones((transport_cell_size, transport_cell_size))),
        )
        assert xp.allclose(
            xp.tril(sparse_hamiltonian[block_slice, block_slice].todense(), k=-1),
            xp.zeros((transport_cell_size, transport_cell_size)),
        )

        if i < num_transport_cells - 1:
            off_diagonal_block_slice = slice(
                (i + 1) * transport_cell_size, (i + 2) * transport_cell_size
            )
            h_upper = sparse_hamiltonian[
                block_slice, off_diagonal_block_slice
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
