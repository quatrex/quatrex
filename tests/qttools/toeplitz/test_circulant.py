# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes tests for the qttools.toeplitz module."""

import pytest

from qttools import xp
from qttools.toeplitz.circulant import check_nested_phi_circulant, expand_transverse
from quatrex.device.inputs import expand_circulant_cell


def _generate_dict(
    rng,
    transport_ind: int,
    grid_shape: tuple[int, int, int],
    block_size: int,
) -> dict:
    """Generates a dictionary of random matrices with keys corresponding to the grid shape."""
    matrix_dict = {}
    for i, j, k in xp.ndindex(grid_shape):
        key = [i, j, k]
        key[transport_ind] -= grid_shape[transport_ind] // 2
        key = tuple(key)
        matrix_dict[key] = rng.random((block_size, block_size))

    return matrix_dict


def test_transverse_upscale(
    grid_shape: tuple[int, int, int],
    transport_ind: int,
    block_size: int,
    phases: tuple[complex, complex],
):
    """Tests the transverse upscale of the tight binding matrix."""
    rng = xp.random.default_rng(seed=42)
    matrix_dict = _generate_dict(rng, transport_ind, grid_shape, block_size)

    transverse_inds = [i for i in range(3) if i != transport_ind]
    sections = tuple(grid_shape[i] for i in transverse_inds)

    matrix_dict = expand_transverse(matrix_dict, transport_ind, sections, phases)

    for matrix in matrix_dict.values():
        reference_shape = (
            block_size * grid_shape[transverse_inds[0]] * grid_shape[transverse_inds[1]]
        )
        assert matrix.shape == (reference_shape, reference_shape)

    for i in range(grid_shape[transport_ind]):
        test_key = [0, 0, 0]
        test_key[transport_ind] = i - grid_shape[transport_ind] // 2
        test_key = tuple(test_key)
        test_matrix = matrix_dict[test_key]
        assert check_nested_phi_circulant(test_matrix, sections, phases)


def test_full_upscale(
    grid_shape: tuple[int, int, int],
    transport_ind: int,
    block_size: int,
    phases: tuple[complex, complex],
):
    """Tests the full upscale of the tight binding matrix."""

    if grid_shape[transport_ind] < 3:
        pytest.skip("Grid shape too small for full upscale test.")

    rng = xp.random.default_rng(seed=42)
    matrix_dict = _generate_dict(rng, transport_ind, grid_shape, block_size)

    transverse_inds = [i for i in range(3) if i != transport_ind]
    sections = tuple(grid_shape[i] for i in transverse_inds)

    reference_shape = (
        block_size
        * grid_shape[transverse_inds[0]]
        * grid_shape[transverse_inds[1]]
        * (grid_shape[transport_ind] // 2)
    )

    for index in [-1, 0, 1]:
        test_matrix = expand_circulant_cell(
            matrix_dict=matrix_dict,
            transport_cell_size=grid_shape[transport_ind] // 2,
            transport_ind=transport_ind,
            index=index,
            sections=sections,
            phases=phases,
        )
        assert test_matrix.shape == (reference_shape, reference_shape)

        for m in range(grid_shape[transport_ind] // 2):
            for n in range(grid_shape[transport_ind] // 2):
                block_size = test_matrix.shape[-1] // (grid_shape[transport_ind] // 2)
                slice_m = slice(m * block_size, (m + 1) * block_size)
                slice_n = slice(n * block_size, (n + 1) * block_size)
                assert check_nested_phi_circulant(
                    test_matrix[slice_m, slice_n], sections, phases
                )
