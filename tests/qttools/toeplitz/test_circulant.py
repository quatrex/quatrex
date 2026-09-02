# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes tests for the qttools.toeplitz module."""

import pytest

from qttools import NDArray, xp
from qttools.toeplitz.circulant import construct_circulant_cell, expand_transverse


def check_phi_circulant(
    a: NDArray,
    sections: int,
    phase: complex = 1.0,
) -> bool:
    """Check if a matrix is block phi circulant with the given number of sections.

    Parameters
    ----------
    a : NDArray
        The matrix to check.
    sections : int
        The number of sections in the block circulant structure.
    phase : complex, optional
        The phase factor to apply to the blocks, by default 1.0.

    Returns
    -------
    bool
        True if a is block circulant with the given number of sections, False otherwise.

    """
    if a.shape[-1] % sections != 0:
        raise ValueError("The last dimension of a must be divisible by sections.")

    if a.shape[-2] != a.shape[-1]:
        raise ValueError(
            "The second to last dimension of a must be equal to the last dimension of a."
        )

    block_size = a.shape[-1] // sections
    # Take the first block-row (top n rows)
    block_layer = a[..., :block_size, :]
    blocks = xp.split(block_layer, sections, axis=-1)

    for i in range(sections):
        if i == 0:
            shifted_blocks = blocks
        else:
            # Scale wrapped blocks by phase, leave remaining blocks unscaled
            shifted_blocks = [phase * b for b in blocks[-i:]] + blocks[:-i]

        expected = xp.concatenate(shifted_blocks, axis=-1)
        actual = a[..., i * block_size : (i + 1) * block_size, :]

        if not xp.allclose(actual, expected):
            return False

    return True


def check_nested_phi_circulant(
    a: NDArray,
    sections: tuple[int, int],
    phases: tuple[complex, complex] = (1.0, 1.0),
) -> bool:
    """Check if a matrix is block phi circulant in 2D with the given
    number of sections.

    Parameters
    ----------
    a : NDArray
        The matrix to check.
    sections : tuple[int, int]
        The number of sections in the block circulant structure for each
        dimension.
    phases : tuple[complex, complex], optional
        The phase factors to apply to the blocks for each dimension, by
        default (1.0, 1.0).

    Returns
    -------
    bool
        True if a is block circulant in 2D with the given number of
        sections, False otherwise.

    """
    if not check_phi_circulant(a, sections[1], phases[1]):
        return False

    for i in range(sections[1]):
        for j in range(sections[1]):
            block_size = a.shape[-1] // sections[1]
            slice_i = slice(i * block_size, (i + 1) * block_size)
            slice_j = slice(j * block_size, (j + 1) * block_size)
            if not check_phi_circulant(a[slice_i, slice_j], sections[0], phases[0]):
                return False

    return True


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

    for block_index in [-1, 0, 1]:
        test_matrix = construct_circulant_cell(
            matrix_dict=matrix_dict,
            transport_cell_size=grid_shape[transport_ind] // 2,
            transport_ind=transport_ind,
            block_index=block_index,
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
