# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, xp
from qttools.toeplitz.circulant import (
    _detransform_matrix,
    _get_dft_matrix,
    _get_idft_matrix,
    _transform_matrix,
    check_circulant,
)


def _make_1D_block_circulant(
    a: NDArray,
    sections: int,
) -> NDArray:
    """Helper function to transform a matrix into a block circulant matrix with
    the given number of sections."""

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

    matrix = xp.zeros_like(a)
    for i in range(sections):
        shifted_blocks = blocks[-i:] + blocks[:-i]
        matrix[..., i * block_size : (i + 1) * block_size, :] = xp.concatenate(
            shifted_blocks, axis=-1
        )

    return matrix


def _make_2D_block_circulant(
    a: NDArray,
    sections_x: int,
    sections_y: int,
) -> NDArray:
    """Helper function to transform a matrix into a block circulant matrix with
    the given number of sections."""

    if a.shape[-1] % sections_x != 0:
        raise ValueError("The last dimension of a must be divisible by sections_x.")
    if a.shape[-1] % sections_y != 0:
        raise ValueError("The last dimension of a must be divisible by sections_y.")
    if a.shape[-1] % (sections_x * sections_y) != 0:
        raise ValueError(
            "The last dimension of a must be divisible by the section product."
        )

    if a.shape[-2] != a.shape[-1]:
        raise ValueError(
            "The second to last dimension of a must be equal to the last dimension of a."
        )

    block_size_x = a.shape[-1] // sections_x

    # make first circulant in the y direction
    for i in range(0, a.shape[-1], block_size_x):
        a[..., :block_size_x, i : i + block_size_x] = _make_1D_block_circulant(
            a[..., :block_size_x, i : i + block_size_x],
            sections=sections_y,
        )

    return _make_1D_block_circulant(a, sections=sections_x)


def test_dft_matrix(block_sections: int):
    """Test the properties of the DFT and IDFT matrices."""

    W = _get_dft_matrix(block_sections)
    W_inv = _get_idft_matrix(block_sections)

    test = W @ W_inv
    assert xp.allclose(test, xp.eye(block_sections))

    test = W_inv @ W
    assert xp.allclose(test, xp.eye(block_sections))

    # The DFT matrix should be unitary
    assert xp.allclose(W @ W.conj().T, xp.eye(block_sections))
    assert xp.allclose(W.conj().T @ W, xp.eye(block_sections))
    assert xp.allclose(W_inv @ W_inv.conj().T, xp.eye(block_sections))
    assert xp.allclose(W_inv.conj().T @ W_inv, xp.eye(block_sections))


def test_transform_1D_circulant(batch_size: int, block_size: int, block_sections: int):
    """Test the transformation of a 1D block circulant matrix to block diagonal form."""

    if block_size % block_sections != 0:
        pytest.skip("The block size must be divisible by the number of block sections.")

    rng = xp.random.default_rng(seed=0)
    if batch_size == 1:
        a = rng.random((block_size, block_size)) + 1j * rng.random(
            (block_size, block_size)
        )
    else:
        a = rng.random((batch_size, block_size, block_size)) + 1j * rng.random(
            (batch_size, block_size, block_size)
        )

    a_circulant = _make_1D_block_circulant(a, block_sections)

    if not check_circulant(a_circulant, block_sections):
        raise ValueError("The generated matrix is not block circulant.")

    a_diagonal = _transform_matrix(a_circulant, sections_x=block_sections)

    a_test = _detransform_matrix(a_diagonal, sections_x=block_sections)

    assert xp.allclose(a_circulant, a_test)


def test_transform_2D_circulant(
    batch_size: int, block_size: int, block_sections_x: int, block_sections_y: int
):
    """Test the transformation of a 2D block circulant matrix to block diagonal form."""

    if block_size % block_sections_x != 0:
        pytest.skip("The block size must be divisible by the number of block sections.")
    if block_size % block_sections_y != 0:
        pytest.skip("The block size must be divisible by the number of block sections.")
    if block_size % (block_sections_x * block_sections_y) != 0:
        pytest.skip("The block size must be divisible by the section product.")

    rng = xp.random.default_rng(seed=0)
    if batch_size == 1:
        a = rng.random((block_size, block_size)) + 1j * rng.random(
            (block_size, block_size)
        )
    else:
        a = rng.random((batch_size, block_size, block_size)) + 1j * rng.random(
            (batch_size, block_size, block_size)
        )

    a_circulant = _make_2D_block_circulant(
        a, sections_x=block_sections_x, sections_y=block_sections_y
    )

    a_diagonal = _transform_matrix(
        a_circulant, sections_x=block_sections_x, sections_y=block_sections_y
    )

    a_test = _detransform_matrix(
        a_diagonal, sections_x=block_sections_x, sections_y=block_sections_y
    )

    assert xp.allclose(a_circulant, a_test)
