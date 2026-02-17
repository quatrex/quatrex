# Copyright (c) 2025 ETH Zurich and the authors of the quatrex package.

import pytest

from qttools import xp
from quatrex.bandstructure.contact import extract_blocks, extract_sub_arrow_blocks


def test_extract_blocks():
    big_block_size = 6
    small_block_size = 2
    leading_shape = (3, 4)

    # Create a test matrix
    mat = xp.arange(
        xp.prod(xp.array(leading_shape)) * big_block_size * big_block_size
    ).reshape((*leading_shape, big_block_size, big_block_size))

    # Extract blocks
    blocks = extract_blocks(mat, small_block_size)

    small_blocks_per_side = big_block_size // small_block_size
    # Check shape
    expected_shape = (
        *leading_shape,
        small_blocks_per_side,
        small_blocks_per_side,
        small_block_size,
        small_block_size,
    )
    assert blocks.shape == expected_shape

    # Loop to check content
    for i in range(small_blocks_per_side):
        for j in range(small_blocks_per_side):
            expected_block = mat[
                ...,
                i * small_block_size : (i + 1) * small_block_size,
                j * small_block_size : (j + 1) * small_block_size,
            ]
            assert xp.allclose(blocks[..., i, j, :, :], expected_block)


@pytest.mark.parametrize("side", ["left", "right"])
def test_extract_sub_arrow_blocks(side):
    big_block_size = 6
    small_block_size = 2
    leading_shape = (3, 4)

    # Create test matrices
    mat_nm = xp.arange(
        xp.prod(xp.array(leading_shape)) * big_block_size * big_block_size
    ).reshape((*leading_shape, big_block_size, big_block_size))
    mat_mm = mat_nm + 1000
    mat_mn = mat_nm + 2000

    # Extract sub arrow blocks
    tiled_blocks = extract_sub_arrow_blocks(
        mat_nm, mat_mm, mat_mn, small_block_size, side=side
    )

    # Check shape
    total_small_blocks = 4 * (big_block_size // small_block_size) - 1
    expected_shape = (
        *leading_shape,
        total_small_blocks,
        small_block_size,
        small_block_size,
    )
    assert tiled_blocks.shape == expected_shape

    if side == "left":
        # Check first block (low left)
        expected_block = mat_nm[
            ..., big_block_size - small_block_size : big_block_size, 0:small_block_size
        ]
        assert xp.allclose(tiled_blocks[..., 0, :, :], expected_block)

        # Check middle index
        mid_index = total_small_blocks // 2
        expected_block = mat_mm[..., 0:small_block_size, 0:small_block_size]
        assert xp.allclose(tiled_blocks[..., mid_index, :, :], expected_block)

        # Check last block (up right)
        expected_block = mat_mn[
            ..., 0:small_block_size, big_block_size - small_block_size : big_block_size
        ]
        assert xp.allclose(tiled_blocks[..., -1, :, :], expected_block)
    else:
        # Check first block (lower left)
        expected_block = mat_mn[
            ..., big_block_size - small_block_size : big_block_size, 0:small_block_size
        ]
        assert xp.allclose(tiled_blocks[..., 0, :, :], expected_block)

        # Check middle index
        mid_index = total_small_blocks // 2
        expected_block = mat_mm[
            ...,
            big_block_size - small_block_size : big_block_size,
            big_block_size - small_block_size : big_block_size,
        ]
        assert xp.allclose(tiled_blocks[..., mid_index, :, :], expected_block)

        # Check last block (upper right)
        expected_block = mat_nm[
            ..., :small_block_size, big_block_size - small_block_size : big_block_size
        ]
        assert xp.allclose(tiled_blocks[..., -1, :, :], expected_block)
