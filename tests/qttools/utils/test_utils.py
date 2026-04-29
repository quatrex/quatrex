# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
import pytest

from qttools import NDArray, xp
from quatrex.core.utils import expand_periodic_superblocks, get_periodic_superblocks


def _test_periodicity(
    a_ji: NDArray,
    a_ii: NDArray,
    a_ij: NDArray,
    block_sections: int,
):
    """Test that the periodic superblock structure is correctly constructed from the given blocks."""

    block_size = a_ii.shape[-1]

    m_ji, m_ii, m_ij = get_periodic_superblocks(
        a_ji=a_ji,
        a_ii=a_ii,
        a_ij=a_ij,
        block_sections=block_sections,
    )

    small_block_size = block_size // block_sections
    a = xp.concatenate(
        [a_ii[..., :small_block_size, :], a_ij[..., :small_block_size, :]],
        axis=-1,
    )

    for i in range(1, block_sections):
        a = xp.concatenate(
            [
                a_ii[
                    ...,
                    i * small_block_size : (i + 1) * small_block_size,
                    :small_block_size,
                ],
                a,
            ],
            axis=-1,
        )
    for i in range(block_sections):
        a = xp.concatenate(
            [
                a_ji[
                    ...,
                    i * small_block_size : (i + 1) * small_block_size,
                    :small_block_size,
                ],
                a,
            ],
            axis=-1,
        )

    m = xp.concatenate(
        [m_ji, m_ii, m_ij],
        axis=-1,
    )

    for i in range(block_sections * small_block_size, small_block_size):
        end = min(
            3 * block_sections * small_block_size - i,
            3 * block_sections * small_block_size,
        )
        assert xp.allclose(
            m[..., i : (i + small_block_size), i : i + end], a[..., :, :end]
        )


@pytest.mark.parametrize(
    "block_size, block_sections",
    [
        (4, 2),
        (9, 3),
        (11, 1),
    ],
)
@pytest.mark.parametrize(
    "batch_shape",
    [
        None,
        (2,),
        (3, 3),
    ],
)
def test_expand_periodic_superblocks(
    block_size: int,
    block_sections: int,
    batch_shape: tuple | None,
):
    """Test that the periodic superblock structure is correctly constructed."""

    rng = xp.random.default_rng(0)
    if batch_shape is not None:
        a_ii = rng.random((*batch_shape, block_size, block_size))
        a_ij = rng.random((*batch_shape, block_size, block_size))
        a_ji = rng.random((*batch_shape, block_size, block_size))
    else:
        a_ii = rng.random((block_size, block_size))
        a_ij = rng.random((block_size, block_size))
        a_ji = rng.random((block_size, block_size))

    _test_periodicity(
        a_ji=a_ji,
        a_ii=a_ii,
        a_ij=a_ij,
        block_sections=block_sections,
    )


@pytest.mark.parametrize(
    "block_size, block_sections",
    [
        (4, 2),
        (9, 3),
        (11, 1),
    ],
)
@pytest.mark.parametrize(
    "repetitions",
    [
        2,
        3,
    ],
)
@pytest.mark.parametrize(
    "batch_shape",
    [
        None,
        (2,),
        (3, 3),
    ],
)
def test_get_periodic_superblocks_ref(
    block_size: int,
    block_sections: int,
    repetitions: int,
    batch_shape: tuple | None,
):
    """Test that the expanded periodic superblock structure
    is correctly constructed."""

    rng = xp.random.default_rng(0)
    if batch_shape is not None:
        a_ii = rng.random((*batch_shape, block_size, block_size))
        a_ij = rng.random((*batch_shape, block_size, block_size))
        a_ji = rng.random((*batch_shape, block_size, block_size))
    else:
        a_ii = rng.random((block_size, block_size))
        a_ij = rng.random((block_size, block_size))
        a_ji = rng.random((block_size, block_size))

    m_ji_test, m_ii_test, m_ij_test = expand_periodic_superblocks(
        a_ji=a_ji,
        a_ii=a_ii,
        a_ij=a_ij,
        block_sections=block_sections,
        repetitions=repetitions,
    )

    a_ii_test = m_ii_test[..., :block_size, :block_size]
    a_ji_test = m_ii_test[..., block_size : 2 * block_size, :block_size]
    a_ij_test = m_ii_test[..., :block_size, block_size : 2 * block_size]

    _test_periodicity(
        a_ji=a_ji_test,
        a_ii=a_ii_test,
        a_ij=a_ij_test,
        block_sections=block_sections,
    )

    assert xp.allclose(m_ji_test[..., :block_size, -block_size:], a_ji_test)
    assert xp.allclose(m_ij_test[..., -block_size:, :block_size], a_ij_test)

    m_ji_test[..., :block_size, -block_size:] = 0
    m_ij_test[..., -block_size:, :block_size] = 0

    assert xp.allclose(m_ji_test, xp.zeros_like(m_ji_test))
    assert xp.allclose(m_ij_test, xp.zeros_like(m_ij_test))
