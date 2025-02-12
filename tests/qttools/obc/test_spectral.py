# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, xp
from qttools.datastructures.dsbsparse import _block_view
from qttools.nevp import NEVP
from qttools.obc import OBCMemoizer, Spectral


def _make_periodic(
    a_xx: tuple[NDArray, ...], block_sections: int
) -> tuple[NDArray, ...]:
    """Enforces that the layer has periodic subblocks.

    Parameters
    ----------
    a_xx : tuple[NDArray, ...]
        The boundary blocks.
    block_sections : int
        The number of block sections.

    Returns
    -------
    a_xx : tuple[NDArray, ...]
        The boundary blocks with periodic subblocks.

    """
    if block_sections == 1:
        return a_xx
    layer = xp.concatenate(a_xx, axis=-1)
    view = _block_view(_block_view(layer, -1, 3 * block_sections), -2, block_sections)
    # Make periodic layers.
    for i in range(block_sections):
        view[i, :] = xp.roll(view[0, :], i, axis=0)

    layer = xp.concatenate(xp.concatenate(view, axis=-2), axis=-1)
    return xp.array_split(layer, 3, axis=-1)


@pytest.mark.usefixtures(
    "nevp",
    "x_ii_formula",
    "block_sections",
    "two_sided",
    "treat_pairwise",
)
def test_correctness(
    a_xx: tuple[NDArray, ...],
    nevp: NEVP,
    block_sections: int,
    x_ii_formula: str,
    contact: str,
    two_sided: bool,
    treat_pairwise: bool,
):
    """Tests that the OBC return the correct result."""
    spectral = Spectral(
        nevp=nevp,
        block_sections=block_sections,
        x_ii_formula=x_ii_formula,
        two_sided=two_sided,
        treat_pairwise=treat_pairwise,
    )
    a_ji, a_ii, a_ij = _make_periodic(a_xx, block_sections)
    x_ii = spectral(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert xp.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), atol=1e-5)


@pytest.mark.usefixtures(
    "nevp",
    "x_ii_formula",
    "block_sections",
    "two_sided",
    "treat_pairwise",
)
def test_correctness_batch(
    a_xx: tuple[NDArray, ...],
    nevp: NEVP,
    block_sections: int,
    x_ii_formula: str,
    contact: str,
    two_sided: bool,
    treat_pairwise: bool,
):
    """Tests that the OBC return the correct result."""
    spectral = Spectral(
        nevp=nevp,
        block_sections=block_sections,
        x_ii_formula=x_ii_formula,
        two_sided=two_sided,
        treat_pairwise=treat_pairwise,
    )
    a_ji, a_ii, a_ij = _make_periodic(a_xx, block_sections)
    a_ji = xp.stack([a_ji for __ in range(10)])
    a_ii = xp.stack([a_ii for __ in range(10)])
    a_ij = xp.stack([a_ij for __ in range(10)])
    x_ii = spectral(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert xp.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), atol=1e-5)


@pytest.mark.usefixtures(
    "nevp",
    "x_ii_formula",
    "block_sections",
)
def test_memoizer(
    a_xx: tuple[NDArray, ...],
    nevp: NEVP,
    block_sections: int,
    x_ii_formula: str,
    contact: str,
):
    """Tests that the Memoization works."""
    spectral = Spectral(
        nevp=nevp, block_sections=block_sections, x_ii_formula=x_ii_formula
    )
    spectral = OBCMemoizer(spectral)
    a_ji, a_ii, a_ij = _make_periodic(a_xx, block_sections)
    x_ii = spectral(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert xp.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), atol=1e-5)
    # Add a little noise to the input matrices.
    a_ji += xp.random.randn(*a_ji.shape)
    a_ii += xp.random.randn(*a_ii.shape)
    a_ij += xp.random.randn(*a_ij.shape)
    a_ji, a_ii, a_ij = _make_periodic((a_ji, a_ii, a_ij), block_sections)

    x_ii = spectral(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert xp.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), atol=2e-5)
