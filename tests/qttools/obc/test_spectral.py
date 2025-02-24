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
    "residual_normalization",
)
def test_correctness(
    a_xx: tuple[NDArray, ...],
    nevp: NEVP,
    block_sections: int,
    x_ii_formula: str,
    contact: str,
    two_sided: bool,
    treat_pairwise: bool,
    residual_normalization: str | None,
):
    """Tests that the OBC return the correct result."""
    spectral = Spectral(
        nevp=nevp,
        block_sections=block_sections,
        x_ii_formula=x_ii_formula,
        two_sided=two_sided,
        treat_pairwise=treat_pairwise,
        residual_normalization=residual_normalization,
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
    "residual_normalization",
)
def test_correctness_batch(
    a_xx: tuple[NDArray, ...],
    nevp: NEVP,
    block_sections: int,
    x_ii_formula: str,
    contact: str,
    two_sided: bool,
    treat_pairwise: bool,
    residual_normalization: str,
):
    """Tests that the OBC return the correct result."""
    spectral = Spectral(
        nevp=nevp,
        block_sections=block_sections,
        x_ii_formula=x_ii_formula,
        two_sided=two_sided,
        treat_pairwise=treat_pairwise,
        residual_normalization=residual_normalization,
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


@pytest.mark.usefixtures(
    "block_size",
    "batch_size",
    "nevp",
    "block_sections",
)
def test_upscaling(
    block_size: int,
    batch_size: int,
    nevp: NEVP,
    block_sections: int,
):
    """Tests that the eigenmode upscaling works."""

    spectral = Spectral(nevp=nevp, block_sections=block_sections)

    rng = xp.random.default_rng()

    ws = rng.random((batch_size, block_size)) + 1j * rng.random(
        (batch_size, block_size)
    )

    vs = rng.random(
        (batch_size, block_size // block_sections, block_size)
    ) + 1j * rng.random((batch_size, block_size // block_sections, block_size))

    _, vs_upscaled = spectral._upscale_eigenmodes(ws, vs)

    vs_upscaled_ref = xp.zeros((batch_size, block_size, block_size), dtype=vs.dtype)
    for i in range(batch_size):
        for j, w in enumerate(ws[i]):
            vs_upscaled_ref[i, :, j] = xp.kron(
                xp.array([w**n for n in range(block_sections)]), vs[i, :, j]
            )
            vs_upscaled_ref[i, :, j] /= xp.linalg.norm(vs_upscaled_ref[i, :, j])

    assert xp.allclose(vs_upscaled, vs_upscaled_ref)


@pytest.mark.usefixtures(
    "nevp",
    "two_sided",
)
def test_compute_dE_dk(
    a_xx: tuple[NDArray, ...],
    nevp: NEVP,
    two_sided: bool,
):
    """Tests that the eigenmode upscaling works."""

    if a_xx[0].ndim == 2:
        a_xx = tuple(a[xp.newaxis, ...] for a in a_xx)

    batch_size = a_xx[0].shape[0]
    block_size = a_xx[0].shape[-1]
    b = len(a_xx) // 2

    spectral = Spectral(nevp=nevp, two_sided=two_sided)

    rng = xp.random.default_rng()

    ws = rng.random((batch_size, block_size)) + 1j * rng.random(
        (batch_size, block_size)
    )

    vrs = rng.random((batch_size, block_size, block_size)) + 1j * rng.random(
        (batch_size, block_size, block_size)
    )

    if two_sided:
        vls = rng.random((batch_size, block_size, block_size)) + 1j * rng.random(
            (batch_size, block_size, block_size)
        )
    else:
        vls = None

    dEk_dk = spectral._compute_dE_dk(ws, vrs, a_xx, vls=vls)

    dEk_dk_ref = xp.zeros_like(ws)
    for i in range(batch_size):
        for j, w in enumerate(ws[i]):
            a = -sum(
                (1j * n) * w**n * a_xn[i] for a_xn, n in zip(a_xx, range(-b, b + 1))
            )

            if two_sided:
                phi_right = vrs[i, :, j]
                phi_left = vls[i, :, j]
            else:
                phi_right = vrs[i, :, j]
                phi_left = vrs[i, :, j]

            dEk_dk_ref[i, j] = (phi_left.conj().T @ a @ phi_right) / (
                phi_left.conj().T @ phi_right
            )

    assert xp.allclose(dEk_dk, dEk_dk_ref)
