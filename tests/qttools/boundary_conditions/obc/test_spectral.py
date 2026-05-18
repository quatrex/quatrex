# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, xp
from qttools.boundary_conditions.obc import OBCSystem, Spectral
from qttools.nevp import NEVP


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

    a_ji, a_ii, a_ij = a_xx

    # repeat the blocks to create a layer with block_sections
    block_size = a_ii.shape[-1] * block_sections
    batch_size = a_ii.shape[0] if a_ii.ndim == 3 else 1
    a_ii_hat = xp.zeros((batch_size, block_size, block_size), dtype=a_ii.dtype)
    a_ij_hat = xp.zeros_like(a_ii_hat)
    a_ji_hat = xp.zeros_like(a_ii_hat)

    for i in range(block_sections):
        a_ii_hat[
            :,
            i * a_ii.shape[-1] : (i + 1) * a_ii.shape[-1],
            i * a_ii.shape[-1] : (i + 1) * a_ii.shape[-1],
        ] = a_ii
        if i < block_sections - 1:
            a_ii_hat[
                :,
                i * a_ij.shape[-1] : (i + 1) * a_ij.shape[-1],
                (i + 1) * a_ij.shape[-1] : (i + 2) * a_ij.shape[-1],
            ] = a_ij
            a_ii_hat[
                :,
                (i + 1) * a_ji.shape[-1] : (i + 2) * a_ji.shape[-1],
                i * a_ji.shape[-1] : (i + 1) * a_ji.shape[-1],
            ] = a_ji

    a_ji_hat[
        :,
        0 : a_ji.shape[-1],
        (block_sections - 1) * a_ji.shape[-1] : block_sections * a_ji.shape[-1],
    ] = a_ji
    a_ij_hat[
        :,
        (block_sections - 1) * a_ij.shape[-1] : block_sections * a_ij.shape[-1],
        0 : a_ij.shape[-1],
    ] = a_ij

    return a_ji_hat, a_ii_hat, a_ij_hat


def test_correctness(
    a_xx: tuple[NDArray, ...],
    nevp: NEVP,
    block_sections: int,
):
    """Tests that the OBC return the correct result."""
    spectral = Spectral(
        nevp=nevp,
        block_sections=block_sections,
    )
    a_ji, a_ii, a_ij = _make_periodic(a_xx, block_sections)
    a_ji, a_ii, a_ij = a_ji[0], a_ii[0], a_ij[0]
    x_ii = spectral(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact="")
    assert xp.all(
        (
            xp.linalg.norm(
                x_ii - xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), axis=(-1, -2)
            )
            / xp.linalg.norm(x_ii, axis=(-1, -2))
        )
        < 5e-3
    )


def test_correctness_batch(
    a_xx: tuple[NDArray, ...],
    nevp: NEVP,
    block_sections: int,
):
    """Tests that the OBC return the correct result."""
    spectral = Spectral(
        nevp=nevp,
        block_sections=block_sections,
        residual_tolerance=1e-1,
        max_decay=20,
    )
    a_ji, a_ii, a_ij = _make_periodic(a_xx, block_sections)
    x_ii = spectral(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact="")
    assert xp.all(
        (
            xp.linalg.norm(
                x_ii - xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), axis=(-1, -2)
            )
            / xp.linalg.norm(x_ii, axis=(-1, -2))
        )
        < 5e-3
    )


def test_memoizer(
    a_xx: tuple[NDArray, ...], nevp: NEVP, block_sections: int, memoization_mode: str
):
    """Tests that the Memoization works."""
    spectral = Spectral(
        nevp=nevp,
        block_sections=block_sections,
        residual_tolerance=1e-1,
        max_decay=20,
    )

    obc_system = OBCSystem(spectral, memoization_mode=memoization_mode)

    # Add a little noise to the input matrices.
    a_ji, a_ii, a_ij = a_xx
    a_ji_hat = a_ji * (1 + 1e-6)
    a_ii_hat = a_ii * (1 + 1e-6)
    a_ij_hat = a_ij * (1 + 1e-6)

    a_ji, a_ii, a_ij = _make_periodic(a_xx, block_sections)
    a_ji_hat, a_ii_hat, a_ij_hat = _make_periodic(
        (a_ji_hat, a_ii_hat, a_ij_hat), block_sections
    )

    x_ii, *__ = obc_system((a_ii, a_ij, a_ji), contact="contact")
    assert xp.all(
        (
            xp.linalg.norm(
                x_ii - xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), axis=(-1, -2)
            )
            / xp.linalg.norm(x_ii, axis=(-1, -2))
        )
        < 5e-3
    )

    x_ii, *__ = obc_system((a_ii_hat, a_ij_hat, a_ji_hat), contact="contact")
    assert xp.all(
        (
            xp.linalg.norm(
                x_ii - xp.linalg.inv(a_ii_hat - a_ji_hat @ x_ii @ a_ij_hat),
                axis=(-1, -2),
            )
            / xp.linalg.norm(x_ii, axis=(-1, -2))
        )
        < 5e-3
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


def test_compute_dE_dk(
    a_xx: tuple[NDArray, ...],
    nevp: NEVP,
):
    """Tests that the eigenmode upscaling works."""

    if a_xx[0].ndim == 2:
        a_xx = tuple(a[xp.newaxis, ...] for a in a_xx)

    batch_size = a_xx[0].shape[0]
    block_size = a_xx[0].shape[-1]
    b = len(a_xx) // 2

    spectral = Spectral(nevp=nevp)

    rng = xp.random.default_rng()

    ws = rng.random((batch_size, block_size)) + 1j * rng.random(
        (batch_size, block_size)
    )

    vrs = rng.random((batch_size, block_size, block_size)) + 1j * rng.random(
        (batch_size, block_size, block_size)
    )

    dEk_dk = spectral._compute_dE_dk(ws, vrs, a_xx)

    dEk_dk_ref = xp.zeros_like(ws)
    for i in range(batch_size):
        for j, w in enumerate(ws[i]):
            a = -sum(
                (1j * n) * w**n * a_xn[i] for a_xn, n in zip(a_xx, range(-b, b + 1))
            )

            phi_right = vrs[i, :, j]
            phi_left = vrs[i, :, j]

            dEk_dk_ref[i, j] = phi_left.conj().T @ a @ phi_right

    assert xp.allclose(dEk_dk, dEk_dk_ref)


def test_injected(
    a_xx: tuple[NDArray, ...],
    nevp: NEVP,
    block_sections: int,
):
    """Tests that the OBC return the correct result when injected modes are
    requested."""

    # skip the test if `Beyn` is used
    if nevp.__class__.__name__ == "Beyn":
        pytest.skip(
            "Beyn is very sensitive and the test fails for the carbon nanotube example."
            "We need to find a better example to test the injection vector calculation with Beyn."
        )

    spectral = Spectral(
        nevp=nevp,
        block_sections=block_sections,
        residual_tolerance=1e-1,
        max_decay=20,
    )
    a_ji, a_ii, a_ij = _make_periodic(a_xx, block_sections)
    x_ii, phi_surfaces = spectral(
        a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact="", return_injected=True
    )

    assert xp.all(
        (
            xp.linalg.norm(
                x_ii - xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), axis=(-1, -2)
            )
            / xp.linalg.norm(x_ii, axis=(-1, -2))
        )
        < 5e-3
    )

    # TODO: There is no simple way to check the correctnes of the injected
    # modes, but we can check that the code produces the same injected modes in
    # a batch and non-batch setting.
    for i in range(a_ii.shape[0]):
        __, phi_surface = spectral(
            a_ii=a_ii[i], a_ij=a_ij[i], a_ji=a_ji[i], contact="", return_injected=True
        )
        assert xp.allclose(phi_surface, phi_surfaces[i])
