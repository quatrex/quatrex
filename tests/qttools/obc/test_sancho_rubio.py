# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, xp
from qttools.obc import OBCMemoizer, SanchoRubio


def test_convergence(a_xx: tuple[NDArray, ...], contact: str):
    """Tests that the OBC return the correct result."""
    sancho_rubio = SanchoRubio()
    a_ji, a_ii, a_ij = a_xx
    x_ii = sancho_rubio(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert xp.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij))


def test_convergence_batch(a_xx: tuple[NDArray, ...], contact: str):
    """Tests that the OBC return the correct result."""
    sancho_rubio = SanchoRubio()
    a_ji, a_ii, a_ij = a_xx
    factors = xp.random.rand(10)
    a_ji = xp.stack([f * a_ji for f in factors])
    a_ii = xp.stack([f * a_ii for f in factors])
    a_ij = xp.stack([f * a_ij for f in factors])
    x_ii = sancho_rubio(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert xp.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij))


def test_max_iterations(a_xx: tuple[NDArray, ...]):
    """Tests that Sancho-Rubio raises Exception after max_iterations."""
    sancho_rubio = SanchoRubio(max_iterations=1, convergence_tol=1e-8)
    a_ji, a_ii, a_ij = a_xx
    with pytest.warns(RuntimeWarning):
        sancho_rubio(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=None)


def test_memoizer(a_xx: tuple[NDArray, ...], contact: str):
    """Tests that the Memoization works."""
    spectral = SanchoRubio()
    spectral = OBCMemoizer(spectral)
    a_ji, a_ii, a_ij = a_xx
    x_ii = spectral(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert xp.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), atol=1e-5)

    # Add a little noise to the input matrices.
    a_ji += xp.random.randn(*a_ji.shape)
    a_ii += xp.random.randn(*a_ii.shape)
    a_ij += xp.random.randn(*a_ij.shape)

    x_ii = spectral(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert xp.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), atol=1e-5)
