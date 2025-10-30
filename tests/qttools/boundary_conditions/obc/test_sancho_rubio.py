# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, xp
from qttools.boundary_conditions.obc import OBCSystem, SanchoRubio


def test_convergence(a_xx: tuple[NDArray, ...]):
    """Tests that the OBC return the correct result."""
    sancho_rubio = SanchoRubio(convergence_tol=1e-10)
    a_ji, a_ii, a_ij = a_xx
    a_ji, a_ii, a_ij = a_ji[0], a_ii[0], a_ij[0]
    assert a_ji.ndim == a_ii.ndim == a_ij.ndim == 2
    x_ii = sancho_rubio(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact="")
    assert xp.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij))


def test_convergence_batch(a_xx: tuple[NDArray, ...]):
    """Tests that the OBC return the correct result."""
    sancho_rubio = SanchoRubio(convergence_tol=1e-10)
    a_ji, a_ii, a_ij = a_xx
    assert a_ji.ndim == a_ii.ndim == a_ij.ndim == 3
    x_ii = sancho_rubio(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact="")
    assert xp.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij))


def test_max_iterations(a_xx: tuple[NDArray, ...]):
    """Tests that Sancho-Rubio raises Exception after max_iterations."""
    sancho_rubio = SanchoRubio(max_iterations=1, convergence_tol=1e-8)
    a_ji, a_ii, a_ij = a_xx
    with pytest.warns(RuntimeWarning):
        sancho_rubio(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact="")


def test_memoizer(a_xx: tuple[NDArray, ...]):
    """Tests that the Memoization works."""
    sacho = SanchoRubio(convergence_tol=1e-10)
    obc_system = OBCSystem(sacho)
    a_ji, a_ii, a_ij = a_xx
    _, _, x_ii = obc_system((a_ii, a_ij, a_ji), contact="contact")
    assert xp.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), atol=1e-5)

    # Add a little noise to the input matrices.
    a_ji = a_ji * (1 + 1e-6)
    a_ii = a_ii * (1 + 1e-6)
    a_ij = a_ij * (1 + 1e-6)

    _, _, x_ii = obc_system((a_ii, a_ij, a_ji), contact="contact")
    assert xp.allclose(x_ii, xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij), atol=1e-5)
