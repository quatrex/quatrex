# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.
import numpy as np
import numpy.linalg as npla
import pytest

from qttools.obc.sancho_rubio import SanchoRubio


def test_convergence(a_xx: tuple[np.ndarray], contact: str):
    """Tests that the OBC return the correct result."""
    sancho_rubio = SanchoRubio()
    a_ji, a_ii, a_ij = a_xx
    x_ii = sancho_rubio(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert np.allclose(x_ii, npla.inv(a_ii - a_ji @ x_ii @ a_ij))


def test_convergence_batch(a_xx: tuple[np.ndarray], contact: str):
    """Tests that the OBC return the correct result."""
    sancho_rubio = SanchoRubio()
    a_ji, a_ii, a_ij = a_xx
    factors = np.random.rand(10)
    a_ji = np.stack([f * a_ji for f in factors])
    a_ii = np.stack([f * a_ii for f in factors])
    a_ij = np.stack([f * a_ij for f in factors])
    x_ii = sancho_rubio(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=contact)
    assert np.allclose(x_ii, npla.inv(a_ii - a_ji @ x_ii @ a_ij))


def test_max_iterations(a_xx: tuple[np.ndarray]):
    """Tests that Sancho-Rubio raises Exception after max_iterations."""
    sancho_rubio = SanchoRubio(max_iterations=1, convergence_tol=1e-8)
    a_ji, a_ii, a_ij = a_xx
    with pytest.warns(RuntimeWarning):
        sancho_rubio(a_ii=a_ii, a_ij=a_ij, a_ji=a_ji, contact=None)
