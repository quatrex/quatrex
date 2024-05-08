# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors. All rights reserved.
import numpy as np
import numpy.linalg as npla
import pytest

from qttools.obc import SanchoRubio


def test_convergence(a_ii: np.ndarray, a_ij: np.ndarray):
    """Tests that the OBC return the correct result."""
    sancho_rubio = SanchoRubio()
    x_ii = sancho_rubio(a_ii=a_ii, a_ij=a_ij, a_ji=a_ij.conj().T, contact=None)
    assert np.allclose(x_ii, npla.inv(a_ii - a_ij.conj().T @ x_ii @ a_ij))


def test_max_iterations(a_ii: np.ndarray, a_ij: np.ndarray):
    """Tests that Sancho-Rubio raises Exception after max_iterations."""
    sancho_rubio = SanchoRubio(max_iterations=1, convergence_tol=1e-8)
    with pytest.raises(RuntimeError):
        sancho_rubio(a_ii, a_ij, a_ij.conj().T, contact=None)
