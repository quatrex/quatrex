# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
import numpy as np
import pytest
from scipy import sparse

from quatrex.electrostatics import solver
from quatrex.electrostatics.solver import solve_linear_system


@pytest.fixture
def system() -> tuple[sparse.csr_matrix, np.ndarray]:
    """1D Laplacian with Dirichlet rows, as assembled for the Poisson problem."""
    n = 200
    matrix = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)).tolil()
    for i in (0, n - 1):
        matrix.rows[i], matrix.data[i] = [i], [1.0]
    rhs = np.random.default_rng(0).uniform(-1, 1, n)
    rhs[[0, -1]] = [0.5, -0.5]
    return matrix.tocsr(), rhs


def test_solve_linear_system(system):
    """Tests that the BiCGStab path solves the system."""
    matrix, rhs = system
    solution = solve_linear_system(matrix, rhs)
    assert np.allclose(matrix @ solution, rhs, atol=1e-4 * np.linalg.norm(rhs))


def test_solve_linear_system_fallback(system, monkeypatch):
    """Tests the direct fallback when BiCGStab does not converge."""
    matrix, rhs = system

    def failing_bicgstab(*args, **kwargs):
        return np.full_like(rhs, 1e6), 1000

    monkeypatch.setattr(solver.sparse.linalg, "bicgstab", failing_bicgstab)
    solution = solve_linear_system(matrix, rhs)
    assert np.allclose(matrix @ solution, rhs)
