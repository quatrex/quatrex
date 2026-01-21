# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.

import importlib.util

import numpy as np
import pytest

from qttools import NDArray, sparse, xp
from qttools.wave_function_solver import MUMPS, SuperLU, cuDSS

mumps_available = importlib.util.find_spec("mumps") is not None
nvmath_available = importlib.util.find_spec("nvmath") is not None


def _assemble_system(
    n: int,
    m: int,
    format: str,
    order="C",
) -> tuple[sparse.csc_matrix, NDArray]:
    """Assembles a random sparse system of equations.

    Parameters
    ----------
    n : int
        Number of rows/columns in the square matrix.
    m : int
        Number of columns in the right-hand side matrix.
    format : str
        The format of the sparse matrix (e.g., 'csc', 'csr', 'coo').
    order : str, optional
        The memory order of the right-hand side matrix, 'C' for row-major or '
        'F' for column-major. Default is 'C'.

    Returns
    -------
    a : sparse.csc_matrix
        The generated sparse system matrix in the specified format.
    b : NDArray
        The right-hand side matrix with shape (n, m).

    """
    a = sparse.random(n, n, density=0.1, format=format, dtype=float)
    a += 1j * sparse.random(n, n, density=0.1, format=format, dtype=float)
    a += sparse.diags([2.0] * n, format=format, dtype=float)
    b = xp.ones((n, m), dtype=xp.complex128, order=order)
    return a, b


class TestLU:
    """Tests for the LU wave function solver."""

    def test_solve(self, n: int, m: int):
        """Tests the wave function solver."""
        a, b = _assemble_system(n, m, format="csc")
        solver = SuperLU()

        x = solver.solve(a, b)

        assert x.shape == (n, m)
        assert xp.allclose(a @ x, b, atol=1e-6)


@pytest.mark.skipif(not mumps_available, reason="Requires python-mumps package")
@pytest.mark.skipif(xp.__name__ != "numpy", reason="Requires numpy backend")
class TestMUMPS:
    def test_solve(self, n: int, m: int):
        """Tests the wave function solver."""
        a, b = _assemble_system(n, m, format="coo")

        solver = MUMPS(reuse_analysis=False)
        x = solver.solve(a, b)

        assert x.shape == (n, m)
        assert np.allclose(a @ x, b, atol=1e-6)

    def test_reuse_analysis(self, n: int, m: int):
        """Tests the wave function solver with reuse of analysis."""
        a, b = _assemble_system(n, m, format="coo")

        solver = MUMPS(reuse_analysis=True)
        x1 = solver.solve(a, b)
        assert np.allclose(a @ x1, b, atol=1e-6)

        # Reuse the analysis phase.
        a.data[:] *= 10  # Modify the matrix to change the solution.

        __, b = _assemble_system(n, 2 * m, format="coo")

        x2 = solver.solve(a, b)
        assert np.allclose(a @ x2, b, atol=1e-6)

    def test_explicit_ordering(self, n: int, m: int):
        """Tests the wave function solver with explicit ordering."""
        a, b = _assemble_system(n, m, format="coo")

        solver = MUMPS(ordering="scotch")
        x = solver.solve(a, b)

        assert x.shape == (n, m)
        assert np.allclose(a @ x, b, atol=1e-6)


@pytest.mark.skipif(not nvmath_available, reason="Requires nvmath-python package")
@pytest.mark.skipif(xp.__name__ != "cupy", reason="Requires cupy backend")
class TestcuDSS:

    def test_solve(self, n: int, m: int):
        """Tests the wave function solver."""
        a, b = _assemble_system(n, m, format="csr", order="F")

        solver = cuDSS()
        x = solver.solve(a, b)

        assert x.shape == (n, m)
        assert xp.allclose(a @ x, b, atol=1e-6)

    def test_explicit_reset_operands(self, n: int, m: int):
        """Tests the wave function solver with explicit reset of operands."""
        a, b = _assemble_system(n, m, format="csr", order="F")

        solver = cuDSS(explicitely_reset_operands="a,b")
        x1 = solver.solve(a, b)
        assert xp.allclose(a @ x1, b, atol=1e-6)

        # Modify the matrix to change the solution.
        a.data[:] *= 10

        __, b = _assemble_system(n, 2 * m, format="csr", order="F")
        x2 = solver.solve(a, b)
        assert xp.allclose(a @ x2, b, atol=1e-6)

    def test_implicit_reset_operands(self, n: int, m: int):
        """Tests the wave function solver without explicit reset of operands."""
        a, b = _assemble_system(n, m, format="csr", order="F")

        solver = cuDSS(explicitely_reset_operands="b")
        x1 = solver.solve(a, b)
        assert xp.allclose(a @ x1, b, atol=1e-6)

        # Modify the matrix to change the solution.
        a.data[:] *= 10

        __, b = _assemble_system(n, 2 * m, format="csr", order="F")
        # The solver should still work without explicit reset.
        x2 = solver.solve(a, b)
        assert xp.allclose(a @ x2, b, atol=1e-6)
