# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, sparse, xp

from .conftest import WFSolverSpec


def _assemble_banded_matrix(n: int, m: int, format: str) -> sparse.spmatrix:
    """Assembles a random diagonally dominant banded matrix.

    Parameters
    ----------
    n : int
        Number of rows/columns in the square matrix.
    m : int
        Number of columns in the right-hand side matrix.
    format : str
        The format of the sparse matrix (e.g., 'csr', 'csc', 'coo').

    Returns
    -------
    a : sparse.spmatrix
        The generated sparse banded system matrix in the specified
        format.

    """
    # NOTE: Important that the number of diagonals is not too small,
    # otherwise the absence of pivoting in the Thomas algorithm will
    # lead to large errors.
    num_diags = max(n // 2, 1)
    offsets = xp.arange(-num_diags // 2, num_diags // 2)

    a = sparse.diags(
        # This roughly mimics the decay of off-diagonal elements in a
        # Hamiltonian matrix.
        xp.exp(-5 * xp.abs(offsets)),
        offsets=offsets,
        shape=(n, n),
        format=format,
    )
    a.data = a.data * (
        xp.random.rand(*a.data.shape) + 1j * xp.random.rand(*a.data.shape)
    )

    # Make the matrix strongly diagonally dominant.
    a.setdiag(xp.abs(a.diagonal() * 10))

    return a


def _assemble_system(
    n: int,
    m: int,
    sparse_format: str,
    order="C",
    use_banded=False,
) -> tuple[sparse.spmatrix, NDArray]:
    """Assembles a random sparse system of equations.

    Parameters
    ----------
    n : int
        Number of rows/columns in the square matrix.
    m : int
        Number of columns in the right-hand side matrix.
    sparse_format : str
        The format of the sparse matrix (e.g., 'csc', 'csr', 'coo').
    order : str, optional
        The memory order of the right-hand side matrix, 'C' for row-major or '
        'F' for column-major. Default is 'C'.
    use_banded : bool, optional Whether to assemble a banded matrix
        instead of a general sparse matrix. Default is False. If True,
        the matrix will have a bandwidth of approximately n//2, which is
        suitable for testing the Thomas solver. If False, a general
        sparse matrix with random sparsity pattern will be generated,
        which is better for testing the other solvers.

    Returns
    -------
    a : sparse.spmatrix
        The generated sparse system matrix in the specified format.
    b : NDArray
        The right-hand side matrix with shape (n, m).

    """
    if use_banded:
        a = _assemble_banded_matrix(n, m, sparse_format)
    else:
        a = sparse.random(n, n, density=0.1, format=sparse_format, dtype=float)
        a += 1j * sparse.random(n, n, density=0.1, format=sparse_format, dtype=float)
        a += sparse.diags([2.0] * n, format=sparse_format, dtype=float)

    b = xp.random.rand(n, m) + 1j * xp.random.rand(n, m)
    b = xp.asarray(b, order=order)
    return a, b


def test_solve(n: int, m: int, solver_spec: WFSolverSpec):
    """Tests the wave function solver."""
    a, b = _assemble_system(
        n,
        m,
        sparse_format=solver_spec.sparse_format,
        order=solver_spec.order,
        use_banded=solver_spec.use_banded,
    )
    solver = solver_spec.solver_type()

    x = solver.solve(a, b, **solver_spec.solve_kwargs)

    assert x.shape == (n, m)
    assert xp.allclose(a @ x, b, atol=1e-6)


def test_reuse_analysis(n: int, m: int, solver_spec: WFSolverSpec):
    """Tests the wave function solver with reuse of analysis."""
    if not solver_spec.supports_reuse_analysis:
        pytest.skip(
            f"{solver_spec.solver_type.__name__} does not support reuse of analysis."
        )

    a, b = _assemble_system(
        n,
        m,
        sparse_format=solver_spec.sparse_format,
        order=solver_spec.order,
        use_banded=solver_spec.use_banded,
    )
    solver = solver_spec.solver_type()

    x1 = solver.solve(a, b, **solver_spec.solve_kwargs)
    assert xp.allclose(a @ x1, b, atol=1e-6)

    # Reuse the analysis phase.
    a.data[:] *= 10  # Modify the matrix to change the solution.

    __, b = _assemble_system(
        n,
        2 * m,
        sparse_format=solver_spec.sparse_format,
        order=solver_spec.order,
        use_banded=solver_spec.use_banded,
    )

    x2 = solver.solve(a, b, reuse_analysis=True, **solver_spec.solve_kwargs)
    assert xp.allclose(a @ x2, b, atol=1e-6)


def test_reuse_factorization(n: int, m: int, solver_spec: WFSolverSpec):
    """Tests the wave function solver with reuse of factorization."""
    if not solver_spec.supports_reuse_factorization:
        pytest.skip(
            f"{solver_spec.solver_type.__name__} does not support reuse of factorization."
        )

    a, b = _assemble_system(
        n,
        m,
        sparse_format=solver_spec.sparse_format,
        order=solver_spec.order,
        use_banded=solver_spec.use_banded,
    )
    solver = solver_spec.solver_type()

    x1 = solver.solve(a, b, **solver_spec.solve_kwargs)
    assert xp.allclose(a @ x1, b, atol=1e-6)

    __, b = _assemble_system(
        n,
        2 * m,
        sparse_format=solver_spec.sparse_format,
        order=solver_spec.order,
        use_banded=solver_spec.use_banded,
    )

    x2 = solver.solve(
        a,
        b,
        reuse_analysis=solver_spec.supports_reuse_analysis,
        reuse_factorization=True,
        **solver_spec.solve_kwargs,
    )
    assert xp.allclose(a @ x2, b, atol=1e-6)


def test_real_symmetric_system(n: int, m: int, solver_spec: WFSolverSpec):
    """Tests the wave function solver on a real symmetric system."""
    if not solver_spec.supports_symmetric:
        pytest.skip(
            f"{solver_spec.solver_type.__name__} does not support symmetric systems."
        )

    a, b = _assemble_system(
        n,
        m,
        sparse_format=solver_spec.sparse_format,
        order=solver_spec.order,
        use_banded=solver_spec.use_banded,
    )

    # Drop the imaginary part and symmetrize the system matrix.
    a = a.astype(xp.float64)
    a = a + a.T
    b = b.astype(xp.float64, order=solver_spec.order)

    solver = solver_spec.solver_type(
        matrix_type="real_symmetric_indefinite", matrix_view="upper"
    )
    x = solver.solve(
        sparse.triu(a, format=solver_spec.sparse_format), b, **solver_spec.solve_kwargs
    )

    assert x.shape == (n, m)
    assert xp.allclose(a @ x, b, atol=1e-6)


def test_complex_hermitian_system(n: int, m: int, solver_spec: WFSolverSpec):
    """Tests the wave function solver on a complex Hermitian system."""
    if not solver_spec.supports_hermitian:
        pytest.skip(
            f"{solver_spec.solver_type.__name__} does not support Hermitian systems."
        )

    a, b = _assemble_system(
        n,
        m,
        sparse_format=solver_spec.sparse_format,
        order=solver_spec.order,
        use_banded=solver_spec.use_banded,
    )

    # Make the system matrix Hermitian.
    a = a.conj().T + a

    solver = solver_spec.solver_type(
        matrix_type="complex_hermitian_indefinite", matrix_view="upper"
    )
    x = solver.solve(
        sparse.triu(a, format=solver_spec.sparse_format), b, **solver_spec.solve_kwargs
    )

    assert x.shape == (n, m)
    assert xp.allclose(a @ x, b, atol=1e-6)
