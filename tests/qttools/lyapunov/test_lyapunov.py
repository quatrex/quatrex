# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numpy as np

from qttools import NDArray, xp
from qttools.lyapunov import LyapunovMemoizer, LyapunovSolver, Vectorize


def test_correctness(inputs: tuple[NDArray, NDArray], lyapunov_solver: LyapunovSolver):
    """Tests that the Lyapunov solver returns the correct result."""
    a, q = inputs
    x = lyapunov_solver(a, q, "contact")

    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)


def test_correctness_batch(
    inputs: tuple[NDArray, NDArray], lyapunov_solver: LyapunovSolver
):
    """Tests that the batched Lyapunov solver returns the correct result."""
    a, q = inputs
    a = xp.stack([a for __ in range(10)])
    q = xp.stack([q for __ in range(10)])

    x = lyapunov_solver(a, q, "contact")

    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)


def test_memoizer(inputs: tuple[NDArray, NDArray], lyapunov_solver: LyapunovSolver):
    """Tests that the Lyapunov memoizer works."""
    a, q = inputs
    lyapunov_solver = LyapunovMemoizer(lyapunov_solver)
    x = lyapunov_solver(a, q, "contact")
    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)

    # Add a little noise to the input matrices.
    a, q = inputs
    a += 1e-3 * xp.random.randn(*a.shape)
    q += 1e-3 * xp.random.randn(*q.shape)

    x = lyapunov_solver(a, q, "contact")
    assert xp.allclose(x, a @ x @ a.conj().swapaxes(-1, -2) + q)


def test_reduced_system(
    inputs: tuple[NDArray, NDArray], lyapunov_solver: LyapunovSolver
):
    """Tests that the batched Lyapunov solver returns the correct result."""
    a, q = inputs

    # Return if lyapunov_solver does vectrization
    if isinstance(lyapunov_solver, Vectorize):
        return

    if hasattr(lyapunov_solver, "reduce_sparsity"):
        lyapunov_solver.reduce_sparsity = True

    row_start = 1 + np.random.randint(0, a.shape[-1] - 1)
    row_end = max(
        row_start + 1 + np.random.randint(0, max(a.shape[-1] - row_start - 1, 1)),
        a.shape[-1] - 1,
    )

    col_start = 1 + np.random.randint(0, a.shape[-1] - 1)
    col_end = max(
        col_start + 1 + np.random.randint(0, max(a.shape[-1] - col_start - 1, 1)),
        a.shape[-1] - 1,
    )

    a_cpy = a.copy()
    a_cpy[:row_start, :] = 0
    a_cpy[row_end:, :] = 0

    x = lyapunov_solver(a_cpy, q, "contact")

    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q)

    a_cpy = a.copy()
    a_cpy[:, :col_start] = 0
    a_cpy[:, col_end:] = 0

    x = lyapunov_solver(a_cpy, q, "contact")

    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q)

    a_cpy = a.copy()
    a_cpy[:row_start, :] = 0
    a_cpy[row_end:, :] = 0
    a_cpy[:, :col_start] = 0
    a_cpy[:, col_end:] = 0

    x = lyapunov_solver(a_cpy, q, "contact")

    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q)

    if hasattr(lyapunov_solver, "reduce_sparsity"):
        lyapunov_solver.reduce_sparsity = False


def test_reduced_system_batch(
    inputs: tuple[NDArray, NDArray], lyapunov_solver: LyapunovSolver
):
    """Tests that the batched Lyapunov solver returns the correct result."""
    a, q = inputs

    # Return if lyapunov_solver does vectrization
    if isinstance(lyapunov_solver, Vectorize):
        return

    if hasattr(lyapunov_solver, "reduce_sparsity"):
        lyapunov_solver.reduce_sparsity = True

    row_start = 1 + np.random.randint(0, a.shape[-1] - 1)
    row_end = max(
        row_start + 1 + np.random.randint(0, max(a.shape[-1] - row_start - 1, 1)),
        a.shape[-1] - 1,
    )

    col_start = 1 + np.random.randint(0, a.shape[-1] - 1)
    col_end = max(
        col_start + 1 + np.random.randint(0, max(a.shape[-1] - col_start - 1, 1)),
        a.shape[-1] - 1,
    )

    a = xp.stack([a for __ in range(10)])
    q = xp.stack([q for __ in range(10)])

    a_cpy = a.copy()
    a_cpy[:, :row_start, :] = 0
    a_cpy[:, row_end:, :] = 0

    x = lyapunov_solver(a_cpy, q, "contact")

    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q)

    a_cpy = a.copy()
    a_cpy[:, :, :col_start] = 0
    a_cpy[:, :, col_end:] = 0

    x = lyapunov_solver(a_cpy, q, "contact")

    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q)

    a_cpy = a.copy()
    a_cpy[:, :row_start, :] = 0
    a_cpy[:, row_end:, :] = 0
    a_cpy[:, :, :col_start] = 0
    a_cpy[:, :, col_end:] = 0

    x = lyapunov_solver(a_cpy, q, "contact")

    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q)

    if hasattr(lyapunov_solver, "reduce_sparsity"):
        lyapunov_solver.reduce_sparsity = False


def test_memoizer_reduced_system(
    inputs: tuple[NDArray, NDArray], lyapunov_solver: LyapunovSolver
):
    """Tests that the Lyapunov memoizer works."""
    a, q = inputs

    # Return if lyapunov_solver does vectrization
    if isinstance(lyapunov_solver, Vectorize):
        return

    lyapunov_solver = LyapunovMemoizer(lyapunov_solver, reduce_sparsity=True)

    row_start = 1 + np.random.randint(0, a.shape[-1] - 1)
    row_end = max(
        row_start + 1 + np.random.randint(0, max(a.shape[-1] - row_start - 1, 1)),
        a.shape[-1] - 1,
    )

    col_start = 1 + np.random.randint(0, a.shape[-1] - 1)
    col_end = max(
        col_start + 1 + np.random.randint(0, max(a.shape[-1] - col_start - 1, 1)),
        a.shape[-1] - 1,
    )

    q_cpy = q + 1e-3 * xp.random.randn(*q.shape)

    a_cpy = a.copy()
    a_cpy[:row_start, :] = 0
    a_cpy[row_end:, :] = 0
    x = lyapunov_solver(a_cpy, q, "contact")
    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q)

    a_cpy += 1e-3 * xp.random.randn(*a.shape)
    a_cpy[:row_start, :] = 0
    a_cpy[row_end:, :] = 0
    x = lyapunov_solver(a_cpy, q_cpy, "contact")
    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q_cpy)

    lyapunov_solver._cache = {}

    a_cpy = a.copy()
    a_cpy[:, :col_start] = 0
    a_cpy[:, col_end:] = 0
    x = lyapunov_solver(a_cpy, q, "contact")
    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q)

    a_cpy += 1e-3 * xp.random.randn(*a.shape)
    a_cpy[:, :col_start] = 0
    a_cpy[:, col_end:] = 0
    x = lyapunov_solver(a_cpy, q_cpy, "contact")
    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q_cpy)

    lyapunov_solver._cache = {}

    a_cpy = a.copy()
    a_cpy[:row_start, :] = 0
    a_cpy[row_end:, :] = 0
    a_cpy[:, :col_start] = 0
    a_cpy[:, col_end:] = 0
    x = lyapunov_solver(a_cpy, q, "contact")
    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q)

    a_cpy += 1e-3 * xp.random.randn(*a.shape)
    a_cpy[:row_start, :] = 0
    a_cpy[row_end:, :] = 0
    a_cpy[:, :col_start] = 0
    a_cpy[:, col_end:] = 0
    x = lyapunov_solver(a_cpy, q_cpy, "contact")
    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q_cpy)


def test_memoizer_reduced_system_batched(
    inputs: tuple[NDArray, NDArray], lyapunov_solver: LyapunovSolver
):
    """Tests that the Lyapunov memoizer works."""
    a, q = inputs

    # Return if lyapunov_solver does vectrization
    if isinstance(lyapunov_solver, Vectorize):
        return

    lyapunov_solver = LyapunovMemoizer(lyapunov_solver, reduce_sparsity=True)

    row_start = 1 + np.random.randint(0, a.shape[-1] - 1)
    row_end = max(
        row_start + 1 + np.random.randint(0, max(a.shape[-1] - row_start - 1, 1)),
        a.shape[-1] - 1,
    )

    col_start = 1 + np.random.randint(0, a.shape[-1] - 1)
    col_end = max(
        col_start + 1 + np.random.randint(0, max(a.shape[-1] - col_start - 1, 1)),
        a.shape[-1] - 1,
    )

    a = xp.stack([a for __ in range(10)])
    q = xp.stack([q for __ in range(10)])

    q_cpy = q + 1e-3 * xp.random.randn(*q.shape)

    a_cpy = a.copy()
    a_cpy[:, :row_start, :] = 0
    a_cpy[:, row_end:, :] = 0
    x = lyapunov_solver(a_cpy, q, "contact")
    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q)

    a_cpy += 1e-3 * xp.random.randn(*a.shape)
    a_cpy[:, :row_start, :] = 0
    a_cpy[:, row_end:, :] = 0
    x = lyapunov_solver(a_cpy, q_cpy, "contact")
    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q_cpy)

    lyapunov_solver._cache = {}

    a_cpy = a.copy()
    a_cpy[:, :, :col_start] = 0
    a_cpy[:, :, col_end:] = 0
    x = lyapunov_solver(a_cpy, q, "contact")
    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q)

    a_cpy += 1e-3 * xp.random.randn(*a.shape)
    a_cpy[:, :, :col_start] = 0
    a_cpy[:, :, col_end:] = 0
    x = lyapunov_solver(a_cpy, q_cpy, "contact")
    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q_cpy)

    lyapunov_solver._cache = {}

    a_cpy = a.copy()
    a_cpy[:, :row_start, :] = 0
    a_cpy[:, row_end:, :] = 0
    a_cpy[:, :, :col_start] = 0
    a_cpy[:, :, col_end:] = 0
    x = lyapunov_solver(a_cpy, q, "contact")
    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q)

    a_cpy += 1e-3 * xp.random.randn(*a.shape)
    a_cpy[:, :row_start, :] = 0
    a_cpy[:, row_end:, :] = 0
    a_cpy[:, :, :col_start] = 0
    a_cpy[:, :, col_end:] = 0
    x = lyapunov_solver(a_cpy, q_cpy, "contact")
    assert xp.allclose(x, a_cpy @ x @ a_cpy.conj().swapaxes(-1, -2) + q_cpy)
