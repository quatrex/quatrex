# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
import numpy as np

from quatrex.core.utils import split_trapezoid

X = np.linspace(-1.0, 1.0, 21)


def test_sum_is_trapezoid():
    """Tests that both parts add up to the full trapezoidal integral."""
    rng = np.random.default_rng(0)
    y = rng.uniform(-1, 1, (4, 3, X.size))
    split = rng.uniform(-1.2, 1.2, (4, 3))
    below, above = split_trapezoid(y, X, split)
    assert np.allclose(below + above, np.trapezoid(y, X, axis=-1))


def test_exact_for_linear_integrand():
    """Tests exact integration up to a split point inside a grid cell."""
    y = np.stack([np.ones_like(X), 2 * X + 1])
    split = np.array([0.033, -0.517])
    below, above = split_trapezoid(y, X, split)

    def antiderivative(x):
        return np.array([x, x**2 + x])

    expected = np.diag(antiderivative(split)) - antiderivative(X[0])
    assert np.allclose(below, expected)
    assert np.allclose(above, antiderivative(X[-1]) - np.diag(antiderivative(split)))


def test_split_outside_grid():
    """Tests that split points outside the grid are clamped."""
    y = np.ones((2, X.size))
    below, above = split_trapezoid(y, X, np.array([-5.0, 5.0]))
    assert np.allclose(below, [0.0, 2.0])
    assert np.allclose(above, [2.0, 0.0])


def test_split_on_grid_point():
    """Tests that a split exactly on a grid point needs no interpolation."""
    y = np.exp(X)[np.newaxis]
    below, __ = split_trapezoid(y, X, np.array([X[7]]))
    assert np.allclose(below, np.trapezoid(y[:, :8], X[:8], axis=-1))
