# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Helper functions having to do with grids."""

from qttools import NDArray, xp


def get_equal_spacing(a: NDArray):
    """
    Asserts that the one-dimensional array `a` is approximately equispaced and returns
    the spacing.
    """

    if len(a.shape) != 1:
        raise ValueError("`a` has multiple dimensions.")

    differences = xp.diff(a)
    spacing = differences[0]

    if not xp.allclose(differences, spacing):
        raise ValueError("`a` is not equispaced.")

    return spacing
