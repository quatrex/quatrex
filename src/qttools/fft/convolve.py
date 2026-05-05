# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from qttools import NDArray, xp


def _naive_convolve(a: NDArray, b: NDArray) -> NDArray:
    """Naive implementation of convolution for testing purposes.

    Parameters
    ----------
    a : NDArray
        First array.
    b : NDArray
        Second array.

    Returns
    -------
    NDArray
        The convolution of the two arrays.

    """
    ne_a = a.shape[0]
    ne_b = b.shape[0]
    ne = ne_a + ne_b - 1
    result_shape = (ne,) + a.shape[1:]
    result = xp.zeros(result_shape, dtype=a.dtype)
    for i in range(ne_a):
        for j in range(ne_b):
            result[i + j] += a[i] * b[j]
    return result
