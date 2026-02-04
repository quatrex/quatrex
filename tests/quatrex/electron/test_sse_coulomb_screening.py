# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, xp
from qttools.fft.convolve import _naive_convolve
from quatrex.electron.sse_coulomb_screening import hilbert_transform


def naive_hilbert_transform(a_full: NDArray, energies: NDArray) -> NDArray:
    """Naive implementation of Hilbert transform for testing purposes."""
    de = energies[1] - energies[0]
    ne = energies.size
    energy_differences = energies - energies[0] + de / 2
    hilbert_kernel = 1 / xp.concatenate([-energy_differences[::-1], energy_differences])
    result = _naive_convolve(a_full, hilbert_kernel)
    return result[2 * ne : 3 * ne]


@pytest.mark.usefixtures("array_shape")
def test_hilbert_transform(array_shape):
    ne = array_shape[0]
    # Also add orbital dimension at the end
    full_array_shape = (3 * ne,) + array_shape[1:] + (1,)
    a_full = xp.random.random(full_array_shape)
    energies = xp.linspace(-10, 10, ne)
    result = hilbert_transform(a_full[:-ne], a_full[ne:], energies)
    x = xp.zeros_like(a_full)
    x[ne:] = a_full[ne:]
    x[:-ne] -= a_full[:-ne]
    expected = naive_hilbert_transform(x, energies)
    assert xp.allclose(result, expected)
