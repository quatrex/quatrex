# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.


from qttools import NDArray, xp
from qttools.fft.convolve import _naive_convolve
from quatrex.coulomb_screening.polarization import hilbert_transform


def naive_hilbert_transform(a: NDArray, energies: NDArray) -> NDArray:
    """Naive implementation of Hilbert transform for polarization for testing purposes.
    `a` should have the symmetry $a(-E) = a^*(E)$.
    """
    de = energies[1] - energies[0]
    ne = energies.size
    # Should satisfy the symmetries of the polarization
    a_full = xp.concatenate([xp.conj(a[::-1]), a])
    energy_differences = energies - energies[0] + de / 2
    hilbert_kernel = 1 / xp.concatenate([-energy_differences[::-1], energy_differences])
    result = _naive_convolve(a_full, hilbert_kernel)
    return result[2 * ne : 3 * ne]


def test_hilbert_transform(array_shape):
    # Use the symmetry of the polarization $a(-\omega)=a^{*}(\omega)$
    ne = array_shape[0]
    # Add empty orbital dimension at the end
    a = xp.random.random(array_shape + (1,)) + 1j * xp.random.random(array_shape + (1,))
    energies = xp.linspace(-10, 10, ne)
    result = hilbert_transform(a, energies)
    expected = naive_hilbert_transform(a, energies)
    assert xp.allclose(result, expected)
