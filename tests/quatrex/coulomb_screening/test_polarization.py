# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.


from types import SimpleNamespace

import pytest

from qttools import NDArray, xp
from qttools.fft.convolve import _naive_convolve
from quatrex.coulomb_screening.polarization import (
    PCoulombScreening,
    hilbert_transform,
)
from quatrex.grid import KGrid


def naive_hilbert_transform(a: NDArray, energies: NDArray) -> NDArray:
    """Naive implementation of Hilbert transform for polarization for testing purposes.
    `a` should have the symmetry $a(-E) = a^*(E)$.
    """
    ne = energies.size
    # Should satisfy the symmetries of the polarization
    a_full = xp.concatenate([xp.conj(a[-1:0:-1]), a])
    energy_differences = energies - energies[0]
    # Remove the singularity by setting the energy difference to inf at the singularity.
    energy_differences[0] = xp.inf
    hilbert_kernel = 1 / xp.concatenate(
        [-energy_differences[-1:0:-1], energy_differences]
    )
    result = _naive_convolve(a_full, hilbert_kernel)
    return result[2 * ne - 2 : 3 * ne - 2]


def test_hilbert_transform(array_shape):
    # Use the symmetry of the polarization $a(-\omega)=a^{*}(\omega)$
    ne = array_shape[0]
    # Add empty orbital dimension at the end
    a = xp.random.random(array_shape + (1,)) + 1j * xp.random.random(array_shape + (1,))
    # E = 0 should be 0 due to the symmetry, so set it explicitly to 0.
    a[0] = 0
    energies = xp.linspace(-10, 10, ne)
    result = hilbert_transform(a, energies)
    expected = naive_hilbert_transform(a, energies)
    assert xp.allclose(result, expected)


class _Matrix:
    """The parts of a DSDBSparse matrix that the polarization needs."""

    symmetry = "skew-hermitian"
    distribution_state = "nnz"

    def __init__(self, data: NDArray) -> None:
        """Initializes the matrix with the data of the nnz distribution."""
        self.data = data

    def dtranspose(self, discard: bool = False) -> None:
        """Only switches the state."""
        self.distribution_state = "stack" if self.distribution_state == "nnz" else "nnz"


def _naive_polarization(
    g_a: NDArray, g_b: NDArray, size: tuple[int, ...], q_ind: int, shift: int
) -> NDArray:
    r"""Sums \(G^a_{ij}(k+q, E+w) G^b_{ji}(k, E)\) over k and E."""
    ne = g_a.shape[0]
    q = xp.unravel_index(xp.asarray(q_ind), size)
    k = xp.unravel_index(xp.arange(g_a.shape[1]), size)
    k_plus_q = xp.ravel_multi_index(
        tuple((ki + qi) % n for ki, qi, n in zip(k, q, size)), size
    )
    result = xp.zeros(g_a.shape[-1], dtype=complex)
    for e in range(max(0, -shift), min(ne, ne - shift)):
        result += xp.sum(g_a[e + shift, k_plus_q] * -g_b[e].conj(), axis=0)
    return result


@pytest.mark.parametrize(
    "size", [(1, 1, 1), (1, 2, 2), (1, 3, 1), (1, 3, 3), (1, 4, 3), (1, 1, 5)]
)
@pytest.mark.parametrize("renormalization", [False, True])
def test_polarization(size: tuple[int, ...], renormalization: bool):
    """Compares the polarization to its definition."""
    ne, nnz = 8, 5
    energies = xp.linspace(0.0, 3.5, ne)
    kgrid, qgrid = KGrid(size), KGrid(size, gamma_centered=True)
    nk = kgrid.num_points

    # In equilibrium, P^> and P^< are the same at zero frequency.
    rng = xp.random.RandomState(0)
    shape = (ne, nk, nnz)
    spectral = rng.uniform(-1, 1, shape) + 1j * rng.uniform(-1, 1, shape)
    occupation = (1 / (1 + xp.exp(2 * (energies - 1.6))))[:, None, None]
    g_lesser = occupation * spectral
    g_greater = -(1 - occupation) * spectral

    config = SimpleNamespace(
        compute=SimpleNamespace(convolve=SimpleNamespace(num_batches=2)),
        coulomb_screening=SimpleNamespace(
            align_polarization_to_complex_axes=False,
            energy_renormalization_terms=(("polarization",) if renormalization else ()),
        ),
    )
    device = SimpleNamespace(kgrid=kgrid, qgrid=qgrid)
    p_lesser, p_greater, p_retarded_hermitian = (
        _Matrix(xp.zeros(shape, dtype=complex)) for __ in range(3)
    )
    polarization = PCoulombScreening(config, device, energies)
    polarization.compute(
        _Matrix(g_lesser),
        _Matrix(g_greater),
        out=(p_lesser, p_greater, p_retarded_hermitian),
    )

    # The positive and negative frequencies, at all the q-points.
    shifts = range(-ne + 1, ne)
    expected_greater, expected_lesser = (
        polarization.prefactor
        * xp.array(
            [
                [_naive_polarization(g_a, g_b, size, q, s) for q in range(nk)]
                for s in shifts
            ]
        )
        for g_a, g_b in ((g_greater, g_lesser), (g_lesser, g_greater))
    )
    assert xp.allclose(p_greater.data, expected_greater[ne - 1 :])
    assert xp.allclose(p_lesser.data, expected_lesser[ne - 1 :])

    if not renormalization:
        assert xp.all(p_retarded_hermitian.data == 0.0)
        return

    spectral = expected_greater - expected_lesser
    expected = xp.zeros(shape, dtype=complex)
    spacing = float(energies[1] - energies[0])
    for i in range(ne):
        for j, s in enumerate(shifts):
            # NOTE: The kernel is truncated to the energy window.
            if s != i and abs(i - s) < ne:
                expected[i] += spectral[j] / (spacing * (i - s))
    expected *= -polarization.prefactor / 2 * polarization.kpoint_volume
    assert xp.allclose(p_retarded_hermitian.data, expected)
