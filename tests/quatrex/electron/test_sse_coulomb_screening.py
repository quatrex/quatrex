# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.


from pathlib import Path

import pytest

from qttools import NDArray, xp
from qttools.fft.convolve import _naive_convolve
from quatrex.core.config import parse_config
from quatrex.electron.sse_coulomb_screening import (
    SigmaCoulombScreening,
    hilbert_transform,
)


def naive_hilbert_transform(a_full: NDArray, energies: NDArray) -> NDArray:
    """Naive implementation of Hilbert transform for testing purposes."""
    ne = energies.size
    energy_differences = energies - energies[0]
    # Remove the singularity by setting the energy difference to inf at the singularity.
    energy_differences[0] = xp.inf
    hilbert_kernel = 1 / xp.concatenate(
        [-energy_differences[-1:0:-1], energy_differences]
    )
    result = _naive_convolve(a_full, hilbert_kernel)
    return result[2 * ne - 1 : 3 * ne - 1]


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


def _reference_convolution(g_: NDArray, w_: NDArray) -> NDArray:
    """Naive convolution implementation for testing purposes."""
    ne = g_.shape[0]
    out_ref = xp.zeros_like(g_)
    for i in range(len(g_)):
        for j in range(len(g_)):
            out_ref[i] += g_[j] * w_[i - j + ne - 1]

    return out_ref


class MockDSDBSparse:
    def __init__(self, data: NDArray) -> None:
        self.data = data


@pytest.mark.parametrize("ne", [10, 100])
@pytest.mark.parametrize("no", [1, 2])
def test_lesser_greater_convolution(ne: int, no: int, gw_example: tuple[Path, bool]):
    """Test the convolution for the lesser and greater self-energies."""

    rng = xp.random.default_rng(seed=42)

    g_lesser = rng.random((ne, no)) + 1j * rng.random((ne, no))
    g_greater = rng.random((ne, no)) + 1j * rng.random((ne, no))
    w_lesser = rng.random((ne, no)) + 1j * rng.random((ne, no))
    w_greater = rng.random((ne, no)) + 1j * rng.random((ne, no))

    # Construct full energy grid
    # using the symmetry: W<>(-E) = -W><(E)*.
    # zero-energy origin to prevent double-counting during concatenation.
    w_lesser[0] = 0
    w_greater[0] = 0

    w_lesser_full = xp.concatenate((-w_greater.conj()[::-1][:-1], w_lesser))
    w_greater_full = xp.concatenate((-w_lesser.conj()[::-1][:-1], w_greater))

    sigma_lesser_ref = _reference_convolution(g_lesser, w_lesser_full)
    sigma_greater_ref = _reference_convolution(g_greater, w_greater_full)

    # arbitrary energies for testing purposes
    electron_energies = xp.linspace(-10, 2, ne)

    example_path, _ = gw_example
    quatrex_config_path = example_path / "quatrex_config.toml"
    config = parse_config(quatrex_config_path)
    sigma_coulomb_screening = SigmaCoulombScreening(config, electron_energies)

    # Process all data at once for testing
    batch = slice(None)

    out = (
        MockDSDBSparse(xp.zeros_like(g_lesser)),
        MockDSDBSparse(xp.zeros_like(g_lesser)),
        MockDSDBSparse(xp.zeros_like(g_lesser)),
    )
    sigma_coulomb_screening._compute_with_correction(
        MockDSDBSparse(g_lesser),
        MockDSDBSparse(g_greater),
        MockDSDBSparse(w_lesser),
        MockDSDBSparse(w_greater),
        out,
        batch,
    )

    assert xp.allclose(
        out[0].data, sigma_coulomb_screening.prefactor * sigma_lesser_ref
    )
    assert xp.allclose(
        out[1].data, sigma_coulomb_screening.prefactor * sigma_greater_ref
    )

    hilbert_kernel_fft = xp.zeros((2 * ne - 1, 1), dtype=g_lesser.dtype)

    out = (
        MockDSDBSparse(xp.zeros_like(g_lesser)),
        MockDSDBSparse(xp.zeros_like(g_lesser)),
        MockDSDBSparse(xp.zeros_like(g_lesser)),
    )
    # NOTE: the "correction" part only affects the retarded part
    sigma_coulomb_screening._compute_without_correction(
        MockDSDBSparse(g_lesser),
        MockDSDBSparse(g_greater),
        MockDSDBSparse(w_lesser),
        MockDSDBSparse(w_greater),
        out,
        batch,
        hilbert_kernel_fft,
    )

    assert xp.allclose(
        out[0].data, sigma_coulomb_screening.prefactor * sigma_lesser_ref
    )
    assert xp.allclose(
        out[1].data, sigma_coulomb_screening.prefactor * sigma_greater_ref
    )
