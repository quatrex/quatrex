# Copyright (c) 2025 ETH Zurich and the authors of the quatrex package.

from qttools import NDArray, xp
from qttools.profiling import Profiler

if xp.__name__ == "cupy":
    cache = xp.fft.config.get_plan_cache()

profiler = Profiler()


@profiler.profile(level="api")
def fft_convolve(a: NDArray, b: NDArray) -> NDArray:
    """Computes the convolution of two arrays using FFT over the first axis (energy axis).

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
    ne = (
        a.shape[0] + b.shape[0]
    )  # Should not have -1 here (otherwise hilbert transform fails)
    a_fft = xp.fft.fft(a, ne, axis=0)
    b_fft = xp.fft.fft(b, ne, axis=0)
    return xp.fft.ifft(a_fft * b_fft, axis=0)


@profiler.profile(level="api")
def fft_circular_convolve(a: xp.ndarray, b: xp.ndarray, axes: tuple[int]) -> xp.ndarray:
    """Computes the circular convolution of two arrays using the FFT.

    Parameters
    ----------
    a : NDArray
        First array.
    b : NDArray
        Second array.
    axes : tuple[int]
        The axes over which to perform the convolution.

    Returns
    -------
    NDArray
        The circular convolution of the two arrays.
    """
    # Extract the shapes of the arrays along the axes as tuples.
    nka = tuple(a.shape[i] for i in axes)
    nkb = tuple(b.shape[i] for i in axes)
    a_fft = xp.fft.fftn(a, nka, axes=axes)
    b_fft = xp.fft.fftn(b, nkb, axes=axes)
    return xp.fft.ifftn(a_fft * b_fft, axes=axes)


@profiler.profile(level="api")
def fft_convolve_kpoints(a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
    """Computes the convolution of two arrays using the FFT.

    The first axis is assumed to be the energy axis, the other
    axes are k-points and the last axis is the orbital index.

    Over the k-point axes, a circular convolution is performed.

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
    ne = a.shape[0] + b.shape[0] - 1
    nka = a.shape[1:-1]
    nkb = b.shape[1:-1]
    a_fft = xp.fft.fftn(a, (ne,) + nka, axes=(0,) + tuple(range(1, len(nka) + 1)))
    b_fft = xp.fft.fftn(b, (ne,) + nkb, axes=(0,) + tuple(range(1, len(nkb) + 1)))
    return xp.fft.ifftn(a_fft * b_fft, axes=(0,) + tuple(range(1, len(nka) + 1)))


@profiler.profile(level="api")
def fft_correlate_kpoints(a: NDArray, b: NDArray) -> NDArray:
    """Computes the correlation of two arrays using FFT.

    The first axis is assumed to be the energy axis, the other
    axes are k-points and the last axis is the orbital index.

    Over the k-point axes, a circular correlation is performed.

    Parameters
    ----------
    a : NDArray
        First array.
    b : NDArray
        Second array.

    Returns
    -------
    NDArray
        The cross-correlation of the two arrays.

    """
    ne = a.shape[0] + b.shape[0] - 1
    nka = a.shape[1:-1]
    nkb = b.shape[1:-1]
    a_fft = xp.fft.fftn(a, (ne,) + nka, axes=(0,) + tuple(range(1, len(nka) + 1)))
    b_fft = xp.fft.fftn(
        xp.flip(b, axis=(0,) + tuple(range(1, len(nkb) + 1))),
        (ne,) + nkb,
        axes=(0,) + tuple(range(1, len(nkb) + 1)),
    )
    # I have to roll the result by one, to get the gamma point at index 0.
    # The ordering is then: [0, 1, 2, ..., -2, -1] in terms of shifts.
    return xp.roll(
        xp.fft.ifftn(a_fft * b_fft, axes=(0,) + tuple(range(1, len(nka) + 1))),
        shift=1,
        axis=tuple(range(1, len(nka) + 1)),
    )


@profiler.profile(level="api")
def hilbert_transform_polarization(a: NDArray, energies: NDArray) -> NDArray:
    """Computes the Hilbert transform of the array a, assuming the symmetries of the
    polarization, i.e \([P^{\lessgtr}_{ij}(\omega)]^{\dagger} = -P^{\gtrless}_{ij}(-\omega)\).
    This becomes \(a(-\omega)=a^{*}(\omega)\), where a is \(a=P^>-P^<\).

    Assumes that the first axis corresponds to the energy axis.

    Parameters
    ----------
    a : NDArray
        The array to transform.
    energies : NDArray
        The energy values corresponding to the first axis of a.
    eta : float, optional
        For the principle part. Small part to avoid singularity, by
        default 1e-8.

    Returns
    -------
    NDArray
         The Hilbert transform of a.

    """
    # eta for removing the singularity. See Cauchy principal value.
    de = energies[1] - energies[0]
    eta = de / 2
    energy_differences = (
        xp.expand_dims(energies - energies[0], tuple(range(1, a.ndim))) + eta
    )
    ne = energies.size

    hilbert_kernel = 1 / energy_differences
    b = fft_convolve(a, hilbert_kernel)[:ne]
    # Negative frequencies of a
    b += fft_convolve(a[::-1].conj(), hilbert_kernel)[-ne:]
    # Negative frequencies of the kernel
    hilbert_kernel = -hilbert_kernel[::-1]
    b += fft_convolve(a, hilbert_kernel)[-ne:]

    return b


@profiler.profile(level="api")
def hilbert_transform_selfenergy(
    sl: NDArray, sg: NDArray, energies: NDArray
) -> NDArray:
    """Computes the Hilbert transform.

    Assumes that the first axis corresponds to the energy axis.

    Parameters
    ----------
    sl : NDArray
        The lesser self-energy on the grid |-----|-----|xxxxx|.
    sg : NDArray
        The greater self-energy on the grid |xxxxx|-----|-----|.
    energies : NDArray
        The energy values corresponding to the first axis of a.

    Returns
    -------
    NDArray
         The Hilbert transform of a.

    """
    ne = energies.size
    nk = sg.shape[1:-1]
    # Add empty dimensions for each k-point.
    energy_differences = (energies - energies[0]).reshape(-1, *(len(nk) + 1) * (1,))
    # eta for removing the singularity. See Cauchy principal value.
    eta = (energies[1] - energies[0]) / 2
    hilbert_kernel = 1 / (energy_differences + eta)

    sr = fft_convolve(sg[:ne] - sl[-ne:], hilbert_kernel)[:ne]
    # Correct for left edge
    sr += fft_convolve(-sl[:ne], hilbert_kernel)[-ne:]
    # Next account for negative frequencies
    hilbert_kernel = -hilbert_kernel[::-1]
    sr += fft_convolve(sg[:ne] - sl[-ne:], hilbert_kernel)[-ne:]
    # Correct for right edge
    sr += fft_convolve(sg[-ne:], hilbert_kernel)[:ne]

    return sr
