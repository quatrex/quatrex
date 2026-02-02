# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

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
def fft_circular_convolve(a: NDArray, b: NDArray, axes: tuple[int]) -> NDArray:
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
def fft_convolve_kpoints(a: NDArray, b: NDArray) -> NDArray:
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
