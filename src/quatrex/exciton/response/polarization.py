# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

from cupyx.profiler import time_range
from qttools.utils.gpu_utils import get_device, get_host, synchronize_current_stream, xp
from mpi4py.MPI import COMM_WORLD as comm

@time_range()
def kron_correlate(a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
    """Convolves two 1D arrays using FFT and performs kronecker."""
    n = a.shape[0] + b.shape[0] - 1
    a_fft = xp.fft.fftn(a, (n,), axes=(0,))
    b_fft = xp.fft.fftn(b[::-1], (n,), axes=(0,))

    with time_range("einsum", color_id=comm.rank):
        x_fft = xp.einsum("ei,ej->eij", a_fft, b_fft)

    return xp.fft.ifftn(x_fft, axes=(0,))


def correlate(a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
    """Computes the correlation of two 1D arrays.

    This is slightly different from the usual definition of correlation
    in signal processing, where the second array is conjugated.

    Here, we use the definition of correlation as the convolution of
    the first array with the reversed second array.


    Parameters
    ----------
    a : np.ndarray
        First array.
    b : np.ndarray
        Second array.

    Returns
    -------
    np.ndarray
        Correlation of `a` and `b` including the "full" correlation.

    """
    return fftconvolve(a, b[::-1])



def fftconvolve(a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
    """Convolves two 1D arrays using FFT.

    Parameters
    ----------
    a : np.ndarray
        First array.
    b : np.ndarray
        Second array.

    Returns
    -------
    np.ndarray
        Convolution of `a` and `b` including the "full" convolution.

    """
    n = len(a) + len(b) - 1
    a_fft = xp.fft.fft(a, n)
    b_fft = xp.fft.fft(b, n)
    return xp.fft.ifft(a_fft * b_fft)
