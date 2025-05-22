# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

from cupyx.profiler import time_range
from qttools.utils.gpu_utils import get_device, get_host, synchronize_current_stream, xp
from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray

from qttools.datastructures import DSDBCOO

def four_point_correlation(GG: NDArray,
                            GL: NDArray,
                            G_rows: NDArray,
                            G_cols: NDArray,
                            L_rows: NDArray,
                            L_cols: NDArray,
                            L_nen: int,
                            L_step_E: int,
                            G_inverse_table: NDArray,
                            prefactor,
                            ):
    """Computes the four-point correlation function.
    This function computes the four-point correlation function
    using the element-wise product of the two-point correlation
    functions GG and GL. The correlation is computed using
    the FFT convolution method. The flipping of the
    second function in convolution is done in the Fourier space, by 
    taking its conjugate.
    Parameters
    ----------
    GG : NDArray
        Two-point Green's function, last dimension is energy, first dimension is space.
    GL : NDArray
        Two-point Green's function, last dimension is energy, first dimension is space.
    Returns
    -------
    NDArray
        Four-point correlation function, last dimension is energy, first dimension is space.
    """
    G_nen = GG.shape[-1]
    n = G_nen + G_nen - 1
    L_nnz = len(L_rows)
    G_nnz = len(G_rows)
    assert L_nnz == len(L_cols)
    assert G_nnz == len(G_cols)
    assert GG.shape[0] == GL.shape[0]

    LG = xp.zeros((L_nnz, L_nen), dtype=GG.dtype)
    LL = xp.zeros((L_nnz, L_nen), dtype=GG.dtype)
    
    GG_fft = xp.fft.fftn(GG, (n,), axes=(-1,))    

    for iinz in range(L_nnz):
        i = G_rows[L_rows[iinz]]
        j = G_cols[L_rows[iinz]]
        k = G_rows[L_cols[iinz]]
        L = G_cols[L_cols[iinz]]

        GL_fft = xp.fft.fftn(GL[G_inverse_table[L,j]], (n,), axes=(-1,))
        L_fft = prefactor * xp.multiply( GG_fft[G_inverse_table[i,k]] , GL_fft.conj() )
        L_t = xp.fft.ifftn(L_fft)
        LG[iinz] = L_t[G_nen - 1 : G_nen - 1 + L_nen * L_step_E : L_step_E]

        GL_fft = xp.fft.fftn(GL[G_inverse_table[i,k]], (n,), axes=(-1,))
        L_fft = prefactor * xp.multiply( GL_fft , GG_fft[G_inverse_table[L,j]].conj() )
        L_t = xp.fft.ifftn(L_fft)
        LL[iinz] = L_t[G_nen - 1 : G_nen - 1 + L_nen * L_step_E : L_step_E]

    return LG, LL

@time_range()
def kron_correlate(a: xp.ndarray, b: xp.ndarray, ) -> xp.ndarray:
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
