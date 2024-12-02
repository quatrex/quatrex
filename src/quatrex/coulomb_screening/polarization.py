# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, xp
from qttools.datastructures import DSBSparse

from quatrex.core.sse import ScatteringSelfEnergy


def fft_correlate(a: NDArray, b: NDArray) -> NDArray:
    """Computes the correlation of two arrays using FFT.

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
    n = a.shape[0] + b.shape[0] - 1
    a_fft = xp.fft.fftn(a, (n,), axes=(0,))
    b_fft = xp.fft.fftn(b[::-1], (n,), axes=(0,))
    return xp.fft.ifftn(a_fft * b_fft, axes=(0,))


class PCoulombScreening(ScatteringSelfEnergy):
    """Computes the dynamic polarization from the electronic system.

    Parameters
    ----------
    coulomb_screening_energies : NDArray
        The energies for the Coulomb screening

    """

    def __init__(self, coulomb_screening_energies: NDArray) -> None:
        """Initializes the polarization."""
        self.energies = coulomb_screening_energies
        self.ne = len(self.energies)
        self.prefactor = -1j / xp.pi * (self.energies[1] - self.energies[0])

    def compute(
        self, g_lesser: DSBSparse, g_greater: DSBSparse, out: tuple[DSBSparse, ...]
    ) -> None:
        """Computes the polarization.

        Parameters
        ----------
        g_lesser : DSBSparse
            The lesser Green's function.
        g_greater : DSBSparse
            The greater Green's function.
        out : tuple[DSBSparse, ...]
            The output matrices for the polarization. The order is
            p_lesser, p_greater, p_retarded.

        """
        p_lesser, p_greater, p_retarded = out
        # Transpose the matrices to nnz distribution.
        for m in (g_lesser, g_greater, p_lesser, p_greater, p_retarded):
            m.dtranspose() if m.distribution_state != "nnz" else None

        p_g_full = self.prefactor * fft_correlate(g_greater.data, -g_lesser.data.conj())
        p_l_full = -p_g_full[::-1].conj()
        # Fill the matrices with the data. Take second part of the energy convolution.
        p_lesser._data[
            p_lesser._stack_padding_mask, ..., : p_lesser.nnz_section_sizes[comm.rank]
        ] = p_l_full[self.ne - 1 :]
        p_greater._data[
            p_greater._stack_padding_mask, ..., : p_greater.nnz_section_sizes[comm.rank]
        ] = p_g_full[self.ne - 1 :]
        p_retarded._data = (p_greater._data - p_lesser._data) / 2

        # Transpose the matrices to stack distribution.
        for m in (g_lesser, g_greater, p_lesser, p_greater, p_retarded):
            m.dtranspose() if m.distribution_state != "stack" else None
