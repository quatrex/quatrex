# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

from qttools import NDArray, xp
from qttools.datastructures import DSBSparse

from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.core.utils import homogenize


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
    quatrex_config : Path
        Quatrex configuration file.
    coulomb_screening_energies : NDArray
        The energies for the Coulomb screening

    """

    def __init__(
        self, quatrex_config: QuatrexConfig, coulomb_screening_energies: NDArray
    ) -> None:
        """Initializes the polarization."""
        self.energies = coulomb_screening_energies
        self.ne = len(self.energies)
        self.prefactor = -1j / xp.pi * xp.abs(self.energies[1] - self.energies[0])
        self.flatband = quatrex_config.electron.flatband

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
        for m in (g_lesser, g_greater, p_lesser, p_greater):
            m.dtranspose() if m.distribution_state != "nnz" else None

        p_g_full = self.prefactor * fft_correlate(g_greater.data, -g_lesser.data.conj())
        p_l_full = -p_g_full[::-1].conj()
        # Fill the matrices with the data. Take second part of the energy convolution.
        p_lesser.data = p_l_full[self.ne - 1 :]
        p_greater.data = p_g_full[self.ne - 1 :]

        # Transpose the matrices to stack distribution.
        for m in (g_lesser, g_greater, p_lesser, p_greater):
            m.dtranspose() if m.distribution_state != "stack" else None

        # Enforce anti-Hermitian symmetry and calculate Pr.
        p_lesser.data = (p_lesser.data - p_lesser.ltranspose(copy=True).data.conj()) / 2
        p_greater.data = (
            p_greater.data - p_greater.ltranspose(copy=True).data.conj()
        ) / 2

        p_lesser._data.real = 0
        p_greater._data.real = 0

        p_retarded.data = (p_greater.data - p_lesser.data) / 2

        # Homogenize in case of flatband.
        if self.flatband:
            homogenize(p_lesser)
            homogenize(p_greater)
            homogenize(p_retarded)


class PCoulombScreening_X(ScatteringSelfEnergy):
    """Computes the dynamic polarization from the electronic system.

    Parameters
    ----------
    quatrex_config : Path
        Quatrex configuration file.
    coulomb_screening_energies : NDArray
        The energies for the Coulomb screening

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        energies_lesser: NDArray,
        number_of_overlap_energies: int,
    ) -> None:
        """Initializes the polarization."""
        self.energies_lesser = energies_lesser
        self.noe = number_of_overlap_energies
        self.prefactor = (
            -1j / xp.pi * xp.abs(self.energies_lesser[1] - self.energies_lesser[0])
        )
        self.flatband = quatrex_config.electron.flatband

    def compute(self, g_x: DSBSparse, out: tuple[DSBSparse, ...]) -> None:
        """Computes the polarization.

        Parameters
        ----------
        g_x : DSBSparse
            Combined lesser and greater Green's function.
        out : tuple[DSBSparse, ...]
            The output matrices for the polarization. The order is
            p_lesser, p_retarded.

        """
        p_lesser, p_retarded = out
        nel = len(self.energies_lesser)
        # Transpose the matrices to nnz distribution.
        for m in (g_x, p_lesser):
            m.dtranspose() if m.distribution_state != "nnz" else None

        # TODO: Maybe make the p_lesser buffer 1 element smaller
        p_lesser._data[1:] = self.prefactor * fft_correlate(
            g_x.data[:nel], g_x.data[nel:].conj()
        )

        # Transpose the matrices to stack distribution.
        for m in (g_x, p_lesser):
            m.dtranspose() if m.distribution_state != "stack" else None

        # Enforce anti-Hermitian symmetry and calculate Pr.
        p_lesser.data = (p_lesser.data - p_lesser.ltranspose(copy=True).data.conj()) / 2
        p_lesser._data.real = 0

        p_retarded.data = -(p_lesser.data) / 2
        # TODO: How to do the below operation?
        # p_retarded.data[-self.noe:] -= p_lesser.data[:-self.noe-1:-1].conj()

        # Homogenize in case of flatband.
        if self.flatband:
            homogenize(p_lesser)
            homogenize(p_retarded)
