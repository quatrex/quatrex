# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the scattering self-energy from the bare Fock interaction."""

import numpy as np

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.fft import fft_circular_convolve
from qttools.profiling import Profiler
from quatrex.core.config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.device.scba import SCBADevice
from quatrex.grid import get_equal_spacing

profiler = Profiler()


class SigmaFock(ScatteringSelfEnergy):
    """Computes the bare Fock self-energy.

    Parameters
    ----------
    config : QuatrexConfig
        The Quatrex configuration.
    device: SCBADevice
    electron_energies : NDArray
        The energies for the electron system.

    """

    def __init__(
        self,
        config: QuatrexConfig,
        device: SCBADevice,
        electron_energies: NDArray,
    ):
        """Initializes the bare Fock self-energy."""
        self.device = device
        self.energies = electron_energies
        self.qpoint_volume = np.prod(self.device.qgrid.size)
        self.prefactor = 1j / (2 * xp.pi) * get_equal_spacing(self.energies)
        (
            device.coulomb_matrix.dtranspose()
            if device.coulomb_matrix.distribution_state != "nnz"
            else None
        )
        self.coulomb_matrix_data = device.coulomb_matrix.data[0]

    @profiler.profile(label="SigmaFock", level="default", comm=comm)
    def compute(self, g_lesser: DSDBSparse, out: tuple[DSDBSparse, ...]) -> None:
        """Computes the Fock self-energy.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The lesser Green's function.
        out : tuple[DSDBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_retarded_hermitian.

        """
        # TODO: Check again if we really need to transpose the matrices
        # here.
        with profiler.profile_range(
            label="SigmaFock: stack->nnz transpose", level="default", comm=comm
        ):
            (sigma_retarded_hermitian,) = out
            for m in (g_lesser, sigma_retarded_hermitian):
                # These should both already be in nnz-distribution.
                m.dtranspose() if m.distribution_state != "nnz" else None

        # Compute the electron density by summing over energies.
        with profiler.profile_range(
            label="SigmaFock: SSE computation", level="default", comm=comm
        ):
            if g_lesser.data.shape[-1] == 0:
                return

            gl_density = xp.reshape(
                self.device.kgrid.expand(
                    self.prefactor * g_lesser.data.sum(axis=0), axis=0
                ),
                (*self.device.kgrid.size, -1),
            )
            out = fft_circular_convolve(
                gl_density,
                xp.reshape(
                    self.device.qgrid.expand(self.coulomb_matrix_data, axis=0),
                    (*self.device.qgrid.size, -1),
                ),
                axes=tuple(range(gl_density.ndim - 1)),
            )

            out = self.device.kgrid.reduce(out.reshape(-1, out.shape[-1]))

            sigma_retarded_hermitian.data += out / self.qpoint_volume

        # NOTE: The electron Green's functions and self-energies must
        # not be transposed back to stack distribution, as they are
        # needed in nnz distribution for the other interactions.
