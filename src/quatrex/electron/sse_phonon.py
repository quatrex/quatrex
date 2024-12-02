# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from qttools import xp
from qttools.datastructures import DSBSparse

from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.core.statistics import bose_einstein


class SigmaPhonon(ScatteringSelfEnergy):
    """Computes the lesser electron-photon self-energy."""

    def __init__(
        self,
        config: QuatrexConfig,
        electron_energies: xp.ndarray | None = None,
    ) -> None:
        """Initializes the self-energy."""

        if config.phonon.model == "negf":
            raise NotImplementedError

        if config.phonon.model == "pseudo-scattering":
            if electron_energies is None:
                raise ValueError(
                    "Electron energies must be provided for deformation potential model."
                )
            self.phonon_energy = config.phonon.phonon_energy
            self.deformation_potential = config.phonon.deformation_potential
            self.occupancy = bose_einstein(
                self.phonon_energy, config.phonon.temperature
            )

            # energy + hbar * omega
            # <=> np.roll(self.electron_energies, -upshift)[:-upshift]
            self.upshift = np.argmin(
                np.abs(electron_energies - (electron_energies[0] + self.phonon_energy))
            )
            # energy - hbar * omega
            # <=> np.roll(self.electron_energies, downshift)[downshift:]
            self.downshift = (
                electron_energies.size
                - np.argmin(
                    np.abs(
                        electron_energies - (electron_energies[-1] - self.phonon_energy)
                    )
                )
                - 1
            )

            self.valid_slice = (
                slice(self.downshift, -self.upshift)
                if self.upshift != 0
                else slice(None)
            )

            totalshift = self.upshift + self.downshift

            self.upslice = slice(None) if totalshift == 0 else slice(-totalshift)
            self.downslice = slice(totalshift, None)

            return

        raise ValueError(f"Unknown phonon model: {config.phonon.model}")

    def compute(
        self, g_lesser: DSBSparse, g_greater: DSBSparse, out: tuple[DSBSparse, ...]
    ) -> None:
        """Computes the electron-phonon self-energy."""
        return self._compute_pseudo_scattering(g_lesser, g_greater, out)

    def _compute_pseudo_scattering(
        self, g_lesser: DSBSparse, g_greater: DSBSparse, out: tuple[DSBSparse, ...]
    ) -> None:
        """Computes the pseudo-phonon self-energy due to a deformation potential.

        Parameters
        ----------
        g_lesser : DSBSparse
            The lesser Green's function.
        g_greater : DSBSparse
            The greater Green's function.
        out : tuple[DSBSparse, ...]
            The lesser, greater and retarded self-energies.

        """
        sigma_lesser, sigma_greater, sigma_retarded = out
        # Transpose the matrices to nnz distribution.
        for m in (g_lesser, g_greater, sigma_lesser, sigma_greater, sigma_retarded):
            m.dtranspose() if m.distribution_state != "nnz" else None

        # ==== Diagonal only ===========================================
        inds = xp.diag_indices(sigma_lesser.shape[-1])

        sigma_lesser.stack[self.valid_slice][*inds] += self.deformation_potential**2 * (
            self.occupancy
            * xp.roll(g_lesser[*inds], self.downshift, axis=0)[self.downslice]
            + (self.occupancy + 1)
            * xp.roll(g_lesser[*inds], -self.upshift, axis=0)[self.upslice]
        )
        sigma_greater.stack[self.valid_slice][
            *inds
        ] += self.deformation_potential**2 * (
            self.occupancy
            * xp.roll(g_greater[*inds], -self.upshift, axis=0)[self.upslice]
            + (self.occupancy + 1)
            * xp.roll(g_greater[*inds], self.downshift, axis=0)[self.downslice]
        )

        # ==== Full matrices ===========================================
        # nnz_stop = sigma_lesser.nnz_section_sizes[comm.rank]
        # stack_padding_inds = sigma_lesser._stack_padding_mask.nonzero()[0][
        #     self.downshift : -self.upshift
        # ]
        # sigma_lesser._data[stack_padding_inds, ..., :nnz_stop] = (
        #     self.deformation_potential**2
        #     * (
        #         self.occupancy
        #         * np.roll(g_lesser.data, self.downshift, axis=0)[self.totalshift :]
        #         + (self.occupancy + 1)
        #         * np.roll(g_lesser.data, -self.upshift, axis=0)[: -self.totalshift]
        #     )
        # )
        # sigma_greater._data[stack_padding_inds, ..., :nnz_stop] = (
        #     self.deformation_potential**2
        #     * (
        #         self.occupancy
        #         * np.roll(g_greater.data, -self.upshift, axis=0)[: -self.totalshift]
        #         + (self.occupancy + 1)
        #         * np.roll(g_greater.data, self.downshift, axis=0)[self.totalshift :]
        #     )
        # )

        # Keep only the imaginary part.
        sigma_lesser._data.real = 0.0
        sigma_greater._data.real = 0.0

        sigma_retarded._data[
            sigma_retarded._stack_padding_mask,
            ...,
            : sigma_retarded.nnz_section_sizes[comm.rank],
        ] += 0.5 * (sigma_greater.data - sigma_lesser.data)

        # Transpose the matrices back to the original stack distribution.
        for m in (g_lesser, g_greater, sigma_lesser, sigma_greater, sigma_retarded):
            m.dtranspose() if m.distribution_state != "stack" else None
