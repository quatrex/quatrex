# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from qttools.datastructures import DSBSparse
from qttools.utils.gpu_utils import xp

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
            self.totalshift = self.upshift + self.downshift

            return

        raise ValueError(f"Unknown phonon model: {config.phonon.model}")

    def compute(
        self, g_lesser: DSBSparse, g_greater: DSBSparse, out: tuple[DSBSparse, ...]
    ) -> None:
        """Computes the electron-phonon self-energy."""
        return self._compute_pseudo_scattering(g_lesser, g_greater, out)

    def _initialize_diag_inds(self, sigma_lesser: DSBSparse) -> None:
        """Computes the diagonal indices for vectorized assignment."""
        stack_padding_mask = sigma_lesser._stack_padding_mask
        stack_padding_inds = stack_padding_mask.nonzero()[0][
            self.downshift : -self.upshift
        ]
        inds = np.zeros(sigma_lesser.shape[-1], dtype=int)
        ranks = np.zeros(sigma_lesser.shape[-1], dtype=int)

        for n in range(sigma_lesser.shape[-1]):
            inds[n] = xp.where((sigma_lesser.rows == n) & (sigma_lesser.cols == n))[0][
                0
            ]
            ranks[n] = xp.where(sigma_lesser.nnz_section_offsets <= inds[n])[0][-1]

        self._local_inds = (
            inds[ranks == comm.rank] - sigma_lesser.nnz_section_offsets[comm.rank]
        )
        self._sigma_inds = xp.ix_(stack_padding_inds, self._local_inds)

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
        if not hasattr(self, "_local_inds") or not hasattr(self, "_sigma_inds"):
            self._initialize_diag_inds(sigma_lesser)

        sigma_lesser._data[self._sigma_inds] = self.deformation_potential**2 * (
            self.occupancy
            * np.roll(g_lesser.data[..., self._local_inds], self.downshift, axis=0)[
                self.totalshift :
            ]
            + (self.occupancy + 1)
            * np.roll(g_lesser.data[..., self._local_inds], -self.upshift, axis=0)[
                : -self.totalshift
            ]
        )
        sigma_greater._data[self._sigma_inds] = self.deformation_potential**2 * (
            self.occupancy
            * np.roll(g_greater.data[..., self._local_inds], -self.upshift, axis=0)[
                : -self.totalshift
            ]
            + (self.occupancy + 1)
            * np.roll(g_greater.data[..., self._local_inds], self.downshift, axis=0)[
                self.totalshift :
            ]
        )

        # Keep only the imaginary part.
        sigma_lesser._data.real = 0.0
        sigma_greater._data.real = 0.0

        sigma_retarded._data[
            sigma_retarded._stack_padding_mask,
            ...,
            : sigma_retarded.nnz_section_sizes[comm.rank],
        ] = 0.5 * (sigma_greater.data - sigma_lesser.data)

        # ==== Full matrices ===========================================
        # nnz_stop = sigma_lesser.nnz_section_sizes[comm.rank]

        # sigma_lesser._data[stack_padding_inds, ..., :nnz_stop] = (
        #     self.deformation_potential**2
        #     * (
        #         self.occupancy
        #         * np.roll(g_lesser.data, self.downshift, axis=0)[self.totalshift :]
        #         + (self.occupancy + 1)
        #         * np.roll(g_lesser.data, -self.upshift, axis=0)[: -self.totalshift]
        #     ).imag
        #     * 1j
        # )
        # sigma_greater._data[stack_padding_inds, ..., :nnz_stop] = (
        #     self.deformation_potential**2
        #     * (
        #         self.occupancy
        #         * np.roll(g_greater.data, -self.upshift, axis=0)[: -self.totalshift]
        #         + (self.occupancy + 1)
        #         * np.roll(g_greater.data, self.downshift, axis=0)[self.totalshift :]
        #     ).imag
        #     * 1j
        # )

        # sigma_retarded._data[stack_padding_mask, ..., :nnz_stop] = 0.5 * (
        #     sigma_greater.data - sigma_lesser.data
        # )

        # Transpose the matrices back to the original stack distribution.
        for m in (g_lesser, g_greater, sigma_lesser, sigma_greater, sigma_retarded):
            m.dtranspose() if m.distribution_state != "stack" else None
