# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from qttools import NDArray, xp
from qttools.datastructures import DSDBSparse
from quatrex.core.config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.core.statistics import bose_einstein


class SigmaPhonon(ScatteringSelfEnergy):
    """Computes the electron-photon self-energy.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration object.
    electron_energies : NDArray, optional
        The electron energies.

    """

    def __init__(
        self,
        config: QuatrexConfig,
        electron_energies: NDArray | None = None,
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

            # energy +- hbar * omega
            self.shift = xp.argmin(
                xp.abs(electron_energies - (electron_energies[0] + self.phonon_energy))
            )
            return

        raise ValueError(f"Unknown phonon model: {config.phonon.model}")

    def compute(
        self, g_lesser: DSDBSparse, g_greater: DSDBSparse, out: tuple[DSDBSparse, ...]
    ) -> None:
        """Computes the electron-phonon self-energy.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The lesser Green's function.
        g_greater : DSDBSparse
            The greater Green's function.
        out : tuple[DSDBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_lesser, sigma_greater, sigma_retarded.

        """
        return self._compute_pseudo_scattering(g_lesser, g_greater, out)

    def _compute_pseudo_scattering(
        self, g_lesser: DSDBSparse, g_greater: DSDBSparse, out: tuple[DSDBSparse, ...]
    ) -> None:
        """Computes the pseudo-phonon self-energy due to a deformation potential.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The lesser Green's function.
        g_greater : DSDBSparse
            The greater Green's function.
        out : tuple[DSDBSparse, ...]
            The lesser, greater and retarded self-energies.

        """
        sigma_lesser, sigma_greater, __ = out
        # Transpose the matrices to nnz distribution.
        for m in (g_lesser, g_greater, sigma_lesser, sigma_greater):
            # These should ideally already be in nnz-distribution.
            m.dtranspose() if m.distribution_state != "nnz" else None

        ne = g_lesser.data.shape[0]

        sl_diag = sigma_lesser.diagonal()
        gl_diag = g_lesser.diagonal()

        sl_diag[: ne - self.shift] += self.deformation_potential**2 * (
            (self.occupancy + 1) * gl_diag[self.shift :]
        )
        sl_diag[self.shift :] += self.deformation_potential**2 * (
            self.occupancy * gl_diag[: ne - self.shift]
        )

        sigma_lesser.fill_diagonal(sl_diag)

        sg_diag = sigma_greater.diagonal()
        gg_diag = g_greater.diagonal()

        sg_diag[: ne - self.shift] += self.deformation_potential**2 * (
            self.occupancy * gg_diag[self.shift :]
        )
        sg_diag[self.shift :] += self.deformation_potential**2 * (
            (self.occupancy + 1) * gg_diag[: ne - self.shift]
        )

        sigma_greater.fill_diagonal(sg_diag)
