# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from qttools import NDArray, xp
from qttools.datastructures import DSDBSparse
from qttools.kernels.mixed_precision import compress, decompress
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

        self.config = config

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
            # <=> xp.roll(self.electron_energies, -upshift)[:-upshift]
            self.upshift = xp.argmin(
                xp.abs(electron_energies - (electron_energies[0] + self.phonon_energy))
            )
            # energy - hbar * omega
            # <=> xp.roll(self.electron_energies, downshift)[downshift:]
            self.downshift = (
                electron_energies.size
                - xp.argmin(
                    xp.abs(
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
        sigma_lesser, sigma_greater, sigma_retarded = out
        # Transpose the matrices to nnz distribution.
        for m in (g_lesser, g_greater, sigma_lesser, sigma_greater, sigma_retarded):
            # These should ideally already be in nnz-distribution.
            m.dtranspose() if m.distribution_state != "nnz" else None

        # ==== Using diagonal() ========================================
        sl_diag = sigma_lesser.diagonal(stack_index=self.valid_slice)

        sl_diag += self.deformation_potential**2 * (
            self.occupancy
            * xp.roll(g_lesser.diagonal(), self.downshift, axis=0)[self.downslice]
            + (self.occupancy + 1)
            * xp.roll(g_lesser.diagonal(), -self.upshift, axis=0)[self.upslice]
        )

        if sigma_lesser.bits is not None:
            _data = decompress(sigma_lesser.data, sigma_lesser.bits)
        else:
            _data = sigma_lesser.data

        sigma_lesser.fill_diagonal(sl_diag, stack_index=self.valid_slice, data=_data)

        if sigma_lesser.bits is not None:
            sigma_lesser.data = compress(_data, sigma_lesser.bits)

        sg_diag = sigma_greater.diagonal(stack_index=self.valid_slice)

        sg_diag += self.deformation_potential**2 * (
            self.occupancy
            * xp.roll(g_greater.diagonal(), -self.upshift, axis=0)[self.upslice]
            + (self.occupancy + 1)
            * xp.roll(g_greater.diagonal(), self.downshift, axis=0)[self.downslice]
        )

        if sigma_greater.bits is not None:
            _data = decompress(sigma_greater.data, sigma_greater.bits)
        else:
            _data = sigma_greater.data

        sigma_greater.fill_diagonal(sg_diag, stack_index=self.valid_slice, data=_data)

        if sigma_greater.bits is not None:
            sigma_greater.data = compress(_data, sigma_greater.bits)
