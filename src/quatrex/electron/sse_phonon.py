# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the scattering self-energy from the electron-phonon interaction."""

from qttools import NDArray, xp
from qttools.datastructures import DSDBSparse
from quatrex.core import constants
from quatrex.core.config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.core.statistics import bose_einstein


class SigmaPhonon(ScatteringSelfEnergy):
    """Computes the electron-phonon self-energy.

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
            # TODO: Get rid of the compute_fn attribute and use a more elegant solution.
            self.compute_fn = self._compute_pseudo_scattering
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

        if config.phonon.model == "long-wavelength":
            self.compute_fn = self._compute_long_wavelength
            if electron_energies is None:
                raise ValueError(
                    "Electron energies must be provided for the long-wavelength model."
                    # TODO: Really?
                )

            # Compute phonon modes and coupling constants
            # Generate a grid of phonon momenta.
            # TODO: 3d momenta
            self.phonon_momenta = xp.linspace(
                -config.phonon.q_grid_maximum,
                config.phonon.q_grid_maximum,
                config.phonon.q_grid_n_target,
            )
            # TODO: The following approach does not allow acoustic/optical to be None.
            self.n_phonon_momenta = len(self.phonon_momenta)
            self.n_acoustic_modes = len(config.phonon.acoustic_deformation_potentials)
            self.n_optical_modes = len(config.phonon.optical_deformation_potentials)
            self.n_modes = self.n_acoustic_modes + self.n_optical_modes
            self.phonon_energies = xp.zeros((self.n_phonon_momenta, self.n_modes))
            self.coupling_constants = xp.zeros((self.n_phonon_momenta, self.n_modes))
            for mode_index in range(self.n_modes):
                if mode_index < self.n_acoustic_modes:
                    # Acoustic phonons
                    list_index = mode_index
                    # TODO: Convert to expected units (e.g. h or hbar?)
                    self.phonon_energies[:, mode_index] = (
                        config.phonon.acoustic_speeds_of_sound[list_index]
                        * xp.abs(self.phonon_momenta)
                        * constants.hbar
                    )
                    # TODO: Check this equation for the coupling
                    # TODO: Add polarization $\epsilon$
                    # TODO: Relax or assert the constraint that the basis functions must
                    #       be orthonormal
                    self.coupling_constants[:, mode_index] = (
                        1j
                        * config.phonon.acoustic_deformation_potentials[list_index]
                        * xp.sqrt(
                            constants.hbar
                            / (
                                2
                                * self.n_phonon_momenta
                                * self.n_modes
                                * self.phonon_energies[:, mode_index]
                            )
                        )
                        * self.phonon_momenta
                    )
                else:
                    # Optical phonons
                    list_index = mode_index - self.n_acoustic_modes
                    self.phonon_energies[:, mode_index] = (
                        config.phonon.optical_phonon_energies[list_index]
                    )
                    self.coupling_constants[
                        :, mode_index
                    ] = config.phonon.optical_deformation_potentials[
                        list_index
                    ] * xp.sqrt(
                        constants.hbar
                        / (
                            2
                            * self.n_phonon_momenta
                            * self.n_modes
                            * self.phonon_energies[:, mode_index]
                        )
                    )

            # Assuming equispaced energies (TODO: assert that)
            energy_spacing = electron_energies[1] - electron_energies[0]
            # phonon_energy_shifts[momentum_index, mode_index] * energy_spacing
            # is the phonon energy rounded to the electron energy grid
            self.phonon_energy_shifts = xp.astype(
                xp.rint(self.phonon_energies / energy_spacing), int
            )
            assert xp.all(self.phonon_energy_shifts >= 0)
            self.occupancies = bose_einstein(
                self.phonon_energies, config.phonon.temperature
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
            sigma_lesser, sigma_greater, sigma_retarded_hermitian.

        """
        return self.compute_fn(g_lesser, g_greater, out)

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

    def _compute_long_wavelength(
        self, g_lesser: DSDBSparse, g_greater: DSDBSparse, out: tuple[DSDBSparse, ...]
    ) -> None:
        """Computes the long-wavelength phonon self-energy.

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
        for m in (g_lesser, g_greater, sigma_lesser, sigma_greater):
            assert m.distribution_state == "nnz"

        ne = g_lesser.data.shape[0]

        # OPTIMIZATION: Mitigate the python loops
        for momentum_index in range(self.n_phonon_momenta):
            for mode_index in range(self.n_modes):
                shift = self.phonon_energy_shifts[momentum_index, mode_index]
                occupancy = self.occupancies[momentum_index, mode_index]
                coupling_constant = self.coupling_constants[momentum_index, mode_index]
                coupling_factor = xp.abs(coupling_constant) ** 2

                sigma_lesser.data[: ne - shift, :] += (
                    coupling_factor * (occupancy + 1) * g_lesser.data[shift:, :]
                )
                sigma_lesser.data[shift:, :] += (
                    coupling_factor * occupancy * g_lesser.data[: ne - shift, :]
                )
                sigma_greater.data[shift:, :] += (
                    coupling_factor * (occupancy + 1) * g_greater.data[: ne - shift, :]
                )
                sigma_greater.data[: ne - shift, :] += (
                    coupling_factor * occupancy * g_greater.data[shift:, :]
                )
