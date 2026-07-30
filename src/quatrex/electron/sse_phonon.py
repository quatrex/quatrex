# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the scattering self-energy from the electron-phonon interaction."""

import time
import h5py  # TODO: Import data elsewhere and remove this import

from qttools import NDArray, xp
from qttools.datastructures import DSDBSparse
from quatrex.core import constants
from quatrex.core.config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.core.statistics import bose_einstein


def _get_equal_spacing(a: NDArray):
    assert len(a.shape) == 1

    differences = xp.diff(a)
    spacing = differences[0]
    assert xp.allclose(differences, spacing), "`a` is not equispaced"

    return spacing


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
            self._compute_fn = self._compute_pseudo_scattering
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
            self._compute_fn = self._compute_long_wavelength
            if electron_energies is None:
                raise ValueError(
                    "Electron energies must be provided for the long-wavelength model."
                )

            # Load phonon modes
            # TODO: Do this elsewhere
            with h5py.File(config.input_dir / "phonon_data.h5", "r") as f:
                # Dimensions:
                # phonon_momenta[qx]
                # phonon_energies[mode, qx]
                # acoustic_mode_indices[polarized along x/y/z]
                # acoustic_epsilon[x/y/z, mode polarized along x/y/z]
                self.phonon_momenta = f["momentum-x"][:]
                phonon_energies_in = constants.hbar * f["omega"][:]
                acoustic_mode_indices = f["acoustic-mode-indices"][:]
                acoustic_epsilon = f["acoustic-epsilon"][:]

            # We ignore the transverse acoustic modes since the corresponding
            # long-wavelength coupling vanishes. This assumes small complex parts
            # of the corresponding epsilon.
            # The longitudinal mode is chosen as the first one.
            longitudinal_mode_index = acoustic_mode_indices[0]
            longitudinal_phonon_energies = phonon_energies_in[
                [longitudinal_mode_index], :
            ]
            self.longitudinal_epsilon = acoustic_epsilon[:, 0]
            self.phonon_energies = xp.delete(
                phonon_energies_in, acoustic_mode_indices, axis=0
            )
            self.phonon_energies = xp.concatenate(
                (longitudinal_phonon_energies, self.phonon_energies), axis=0
            )

            self.n_phonon_momenta = len(self.phonon_momenta)
            self.n_modes = self.phonon_energies.shape[0]
            # There are 3 * "number of atoms in unit cell" modes
            self.n_atoms_unit_cell = phonon_energies_in.shape[0] // 3

            # Compute coupling constants
            self.coupling_constants = xp.zeros((self.n_modes, self.n_phonon_momenta))
            for mode_index in range(self.n_modes):
                # [prefactor] = Å
                prefactor = xp.sqrt(
                    constants.hbar**2
                    / (
                        2
                        * config.phonon.atom_mass
                        * self.phonon_energies[mode_index, :]
                    )
                )
                if mode_index == 0:
                    # Acoustic longitudinal phonons
                    self.coupling_constants[mode_index, :] = (
                        1j
                        * config.phonon.acoustic_deformation_potential
                        * prefactor
                        * self.phonon_momenta
                        * self.longitudinal_epsilon[0]
                    )
                else:
                    # Optical phonons
                    self.coupling_constants[mode_index, :] = (
                        config.phonon.optical_deformation_potential * prefactor
                    )

            energy_spacing = _get_equal_spacing(electron_energies)
            # phonon_energy_shifts[momentum_index, mode_index] * energy_spacing
            # is the phonon energy rounded to the electron energy grid
            self.phonon_energy_shifts = xp.astype(
                xp.rint(self.phonon_energies / energy_spacing), int
            )
            assert xp.all(self.phonon_energy_shifts >= 0)
            self.occupancies = bose_einstein(
                self.phonon_energies, config.phonon.temperature
            )
            breakpoint()
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
        return self._compute_fn(g_lesser, g_greater, out)

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
        start_time = time.perf_counter()
        for mode_index in range(self.n_modes):
            print("Mode", mode_index, ", Time", time.perf_counter() - start_time, "s")
            for momentum_index in range(self.n_phonon_momenta):
                shift = self.phonon_energy_shifts[mode_index, momentum_index]
                occupancy = self.occupancies[mode_index, momentum_index]
                coupling_constant = self.coupling_constants[mode_index, momentum_index]
                prefactor = xp.abs(coupling_constant) ** 2 / (
                    self.n_phonon_momenta * self.n_atoms_unit_cell
                )

                sigma_lesser.data[: ne - shift, :] += (
                    prefactor * (occupancy + 1) * g_lesser.data[shift:, :]
                )
                sigma_lesser.data[shift:, :] += (
                    prefactor * occupancy * g_lesser.data[: ne - shift, :]
                )
                sigma_greater.data[shift:, :] += (
                    prefactor * (occupancy + 1) * g_greater.data[: ne - shift, :]
                )
                sigma_greater.data[: ne - shift, :] += (
                    prefactor * occupancy * g_greater.data[shift:, :]
                )
