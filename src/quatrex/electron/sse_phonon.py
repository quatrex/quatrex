# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the scattering self-energy from the electron-phonon interaction."""

from qttools import NDArray, xp
from qttools.datastructures import DSDBSparse
from quatrex.core import constants
from quatrex.core.config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.core.statistics import bose_einstein


def _get_equal_spacing(a: NDArray):
    """Asserts that `a` is equispaced and returns the spacing."""
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
            with open(config.input_dir / "phonon_dispersion.npy", "rb") as f:
                """
                Specification on phonon_dispersion.npy:
                This file contains the angular velocities `omega[mode, momentum]`
                for the different phonon modes and momenta. The phonon momenta are
                equally spaced as
                `np.linspace(-pi/a, pi/a, n_phonon_momenta)`, with `a` the lattice
                constant.
                The longitudinal acoustic mode along x is the first one
                (`omega[0, :]`), followed by the two transverse acoustic modes.
                The remaining modes are in no particular order.
                """
                # phonon_energies[mode, qx]
                phonon_energies_in = constants.hbar * xp.load(f)

            # We ignore the transverse acoustic modes since the corresponding
            # long-wavelength coupling vanishes.
            phonon_energies = xp.delete(phonon_energies_in, [1, 2], axis=0)

            # Infer quantities from the loaded dispersion
            n_modes, n_phonon_momenta = phonon_energies.shape
            # There are 3 * "number of atoms in unit cell" modes
            assert (
                phonon_energies_in.shape[0] % 3 == 0
            ), "Not the correct amount of modes"
            n_atoms_unit_cell = phonon_energies_in.shape[0] // 3
            max_phonon_momentum = xp.pi / config.phonon.lattice_constant
            phonon_momenta = xp.linspace(
                -max_phonon_momentum, max_phonon_momentum, n_phonon_momenta
            )
            longitudinal_epsilon_x = 1 / xp.sqrt(n_atoms_unit_cell)

            # Compute electron-phonon coupling constants
            coupling_constants = xp.zeros((n_modes, n_phonon_momenta), dtype=complex)
            for mode_index in range(n_modes):
                # [prefactor] = Å
                prefactor = xp.sqrt(
                    constants.hbar**2
                    / (2 * config.phonon.atom_mass * phonon_energies[mode_index, :])
                )
                if mode_index == 0:
                    # Acoustic longitudinal phonons
                    coupling_constants[mode_index, :] = (
                        1j
                        * config.phonon.acoustic_deformation_potential
                        * prefactor
                        * phonon_momenta
                        * longitudinal_epsilon_x
                    )
                else:
                    # Optical phonons
                    coupling_constants[mode_index, :] = (
                        config.phonon.optical_deformation_potential * prefactor
                    )

            energy_spacing = _get_equal_spacing(electron_energies)
            # phonon_energy_shifts[momentum_index, mode_index] * energy_spacing
            # is the phonon energy rounded to the electron energy grid
            phonon_energy_shifts = xp.astype(
                xp.rint(phonon_energies / energy_spacing), int
            )
            assert xp.all(phonon_energy_shifts >= 0)
            occupancies = bose_einstein(phonon_energies, config.phonon.temperature)

            # Compute V
            prefactor = 1 / (n_phonon_momenta * n_atoms_unit_cell)
            coupling_factors = xp.abs(coupling_constants) ** 2
            self.V_em = prefactor * xp.bincount(
                phonon_energy_shifts.flatten(),
                weights=(coupling_factors * (occupancies + 1)).flatten(),
            )
            self.V_abs = prefactor * xp.bincount(
                phonon_energy_shifts.flatten(),
                weights=(coupling_factors * occupancies).flatten(),
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

        for shift in range(len(self.V_em)):
            sigma_lesser.data[: ne - shift, :] += (
                self.V_em[shift] * g_lesser.data[shift:, :]
            )
            sigma_lesser.data[shift:, :] += (
                self.V_abs[shift] * g_lesser.data[: ne - shift, :]
            )
            sigma_greater.data[shift:, :] += (
                self.V_em[shift] * g_greater.data[: ne - shift, :]
            )
            sigma_greater.data[: ne - shift, :] += (
                self.V_abs[shift] * g_greater.data[shift:, :]
            )
