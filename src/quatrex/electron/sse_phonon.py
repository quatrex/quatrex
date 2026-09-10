# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the scattering self-energy from the electron-phonon interaction."""

from qttools import NDArray, sparse, xp
from qttools.datastructures import DSDBSparse
from qttools.utils.mpi_utils import distributed_load
from quatrex.core import constants
from quatrex.core.config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.core.statistics import bose_einstein
from quatrex.grid import get_equal_spacing


def _get_v_matrix(v: NDArray, g_size: int):
    r"""
    Given a one-dimensional array `v`, returns the matrix `m` such that
    (m @ g)[i] = \sum_s v[s] g[i + s] for any vector g of size `g_size`.
    The sum runs over all indices where both factors are defined.

    Note that
    (m.T @ g)[i] = \sum_s v[s] g[i - s].
    """

    if len(v.shape) != 1:
        raise ValueError("`v` has multiple dimensions.")

    v_size = xp.size(v)

    diagonals = [xp.full(g_size - i, v[i]) for i in range(v_size)]
    offsets = xp.arange(v_size)
    m = sparse.diags(diagonals, offsets, format="csc")

    return m


class SigmaPhononPseudoScattering(ScatteringSelfEnergy):
    """
    Computes the electron-phonon self-energy in the pseudo-scattering
    approximation.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration object.
    electron_energies : NDArray
        The electron energies.
    """

    def __init__(
        self,
        config: QuatrexConfig,
        electron_energies: NDArray,
    ) -> None:
        """Initializes the self-energy."""

        self.phonon_energy = config.phonon.phonon_energy
        self.deformation_potential = config.phonon.deformation_potential
        self.occupancy = bose_einstein(self.phonon_energy, config.phonon.temperature)

        # energy +- hbar * omega
        self.shift = xp.argmin(
            xp.abs(electron_energies - (electron_energies[0] + self.phonon_energy))
        )

    def compute(
        self, g_lesser: DSDBSparse, g_greater: DSDBSparse, out: tuple[DSDBSparse, ...]
    ) -> None:
        """
        Computes the electron-phonon self-energy in the pseudo-scattering
        approximation.

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


class SigmaPhononDeformationPotential(ScatteringSelfEnergy):
    """
    Computes the electron-phonon self-energy in the deformation potential
    approximation.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration object.
    electron_energies : NDArray
        The electron energies.
    """

    def __init__(
        self,
        config: QuatrexConfig,
        electron_energies: NDArray,
    ) -> None:
        """Initializes the self-energy."""

        # Load phonon modes
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
        phonon_energies_in = constants.hbar * distributed_load(
            config.input_dir / "phonon_dispersion.npy"
        )

        # We ignore the transverse acoustic modes since the corresponding
        # deformation potential coupling vanishes.
        phonon_energies = xp.delete(phonon_energies_in, [1, 2], axis=0)

        # Infer quantities from the loaded dispersion
        n_modes, n_phonon_momenta = phonon_energies.shape
        if phonon_energies_in.shape[0] % 3 != 0:
            raise ValueError(
                'Not the correct amount of modes: There are supposed to be 3 * "number of atoms in unit cell" modes.'
            )
        n_atoms_unit_cell = phonon_energies_in.shape[0] // 3
        max_phonon_momentum = xp.pi / config.phonon.lattice_constant
        phonon_momenta = xp.linspace(
            -max_phonon_momentum, max_phonon_momentum, n_phonon_momenta
        )

        # Compute electron-phonon coupling constants
        coupling_constants = xp.zeros((n_modes, n_phonon_momenta), dtype=complex)
        atom_mass = config.phonon.atom_mass_u * constants.atomic_mass_unit
        # prefactors[mode, qx]
        prefactors = xp.sqrt(constants.hbar**2 / (2 * atom_mass * phonon_energies))
        # Acoustic longitudinal phonons
        longitudinal_epsilon_x = 1 / xp.sqrt(n_atoms_unit_cell)
        coupling_constants[0, :] = (
            1j
            * config.phonon.acoustic_deformation_potential
            * prefactors[0, :]
            * phonon_momenta
            * longitudinal_epsilon_x
        )
        # Optical phonons
        coupling_constants[1:, :] = (
            config.phonon.optical_deformation_potential * prefactors[1:, :]
        )

        energy_spacing = get_equal_spacing(electron_energies)
        # phonon_energy_shifts[momentum_index, mode_index] * energy_spacing
        # is the phonon energy rounded to the electron energy grid
        phonon_energy_shifts = xp.astype(xp.rint(phonon_energies / energy_spacing), int)
        if not xp.all(phonon_energy_shifts >= 0):
            raise ValueError("Detected negative phonon energies.")
        occupancies = bose_einstein(phonon_energies, config.phonon.temperature)

        # Compute v
        prefactor = 1 / (n_phonon_momenta * n_atoms_unit_cell)
        coupling_factors = xp.abs(coupling_constants) ** 2
        v_em = prefactor * xp.bincount(
            phonon_energy_shifts.flatten(),
            weights=(coupling_factors * (occupancies + 1)).flatten(),
        )
        v_abs = prefactor * xp.bincount(
            phonon_energy_shifts.flatten(),
            weights=(coupling_factors * occupancies).flatten(),
        )

        # Compute corresponding banded matrices
        n_electron_energies = electron_energies.shape[0]
        m_em = _get_v_matrix(v_em, g_size=n_electron_energies)
        m_abs = _get_v_matrix(v_abs, g_size=n_electron_energies)
        # m_greater = m_lesser.T is not stored to save memory
        self.m_lesser = m_em + m_abs.T

    def compute(
        self, g_lesser: DSDBSparse, g_greater: DSDBSparse, out: tuple[DSDBSparse, ...]
    ) -> None:
        """
        Computes the electron-phonon self-energy in the deformation potential
        approximation.

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
            if m.distribution_state != "nnz":
                raise ValueError(
                    'The inputs and outputs of `_compute_deformation_potential` must be in the "nnz" distribution state.'
                )

        sigma_lesser.data = self.m_lesser @ g_lesser.data
        sigma_greater.data = self.m_lesser.T @ g_greater.data
