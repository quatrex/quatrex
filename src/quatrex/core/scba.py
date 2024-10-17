# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import os
from dataclasses import dataclass

import numpy as np
from qttools.datastructures import DSBSparse

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.observables import density
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.coulomb_screening import CoulombScreeningSolver, PCoulombScreening
from quatrex.electron import (
    ElectronSolver,
    SigmaCoulombScreening,
    SigmaPhonon,
    SigmaPhoton,
)
from quatrex.phonon import PhononSolver, PiPhonon
from quatrex.photon import PhotonSolver, PiPhoton


def _get_allocator(dsbsparse_type: DSBSparse, system_matrix: DSBSparse) -> DSBSparse:
    """Returns an allocation factory for the given DSBSparse type.

    Parameters
    ----------
    dsbsparse_type : DSBSparse
        The DSBSparse type to allocate.
    system_matrix : DSBSparse
        The system matrix to allocate the DSBSparse type for. The
        sparsity pattern of the system matrix is used to allocate
        the DSBSparse matrix.

    """

    def _allocator() -> DSBSparse:
        return dsbsparse_type.zeros_like(system_matrix)

    return _allocator


class SCBAData:
    """Data container class for the SCBA.

    Parameters
    ----------
    scba : SCBA
        The SCBA instance.

    """

    def __init__(self, scba: "SCBA") -> None:
        """Initializes the SCBA data."""

        electron_allocator = _get_allocator(
            scba.compute_config.dbsparse_type,
            scba.electron_solver.system_matrix,
        )
        self.sigma_retarded_prev = electron_allocator()
        self.sigma_lesser_prev = electron_allocator()
        self.sigma_greater_prev = electron_allocator()
        self.sigma_retarded = electron_allocator()
        self.sigma_lesser = electron_allocator()
        self.sigma_greater = electron_allocator()
        self.g_retarded = electron_allocator()
        self.g_lesser = electron_allocator()
        self.g_greater = electron_allocator()

        if hasattr(scba, "coulomb_screening_solver"):
            coulomb_screening_allocator = _get_allocator(
                scba.compute_config.dbsparse_type,
                scba.coulomb_screening_solver.system_matrix,
            )
            self.p_retarded = coulomb_screening_allocator()
            self.p_lesser = coulomb_screening_allocator()
            self.p_greater = coulomb_screening_allocator()
            self.w_retarded = coulomb_screening_allocator()
            self.w_lesser = coulomb_screening_allocator()
            self.w_greater = coulomb_screening_allocator()

        if hasattr(scba, "photon_solver"):
            photon_allocator = _get_allocator(
                scba.compute_config.dbsparse_type,
                scba.photon_solver.system_matrix,
            )
            self.pi_photon_retarded = photon_allocator()
            self.pi_photon_lesser = photon_allocator()
            self.pi_photon_greater = photon_allocator()
            self.d_photon_retarded = photon_allocator()
            self.d_photon_lesser = photon_allocator()
            self.d_photon_greater = photon_allocator()

        if hasattr(scba, "phonon_solver"):
            phonon_allocator = _get_allocator(
                scba.compute_config.dbsparse_type,
                scba.phonon_solver.system_matrix,
            )
            self.pi_phonon_retarded = phonon_allocator()
            self.pi_phonon_lesser = phonon_allocator()
            self.pi_phonon_greater = phonon_allocator()
            self.d_phonon_retarded = phonon_allocator()
            self.d_phonon_lesser = phonon_allocator()
            self.d_phonon_greater = phonon_allocator()


@dataclass
class Observables:
    # --- Electrons ----------------------------------------------------
    electron_ldos: np.ndarray = None
    electron_density: np.ndarray = None
    hole_density: np.ndarray = None
    electron_current: np.ndarray = None

    electron_electron_scattering_rate: np.ndarray = None
    electron_photon_scattering_rate: np.ndarray = None
    electron_phonon_scattering_rate: np.ndarray = None

    sigma_retarded_density: np.ndarray = None
    sigma_lesser_density: np.ndarray = None
    sigma_greater_density: np.ndarray = None

    # --- Coulomb screening --------------------------------------------
    w_retarded_density: np.ndarray = None
    w_lesser_density: np.ndarray = None
    w_greater_density: np.ndarray = None

    p_retarded_density: np.ndarray = None
    p_lesser_density: np.ndarray = None
    p_greater_density: np.ndarray = None

    # --- Photons ------------------------------------------------------
    pi_photon_retarded_density: np.ndarray = None
    pi_photon_lesser_density: np.ndarray = None
    pi_photon_greater_density: np.ndarray = None

    d_photon_retarded_density: np.ndarray = None
    d_photon_lesser_density: np.ndarray = None
    d_photon_greater_density: np.ndarray = None

    photon_current_density: np.ndarray = None

    # --- Phonons ------------------------------------------------------
    pi_phonon_retarded_density: np.ndarray = None
    pi_phonon_lesser_density: np.ndarray = None
    pi_phonon_greater_density: np.ndarray = None
    d_phonon_retarded_density: np.ndarray = None
    d_phonon_lesser_density: np.ndarray = None
    d_phonon_greater_density: np.ndarray = None

    thermal_current: np.ndarray = None


class SCBA:
    """Self-consistent Born approximation (SCBA) solver.

    Parameters
    ----------
    quatrex_config : Path
        Quatrex configuration file.
    compute_config : Path, optional
        Compute configuration file.

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig = None,
    ) -> None:
        """Initializes an SCBA instance."""
        self.quatrex_config = quatrex_config

        if compute_config is None:
            compute_config = ComputeConfig()

        self.compute_config = compute_config

        # ----- Electrons ----------------------------------------------
        self.electron_energies = np.load(
            self.quatrex_config.input_dir / "electron_energies.npy"
        )
        self.electron_solver = ElectronSolver(
            self.quatrex_config,
            self.compute_config,
            self.electron_energies,
        )

        # ----- Coulomb screening --------------------------------------
        if self.quatrex_config.scba.coulomb_screening:
            energies_path = (
                self.quatrex_config.input_dir / "coulomb_screening_energies.npy"
            )
            if os.path.isfile(energies_path):
                self.coulomb_screening_energies = np.load(energies_path)
            else:
                self.coulomb_screening_energies = self.electron_energies

            self.p_coulomb_screening = PCoulombScreening(...)
            self.coulomb_screening_solver = CoulombScreeningSolver(
                self.quatrex_config,
                self.compute_config,
                self.coulomb_screening_energies,
                ...,
            )
            self.sigma_coulomb_screening = SigmaCoulombScreening(...)

        # ----- Photons ------------------------------------------------
        if self.quatrex_config.scba.photon:
            energies_path = self.quatrex_config.input_dir / "photon_energies.npy"
            self.photon_energies = np.load(energies_path)
            self.pi_photon = PiPhoton(...)
            self.photon_solver = PhotonSolver(
                self.quatrex_config,
                self.compute_config,
                self.photon_energies,
                ...,
            )
            self.sigma_photon = SigmaPhoton(...)

        # ----- Phonons ------------------------------------------------
        if self.quatrex_config.scba.phonon:
            energies_path = self.quatrex_config.input_dir / "phonon_energies.npy"
            self.phonon_energies = np.load(energies_path)
            if self.quatrex_config.phonon.model == "negf":
                self.pi_phonon = PiPhonon(...)
                self.phonon_solver = PhononSolver(
                    self.quatrex_config,
                    self.compute_config,
                    self.phonon_energies,
                    ...,
                )
            self.sigma_phonon = SigmaPhonon(...)

        self.data = SCBAData(self)
        self.observables = Observables()

    def _swap_sigma(self) -> None:
        """Swaps the current and previous self-energy buffers."""
        self.data.sigma_retarded._data[:], self.data.sigma_retarded_prev._data[:] = (
            self.data.sigma_retarded_prev._data,
            self.data.sigma_retarded._data,
        )
        self.data.sigma_lesser._data[:], self.data.sigma_lesser_prev._data[:] = (
            self.data.sigma_lesser_prev._data,
            self.data.sigma_lesser._data,
        )
        self.data.sigma_greater._data[:], self.data.sigma_greater_prev._data[:] = (
            self.data.sigma_greater_prev._data,
            self.data.sigma_greater._data,
        )

    def _update_sigma(self) -> None:
        """Updates the self-energy with a mixing factor."""
        mixing_factor = self.quatrex_config.scba.mixing_factor
        self.data.sigma_retarded.data[:] = (
            (1 - mixing_factor) * self.data.sigma_retarded_prev.data
            + mixing_factor * self.data.sigma_retarded.data
        )
        self.data.sigma_retarded.data[:] = (
            (1 - mixing_factor) * self.data.sigma_lesser_prev.data
            + mixing_factor * self.data.sigma_lesser.data
        )
        self.data.sigma_greater.data[:] = (
            (1 - mixing_factor) * self.data.sigma_greater_prev.data
            + mixing_factor * self.data.sigma_greater.data
        )

    def _has_converged(self) -> bool:
        """Checks if the SCBA has converged."""
        return False

    def _compute_phonon_interaction(self):
        """Computes the phonon interaction."""
        if self.quatrex_config.phonon.model == "negf":
            raise NotImplementedError

        elif self.quatrex_config.phonon.model == "pseudo-scattering":
            self.sigma_phonon.compute(
                self.data.g_lesser,
                self.data.g_greater,
                out=(
                    self.data.sigma_lesser,
                    self.data.sigma_greater,
                    self.data.sigma_retarded,
                ),
            )

    def _compute_photon_interaction(self):
        """Computes the photon interaction."""
        raise NotImplementedError

    def _compute_coulomb_screening_interaction(self):
        """Computes the Coulomb screening interaction."""
        raise NotImplementedError

    def _compute_observables(self) -> None:
        """Computes observables."""
        self.observables.electron_ldos = -density(
            self.data.g_retarded,
            self.electron_solver.overlap_sparray,
        )
        self.observables.electron_density = density(
            self.data.g_lesser,
            self.electron_solver.overlap_sparray,
        )
        self.observables.hole_density = -density(
            self.data.g_greater,
            self.electron_solver.overlap_sparray,
        )

    def run(self) -> None:
        """Runs the SCBA to convergence."""
        print("Entering SCBA loop...", flush=True)
        for i in range(self.quatrex_config.scba.max_iterations):
            print(f"Iteration {i}", flush=True)
            self.electron_solver.solve(
                self.data.sigma_lesser,
                self.data.sigma_greater,
                self.data.sigma_retarded,
                out=(self.data.g_lesser, self.data.g_greater, self.data.g_retarded),
            )

            # Swap current with previous self-energy buffer.
            self._swap_sigma()

            # Reset current self-energy.
            self.data.sigma_retarded.data[:] = 0.0
            self.data.sigma_lesser.data[:] = 0.0
            self.data.sigma_greater.data[:] = 0.0

            if self.quatrex_config.scba.coulomb_screening:
                self._compute_coulomb_screening_interaction()

            if self.quatrex_config.scba.photon:
                self._compute_photon_interaction()

            if self.quatrex_config.scba.phonon:
                self._compute_phonon_interaction()

            if self._has_converged():
                print(f"SCBA converged after {i} iterations.")
                break

            # Update self-energy for next iteration with mixing factor.
            self._update_sigma()

        else:  # Did not break, i.e. max_iterations reached.
            print(f"SCBA did not converge after {i} iterations.")
            self._compute_observables()
