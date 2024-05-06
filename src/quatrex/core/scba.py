# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import os
from dataclasses import dataclass

import numpy as np
from qttools.datastructures import DBSparse

from quatrex.core.compute_config import ComputeConfig
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
from quatrex.utils.mpi_utils import slice_local_array


@dataclass
class SCBAData:
    """
    Dataclass storing the data used in the Self-Consistent Born Approximation (SCBA). For
    all possible sub-systems.

    Attributes:
        # Electron
        sigma_retarded (DBSparse): Retarded self-energy.
        sigma_lesser (DBSparse): Lesser self-energy.
        sigma_greater (DBSparse): Greater self-energy.
        g_retarded (DBSparse): Retarded Green's function.
        g_lesser (DBSparse): Lesser Green's function.
        g_greater (DBSparse): Greater Green's function.

        # Coulomb screening
        p_retarded (DBSparse): Retarded Coulomb screening.
        p_lesser (DBSparse): Lesser Coulomb screening.
        p_greater (DBSparse): Greater Coulomb screening.
        w_retarded (DBSparse): Retarded screened Coulomb interaction.
        w_lesser (DBSparse): Lesser screened Coulomb interaction.
        w_greater (DBSparse): Greater screened Coulomb interaction.

        # Photon
        pi_photon_retarded (DBSparse): Retarded photon self-energy.
        pi_photon_lesser (DBSparse): Lesser photon self-energy.
        pi_photon_greater (DBSparse): Greater photon self-energy.
        d_photon_retarded (DBSparse): Retarded photon Green's function.
        d_photon_lesser (DBSparse): Lesser photon Green's function.
        d_photon_greater (DBSparse): Greater photon Green's function.

        # Phonon
        pi_phonon_retarded (DBSparse): Retarded phonon self-energy.
        pi_phonon_lesser (DBSparse): Lesser phonon self-energy.
        pi_phonon_greater (DBSparse): Greater phonon self-energy.
        d_phonon_retarded (DBSparse): Retarded phonon Green's function.
        d_phonon_lesser (DBSparse): Lesser phonon Green's function.
        d_phonon_greater (DBSparse): Greater phonon Green's function.
    """

    # ----- Electron data ------------------------------------------------------------
    sigma_retarded_prev: DBSparse = None
    sigma_lesser_prev: DBSparse = None
    sigma_greater_prev: DBSparse = None
    sigma_retarded: DBSparse = None
    sigma_lesser: DBSparse = None
    sigma_greater: DBSparse = None
    g_retarded: DBSparse = None
    g_lesser: DBSparse = None
    g_greater: DBSparse = None

    # ----- Coulomb screening data ---------------------------------------------------
    p_retarded: DBSparse = None
    p_lesser: DBSparse = None
    p_greater: DBSparse = None
    w_retarded: DBSparse = None
    w_lesser: DBSparse = None
    w_greater: DBSparse = None

    # ----- Photon data --------------------------------------------------------------
    pi_photon_retarded: DBSparse = None
    pi_photon_lesser: DBSparse = None
    pi_photon_greater: DBSparse = None
    d_photon_retarded: DBSparse = None
    d_photon_lesser: DBSparse = None
    d_photon_greater: DBSparse = None

    # ----- Phonon data --------------------------------------------------------------
    pi_phonon_retarded: DBSparse = None
    pi_phonon_lesser: DBSparse = None
    pi_phonon_greater: DBSparse = None
    d_phonon_retarded: DBSparse = None
    d_phonon_lesser: DBSparse = None
    d_phonon_greater: DBSparse = None


@dataclass
class Observables:
    # Electron
    electron_ldos: np.ndarray
    electron_density: np.ndarray
    hole_density: np.ndarray
    electron_current: np.ndarray

    electron_electron_scattering_rate: np.ndarray
    electron_photon_scattering_rate: np.ndarray
    electron_phonon_scattering_rate: np.ndarray

    sigma_retarded_density: np.ndarray
    sigma_lesser_density: np.ndarray
    sigma_greater_density: np.ndarray

    # Coulomb Screening
    w_retarded_density: np.ndarray
    w_lesser_density: np.ndarray
    w_greater_density: np.ndarray

    p_retarded_density: np.ndarray
    p_lesser_density: np.ndarray
    p_greater_density: np.ndarray

    # Photon
    pi_photon_retarded_density: np.ndarray
    pi_photon_lesser_density: np.ndarray
    pi_photon_greater_density: np.ndarray

    d_photon_retarded_density: np.ndarray
    d_photon_lesser_density: np.ndarray
    d_photon_greater_density: np.ndarray

    photon_current_density: np.ndarray

    # Phonon
    pi_phonon_retarded_density: np.ndarray
    pi_phonon_lesser_density: np.ndarray
    pi_phonon_greater_density: np.ndarray
    d_phonon_retarded_density: np.ndarray
    d_phonon_lesser_density: np.ndarray
    d_phonon_greater_density: np.ndarray

    thermal_current: np.ndarray


class SCBA:
    """Computes the self-consistent Born approximation to convergence.

    Parameters
    ----------
    quatrex_config : Path
        Quatrex configuration file.
    compute_config : Path, optional
        Compute configuration file.

    """

    def __init__(
        self, quatrex_config: QuatrexConfig, compute_config: ComputeConfig = None
    ) -> None:
        self.quatrex_config = quatrex_config
        if compute_config is None:
            compute_config = ComputeConfig()

        self.compute_config = compute_config
        self.dbsparse = compute_config.dbsparse

        self.data = SCBAData()

        # ----- Electron init ----------------------------------------------------------
        self.electron_energies = np.load(
            self.quatrex_config.input_dir / "electron_energies.npy"
        )
        self.local_electron_energies = slice_local_array(self.electron_energies)

        self.electron_solver = ElectronSolver(self.quatrex_config)

        self._initialize_electron_data()

        # ----- Coulomb screening init -------------------------------------------------
        if self.quatrex_config.scba.coulomb_screening:
            energies_path = (
                self.quatrex_config.input_dir / "coulomb_screening_energies.npy"
            )

            if os.path.isfile(energies_path):
                self.coulomb_screening_energies = np.load(energies_path)
            else:
                self.coulomb_screening_energies = self.electron_energies

            self.local_coulomb_screening_energies = slice_local_array(
                self.coulomb_screening_energies
            )

            self.pol_coulomb_screening = PCoulombScreening(...)
            self.coulomb_screening_solver = CoulombScreeningSolver(...)
            self.sse_coulomb_screening = SigmaCoulombScreening(...)

            self._initialize_coulomb_screening_data()

        # ----- Photon init ------------------------------------------------------------
        if self.quatrex_config.scba.photon:
            energies_path = self.quatrex_config.input_dir / "photon_energies.npy"

            self.photon_energies = np.load(energies_path)

            self.local_photon_energies = slice_local_array(self.photon_energies)

            self.pol_photon = PiPhoton(...)
            self.photon_solver = PhotonSolver(...)
            self._initialize_photon_data()

            self.sse_photon = SigmaPhoton(...)

        # ----- Phonon init ------------------------------------------------------------
        if self.quatrex_config.scba.phonon:
            energies_path = self.quatrex_config.input_dir / "phonon_energies.npy"

            self.phonon_energies = np.load(energies_path)

            self.local_phonon_energies = slice_local_array(self.phonon_energies)

            if self.quatrex_config.phonon.model == "negf":
                self.pol_phonon = PiPhonon(...)
                self.phonon_solver = PhononSolver(...)
                self._initialize_phonon_data()

            self.sse_phonon = SigmaPhonon(...)

    def _initialize_electron_data(self) -> None:
        self.data.sigma_retarded_prev = self.dbsparse.zeros_like(
            self.electron_solver.system_matrix
        )
        self.data.sigma_lesser_prev = self.dbsparse.zeros_like(
            self.electron_solver.system_matrix
        )
        self.data.sigma_greater_prev = self.dbsparse.zeros_like(
            self.electron_solver.system_matrix
        )
        self.data.sigma_retarded = self.dbsparse.zeros_like(
            self.electron_solver.system_matrix
        )
        self.data.sigma_lesser = self.dbsparse.zeros_like(
            self.electron_solver.system_matrix
        )
        self.data.sigma_greater = self.dbsparse.zeros_like(
            self.electron_solver.system_matrix
        )
        self.data.g_retarded = self.dbsparse.zeros_like(
            self.electron_solver.system_matrix
        )
        self.data.g_lesser = self.dbsparse.zeros_like(
            self.electron_solver.system_matrix
        )
        self.data.g_greater = self.dbsparse.zeros_like(
            self.electron_solver.system_matrix
        )

    def _initialize_coulomb_screening_data(self) -> None:
        self.data.p_retarded = self.dbsparse.zeros_like(
            self.coulomb_screening_solver.system_matrix
        )
        self.data.p_lesser = self.dbsparse.zeros_like(
            self.coulomb_screening_solver.system_matrix
        )
        self.data.p_greater = self.dbsparse.zeros_like(
            self.coulomb_screening_solver.system_matrix
        )
        self.data.w_retarded = self.dbsparse.zeros_like(
            self.coulomb_screening_solver.system_matrix
        )
        self.data.w_lesser = self.dbsparse.zeros_like(
            self.coulomb_screening_solver.system_matrix
        )
        self.data.w_greater = self.dbsparse.zeros_like(
            self.coulomb_screening_solver.system_matrix
        )

    def _initialize_photon_data(self) -> None:
        self.data.pi_photon_retarded = self.dbsparse.zeros_like(
            self.photon_solver.system_matrix
        )
        self.data.pi_photon_lesser = self.dbsparse.zeros_like(
            self.photon_solver.system_matrix
        )
        self.data.pi_photon_greater = self.dbsparse.zeros_like(
            self.photon_solver.system_matrix
        )
        self.data.d_photon_retarded = self.dbsparse.zeros_like(
            self.photon_solver.system_matrix
        )
        self.data.d_photon_lesser = self.dbsparse.zeros_like(
            self.photon_solver.system_matrix
        )
        self.data.d_photon_greater = self.dbsparse.zeros_like(
            self.photon_solver.system_matrix
        )

    def _initialize_phonon_data(self) -> None:
        self.data.pi_phonon_retarded = self.dbsparse.zeros_like(
            self.phonon_solver.system_matrix
        )
        self.data.pi_phonon_lesser = self.dbsparse.zeros_like(
            self.phonon_solver.system_matrix
        )
        self.data.pi_phonon_greater = self.dbsparse.zeros_like(
            self.phonon_solver.system_matrix
        )
        self.data.d_phonon_retarded = self.dbsparse.zeros_like(
            self.phonon_solver.system_matrix
        )
        self.data.d_phonon_lesser = self.dbsparse.zeros_like(
            self.phonon_solver.system_matrix
        )
        self.data.d_phonon_greater = self.dbsparse.zeros_like(
            self.phonon_solver.system_matrix
        )

    def _compute_phonon_interaction(self):
        if self.quatrex_config.phonon.model == "negf":
            raise NotImplementedError

        elif self.quatrex_config.phonon.model == "pseudo-scattering":
            self.sse_phonon.compute(
                self.data.g_lesser,
                self.data.g_greater,
                out=(
                    self.data.sigma_lesser,
                    self.data.sigma_greater,
                    self.data.sigma_retarded,
                ),
            )

    def _compute_photon_interaction(self):
        raise NotImplementedError

    def _compute_coulomb_screening_interaction(self):
        raise NotImplementedError

    def run(self) -> None:
        """Runs the SCBA to convergence."""
        for __ in range(self.quatrex_config.scba.max_iterations):
            self.electron_solver.solve_lesser_greater(
                self.data.sigma_lesser,
                self.data.sigma_greater,
                self.data.sigma_retarded,
                out=(self.data.g_lesser, self.data.g_greater, self.data.g_retarded),
            )

            # Swap current and previous self-energy buffers.
            self.data.sigma_retarded.data, self.data.sigma_retarded_prev.data = (
                self.data.sigma_retarded_prev.data,
                self.data.sigma_retarded.data,
            )
            self.data.sigma_lesser.data, self.data.sigma_lesser_prev.data = (
                self.data.sigma_lesser_prev.data,
                self.data.sigma_lesser.data,
            )
            self.data.sigma_greater.data, self.data.sigma_greater_prev.data = (
                self.data.sigma_greater_prev.data,
                self.data.sigma_greater.data,
            )

            # Reset self-energy.
            self.data.sigma_retarded.data[:] = 0.0
            self.data.sigma_lesser.data[:] = 0.0
            self.data.sigma_greater.data[:] = 0.0

            if self.quatrex_config.scba.coulomb_screening:
                self._compute_coulomb_screening_interaction()

            if self.quatrex_config.scba.photon:
                self._compute_photon_interaction()

            if self.quatrex_config.scba.phonon:
                self._compute_phonon_interaction()

            if self.has_converged():
                print(f"SCBA converged after {__} iterations.")
                break

            # Update self-energy with mixing factor.
            m = self.quatrex_config.scba.mixing_factor
            self.data.sigma_retarded = (
                1 - m
            ) * self.data.sigma_retarded_prev + m * self.data.sigma_retarded
            self.data.sigma_retarded = (
                1 - m
            ) * self.data.sigma_lesser_prev + m * self.data.sigma_lesser
            self.data.sigma_greater = (
                1 - m
            ) * self.data.sigma_greater_prev + m * self.data.sigma_greater

        else:  # Did not break, i.e. max_iterations reached.
            print(f"SCBA did not converge after {__} iterations.")
