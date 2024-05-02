# Copyright 2023-2024 ETH Zurich and QuaTrEx authors. All rights reserved.
import os

import numpy.linalg as npla
import numpy as np


from qttools.datastructures.dbcsr import DBCSR


from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.compute_config import ComputeConfig

from quatrex.electron.electron_solver import ElectronSolver
from quatrex.electron.sigma_phonon import SigmaPhonon

from quatrex.coulomb_screening.coulomb_screening_solver import CoulombScreeningSolver
from quatrex.photon.photon_solver import PhotonSolver
from quatrex.phonon.phonon_solver import PhononSolver

from quatrex.utils.dist_utils import slice_local_array


@dataclass
class SCBAData:
    """
    Dataclass storing the data used in the Self-Consistent Born Approximation (SCBA). For
    all possible sub-systems.

    Attributes:
        # Electron
        sigma_retarded (DBCSR): Retarded self-energy.
        sigma_lesser (DBCSR): Lesser self-energy.
        sigma_greater (DBCSR): Greater self-energy.
        g_retarded (DBCSR): Retarded Green's function.
        g_lesser (DBCSR): Lesser Green's function.
        g_greater (DBCSR): Greater Green's function.

        # Coulomb screening
        p_retarded (DBCSR): Retarded Coulomb screening.
        p_lesser (DBCSR): Lesser Coulomb screening.
        p_greater (DBCSR): Greater Coulomb screening.
        w_retarded (DBCSR): Retarded screened Coulomb interaction.
        w_lesser (DBCSR): Lesser screened Coulomb interaction.
        w_greater (DBCSR): Greater screened Coulomb interaction.

        # Photon
        pi_photon_retarded (DBCSR): Retarded photon self-energy.
        pi_photon_lesser (DBCSR): Lesser photon self-energy.
        pi_photon_greater (DBCSR): Greater photon self-energy.
        d_photon_retarded (DBCSR): Retarded photon Green's function.
        d_photon_lesser (DBCSR): Lesser photon Green's function.
        d_photon_greater (DBCSR): Greater photon Green's function.

        # Phonon
        pi_phonon_retarded (DBCSR): Retarded phonon self-energy.
        pi_phonon_lesser (DBCSR): Lesser phonon self-energy.
        pi_phonon_greater (DBCSR): Greater phonon self-energy.
        d_phonon_retarded (DBCSR): Retarded phonon Green's function.
        d_phonon_lesser (DBCSR): Lesser phonon Green's function.
        d_phonon_greater (DBCSR): Greater phonon Green's function.
    """

    # ----- Electron data ------------------------------------------------------------
    sigma_retarded: DBCSR = None
    sigma_lesser: DBCSR = None
    sigma_greater: DBCSR = None
    g_retarded: DBCSR = None
    g_lesser: DBCSR = None
    g_greater: DBCSR = None

    # ----- Coulomb screening data ---------------------------------------------------
    p_retarded: DBCSR = None
    p_lesser: DBCSR = None
    p_greater: DBCSR = None
    w_retarded: DBCSR = None
    w_lesser: DBCSR = None
    w_greater: DBCSR = None

    # ----- Photon data --------------------------------------------------------------
    pi_photon_retarded: DBCSR = None
    pi_photon_lesser: DBCSR = None
    pi_photon_greater: DBCSR = None
    d_photon_retarded: DBCSR = None
    d_photon_lesser: DBCSR = None
    d_photon_greater: DBCSR = None

    # ----- Phonon data --------------------------------------------------------------
    pi_phonon_retarded: DBCSR = None
    pi_phonon_lesser: DBCSR = None
    pi_phonon_greater: DBCSR = None
    d_phonon_retarded: DBCSR = None
    d_phonon_lesser: DBCSR = None
    d_phonon_greater: DBCSR = None

@dataclass
class Observables:
    #Electron
    electron_ldos:
    electron_density:
    hole_density:
    electron_current:

    electron_electron_scattering_rate:
    electron_photon_scattering_rate:
    electron_phonon_scattering_rate:

    sigma_retarded_density:
    sigma_lesser_density:
    sigma_greater_density:

    #Coulomb Screening
    w_retarded_density:
    w_lesser_density:
    w_greater_density:

    p_retarded_density:
    p_lesser_density:
    p_greater_density:

    #Photon
    pi_photon_retarded:
    pi_photon_lesser:
    pi_photon_greater:

    d_photon_retarded:
    d_photon_lesser:
    d_photon_greater:

    photon_current:

    #Phonon
    pi_phonon_retarded:
    pi_phonon_lesser:
    pi_phonon_greater:
    d_phonon_retarded:
    d_phonon_lesser:
    d_phonon_greater:

    thermal_current:


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
            self.compute_config = ComputeConfig()
        else:
            self.compute_config = compute_config

        self.data = SCBAData()

        # ----- Electron init ----------------------------------------------------------
        self.electron_energies = np.load(
            self.quatrex_config.input_dir() + "/electron_energies.npy"
        )
        self.local_electron_energies = self.slice_local_array(self.electron_energies)

        self.electron_solver = ElectronSolver(self.quatrex_config)

        self._initialize_electron_data()

        # ----- Coulomb screening init -------------------------------------------------
        if self.quatrex_config.scba.coulomb_screening:
            energies_path = (
                self.quatrex_config.input_dir() + "/coulomb_screening_energies.npy"
            )

            if os.path.isfile(energies_path):
                self.coulomb_screening_energies = np.load(energies_path)
            else:
                self.coulomb_screening_energies = self.electron_energies

            self.local_coulomb_screening_energies = self.slice_local_array(
                self.coulomb_screening_energies
            )

            self.coulomb_screening_solver = CoulombScreeningSolver(
                self.quatrex_config,
                self.compute_config,
                self.local_coulomb_screening_energies,
            )

            self._initialize_coulomb_screening_data()

        # ----- Photon init ------------------------------------------------------------
        if self.quatrex_config.scba.photon:
            energies_path = self.quatrex_config.input_dir() + "/photon_energies.npy"

            self.photon_energies = np.load(energies_path)

            self.local_photon_energies = self.slice_local_array(self.photon_energies)

            self.coulomb_screening_solver = PhotonSolver(
                self.quatrex_config,
                self.compute_config,
                self.local_photon_energies,
            )

            self._initialize_photon_data()

        # ----- Phonon init ------------------------------------------------------------
        if self.quatrex_config.scba.phonon:
            energies_path = self.quatrex_config.input_dir() + "/phonon_energies.npy"

            self.phonon_energies = np.load(energies_path)

            self.local_phonon_energies = self.slice_local_array(self.phonon_energies)

            self.coulomb_screening_solver = PhononSolver(
                self.quatrex_config,
                self.compute_config,
                self.local_phonon_energies,
            )

            self._initialize_phonon_data()

    def _initialize_electron_data(self) -> None:
        self.data.sigma_retarded = DBCSR.zeros_like(
            self.electron_solver.hamiltonian_dbsparse
        )
        self.data.sigma_lesser = DBCSR.zeros_like(
            self.electron_solver.hamiltonian_dbsparse
        )
        self.data.sigma_greater = DBCSR.zeros_like(
            self.electron_solver.hamiltonian_dbsparse
        )
        self.data.g_retarded = DBCSR.zeros_like(
            self.electron_solver.hamiltonian_dbsparse
        )
        self.data.g_lesser = DBCSR.zeros_like(self.electron_solver.hamiltonian_dbsparse)
        self.data.g_greater = DBCSR.zeros_like(
            self.electron_solver.hamiltonian_dbsparse
        )

    def _initialize_coulomb_screening_data(self) -> None: ...

    def _initialize_photon_data(self) -> None: ...

    def _initialize_phonon_data(self) -> None: ...

    def run(self) -> None:

        scba = self._warm_up_iteration(self.quatrex_config)

        for __ in range(1, self.quatrex_config.scba.max_iterations + 1):
            g_lesser, g_greater = scba.electron_solver.solve_lesser_greater()

            sigma_lesser = None
            sigma_greater = None

            if self.quatrex_config.scba.phonon:
                if self.quatrex_config.phonon.model == "greens_function":
                    raise NotImplementedError

                elif self.quatrex_config.phonon.model == "deformation_potential":
                    scba.sigma_phonon.g_lesser = g_lesser
                    scba.sigma_phonon.g_greater = g_greater

                    sigma_phonon_lesser, sigma_phonon_greater = (
                        scba.sigma_phonon.compute()
                    )

                    sigma_lesser += sigma_phonon_lesser
                    sigma_greater += sigma_phonon_greater

            # Compute difference between current and preceeding self-energy.
            sigma_retarded = 0.5 * (sigma_greater - sigma_lesser)

            prev_sigma_retarded = 0.5 * (
                scba.electron_solver.sigma_greater - scba.electron_solver.sigma_lesser
            )

            diff_causal = sigma_retarded - prev_sigma_retarded

            abs_norm_diff_causal = npla.norm(diff_causal)
            rel_norm_diff_causal = abs_norm_diff_causal / npla.norm(sigma_retarded)

            # Update self-energy.
            m = self.quatrex_config.scba.mixing_factor
            scba.electron_solver.sigma_lesser = (
                1 - m
            ) * scba.electron_solver.sigma_lesser + m * sigma_lesser
            scba.electron_solver.sigma_greater = (
                1 - m
            ) * scba.electron_solver.sigma_greater + m * sigma_greater

            if rel_norm_diff_causal < self.quatrex_config.scba.tolerance:
                print(f"SCBA converged after {__} iterations.")
                break

        else:  # Did not break, i.e. max_iterations reached.
            print(f"SCBA did not converge after {__} iterations.")
