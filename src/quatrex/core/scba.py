# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import os
import time
from dataclasses import dataclass, field

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from qttools.datastructures import DSBSparse

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.observables import contact_currents, density
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

        allocate_electron_quantity = _get_allocator(
            scba.compute_config.dbsparse_type,
            scba.electron_solver.system_matrix,
        )
        self.sigma_retarded_prev = allocate_electron_quantity()
        self.sigma_lesser_prev = allocate_electron_quantity()
        self.sigma_greater_prev = allocate_electron_quantity()
        self.sigma_retarded = allocate_electron_quantity()
        self.sigma_lesser = allocate_electron_quantity()
        self.sigma_greater = allocate_electron_quantity()
        self.g_retarded = allocate_electron_quantity()
        self.g_lesser = allocate_electron_quantity()
        self.g_greater = allocate_electron_quantity()

        if hasattr(scba, "coulomb_screening_solver"):
            allocate_coulomb_screening_quantity = _get_allocator(
                scba.compute_config.dbsparse_type,
                scba.coulomb_screening_solver.system_matrix,
            )
            self.p_retarded = allocate_coulomb_screening_quantity()
            self.p_lesser = allocate_coulomb_screening_quantity()
            self.p_greater = allocate_coulomb_screening_quantity()
            self.w_retarded = allocate_coulomb_screening_quantity()
            self.w_lesser = allocate_coulomb_screening_quantity()
            self.w_greater = allocate_coulomb_screening_quantity()

        if hasattr(scba, "photon_solver"):
            allocate_photon_quantity = _get_allocator(
                scba.compute_config.dbsparse_type,
                scba.photon_solver.system_matrix,
            )
            self.pi_photon_retarded = allocate_photon_quantity()
            self.pi_photon_lesser = allocate_photon_quantity()
            self.pi_photon_greater = allocate_photon_quantity()
            self.d_photon_retarded = allocate_photon_quantity()
            self.d_photon_lesser = allocate_photon_quantity()
            self.d_photon_greater = allocate_photon_quantity()

        if hasattr(scba, "phonon_solver"):
            allocate_phonon_quantity = _get_allocator(
                scba.compute_config.dbsparse_type,
                scba.phonon_solver.system_matrix,
            )
            self.pi_phonon_retarded = allocate_phonon_quantity()
            self.pi_phonon_lesser = allocate_phonon_quantity()
            self.pi_phonon_greater = allocate_phonon_quantity()
            self.d_phonon_retarded = allocate_phonon_quantity()
            self.d_phonon_lesser = allocate_phonon_quantity()
            self.d_phonon_greater = allocate_phonon_quantity()


@dataclass
class Observables:
    # --- Electrons ----------------------------------------------------
    electron_ldos: np.ndarray = None
    electron_density: np.ndarray = None
    hole_density: np.ndarray = None
    electron_current: dict = field(default_factory=dict)

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
            if self.quatrex_config.phonon.model == "negf":
                energies_path = self.quatrex_config.input_dir / "phonon_energies.npy"
                self.phonon_energies = np.load(energies_path)
                self.pi_phonon = PiPhonon(...)
                self.phonon_solver = PhononSolver(
                    self.quatrex_config,
                    self.compute_config,
                    self.phonon_energies,
                    ...,
                )
                self.sigma_phonon = SigmaPhonon(...)

            elif self.quatrex_config.phonon.model == "pseudo-scattering":
                self.sigma_phonon = SigmaPhonon(quatrex_config, self.electron_energies)

        self.data = SCBAData(self)
        self.observables = Observables()

    def _swap_sigma(self) -> None:
        """Swaps the current and previous self-energy buffers."""
        self.data.sigma_lesser._data[:], self.data.sigma_lesser_prev._data[:] = (
            self.data.sigma_lesser_prev._data,
            self.data.sigma_lesser._data,
        )
        self.data.sigma_greater._data[:], self.data.sigma_greater_prev._data[:] = (
            self.data.sigma_greater_prev._data,
            self.data.sigma_greater._data,
        )
        self.data.sigma_retarded._data[:], self.data.sigma_retarded_prev._data[:] = (
            self.data.sigma_retarded_prev._data,
            self.data.sigma_retarded._data,
        )

    def _update_sigma(self) -> None:
        """Updates the self-energy with a mixing factor."""
        mixing_factor = self.quatrex_config.scba.mixing_factor
        self.data.sigma_lesser.data[:] = (
            (1 - mixing_factor) * self.data.sigma_lesser_prev.data
            + mixing_factor * self.data.sigma_lesser.data
        )
        self.data.sigma_greater.data[:] = (
            (1 - mixing_factor) * self.data.sigma_greater_prev.data
            + mixing_factor * self.data.sigma_greater.data
        )
        self.data.sigma_retarded.data[:] = (
            (1 - mixing_factor) * self.data.sigma_retarded_prev.data
            + mixing_factor * self.data.sigma_retarded.data
        )

    def _has_converged(self) -> bool:
        """Checks if the SCBA has converged."""
        # Relative infinity norm of the self-energy update.
        diff = self.data.sigma_retarded.data - self.data.sigma_retarded_prev.data
        max_diff = np.max(np.abs(diff))
        # # rel_max_diff = max_diff / np.max(np.abs(self.data.sigma_retarded.data))
        max_diff = comm.allreduce(max_diff, op=MPI.MAX)
        (
            print(f"Maximum Self-Energy Update: {max_diff}", flush=True)
            if comm.rank == 0
            else None
        )
        # if max_diff < self.quatrex_config.scba.convergence_tol:
        #     return True
        # return False
        i_left, i_right = contact_currents(self.electron_solver)
        change_left = np.linalg.norm(
            i_left.real - self.observables.electron_current.get("left", 0.0)
        )
        change_right = np.linalg.norm(
            i_right.real - self.observables.electron_current.get("right", 0.0)
        )
        ave_change = 0.5 * (change_left + change_right)
        (
            print(f"Average Current Change: {ave_change}", flush=True)
            if comm.rank == 0
            else None
        )
        rel_change_left = change_left / np.linalg.norm(i_left.real)
        rel_change_right = change_right / np.linalg.norm(i_right.real)
        rel_ave_change = 0.5 * (rel_change_left + rel_change_right)
        (
            print(f"Relative Average Current Change: {rel_ave_change}", flush=True)
            if comm.rank == 0
            else None
        )

        diff = i_left.real.sum() + i_right.real.sum()
        print(f"Current Difference: {diff}", flush=True) if comm.rank == 0 else None

        rel_diff = diff / (i_left.real.sum() - i_right.real.sum())
        (
            print(f"Relative Current Conservation: {rel_diff}", flush=True)
            if comm.rank == 0
            else None
        )

        # if ave_change < self.quatrex_config.scba.convergence_tol:
        #     return True
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

        self.observables.electron_current = dict(
            zip(("left", "right"), contact_currents(self.electron_solver))
        )

    def run(self) -> None:
        """Runs the SCBA to convergence."""
        print("Entering SCBA loop...", flush=True) if comm.rank == 0 else None
        times = []
        for i in range(self.quatrex_config.scba.max_iterations):
            print(f"Iteration {i}", flush=True) if comm.rank == 0 else None
            times.append(time.perf_counter())

            times.append(time.perf_counter())
            self.electron_solver.solve(
                self.data.sigma_lesser,
                self.data.sigma_greater,
                self.data.sigma_retarded,
                out=(self.data.g_lesser, self.data.g_greater, self.data.g_retarded),
            )
            t_solve = time.perf_counter() - times.pop()
            (
                print(f"Time for electron solver: {t_solve:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )
            # Swap current with previous self-energy buffer.
            times.append(time.perf_counter())
            self._swap_sigma()
            t_swap = time.perf_counter() - times.pop()
            (
                print(f"Time for swapping: {t_swap:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            # Reset current self-energy.
            times.append(time.perf_counter())
            self.data.sigma_retarded._data[:] = 0.0
            self.data.sigma_lesser._data[:] = 0.0
            self.data.sigma_greater._data[:] = 0.0
            t_reset = time.perf_counter() - times.pop()
            (
                print(f"Time for resetting: {t_reset:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            if self.quatrex_config.scba.coulomb_screening:
                self._compute_coulomb_screening_interaction()

            if self.quatrex_config.scba.photon:
                self._compute_photon_interaction()

            times.append(time.perf_counter())
            if self.quatrex_config.scba.phonon:
                self._compute_phonon_interaction()
            t_phonon = time.perf_counter() - times.pop()
            (
                print(f"Time for phonon interaction: {t_phonon:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            self.observables.electron_current = dict(
                zip(("left", "right"), contact_currents(self.electron_solver))
            )

            times.append(time.perf_counter())
            if self._has_converged():
                (
                    print(f"SCBA converged after {i} iterations.")
                    if comm.rank == 0
                    else None
                )
                break
            t_convergence = time.perf_counter() - times.pop()
            (
                print(f"Time for convergence check: {t_convergence:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            # Update self-energy for next iteration with mixing factor.
            times.append(time.perf_counter())
            self._update_sigma()
            t_update = time.perf_counter() - times.pop()
            (
                print(f"Time for updating: {t_update:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            t_iteration = time.perf_counter() - times.pop()
            (
                print(f"Time for iteration: {t_iteration:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

        else:  # Did not break, i.e. max_iterations reached.
            (
                print(f"SCBA did not converge after {i} iterations.")
                if comm.rank == 0
                else None
            )
            self._compute_observables()
