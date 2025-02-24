# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import os
import time
from dataclasses import dataclass, field

from cupyx.profiler import time_range
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, xp
from qttools.utils.mpi_utils import distributed_load

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.observables import contact_currents, density, device_current
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.utils import compute_num_connected_blocks, compute_sparsity_pattern
from quatrex.coulomb_screening import CoulombScreeningSolver, PCoulombScreening
from quatrex.electron import (
    ElectronSolver,
    SigmaCoulombScreening,
    SigmaFock,
    SigmaPhonon,
    SigmaPhoton,
)
from quatrex.phonon import PhononSolver, PiPhonon
from quatrex.photon import PhotonSolver, PiPhoton


class SCBAData:
    """Data container class for the SCBA.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The Quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
    ) -> None:
        """Initializes the SCBA data."""
        # Load orbital positions, energy vector and block-sizes.
        grid = distributed_load(quatrex_config.input_dir / "grid.npy")
        electron_energies = distributed_load(
            quatrex_config.input_dir / "electron_energies.npy"
        )
        block_sizes = distributed_load(quatrex_config.input_dir / "block_sizes.npy")

        # Find the maximum interaction cutoff.
        max_interaction_cutoff = 0.0
        if quatrex_config.scba.coulomb_screening:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                quatrex_config.coulomb_screening.interaction_cutoff,
            )
        if quatrex_config.scba.photon:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                quatrex_config.photon.interaction_cutoff,
            )
        if quatrex_config.scba.phonon:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                quatrex_config.phonon.interaction_cutoff,
            )

        if comm.rank == 0:
            print(f"Max Interaction Cutoff: {max_interaction_cutoff}", flush=True)

        self.sparsity_pattern = compute_sparsity_pattern(grid, max_interaction_cutoff)

        dsbsparse_type = compute_config.dsbsparse_type

        self.g_retarded = dsbsparse_type.from_sparray(
            self.sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=electron_energies.shape,
        )
        self.g_retarded._data[:] = 0.0  # Initialize to zero.
        self.g_lesser = dsbsparse_type.zeros_like(self.g_retarded)
        self.g_greater = dsbsparse_type.zeros_like(self.g_retarded)

        self.sigma_retarded_prev = dsbsparse_type.zeros_like(self.g_retarded)
        self.sigma_lesser_prev = dsbsparse_type.zeros_like(self.g_retarded)
        self.sigma_greater_prev = dsbsparse_type.zeros_like(self.g_retarded)
        self.sigma_retarded = dsbsparse_type.zeros_like(self.g_retarded)
        self.sigma_lesser = dsbsparse_type.zeros_like(self.g_retarded)
        self.sigma_greater = dsbsparse_type.zeros_like(self.g_retarded)

        if quatrex_config.scba.coulomb_screening:
            # NOTE: The polarization has the same sparsity pattern as
            # the electronic system (the interactions are local in real
            # space). However, we need to change the block sizes of the
            # screened Coulomb interaction.
            self.p_retarded = dsbsparse_type.zeros_like(self.g_retarded)
            self.p_lesser = dsbsparse_type.zeros_like(self.g_retarded)
            self.p_greater = dsbsparse_type.zeros_like(self.g_retarded)

            num_connected_blocks = quatrex_config.coulomb_screening.num_connected_blocks
            if num_connected_blocks == "auto":
                num_connected_blocks = compute_num_connected_blocks(
                    self.sparsity_pattern, block_sizes
                )

            if comm.rank == 0:
                print(f"Number of connected blocks: {num_connected_blocks}", flush=True)

            # TODO: This only works for constant block sizes.
            coulomb_screening_block_sizes = (
                block_sizes[: len(block_sizes) // num_connected_blocks]
                * num_connected_blocks
            )

            self.w_lesser = dsbsparse_type.from_sparray(
                self.sparsity_pattern.astype(xp.complex128),
                block_sizes=coulomb_screening_block_sizes,
                global_stack_shape=electron_energies.shape,
            )
            self.w_greater = dsbsparse_type.zeros_like(self.w_lesser)

        # TODO: The interactions with photons and phonons are not yet
        # implemented.
        if quatrex_config.scba.photon:
            raise NotImplementedError

        if quatrex_config.scba.phonon and quatrex_config.phonon.model == "negf":
            raise NotImplementedError


@dataclass
class Observables:
    """Observable quantities for the SCBA."""

    # --- Electrons ----------------------------------------------------
    electron_ldos: NDArray = None
    electron_density: NDArray = None
    hole_density: NDArray = None
    electron_current: dict = field(default_factory=dict)

    valence_band_edges: NDArray = None
    conduction_band_edges: NDArray = None

    excess_charge_density: NDArray = None

    electron_electron_scattering_rate: NDArray = None
    electron_photon_scattering_rate: NDArray = None
    electron_phonon_scattering_rate: NDArray = None

    sigma_retarded_density: NDArray = None
    sigma_lesser_density: NDArray = None
    sigma_greater_density: NDArray = None

    # --- Coulomb screening --------------------------------------------
    w_lesser_density: NDArray = None
    w_greater_density: NDArray = None

    p_retarded_density: NDArray = None
    p_lesser_density: NDArray = None
    p_greater_density: NDArray = None

    # --- Photons ------------------------------------------------------
    pi_photon_retarded_density: NDArray = None
    pi_photon_lesser_density: NDArray = None
    pi_photon_greater_density: NDArray = None

    d_photon_retarded_density: NDArray = None
    d_photon_lesser_density: NDArray = None
    d_photon_greater_density: NDArray = None

    photon_current_density: NDArray = None

    # --- Phonons ------------------------------------------------------
    pi_phonon_retarded_density: NDArray = None
    pi_phonon_lesser_density: NDArray = None
    pi_phonon_greater_density: NDArray = None
    d_phonon_retarded_density: NDArray = None
    d_phonon_lesser_density: NDArray = None
    d_phonon_greater_density: NDArray = None

    thermal_current: NDArray = None


class SCBA:
    """Self-consistent Born approximation (SCBA) solver.

    Parameters
    ----------
    quatrex_config : Path
        Quatrex configuration file.
    compute_config : Path, optional
        Compute configuration file, by default None. If None, the
        default compute parameters are used.

    """

    @time_range()
    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig | None = None,
    ) -> None:
        """Initializes an SCBA instance."""
        self.quatrex_config = quatrex_config

        if compute_config is None:
            compute_config = ComputeConfig()

        self.compute_config = compute_config

        self.data = SCBAData(quatrex_config, compute_config)
        self.observables = Observables()
        self.mixing_factor = self.quatrex_config.scba.mixing_factor

        # ----- Electrons ----------------------------------------------
        self.electron_energies = distributed_load(
            self.quatrex_config.input_dir / "electron_energies.npy"
        )
        self.electron_solver = ElectronSolver(
            self.quatrex_config,
            self.compute_config,
            self.electron_energies,
            sparsity_pattern=self.data.sparsity_pattern,
        )

        # ----- Coulomb screening --------------------------------------
        if self.quatrex_config.scba.coulomb_screening:
            energies_path = (
                self.quatrex_config.input_dir / "coulomb_screening_energies.npy"
            )
            if os.path.isfile(energies_path):
                self.coulomb_screening_energies = distributed_load(energies_path)
            else:
                self.coulomb_screening_energies = (
                    self.electron_energies - self.electron_energies[0]
                )
                # Remove the zero energy to avoid division by zero.
                self.coulomb_screening_energies += 1e-6

            self.sigma_fock = SigmaFock(
                self.quatrex_config,
                self.compute_config,
                self.electron_energies,
                sparsity_pattern=self.data.sparsity_pattern,
            )
            # NOTE: No sparsity information required here.
            self.p_coulomb_screening = PCoulombScreening(
                self.quatrex_config,
                self.coulomb_screening_energies,
            )
            self.coulomb_screening_solver = CoulombScreeningSolver(
                self.quatrex_config,
                self.compute_config,
                self.coulomb_screening_energies,
                sparsity_pattern=self.data.sparsity_pattern,
            )
            self.sigma_coulomb_screening = SigmaCoulombScreening(
                self.quatrex_config,
                self.compute_config,
                self.electron_energies,
            )

        # ----- Photons ------------------------------------------------
        if self.quatrex_config.scba.photon:
            energies_path = self.quatrex_config.input_dir / "photon_energies.npy"
            self.photon_energies = distributed_load(energies_path)
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
                self.phonon_energies = distributed_load(energies_path)
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

    def _stash_sigma(self) -> None:
        """Stash the current into the previous self-energy buffers."""
        self.data.sigma_lesser_prev._data[:] = self.data.sigma_lesser._data
        self.data.sigma_greater_prev._data[:] = self.data.sigma_greater._data
        self.data.sigma_retarded_prev._data[:] = self.data.sigma_retarded._data

        self.data.sigma_retarded._data[:] = 0.0
        self.data.sigma_lesser._data[:] = 0.0
        self.data.sigma_greater._data[:] = 0.0

    def _update_sigma(self) -> None:
        """Updates the self-energy with a mixing factor."""

        self.data.sigma_lesser.data = (
            (1 - self.mixing_factor) * self.data.sigma_lesser_prev.data
            + self.mixing_factor * self.data.sigma_lesser.data
        )
        self.data.sigma_greater.data = (
            (1 - self.mixing_factor) * self.data.sigma_greater_prev.data
            + self.mixing_factor * self.data.sigma_greater.data
        )
        self.data.sigma_retarded.data = (
            (1 - self.mixing_factor) * self.data.sigma_retarded_prev.data
            + self.mixing_factor * self.data.sigma_retarded.data
        )

        # Symmetrization.
        self.data.sigma_lesser.data = 0.5 * (
            self.data.sigma_lesser.data
            - self.data.sigma_lesser.ltranspose(copy=True).data.conj()
        )
        self.data.sigma_greater.data = 0.5 * (
            self.data.sigma_greater.data
            - self.data.sigma_greater.ltranspose(copy=True).data.conj()
        )
        self.data.sigma_lesser._data.real = 0
        self.data.sigma_greater._data.real = 0

        self.data.sigma_retarded._data.imag = 0.0

        # Make the remaining real part Hermitian.
        self.data.sigma_retarded.data = 0.5 * (
            self.data.sigma_retarded.data
            + self.data.sigma_retarded.ltranspose(copy=True).data.conj()
        )
        # Now add the imaginary, skew-Hermitian part back.
        self.data.sigma_retarded.data += 0.5 * (
            self.data.sigma_greater.data - self.data.sigma_lesser.data
        )

    def _has_converged(self) -> bool:
        """Checks if the SCBA has converged."""
        # Infinity norm of the self-energy update.
        diff = self.data.sigma_retarded.data - self.data.sigma_retarded_prev.data
        max_diff = xp.max(xp.abs(diff))
        max_diff = comm.allreduce(max_diff, op=MPI.MAX)

        i_left, i_right = contact_currents(
            self.data.g_lesser,
            self.data.g_greater,
            self.electron_solver.obc_blocks,
        )
        change_left = xp.linalg.norm(
            i_left.real - self.observables.electron_current.get("left", 0.0)
        )
        change_right = xp.linalg.norm(
            i_right.real - self.observables.electron_current.get("right", 0.0)
        )
        ave_change = 0.5 * (change_left + change_right)

        if comm.rank == 0:
            print(f"Maximum Self-Energy Update: {max_diff}", flush=True)
            print(f"Average Current Change: {ave_change}", flush=True)
            print(f"Current Difference: {diff}", flush=True)

        # if ave_change < self.quatrex_config.scba.convergence_tol:
        #     return True

        return False  # TODO: :-)

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
        times = []
        times.append(time.perf_counter())
        self.p_coulomb_screening.compute(
            self.data.g_lesser,
            self.data.g_greater,
            out=(self.data.p_lesser, self.data.p_greater, self.data.p_retarded),
        )
        t_polarization = time.perf_counter() - times.pop()
        if comm.rank == 0:
            print(f"Time for polarization: {t_polarization:.2f} s", flush=True)

        times.append(time.perf_counter())
        self.coulomb_screening_solver.solve(
            self.data.p_lesser,
            self.data.p_greater,
            self.data.p_retarded,
            out=(self.data.w_lesser, self.data.w_greater),
        )
        t_solve = time.perf_counter() - times.pop()
        if comm.rank == 0:
            print(f"Time for Coulomb screening solver: {t_solve:.2f} s", flush=True)

        self._compute_coulomb_screening_observables()

        times.append(time.perf_counter())
        self.sigma_fock.compute(
            self.data.g_lesser,
            out=(self.data.sigma_retarded,),
        )
        t_fock = time.perf_counter() - times.pop()
        if comm.rank == 0:
            print(f"Time for Fock self-energy: {t_fock:.2f} s", flush=True)

        times.append(time.perf_counter())
        self.sigma_coulomb_screening.compute(
            self.data.g_lesser,
            self.data.g_greater,
            self.data.w_lesser,
            self.data.w_greater,
            out=(
                self.data.sigma_lesser,
                self.data.sigma_greater,
                self.data.sigma_retarded,
            ),
        )
        t_sigma = time.perf_counter() - times.pop()
        if comm.rank == 0:
            print(
                f"Time for Coulomb screening self-energy: {t_sigma:.2f} s", flush=True
            )

    def _compute_electron_observables(self) -> None:
        """Computes electron observables."""
        self.observables.electron_ldos = -density(
            self.data.g_retarded,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)
        self.observables.electron_density = density(
            self.data.g_lesser,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)
        self.observables.hole_density = -density(
            self.data.g_greater,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)

        self.observables.electron_current = dict(
            zip(
                ("left", "right"),
                contact_currents(
                    self.data.g_lesser,
                    self.data.g_greater,
                    self.electron_solver.obc_blocks,
                ),
            )
        )
        self.observables.electron_current["device"] = device_current(
            self.data.g_lesser, self.electron_solver.hamiltonian_sparray
        )
        if self.quatrex_config.electron.solver.compute_current:
            self.observables.electron_current["meir-wingreen"] = xp.vstack(
                comm.allgather(self.electron_solver.meir_wingreen_current)
            )

        self.observables.sigma_retarded_density = -density(
            self.data.sigma_retarded,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)
        self.observables.sigma_lesser_density = density(
            self.data.sigma_lesser,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)
        self.observables.sigma_greater_density = -density(
            self.data.sigma_greater,
            self.electron_solver.overlap_sparray,
        ) / (2 * xp.pi)

    def _compute_coulomb_screening_observables(self) -> None:
        self.observables.p_retarded_density = -density(self.data.p_retarded) / (
            2 * xp.pi
        )
        self.observables.p_lesser_density = density(self.data.p_lesser) / (2 * xp.pi)
        self.observables.p_greater_density = -density(self.data.p_greater) / (2 * xp.pi)

        self.observables.w_lesser_density = density(self.data.w_lesser) / (2 * xp.pi)
        self.observables.w_greater_density = -density(self.data.w_greater) / (2 * xp.pi)

    def _write_iteration_outputs(self, iteration: int):
        """Writes output for the current iteration on rank zero."""

        if comm.rank != 0:
            return

        print(f"Writing output for iteration {iteration}...", flush=True)

        output_dir = self.quatrex_config.simulation_dir / "outputs"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        xp.save(
            f"{output_dir}/electron_ldos_{iteration}.npy",
            self.observables.electron_ldos,
        )
        xp.save(
            f"{output_dir}/electron_density_{iteration}.npy",
            self.observables.electron_density,
        )
        xp.save(
            f"{output_dir}/hole_density_{iteration}.npy", self.observables.hole_density
        )
        xp.save(
            f"{output_dir}/i_left_{iteration}.npy",
            self.observables.electron_current["left"],
        )
        xp.save(
            f"{output_dir}/i_right_{iteration}.npy",
            self.observables.electron_current["right"],
        )

        xp.save(
            f"{output_dir}/device_current_{iteration}.npy",
            self.observables.electron_current["device"],
        )

        if self.quatrex_config.electron.solver.compute_current:
            xp.save(
                f"{output_dir}/meir_wingreen_current_{iteration}.npy",
                self.observables.electron_current["meir-wingreen"],
            )

        if self.quatrex_config.scba.coulomb_screening:
            xp.save(
                f"{output_dir}/p_lesser_density_{iteration}.npy",
                self.observables.p_lesser_density,
            )
            xp.save(
                f"{output_dir}/p_greater_density_{iteration}.npy",
                self.observables.p_greater_density,
            )
            xp.save(
                f"{output_dir}/p_retarded_density_{iteration}.npy",
                self.observables.p_retarded_density,
            )

            xp.save(
                f"{output_dir}/w_lesser_density_{iteration}.npy",
                self.observables.w_lesser_density,
            )
            xp.save(
                f"{output_dir}/w_greater_density_{iteration}.npy",
                self.observables.w_greater_density,
            )

        xp.save(
            f"{output_dir}/sigma_retarded_density_{iteration}.npy",
            self.observables.sigma_retarded_density,
        )
        xp.save(
            f"{output_dir}/sigma_lesser_density_{iteration}.npy",
            self.observables.sigma_lesser_density,
        )
        xp.save(
            f"{output_dir}/sigma_greater_density_{iteration}.npy",
            self.observables.sigma_greater_density,
        )

    def run(self) -> None:
        """Runs the SCBA to convergence."""
        print("Entering SCBA loop...", flush=True) if comm.rank == 0 else None
        times = []
        for i in range(self.quatrex_config.scba.max_iterations):
            print(f"Iteration {i}", flush=True) if comm.rank == 0 else None
            # append for iteration time
            times.append(time.perf_counter())

            times.append(time.perf_counter())
            self.electron_solver.solve(
                self.data.sigma_lesser,
                self.data.sigma_greater,
                self.data.sigma_retarded,
                out=(self.data.g_lesser, self.data.g_greater, self.data.g_retarded),
            )
            t_solve = time.perf_counter() - times.pop()
            if comm.rank == 0:
                print(f"Time for electron solver: {t_solve:.2f} s", flush=True)

            self._compute_electron_observables()

            # Stash current into previous self-energy buffer.
            times.append(time.perf_counter())
            self._stash_sigma()
            t_swap = time.perf_counter() - times.pop()
            if comm.rank == 0:
                print(f"Time for swapping: {t_swap:.2f} s", flush=True)

            # Transpose to nnz distribution.
            # NOTE: While computing all interactions, we only ever need
            # to access the Green's function and the self-energies in
            # their nnz-distributed state.
            t0 = time.perf_counter()
            for m in (self.data.g_lesser, self.data.g_greater):
                m.dtranspose(discard=False)  # This must not be discarded.
                assert m.distribution_state == "nnz"
            for m in (
                self.data.sigma_lesser,
                self.data.sigma_greater,
                self.data.sigma_retarded,
            ):
                m.dtranspose(discard=True)  # These can be safely discarded.
                assert m.distribution_state == "nnz"
            t1 = time.perf_counter()
            if comm.rank == 0:
                print(f"scba: Time for transposing forth: {t1 - t0:.2f} s", flush=True)

            if self.quatrex_config.scba.coulomb_screening:
                times.append(time.perf_counter())
                self._compute_coulomb_screening_interaction()
                t_coulomb = time.perf_counter() - times.pop()
                if comm.rank == 0:
                    print(
                        f"Time for Coulomb screening interaction: {t_coulomb:.2f} s",
                        flush=True,
                    )

            if self.quatrex_config.scba.photon:
                self._compute_photon_interaction()

            if self.quatrex_config.scba.phonon:
                times.append(time.perf_counter())
                self._compute_phonon_interaction()
                t_phonon = time.perf_counter() - times.pop()
                (
                    print(f"Time for phonon interaction: {t_phonon:.2f} s", flush=True)
                    if comm.rank == 0
                    else None
                )

            # Transpose back to stack distribution.
            t0 = time.perf_counter()
            for m in (self.data.g_lesser, self.data.g_greater):
                m.dtranspose(discard=True)  # These can be safely discarded.
                assert m.distribution_state == "stack"
            for m in (
                self.data.sigma_lesser,
                self.data.sigma_greater,
                self.data.sigma_retarded,
            ):
                m.dtranspose(discard=False)  # This must not be discarded.
                assert m.distribution_state == "stack"
            t1 = time.perf_counter()
            if comm.rank == 0:
                print(f"scba: Time for transposing back: {t1 - t0:.2f} s", flush=True)

            times.append(time.perf_counter())
            if self._has_converged():
                if comm.rank == 0:
                    print(f"SCBA converged after {i} iterations.", flush=True)

                break

            t_convergence = time.perf_counter() - times.pop()
            if comm.rank == 0:
                print(f"Time for convergence check: {t_convergence:.2f} s", flush=True)

            times.append(time.perf_counter())

            # Update self-energy for next iteration with mixing factor.
            self._update_sigma()

            t_update = time.perf_counter() - times.pop()
            if comm.rank == 0:
                print(f"Time for updating: {t_update:.2f} s", flush=True)

            t_iteration = time.perf_counter() - times.pop()
            if comm.rank == 0:
                print(f"Time for iteration: {t_iteration:.2f} s", flush=True)

            if i % self.quatrex_config.scba.output_interval == 0:
                self._write_iteration_outputs(i)

        else:  # Did not break, i.e. max_iterations reached.
            if comm.rank == 0:
                print(f"SCBA did not converge after {i} iterations.")
