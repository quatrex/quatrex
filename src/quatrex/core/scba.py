# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import os
import time
from dataclasses import dataclass, field

from cupyx.profiler import time_range
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from qttools import NCCL_AVAILABLE, NDArray, nccl_comm, xp
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_host, synchronize_device
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

profiler = Profiler()


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
        block_sizes = get_host(
            distributed_load(quatrex_config.input_dir / "block_sizes.npy")
        )

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
                self.compute_config,
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

    @profiler.profile(level="api")
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
        synchronize_device()
        time_start = time.perf_counter()

        self.data.sigma_lesser.symmetrize(xp.subtract)
        self.data.sigma_greater.symmetrize(xp.subtract)

        self.data.sigma_lesser._data.real = 0
        self.data.sigma_greater._data.real = 0

        self.data.sigma_retarded._data.imag = 0.0

        # Make the remaining real part Hermitian.
        self.data.sigma_retarded.symmetrize(xp.add)

        # Now add the imaginary, skew-Hermitian part back.
        self.data.sigma_retarded.data += 0.5 * (
            self.data.sigma_greater.data - self.data.sigma_lesser.data
        )
        synchronize_device()
        time_end = time.perf_counter()
        if comm.rank == 0:
            print(f"Symmetrization time: {time_end-time_start}", flush=True)

    @profiler.profile(level="api")
    def _has_converged(self) -> bool:
        """Checks if the SCBA has converged."""
        # Infinity norm of the self-energy update.
        diff = self.data.sigma_retarded.data - self.data.sigma_retarded_prev.data
        local_max_diff = xp.max(xp.abs(diff))
        if not NCCL_AVAILABLE:
            max_diff = comm.allreduce(local_max_diff, op=MPI.MAX)
        else:
            max_diff = xp.empty_like(local_max_diff)
            synchronize_device()
            nccl_comm.all_reduce(local_max_diff, max_diff, op="max")
            synchronize_device()

        i_left = xp.real(self.observables.electron_current.get("left", 0.0))
        i_right = xp.real(self.observables.electron_current.get("right", 0.0))

        dE = self.electron_energies[1] - self.electron_energies[0]
        current_diff = xp.abs(xp.sum(i_left) * dE - xp.sum(i_right) * dE)

        if comm.rank == 0:
            print(f"Maximum Self-Energy Update: {max_diff}", flush=True)
            print(f"Contact Current Difference: {current_diff}", flush=True)

        # if ave_change < self.quatrex_config.scba.convergence_tol:
        #     return True

        return False  # TODO: :-)

    @profiler.profile(level="api")
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

    @profiler.profile(level="api")
    def _compute_photon_interaction(self):
        """Computes the photon interaction."""
        raise NotImplementedError

    @profiler.profile(level="api")
    def _compute_coulomb_screening_interaction(self):
        """Computes the Coulomb screening interaction."""

        t_polarization_start = time.perf_counter()
        self.p_coulomb_screening.compute(
            self.data.g_lesser,
            self.data.g_greater,
            out=(self.data.p_lesser, self.data.p_greater, self.data.p_retarded),
        )
        synchronize_device()
        t_polarization_end = time.perf_counter()
        comm.Barrier()
        t_polarization_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"  Time for polarization: {t_polarization_end - t_polarization_start:.3f} s",
                flush=True,
            )
            print(
                f"  Time for polarization all: {t_polarization_end_all - t_polarization_start:.3f} s",
                flush=True,
            )

        t_coulomb_start = time.perf_counter()
        self.coulomb_screening_solver.solve(
            self.data.p_lesser,
            self.data.p_greater,
            self.data.p_retarded,
            out=(self.data.w_lesser, self.data.w_greater),
        )
        synchronize_device()
        t_coulomb_end = time.perf_counter()
        comm.Barrier()
        t_coulomb_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"  Time for Coulomb screening: {t_coulomb_end - t_coulomb_start:.3f} s",
                flush=True,
            )
            print(
                f"  Time for Coulomb screening all: {t_coulomb_end_all - t_coulomb_start:.3f} s",
                flush=True,
            )

        t_coulomb_observables = time.perf_counter()
        self._compute_coulomb_screening_observables()
        synchronize_device()
        t_coulomb_observables_end = time.perf_counter()
        comm.Barrier()
        t_coulomb_observables_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"  Time for Coulomb screening observables: {t_coulomb_observables_end - t_coulomb_observables:.3f} s",
                flush=True,
            )
            print(
                f"  Time for Coulomb screening observables all: {t_coulomb_observables_end_all - t_coulomb_observables:.3f} s",
                flush=True,
            )

        t_sigma_fock_start = time.perf_counter()
        self.sigma_fock.compute(
            self.data.g_lesser,
            out=(self.data.sigma_retarded,),
        )
        synchronize_device()
        t_sigma_fock_end = time.perf_counter()
        comm.Barrier()
        t_sigma_fock_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"  Time for Fock self-energy: {t_sigma_fock_end - t_sigma_fock_start:.3f} s",
                flush=True,
            )
            print(
                f"  Time for Fock self-energy all: {t_sigma_fock_end_all - t_sigma_fock_start:.3f} s",
                flush=True,
            )

        t_sigma_start = time.perf_counter()
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
        synchronize_device()
        t_sigma_end = time.perf_counter()
        comm.Barrier()
        t_sigma_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"  Time for Coulomb screening self-energy: {t_sigma_end - t_sigma_start:.3f} s",
                flush=True,
            )
            print(
                f"  Time for Coulomb screening self-energy all: {t_sigma_end_all - t_sigma_start:.3f} s",
                flush=True,
            )

    @profiler.profile(level="debug")
    def _compute_electron_observables(self) -> None:
        """Computes electron observables."""
        if self.quatrex_config.outputs.electron_ldos:
            self.observables.electron_ldos = -density(
                self.data.g_retarded,
                self.electron_solver.overlap_sparray,
            ) / (2 * xp.pi)
        if self.quatrex_config.outputs.electron_density:
            self.observables.electron_density = density(
                self.data.g_lesser,
                self.electron_solver.overlap_sparray,
            ) / (2 * xp.pi)
        if self.quatrex_config.outputs.hole_density:
            self.observables.hole_density = -density(
                self.data.g_greater,
                self.electron_solver.overlap_sparray,
            ) / (2 * xp.pi)

        if self.quatrex_config.outputs.contact_currents:
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
        if self.quatrex_config.outputs.device_currents:
            self.observables.electron_current["device"] = device_current(
                self.data.g_lesser, self.electron_solver.hamiltonian_sparray
            )
            if self.quatrex_config.electron.solver.compute_current:
                if not NCCL_AVAILABLE:
                    meir_wingreen_current = xp.vstack(
                        comm.allgather(self.electron_solver.meir_wingreen_current)
                    )
                else:
                    # NOTE: NCCL does not expose all_gather_v. This is a hack.
                    local_current = self.electron_solver.meir_wingreen_current
                    pad_width = (
                        self.data.g_lesser.total_stack_size // comm.size
                        - local_current.shape[0]
                    )
                    local_current = xp.pad(local_current, ((0, pad_width), (0, 0)))
                    meir_wingreen_current = xp.empty(
                        (self.data.g_lesser.total_stack_size, local_current.shape[-1]),
                        dtype=local_current.dtype,
                    )
                    synchronize_device()
                    nccl_comm.all_gather(
                        local_current, meir_wingreen_current, local_current.size
                    )
                    synchronize_device()
                    meir_wingreen_current = meir_wingreen_current[
                        self.data.g_lesser._stack_padding_mask, ...
                    ]

                self.observables.electron_current["meir-wingreen"] = (
                    meir_wingreen_current
                )

        if self.quatrex_config.outputs.self_energy_density:
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

    @profiler.profile(level="debug")
    def _compute_coulomb_screening_observables(self) -> None:

        if self.quatrex_config.outputs.polarization_density:
            self.observables.p_retarded_density = -density(self.data.p_retarded) / (
                2 * xp.pi
            )
            self.observables.p_lesser_density = density(self.data.p_lesser) / (
                2 * xp.pi
            )
            self.observables.p_greater_density = -density(self.data.p_greater) / (
                2 * xp.pi
            )

        if self.quatrex_config.outputs.coulomb_screening_density:
            self.observables.w_lesser_density = density(self.data.w_lesser) / (
                2 * xp.pi
            )
            self.observables.w_greater_density = -density(self.data.w_greater) / (
                2 * xp.pi
            )

    @profiler.profile(level="debug")
    def _write_iteration_outputs(self, iteration: int):
        """Writes output for the current iteration on rank zero."""

        if self.quatrex_config.outputs.profiling_stats:
            profiler.dump_stats(self.quatrex_config.output_dir / "profiling_stats")

        if comm.rank != 0:
            return

        outputs = {}

        if self.quatrex_config.outputs.electron_ldos:
            outputs[f"electron_ldos_{iteration}.npy"] = self.observables.electron_ldos
        if self.quatrex_config.outputs.electron_density:
            outputs[f"electron_density_{iteration}.npy"] = (
                self.observables.electron_density
            )
        if self.quatrex_config.outputs.hole_density:
            outputs[f"hole_density_{iteration}.npy"] = self.observables.hole_density

        if self.quatrex_config.outputs.contact_currents:
            outputs.update(
                {
                    f"i_{contact}_{iteration}.npy": current
                    for contact, current in self.observables.electron_current.items()
                }
            )
        if self.quatrex_config.outputs.device_currents:
            outputs[f"device_current_{iteration}.npy"] = (
                self.observables.electron_current["device"]
            )
            if self.quatrex_config.electron.solver.compute_current:
                outputs[f"meir_wingreen_current_{iteration}.npy"] = (
                    self.observables.electron_current["meir-wingreen"]
                )

        if self.quatrex_config.scba.coulomb_screening:
            if self.quatrex_config.outputs.polarization_density:
                outputs.update(
                    {
                        f"p_lesser_density_{iteration}.npy": self.observables.p_lesser_density,
                        f"p_greater_density_{iteration}.npy": self.observables.p_greater_density,
                        f"p_retarded_density_{iteration}.npy": self.observables.p_retarded_density,
                    }
                )
            if self.quatrex_config.outputs.coulomb_screening_density:
                outputs.update(
                    {
                        f"w_lesser_density_{iteration}.npy": self.observables.w_lesser_density,
                        f"w_greater_density_{iteration}.npy": self.observables.w_greater_density,
                    }
                )

        if self.quatrex_config.outputs.self_energy_density:
            outputs.update(
                {
                    f"sigma_retarded_density_{iteration}.npy": self.observables.sigma_retarded_density,
                    f"sigma_lesser_density_{iteration}.npy": self.observables.sigma_lesser_density,
                    f"sigma_greater_density_{iteration}.npy": self.observables.sigma_greater_density,
                }
            )

        print(f"Writing output for iteration {iteration}...", flush=True)

        if not os.path.exists(self.quatrex_config.output_dir):
            os.mkdir(self.quatrex_config.output_dir)

        for filename, data in outputs.items():
            xp.save(self.quatrex_config.output_dir / filename, data)

    @profiler.profile(level="basic")
    def run(self) -> None:
        """Runs the SCBA to convergence."""
        print("Entering SCBA loop...", flush=True) if comm.rank == 0 else None

        for i in range(self.quatrex_config.scba.max_iterations):
            print(f"Iteration {i}", flush=True) if comm.rank == 0 else None
            # append for iteration time
            synchronize_device()
            comm.Barrier()
            t_iteration_start = time.perf_counter()

            t_solve_start = time.perf_counter()
            self.electron_solver.solve(
                self.data.sigma_lesser,
                self.data.sigma_greater,
                self.data.sigma_retarded,
                out=(self.data.g_lesser, self.data.g_greater, self.data.g_retarded),
            )
            synchronize_device()
            t_solve_end = time.perf_counter()
            comm.Barrier()
            t_solve_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"Time for electron solver: {t_solve_end - t_solve_start:.3f} s",
                    flush=True,
                )
                print(
                    f"Time for electron solver all: {t_solve_end_all - t_solve_start:.3f} s",
                    flush=True,
                )

            t_oberservables_start = time.perf_counter()
            self._compute_electron_observables()
            synchronize_device()
            t_oberservables_end = time.perf_counter()
            comm.Barrier()

            t_oberservables_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"Time for computing observables: {t_oberservables_end - t_oberservables_start:.3f} s",
                    flush=True,
                )
                print(
                    f"Time for computing observables all: {t_oberservables_end_all - t_oberservables_start:.3f} s",
                    flush=True,
                )

            # Stash current into previous self-energy buffer.
            t_stash_start = time.perf_counter()
            self._stash_sigma()
            synchronize_device()
            t_stash_end = time.perf_counter()
            comm.Barrier()
            t_stash_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"Time for swapping: {t_stash_end - t_stash_start:.3f} s",
                    flush=True,
                )
                print(
                    f"Time for swapping all: {t_stash_end_all - t_stash_start:.3f} s",
                    flush=True,
                )

            # Transpose to nnz distribution.
            # NOTE: While computing all interactions, we only ever need
            # to access the Green's function and the self-energies in
            # their nnz-distributed state.
            t_start_transpose = time.perf_counter()
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
            synchronize_device()
            t_end_transpose = time.perf_counter()
            comm.Barrier()
            t_end_transpose_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"scba: Time for transposing forth: {t_end_transpose - t_start_transpose:.3f} s",
                    flush=True,
                )
                print(
                    f"scba: Time for transposing forth all: {t_end_transpose_all - t_start_transpose:.3f} s",
                    flush=True,
                )

            if self.quatrex_config.scba.coulomb_screening:
                t_start_coulomb = time.perf_counter()
                self._compute_coulomb_screening_interaction()
                synchronize_device()
                t_end_coulomb = time.perf_counter()
                comm.Barrier()
                t_end_coulomb_all = time.perf_counter()
                if comm.rank == 0:
                    print(
                        f"Time for Coulomb screening interaction: {t_end_coulomb - t_start_coulomb:.3f} s",
                        flush=True,
                    )
                    print(
                        f"Time for Coulomb screening interaction all: {t_end_coulomb_all - t_start_coulomb:.3f} s",
                        flush=True,
                    )

            if self.quatrex_config.scba.photon:
                self._compute_photon_interaction()

            if self.quatrex_config.scba.phonon:
                t_start_phonon = time.perf_counter()
                self._compute_phonon_interaction()
                synchronize_device()
                t_end_phonon = time.perf_counter()
                comm.Barrier()
                t_end_phonon_all = time.perf_counter()
                if comm.rank == 0:
                    print(
                        f"Time for phonon interaction: {t_end_phonon - t_start_phonon:.3f} s",
                        flush=True,
                    )
                    print(
                        f"Time for phonon interaction all: {t_end_phonon_all - t_start_phonon:.3f} s",
                        flush=True,
                    )

            # Transpose back to stack distribution.
            t_transpose_sigma_start = time.perf_counter()
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
            synchronize_device()
            t_transpose_sigma_end = time.perf_counter()
            comm.Barrier()
            t_transpose_sigma_end_all = time.perf_counter()

            if comm.rank == 0:
                print(
                    f"scba: Time for transposing back: {t_transpose_sigma_end - t_transpose_sigma_start:.3f} s",
                    flush=True,
                )
                print(
                    f"scba: Time for transposing back all: {t_transpose_sigma_end_all - t_transpose_sigma_start:.3f} s",
                    flush=True,
                )

            t_convergence_start = time.perf_counter()
            if self._has_converged():
                if comm.rank == 0:
                    print(f"SCBA converged after {i} iterations.", flush=True)

                break
            synchronize_device()
            t_convergence_end = time.perf_counter()
            comm.Barrier()
            t_convergence_end_all = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"Time for convergence check: {t_convergence_end - t_convergence_start:.3f} s",
                    flush=True,
                )
                print(
                    f"Time for convergence check all: {t_convergence_end_all - t_convergence_start:.3f} s",
                    flush=True,
                )

            t_sigma_update_start = time.perf_counter()
            # Update self-energy for next iteration with mixing factor.
            self._update_sigma()
            synchronize_device()
            t_sigma_update_end = time.perf_counter()
            comm.Barrier()
            t_sigma_update_end_all = time.perf_counter()

            if comm.rank == 0:
                print(
                    f"Time for updating: {t_sigma_update_end - t_sigma_update_start:.3f} s",
                    flush=True,
                )
                print(
                    f"Time for updating all: {t_sigma_update_end_all - t_sigma_update_start:.3f} s",
                    flush=True,
                )

            t_iteration = time.perf_counter() - t_iteration_start
            if comm.rank == 0:
                print(f"Time for iteration all: {t_iteration:.3f} s", flush=True)

            free_memory, total_memory = xp.cuda.Device().mem_info
            usage = (total_memory - free_memory) / total_memory
            if not NCCL_AVAILABLE:
                average_usage = comm.allreduce(usage, op=MPI.SUM) / comm.size
            else:
                average_usage = xp.empty(1)
                synchronize_device()
                nccl_comm.all_reduce(xp.array(usage), average_usage, op="sum")
                synchronize_device()
                average_usage = float(average_usage[0]) / comm.size
            if comm.rank == 0:
                print(
                    f"Rank-average device memory usage: {average_usage * 100:.4f}%",
                    flush=True,
                )

            if i % self.quatrex_config.scba.output_interval == 0:
                synchronize_device()
                comm.Barrier()
                t_write_start = time.perf_counter()
                self._write_iteration_outputs(i)
                synchronize_device()
                t_write_end = time.perf_counter()
                comm.Barrier()
                t_write_end_all = time.perf_counter()
                if comm.rank == 0:
                    print(
                        f"Time for writing outputs: {t_write_end_all - t_write_start:.3f} s",
                        flush=True,
                    )
                    print(
                        f"Time for writing outputs all: {t_write_end - t_write_start:.3f} s",
                        flush=True,
                    )

        else:  # Did not break, i.e. max_iterations reached.
            if comm.rank == 0:
                print(f"SCBA did not converge after {i} iterations.")
