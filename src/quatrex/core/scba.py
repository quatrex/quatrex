# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import os
from dataclasses import dataclass, field
from time import sleep

import numpy as np
from scipy.interpolate import make_interp_spline    
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_host
from qttools.utils.mpi_utils import distributed_load, get_section_sizes, gather_array_stack, get_section_sizes, reduce_matrix_over_stack
from quatrex.core.config import QuatrexConfig
from quatrex.core.observables import (
    contact_currents,
    current_conservation,
    density,
    device_current,
)
from quatrex.core.utils import compute_num_connected_blocks, compute_sparsity_pattern
from quatrex.coulomb_screening import CoulombScreeningSolver, PCoulombScreening
from quatrex.device.inputs import (
    create_coordinate_grid,
    distributed_read_xyz,
    load_matrix,
)
from quatrex.electron import (
    ElectronSolver,
    SigmaCoulombScreening,
    SigmaFock,
    SigmaPhonon,
    SigmaPhoton,
)
from quatrex.grid import get_electron_energies
from quatrex.phonon import PhononSolver, PiPhonon
from quatrex.photon import PhotonSolver, PiPhoton

profiler = Profiler()

class SCBAData:
    """Data container class for the SCBA.

    Parameters
    ----------
    config : QuatrexConfig
        The Quatrex configuration.

    """

    def __init__(self, config: QuatrexConfig, electron_energies: NDArray) -> None:
        """Initializes the SCBA data."""
        # Load orbital positions, energy vector and block-sizes.

        structure_file = config.input_dir / "structure.xyz"
        if not structure_file.exists():
            raise FileNotFoundError(f"Structure file {structure_file} not found.")
        lattice_vectors, atom_coordinates, atomic_species = distributed_read_xyz(
            structure_file
        )

        orbitals_per_atom = [
            config.device.num_orbitals_per_atom.get(s, 1) for s in atomic_species
        ]
        atom_coordinates = xp.asarray(atom_coordinates)
        grid = xp.repeat(atom_coordinates, orbitals_per_atom, axis=0)

        if config.device.construct_from_unit_cell:
            # The neighbor cell cutoff along the transport direction
            # determines the size of the transport cell.
            transport_ind = "xyz".index(config.device.transport_direction)
            unit_cells_per_transport_cell = [1, 1, 1]
            unit_cells_per_transport_cell[transport_ind] = (
                config.device.neighbor_cell_cutoff[transport_ind]
            )
            device_cell = unit_cells_per_transport_cell.copy()
            device_cell[transport_ind] *= config.device.num_transport_cells

            block_sizes = np.array(
                [unit_cells_per_transport_cell[transport_ind] * grid.shape[0]]
                * config.device.num_transport_cells
            )

            grid = create_coordinate_grid(
                grid, tuple(device_cell), xp.asarray(lattice_vectors)
            )

        else:
            block_sizes = config.device.block_size
            if isinstance(block_sizes, int):
                num_blocks, remainder = divmod(grid.shape[0], block_sizes)
                if remainder != 0:
                    raise ValueError(
                        f"Block size {block_sizes} does not evenly divide the number of orbitals {grid.shape[0]}."
                    )
                block_sizes = [block_sizes] * num_blocks

            block_sizes = np.array(block_sizes)

            if block_sizes.sum() != grid.shape[0]:
                raise ValueError(
                    f"Sum of block sizes {block_sizes.sum()} does not match the number of orbitals {grid.shape[0]}."
                )

        kpoint_grid = config.device.kpoint_grid
        # Find the maximum interaction cutoff.
        max_interaction_cutoff = 0.0
        if config.scba.coulomb_screening:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                config.coulomb_screening.interaction_cutoff,
            )
        if config.scba.photon:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                config.photon.interaction_cutoff,
            )
        if config.scba.phonon:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                config.phonon.interaction_cutoff,
            )
        if max_interaction_cutoff == 0.0:
            raise NotImplementedError(
                "At least one interaction must be enabled in the SCBA."
                "Ballistic transport is not properly supported yet."
            )

        if comm.rank == 0:
            print(f"Max Interaction Cutoff: {max_interaction_cutoff}", flush=True)

        with profiler.profile_range(
            label="SCBA: Sparsity Pattern", level="default", comm=comm
        ):
            # Determine the local slice of the data.
            # NOTE: This is arrow-wise partitioning.
            # TODO: Allow more options, e.g., block row-wise partitioning.
            section_sizes, __ = get_section_sizes(len(block_sizes), comm.block.size)
            section_offsets = np.hstack(([0], np.cumsum(section_sizes)))
            block_offsets = np.hstack(([0], np.cumsum(block_sizes)))
            start_idx = block_offsets[section_offsets[comm.block.rank]]
            end_idx = block_offsets[section_offsets[comm.block.rank + 1]]

            self.sparsity_pattern = compute_sparsity_pattern(
                grid,
                max_interaction_cutoff,
                transport_direction=config.device.transport_direction,
                start_idx=start_idx,
                end_idx=end_idx,
            )

        dsdbsparse_type = config.compute.dsdbsparse_type

        self.g_retarded = dsdbsparse_type.from_sparray(
            self.sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=electron_energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
        )
        self.g_retarded.data[:] = 0.0  # Initialize to zero.

        self.g_lesser = dsdbsparse_type.from_sparray(
            self.sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=electron_energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
            symmetry=config.scba.symmetric,
            symmetry_op=lambda a: -a.conj(),
        )
        self.g_greater = dsdbsparse_type.zeros_like(self.g_lesser)

        self.sigma_lesser_prev = dsdbsparse_type.zeros_like(self.g_lesser)
        self.sigma_lesser = dsdbsparse_type.zeros_like(self.g_lesser)
        self.sigma_greater_prev = dsdbsparse_type.zeros_like(self.g_lesser)
        self.sigma_greater = dsdbsparse_type.zeros_like(self.g_lesser)

        self.sigma_retarded_prev = dsdbsparse_type.zeros_like(self.g_lesser)
        self.sigma_retarded = dsdbsparse_type.zeros_like(self.g_lesser)
        if config.scba.symmetric:
            self.sigma_retarded.symmetry_op = lambda a: a
            self.sigma_retarded_prev.symmetry_op = lambda a: a

        if config.scba.coulomb_screening:
            # NOTE: The polarization has the same sparsity pattern as
            # the electronic system (the interactions are local in real
            # space). However, we need to change the block sizes of the
            # screened Coulomb interaction.
            self.p_retarded = dsdbsparse_type.zeros_like(self.g_lesser)
            self.p_lesser = dsdbsparse_type.zeros_like(self.g_lesser)
            self.p_greater = dsdbsparse_type.zeros_like(self.g_lesser)

            num_connected_blocks = config.coulomb_screening.num_connected_blocks
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

            self.w_lesser = dsdbsparse_type.from_sparray(
                self.sparsity_pattern.astype(xp.complex128),
                block_sizes=coulomb_screening_block_sizes,
                global_stack_shape=electron_energies.shape
                + tuple([k for k in kpoint_grid if k > 1]),
                symmetry=config.scba.symmetric,
                symmetry_op=lambda a: -a.conj(),
            )
            self.w_greater = dsdbsparse_type.zeros_like(self.w_lesser)

        # TODO: The interactions with photons and phonons are not yet
        # implemented.
        if config.scba.photon:
            raise NotImplementedError

        if config.scba.phonon and config.phonon.model == "negf":
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
    config : QuatrexConfig
        Quatrex configuration object.

    """

    def __init__(self, config: QuatrexConfig) -> None:
        """Initializes an SCBA instance."""
        self.config = config

        self.observables = Observables()
        electron_energies = xp.zeros((comm.size,))
        self.data = SCBAData(config, electron_energies=electron_energies)  # dummy data
        self.mixing_factor = self.config.scba.mixing_factor

        # ----- Electrons ----------------------------------------------
        self.electron_energies = get_electron_energies(config)

        min_energy = self.electron_energies[0]
        max_energy = self.electron_energies[-1]
        num_energies = len(self.electron_energies)
        energy_resolution = self.electron_energies[1] - self.electron_energies[0]
        num_energies_per_rank = num_energies // comm.stack.size
        if comm.rank == 0:
            print(
                f"Energy window: {min_energy} to {max_energy} eV with {num_energies} grid points.",
                flush=True,
            )
            print(f"Resolution is {energy_resolution} eV.", flush=True)
            print(
                f"Each comm.block has {num_energies_per_rank} grid points.", flush=True
            )
        
        # initial adaptive energy grid is just the linear grid
        if self.config.scba.adaptive:
            self.adaptive_electron_energies_for_g_sigma = xp.copy(self.electron_energies)
            self.adaptive_electron_energies_for_p_w = xp.copy(self.electron_energies)
        
        self.electron_solver = ElectronSolver(
            self.config,
            self.electron_energies,
            sparsity_pattern=self.data.sparsity_pattern,
        )

        # ----- Coulomb screening --------------------------------------
        if self.config.scba.coulomb_screening:
            # Load the Coulomb matrix.
            coulomb_matrix, __ = load_matrix(
                config=config,
                matrix_name="coulomb_matrix",
                sparsity_pattern=self.data.sparsity_pattern,
                shift_kpoints=True,
            )

            # Make sure the Coulomb matrix is hermitian.
            # TODO: Check that this is correct for kpoints.
            if not coulomb_matrix.symmetry:
                coulomb_matrix.symmetrize()
            coulomb_matrix._data /= config.coulomb_screening.epsilon_r

            energies_path = self.config.input_dir / "coulomb_screening_energies.npy"
            if os.path.isfile(energies_path):
                self.coulomb_screening_energies = distributed_load(energies_path)
            else:
                # liyongda (12 Mar 2026): shift the energies to start from 0
                self.coulomb_screening_energies = (
                    self.electron_energies - self.electron_energies[0]
                )
                # Remove the zero energy to avoid division by zero.
                self.coulomb_screening_energies += 1e-6

            (
                coulomb_matrix.dtranspose()
                if coulomb_matrix.distribution_state != "nnz"
                else None
            )
            self.sigma_fock = SigmaFock(
                self.config,
                coulomb_matrix,
                self.electron_energies,
            )
            # Have to transpose the coulomb matrix back to the original distribution.
            (
                coulomb_matrix.dtranspose()
                if coulomb_matrix.distribution_state == "nnz"
                else None
            )

            # NOTE: No sparsity information required here.
            self.p_coulomb_screening = PCoulombScreening(
                self.config,
                self.coulomb_screening_energies,
            )
            self.coulomb_screening_solver = CoulombScreeningSolver(
                self.config,
                coulomb_matrix,
                self.coulomb_screening_energies,
                sparsity_pattern=self.data.sparsity_pattern,
            )
            self.sigma_coulomb_screening = SigmaCoulombScreening(
                self.config,
                self.electron_energies,
            )

        # ----- Photons ------------------------------------------------
        if self.config.scba.photon:
            energies_path = self.config.input_dir / "photon_energies.npy"
            self.photon_energies = distributed_load(energies_path)
            self.pi_photon = PiPhoton(...)
            self.photon_solver = PhotonSolver(
                self.config,
                self.photon_energies,
                ...,
            )
            self.sigma_photon = SigmaPhoton(...)

        # ----- Phonons ------------------------------------------------
        if self.config.scba.phonon:
            if self.config.phonon.model == "negf":
                energies_path = self.config.input_dir / "phonon_energies.npy"
                self.phonon_energies = distributed_load(energies_path)
                self.pi_phonon = PiPhonon(...)
                self.phonon_solver = PhononSolver(
                    self.config,
                    self.phonon_energies,
                    ...,
                )
                self.sigma_phonon = SigmaPhonon(...)

            elif self.config.phonon.model == "pseudo-scattering":
                self.sigma_phonon = SigmaPhonon(config, self.electron_energies)

        self.data = SCBAData(
            config, electron_energies=self.electron_energies
        )  # real data

    def _stash_sigma(self) -> None:
        """Stash the current into the previous self-energy buffers."""
        self.data.sigma_lesser_prev.data[:] = self.data.sigma_lesser.data
        self.data.sigma_greater_prev.data[:] = self.data.sigma_greater.data
        self.data.sigma_retarded_prev.data[:] = self.data.sigma_retarded.data

        self.data.sigma_retarded.data[:] = 0.0
        self.data.sigma_lesser.data[:] = 0.0
        self.data.sigma_greater.data[:] = 0.0

    @profiler.profile(label="SCBA: Symmetrize Sigma", level="default", comm=comm)
    def _symmetrize_sigma(self) -> None:
        # Symmetrization.
        if not self.config.scba.symmetric:
            self.data.sigma_lesser.symmetrize(xp.subtract)
            self.data.sigma_greater.symmetrize(xp.subtract)
            # Make the self-energy Hermitian (removing the skew-Hermitian part).
            self.data.sigma_retarded.symmetrize(xp.add)

        if self.config.coulomb_screening.discard_real_parts:
            self.data.sigma_lesser._data.real = 0
            self.data.sigma_greater._data.real = 0
            # Make sure that the imaginary part comes only from
            # sigma_greater - sigma_lesser.
            self.data.sigma_retarded._data.imag = 0

        # Now add the imaginary, skew-Hermitian part back.
        self.data.sigma_retarded.data += 0.5 * (
            self.data.sigma_greater.data - self.data.sigma_lesser.data
        )

    @profiler.profile(label="SCBA: Update Sigma", level="default", comm=comm)
    def _update_sigma(self) -> None:
        """Updates the self-energy with a mixing factor."""
        self.data.sigma_lesser.data[:] = (
            (1 - self.mixing_factor) * self.data.sigma_lesser_prev.data
            + self.mixing_factor * self.data.sigma_lesser.data
        )
        self.data.sigma_greater.data[:] = (
            (1 - self.mixing_factor) * self.data.sigma_greater_prev.data
            + self.mixing_factor * self.data.sigma_greater.data
        )
        self.data.sigma_retarded.data[:] = (
            (1 - self.mixing_factor) * self.data.sigma_retarded_prev.data
            + self.mixing_factor * self.data.sigma_retarded.data
        )

    @profiler.profile(label="SCBA: Convergence test", level="default", comm=comm)
    def _has_converged(self) -> bool:
        """Checks if the SCBA has converged."""
        # Infinity norm of the self-energy update.
        diff = self.data.sigma_retarded.data - self.data.sigma_retarded_prev.data
        local_max_diff = get_host(xp.max(xp.abs(diff)))
        max_diff = np.empty_like(local_max_diff)
        global_comm.Allreduce(local_max_diff, max_diff, op=MPI.MAX)

        i_left = xp.real(self.observables.electron_current.get("left", 0.0))
        i_right = xp.real(self.observables.electron_current.get("right", 0.0))

        dE = self.electron_energies[1] - self.electron_energies[0]
        current_diff = xp.abs(xp.sum(i_left) * dE + xp.sum(i_right) * dE)

        current_conservation_abs, current_conservation_rel = current_conservation(
            self.data.g_lesser,
            self.data.g_greater,
            self.data.sigma_lesser,
            self.data.sigma_greater,
        )

        if comm.rank == 0:
            print(f"Maximum Self-Energy Update: {max_diff}", flush=True)
            print(f"Contact Current Difference: {current_diff}", flush=True)
            print(f"Current Conservation abs: {current_conservation_abs}", flush=True)
            print(f"Current Conservation rel: {current_conservation_rel}", flush=True)

        return False  # TODO: :-)
    
    # liyongda (12 Mar 2026): profiler has an internal sync `comm.barrier()`.
    #   but since this function is only called by rank 0, the other ranks never arrive
    #   and we observe an MPI stall
    # @profiler.profile(label="SCBA: Compute adaptive grid", level="default", comm=comm)
    def _compute_adaptive_grid(self, reference_func: NDArray) -> NDArray:
        """Computes an adaptive energy grid based on gradient of sum(abs(VAR))."""
        def _monitor(x, type='gradient'):
            # lose real/imag parts here
            if type == 'gradient':
                return xp.abs(xp.gradient(x))
            elif type == 'curvature':
                return xp.abs(xp.gradient(xp.gradient(x)))
            else:
                raise ValueError("type must be 'gradient' or 'curvature'")
            
        def _adaptive_grid_from_monitor(x, monitor_type='gradient', N_target=1000):
            """Generate adaptive points based on the monitor function
            
            Parameters
            ----------
            x : np.ndarray
                The input data array.
            monitor_type : str, optional
                The type of monitor function to use ('gradient' or 'curvature'), by default 'gradient'.
            N_target : int, optional
                The number of target adaptive points, by default 1000.

            Returns
            -------
            adaptive_points : np.ndarray
                The adaptive grid points.
            monitor : np.ndarray
                The monitor function values.
            cumsum : np.ndarray
                The cumulative distribution function of the monitor.
            """
            monitor = _monitor(x, type=monitor_type)
            cumsum = xp.cumsum(monitor)
            cumsum = cumsum / cumsum[-1]    # normalize to [0,1]

            # equally space the adaptive points in cumulative distribution (of monitor)
            targets = xp.linspace(0, 1, N_target)

            # force use linearly spaced energy grid (self.electron_energies may be non-uniform)
            linear_electron_energies = xp.linspace(self.config.electron.energy_window_min,
                                             self.config.electron.energy_window_max,
                                             self.config.electron.energy_window_num)

            # reverse the (x,y) to (y,x) for interpolation back to the original x-axis
            adaptive_points = xp.interp(targets, cumsum, linear_electron_energies)

            return adaptive_points, monitor, cumsum
        
        # calling function ensures only rank 0 computes the adaptive grid
        adaptive_points, monitor, cumsum = _adaptive_grid_from_monitor(reference_func,
                                                    monitor_type='gradient',
                                                    N_target=self.config.scba.adaptive_num_points)
        return adaptive_points

        # monitor = xp.abs(xp.gradient(reference_func))
        # cumsum = xp.cumsum(monitor)
        # cumsum = cumsum / cumsum[-1]    # normalize to [0,1]

        # equally space the adaptive points in cumulative distribution (of monitor)
        # N_target = self.config.scba.adaptive_num_points
        # targets = xp.linspace(0, 1, N_target)

        # force use linearly spaced energy grid (self.electron_energies may be non-uniform)
        # linear_electron_energies = xp.linspace(self.config.electron.energy_window_min,
        #                                     self.config.electron.energy_window_max,
        #                                     self.config.electron.energy_window_num)

        # reverse the (x,y) to (y,x) for interpolation back to the original x-axis
        # adaptive_points = xp.interp(targets, cumsum, linear_electron_energies)

        # return adaptive_points
        # return xp.asarray([1])

    @profiler.profile(label="SCBA: Phonon interactions", level="default", comm=comm)
    def _compute_phonon_interaction(self):
        """Computes the phonon interaction."""
        if self.config.phonon.model == "negf":
            raise NotImplementedError

        elif self.config.phonon.model == "pseudo-scattering":
            self.sigma_phonon.compute(
                self.data.g_lesser,
                self.data.g_greater,
                out=(
                    self.data.sigma_lesser,
                    self.data.sigma_greater,
                    self.data.sigma_retarded,
                ),
            )

    @profiler.profile(label="SCBA: Photon interactions", level="default", comm=comm)
    def _compute_photon_interaction(self):
        """Computes the photon interaction."""
        raise NotImplementedError

    @profiler.profile(label="SCBA: Electron interactions", level="default", comm=comm)
    def _compute_coulomb_screening_interaction(self, iteration: int, archive_file_prefix: None or str):
        """Computes the Coulomb screening interaction."""

        self.data.p_greater.allocate_data()
        self.data.p_lesser.allocate_data()
        self.data.p_retarded.allocate_data()

        # compute polarization, pass in adaptive grid if needed, else None
        source_adaptive_points = None
        target_adaptive_points = None

        if self.config.scba.adaptive and iteration >= self.config.scba.adaptive_start_iteration:
            source_adaptive_points = self.adaptive_electron_energies_for_g_sigma
            target_adaptive_points = self.adaptive_electron_energies_for_p_w
        self.p_coulomb_screening.compute(
            self.data.g_lesser,
            self.data.g_greater,
            source_adaptive_points=source_adaptive_points,
            target_adaptive_points=target_adaptive_points,
            out=(self.data.p_lesser, self.data.p_greater, self.data.p_retarded)
        )

        # create adaptive grid and interpolate the p functions onto the new grid
        #   create the grid in iteration n-1, to be used in iteration n
        if self.config.scba.adaptive and iteration == self.config.scba.adaptive_start_iteration-1:

            # reduction must be called by all ranks, will hang if it's in a if comm.rank==0 block
            #   but result is only placed in rank 0
            p_retarded_reduced = reduce_matrix_over_stack(self.data.p_retarded, global_comm)

            comm.barrier()
            adaptive_electron_energies_for_p_w = np.empty(self.config.scba.adaptive_num_points, dtype=np.float64)
            # rank 0 computes adaptive grid and broadcast to other ranks
            if comm.rank == 0:
                print(f"rank {comm.rank} - computing adaptive grid", flush=True)
                # with profiler.profile_range(
                #     label="SCBA: compute adaptive grid", level="default", comm=comm
                # ):
                # liyongda (13 Mar 2026): adding profiler decorator to function causes MPI stall, 
                #   because only rank 0 calls the function and there is an internal sync in the profiler
                #       --> other ranks never sync because they never executed the function
                #   tried doing the profiler with a `with`, still didn't work
                #   computing grid is is O(N) and not time critical. Ignoring it for now
                adaptive_electron_energies_for_p_w = self._compute_adaptive_grid(p_retarded_reduced)
            
            comm.barrier()
            comm.Bcast(adaptive_electron_energies_for_p_w, root=0)

            # update adaptive grid from None to the computed grid
            self.adaptive_electron_energies_for_p_w = adaptive_electron_energies_for_p_w

            # debugging and save
            if comm.rank == 0:
                print(f"Adaptive energy grid from computed p_retarded_reduced with {len(self.adaptive_electron_energies_for_p_w)} points.", flush=True)
                print(f"Original linear grid had {len(self.electron_energies)} points from {self.electron_energies[0]} to {self.electron_energies[-1]} (dE = {self.electron_energies[1] - self.electron_energies[0]} eV)", flush=True)
                
                xp.save(self.config.output_dir / "adaptive_electron_energies_for_p_w.npy", self.adaptive_electron_energies_for_p_w)
                print(f"Saved adaptive energy grid to output directory.", flush=True)
            
            # 04 Feb 2026: will get errors from resizing the memory buffer if the number of adaptive points
            #   is different from the original grid.
            #   solve the error by forcing number of adaptive points to be the same as the original grid
            #   discussions with AlexMaeder, Nicolas, and Anders support this

            # debugging
            # comm.barrier() 
            # print(f"rank {comm.rank} has g_lesser shape before interp: {self.data.g_lesser.data.shape}", flush=True)

            comm.barrier()

            # each rank needs all energies, but only some nnz
            # transpose from stack to nnz distribution
            if self.data.p_lesser.distribution_state != "nnz":
                if comm.rank == 0:
                    print(f"transposing p functions from stack to nnz distribution for interpolation", flush=True)
                for m in [ self.data.p_lesser, self.data.p_greater, self.data.p_retarded ]:
                    m.dtranspose(discard=False)  # This must not be discarded.
                    assert m.distribution_state == "nnz"

            # bsplit with k=1 for linear interpolation
            # able to be batched, ex. x=(11,), y=(11,1000), 1000 sets of y-data to be interpolated on the same x-axis
            # https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#batches-of-y
            k = self.config.scba.adaptive_interpolation_order   # 1 (linear), 2 (quadratic), 3 (cubic)
            bspl_lesser = make_interp_spline(self.electron_energies, self.data.p_lesser.data, k=k)
            bspl_greater = make_interp_spline(self.electron_energies, self.data.p_greater.data, k=k)
            bspl_retarded = make_interp_spline(self.electron_energies, self.data.p_retarded.data, k=k)
            
            self.data.p_lesser.data = bspl_lesser(self.adaptive_electron_energies_for_p_w)
            self.data.p_greater.data = bspl_greater(self.adaptive_electron_energies_for_p_w)
            self.data.p_retarded.data = bspl_retarded(self.adaptive_electron_energies_for_p_w)

            # transpose back from nnz to stack
            if self.data.p_lesser.distribution_state != "stack":
                if comm.rank == 0:
                    print(f"transposing p functions back from nnz to stack distribution after interpolation", flush=True)
                for m in [ self.data.p_lesser, self.data.p_greater, self.data.p_retarded ]:
                    m.dtranspose(discard=False)  # This must not be discarded.
                    assert m.distribution_state == "stack"

        comm.barrier()

        # save p (polarization)
        if self.config.outputs.save_reduced_functions:
            p_lesser_reduced = reduce_matrix_over_stack(self.data.p_lesser, global_comm)
            p_greater_reduced = reduce_matrix_over_stack(self.data.p_greater, global_comm)
            p_retarded_reduced = reduce_matrix_over_stack(self.data.p_retarded, global_comm)
            if comm.rank == 0:
                xp.save(self.config.output_dir / f"p_lesser_reduced_step_{iteration}.npy", np.array(p_lesser_reduced).flatten())
                xp.save(self.config.output_dir / f"p_greater_reduced_step_{iteration}.npy", np.array(p_greater_reduced).flatten())
                xp.save(self.config.output_dir / f"p_retarded_reduced_step_{iteration}.npy", np.array(p_retarded_reduced).flatten())
                print(f"saved reduced p for iteration {iteration}", flush=True)
            comm.barrier()

        if self.config.outputs.save_scba_iteration_data:
            # all ranks load the random sample indices and perform gather
            sample_indices = np.load(f"{archive_file_prefix}_sample_indices.npy")

            p_lesser_concat = gather_array_stack(self.data.p_lesser.data, global_comm, sample_indices)
            p_greater_concat = gather_array_stack(self.data.p_greater.data, global_comm, sample_indices)
            p_retarded_concat = gather_array_stack(self.data.p_retarded.data, global_comm, sample_indices)
            if comm.rank == 0:
                xp.save(f"{archive_file_prefix}_p_lesser_iter{iteration:02}.npy", p_lesser_concat)
                xp.save(f"{archive_file_prefix}_p_greater_iter{iteration:02}.npy", p_greater_concat)
                xp.save(f"{archive_file_prefix}_p_retarded_iter{iteration:02}.npy", p_retarded_concat)
                print(f"saved p files for iteration {iteration}", flush=True)
            comm.barrier()

        comm.barrier()

        self.data.w_greater.allocate_data()
        self.data.w_lesser.allocate_data()

        # solve for w (screened Coulomb interaction) using p (polarization)
        # liyongda (05 Feb 2026): don't think I need to do anything special for adaptive grid here\
        # liyongda (13 Mar 2026): confirmed with Anders. No explicit energy dependence. No changes needed
        self.coulomb_screening_solver.solve(
            self.data.p_lesser,
            self.data.p_greater,
            self.data.p_retarded,
            out=(self.data.w_lesser, self.data.w_greater),
        )
        comm.barrier()

        # save w (coulomb screened interaction)
        if self.config.outputs.save_reduced_functions:
            w_lesser_reduced = reduce_matrix_over_stack(self.data.w_lesser, global_comm)
            w_greater_reduced = reduce_matrix_over_stack(self.data.w_greater, global_comm)
            if comm.rank == 0:
                xp.save(self.config.output_dir /  f"w_lesser_reduced_step_{iteration}.npy", np.array(w_lesser_reduced).flatten())
                xp.save(self.config.output_dir /  f"w_greater_reduced_step_{iteration}.npy", np.array(w_greater_reduced).flatten())
                print(f"saved reduced w for iteration {iteration}", flush=True)
            comm.barrier()

        if self.config.outputs.save_scba_iteration_data:
            w_lesser_concat = gather_array_stack(self.data.w_lesser.data, global_comm, sample_indices)
            w_greater_concat = gather_array_stack(self.data.w_greater.data, global_comm, sample_indices)
            if comm.rank == 0:
                xp.save(f"{archive_file_prefix}_w_lesser_iter{iteration:02}.npy", w_lesser_concat)
                xp.save(f"{archive_file_prefix}_w_greater_iter{iteration:02}.npy", w_greater_concat)
                print(f"saved w files for iteration {iteration}", flush=True)
            comm.barrier()

        comm.barrier()
        self._compute_coulomb_screening_observables()

        self.data.p_lesser.free_data()
        self.data.p_greater.free_data()
        self.data.p_retarded.free_data()

        # update the energy grid for sigma fock computation if we have a new adaptive grid
        if self.config.scba.adaptive and iteration >= self.config.scba.adaptive_start_iteration:
            self.sigma_fock.update_energies(self.adaptive_electron_energies_for_g_sigma)
        comm.barrier()

        # up to the user to update the energy grid to adaptive grid before calling `sigma_fock.compute`
        #   --> computes integral of g_lesser over the grid
        self.sigma_fock.compute(
            self.data.g_lesser,
            use_adaptive = self.config.scba.adaptive and iteration >= self.config.scba.adaptive_start_iteration,
            adaptive_integration_method = self.config.scba.adaptive_integration_method,
            out=(self.data.sigma_retarded,)
        )

        comm.barrier()

        # update the energy grid for sigma coulomb screening computation if we have a new adaptive grid
        if self.config.scba.adaptive and iteration >= self.config.scba.adaptive_start_iteration:
            self.sigma_coulomb_screening.update_energies(self.adaptive_electron_energies_for_g_sigma)
        comm.barrier()

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
            use_adaptive = self.config.scba.adaptive and iteration >= self.config.scba.adaptive_start_iteration
        )

        comm.barrier()

        self.data.w_greater.free_data()
        self.data.w_lesser.free_data()

    @profiler.profile(label="SCBA: G observables", level="default", comm=comm)
    def _compute_electron_observables(self) -> None:
        """Computes electron observables."""
        overlap = (
            None
            if self.electron_solver.orthogonal_basis
            else self.electron_solver.overlap
        )
        if self.config.outputs.electron_ldos:
            self.observables.electron_ldos = -density(
                self.data.g_retarded,
                overlap,
            ) / (2 * xp.pi)
        if self.config.outputs.electron_density:
            self.observables.electron_density = density(
                self.data.g_lesser,
                overlap,
            ) / (2 * xp.pi)
        if self.config.outputs.hole_density:
            self.observables.hole_density = -density(
                self.data.g_greater,
                overlap,
            ) / (2 * xp.pi)

        if self.config.outputs.contact_currents:
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
        if self.config.outputs.device_currents:
            self.observables.electron_current["device"] = device_current(
                self.data.g_lesser, self.electron_solver.hamiltonian
            )
            if self.config.electron.solver.compute_current:
                if comm.block.size > 1:
                    raise NotImplementedError(
                        "Meir-Wingreen current is not implemented for distributed SCBA."
                    )

                local_current = self.electron_solver.meir_wingreen_current
                meir_wingreen_current = comm.stack.all_gather_v(
                    local_current,
                    axis=0,
                    mask=self.data.g_lesser._stack_padding_mask,
                )

                self.observables.electron_current["meir-wingreen"] = (
                    meir_wingreen_current
                )

        if self.config.outputs.self_energy_density:
            self.observables.sigma_retarded_density = -density(
                self.data.sigma_retarded,
                overlap,
            ) / (2 * xp.pi)
            self.observables.sigma_lesser_density = density(
                self.data.sigma_lesser,
                overlap,
            ) / (2 * xp.pi)
            self.observables.sigma_greater_density = -density(
                self.data.sigma_greater,
                overlap,
            ) / (2 * xp.pi)

    @profiler.profile(label="SCBA: W observables", level="default", comm=comm)
    def _compute_coulomb_screening_observables(self) -> None:

        # NOTE: The overlap is maybe missing here (it is not used)
        if self.config.outputs.polarization_density:
            self.observables.p_retarded_density = -density(self.data.p_retarded) / (
                2 * xp.pi
            )
            self.observables.p_lesser_density = density(self.data.p_lesser) / (
                2 * xp.pi
            )
            self.observables.p_greater_density = -density(self.data.p_greater) / (
                2 * xp.pi
            )

        if self.config.outputs.coulomb_screening_density:
            self.observables.w_lesser_density = density(self.data.w_lesser) / (
                2 * xp.pi
            )
            self.observables.w_greater_density = -density(self.data.w_greater) / (
                2 * xp.pi
            )

    @profiler.profile(label="SCBA: Write outputs", level="default", comm=comm)
    def _write_iteration_outputs(self, iteration: int):
        """Writes output for the current iteration on rank zero."""

        if comm.rank != 0:
            return

        outputs = {}

        if self.config.outputs.electron_ldos:
            outputs[f"electron_ldos_{iteration}.npy"] = self.observables.electron_ldos
        if self.config.outputs.electron_density:
            outputs[f"electron_density_{iteration}.npy"] = (
                self.observables.electron_density
            )
        if self.config.outputs.hole_density:
            outputs[f"hole_density_{iteration}.npy"] = self.observables.hole_density

        if self.config.outputs.contact_currents:
            outputs.update(
                {
                    f"i_{contact}_{iteration}.npy": current
                    for contact, current in self.observables.electron_current.items()
                }
            )
        if self.config.outputs.device_currents:
            outputs[f"device_current_{iteration}.npy"] = (
                self.observables.electron_current["device"]
            )
            if self.config.electron.solver.compute_current:
                if comm.block.size > 1:
                    raise NotImplementedError(
                        "Meir-Wingreen current is not implemented for distributed SCBA."
                    )

                outputs[f"meir_wingreen_current_{iteration}.npy"] = (
                    self.observables.electron_current["meir-wingreen"]
                )

        if self.config.scba.coulomb_screening:
            if self.config.outputs.polarization_density:
                outputs.update(
                    {
                        f"p_lesser_density_{iteration}.npy": self.observables.p_lesser_density,
                        f"p_greater_density_{iteration}.npy": self.observables.p_greater_density,
                        f"p_retarded_density_{iteration}.npy": self.observables.p_retarded_density,
                    }
                )
            if self.config.outputs.coulomb_screening_density:
                outputs.update(
                    {
                        f"w_lesser_density_{iteration}.npy": self.observables.w_lesser_density,
                        f"w_greater_density_{iteration}.npy": self.observables.w_greater_density,
                    }
                )

        if self.config.outputs.self_energy_density:
            outputs.update(
                {
                    f"sigma_retarded_density_{iteration}.npy": self.observables.sigma_retarded_density,
                    f"sigma_lesser_density_{iteration}.npy": self.observables.sigma_lesser_density,
                    f"sigma_greater_density_{iteration}.npy": self.observables.sigma_greater_density,
                }
            )

        print(f"Writing output for iteration {iteration}...", flush=True)

        if not os.path.exists(self.config.output_dir):
            os.mkdir(self.config.output_dir)

        for filename, data in outputs.items():
            xp.save(self.config.output_dir / filename, data)

    @profiler.profile(label="SCBA", level="default", comm=comm)
    def run(self) -> None:
        """Runs the SCBA to convergence."""
        print("Entering SCBA loop...", flush=True) if comm.rank == 0 else None

        archive_file_prefix = None
        sample_indices = None
        if self.config.outputs.save_scba_iteration_data:
            archive_file_prefix = self.config.output_dir / "visualize_scba"
            
            # random samples must be less than the total number of non-zero elements
            num_random_samples = min(self.config.outputs.num_nnz_samples_scba_iteration_data, self.data.g_lesser.data.shape[1])

            # only rank 0 generate random indices and saves it, for the other ranks to use
            if comm.rank == 0:
                # generate unique random sample indices between 0-43824 (number of non-zero indices)
                rng = np.random.default_rng(42)
                sample_indices = rng.choice(self.data.g_lesser.data.shape[1], size=num_random_samples, replace=False)
                sample_indices = np.sort(sample_indices)

                # save indices
                np.save(f"{archive_file_prefix}_sample_indices.npy", sample_indices)


        # save electron energies used, used in post-processing.ipynb
        if comm.rank == 0:
            xp.save(self.config.output_dir / "electron_energies.npy", get_host(self.electron_energies))
        comm.barrier()

        for i in range(self.config.scba.max_iterations):
            print(f"Iteration {i}", flush=True) if comm.rank == 0 else None

            with profiler.profile_range(
                label="SCBA: Iteration", level="default", comm=comm
            ):
                # we're in adaptive mode
                if self.config.scba.adaptive and i >= self.config.scba.adaptive_start_iteration:
                    self.electron_solver.update_energies(self.adaptive_electron_energies_for_g_sigma)
                    
                # compute the Green's function from scattering self-energies
                # liyongda (23 Feb 2026): iteration 0, all sigma data is zero (checked with debugger and np.allclose)
                self.electron_solver.solve(
                    self.data.sigma_lesser,
                    self.data.sigma_greater,
                    self.data.sigma_retarded,
                    out=(self.data.g_lesser, self.data.g_greater, self.data.g_retarded)
                )

            comm.barrier()

            # save g
            if self.config.outputs.save_reduced_functions:
                g_lesser_reduced = reduce_matrix_over_stack(self.data.g_lesser, global_comm)
                g_greater_reduced = reduce_matrix_over_stack(self.data.g_greater, global_comm)
                g_retarded_reduced = reduce_matrix_over_stack(self.data.g_retarded, global_comm)
                if comm.rank == 0:
                    xp.save(self.config.output_dir /  f"g_lesser_reduced_step_{i}.npy", np.array(g_lesser_reduced).flatten())
                    xp.save(self.config.output_dir /  f"g_greater_reduced_step_{i}.npy", np.array(g_greater_reduced).flatten())
                    xp.save(self.config.output_dir /  f"g_retarded_reduced_step_{i}.npy", np.array(g_retarded_reduced).flatten())
                    print(f"saved reduced g for iteration {i}", flush=True)
            comm.barrier()

            if self.config.outputs.save_scba_iteration_data:
                # all ranks load the random sample indices and perform gather
                sample_indices = np.load(f"{archive_file_prefix}_sample_indices.npy")

                g_lesser_concat = gather_array_stack(self.data.g_lesser.data, global_comm, sample_indices)
                g_greater_concat = gather_array_stack(self.data.g_greater.data, global_comm, sample_indices)
                g_retarded_concat = gather_array_stack(self.data.g_retarded.data, global_comm, sample_indices)
                if comm.rank == 0:
                    xp.save(f"{archive_file_prefix}_g_lesser_iter{i:02}.npy", g_lesser_concat)
                    xp.save(f"{archive_file_prefix}_g_greater_iter{i:02}.npy", g_greater_concat)
                    xp.save(f"{archive_file_prefix}_g_retarded_iter{i:02}.npy", g_retarded_concat)
                    print(f"saved g files for iteration {i}", flush=True)

            comm.barrier()
            # create adaptive grid and interpolate the g functions onto the new grid
            #   create the grid in iteration n-1, to be used in iteration n
            if self.config.scba.adaptive and i == self.config.scba.adaptive_start_iteration-1:
                # reduction must be called by all ranks, will hang if it's in a if comm.rank==0 block
                #   but result is only placed in rank 0
                g_retarded_reduced = reduce_matrix_over_stack(self.data.g_retarded, global_comm)

                comm.barrier()
                adaptive_electron_energies_for_g_sigma = np.empty(self.config.scba.adaptive_num_points, dtype=np.float64)
                # rank 0 computes adaptive grid and broadcast to other ranks
                if comm.rank == 0:
                    print(f"rank {comm.rank} - computing adaptive grid", flush=True)
                    # with profiler.profile_range(
                    #     label="SCBA: compute adaptive grid", level="default", comm=comm
                    # ):
                    # liyongda (13 Mar 2026): adding profiler decorator to function causes MPI stall, 
                    #   because only rank 0 calls the function and there is an internal sync in the profiler
                    #       --> other ranks never sync because they never executed the function
                    #   tried doing the profiler with a `with`, still didn't work
                    #   computing grid is is O(N) and not time critical. Ignoring it for now
                    adaptive_electron_energies_for_g_sigma = self._compute_adaptive_grid(g_retarded_reduced)
                
                comm.barrier()
                comm.Bcast(adaptive_electron_energies_for_g_sigma, root=0)

                # update adaptive grid from None to the computed grid
                self.adaptive_electron_energies_for_g_sigma = adaptive_electron_energies_for_g_sigma

                # debugging and save
                if comm.rank == 0:
                    print(f"Adaptive energy grid from computed g_retarded_reduced with {len(self.adaptive_electron_energies_for_g_sigma)} points.", flush=True)
                    print(f"Original linear grid had {len(self.electron_energies)} points from {self.electron_energies[0]} to {self.electron_energies[-1]} (dE = {self.electron_energies[1] - self.electron_energies[0]} eV)", flush=True)
                    
                    xp.save(self.config.output_dir / "adaptive_electron_energies_for_g_sigma.npy", self.adaptive_electron_energies_for_g_sigma)
                    print(f"Saved adaptive energy grid to output directory.", flush=True)
                
                # 04 Feb 2026: will get errors from resizing the memory buffer if the number of adaptive points
                #   is different from the original grid.
                #   solve the error by forcing number of adaptive points to be the same as the original grid
                #   discussions with AlexMaeder, Nicolas, and Anders support this

                # debugging
                # comm.barrier() 
                # print(f"rank {comm.rank} has g_lesser shape before interp: {self.data.g_lesser.data.shape}", flush=True)

                comm.barrier()

                # each rank needs all energies, but only some nnz
                # transpose from stack to nnz distribution
                for m in [ self.data.g_lesser, self.data.g_greater, self.data.g_retarded ]:
                    m.dtranspose(discard=False)  # This must not be discarded.
                    assert m.distribution_state == "nnz"

                # bsplit with k=1 for linear interpolation
                # able to be batched, ex. x=(11,), y=(11,1000), 1000 sets of y-data to be interpolated on the same x-axis
                # https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#batches-of-y
                k = self.config.scba.adaptive_interpolation_order   # 1 (linear), 2 (quadratic), 3 (cubic)
                bspl_lesser = make_interp_spline(self.electron_energies, self.data.g_lesser.data, k=k)
                bspl_greater = make_interp_spline(self.electron_energies, self.data.g_greater.data, k=k)
                bspl_retarded = make_interp_spline(self.electron_energies, self.data.g_retarded.data, k=k)
                
                self.data.g_lesser.data = bspl_lesser(self.adaptive_electron_energies_for_g_sigma)
                self.data.g_greater.data = bspl_greater(self.adaptive_electron_energies_for_g_sigma)
                self.data.g_retarded.data = bspl_retarded(self.adaptive_electron_energies_for_g_sigma)

                # transpose back from nnz to stack
                for m in [ self.data.g_lesser, self.data.g_greater, self.data.g_retarded ]:
                    m.dtranspose(discard=False)  # This must not be discarded.
                    assert m.distribution_state == "stack"

            comm.barrier()
            self._compute_electron_observables()

            # Stash current into previous self-energy buffer.
            self._stash_sigma()

            with profiler.profile_range(
                    label="SCBA: stack->nnz transpose", level="default", comm=comm
            ):
                # Transpose to nnz distribution.
                # NOTE: While computing all interactions, we only ever need
                # to access the Green's function and the self-energies in
                # their nnz-distributed state.
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

            # compute polarization, then screened Coulomb interaction, then self-energy from the screened interaction
            if self.config.scba.coulomb_screening:
                self._compute_coulomb_screening_interaction(
                    iteration=i,
                    archive_file_prefix=archive_file_prefix
                )

            if self.config.scba.photon:
                self._compute_photon_interaction()

            if self.config.scba.phonon:
                self._compute_phonon_interaction()

            with profiler.profile_range(
                    label="SCBA: stack->nnz transpose back", level="default", comm=comm
                ):
                # Transpose back to stack distribution.
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

            # save sigma
            if self.config.outputs.save_reduced_functions:
                # save g at start of loop immediately after it's computed
                # save p and w inside _compute_coulomb_screening_interaction because it's freed in the function

                # save Sigma after it's been transposed back to stack distribution
                sigma_lesser_reduced = reduce_matrix_over_stack(self.data.sigma_lesser, global_comm)
                sigma_greater_reduced = reduce_matrix_over_stack(self.data.sigma_greater, global_comm)
                sigma_retarded_reduced = reduce_matrix_over_stack(self.data.sigma_retarded, global_comm)
                if comm.rank == 0:
                    # save to file
                    xp.save(self.config.output_dir /  f"sigma_lesser_reduced_step_{i}.npy", np.array(sigma_lesser_reduced).flatten())
                    xp.save(self.config.output_dir /  f"sigma_greater_reduced_step_{i}.npy", np.array(sigma_greater_reduced).flatten())
                    xp.save(self.config.output_dir /  f"sigma_retarded_reduced_step_{i}.npy", np.array(sigma_retarded_reduced).flatten())
                    print(f"saved reduced sigma for iteration {i}", flush=True)
                comm.barrier()

            if self.config.outputs.save_scba_iteration_data:
                # save g at start of loop immediately after it's computed
                # save p and w inside _compute_coulomb_screening_interaction because it's freed right after use

                # save Sigma after it's been transposed back to stack distribution
                sigma_lesser_concat = gather_array_stack(self.data.sigma_lesser.data, global_comm, sample_indices)
                sigma_greater_concat = gather_array_stack(self.data.sigma_greater.data, global_comm, sample_indices)
                sigma_retarded_concat = gather_array_stack(self.data.sigma_retarded.data, global_comm, sample_indices)
                if comm.rank == 0:
                    xp.save(f"{archive_file_prefix}_sigma_lesser_iter{i:02}.npy", sigma_lesser_concat)
                    xp.save(f"{archive_file_prefix}_sigma_greater_iter{i:02}.npy", sigma_greater_concat)
                    xp.save(f"{archive_file_prefix}_sigma_retarded_iter{i:02}.npy", sigma_retarded_concat)

                print(f"saved sigma files for iteration {i}", flush=True) if comm.rank == 0 else None
                comm.barrier()


            comm.barrier()

            # Symmetrize the self-energy.
            self._symmetrize_sigma()

            if self._has_converged():
                if comm.rank == 0:
                    print(f"SCBA converged after {i} iterations.", flush=True)
                break

            # Update self-energy for next iteration with mixing factor.
            self._update_sigma()

            if xp.__name__ == "cupy":
                free_memory, total_memory = xp.cuda.Device().mem_info
                usage = np.array((total_memory - free_memory) / total_memory)
                average_usage = np.empty(1)
                max_usage = np.empty(1)
                global_comm.Allreduce(usage, average_usage, op=MPI.SUM)
                global_comm.Allreduce(usage, max_usage, op=MPI.MAX)
                average_usage /= comm.size

                if comm.rank == 0:
                    print(
                        f"Rank-average device memory usage: {average_usage[0] * 100:.4f}%",
                        flush=True,
                    )
                    print(
                        f"Max device memory usage: {max_usage[0] * 100:.4f}%",
                        flush=True,
                    )

            if i % self.config.scba.output_interval == 0:
                self._write_iteration_outputs(i)

        else:  # Did not break, i.e. max_iterations reached.
            if comm.rank == 0:
                print(f"SCBA did not converge after {i} iterations.")
