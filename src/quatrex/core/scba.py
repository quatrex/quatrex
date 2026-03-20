# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import os
from dataclasses import dataclass, field

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.kernels.mixed_precision import compress, decompress
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import create_stream, get_host, synchronize_stream
from qttools.utils.mpi_utils import distributed_load, get_section_sizes
from quatrex.core.config import QuatrexConfig
from quatrex.core.observables import (  # current_conservation,
    contact_currents,
    density,
    device_current,
)
from quatrex.core.utils import compute_num_connected_blocks, compute_sparsity_pattern
from quatrex.coulomb_screening import CoulombScreeningSolver, PCoulombScreening
from quatrex.device.inputs import (
    assemble_matrix,
    create_coordinate_grid,
    distributed_read_xyz,
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

            block_sizes = np.array(
                [config.device.neighbor_cell_cutoff[transport_ind] * grid.shape[0]]
                * config.device.num_transport_cells
            )

            grid = create_coordinate_grid(
                grid,
                config.device.num_transport_cells
                * config.device.neighbor_cell_cutoff[transport_ind],
                transport_ind,
                xp.asarray(lattice_vectors),
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
        if comm.rank == 0:
            print(f"Grid shape: {grid.shape}", flush=True)
            print(f"Block size: {block_sizes[0]}", flush=True)

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

            rows, cols = compute_sparsity_pattern(
                grid,
                max_interaction_cutoff,
                transport_direction=config.device.transport_direction,
                start_idx=start_idx,
                end_idx=end_idx,
            )
            self.rows = rows
            self.cols = cols

        dsdbsparse_type = config.compute.dsdbsparse_type

        self.g_retarded = dsdbsparse_type.from_sparray(
            rows,
            cols,
            block_sizes=block_sizes,
            global_stack_shape=electron_energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
            bits=config.compute.num_bits,
        )

        self.g_lesser = dsdbsparse_type.from_sparray(
            rows,
            cols,
            block_sizes=block_sizes,
            global_stack_shape=electron_energies.shape
            + tuple([k for k in kpoint_grid if k > 1]),
            symmetry=config.scba.symmetric,
            symmetry_op=lambda a: -a.conj(),
            bits=config.compute.num_bits,
        )
        self.g_greater = dsdbsparse_type.empty_like(self.g_lesser)

        self.sigma_lesser_prev = dsdbsparse_type.empty_like(self.g_lesser)
        self.sigma_lesser = dsdbsparse_type.empty_like(self.g_lesser)
        self.sigma_greater_prev = dsdbsparse_type.empty_like(self.g_lesser)
        self.sigma_greater = dsdbsparse_type.empty_like(self.g_lesser)

        self.sigma_retarded_prev = dsdbsparse_type.empty_like(self.g_lesser)
        self.sigma_retarded = dsdbsparse_type.empty_like(self.g_lesser)

        if config.scba.symmetric:
            self.sigma_retarded.symmetry_op = lambda a: a
            self.sigma_retarded_prev.symmetry_op = lambda a: a

        if config.scba.coulomb_screening:
            # NOTE: The polarization has the same sparsity pattern as
            # the electronic system (the interactions are local in real
            # space). However, we need to change the block sizes of the
            # screened Coulomb interaction.
            self.p_retarded = dsdbsparse_type.empty_like(self.g_lesser)
            self.p_lesser = dsdbsparse_type.empty_like(self.g_lesser)
            self.p_greater = dsdbsparse_type.empty_like(self.g_lesser)

            num_connected_blocks = config.coulomb_screening.num_connected_blocks
            if num_connected_blocks == "auto":
                num_connected_blocks = compute_num_connected_blocks(
                    rows, cols, block_sizes
                )

            if comm.rank == 0:
                print(f"Number of connected blocks: {num_connected_blocks}", flush=True)

            # TODO: This only works for constant block sizes.
            coulomb_screening_block_sizes = (
                block_sizes[: len(block_sizes) // num_connected_blocks]
                * num_connected_blocks
            )

            self.w_lesser = dsdbsparse_type.from_sparray(
                rows,
                cols,
                block_sizes=coulomb_screening_block_sizes,
                global_stack_shape=electron_energies.shape
                + tuple([k for k in kpoint_grid if k > 1]),
                symmetry=config.scba.symmetric,
                symmetry_op=lambda a: -a.conj(),
                bits=config.compute.num_bits,
            )
            self.w_greater = dsdbsparse_type.empty_like(self.w_lesser)

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

        self.electron_solver = ElectronSolver(
            self.config,
            self.electron_energies,
            rows=self.data.rows,
            cols=self.data.cols,
        )

        # ----- Coulomb screening --------------------------------------
        if self.config.scba.coulomb_screening:
            # Load the Coulomb matrix.
            coulomb_matrix, __ = assemble_matrix(
                config=config,
                matrix_name="coulomb_matrix",
                sparsity_pattern=(self.data.rows, self.data.cols),
                shift_kpoints=True,
            )

            # Make sure the Coulomb matrix is hermitian.
            # TODO: Check that this is correct for kpoints.
            if not coulomb_matrix.symmetry:
                coulomb_matrix.symmetrize()
            if config.compute.num_bits is None:
                coulomb_matrix._data /= config.coulomb_screening.epsilon_r
            else:
                coulomb_matrix._data = compress(
                    decompress(coulomb_matrix._data, config.compute.num_bits)
                    / config.coulomb_screening.epsilon_r,
                    config.compute.num_bits,
                )

            energies_path = self.config.input_dir / "coulomb_screening_energies.npy"
            if os.path.isfile(energies_path):
                self.coulomb_screening_energies = distributed_load(energies_path)
            else:
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
                rows=self.data.rows,
                cols=self.data.cols,
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

        self._copy_stream = create_stream()

    def _stash_sigma(self) -> None:
        """Stash the current into the previous self-energy buffers."""

        # cursed?
        # does not copy the data, but just the reference to the data, so it is very cheap.
        self.data.sigma_lesser_prev._host_data = self.data.sigma_lesser._host_data
        self.data.sigma_greater_prev._host_data = self.data.sigma_greater._host_data
        self.data.sigma_retarded_prev._host_data = self.data.sigma_retarded._host_data

        self.data.sigma_lesser._host_data = None
        self.data.sigma_greater._host_data = None
        self.data.sigma_retarded._host_data = None

        self.data.sigma_lesser.free_data()
        self.data.sigma_greater.free_data()
        self.data.sigma_retarded.free_data()

    @profiler.profile(label="SCBA: Symmetrize Sigma", level="default", comm=comm)
    def _symmetrize_sigma(self) -> None:
        # Symmetrization.
        if not self.config.scba.symmetric:
            self.data.sigma_lesser.symmetrize(xp.subtract)
            self.data.sigma_greater.symmetrize(xp.subtract)
            # Make the self-energy Hermitian (removing the skew-Hermitian part).
            self.data.sigma_retarded.symmetrize(xp.add)

        if self.config.coulomb_screening.discard_real_parts:

            if self.data.sigma_lesser.bits is not None:
                _data = decompress(
                    self.data.sigma_lesser._data, self.data.sigma_lesser.bits
                )
            else:
                _data = self.data.sigma_lesser._data
            _data.real = 0
            if self.data.sigma_lesser.bits is not None:
                self.data.sigma_lesser._data = compress(
                    _data, self.data.sigma_lesser.bits
                )

            if self.data.sigma_greater.bits is not None:
                _data = decompress(
                    self.data.sigma_greater._data, self.data.sigma_greater.bits
                )
            else:
                _data = self.data.sigma_greater._data
            _data.real = 0
            if self.data.sigma_greater.bits is not None:
                self.data.sigma_greater._data = compress(
                    _data, self.data.sigma_greater.bits
                )
            # Make sure that the imaginary part comes only from
            # sigma_greater - sigma_lesser.
            if self.data.sigma_retarded.bits is not None:
                _data = decompress(
                    self.data.sigma_retarded._data, self.data.sigma_retarded.bits
                )
            else:
                _data = self.data.sigma_retarded._data
            _data.imag = 0
            if self.data.sigma_retarded.bits is not None:
                self.data.sigma_retarded._data = compress(
                    _data, self.data.sigma_retarded.bits
                )

        # Now add the imaginary, skew-Hermitian part back.
        if self.data.sigma_retarded.bits is not None:
            self.data.sigma_retarded.data = compress(
                0.5
                * (
                    decompress(
                        self.data.sigma_greater.data, self.data.sigma_greater.bits
                    )
                    - decompress(
                        self.data.sigma_lesser.data, self.data.sigma_greater.bits
                    )
                )
                + decompress(
                    self.data.sigma_retarded.data, self.data.sigma_retarded.bits
                ),
                self.data.sigma_retarded.bits,
            )
        else:
            self.data.sigma_retarded.data += 0.5 * (
                self.data.sigma_greater.data - self.data.sigma_lesser.data
            )

    @profiler.profile(label="SCBA: Update Sigma", level="default", comm=comm)
    def _update_sigma(self) -> None:
        """Updates the self-energy with a mixing factor."""

        if self.data.sigma_lesser.bits is not None:
            bits = self.data.sigma_lesser.bits
            self.data.sigma_lesser.data[:] = compress(
                (
                    (1 - self.mixing_factor)
                    * decompress(self.data.sigma_lesser_prev.data, bits)
                    + self.mixing_factor * decompress(self.data.sigma_lesser.data, bits)
                ),
                bits,
            )
            self.data.sigma_greater.data[:] = compress(
                (
                    (1 - self.mixing_factor)
                    * decompress(self.data.sigma_greater_prev.data, bits)
                    + self.mixing_factor
                    * decompress(self.data.sigma_greater.data, bits)
                ),
                bits,
            )
            self.data.sigma_retarded.data[:] = compress(
                (
                    (1 - self.mixing_factor)
                    * decompress(self.data.sigma_retarded_prev.data, bits)
                    + self.mixing_factor
                    * decompress(self.data.sigma_retarded.data, bits)
                ),
                bits,
            )
        else:
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
        if self.data.sigma_lesser.bits is not None:
            diff = decompress(
                self.data.sigma_retarded.data, self.data.sigma_lesser.bits
            ) - decompress(
                self.data.sigma_retarded_prev.data, self.data.sigma_lesser.bits
            )
        else:
            diff = self.data.sigma_retarded.data - self.data.sigma_retarded_prev.data

        local_max_diff = get_host(xp.max(xp.abs(diff)))
        max_diff = np.empty_like(local_max_diff)
        global_comm.Allreduce(local_max_diff, max_diff, op=MPI.MAX)

        i_left = xp.real(self.observables.electron_current.get("left", 0.0))
        i_right = xp.real(self.observables.electron_current.get("right", 0.0))

        dE = self.electron_energies[1] - self.electron_energies[0]
        current_diff = xp.abs(xp.sum(i_left) * dE + xp.sum(i_right) * dE)

        # current_conservation_abs, current_conservation_rel = current_conservation(
        #     self.data.g_lesser,
        #     self.data.g_greater,
        #     self.data.sigma_lesser,
        #     self.data.sigma_greater,
        # )

        if comm.rank == 0:
            print(f"Maximum Self-Energy Update: {max_diff}", flush=True)
            print(f"Contact Current Difference: {current_diff}", flush=True)
            # print(f"Current Conservation abs: {current_conservation_abs}", flush=True)
            # print(f"Current Conservation rel: {current_conservation_rel}", flush=True)

        return False  # TODO: :-)

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
    def _compute_coulomb_screening_interaction(self):
        """Computes the Coulomb screening interaction."""

        self.data.p_greater.allocate_data()
        self.data.p_lesser.allocate_data()
        self.data.p_retarded.allocate_data()
        self.data.g_lesser.to_host(
            delete_device=False, stream=self._copy_stream, sync=False
        )
        self.data.g_greater.to_host(
            delete_device=False, stream=self._copy_stream, sync=False
        )

        self.p_coulomb_screening.compute(
            self.data.g_lesser,
            self.data.g_greater,
            out=(self.data.p_lesser, self.data.p_greater, self.data.p_retarded),
        )
        self.data.g_lesser.free_data()
        self.data.g_greater.free_data()

        # TODO: put G to the host before
        # computing W

        self.data.w_greater.allocate_data()
        self.data.w_lesser.allocate_data()

        self.coulomb_screening_solver.solve(
            self.data.p_lesser,
            self.data.p_greater,
            self.data.p_retarded,
            out=(self.data.w_lesser, self.data.w_greater),
        )

        synchronize_stream(self._copy_stream)
        self.data.g_lesser.to_device(
            delete_host=True, stream=self._copy_stream, sync=False
        )
        self.data.g_greater.to_device(
            delete_host=True, stream=self._copy_stream, sync=False
        )

        self._compute_coulomb_screening_observables()

        self.data.p_lesser.free_data()
        self.data.p_greater.free_data()
        self.data.p_retarded.free_data()

        with profiler.profile_range(
            label="SCBA: sigma stack->nnz transpose", level="default", comm=comm
        ):
            for m in (
                self.data.sigma_lesser,
                self.data.sigma_greater,
                self.data.sigma_retarded,
            ):
                m.allocate_data()
                m.dtranspose(discard=True)  # These can be safely discarded.
                assert m.distribution_state == "nnz"

        synchronize_stream(self._copy_stream)

        self.sigma_fock.compute(
            self.data.g_lesser,
            out=(self.data.sigma_retarded,),
        )

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

        self.data.g_lesser.allocate_data()
        self.data.g_greater.allocate_data()
        self.data.sigma_lesser.allocate_data()
        self.data.sigma_greater.allocate_data()
        self.data.sigma_retarded.allocate_data()

        if comm.rank == 0:
            print("len(G) ", self.data.g_lesser.total_nnz_size, flush=True)
            print(
                "len(MG) ",
                self.electron_solver.system_matrix.total_nnz_size,
                flush=True,
            )
            print(
                "len(H) ", self.electron_solver.hamiltonian.total_nnz_size, flush=True
            )

            print(
                "len(MW) ",
                self.coulomb_screening_solver.system_matrix.total_nnz_size,
                flush=True,
            )
            print(
                "len(L) ",
                self.coulomb_screening_solver.l_lesser.total_nnz_size,
                flush=True,
            )

        # NOTE: benchmark mode
        self.data.sigma_lesser.data[:] = 0.0
        self.data.sigma_greater.data[:] = 0.0
        self.data.sigma_retarded.data[:] = 0.0

        for i in range(self.config.scba.max_iterations):
            print(f"Iteration {i}", flush=True) if comm.rank == 0 else None

            with profiler.profile_range(
                label="SCBA: Iteration", level="default", comm=comm
            ):
                self.data.g_retarded.allocate_data()
                self.electron_solver.solve(
                    self.data.sigma_lesser,
                    self.data.sigma_greater,
                    self.data.sigma_retarded,
                    out=(self.data.g_lesser, self.data.g_greater, self.data.g_retarded),
                )
                self._compute_electron_observables()
                self.electron_solver.hamiltonian.set_to_device()

                self.data.g_retarded.free_data()

                # Stash current into previous self-energy buffer.
                self._stash_sigma()

                for m in (self.data.g_lesser, self.data.g_greater):
                    m.dtranspose(discard=False)  # This must not be discarded.
                    assert m.distribution_state == "nnz"

                if self.config.scba.coulomb_screening:
                    self._compute_coulomb_screening_interaction()

                if self.config.scba.photon:
                    self._compute_photon_interaction()

                if self.config.scba.phonon:
                    self._compute_phonon_interaction()

                with profiler.profile_range(
                    label="SCBA: stack->nnz transpose back", level="default", comm=comm
                ):
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

            # Symmetrize the self-energy.
            self._symmetrize_sigma()

            self.data.sigma_lesser_prev.to_device()
            self.data.sigma_greater_prev.to_device()
            self.data.sigma_retarded_prev.to_device()

            if self._has_converged():
                if comm.rank == 0:
                    print(f"SCBA converged after {i} iterations.", flush=True)
                break

            # Update self-energy for next iteration with mixing factor.
            self._update_sigma()

            self.data.sigma_lesser_prev.free_data()
            self.data.sigma_greater_prev.free_data()
            self.data.sigma_retarded_prev.free_data()
            self.data.sigma_lesser_prev._host_data = None
            self.data.sigma_greater_prev._host_data = None
            self.data.sigma_retarded_prev._host_data = None

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
