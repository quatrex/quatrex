# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the SCBA device class for electronic transport calculations."""

import os
import warnings
from types import SimpleNamespace

import numpy as np

from qttools import NDArray, sparse
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_host
from qttools.utils.mpi_utils import get_section_sizes
from quatrex.contact.scba import SCBAContact
from quatrex.core.config import QuatrexConfig
from quatrex.core.utils import compute_num_connected_blocks, compute_sparsity_pattern
from quatrex.device.base import BaseDevice
from quatrex.device.inputs import assemble_matrix
from quatrex.grid import KGrid
from quatrex.grid.orbit_packing import (
    find_orbital_operations,
    measure_symmetry_breaking,
    repeat_orbital_operation,
)

profiler = Profiler()


class SCBADevice(BaseDevice):
    """A quantum device for electronic transport calculations.

    Parameters
    ----------
    config : QuatrexConfig
        Configuration object containing input paths, device parameters,
        and computational settings.

    Attributes
    ----------
    config : QuatrexConfig
        Reference to the configuration object.
    orbital_coordinates : NDArray
        Array of orbital coordinates.
    atom_coordinates : NDArray
        Array of atomic coordinates.
    atomic_species : NDArray
        Array of atom symbols for each atom. NOTE: This array is always on the
        host since CuPy does not support string arrays.
    orbital_offsets : NDArray
        Array of cumulative orbital counts, used to map from atoms to
        orbitals. orbital_offsets[i] gives the starting orbital index
        for atom i.
    potential : NDArray, optional
        Array of electrostatic potential for each orbital.
        Can be either None if no potential is provided or a 1D array
        where the 1D index corresponds to the orbital index or the
        atom index depending on the shape of the provided potential.
    contacts : list[SCBAContact]
        List of Contact objects representing the semi-infinite leads
        connected to this device.
    orbital_operations : list[OrbitalOperation] | None
        The symmetry operations of the device. None if the interactions
        are not computed from the symmetry-unique elements.

    """

    def __init__(self, config: QuatrexConfig) -> None:
        super().__init__(config)

        # NOTE: The k-points are used for the Hamiltonian and the
        # q-points are used for any interacting systems (e.g., phonons,
        # photons, etc.). The q-points are always gamma-centered.
        # Therefore the number of irreducible k-points and q-points may
        # differ, even if the same grid size is used!
        self.qgrid = KGrid(
            size=self.device_config.kpoint_grid,
            gamma_centered=True,  # Important!
            symmetry_reduction=self.device_config.kpoint_symmetry_reduction,
            structure=self.structure,
            operations=self.kgrid.operations if self.uses_packing else None,
        )
        if comm.rank == 0:
            if not os.path.exists(config.output_dir):
                os.mkdir(config.output_dir)

            if self.device_config.kpoint_symmetry_reduction:
                print(
                    f"Using symmetry-reduced q-point grid. "
                    f"Mapped {self.qgrid.num_full_points} q-points to "
                    f"{self.qgrid.num_irreducible_points} irreducible q-points.",
                    flush=True,
                )
            else:
                print(
                    f"Using full q-point grid with {self.qgrid.num_points} q-points.",
                    flush=True,
                )

        self.hamiltonians, self.overlap_matrices, hamiltonian_sparsity_pattern = (
            self._init_hamiltonian()
        )
        self.block_sizes = self.hamiltonians.block_sizes

        self._allocate_sparsity_pattern()
        # NOTE: Maybe not the greatest that the subsystem specific
        # things are allocated in the device. Previously, they were
        # allocated in the main SCBA class.
        if self.config.scba.coulomb_screening:
            self.coulomb_num_connected_blocks, self.coulomb_block_sizes = (
                self._coulomb_block_sizes()
            )
            self.coulomb_matrix = self._init_coulomb_matrix()

        if self.uses_packing:
            self._check_symmetry()

        # NOTE: Contacts are added at the end since the need parameters
        # from the previous method.
        self._add_contacts(hamiltonian_sparsity_pattern)

    @property
    def uses_packing(self) -> bool:
        """Whether only the symmetry-unique elements are used to compute
        the interactions."""
        return self.orbital_operations is not None

    def _create_kgrid(self) -> KGrid:
        """Creates the k-point grid.

        If the orbitals are known, the symmetry operations of the device
        are limited to the ones that also apply to the interactions.

        """
        self.orbital_operations = None
        device_config = self.device_config
        if not device_config.kpoint_symmetry_reduction:
            return super()._create_kgrid()

        if device_config.orbital_labels is None:
            kgrid = super()._create_kgrid()
            if comm.rank == 0 and kgrid.num_points < kgrid.num_full_points:
                warnings.warn(
                    "The matrices at symmetry-equivalent k-points are assumed to be "
                    "identical, which is not true in general. Set "
                    "`device.orbital_labels` to account for the symmetry exactly, or "
                    "disable `device.kpoint_symmetry_reduction`."
                )
            return kgrid

        if not self.config.scba.symmetric:
            raise ValueError("`device.orbital_labels` requires `scba.symmetric`.")
        if comm.block.size > 1:
            raise NotImplementedError(
                "`device.orbital_labels` is not implemented for domain-distributed "
                "solvers."
            )

        transport_ind = "abc".index(device_config.transport_direction)
        symmetry_operations, orbital_operations = find_orbital_operations(
            self.structure, device_config.orbital_labels, transport_ind
        )
        kgrid = KGrid(
            size=device_config.kpoint_grid,
            shift=device_config.kpoint_shift,
            symmetry_reduction=True,
            operations=symmetry_operations,
        )
        num_repetitions = 1
        if device_config.construct_from_unit_cell:
            num_repetitions = (
                device_config.num_transport_cells
                * device_config.neighbor_cell_cutoff[transport_ind]
            )
        # Not all of the operations have to preserve the grid.
        self.orbital_operations = [
            repeat_orbital_operation(orbital_operation, num_repetitions)
            for symmetry_operation, orbital_operation in zip(
                symmetry_operations, orbital_operations
            )
            if any(symmetry_operation is op for op in kgrid.operations)
        ]
        return kgrid

    @profiler.profile("Device: Check symmetry", level="default", comm=comm)
    def _check_symmetry(self) -> None:
        """Checks that the matrices are symmetric.

        For this, the matrices are assembled at all the images of a
        single k-point that is not special.

        """
        size = np.asarray(self.kgrid.size)
        kpoint = np.where(size > 1, [0.1372, 0.2158, 0.0794], 0.0)
        kpoints = np.array(
            [kpoint @ np.linalg.inv(op.rotation) for op in self.orbital_operations]
        )
        star = SimpleNamespace(
            size=tuple(size), points=kpoints, num_points=len(kpoints)
        )

        names = {"hamiltonian": (True, None)}
        if self.overlap_matrices is not None:
            names["overlap"] = (True, None)
        if self.config.scba.coulomb_screening:
            names["coulomb_matrix"] = (False, self.sparsity_pattern)

        for name, (signed, sparsity_pattern) in names.items():
            matrix, __ = assemble_matrix(
                config=self.config,
                matrix_name=name,
                kgrid=star,
                sparsity_pattern=sparsity_pattern,
            )
            deviation = measure_symmetry_breaking(
                matrix.data[0],
                *matrix.spy(),
                kpoints,
                self.orbital_operations,
                signed,
            )
            self._check_deviation(name, deviation)

        if comm.rank == 0:
            print(
                f"Using {len(self.orbital_operations)} symmetry operations.", flush=True
            )

    def _check_deviation(self, name: str, deviation: float) -> None:
        """Checks the deviation from the symmetry."""
        tolerance = self.device_config.kpoint_symmetry_tolerance
        if deviation > tolerance:
            raise ValueError(
                f"The {name} breaks the symmetry by {deviation:.2e}, which is more "
                f"than `device.kpoint_symmetry_tolerance` = {tolerance:.2e}. Either "
                "the `device.orbital_labels` are wrong, or "
                "`device.kpoint_symmetry_reduction` has to be disabled."
            )

    def check_potential(self, potential: NDArray) -> None:
        """Checks that the potential is symmetric.

        Parameters
        ----------
        potential : NDArray
            The potential of each orbital.

        """
        potential = get_host(potential)
        deviation = max(
            float(np.abs(potential[op.targets] - potential).max())
            for op in self.orbital_operations
        )
        self._check_deviation("potential", deviation)

    def _add_contacts(self, hamiltonian_sparsity_pattern: sparse.coo_matrix):
        """Initializes and attaches contacts to the device.

        Parameters
        ----------
        hamiltonian_sparsity_pattern : sparse.coo_matrix
            Sparsity pattern of the device Hamiltonian, used to
            determine the periodicity of the contacts and their coupling
            to the device.

        Creates Contact objects for each contact defined in the device
        configuration. Each contact represents a semi-infinite lead
        connected to the finite device region, providing boundary
        conditions for transport calculations.

        """
        for contact_config in self.device_config.contacts:
            # NOTE: Conversion to CSR is not efficient, but needed
            # matrix is sliced.
            self.contacts.append(
                SCBAContact(
                    device=self,
                    contact_config=contact_config,
                    sparsity_pattern=hamiltonian_sparsity_pattern.tocsr(),
                )
            )

        if comm.rank == 0:
            print(
                f"Device initialized with {len(self.contacts)} contacts.",
                flush=True,
            )

    def _init_hamiltonian(
        self,
    ) -> tuple[DSDBSparse, DSDBSparse | None, sparse.coo_matrix]:
        """Initializes Hamiltonian and overlap matrices from files."""

        # Load the device Hamiltonian.
        hamiltonians, hamiltonian_sparsity_pattern = assemble_matrix(
            config=self.config,
            matrix_name="hamiltonian",
            kgrid=self.kgrid,
            sparsity_pattern=None,
        )

        try:
            # Attempt to load the device overlap matrix.
            overlap_matrices, __ = assemble_matrix(
                config=self.config,
                matrix_name="overlap",
                kgrid=self.kgrid,
                sparsity_pattern=None,
            )

            # Check that the overlap matrix and Hamiltonian matrix match.
            if overlap_matrices.shape != hamiltonians.shape:
                raise ValueError(
                    "Overlap matrix and Hamiltonian matrix have different shapes."
                )

            if comm.rank == 0:
                print("Non-orthogonal basis detected.", flush=True)

        except FileNotFoundError:
            overlap_matrices = None
            if comm.rank == 0:
                print("No overlap matrix found. Assuming orthogonal basis.", flush=True)

        return hamiltonians, overlap_matrices, hamiltonian_sparsity_pattern

    @profiler.profile("Device: Sparsity Pattern", level="default", comm=comm)
    def _allocate_sparsity_pattern(self):
        """Allocates the sparsity pattern."""
        max_interaction_cutoff = 0.0
        if self.config.scba.coulomb_screening:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                self.config.coulomb_screening.interaction_cutoff,
            )
        if self.config.scba.photon:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                self.config.photon.interaction_cutoff,
            )
        if self.config.scba.phonon:
            max_interaction_cutoff = max(
                max_interaction_cutoff,
                self.config.phonon.interaction_cutoff,
            )
        if max_interaction_cutoff == 0.0:
            raise NotImplementedError(
                "At least one interaction must be enabled in the SCBA."
                "Ballistic transport is not properly supported yet."
            )

        if comm.rank == 0:
            print(f"Max Interaction Cutoff: {max_interaction_cutoff}", flush=True)

        # Determine the local slice of the data.
        # NOTE: This is arrow-wise partitioning.
        # TODO: Allow more options, e.g., block row-wise partitioning.
        section_sizes, __ = get_section_sizes(len(self.block_sizes), comm.block.size)
        section_offsets = np.hstack(([0], np.cumsum(section_sizes)))
        block_offsets = np.hstack(([0], np.cumsum(self.block_sizes)))
        start_idx = block_offsets[section_offsets[comm.block.rank]]
        end_idx = block_offsets[section_offsets[comm.block.rank + 1]]

        # NOTE: The sparsity pattern is kept in memory.
        # TODO: Potentially could be deallocated or put on the host.
        self.sparsity_pattern = compute_sparsity_pattern(
            self.orbital_coordinates,
            max_interaction_cutoff,
            transport_direction=self.device_config.transport_direction,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    def _coulomb_block_sizes(self) -> tuple[int, NDArray]:
        """Computes the block sizes for the Coulomb Screening."""
        coulomb_num_connected_blocks = (
            self.config.coulomb_screening.num_connected_blocks
        )
        if coulomb_num_connected_blocks == "auto":
            coulomb_num_connected_blocks = compute_num_connected_blocks(
                self.sparsity_pattern, self.block_sizes
            )

        if comm.rank == 0:
            print(
                f"Number of connected blocks: {coulomb_num_connected_blocks}",
                flush=True,
            )

        if len(self.block_sizes) % coulomb_num_connected_blocks != 0:
            # Not implemented yet.
            raise ValueError(
                f"Number of blocks must be divisible by {coulomb_num_connected_blocks}."
            )

        coulomb_block_sizes = (
            self.block_sizes[: len(self.block_sizes) // coulomb_num_connected_blocks]
            * coulomb_num_connected_blocks
        )

        # Check that the provided block sizes match the Hamiltonian.
        if coulomb_block_sizes.sum() != self.hamiltonians.shape[-2]:
            raise ValueError(
                "Block sizes do not match Hamiltonian. "
                f"{coulomb_block_sizes.sum()} != {self.hamiltonians.shape[-2]}"
            )

        return coulomb_num_connected_blocks, coulomb_block_sizes

    def _init_coulomb_matrix(self) -> DSDBSparse:
        """Initializes the Coulomb matrix."""
        # Load the Coulomb matrix.
        coulomb_matrix, __ = assemble_matrix(
            config=self.config,
            matrix_name="coulomb_matrix",
            kgrid=self.qgrid,
            sparsity_pattern=self.sparsity_pattern,
        )

        # Make sure the Coulomb matrix is hermitian.
        # TODO: Check that this is correct for kpoints.
        if coulomb_matrix.symmetry is None:
            coulomb_matrix.symmetrize("hermitian")

        coulomb_matrix.data /= self.config.coulomb_screening.epsilon_r

        return coulomb_matrix
