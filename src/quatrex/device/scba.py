# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the SCBA device class for electronic transport calculations."""

import numpy as np

from qttools import NDArray, sparse
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.profiling import Profiler
from qttools.utils.mpi_utils import get_section_sizes
from quatrex.contact.scba import SCBAContact
from quatrex.core.config import QuatrexConfig
from quatrex.core.utils import compute_num_connected_blocks, compute_sparsity_pattern
from quatrex.device.base import BaseDevice
from quatrex.device.inputs import assemble_matrix

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

    """

    def __init__(self, config: QuatrexConfig) -> None:
        super().__init__(config)
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

        # NOTE: Contacts are added at the end since the need parameters
        # from the previous method.
        self._add_contacts(hamiltonian_sparsity_pattern)

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
            sparsity_pattern=None,
            shift_kpoints=False,
        )

        try:
            # Attempt to load the device overlap matrix.
            overlap_matrices, __ = assemble_matrix(
                config=self.config,
                matrix_name="overlap",
                sparsity_pattern=None,
                shift_kpoints=False,
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
            sparsity_pattern=self.sparsity_pattern,
            shift_kpoints=True,
        )

        # Make sure the Coulomb matrix is hermitian.
        # TODO: Check that this is correct for kpoints.
        if coulomb_matrix.symmetry is None:
            coulomb_matrix.symmetrize("hermitian")

        coulomb_matrix.data /= self.config.coulomb_screening.epsilon_r

        return coulomb_matrix
