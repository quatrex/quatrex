# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, sparse, xp
from qttools.utils.gpu_utils import get_host
from qttools.utils.mpi_utils import distributed_load
from quatrex.core.config import QuatrexConfig
from quatrex.device.contact import Contact
from quatrex.device.inputs import (
    create_coordinate_grid,
    distributed_read_xyz,
    load_matrices,
)


class Device:
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
    hamiltonians : dict
        Dictionary of Hamiltonian matrices indexed by (i, j, k) lattice
        vectors. Keys are tuples representing the lattice vector
        indices, values are sparse CSR matrices.
    overlap_matrices : dict
        Dictionary of overlap matrices with the same indexing as
        hamiltonians. For orthogonal basis sets, defaults to identity
        matrices.
    gamma_only : bool
        True if only the Gamma point (0,0,0) Hamiltonian is available,
        indicating that k-point calculations are not possible.
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
    contacts : list[Contact]
        List of Contact objects representing the semi-infinite leads
        connected to this device.

    Methods
    -------
    apply_potential()
        Apply the electrostatic potential to the Hamiltonian matrices.

    """

    def __init__(self, config: QuatrexConfig) -> None:
        """Initializes a Device object from configuration."""

        self.config = config

        self._init_hamiltonian()
        __, self.atom_coordinates, self.atomic_species = self.load_structure(config)
        # TODO QTBM Device/Contact currently assumes that these quantities are on the host
        self.atom_coordinates = get_host(self.atom_coordinates)

        self._init_orbitals()
        self.potential = self.load_potential(
            self.config.input_dir,
            self.atom_coordinates,
            self.atomic_species,
            self.config.device.num_orbitals_per_atom,
        )
        self.apply_potential()
        self._add_contacts()

        if comm.rank == 0:
            print(
                f"Device initialized with {len(self.contacts)} contacts.",
                flush=True,
            )

    @staticmethod
    def load_potential(
        input_dir: Path,
        atom_coordinates: NDArray,
        atomic_species: NDArray,
        num_orbitals_per_atom: dict[str, int],
    ) -> NDArray:
        """Loads electrostatic potential data from input files.

        Attempts to load the electrostatic potential from potential.npy in the
        input directory. The potential can be provided either at the atomic
        level or at the orbital level.

        Parameters
        ----------
        input_dir : Path
            Directory containing the `potential.npy` file.
        atom_coordinates : NDArray
            Array of atomic coordinates.
        atomic_species : NDArray
            Array of atom symbols for each atom. NOTE: This array is always on the
            host since CuPy does not support string arrays.
        num_orbitals_per_atom : dict[str, int]
            Dictionary mapping atomic species to the number of orbitals per
            atom.

        Returns
        -------
        NDArray
            The electrostatic potential array. If no potential file is found,
            returns an array of zeros with length equal to the total number of
            orbitals.


        """
        orbitals_per_atom = [
            num_orbitals_per_atom.get(species, 1) for species in atomic_species
        ]
        num_orbitals = np.sum(np.array(orbitals_per_atom))

        try:
            potential = distributed_load(input_dir / "potential.npy")

            # NOTE: If atom species is only 'X', then we still repeat.
            if potential.shape[0] == atom_coordinates.shape[0]:
                # Upscale the potential to the number of orbitals
                potential = xp.repeat(potential, orbitals_per_atom, axis=0)
            elif potential.shape[0] != num_orbitals:
                raise ValueError(
                    "Potential shape does not match number of atoms or orbitals."
                )

        except FileNotFoundError:
            potential = xp.zeros(num_orbitals)

        return potential

    @staticmethod
    def load_structure(
        config: QuatrexConfig,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Loads the orbital coordinates, atom coordinates, and atomic
        species for the device.

        Parameters
        ----------
        config : QuatrexConfig
            The Quatrex configuration.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
            The orbital coordinates, atom coordinates, and atomic
            species.

        """

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
        orbital_coordinates = xp.repeat(atom_coordinates, orbitals_per_atom, axis=0)

        if config.device.construct_from_unit_cell:

            transport_ind = "xyz".index(config.device.transport_direction)

            orbital_coordinates = create_coordinate_grid(
                orbital_coordinates,
                config.device.num_transport_cells
                * config.device.neighbor_cell_cutoff[transport_ind],
                transport_ind,
                xp.asarray(lattice_vectors),
            )

            atom_coordinates = create_coordinate_grid(
                atom_coordinates,
                config.device.num_transport_cells
                * config.device.neighbor_cell_cutoff[transport_ind],
                transport_ind,
                xp.asarray(lattice_vectors),
            )

            atomic_species = np.concatenate(
                [atomic_species]
                * config.device.neighbor_cell_cutoff[transport_ind]
                * config.device.num_transport_cells
            )

        return orbital_coordinates, atom_coordinates, atomic_species

    def _init_hamiltonian(self) -> None:
        """Initializes Hamiltonian and overlap matrices from files.

        Loads sparse matrices from .mat files in the input directory.
        Files should be named "hamiltonian.mat" and
        "overlap.mat" where the keys are strings of [i,j,k] represent
        lattice vector indices.

        For missing overlap matrices, identity matrices are assumed
        (orthogonal basis). The (0,0,0) Hamiltonian matrix is mandatory
        and its absence raises an error.

        """

        self.gamma_only = False

        if not (self.config.input_dir / "hamiltonian.mat").exists():
            raise ValueError("Hamiltonian matrix not found.")

        # NOTE: Contains only the upper triangular parts of the matrices.
        hamiltonians = load_matrices(self.config, "hamiltonian")

        self.hamiltonians = {}
        for r, h_r in hamiltonians.items():
            # TODO: Check data type handling.
            # Gamma point can be real depending on the basis.
            # TODO: Do not unsymmetrize
            flipped_r = tuple(-i for i in r)
            hamiltonian_flipped = hamiltonians[flipped_r]
            self.hamiltonians[r] = (
                hamiltonians[r] + sparse.triu(hamiltonian_flipped, k=1).conj().T
            )
        for r in hamiltonians.keys():
            self.hamiltonians[r] = sparse.csr_matrix(self.hamiltonians[r])

        size = self.hamiltonians[(0, 0, 0)].shape[0]

        if (self.config.input_dir / "overlap.mat").exists():
            overlap_matrices = load_matrices(self.config, "overlap")

            self.overlap_matrices = {}
            for r, s_r in overlap_matrices.items():
                assert (
                    s_r.shape[0] == size
                ), f"Overlap matrix at index {r} has incompatible size with Hamiltonian. Expected {size}, got {s_r.shape[0]}."

                assert (
                    s_r.shape[1] == size
                ), f"Overlap matrix at index {r} has incompatible size with Hamiltonian. Expected {size}, got {s_r.shape[1]}."

                # TODO: Check data type handling.
                # Gamma point can be real depending on the basis.
                # TODO: Do not unsymmetrize
                flipped_r = tuple(-i for i in r)
                overlap_matrix_flipped = overlap_matrices[flipped_r]
                self.overlap_matrices[r] = (
                    overlap_matrices[r]
                    + sparse.triu(overlap_matrix_flipped, k=1).conj().T
                )

            if len(self.overlap_matrices) < len(self.hamiltonians):
                raise ValueError(
                    "Some overlap matrices are missing while others are present. All or none must be provided."
                )
            for r in overlap_matrices.keys():
                self.overlap_matrices[r] = sparse.csr_matrix(self.overlap_matrices[r])

        else:
            if comm.rank == 0:
                warnings.warn(
                    "No overlap matrices found. Assuming identity matrix.",
                )
            self.overlap_matrices = {
                (0, 0, 0): sparse.eye(size, dtype=xp.complex128, format="csr")
            }

        if comm.rank == 0:
            print(f"Loaded {len(self.hamiltonians)} Hamiltonian matrices", flush=True)
            print(f"Loaded {len(self.overlap_matrices)} overlap matrices", flush=True)

        if len(self.hamiltonians) == 1:
            self.gamma_only = True

    def _init_orbitals(self) -> None:
        """Initializes the orbital indexing system for the device.

        Sets up the mapping between atoms and orbitals by determining
        how many orbitals each atom has and creating cumulative indexing
        arrays.

        The number of orbitals per atom is determined from the
        configuration, with a default of 1 orbital per atom if not
        specified.

        """
        orbitals_per_atom = np.fromiter(
            map(
                defaultdict(lambda: 1, self.config.device.num_orbitals_per_atom).get,
                self.atomic_species,
            ),
            dtype=np.int32,
        )
        # Create a vector with the starting orbital for each atom
        self.orbital_offsets = np.hstack(([0], np.cumsum(orbitals_per_atom)))

    # TODO: THis should probably not happen directly in the Hamiltonian,
    # but rather during the construction of the system matrix.
    def apply_potential(self) -> None:
        """Applies electrostatic potential to device Hamiltonian."""

        potential = self.potential + 1e-10

        for r, s_r in self.overlap_matrices.items():
            self.hamiltonians[r] += (
                s_r.multiply(potential[:, xp.newaxis]) + s_r.multiply(potential)
            ) / 2

    def _add_contacts(self):
        """Initializes and attaches contacts to the device.

        Creates Contact objects for each contact defined in the device
        configuration. Each contact represents a semi-infinite lead
        connected to the finite device region, providing boundary
        conditions for transport calculations.

        """

        contacts = []
        for contact_config in self.config.device.contacts:
            contacts.append(
                Contact(
                    device=self,
                    name=contact_config.name,
                    origin=contact_config.origin,
                    lattice_vectors=contact_config.lattice_vectors,
                    direction=contact_config.direction,
                    fermi_level=contact_config.fermi_level,
                )
            )

        self.contacts = contacts
