import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, sparse, xp
from qttools.utils.mpi_utils import distributed_load
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.contact import Contact
from quatrex.core.quatrex_config import QuatrexConfig


def get_orbital_potential(potential: NDArray, orbital_offsets: NDArray) -> NDArray:
    """Converts atom-resolved potential to orbital-resolved potential.

    Parameters
    ----------
    potential : NDArray
        Electrostatic potential at each atomic site.
    orbital_offsets : NDArray
        Cumulative orbital count array where orbital_offsets[i] gives
        the starting orbital index for atom i.

    Returns
    -------
    NDArray
        Electrostatic potential for each orbital.

    """
    orbitals_per_atom = list(np.diff(orbital_offsets))
    orbital_potential = xp.repeat(potential, orbitals_per_atom, axis=0)
    return orbital_potential


def distributed_read_xyz(filename: Path) -> tuple[NDArray, NDArray, NDArray]:
    """Reads atomic structure data from an XYZ file.

    Parameters
    ----------
    filename : Path
        Path to the XYZ file containing the atomic structure. The file
        should have the standard XYZ format with lattice parameters on
        the second line.

    Returns
    -------
    lattice : NDArray
        3x3 array containing the lattice vectors (in rows).
    atom_coordinates : NDArray
        (N_atoms, 3) array containing atomic coordinates.
    atom_types : NDArray
        (N_atoms,) array containing atom symbol for each atom.

    """

    lattice = None
    atom_coordinates = None
    atom_types = None

    if comm.rank == 0:
        # Read only the second line of the file (this contains the
        # lattice parameters)
        with open(filename, "r") as f:
            __ = f.readline()
            lattice_line = f.readline().strip()

        if not lattice_line.startswith("Lattice="):
            raise ValueError(
                f"Invalid lattice line in {filename}. Expected 'Lattice=', got '{lattice_line}'"
            )

        lattice = lattice_line.split("=")[1].strip().split('"')[1]
        lattice = np.fromstring(lattice, dtype=np.float64, sep=" ").reshape(3, 3)
        atom_coordinates = np.loadtxt(filename, skiprows=2, usecols=(1, 2, 3))
        atom_types = np.loadtxt(filename, skiprows=2, usecols=(0,), dtype=str)

    # Broadcast the data to all the ranks
    lattice = comm.bcast(lattice, root=0)
    atom_coordinates = comm.bcast(atom_coordinates, root=0)
    atom_types = comm.bcast(atom_types, root=0)

    return lattice, atom_coordinates, atom_types


class Device:
    """A quantum device for electronic transport calculations.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        Configuration object containing input paths, device parameters,
        and computational settings.
    compute_config : ComputeConfig
        Configuration object specifying computational resources and
        parallelization settings.

    Attributes
    ----------
    quatrex_config : QuatrexConfig
        Reference to the configuration object.
    compute_config : ComputeConfig
        Reference to the compute configuration object.
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
    lattice_vector : NDArray
        3x3 array containing the lattice vectors of the device unit
        cell.
    atom_coordinates : NDArray
        Array of atomic coordinates.
    atom_type : NDArray
        Array of atom symbols for each atom.
    orbital_offsets : NDArray
        Array of cumulative orbital counts, used to map from atoms to
        orbitals. orbital_offsets[i] gives the starting orbital index
        for atom i.
    orbital_potential : NDArray, optional
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

    def __init__(
        self, quatrex_config: QuatrexConfig, compute_config: ComputeConfig
    ) -> None:
        """Initializes a Device object from configuration."""

        self.quatrex_config = quatrex_config
        self.compute_config = compute_config

        self._init_hamiltonian()
        self._init_lattice()
        self._init_orbitals()
        self._load_potential()
        self.apply_potential()
        self._add_contacts()

        if comm.rank == 0:
            print(
                f"Device initialized with {len(self.contacts)} contacts.",
                flush=True,
            )

    def _init_hamiltonian(self) -> None:
        """Initializes Hamiltonian and overlap matrices from files.

        Loads sparse matrices from .npz files in the input directory.
        Files should be named "hamiltonian_i_j_k.npz" and
        "overlap_i_j_k.npz" where (i,j,k) represent lattice vector
        indices. The method automatically discovers all available matrix
        files and loads them into dictionaries.

        For missing overlap matrices, identity matrices are assumed
        (orthogonal basis). The (0,0,0) Hamiltonian matrix is mandatory
        and its absence raises an error.

        """

        self.hamiltonians = {}
        self.overlap_matrices = {}
        self.gamma_only = False

        if not (self.quatrex_config.input_dir / "hamiltonian_0_0_0.npz").exists():
            raise ValueError("Hamiltonian matrix for (0,0,0) not found.")

        # Load all Hamiltonian files with format hamiltonian_x_y_z.npz
        # Find all hamiltonian files in the input directory
        hamiltonian_paths = self.quatrex_config.input_dir.glob("hamiltonian_*_*_*.npz")

        # Parse indices from filenames and load files
        for hamiltonian_path in hamiltonian_paths:

            indices = tuple(map(int, hamiltonian_path.stem.split("_")[1:]))

            self.hamiltonians[indices] = distributed_load(hamiltonian_path).tocsr()
            assert (
                self.hamiltonians[indices].shape[0]
                == self.hamiltonians[indices].shape[1]
            ), f"Hamiltonian matrix at {hamiltonian_path} is not square."

            # TODO: Check data type handling.
            # Gamma point can be real depending on the basis.
            if not all(index == 0 for index in indices):
                self.hamiltonians[indices] = self.hamiltonians[indices].astype(
                    xp.complex128
                )

            if not self.hamiltonians[indices].has_canonical_format:
                self.hamiltonians[indices].sum_duplicates()
                self.hamiltonians[indices].sort_indices()

            overlap_path = (
                self.quatrex_config.input_dir
                / f"overlap_{indices[0]}_{indices[1]}_{indices[2]}.npz"
            )

            if overlap_path.exists():
                self.overlap_matrices[indices] = distributed_load(overlap_path).tocsr()
                assert (
                    self.overlap_matrices[indices].shape[0]
                    == self.overlap_matrices[indices].shape[1]
                ), f"Overlap matrix at {overlap_path} is not square."

                if not all(index == 0 for index in indices):
                    self.overlap_matrices[indices] = self.overlap_matrices[
                        indices
                    ].astype(xp.complex128)

                if not self.overlap_matrices[indices].has_canonical_format:
                    self.overlap_matrices[indices].sum_duplicates()
                    self.overlap_matrices[indices].sort_indices()

        # TODO: Mechanism to handle orthogonal basis sets.
        if len(self.overlap_matrices) == 0:
            # NOTE: Dangerous to automatically assume identity matrix.
            # Better to add a config option to specify orthogonal basis.
            if comm.rank == 0:
                warnings.warn(
                    "No overlap matrices found. Assuming identity matrix.",
                )
            size = self.hamiltonians[0, 0, 0].shape[0]
            self.overlap_matrices[0, 0, 0] = sparse.eye(
                size, dtype=xp.complex128, format="csr"
            )

        elif len(self.overlap_matrices) < len(self.hamiltonians):
            raise ValueError(
                "Some overlap matrices are missing while others are present. All or none must be provided."
            )

        if comm.rank == 0:
            print(f"Loaded {len(self.hamiltonians)} Hamiltonian matrices", flush=True)
            print(f"Loaded {len(self.overlap_matrices)} overlap matrices", flush=True)

        if len(self.hamiltonians) == 1:
            self.gamma_only = True

    def _init_lattice(self) -> None:
        """Initializes the atomic structure and lattice parameters of
        the device."""

        # Load the lattice structure from file.
        lattice_file = self.quatrex_config.input_dir / "lattice.xyz"
        if not lattice_file.exists():
            raise FileNotFoundError(f"Lattice file {lattice_file} not found.")
        self.lattice_vector, self.atom_coordinates, self.atom_type = (
            distributed_read_xyz(lattice_file)
        )

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
                defaultdict(
                    lambda: 1, self.quatrex_config.device.num_orbitals_per_atom
                ).get,
                self.atom_type,
            ),
            dtype=np.int32,
        )
        # Create a vector with the starting orbital for each atom
        self.orbital_offsets = np.hstack(([0], np.cumsum(orbitals_per_atom)))

    def _load_potential(self) -> None:
        """Loads electrostatic potential data from input files.

        Attempts to load the electrostatic potential from potential.npy
        in the input directory. The potential can be provided either
        at the atomic level or at the orbital level.

        """

        self.orbital_potential = None

        try:
            potential = distributed_load(
                self.quatrex_config.input_dir / "potential.npy"
            )

            if potential.shape[0] == self.atom_coordinates.shape[0]:
                # Upscale the potential to the number of orbitals
                self.orbital_potential = get_orbital_potential(
                    potential, self.orbital_offsets
                )
            elif potential.shape[0] == self.orbital_offsets[-1]:
                self.orbital_potential = potential
            else:
                raise ValueError(
                    "Potential shape does not match number of atoms or orbitals."
                )

        except FileNotFoundError:
            if comm.rank == 0:
                print("No external potential is provided.", flush=True)

    # TODO: THis should probably not happen directly in the Hamiltonian,
    # but rather during the construction of the system matrix.
    def apply_potential(self) -> None:
        """Applies electrostatic potential to device Hamiltonian."""

        if self.orbital_potential is None:
            if comm.rank == 0:
                print(
                    "No potential loaded. Skipping potential application.", flush=True
                )
            return

        potential = self.orbital_potential + 1e-10

        for lattice_index, overlap in self.overlap_matrices.items():
            self.hamiltonians[lattice_index] += (
                overlap.multiply(potential[:, np.newaxis]) + overlap.multiply(potential)
            ) / 2

    def _add_contacts(self):
        """Initializes and attaches contacts to the device.

        Creates Contact objects for each contact defined in the device
        configuration. Each contact represents a semi-infinite lead
        connected to the finite device region, providing boundary
        conditions for transport calculations.

        """

        contacts = []
        for contact_config in self.quatrex_config.device.contacts:
            contacts.append(
                Contact(
                    device=self,
                    name=contact_config.name,
                    origin=contact_config.origin,
                    lattice_vectors=contact_config.size,
                    direction=contact_config.direction,
                    fermi_level=contact_config.fermi_level,
                )
            )

        self.contacts = contacts
