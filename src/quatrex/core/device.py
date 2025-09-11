from collections import defaultdict
from pathlib import Path

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, sparse, xp
from qttools.utils.mpi_utils import distributed_load
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


def distributed_read_xyz(filename: Path) -> tuple[NDArray, list, NDArray, NDArray]:
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
    unique_kinds : list
        List of unique atom symbols/types present in the structure.
    coords : NDArray
        (N_atoms, 3) array containing atomic coordinates.
    atom_types : NDArray
        (N_atoms,) array containing atom symbol for each atom.

    """

    lattice = None
    coords = None
    kinds = None

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

        lattice = lattice_line.split("=")[1].strip('"')
        lattice = np.fromstring(lattice, dtype=np.float64, sep=" ").reshape(3, 3)
        coords = np.loadtxt(filename, skiprows=2, usecols=(1, 2, 3))
        kinds = np.loadtxt(filename, skiprows=2, usecols=(0,), dtype=str)

    # Broadcast the data to all the ranks
    lattice = comm.bcast(lattice, root=0)
    coords = comm.bcast(coords, root=0)
    kinds = comm.bcast(kinds, root=0)

    return lattice, np.unique(kinds), coords, kinds


class Device:
    """A quantum device for electronic transport calculations.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        Configuration object containing input paths, device parameters,
        and computational settings.

    Attributes
    ----------
    config : QuatrexConfig
        Reference to the configuration object.
    hamiltonian : dict
        Dictionary of Hamiltonian matrices indexed by (i, j, k) lattice
        vectors. Keys are tuples representing the lattice vector
        indices, values are sparse CSR matrices.
    overlap : dict
        Dictionary of overlap matrices with the same indexing as
        hamiltonian. For orthogonal basis sets, defaults to identity
        matrices.
    num_hamiltonians : int
        Total number of Hamiltonian matrices loaded from input files.
    num_overlaps : int
        Total number of overlap matrices loaded from input files.
    gamma_only : bool
        True if only the Gamma point (0,0,0) Hamiltonian is available,
        indicating that k-point calculations are not possible.
    lattice_vector : NDArray
        3x3 array containing the lattice vectors of the device unit
        cell.
    atoms_list : list
        List of unique atom symbols present in the device.
    coords : NDArray
        Array of atomic coordinates.
    atom_type : NDArray
        Array of atom symbols for each atom.
    orbital_offsets : NDArray
        Array of cumulative orbital counts, used to map from atoms to
        orbitals. orbital_offsets[i] gives the starting orbital index
        for atom i.
    atom_potential : NDArray, optional
        Array of electrostatic potential at each atom site. None if no
        external potential is provided.
    orbital_potential : NDArray, optional
        Array of electrostatic potential for each orbital. Derived from
        atom_potential by replication according to orbitals per atom.
    contacts : list[Contact]
        List of Contact objects representing the semi-infinite leads
        connected to this device.

    Methods
    -------
    apply_potential()
        Apply the electrostatic potential to the Hamiltonian matrices.

    """

    def __init__(self, quatrex_config: QuatrexConfig) -> None:
        """Initializes a Device object from configuration."""

        self.config = quatrex_config

        self._init_hamiltonian()
        self._init_lattice()
        self._init_orbitals()
        self._load_potential()
        self.apply_potential()
        self.contacts = self._add_contacts()

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

        self.hamiltonian = {}
        self.overlap = {}
        self.gamma_only = False

        # Load all Hamiltonian files with format hamiltonian_x_y_z.npz
        # Find all hamiltonian files in the input directory
        hamiltonian_paths = self.config.input_dir.glob("hamiltonian_*_*_*.npz")

        # Parse indices from filenames and load files
        for hamiltonian_path in hamiltonian_paths:
            x_index, y_index, z_index = map(int, hamiltonian_path.stem.split("_")[1:])
            try:
                self.hamiltonian[x_index, y_index, z_index] = (
                    distributed_load(hamiltonian_path).astype(xp.complex128).tocsr()
                )
                if comm.rank == 0:
                    print(
                        f"Loaded Hamiltonian ({x_index} {y_index} {z_index})",
                        flush=True,
                    )

            except Exception as e:
                if comm.rank == 0:
                    print(f"Failed to load {hamiltonian_path.stem}: {e}", flush=True)

            overlap_path = (
                self.config.input_dir / f"overlap_{x_index}_{y_index}_{z_index}.npz"
            )
            # TODO: Mechanism to handle orthogonal basis sets.
            try:
                self.overlap[x_index, y_index, z_index] = (
                    distributed_load(overlap_path).astype(xp.complex128).tocsr()
                )
                if comm.rank == 0:
                    print(f"Loaded Overlap ({x_index} {y_index} {z_index})", flush=True)
            except Exception as e:
                if comm.rank == 0:
                    print(f"Failed to load {overlap_path.stem}: {e}", flush=True)

        if (0, 0, 0) not in self.hamiltonian:
            raise ValueError(
                "Hamiltonian matrix for (0,0,0) not found. Please check the input files."
            )

        self.num_hamiltonians = len(self.hamiltonian)
        if comm.rank == 0:
            print(f"Loaded {self.num_hamiltonians} Hamiltonian matrices", flush=True)

        if self.num_hamiltonians == 1:
            if comm.rank == 0:
                print(
                    "Only Gamma point Hamiltonian found. K-Points calculations are not possible.",
                    flush=True,
                )
            self.gamma_only = True

        if (0, 0, 0) not in self.overlap:
            if comm.rank == 0:
                print(
                    "Overlap matrix for (0,0,0) not found. Assuming identity matrix.",
                    flush=True,
                )
            self.overlap[(0, 0, 0)] = sparse.eye(
                self.hamiltonian[0, 0, 0].shape[0], dtype=xp.complex128, format="csr"
            )

        self.num_overlaps = len(self.overlap)
        if comm.rank == 0:
            print(f"Loaded {self.num_overlaps} overlap matrices", flush=True)

    def _init_lattice(self) -> None:
        """Initializes the atomic structure and lattice parameters of
        the device."""

        # Load the lattice structure from file.
        lattice_file = self.config.input_dir / "lattice.xyz"
        if not lattice_file.exists():
            raise FileNotFoundError(f"Lattice file {lattice_file} not found.")
        self.lattice_vector, self.atoms_list, self.coords, self.atom_type = (
            distributed_read_xyz(lattice_file)
        )
        if comm.rank == 0:
            print("Lattice structure loaded successfully.", flush=True)

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
                self.atom_type,
            ),
            dtype=np.int32,
        )
        # Create a vector with the starting orbital for each atom
        self.orbital_offsets = np.hstack(([0], np.cumsum(orbitals_per_atom)))

    def _load_potential(self) -> None:
        """Loads electrostatic potential data from input files.

        Attempts to load the electrostatic potential from potential.npy
        in the input directory. If found, the atom-resolved potential is
        converted to orbital-resolved potential.

        """

        self.orbital_potential = None

        try:
            self.atom_potential = distributed_load(
                self.config.input_dir / "potential.npy"
            )
            # Upscale the potential to the number of orbitals
            self.orbital_potential = get_orbital_potential(
                self.atom_potential, self.orbital_offsets
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

        for r, s_r in self.overlap.items():
            SV1 = s_r.multiply(self.orbital_potential).tocsr()
            SV2 = s_r.multiply(self.orbital_potential.T).tocsr()
            self.hamiltonian[r] += (SV1 + SV2) / 2
            self.hamiltonian[r].eliminate_zeros()

    def _add_contacts(self) -> list[Contact]:
        """Initializes and attaches contacts to the device.

        Creates Contact objects for each contact defined in the device
        configuration. Each contact represents a semi-infinite lead
        connected to the finite device region, providing boundary
        conditions for transport calculations.

        Returns
        -------
        list[Contact]
            List of initialized Contact objects, one for each contact
            specified in the device configuration.

        """

        contacts = []
        for contact_config in self.config.device.contacts:
            contacts.append(
                Contact(
                    device=self,
                    name=contact_config.name,
                    origin=contact_config.origin,
                    vectors=contact_config.size,
                    direction=contact_config.direction,
                )
            )
        return contacts
