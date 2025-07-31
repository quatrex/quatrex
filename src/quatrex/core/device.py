from collections import defaultdict
from pathlib import Path

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp
from qttools.utils.mpi_utils import distributed_load

from quatrex.core.contact import Contact
from quatrex.core.quatrex_config import QuatrexConfig


def get_orbital_potential(potential: NDArray, orbital_offsets: NDArray) -> NDArray:
    """
    Computes the potential for each orbital.

    Parameters
    ----------
    potential : NDArray
        The potential.
    orbitals : NDArray
        The starting orbital (cumulative) for each atom.

    Returns
    -------
    orb_potential : NDArray
        The potential for each orbital.

    """
    orbitals_per_atom = np.diff(orbital_offsets, prepend=0)
    orbital_potential = np.repeat(potential, orbitals_per_atom, axis=0)
    return orbital_potential


def distributed_read_xyz(filename: Path) -> tuple[NDArray, list, NDArray, NDArray]:
    """Reads data from xyz files

    Parameters
    ----------
    filename : str
        The name of the file to read (*.xyz)

    Returns
    -------
    lattice : NDArray
        A (3x3) array containing the (rectangular) unit cell size
    atoms : list
        A list containing the atom symbols
    coords : NDArray
        A (*x3) array containing the atomic coordinates
    coordsType : NDArray
        An array containing each type of atom (as integer)

    """

    lattice = None
    coords = None
    kinds = None

    if comm.rank == 0:
        # Read only the second line of the file (this contains the lattice parameters)
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
    """
    A class to represent a device in the Quatrex framework.

    Attributes
    ----------

    name : str
        The name of the device.
    hamiltonian : dict
        The Hamiltonian of the device.
    num_hamiltonians : int
        The number of Hamiltonian matrices loaded.
    num_overlaps : int
        The number of overlap matrices loaded.
    overlap : dict
        The overlap matrix of the device.
    lattice_vector : NDArray
        The lattice vector of the device.
    atoms_list : list
        A list of atom symbols in the device.
    coords : NDArray
        The coordinates of the atoms in the device.
    atom_type : NDArray
        An array containing the type of each atom in the device (as an integer).
    orbitals_per_at : NDArray
        The number of orbitals for each atom type.
    orbitals_vec : NDArray
        A vector containing the starting orbital for each atom type.
    atom_potential : NDArray
        The potential for each atom type.
    orb_potential : NDArray
        The potential for each orbital.

    Methods
    -------
    load_hamiltonian(quatrex_config: QuatrexConfig) -> None
        Load the Hamiltonian and overlap matrix from the specified configuration.
    load_lattice(quatrex_config: QuatrexConfig) -> None
        Load the lattice structure from the specified configuration.

    """

    def __init__(self, quatrex_config: QuatrexConfig) -> None:

        self.config = quatrex_config

        self._init_hamiltonian()
        self._init_lattice()
        self._init_orbitals()
        self._load_potential()
        self.apply_potential()
        self.contacts = self._add_contacts()

    def _init_hamiltonian(self) -> None:
        """Initializes Hamiltonian and overlap matrices."""

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
        """Initalizes the lattice structure of the device."""

        # Load the lattice structure from file.
        lattice_file = self.config.input_dir / "lattice.xyz"
        if not lattice_file.exists():
            raise FileNotFoundError(f"Lattice file {lattice_file} not found.")
        self.lattice_vector, self.atoms_list, self.coords, self.atom_type = (
            distributed_read_xyz(lattice_file)
        )
        if comm.rank == 0:
            print("Lattice structure loaded successfully.", flush=True)

    def _init_orbitals(self):
        """Initializes the orbitals of the device."""
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
        """Loads the potential from the specified configuration."""

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
        """Applies the potential to the Hamiltonian."""

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
        """
        Add a contact to the device.

        Parameters
        ----------
        name : str
            The name of the contact.
        vectors : NDArray
            The vectors defining the contact.
        origin : NDArray
            The origin of the contact.
        direction : int
            The direction of the contact.

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
