import glob
import re
from pathlib import Path

from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp
from qttools.utils.mpi_utils import distributed_load

from quatrex.core.contact import Contact
from quatrex.core.quatrex_config import QuatrexConfig


def distributed_load_contact(filename: Path) -> tuple[int, list, list, list, list]:
    """
    Loads the contact data from a file.

    Parameters
    ----------
    filename : Path
        The path to the contact file.

    Returns
    -------
    n : int
        The number of contacts.
    name : list
        List containing the names of the contacts.
    origin : NDArray
        List containint the origin of the contacts.
    vectors : NDArray
        List containing the vectors of the contacts.
    direction : NDArray
        List containint the direction of the contacts.

    """

    origin = []
    vectors = []
    direction = []
    name = []
    n = None

    if comm.rank == 0:
        with open(filename, "rt") as myfile:

            # Read the number of contacts
            n = int(myfile.readline())
            for i in range(n):
                # Read the name
                name.append(myfile.readline().rstrip("\n"))
                origin.append(xp.asarray(myfile.readline().split(), dtype=xp.float64))
                vectors.append(xp.zeros((3, 3), dtype=xp.float64))  # Initialize vectors
                for i in range(3):
                    vectors[-1][i, :] = xp.asarray(
                        myfile.readline().split(), dtype=xp.float64
                    )
                direction.append(int(myfile.readline()))  # Read the direction

    # Broadcast the data to all the ranks
    n = comm.bcast(n, root=0)
    name = comm.bcast(name, root=0)
    origin = comm.bcast(origin, root=0)
    vectors = comm.bcast(vectors, root=0)
    direction = comm.bcast(direction, root=0)

    return n, name, origin, vectors, direction


def _get_orb_potential(potential: NDArray, orbitals_per_atom: NDArray) -> NDArray:
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

    n_orb_tot = (
        orbitals_per_atom[-1].get().item()
        if hasattr(orbitals_per_atom[-1], "get")
        else orbitals_per_atom[-1].item()
    )

    orb_potential = xp.zeros((n_orb_tot, 1), dtype=xp.float64)
    for i in range(potential.shape[0]):
        orb_potential[orbitals_per_atom[i] : orbitals_per_atom[i + 1]] = potential[i]

    return orb_potential


def get_orb_potential(potential: NDArray, orbitals_per_atom: NDArray) -> NDArray:
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

    n_orb_tot = (
        orbitals_per_atom[-1].get().item()
        if hasattr(orbitals_per_atom[-1], "get")
        else orbitals_per_atom[-1].item()
    )

    orb_potential = xp.zeros((n_orb_tot, 1), dtype=xp.float64)
    for i in range(potential.shape[0]):
        orb_potential[orbitals_per_atom[i] : orbitals_per_atom[i + 1]] = potential[i]

    return orb_potential


def distributed_read_orbitals(filename: Path) -> NDArray:
    """
    Reads the number of orbitals for each atom type from a file.

    Parameters
    ----------
    filename : Path
        The path to the file containing the number of orbitals.

    Returns
    -------
    orbitals : NDArray
        The number of orbitals for each atom type.

    """

    if comm.rank == 0:
        orbitals = xp.reshape(
            xp.loadtxt(filename, dtype=xp.int32), (-1, 1)
        )  # Read the number of orbitals for each atom type
    else:
        orbitals = None

    orbitals = comm.bcast(orbitals, root=0)  # Broadcast the data to all the ranks

    return orbitals


def distributed_read_xyz(filename: Path) -> tuple[NDArray, list, NDArray, NDArray]:
    """
    Reads data from xyz files

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

    atoms = []
    coords = []
    coordsType = []
    lattice = []

    if comm.rank == 0:

        with open(filename, "rt") as myfile:
            for line in myfile:
                # num_atoms line
                if len(line.split()) == 1:
                    pass
                # blank line
                elif len(line.split()) == 0:
                    pass
                # line with cell parameters
                elif "Lattice=" in line:
                    lattice = line.replace('Lattice="', "")
                    lattice = lattice.replace('"', "")
                    lattice = lattice.split()[0:9]
                    lattice = [float(item) for item in lattice]
                    lattice = xp.array(lattice)
                    lattice = xp.reshape(lattice, (3, -1))

                # line with atoms and positions
                elif len(line.split()) == 4:
                    c = line.split()[0]
                    if atoms.count(c) == 0:
                        atoms.append(c)
                    coordsType.append(atoms.index(c))
                    coords.append(line.split()[1:])
                else:
                    pass

        coords = xp.asarray(coords, dtype=xp.float64)
        coordsType = xp.asarray(coordsType, dtype=xp.int16)
        lattice = xp.asarray(lattice, dtype=xp.float64)

    # Broadcast the data to all the ranks
    lattice = comm.bcast(lattice, root=0)
    atoms = comm.bcast(atoms, root=0)
    coords = comm.bcast(coords, root=0)
    coordsType = comm.bcast(coordsType, root=0)

    return lattice, atoms, coords, coordsType


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

    def __init__(self):

        self.hamiltonian: dict = {}
        self.overlap: dict = {}

        self.num_hamiltonians = 0
        self.num_overlaps = 0

        self.lattice_vector: NDArray = None
        self.atoms_list: list = None
        self.coords: NDArray = None
        self.atom_type: NDArray = None

        self.orbitals_per_at: NDArray = None
        self.orbitals_vec: NDArray = None

        self.atom_potential: NDArray = None
        self.orb_potential: NDArray = None

        self.gamma_only = False

    def load_hamiltonian(self, quatrex_config: QuatrexConfig) -> None:
        """
        Load the Hamiltonian and overlap matrix from the specified configuration.
        Parameters
        ----------
        quatrex_config : QuatrexConfig
            The configuration object containing the input directory and other settings.
        """

        # Load all Hamiltonian files with format hamiltonian_x_y.npz
        # Find all hamiltonian files in the input directory
        hamiltonian_pattern = str(quatrex_config.input_dir / "hamiltonian_*.npz")
        hamiltonian_files = glob.glob(hamiltonian_pattern)

        # Parse indices from filenames and load files
        for file_path in hamiltonian_files:
            filename = file_path.split("/")[-1]  # Get just the filename
            match = re.match(r"hamiltonian_(-?\d+)_(-?\d+)_(-?\d+)\.npz", filename)
            if match:
                x_index = int(match.group(1))
                y_index = int(match.group(2))
                z_index = int(match.group(3))
                try:
                    self.hamiltonian[(x_index, y_index, z_index)] = (
                        distributed_load(Path(file_path)).astype(xp.complex128).tocsr()
                    )
                    print(f"Loaded Hamiltonian ({x_index} {y_index} {z_index})")
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")

        if (0, 0, 0) not in self.hamiltonian:
            raise ValueError(
                "Hamiltonian matrix for (0,0,0) not found. Please check the input files."
            )

        print(f"Loaded {len(self.hamiltonian)} Hamiltonian matrices")
        self.num_hamiltonians = len(self.hamiltonian)
        if self.num_hamiltonians == 1:
            print(
                "Only Gamma point Hamiltonian found. K-Points calculations are not possible."
            )
            self.gamma_only = True

        # Load all overlap files with format overlap_x_y.npz
        overlap_pattern = str(quatrex_config.input_dir / "overlap_*.npz")
        overlap_files = glob.glob(overlap_pattern)

        # Parse indices from filenames and load files
        for file_path in overlap_files:
            filename = file_path.split("/")[-1]  # Get just the filename
            match = re.match(r"overlap_(-?\d+)_(-?\d+)_(-?\d+)\.npz", filename)
            if match:
                x_index = int(match.group(1))
                y_index = int(match.group(2))
                z_index = int(match.group(3))
                try:
                    self.overlap[(x_index, y_index, z_index)] = (
                        distributed_load(Path(file_path)).astype(xp.complex128).tocsr()
                    )
                    print(f"Loaded Overlap ({x_index} {y_index} {z_index})")
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")

        if (0, 0, 0) not in self.overlap:
            print("Overlap matrix for (0,0,0) not found. Assuming identity matrix.")
            self.overlap[(0, 0, 0)] = sparse.eye(
                self.hamiltonian[(0, 0, 0)].shape[0], dtype=xp.complex128, format="csr"
            )

        self.num_overlaps = len(self.overlap)
        print(f"Loaded {self.num_overlaps} overlap matrices")

        # TODO # Check if the number of Hamiltonians and overlaps match

    def load_lattice(self, quatrex_config: QuatrexConfig) -> None:
        """
        Load the lattice structure from the specified configuration.
        Parameters
        ----------
        quatrex_config : QuatrexConfig
            The configuration object containing the input directory and other settings.
        """
        # Load lattice structure from file

        lattice_file = quatrex_config.input_dir / "lattice.xyz"
        if not lattice_file.exists():
            raise FileNotFoundError(f"Lattice file {lattice_file} not found.")
        self.lattice_vector, self.atoms_list, self.coords, self.atom_type = (
            distributed_read_xyz(lattice_file)
        )
        print("Lattice structure loaded successfully.")

    def load_orbitals(self, quatrex_config: QuatrexConfig) -> NDArray:
        """
        Load the number of orbitals for each atom type from the specified configuration.
        Parameters
        ----------
        quatrex_config : QuatrexConfig
            The configuration object containing the input directory and other settings.

        Returns
        -------
        NDArray
            The number of orbitals for each atom type.
        """
        self.orbitals_per_at = distributed_read_orbitals(
            quatrex_config.input_dir / "orb.dat"
        )

        # Create a vector with the starting orbital for each atom
        self.orbitals_vec = xp.concatenate(
            (xp.array([0]), xp.cumsum(self.orbitals_per_at[self.atom_type])),
            dtype=xp.int32,
        )

    def load_potential(self, quatrex_config: QuatrexConfig) -> None:
        """
        Load the potential from the specified configuration.
        Parameters
        ----------
        quatrex_config : QuatrexConfig
            The configuration object containing the input directory and other settings.
        """

        self.orb_potential = None

        try:
            self.atom_potential = distributed_load(
                quatrex_config.input_dir / "potential.npy"
            )
            # Upscale the potential to the number of orbitals
            self.orb_potential = get_orb_potential(
                self.atom_potential, self.orbitals_vec
            )

        except FileNotFoundError:
            (
                print("No external potential is provided.", flush=True)
                if comm.rank == 0
                else None
            )

    def apply_potential(
        self,
    ) -> None:

        if self.orb_potential is None:
            print("No potential loaded. Skipping potential application.")
            return

        for key, value in self.overlap.items():

            SV1 = value.multiply(self.orb_potential).tocsr()
            SV2 = value.multiply(self.orb_potential.T).tocsr()
            self.hamiltonian[key] += (SV1 + SV2) / 2
            self.hamiltonian[key].eliminate_zeros()

    def add_contacts(self, config) -> None:
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
        self.contacts = []
        n, name, origin, vectors, direction = distributed_load_contact(
            config.input_dir / "cont.dat"
        )
        for i in range(n):
            self.contacts.append(
                Contact(name[i], self, vectors[i], origin[i], direction[i], config)
            )
