# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import os
import time
from dataclasses import dataclass, field

from pathlib import Path

from cupyx.profiler import time_range
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp, obc
from qttools.utils.mpi_utils import distributed_load

from scipy import sparse as sp_sparse

try:
    from qttools.cuDSS_binding.cudss_wrapp import spsolve_with_CUDSS
    CUDSS_AVAILABLE = True
    print("CUDSS available") if comm.rank == 0 else None
except ImportError:
    CUDSS_AVAILABLE = False


from qttools.datastructures.dsbsparse import _block_view

if xp.__name__ == "numpy":
    from scipy.sparse.linalg import spsolve
    from scipy.sparse.linalg import splu
if xp.__name__ == "cupy":
    from cupyx.scipy.sparse.linalg import spsolve
    from cupyx.scipy.sparse.linalg import splu

if xp.__name__ == "numpy":
    from numpy.linalg import lstsq
if xp.__name__ == "cupy":
    from cupy.linalg import lstsq

from qttools.utils.mpi_utils import get_local_slice

from quatrex.core.statistics import fermi_dirac
from quatrex.core.compute_config import ComputeConfig
from qttools.nevp import NEVP, Beyn, Full

from quatrex.core.quatrex_config import (
    OBCConfig,
    QuatrexConfig,
)

def get_periodic_superblocks_no_flip(
    as0: NDArray, a10: NDArray, a01: NDArray, as1: NDArray, block_sections: int
) -> NDArray:
    """
    Constructs the periodic superblocks for the OBC.

    Parameters
    ----------
    as0 : NDArray
        The superblock element as0. (Contains just one block, it connects two superblocks)
    a10 : NDArray
        The superblock element a10. (Contains block_sections blocks, it forms the in-coupling superblock)
    a01 : NDArray
        The superblock element a01. (Contains block_sections blocks, it forms the in-coupling superblock)
    as1 : NDArray
        The superblock element as1. (Contains just one block, it connects two superblocks)
    block_sections : int
        The number of block sections.

    Returns
    -------
    periodic_blocks : NDArray
        The periodic superblocks.
    """
    
    # Get the shape (including batching dimension) shape of the subbblocks.
    subblock_shape = a01.shape[:-2] + (a01.shape[-1] // block_sections,) * 2

    # Gets the total number of orbitals in the subblocks.
    orb_slab = a01.shape[-2]

    #Slice the superblocks
    as0 = _block_view(as0, -2, 1)
    a10 = _block_view(a10, -2, block_sections)
    a01 = _block_view(a01, -1, block_sections)
    as1 = _block_view(as1, -1, 1)

    if block_sections > 1:
        #Add a zero block
        z_j = _block_view(xp.zeros((orb_slab,orb_slab*(block_sections-1))),-1, block_sections-1)
         #Stack the blocks in the right (transport) order
        periodic_layer = xp.vstack((as0, a10[block_sections::-1], a01[1:], as1, z_j))
    else:
         #Stack the blocks in the right (transport) order
        periodic_layer = xp.vstack((as0, a10[block_sections::-1], a01[1:], as1))
    
   

    # Stack the periodic layer to form a periodic superblock structure.
    
    periodic_blocks = xp.zeros(
        (block_sections, 3 * block_sections, *subblock_shape),
        dtype=a01.dtype,
    )
    for i in range(block_sections):
        periodic_blocks[i, :] = xp.roll(periodic_layer, i, axis=0)

    # Recover the correct superbblock structure form the subblocks.
    periodic_blocks = xp.concatenate(xp.concatenate(periodic_blocks, -2), -1)
    return _block_view(periodic_blocks, -1, 3)


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
    corner1 : NDArray
        List containint the first corner of the contacts.
    corner2 : NDArray
        List containing the second corner of the contacts.
    direction : NDArray
        List containint the direction of the contacts. 

    """

    corner1 = []
    corner2 = []
    direction = []
    name = []
    n = None

    if comm.rank == 0:
        with open(filename, "rt") as myfile:

            # Read the number of contacts
            n = int(myfile.readline())
            for i in range(n):
                name.append(myfile.readline().rstrip('\n'))                         # Read the name of the contact
                corner1.append(xp.asarray(myfile.readline().split(),dtype=float))   # Read the corner 1
                corner2.append(xp.asarray(myfile.readline().split(),dtype=float))   # Read the corner 2
                direction.append(xp.asarray(myfile.readline().split(),dtype=float)) # Read the direction
    
    #Broadcast the data to all the ranks
    n = comm.bcast(n, root=0)
    name = comm.bcast(name, root=0)
    corner1 = comm.bcast(corner1, root=0)
    corner2 = comm.bcast(corner2, root=0)
    direction = comm.bcast(direction, root=0)

    return n,name, corner1, corner2, direction

def distributed_read_slabs(filename: Path) -> tuple[int, int]:
    """
    Reads the number of slabs in x and y from a file.

    Parameters
    ----------
    filename : Path
        The path to the file containing the number of slabs.
    
    Returns
    -------
    slab_x : int
        The number of slabs in x.
    slab_y : int
        The number of slabs in y.
    
    """

    if comm.rank == 0:
        slabs = xp.loadtxt(filename,dtype=xp.int32) #Read the number of slabs in x and y
    else:
        slabs = None
    
    slabs = comm.bcast(slabs, root=0) #Broadcast the data to all the ranks

    #Move the data to the CPU (it is just a single number)
    slab_x = slabs[0].get().item() if hasattr(slabs[0], 'get') else slabs[0].item()
    slab_y = slabs[1].get().item() if hasattr(slabs[1], 'get') else slabs[1].item()

    return slab_x, slab_y

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
        orbitals = xp.reshape(xp.loadtxt(filename,dtype=xp.int32),(-1,1)) #Read the number of orbitals for each atom type
    else:
        orbitals = None
    
    orbitals = comm.bcast(orbitals, root=0) #Broadcast the data to all the ranks

    return orbitals

def distributed_read_xyz(filename: Path) -> tuple[NDArray, list, NDArray, NDArray]:
    ''' 
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
    '''

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
                elif 'Lattice=' in line:
                    lattice = line.replace('Lattice="', '')
                    lattice = lattice.replace('"', '')
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

    #Broadcast the data to all the ranks
    lattice = comm.bcast(lattice, root=0)
    atoms = comm.bcast(atoms, root=0)
    coords = comm.bcast(coords, root=0)
    coordsType = comm.bcast(coordsType, root=0)

    return lattice, atoms, coords, coordsType

def compute_slab_vector_X(coords: NDArray, n_slabs: int, orbitals: NDArray) -> tuple[NDArray, NDArray]:
    """
    Computes the elements (atom,orbitals) for each slab in the x direction.

    Parameters
    ----------
    coords : NDArray
        The atomic coordinates.
    n_slabs : int
        The number of slabs in the x direction.
    orbitals : NDArray
        The starting orbital (cumulative) for each atom.
    
    Returns
    -------
    vec_atoms : NDArray
        Every atom in each slab.
    vec_orb : NDArray
        Every orbital in each slab.
    """

    vec_atoms = []
    vec_orb = []

    #dx is needed to allow every atom to be included in one slab slab
    dx = 0.001

    #Get the min and max x coordinates
    xMin=coords[:,0].min()
    xMax=coords[:,0].max()

    #Compute the width of each slab
    t_slab=(xMax-xMin)/n_slabs

    #Assign some group of atoms to each slab
    for i in range(n_slabs):
        if i != n_slabs-1:
            vec_atoms.append(xp.nonzero(xp.logical_and(coords[:,0]>=xMin+i*t_slab-dx,coords[:,0]<xMin+(i+1)*t_slab-dx))[0])
        else:
            vec_atoms.append(xp.nonzero(xp.logical_and(coords[:,0]>=xMin+i*t_slab-dx,coords[:,0]<=xMin+(i+1)*t_slab+dx))[0])
    
    #Assign the orbitals to each slab
    for i in range(n_slabs):
        vec_orb_loc = xp.array([],dtype=xp.int32)
        for j in range(vec_atoms[i].shape[0]):
            #NEED TO MOVE THE INDEX ON THE CPU
            #I USED A QUICK WORKAROUND FOR NOW
            index = int(vec_atoms[i][j].get() if hasattr(vec_atoms[i][j], 'get') else vec_atoms[i][j])
            k1 = int(orbitals[index].get() if hasattr(orbitals[index], 'get') else orbitals[index])
            k2 = int(orbitals[index+1].get() if hasattr(orbitals[index + 1], 'get') else orbitals[index + 1])
            vec_orb_loc = xp.concatenate((vec_orb_loc,xp.arange(k1, k2)))
        
        vec_orb.append(vec_orb_loc[None,:])
        print(f"Slab {i} has {vec_atoms[i].shape[0]} atoms and {vec_orb[i].shape[1]} orbitals", flush=True) if comm.rank == 0 else None

    return vec_atoms, vec_orb

class Contact:
    """Contact class"""

    name : str = None #Name of the contact

    corner_1: NDArray = None #First corner of the contact
    corner_2: NDArray = None #Second corner of the contact
    direction: NDArray = None #Direction of the contact

    vec_atoms_cont: NDArray = None #"in-coupling" atoms in the contact superblock
    vec_atoms_last_block: NDArray = None #"out-coupling" atoms in the last contact block

    vec_orb_cont: NDArray = None #in-coupling" orbitals in the contact superblock
    vec_orb_last_block = None #"out-coupling" orbitals in the last contact block

    vec_atoms_first_block: NDArray = None #atoms in the first contact block
    vec_orb_first_block: NDArray = None #orbitals in the first contact block
    
    N_coup: int = None #Number of sub-blocks in the contact superblock

    #Constructor
    def __init__(self, corner_1, corner_2, direction, name):
        self.corner_1 = corner_1
        self.corner_2 = corner_2
        self.direction = direction
        self.name = name

        self.N_coup = 0

    def sort_cont_at(self,vec_atoms: NDArray, delta_corner: NDArray, coords: NDArray, coordsType: NDArray) -> NDArray:
        """
        Sorts the atoms in the contact.

        Parameters
        ----------
        vec_atoms : NDArray
            The atoms in the contact.
        delta_corner : NDArray
            The shift of the corner.
        coords : NDArray
            The atomic coordinates.
        coordsType : NDArray
            The type of each atom.
        
        Returns
        -------
        vec_atoms : NDArray
            The sorted atoms in the contact.
        sorted : int
            1 if the atoms needed to be sorted, 0 otherwise.
        """

        sorted = 0 

        dx = 0.1 #A small margin used to ensure that the atoms can be sorted correctly

        vec_sorted = []

        #Position and type of atoms in the first contact block (shifted)
        shifted_coords_first_block = coords[self.vec_atoms_first_block.squeeze(),:] + delta_corner
        coordsType_first_block = coordsType[self.vec_atoms_first_block.squeeze()]

        if vec_atoms.squeeze().shape[0] != self.vec_atoms_first_block.squeeze().shape[0]:
            print("Number of atoms in the shifted contact block does not match the number of atoms in the first block")
            raise Exception("Error: number of atoms in the contact does not match the number of atoms in the first block")

        #For every atom in the shifted first contact block, find the index of the corresponding atom in the current contact block
        for i in range(len(vec_atoms)):
            flag = coords[vec_atoms,0] < shifted_coords_first_block[i,0] + dx
            flag &= coords[vec_atoms,0] > shifted_coords_first_block[i,0] - dx
            flag &= coords[vec_atoms,1] < shifted_coords_first_block[i,1] + dx
            flag &= coords[vec_atoms,1] > shifted_coords_first_block[i,1] - dx
            flag &= coords[vec_atoms,2] < shifted_coords_first_block[i,2] + dx
            flag &= coords[vec_atoms,2] > shifted_coords_first_block[i,2] - dx
            flag &= coordsType[vec_atoms] == coordsType_first_block[i]
            
            index = xp.nonzero(flag)[0] 
            numind = index.shape[0]
            if numind != 1:
                print("Number of compatible atoms found in the translated cell: ",numind)
                raise Exception("Error: more than one atom (or no one) found in the translated cell")
            if vec_atoms[index[0]] != vec_atoms[i]:
                sorted = 1
            vec_sorted.append(vec_atoms[index[0]])
        
        return xp.asarray(vec_sorted), sorted
    
    def compute_vector(self,coords: NDArray, orbitals: NDArray, coordsType, delta_corner: NDArray = xp.array([0,0,0])) -> tuple[NDArray, NDArray]:
        """
        Computes the vectors for the contact element.

        Parameters
        ----------
        coords : NDArray
            The atomic coordinates.
        orbitals : NDArray
            The starting orbital (cumulative) for each atom.
        delta_corner : NDArray, optional
            The shift of the corner, by default xp.array([0,0,0]). (only needed to compute the out-coupling elements)
        
        Returns
        -------
        vec_atoms : NDArray
            Every atom in the contact.
        vec_orb : NDArray
            Every orbital in the contact.
        """

        sorted = 0

        shifted_corner_1 = self.corner_1 + delta_corner #Shift the first corner (only needed for the out-coupling elements)
        shifted_corner_2 = self.corner_2 + delta_corner #Shift the second corner (only needed for the out-coupling elements)

        #Compute the atoms inside the contact slab
        vec_atoms = coords[:,0] >= shifted_corner_1[0] 
        vec_atoms &= coords[:,0] < shifted_corner_2[0]
        vec_atoms &= coords[:,1] >= shifted_corner_1[1]
        vec_atoms &= coords[:,1] < shifted_corner_2[1]
        vec_atoms &= coords[:,2] >= shifted_corner_1[2]
        vec_atoms &= coords[:,2] < shifted_corner_2[2]
        vec_atoms = xp.nonzero(vec_atoms)[0]

        if self.vec_atoms_first_block is not None:
            vec_atoms, sorted = self.sort_cont_at(vec_atoms,delta_corner,coords,coordsType)

        #Compute the orbitals inside the contact slab
        vec_orb = xp.array([],dtype=xp.int32)
        for i in range(vec_atoms.shape[0]):
            #NEED TO MOVE THE INDEX ON THE CPU
            #I USED A QUICK WORKAROUND FOR NOW
            index = int(vec_atoms[i].get() if hasattr(vec_atoms[i], 'get') else vec_atoms[i])
            k1 = int(orbitals[index].get() if hasattr(orbitals[index], 'get') else orbitals[index])
            k2 = int(orbitals[index+1].get() if hasattr(orbitals[index + 1], 'get') else orbitals[index + 1])
            vec_orb = xp.concatenate((vec_orb,xp.arange(k1, k2)))
        
        return vec_atoms, vec_orb, sorted


    def set_vector(self,coords: NDArray, orbitals: NDArray, ham: sparse.csr_matrix, coordsType: NDArray):
        """
        Sets the the contact elements.

        Parameters
        ----------
        coords : NDArray
            The atomic coordinates.
        orbitals : NDArray
            The starting orbital (cumulative) for each atom.
        """

        #Compute the vector for the first block of the contact element
        self.vec_atoms_first_block, self.vec_orb_first_block, sorted = self.compute_vector(coords,orbitals,coordsType) 

        #First initiate the contact superblock elements with the first block (will append stuff later)
        self.vec_atoms_cont = self.vec_atoms_first_block.copy()
        self.vec_orb_cont = self.vec_orb_first_block.copy()

        self.vec_atoms_first_block = self.vec_atoms_first_block[None,:]
        self.vec_orb_first_block = self.vec_orb_first_block[None,:]
        
        if sorted == 1:
            print(f"Contact {self.name}, slab {self.N_coup} has {self.vec_atoms_first_block.shape[1]} atoms and {self.vec_orb_first_block.shape[1]} orbitals. Sorted!", flush=True) if comm.rank == 0 else None
        else:
            print(f"Contact {self.name}, slab {self.N_coup} has {self.vec_atoms_first_block.shape[1]} atoms and {self.vec_orb_first_block.shape[1]} orbitals", flush=True) if comm.rank == 0 else None

        self.N_coup += 1

        self.N_coup_force = -1 #Forcing the number of contact blocks (for debug)

        while True:
            #Increase the corner of the contact element
            delta_corner = xp.multiply(-self.direction*(self.N_coup),(self.corner_2-self.corner_1))

            #Compute the vector for the new contact block
            vec_atoms_to_append, vec_orb_to_append, sorted= self.compute_vector(coords,orbitals,coordsType,delta_corner) 

            #Check if the new contact block is empty
            if ham[self.vec_orb_first_block.T,vec_orb_to_append[None,:]].nnz == 0:
                if self.N_coup_force == -1: #If yes exit the loop
                    break
                else:
                    if self.N_coup_force < self.N_coup:
                        raise Exception("Forcing a too small number of contacts element")
                    if self.N_coup_force == self.N_coup:
                        break
                        
            #Append the new contact block to the contact superblock
            #By doing this, we are forcing the right order (always entering inside the device)
            self.vec_atoms_cont = xp.concatenate((self.vec_atoms_cont,vec_atoms_to_append))
            self.vec_orb_cont = xp.concatenate((self.vec_orb_cont,vec_orb_to_append))
            if sorted == 1:
                print(f"Contact {self.name}, slab {self.N_coup} has {vec_atoms_to_append.shape[0]} atoms and {vec_orb_to_append.shape[0]} orbitals. Sorted!", flush=True) if comm.rank == 0 else None
            else:
                print(f"Contact {self.name}, slab {self.N_coup} has {vec_atoms_to_append.shape[0]} atoms and {vec_orb_to_append.shape[0]} orbitals", flush=True) if comm.rank == 0 else None

            self.N_coup += 1

        n_orb_block = self.vec_orb_cont.shape[0] // self.N_coup
        n_at_block = self.vec_atoms_cont.shape[0] // self.N_coup
        
        #The last block is separated from the rest (it is the out-coupling block)
        self.vec_orb_last_block = self.vec_orb_cont[-n_orb_block:][None,:]
        self.vec_atoms_last_block = self.vec_atoms_cont[-n_at_block:][None,:]

        self.vec_orb_cont = self.vec_orb_cont[:-n_orb_block][None,:]
        self.vec_atoms_cont = self.vec_atoms_cont[:-n_at_block][None,:]

        self.N_coup -=1

        print(f"Contact {self.name} has {self.vec_atoms_cont.shape[1]} atoms and {self.vec_orb_cont.shape[1]} orbitals", flush=True) if comm.rank == 0 else None

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

    n_orb_tot = orbitals_per_atom[-1].get().item() if hasattr(orbitals_per_atom[-1], 'get') else orbitals_per_atom[-1].item()

    orb_potential = xp.zeros((n_orb_tot,1),dtype=xp.float64)
    for i in range(potential.shape[0]):
        orb_potential[orbitals_per_atom[i]:orbitals_per_atom[i+1]] = potential[i]

    return orb_potential

@dataclass
class Observables:
    """Observable quantities for the SCBA."""

    # --- Electrons ----------------------------------------------------
    electron_ldos: NDArray = None
    electron_density: NDArray = None
    hole_density: NDArray = None
    electron_current: dict = field(default_factory=dict)
    
    electron_transmission_contacts : NDArray = None
    electron_transmission_contacts_labels = []

    electron_transmission_x_slabs: NDArray = None

    electron_DOS_x_slabs: NDArray = None

    valence_band_edges: NDArray = None
    conduction_band_edges: NDArray = None

    excess_charge_density: NDArray = None


class QTBM:
    """Quantum Transmitting Boundary Method (QTBM) solver.

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
        """Initializes a QTBM instance."""
        self.quatrex_config = quatrex_config

        if compute_config is None:
            compute_config = ComputeConfig()

        self.compute_config = compute_config

        self.observables = Observables()

        # Load the electron energies.
        self.electron_energies = distributed_load(
            self.quatrex_config.input_dir / "electron_energies.npy"
        )
        self.local_energies = get_local_slice(self.electron_energies) #Get the local slice of the electron energies
        
        self.obc = self._configure_obc(getattr(quatrex_config, "electron").obc) 

        # Load the device Hamiltonian.
        self.hamiltonian_sparray = distributed_load(
            quatrex_config.input_dir / "hamiltonian.npz"
        ).astype(xp.complex128)
        # Load the overlap matrix.
        try:
            self.overlap_sparray = distributed_load(
                quatrex_config.input_dir / "overlap.npz"
            ).astype(xp.complex128)
        except FileNotFoundError:
            # No overlap provided. Assume orthonormal basis.
            self.overlap_sparray = sparse.eye(
                self.hamiltonian_sparray.shape[0],
                format="coo",
                dtype=self.hamiltonian_sparray.dtype,
            )
        
        # Convert the sparse matrices to CSR format
        self.hamiltonian_sparray = self.hamiltonian_sparray.tocsr()
        self.overlap_sparray = self.overlap_sparray.tocsr()

        # Load the lattice and atomic coordinates
        self.lattice, self.atoms, self.coords, self.coordstType = distributed_read_xyz(quatrex_config.input_dir / "lattice.xyz")

        # Load the number of orbitals for each atom type
        self.orbitals_per_at = distributed_read_orbitals(quatrex_config.input_dir / "orb.dat")
 
        # Create a vector with the starting orbital for each atom
        self.orbitals_vec = xp.concatenate((xp.array([0]),xp.cumsum(self.orbitals_per_at[self.coordstType])),dtype=xp.int32)

        # Check that the overlap matrix and Hamiltonian matrix match.
        if self.overlap_sparray.shape != self.hamiltonian_sparray.shape:
            raise ValueError(
                "Overlap matrix and Hamiltonian matrix have different shapes."
            )

        #Load potential
        try: 
            self.atom_potential = distributed_load(
            self.quatrex_config.input_dir / "potential.npy"
                )
            #Upscale the potential to the number of orbitals
            self.orb_potential = get_orb_potential(self.atom_potential, self.orbitals_vec) 

            #Add the potential to the Hamiltonian
            SV1 = self.overlap_sparray.multiply(self.orb_potential).tocsr()
            SV2 = self.overlap_sparray.multiply(self.orb_potential.T).tocsr()
            self.hamiltonian_sparray += (SV1+SV2)/2
            self.hamiltonian_sparray.eliminate_zeros()

        except FileNotFoundError:
            print("No pot provided.", flush=True) if comm.rank == 0 else None
            
        
        self.flatband = quatrex_config.electron.flatband
        self.eta_obc = quatrex_config.electron.eta_obc
        self.block_sections = quatrex_config.electron.obc.block_sections

        # Load contact info
        self.n_cont,self.cont_names,self.corner1,self.corner2,self.corner_direction = distributed_load_contact(quatrex_config.input_dir / "cont.dat")

        # CREATE CONTACT LIST
        self.contacts = []
        for n in range(self.n_cont):
            self.contacts.append(Contact(self.corner1[n],self.corner2[n],self.corner_direction[n],self.cont_names[n]))
            self.contacts[n].set_vector(self.coords, self.orbitals_vec, self.hamiltonian_sparray, self.coordstType)

        # CREATE VECTORS FOR EVERY SLAB
        self.n_slabs_x, self.n_slab_y = distributed_read_slabs(quatrex_config.input_dir / "slabs.dat")
        self.slab_vec_x_at, self.slab_vec_x_orb = compute_slab_vector_X(self.coords,self.n_slabs_x,self.orbitals_vec)

        # Look for all the combinations of contacts
        self.n_transmissions = int((self.n_cont**2-self.n_cont)/2)
        cont_1 = 0
        cont_2 = 1
        for n in range(self.n_transmissions):
            # Append the label for every transmission
            self.observables.electron_transmission_contacts_labels.append(self.contacts[cont_1].name[0] + '->' + self.contacts[cont_2].name[0])
            cont_2 += 1
            if cont_2 == self.n_cont:
                cont_1 += 1
                cont_2 = cont_1+1

        # Initialize the observables
        self.observables.electron_transmission_contacts = xp.zeros((self.n_transmissions,self.local_energies.shape[0]),dtype=xp.float64)
        self.observables.electron_transmission_x_slabs = xp.zeros((self.n_cont,self.n_slabs_x,self.local_energies.shape[0]),dtype=xp.float64)
        self.observables.electron_DOS_x_slabs = xp.zeros((self.n_cont,self.n_slabs_x,self.local_energies.shape[0]),dtype=xp.float64)

        # Band edges and Fermi levels.
        # TODO: This only works for small potential variations accross
        # the device.
        # TODO: During this initialization we should compute the contact
        # band structures and extract the correct fermi levels & band
        # edges from there.
        #self.band_edge_tracking = quatrex_config.electron.band_edge_tracking
        #self.delta_fermi_level_conduction_band = (
        #    quatrex_config.electron.conduction_band_edge
        #    - quatrex_config.electron.fermi_level
        #)
        #self.left_mid_gap_energy = quatrex_config.electron.left_fermi_level
        #self.right_mid_gap_energy = quatrex_config.electron.right_fermi_level

        self.temperature = quatrex_config.electron.temperature

        self.left_fermi_level = quatrex_config.electron.left_fermi_level
        self.right_fermi_level = quatrex_config.electron.right_fermi_level

        self.left_occupancies = fermi_dirac(
            self.local_energies - self.left_fermi_level, self.temperature
        )
        self.right_occupancies = fermi_dirac(
            self.local_energies - self.right_fermi_level, self.temperature
        )

    def _configure_nevp(self, obc_config: OBCConfig) -> NEVP:
        """Configures the NEVP solver from the config.

        Parameters
        ----------
        obc_config : OBCConfig
            The OBC configuration.

        Returns
        -------
        NEVP
            The configured NEVP solver.

        """
        if obc_config.nevp_solver == "beyn":
            return Beyn(
                r_o=obc_config.r_o,
                r_i=obc_config.r_i,
                m_0=obc_config.m_0,
                num_quad_points=obc_config.num_quad_points,
            )
        if obc_config.nevp_solver == "full":
            return Full()

        raise NotImplementedError(
            f"NEVP solver '{obc_config.nevp_solver}' not implemented."
        )

    def _configure_obc(self, obc_config: OBCConfig) -> obc.OBCSolver:
        """Configures the OBC algorithm from the config.

        Parameters
        ----------
        obc_config : OBCConfig
            The OBC configuration.

        Returns
        -------
        obc.OBCSolver
            The configured OBC solver.

        """
        if obc_config.algorithm == "sancho-rubio":
            raise NotImplementedError(
                f"Sancho-rubio OBC algorithm does not work with QTBM, please use spectral OBC solver."
            )

        elif obc_config.algorithm == "spectral":
            nevp = self._configure_nevp(obc_config)
            obc_solver = obc.Spectral(
                nevp=nevp,
                block_sections=obc_config.block_sections,
                min_decay=obc_config.min_decay,
                max_decay=obc_config.max_decay,
                num_ref_iterations=obc_config.num_ref_iterations,
                x_ii_formula=obc_config.x_ii_formula,
                two_sided=obc_config.two_sided,
                treat_pairwise=obc_config.treat_pairwise,
                pairing_threshold=obc_config.pairing_threshold,
                min_propagation=obc_config.min_propagation,
            )

        else:
            raise NotImplementedError(
                f"OBC algorithm '{obc_config.algorithm}' not implemented."
            )

        if obc_config.memoizer.enable:
            obc_solver = obc.OBCMemoizer(
                obc_solver,
                obc_config.memoizer.num_ref_iterations,
                obc_config.memoizer.convergence_tol,
            )

        return obc_solver
    
    def compute_observables(self,phi: NDArray, inj_ind: list, i: int, E: float, S:list):
        """
        Compute observables for the current iteration.

        Parameters
        ----------
        phi : NDArray
            The wavefunction.
        inj_ind : list
            The indices of the injection vectors.
        i : int
            The iteration number.
        w : NDArray
            The injected phase factor (per every injected vector)
        """

        if phi.size == 0:
            return
        #Compute transmissions for all the possible contact couples
        cont_1 = 0
        cont_2 = 1
        for n in range(self.n_transmissions):

            #Get the all the wavefunctions injected from contact 1 and extract the elements inside contact 2
            phi_n = phi[self.contacts[cont_2].vec_orb_cont.T,inj_ind[cont_1]]

            #Compute the transmission
            if(phi_n.size != 0):
                self.observables.electron_transmission_contacts[n,i] = xp.trace(xp.real(1j*phi_n.T.conj() @ (S[cont_2]-S[cont_2].T.conj()) @phi_n))
            
            cont_2 += 1
            if cont_2 == self.n_cont:
                cont_1 += 1
                cont_2 = cont_1+1

        #Compute transmission for all the x slabs and all the contacts
        for n in range(self.n_cont):
            for s in range(self.n_slabs_x-1):

                #For every slab, get the wavefunction injected from the contact
                phi_1 = phi[self.slab_vec_x_orb[s].T,inj_ind[n]]
                phi_2 = phi[self.slab_vec_x_orb[s+1].T,inj_ind[n]]

                #Get the transmission matrix between the slab and the next one
                T01 = self.system_matrix[self.slab_vec_x_orb[s].T,self.slab_vec_x_orb[s+1]]

                if(phi_1.size != 0):
                    self.observables.electron_transmission_x_slabs[n,s,i] = xp.trace(2*xp.imag(phi_1.T.conj() @ T01 @phi_2))

        #Compute DOS

        #Spill over correction
        phi_ortho = self.overlap_sparray @ phi #"Orthogonalize" the wavefunction
        for n in range(self.n_cont):
            #Get the off-couping superblocks for Ham and Overlap (Could also save them in the contact class)
            H_off_coup_cont,_,_ = get_periodic_superblocks_no_flip(
                    self.hamiltonian_sparray[self.contacts[n].vec_orb_last_block.T,self.contacts[n].vec_orb_first_block].toarray(),
                    self.hamiltonian_sparray[self.contacts[n].vec_orb_cont.T,self.contacts[n].vec_orb_first_block].toarray(),
                    self.hamiltonian_sparray[self.contacts[n].vec_orb_first_block.T,self.contacts[n].vec_orb_cont].toarray(),
                    self.hamiltonian_sparray[self.contacts[n].vec_orb_first_block.T,self.contacts[n].vec_orb_last_block].toarray(),
                    block_sections=self.contacts[n].N_coup,
                )
            S_off_coup_cont,_,_ = get_periodic_superblocks_no_flip(
                    self.overlap_sparray[self.contacts[n].vec_orb_last_block.T,self.contacts[n].vec_orb_first_block].toarray(),
                    self.overlap_sparray[self.contacts[n].vec_orb_cont.T,self.contacts[n].vec_orb_first_block].toarray(),
                    self.overlap_sparray[self.contacts[n].vec_orb_first_block.T,self.contacts[n].vec_orb_cont].toarray(),
                    self.overlap_sparray[self.contacts[n].vec_orb_first_block.T,self.contacts[n].vec_orb_last_block].toarray(),
                    block_sections=self.contacts[n].N_coup,
                )
            
            #Solve a small system to propagate the WF inside the contact
            #RHS of the system
            B = self.system_matrix[self.contacts[n].vec_orb_cont.squeeze(),:] @ phi
            #Solve the system of equations
            phi_cont, _, _,_  = lstsq(H_off_coup_cont - E*S_off_coup_cont, -B,rcond=None)
            #Add the spill over contribution
            phi_ortho[self.contacts[n].vec_orb_cont.squeeze(),:] += S_off_coup_cont @ phi_cont 

        #Compute the DOS for every injected wavefunction
        for n in range(self.n_cont):
            for s in range(self.n_slabs_x):
                phi_D = phi[self.slab_vec_x_orb[s].T,inj_ind[n]].squeeze() #Get the wavefunction in the slab
                phi_D_ortho = phi_ortho[self.slab_vec_x_orb[s].T,inj_ind[n]].squeeze() #Get the "orthogonalized" wavefunction in the slab
                if(phi_D.size != 0):
                    self.observables.electron_DOS_x_slabs[n,s,i]=xp.real(xp.sum(xp.multiply(phi_D.conj(), phi_D_ortho))/(2*xp.pi)) #Compute the DOS

    def run(self) -> None:
        """Runs the QTBM"""
        print("Entering QTBM calculation", flush=True) if comm.rank == 0 else None
        times = []
        comm.Barrier()
        for i,E in enumerate(self.local_energies):

            print(f"Iteration {i}", flush=True) if comm.rank == 0 else None

            # append for iteration time
            times.append(time.perf_counter())

            times.append(time.perf_counter())

            self.system_matrix = self.hamiltonian_sparray - E * self.overlap_sparray

            t_solve = time.perf_counter() - times.pop()
            (
                print(f"Time for constructing bare sys. matrix: {t_solve:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            times.append(time.perf_counter())

            sigma_b = []
            inj = []
            inj_ind = []
            w = []
            # Compute the boundary self-energy and the injection vector
            ind_0 = 0
            for n in range(self.n_cont):

                #Get the contact superblocks
                m_10, m_00, m_01 = get_periodic_superblocks_no_flip(
                    self.system_matrix[self.contacts[n].vec_orb_last_block.T,self.contacts[n].vec_orb_first_block].toarray(),
                    self.system_matrix[self.contacts[n].vec_orb_cont.T,self.contacts[n].vec_orb_first_block].toarray(),
                    self.system_matrix[self.contacts[n].vec_orb_first_block.T,self.contacts[n].vec_orb_cont].toarray(),
                    self.system_matrix[self.contacts[n].vec_orb_first_block.T,self.contacts[n].vec_orb_last_block].toarray(),
                    block_sections=self.contacts[n].N_coup,
                )

                #Set the block sections for the OBC
                self.obc.block_sections = self.contacts[n].N_coup
                _ , sigma_b_n, inj_n, _ = self.obc(
                    m_00,
                    m_01,
                    m_10,
                    "left",
                    return_injected = True,
                )

                sigma_b.append(sigma_b_n)
                inj.append(inj_n)
                inj_ind.append(xp.arange(ind_0,ind_0+inj_n.shape[1])[None,:])
                ind_0 += inj_n.shape[1]

            t_solve = time.perf_counter() - times.pop()
            (
                print(f"Time for OBC: {t_solve:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            times.append(time.perf_counter())

            # Set up sytem matrix and rhs for electron solver.
            inj_V = xp.zeros((self.system_matrix.shape[0],ind_0), dtype=xp.complex128, order="F") #Set the injection vector as a zero matrix

            ind1 = []
            ind2 = []
            sig_flat = []
            #Iterate over contacts
            for n in range(self.n_cont):
                ind1.append(xp.repeat(self.contacts[n].vec_orb_cont.squeeze(), self.contacts[n].vec_orb_cont.shape[1]))
                ind2.append(xp.tile(self.contacts[n].vec_orb_cont.squeeze(), self.contacts[n].vec_orb_cont.shape[1]))
                sig_flat.append(sigma_b[n].flatten())
                inj_V[self.contacts[n].vec_orb_cont.T,inj_ind[n]] = inj[n] #Add the injection vector in the contact elements of the rhs
            
            #Concatenate the indices and the self-energies
            ind1 = xp.concatenate(ind1)
            ind2 = xp.concatenate(ind2)
            sig_flat = xp.concatenate(sig_flat)

            upd_0 = sparse.coo_matrix((sig_flat, (ind1, ind2)),shape=self.system_matrix.shape).tocsr()

            #Update the system matrix with the self-energies
            self.system_matrix -= upd_0

            #Iterate over contacts
            #for n in range(self.n_cont):
            #    sigma_cpu = sigma_b[n].get() #Move the self-energy to the CPU
            #    vec_orb_cont_cpu = self.contacts[n].vec_orb_cont.get() #Move the contact elements to the CPU
            #    self.system_matrix[vec_orb_cont_cpu.T,vec_orb_cont_cpu] -= sigma_cpu #Subtract the self-energy in the contact elements
            #    inj_V[self.contacts[n].vec_orb_cont.T,inj_ind[n]] = inj[n] #Add the injection vector in the contact elements of the rhs

            #self.system_matrix = sparse.csr_matrix(self.system_matrix)

            #Eliminate the zeros that were added in the system matrix
            self.system_matrix.eliminate_zeros()

            t_solve = time.perf_counter() - times.pop()
            (
                print(f"Time to set up system of eq.: {t_solve:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            times.append(time.perf_counter())

            # Solve for the wavefunction
            #phi = spsolve(self.system_matrix, inj_V)

            if inj_V.size != 0:
                if CUDSS_AVAILABLE and xp.__name__ == "cupy":
                    #USE CUDSS
                    phi = spsolve_with_CUDSS(self.system_matrix, inj_V)
                else:
                    lu = splu(self.system_matrix)
                    phi = lu.solve(inj_V)


            t_solve = time.perf_counter() - times.pop()
            (
                print(f"Time for electron solver: {t_solve:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            #self.system_matrix = self.system_matrix.get()
            
            # Get the bare system matrix back, needed for transmission calculation 
            #for n in range(self.n_cont):
            #    sigma_cpu = sigma_b[n].get()
            #    vec_orb_cont_cpu = self.contacts[n].vec_orb_cont.get()
            #   self.system_matrix[vec_orb_cont_cpu.T,vec_orb_cont_cpu] += sigma_cpu #Add the self-energy back

            #self.system_matrix = sparse.csr_matrix(self.system_matrix)
            self.system_matrix += upd_0
            
            if inj_V.size != 0:
                # Compute observables (DOS and Transmission)
                self.compute_observables(phi,inj_ind,i,E,sigma_b)

            t_iteration = time.perf_counter() - times.pop()
            (
                print(f"Time for iteration: {t_iteration:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )
        
        # Gather the observables
        self.observables.electron_transmission_x_slabs = xp.concatenate(comm.allgather(self.observables.electron_transmission_x_slabs),axis=-1)
        self.observables.electron_transmission_contacts = xp.hstack(comm.allgather(self.observables.electron_transmission_contacts))
        self.observables.electron_DOS_x_slabs = xp.concatenate(comm.allgather(self.observables.electron_DOS_x_slabs),axis=-1)