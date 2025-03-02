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

if xp.__name__ == "numpy":
    from scipy.sparse.linalg import spsolve
if xp.__name__ == "cupy":
    from cupyx.scipy.sparse.linalg import spsolve

from qttools.utils.mpi_utils import get_local_slice

from quatrex.core.statistics import fermi_dirac
from quatrex.core.compute_config import ComputeConfig
from qttools.nevp import NEVP, Beyn, Full

from quatrex.core.quatrex_config import (
    OBCConfig,
    QuatrexConfig,
)


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

def compute_slab_mask_X(coords: NDArray, n_slabs: int, orbitals: NDArray) -> tuple[NDArray, NDArray]:
    """
    Computes the mask for each slab in the x direction.

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
    mask_atoms : NDArray
        List of every atom in each slab.
    mask_orb : NDArray
        List of every orbital in each slab.
    """

    mask_atoms = []
    mask_orb = []

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
            mask_atoms.append(xp.nonzero(xp.logical_and(coords[:,0]>=xMin+i*t_slab-dx,coords[:,0]<xMin+(i+1)*t_slab-dx))[0])
        else:
            mask_atoms.append(xp.nonzero(xp.logical_and(coords[:,0]>=xMin+i*t_slab-dx,coords[:,0]<=xMin+(i+1)*t_slab+dx))[0])
    
    #Assign the orbitals to each slab
    for i in range(n_slabs):
        mask_orb_loc = xp.array([],dtype=xp.int32)
        for j in range(mask_atoms[i].shape[0]):
            #NEED TO MOVE THE INDEX ON THE CPU
            #I USED A QUICK WORKAROUND FOR NOW
            index = int(mask_atoms[i][j].get() if hasattr(mask_atoms[i][j], 'get') else mask_atoms[i][j])
            k1 = int(orbitals[index].get() if hasattr(orbitals[index], 'get') else orbitals[index])
            k2 = int(orbitals[index+1].get() if hasattr(orbitals[index + 1], 'get') else orbitals[index + 1])
            mask_orb_loc = xp.concatenate((mask_orb_loc,xp.arange(k1, k2)))
        
        mask_orb.append(mask_orb_loc[None,:])

    return mask_atoms, mask_orb

class Contact:
    """Contact class"""

    name : str = None #Name of the contact

    corner_1: NDArray = None #First corner of the contact
    corner_2: NDArray = None #Second corner of the contact
    direction: NDArray = None #Direction of the contact

    mask_atoms_i: NDArray = None #Mask for the "in-coupling" atoms in the contact slab
    mask_atoms_j: NDArray = None #Mask for the "out-coupling" atoms in the contact slab

    mask_orb_i: NDArray = None #Mask for the "in-coupling" orbitals in the contact slab
    mask_orb_j: NDArray = None #Mask for the "out-coupling" orbitals in the contact slab

    #Constructor
    def __init__(self, corner_1, corner_2, direction, name):
        self.corner_1 = corner_1
        self.corner_2 = corner_2
        self.direction = direction
        self.name = name

    def compute_mask(self,coords: NDArray, orbitals: NDArray, delta_corner: NDArray = xp.array([0,0,0])) -> tuple[NDArray, NDArray]:
        """
        Computes the mask for the contact element.

        Parameters
        ----------
        coords : NDArray
            The atomic coordinates.
        orbitals : NDArray
            The starting orbital (cumulative) for each atom.
        delta_corner : NDArray, optional
            The shift of the corner, by default xp.array([0,0,0]). (only needed to compute the out-coupling mask)
        
        Returns
        -------
        mask_atoms : NDArray
            List of every atom in the contact.
        mask_orb : NDArray
            List of every orbital in the contact.
        """

        shifted_corner_1 = self.corner_1 + delta_corner #Shift the first corner (only needed for the out-coupling mask)
        shifted_corner_2 = self.corner_2 + delta_corner #Shift the second corner (only needed for the out-coupling mask)

        #Compute the atoms inside the contact element
        mask_atoms = coords[:,0] >= shifted_corner_1[0] 
        mask_atoms &= coords[:,0] < shifted_corner_2[0]
        mask_atoms &= coords[:,1] >= shifted_corner_1[1]
        mask_atoms &= coords[:,1] < shifted_corner_2[1]
        mask_atoms &= coords[:,2] >= shifted_corner_1[2]
        mask_atoms &= coords[:,2] < shifted_corner_2[2]
        mask_atoms = xp.nonzero(mask_atoms)[0]

        #Compute the orbitals inside the contact element
        mask_orb = xp.array([],dtype=xp.int32)
        for i in range(mask_atoms.shape[0]):
            #NEED TO MOVE THE INDEX ON THE CPU
            #I USED A QUICK WORKAROUND FOR NOW
            index = int(mask_atoms[i].get() if hasattr(mask_atoms[i], 'get') else mask_atoms[i])
            k1 = int(orbitals[index].get() if hasattr(orbitals[index], 'get') else orbitals[index])
            k2 = int(orbitals[index+1].get() if hasattr(orbitals[index + 1], 'get') else orbitals[index + 1])
            mask_orb = xp.concatenate((mask_orb,xp.arange(k1, k2)))
        
        return mask_atoms[None,:], mask_orb[None,:]


    def set_mask(self,coords: NDArray, orbitals: NDArray):
        """
        Sets the mask for the contact element.

        Parameters
        ----------
        coords : NDArray
            The atomic coordinates.
        orbitals : NDArray
            The starting orbital (cumulative) for each atom.
        """
        
        #Compute mask for the contact element   
        self.mask_atoms_i, self.mask_orb_i = self.compute_mask(coords,orbitals)

        #Compute mask for the shifted contact element (needed for off-diagonal elements)
        delta_corner = xp.multiply(-self.direction,(self.corner_2-self.corner_1))
        self.mask_atoms_j, self.mask_orb_j = self.compute_mask(coords,orbitals,delta_corner)

        if self.mask_atoms_i.shape[1] != self.mask_atoms_j.shape[1]:
            raise ValueError(
                f"The number of {self.name} contact atoms in the slab i ({self.mask_atoms_i.shape[1]}) and slab j ({self.mask_atoms_j.shape[1]}) are different."
            )

        print(f"Contact {self.name} has {self.mask_atoms_i.shape[1]} atoms and {self.mask_orb_i.shape[1]} orbitals", flush=True) if comm.rank == 0 else None
    

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

        #Load potential (TODO)

        self.flatband = quatrex_config.electron.flatband
        self.eta_obc = quatrex_config.electron.eta_obc

        # Load contact info
        self.n_cont,self.cont_names,self.corner1,self.corner2,self.corner_direction = distributed_load_contact(quatrex_config.input_dir / "cont.dat")

        # CREATE CONTACT LIST
        self.contacts = []
        for n in range(self.n_cont):
            self.contacts.append(Contact(self.corner1[n],self.corner2[n],self.corner_direction[n],self.cont_names[n]))
            self.contacts[n].set_mask(self.coords, self.orbitals_vec)

        # CREATE MASK FOR EVERY SLAB
        self.n_slabs_x, self.n_slab_y = distributed_read_slabs(quatrex_config.input_dir / "slabs.dat")
        self.slab_mask_x_at, self.slab_mask_x_orb = compute_slab_mask_X(self.coords,self.n_slabs_x,self.orbitals_vec)

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
    
    def compute_observables(self,phi: NDArray, inj_ind: list, i: int, w: NDArray):
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

        #Compute transmissions for all the possible contact couples
        cont_1 = 0
        cont_2 = 1
        for n in range(self.n_transmissions):

            #Get the all the wavefunction injected from contact 1 and extract the elements inside contact 2
            phi_1 = phi[self.contacts[cont_2].mask_orb_j.T,inj_ind[cont_1]]
            phi_2 = phi[self.contacts[cont_2].mask_orb_i.T,inj_ind[cont_1]]

            #Get the transmission matrix between the two contact blocks
            T01 = self.system_matrix[self.contacts[cont_2].mask_orb_j.T,self.contacts[cont_2].mask_orb_i]

            #Compute the transmission
            if(phi_1.size != 0):
                self.observables.electron_transmission_contacts[n,i] = xp.trace(2*xp.imag(phi_1.T.conj() @ T01 @phi_2))

            cont_2 += 1
            if cont_2 == self.n_cont:
                cont_1 += 1
                cont_2 = cont_1+1

        #Compute transmission for all the x slabs and all the contacts
        for n in range(self.n_cont):
            for s in range(self.n_slabs_x-1):

                #For every slab, get the wavefunction injected from the contact
                phi_1 = phi[self.slab_mask_x_orb[s].T,inj_ind[n]]
                phi_2 = phi[self.slab_mask_x_orb[s+1].T,inj_ind[n]]

                #Get the transmission matrix between the slab and the next one
                T01 = self.system_matrix[self.slab_mask_x_orb[s].T,self.slab_mask_x_orb[s+1]]

                if(phi_1.size != 0):
                    self.observables.electron_transmission_x_slabs[n,s,i] = xp.trace(2*xp.imag(phi_1.T.conj() @ T01 @phi_2))

        #Compute DOS
        phi_ortho = self.overlap_sparray @ phi
        for n in range(self.n_cont):
            S_off_coup_cont = self.overlap_sparray[self.contacts[n].mask_orb_j.T,self.contacts[n].mask_orb_i]
            w_sign = w.copy()
            w_sign[inj_ind[n]] = 1/w_sign[inj_ind[n]]
            
            phi_ortho[self.contacts[n].mask_orb_i.squeeze(),:] += S_off_coup_cont @ xp.multiply(phi[self.contacts[n].mask_orb_i.squeeze(),:],w_sign)

        for n in range(self.n_cont):
            for s in range(self.n_slabs_x):

                phi_D = phi[self.slab_mask_x_orb[s].T,inj_ind[n]].squeeze()
                phi_D_ortho = phi_ortho[self.slab_mask_x_orb[s].T,inj_ind[n]].squeeze()
                if(phi_D.size != 0):
                    self.observables.electron_DOS_x_slabs[n,s,i]=xp.real(xp.sum(xp.multiply(phi_D.conj(), phi_D_ortho))/(2*xp.pi))

    def run(self) -> None:
        """Runs the QTBM"""
        print("Entering QTBM calculation", flush=True) if comm.rank == 0 else None
        times = []
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

            S = []
            inj = []
            inj_ind = []
            w = []
            # Compute the boundary self-energy and the injection vector
            ind_0 = 0
            for n in range(self.n_cont):
                S_n, inj_n, w_n = self.obc(
                    self.system_matrix[self.contacts[n].mask_orb_i.T,self.contacts[n].mask_orb_i].toarray(),
                    self.system_matrix[self.contacts[n].mask_orb_i.T,self.contacts[n].mask_orb_j].toarray(),
                    self.system_matrix[self.contacts[n].mask_orb_j.T,self.contacts[n].mask_orb_i].toarray(),
                    "left",
                    return_inj = True,
                )
                S.append(S_n)
                inj.append(inj_n)
                inj_ind.append(xp.arange(ind_0,ind_0+inj_n.shape[1])[None,:])
                ind_0 += inj_n.shape[1]
                w.append(w_n)

            w = xp.concatenate(w)

            t_solve = time.perf_counter() - times.pop()
            (
                print(f"Time for OBC: {t_solve:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            times.append(time.perf_counter())

            # Set up sytem matrix and rhs for electron solver.
            inj_V = xp.zeros((self.system_matrix.shape[0],ind_0), dtype=xp.complex128) #Set the injection vector as a zero matrix

            #Iterate over contacts
            for n in range(self.n_cont):
                self.system_matrix[self.contacts[n].mask_orb_i.T,self.contacts[n].mask_orb_i] -= S[n] #Subtract the self-energy in the contact elements
                inj_V[self.contacts[n].mask_orb_i.T,inj_ind[n]] = inj[n] #Add the injection vector in the contact elements of the rhs

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
            phi = spsolve(self.system_matrix, inj_V)

            t_solve = time.perf_counter() - times.pop()
            (
                print(f"Time for electron solver: {t_solve:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            # Get the bare system matrix back, needed for transmission calculation 
            for n in range(self.n_cont):
                self.system_matrix[self.contacts[n].mask_orb_i.T,self.contacts[n].mask_orb_i] += S[n] #Add the self-energy back
            
            # Compute observables (DOS and Transmission)
            self.compute_observables(phi,inj_ind,i,w)

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