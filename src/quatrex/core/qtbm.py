# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

# import os
import time
from dataclasses import dataclass, field
from pathlib import Path

# from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp
from qttools.utils.mpi_utils import distributed_load

from quatrex.core.device import Device

# from scipy import sparse as sp_sparse

try:
    from qttools.cuDSS_binding.cudss_wrapp import CuDSS

    CUDSS_AVAILABLE = True
    print("CUDSS available") if comm.rank == 0 else None
    cuDSS = CuDSS()
except ImportError:
    CUDSS_AVAILABLE = False


try:
    import mumps

    MUMPS_AVAILABLE = True
    print("MUMPS available") if comm.rank == 0 else None
except ImportError:
    MUMPS_AVAILABLE = False


if xp.__name__ == "numpy":
    from scipy.sparse.linalg import splu  # , spsolve
if xp.__name__ == "cupy":
    from cupyx.scipy.sparse.linalg import splu  # , spsolve

from qttools.utils.mpi_utils import get_local_slice

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.statistics import fermi_dirac


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
        slabs = xp.loadtxt(
            filename, dtype=xp.int32
        )  # Read the number of slabs in x and y
    else:
        slabs = None

    slabs = comm.bcast(slabs, root=0)  # Broadcast the data to all the ranks

    # Move the data to the CPU (it is just a single number)
    slab_x = slabs[0].get().item() if hasattr(slabs[0], "get") else slabs[0].item()
    slab_y = slabs[1].get().item() if hasattr(slabs[1], "get") else slabs[1].item()

    return slab_x, slab_y


def compute_slab_vector_X(
    coords: NDArray, n_slabs: int, orbitals: NDArray
) -> tuple[NDArray, NDArray]:
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

    # dx is needed to allow every atom to be included in one slab slab
    dx = 0.001

    # Get the min and max x coordinates
    xMin = coords[:, 0].min()
    xMax = coords[:, 0].max()

    # Compute the width of each slab
    t_slab = (xMax - xMin) / n_slabs

    # Assign some group of atoms to each slab
    for i in range(n_slabs):
        if i != n_slabs - 1:
            vec_atoms.append(
                xp.nonzero(
                    xp.logical_and(
                        coords[:, 0] >= xMin + i * t_slab - dx,
                        coords[:, 0] < xMin + (i + 1) * t_slab - dx,
                    )
                )[0]
            )
        else:
            vec_atoms.append(
                xp.nonzero(
                    xp.logical_and(
                        coords[:, 0] >= xMin + i * t_slab - dx,
                        coords[:, 0] <= xMin + (i + 1) * t_slab + dx,
                    )
                )[0]
            )

    # Assign the orbitals to each slab
    for i in range(n_slabs):
        vec_orb_loc = xp.array([], dtype=xp.int32)
        for j in range(vec_atoms[i].shape[0]):
            # NEED TO MOVE THE INDEX ON THE CPU
            # I USED A QUICK WORKAROUND FOR NOW
            index = int(
                vec_atoms[i][j].get()
                if hasattr(vec_atoms[i][j], "get")
                else vec_atoms[i][j]
            )
            k1 = int(
                orbitals[index].get()
                if hasattr(orbitals[index], "get")
                else orbitals[index]
            )
            k2 = int(
                orbitals[index + 1].get()
                if hasattr(orbitals[index + 1], "get")
                else orbitals[index + 1]
            )
            vec_orb_loc = xp.concatenate((vec_orb_loc, xp.arange(k1, k2)))

        vec_orb.append(vec_orb_loc[None, :])
        (
            print(
                f"Slab {i} has {vec_atoms[i].shape[0]} atoms and {vec_orb[i].shape[1]} orbitals",
                flush=True,
            )
            if comm.rank == 0
            else None
        )

    return vec_atoms, vec_orb


@dataclass
class Observables:
    """Observable quantities for the SCBA."""

    # --- Electrons ----------------------------------------------------
    electron_ldos: NDArray = None
    electron_density: NDArray = None
    hole_density: NDArray = None
    electron_current: dict = field(default_factory=dict)

    spill_over_error: NDArray = None

    electron_transmission_contacts: NDArray = None
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

    def __init__(
        self,
        device: Device,
        k: tuple,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig | None = None,
    ) -> None:
        """Initializes a QTBM instance."""

        self.device = device
        self.n_cont = len(device.contacts)
        self.quatrex_config = quatrex_config
        self.k = k  # The wavevector for the QTBM

        if self.device.gamma_only and (
            self.k[0] != 0 or self.k[1] != 0 or self.k[2] != 0
        ):
            raise ValueError(
                "The device only has a Gamma point Hamiltonian, "
                "but the wavevector is not (0,0,0)."
            )

        if compute_config is None:
            compute_config = ComputeConfig()

        self.compute_config = compute_config

        self.observables = Observables()

        self.flatband = quatrex_config.electron.flatband
        self.eta_obc = quatrex_config.electron.eta_obc
        self.block_sections = quatrex_config.electron.obc.block_sections

        # Load the electron energies.
        self.electron_energies = distributed_load(
            quatrex_config.input_dir / "electron_energies.npy"
        )
        # Get the local slice of the electron energies
        self.local_energies = get_local_slice(self.electron_energies)

        # CREATE VECTORS FOR EVERY SLAB
        self.n_slabs_x, self.n_slab_y = distributed_read_slabs(
            quatrex_config.input_dir / "slabs.dat"
        )
        self.slab_vec_x_at, self.slab_vec_x_orb = compute_slab_vector_X(
            device.coords, self.n_slabs_x, device.orbitals_vec
        )

        # Look for all the combinations of contacts
        self.n_transmissions = int((self.n_cont**2 - self.n_cont) / 2)
        cont_1 = 0
        cont_2 = 1
        for n in range(self.n_transmissions):
            # Append the label for every transmission
            self.observables.electron_transmission_contacts_labels.append(
                device.contacts[cont_1].name[0] + "->" + device.contacts[cont_2].name[0]
            )
            cont_2 += 1
            if cont_2 == self.n_cont:
                cont_1 += 1
                cont_2 = cont_1 + 1

        # Initialize the observables
        self.observables.electron_transmission_contacts = xp.zeros(
            (self.n_transmissions, self.local_energies.shape[0]), dtype=xp.float64
        )
        self.observables.electron_transmission_x_slabs = xp.zeros(
            (self.n_cont, self.n_slabs_x, self.local_energies.shape[0]),
            dtype=xp.float64,
        )
        self.observables.electron_DOS_x_slabs = xp.zeros(
            (self.n_cont, self.n_slabs_x, self.local_energies.shape[0]),
            dtype=xp.float64,
        )

        # Band edges and Fermi levels.
        # TODO: This only works for small potential variations accross
        # the device.
        # TODO: During this initialization we should compute the contact
        # band structures and extract the correct fermi levels & band
        # edges from there.
        # self.band_edge_tracking = quatrex_config.electron.band_edge_tracking
        # self.delta_fermi_level_conduction_band = (
        #    quatrex_config.electron.conduction_band_edge
        #    - quatrex_config.electron.fermi_level
        # )
        # self.left_mid_gap_energy = quatrex_config.electron.left_fermi_level
        # self.right_mid_gap_energy = quatrex_config.electron.right_fermi_level

        self.temperature = quatrex_config.electron.temperature

        self.left_fermi_level = quatrex_config.electron.left_fermi_level
        self.right_fermi_level = quatrex_config.electron.right_fermi_level

        self.left_occupancies = fermi_dirac(
            self.local_energies - self.left_fermi_level, self.temperature
        )
        self.right_occupancies = fermi_dirac(
            self.local_energies - self.right_fermi_level, self.temperature
        )

        self.sys_mat_shape = self.device.hamiltonian[(0, 0, 0)].shape[0]

        self.reuse_sym = 0

        self.hamiltonian_phase = sparse.csr_matrix(
            self.device.hamiltonian[(0, 0, 0)].shape, dtype=xp.complex128
        )
        for key, value in self.device.hamiltonian.items():
            self.hamiltonian_phase += value * xp.exp(
                1j * self.k[0] * key[0]
                + 1j * self.k[1] * key[1]
                + 1j * self.k[2] * key[2]
            )
        self.overlap_phase = sparse.csr_matrix(
            self.device.overlap[(0, 0, 0)].shape, dtype=xp.complex128
        )
        for key, value in self.device.overlap.items():
            self.overlap_phase += value * xp.exp(
                1j * self.k[0] * key[0]
                + 1j * self.k[1] * key[1]
                + 1j * self.k[2] * key[2]
            )

    def compute_observables(
        self, phi: NDArray, inj_ind: list, i: int, S: list, K, T, E
    ):
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
        # Compute transmissions for all the possible contact couples

        cont_1 = 0
        cont_2 = 1
        for n in range(self.n_transmissions):

            # Get the all the wavefunctions injected from contact 1 and extract the elements inside contact 2
            phi_n = phi[
                self.device.contacts[cont_2].orbitals_contact.T, inj_ind[cont_1]
            ]

            # Compute the transmission
            if phi_n.size != 0:
                self.observables.electron_transmission_contacts[n, i] = xp.trace(
                    xp.real(
                        1j * phi_n.T.conj() @ (S[cont_2] - S[cont_2].T.conj()) @ phi_n
                    )
                )

                (
                    print(
                        f"Transmission {self.observables.electron_transmission_contacts_labels[n]}: {self.observables.electron_transmission_contacts[n, i]}",
                        flush=True,
                    )
                    if comm.rank == 0
                    else None
                )

            cont_2 += 1
            if cont_2 == self.n_cont:
                cont_1 += 1
                cont_2 = cont_1 + 1

        # Compute transmission for all the x slabs and all the contacts
        for n in range(self.n_cont):
            for s in range(self.n_slabs_x - 1):

                # For every slab, get the wavefunction injected from the contact
                phi_1 = phi[self.slab_vec_x_orb[s].T, inj_ind[n]]
                phi_2 = phi[self.slab_vec_x_orb[s + 1].T, inj_ind[n]]

                # Get the transmission matrix between the slab and the next one
                T01 = self.system_matrix[
                    self.slab_vec_x_orb[s].T, self.slab_vec_x_orb[s + 1]
                ]

                if phi_1.size != 0:
                    self.observables.electron_transmission_x_slabs[n, s, i] = xp.trace(
                        2 * xp.imag(phi_1.T.conj() @ T01 @ phi_2)
                    )

        phi_ortho = self.overlap_phase @ phi  # "Orthogonalize" the wavefunction
        for n in range(self.n_cont):

            phi_cont = (
                K[self.device.contacts[n].orbitals_contact.squeeze(), :]
                + T[n] @ phi[self.device.contacts[n].orbitals_contact.squeeze(), :]
            )
            # TODO Add the spill over contribution
            phi_ortho[self.device.contacts[n].orbitals_contact.squeeze(), :] += (
                self.device.contacts[n].get_10(self.overlap_phase) @ phi_cont
            )
            # CHECK SPILL OVER ERROR (DEBUG)
            error = xp.linalg.norm(
                (
                    E * self.device.contacts[n].get_10(self.overlap_phase)
                    - self.device.contacts[n].get_10(self.hamiltonian_phase)
                )
                @ phi_cont
                + self.system_matrix[
                    self.device.contacts[n].orbitals_contact.squeeze(), :
                ]
                @ phi
            )
            print(
                f"    Spill over error for contact {self.device.contacts[n].name[0]} at energy {E}: {error}"
            )

        # Compute the DOS for every injected wavefunction
        for n in range(self.n_cont):
            for s in range(self.n_slabs_x):
                phi_D = phi[
                    self.slab_vec_x_orb[s].T, inj_ind[n]
                ].squeeze()  # Get the wavefunction in the slab
                phi_D_ortho = phi_ortho[
                    self.slab_vec_x_orb[s].T, inj_ind[n]
                ].squeeze()  # Get the "orthogonalized" wavefunction in the slab
                if phi_D.size != 0:
                    self.observables.electron_DOS_x_slabs[n, s, i] = xp.real(
                        xp.sum(xp.multiply(phi_D.conj(), phi_D_ortho)) / (2 * xp.pi)
                    )  # Compute the DOS

    def run(self) -> None:
        """Runs the QTBM"""
        print("Entering QTBM calculation", flush=True) if comm.rank == 0 else None
        times = []
        comm.Barrier()
        OBC_batch_size = 10
        self.system_matrix = None  # Initialize the system matrix

        times.append(time.perf_counter())

        for batch_start in range(0, len(self.local_energies), OBC_batch_size):
            E_batch_OBC = self.local_energies[
                batch_start : batch_start + OBC_batch_size
            ]

            (
                print(
                    f"Processing energies {batch_start} to {batch_start + len(E_batch_OBC) - 1}",
                    flush=True,
                )
                if comm.rank == 0
                else None
            )

            # append for iteration time
            times.append(time.perf_counter())

            times.append(time.perf_counter())
            sigma_b = []
            inj = []
            inj_ind = []
            K = []
            T = []
            # Compute the boundary self-energy and the injection vector
            ind_0 = xp.zeros(len(E_batch_OBC), dtype=xp.int32)
            for n in range(self.n_cont):
                times.append(time.perf_counter())

                sigma_b_cont, inj_cont, num_inj, T_cont, K_cont = self.device.contacts[
                    n
                ].compute_boundary(self.k[0], self.k[1], self.k[2], E_batch_OBC)

                sigma_b.append(sigma_b_cont)
                inj.append(inj_cont)
                K.append(K_cont)
                T.append(T_cont)

                # For every Energy in batch, compute a list with the indeces of evert injected vector
                inj_ind_temp = []
                for i in range(len(E_batch_OBC)):
                    i0 = (
                        ind_0[i].get().item()
                        if hasattr(ind_0[i], "get")
                        else ind_0[i].item()
                    )
                    n_i = (
                        num_inj[i].get().item()
                        if hasattr(num_inj[i], "get")
                        else num_inj[i].item()
                    )
                    inj_ind_temp.append(xp.arange(i0, i0 + n_i))
                inj_ind.append(inj_ind_temp)
                ind_0 += num_inj

                t_solve = time.perf_counter() - times.pop()
                (
                    print(
                        f"Time for OBC in contact {self.device.contacts[n].name[0]}: {t_solve:.2f} s",
                        flush=True,
                    )
                    if comm.rank == 0
                    else None
                )

            t_solve = time.perf_counter() - times.pop()
            (
                print(f"Time for OBC: {t_solve:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            for i, E in enumerate(E_batch_OBC):

                times.append(time.perf_counter())

                # Set up sytem matrix and rhs for electron solver.
                i0 = (
                    ind_0[i].get().item()
                    if hasattr(ind_0[i], "get")
                    else ind_0[i].item()
                )
                inj_V = xp.zeros(
                    (self.sys_mat_shape, i0), dtype=xp.complex128, order="F"
                )  # Set the injection vector as a zero matrix
                K_V = xp.zeros(
                    (self.sys_mat_shape, i0), dtype=xp.complex128, order="F"
                )  # Set the K vector as a zero matrix

                ind1 = []
                ind2 = []
                sig_flat = []
                # Iterate over contacts
                for n in range(self.n_cont):
                    ind1.append(
                        xp.repeat(
                            self.device.contacts[n].orbitals_contact.squeeze(),
                            self.device.contacts[n].orbitals_contact.shape[1],
                        )
                    )
                    ind2.append(
                        xp.tile(
                            self.device.contacts[n].orbitals_contact.squeeze(),
                            self.device.contacts[n].orbitals_contact.shape[1],
                        )
                    )
                    sig_flat.append(sigma_b[n][i, :, :].flatten())
                    inj_V[
                        self.device.contacts[n].orbitals_contact.T, inj_ind[n][i]
                    ] = inj[n][
                        i
                    ]  # Add the injection vector in the contact elements of the rhs
                    K_V[self.device.contacts[n].orbitals_contact.T, inj_ind[n][i]] = K[
                        n
                    ][
                        i
                    ]  # Add the K vector in the contact elements of the rhs

                # Concatenate the indices and the self-energies
                ind1 = xp.concatenate(ind1)
                ind2 = xp.concatenate(ind2)
                sig_flat = xp.concatenate(sig_flat)

                upd_0 = sparse.coo_matrix(
                    (sig_flat, (ind1, ind2)), shape=self.hamiltonian_phase.shape
                ).tocsr()
                upd_0.eliminate_zeros()  # Remove zeros from the self-energy matrix
                if i == 0 and batch_start == 0:
                    self.system_matrix = (
                        E * self.overlap_phase - self.hamiltonian_phase - upd_0
                    )
                else:
                    self.system_matrix.data[:] = (
                        E * self.overlap_phase - self.hamiltonian_phase - upd_0
                    ).data

                # if i==0 and batch_start == 0:
                #    self.system_matrix =  E * self.overlap_phase - self.hamiltonian_phase
                # else:
                #    self.system_matrix.data[:] = - h_V
                #    self.system_matrix.data[:] += E * s_V

                # Update the system matrix with the self-energies
                # if i==0 and batch_start == 0:
                #    self.system_matrix -= upd_0

                #    self.system_matrix = self.system_matrix.tocoo()
                #    rows = self.system_matrix.row
                #    cols = self.system_matrix.col

                #    self.system_matrix = self.system_matrix.tocsr()

                #    if hasattr(self.hamiltonian_phase[rows, cols], 'A'):
                #        h_V = self.hamiltonian_phase[rows, cols].A.ravel()
                #        s_V = self.overlap_phase[rows, cols].A.ravel()
                #    else:
                #        h_V = self.hamiltonian_phase[rows, cols].ravel()
                #        s_V = self.overlap_phase[rows, cols].ravel()

                # else:
                #    if hasattr(upd_0.tocsr()[rows, cols], 'A'):
                #        self.system_matrix.data[:] -= upd_0.tocsr()[rows, cols].A.ravel()
                #    else:
                #        self.system_matrix.data[:] -= upd_0.tocsr()[rows, cols].ravel()

                t_solve = time.perf_counter() - times.pop()
                (
                    print(f"Time to set up system of eq.: {t_solve:.2f} s", flush=True)
                    if comm.rank == 0
                    else None
                )
                times.append(time.perf_counter())
                # Solve for the wavefunction

                if inj_V.size != 0:
                    if CUDSS_AVAILABLE and xp.__name__ == "cupy":
                        # USE CUDSS
                        phi = cuDSS.spsolve_with_CUDSS(self.system_matrix, inj_V)
                        self.reuse_sym = 1
                    else:
                        if MUMPS_AVAILABLE:
                            # USE MUMPS
                            inst = mumps.Context()
                            t_mumps = time.perf_counter()
                            inst.analyze(self.system_matrix)
                            t_analyze = time.perf_counter() - t_mumps
                            (
                                print(
                                    f"Time for MUMPS analyze: {t_analyze:.2f} s",
                                    flush=True,
                                )
                                if comm.rank == 0
                                else None
                            )

                            t_mumps = time.perf_counter()
                            inst.factor(self.system_matrix)
                            t_factor = time.perf_counter() - t_mumps
                            (
                                print(
                                    f"Time for MUMPS factor: {t_factor:.2f} s",
                                    flush=True,
                                )
                                if comm.rank == 0
                                else None
                            )

                            t_mumps = time.perf_counter()
                            phi = inst.solve(inj_V)
                            t_solve = time.perf_counter() - t_mumps
                            (
                                print(
                                    f"Time for MUMPS solve: {t_solve:.2f} s", flush=True
                                )
                                if comm.rank == 0
                                else None
                            )
                        else:
                            lu = splu(self.system_matrix)
                            phi = lu.solve(inj_V)

                t_solve = time.perf_counter() - times.pop()
                (
                    print(f"Time for electron solver: {t_solve:.2f} s", flush=True)
                    if comm.rank == 0
                    else None
                )
                times.append(time.perf_counter())
                # Get the bare system matrix back, needed for transmission calculation
                upd_0.data[:] = (
                    1e-15  # Set a small value to the self-energy matrix to avoid numerical issues
                )
                self.system_matrix.data[:] = (
                    E * self.overlap_phase - self.hamiltonian_phase - upd_0
                ).data
                # LL = upd_0.tocsr()[rows, cols]
                # if hasattr(LL, 'A'):
                #    self.system_matrix.data[:] += LL.A.ravel()
                # else:
                #    self.system_matrix.data[:] += LL.ravel()

                if inj_V.size != 0:
                    # Compute observables (DOS and Transmission)
                    sigma_b_t = []
                    inj_ind_t = []
                    T_t = []
                    for nn in range(self.n_cont):
                        sigma_b_t.append(sigma_b[nn][i, :, :])
                        inj_ind_t.append(inj_ind[nn][i])
                        T_t.append(T[nn][i, :, :])
                    self.compute_observables(
                        phi, inj_ind_t, batch_start + i, sigma_b_t, K_V, T_t, E
                    )

                t_iteration = time.perf_counter() - times.pop()
                (
                    print(
                        f"Time for computing observables: {t_iteration:.2f} s",
                        flush=True,
                    )
                    if comm.rank == 0
                    else None
                )

            t_iteration = time.perf_counter() - times.pop()
            (
                print(f"Time for iteration: {t_iteration:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

        t_iteration = time.perf_counter() - times.pop()
        (
            print(f"Time for QTBM: {t_iteration:.2f} s", flush=True)
            if comm.rank == 0
            else None
        )
        # Gather the observables
        comm.Barrier()
        self.observables.electron_transmission_x_slabs = xp.concatenate(
            comm.allgather(self.observables.electron_transmission_x_slabs), axis=-1
        )
        self.observables.electron_transmission_contacts = xp.hstack(
            comm.allgather(self.observables.electron_transmission_contacts)
        )
        self.observables.electron_DOS_x_slabs = xp.concatenate(
            comm.allgather(self.observables.electron_DOS_x_slabs), axis=-1
        )
