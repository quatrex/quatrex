# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time
from dataclasses import dataclass, field

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp
from qttools.utils.mpi_utils import distributed_load, get_local_slice

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.device import Device
from quatrex.core.quatrex_config import QuatrexConfig

# ------------------------------------------------------------
# TODO: This can be refactored into a common solver interface.
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
    from scipy.sparse.linalg import splu
if xp.__name__ == "cupy":
    from cupyx.scipy.sparse.linalg import splu

# ------------------------------------------------------------


def monkhorst_pack(size: tuple[int]) -> NDArray:
    """Constructs a Monkhorst-Pack grid of k-points.

    Parameters
    ----------
    size : tuple
        The size of the grid in each direction.
    Returns
    -------
    kpts : NDArray
        The Monkhorst-Pack grid of k-points.

    """
    kpts = np.indices(size).transpose((1, 2, 3, 0)).reshape((-1, 3))
    return (kpts + 0.5) / size - 0.5


def compute_slab_vector_x(
    atom_coords: NDArray, num_slabs: int, orbital_offsets: NDArray, dx: float = 1e-3
) -> tuple[NDArray, NDArray]:
    """Computes the elements (atom,orbitals) for each slab in the x direction.

    Parameters
    ----------
    coords : NDArray
        The atomic coordinates.
    n_slabs : int
        The number of slabs in the x direction.
    orbitals : NDArray
        The starting orbital (cumulative) for each atom.
    dx is needed to allow every atom to be included in one slab
        dx = 0.001

    Returns
    -------
    vec_atoms : NDArray
        Every atom in each slab.
    vec_orb : NDArray
        Every orbital in each slab.

    """
    # Get the min and max x coordinates
    x_min = atom_coords[:, 0].min()
    x_max = atom_coords[:, 0].max()

    edges = np.linspace(x_min, x_max, num_slabs + 1, endpoint=True) - 1e-3
    atom_slab_inds = np.digitize(atom_coords[:, 0], edges) - 1
    atom_inds = [np.where(atom_slab_inds == i)[0] for i in range(num_slabs)]

    # TODO: A bit clunky to have to recompute the number of orbitals per
    # atom every time.
    orbitals_per_atom = np.diff(orbital_offsets)
    orbital_coords_x = np.repeat(atom_coords[:, 0], orbitals_per_atom, axis=0)

    orbital_slab_inds = np.digitize(orbital_coords_x, edges) - 1
    # TODO: Orbital indices need to have a new axis for some reason.
    orbital_inds = [
        np.where(orbital_slab_inds == i)[0][np.newaxis] for i in range(num_slabs)
    ]

    return atom_inds, orbital_inds


@dataclass
class Observables:
    """Observable quantities for the QTBM."""

    electron_ldos: NDArray = None
    electron_density: NDArray = None
    hole_density: NDArray = None
    electron_current: dict = field(default_factory=dict)

    spill_over_error: NDArray = None

    electron_transmission_contacts: NDArray = None
    electron_transmission_contacts_labels = []

    electron_transmission_x_slabs: NDArray = None

    electron_dos_x_slabs: NDArray = None

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
        self.num_contacts = len(device.contacts)
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
        self.num_slabs_x, self.num_slabs_y = quatrex_config.device.num_slabs

        self.slab_vec_x_at, self.slab_vec_x_orb = compute_slab_vector_x(
            device.coords, self.num_slabs_x, device.orbital_offsets
        )

        # Look for all the combinations of contacts
        self.num_transmissions = int((self.num_contacts**2 - self.num_contacts) / 2)
        cont_1 = 0
        cont_2 = 1
        for __ in range(self.num_transmissions):
            # Append the label for every transmission
            self.observables.electron_transmission_contacts_labels.append(
                device.contacts[cont_1].name[0] + "->" + device.contacts[cont_2].name[0]
            )
            cont_2 += 1
            if cont_2 == self.num_contacts:
                cont_1 += 1
                cont_2 = cont_1 + 1

        # Initialize the observables
        self.observables.electron_transmission_contacts = xp.zeros(
            (self.num_transmissions, self.local_energies.shape[0]), dtype=xp.float64
        )
        self.observables.electron_transmission_x_slabs = xp.zeros(
            (self.num_contacts, self.num_slabs_x, self.local_energies.shape[0]),
            dtype=xp.float64,
        )
        self.observables.electron_dos_x_slabs = xp.zeros(
            (self.num_contacts, self.num_slabs_x, self.local_energies.shape[0]),
            dtype=xp.float64,
        )

        self.num_orbitals = self.device.hamiltonian[0, 0, 0].shape[0]

        # TODO: Hamiltonian should be assembled for each k-point.
        # (This can easily be vectorized)
        self.hamiltonian_phase = sparse.csr_matrix(
            self.device.hamiltonian[0, 0, 0].shape, dtype=xp.complex128
        )
        for r, h_r in self.device.hamiltonian.items():
            self.hamiltonian_phase += h_r * xp.exp(
                1j * self.k[0] * r[0] + 1j * self.k[1] * r[1] + 1j * self.k[2] * r[2]
            )
        self.overlap_phase = sparse.csr_matrix(
            self.device.overlap[0, 0, 0].shape, dtype=xp.complex128
        )
        for r, s_r in self.device.overlap.items():
            self.overlap_phase += s_r * xp.exp(
                1j * self.k[0] * r[0] + 1j * self.k[1] * r[1] + 1j * self.k[2] * r[2]
            )

    def compute_observables(
        self, phi: NDArray, inj_ind: list, i: int, S: list, K, T, E
    ):
        """Computes observables for the current iteration.

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
        for n in range(self.num_transmissions):

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
                if comm.rank == 0:
                    print(
                        f"Transmission {self.observables.electron_transmission_contacts_labels[n]}: {self.observables.electron_transmission_contacts[n, i]}",
                        flush=True,
                    )

            cont_2 += 1
            if cont_2 == self.num_contacts:
                cont_1 += 1
                cont_2 = cont_1 + 1

        # Compute transmission for all the x slabs and all the contacts
        for n in range(self.num_contacts):
            for s in range(self.num_slabs_x - 1):

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
        for n, contact in enumerate(self.device.contacts):

            phi_cont = (
                K[contact.orbitals_contact.squeeze(), :]
                + T[n] @ phi[contact.orbitals_contact.squeeze(), :]
            )
            # TODO Add the spill over contribution
            phi_ortho[contact.orbitals_contact.squeeze(), :] += (
                contact.get_10(self.overlap_phase) @ phi_cont
            )
            # CHECK SPILL OVER ERROR (DEBUG)
            error = xp.linalg.norm(
                (
                    E * contact.get_10(self.overlap_phase)
                    - contact.get_10(self.hamiltonian_phase)
                )
                @ phi_cont
                + self.system_matrix[contact.orbitals_contact.squeeze(), :] @ phi
            )
            print(
                f"    Spill over error for contact {contact.name[0]} at energy {E}: {error}"
            )

        # Compute the DOS for every injected wavefunction
        for n in range(self.num_contacts):
            for s in range(self.num_slabs_x):
                phi_D = phi[
                    self.slab_vec_x_orb[s].T, inj_ind[n]
                ].squeeze()  # Get the wavefunction in the slab
                phi_D_ortho = phi_ortho[
                    self.slab_vec_x_orb[s].T, inj_ind[n]
                ].squeeze()  # Get the "orthogonalized" wavefunction in the slab
                if phi_D.size != 0:
                    self.observables.electron_dos_x_slabs[n, s, i] = xp.real(
                        xp.sum(xp.multiply(phi_D.conj(), phi_D_ortho)) / (2 * xp.pi)
                    )  # Compute the DOS

    def run(self) -> None:
        """Runs the QTBM calculation."""
        if comm.rank == 0:
            print("Entering QTBM calculation", flush=True)

        times = []
        comm.Barrier()
        OBC_batch_size = 10
        self.system_matrix = None  # Initialize the system matrix

        times.append(time.perf_counter())

        for batch_start in range(0, len(self.local_energies), OBC_batch_size):
            energy_batch = self.local_energies[
                batch_start : batch_start + OBC_batch_size
            ]

            if comm.rank == 0:
                print(
                    f"Processing energies {batch_start} to {batch_start + len(energy_batch) - 1}",
                    flush=True,
                )

            # append for iteration time
            times.append(time.perf_counter())

            times.append(time.perf_counter())

            sigma_obcs = []
            injs = []
            inj_inds = []
            Ks = []
            Ts = []
            # Compute the boundary self-energy and the injection vector.
            ind_0 = np.zeros(len(energy_batch), dtype=np.int32)
            for contact in self.device.contacts:
                times.append(time.perf_counter())

                sigma_obc, inj, num_inj, T, K = contact.compute_boundary(
                    self.k[0], self.k[1], self.k[2], energy_batch
                )

                sigma_obcs.append(sigma_obc)
                injs.append(inj)
                Ks.append(K)
                Ts.append(T)

                # For every energy in batch, compute a list with the
                # indices of every injected vector.
                inj_ind_temp = []
                for i in range(len(energy_batch)):
                    i0 = ind_0[i]
                    n_i = (
                        num_inj[i].get().item()
                        if hasattr(num_inj[i], "get")
                        else num_inj[i].item()
                    )
                    inj_ind_temp.append(xp.arange(i0, i0 + n_i))
                inj_inds.append(inj_ind_temp)
                ind_0 += num_inj

                t_solve = time.perf_counter() - times.pop()
                if comm.rank == 0:
                    print(
                        f"Time for OBC in contact {contact.name[0]}: {t_solve:.2f} s",
                        flush=True,
                    )

            t_solve = time.perf_counter() - times.pop()
            if comm.rank == 0:
                print(f"Time for OBC: {t_solve:.2f} s", flush=True)

            for i, energy in enumerate(energy_batch):

                times.append(time.perf_counter())

                # Set up sytem matrix and rhs for electron solver.
                i0 = (
                    ind_0[i].get().item()
                    if hasattr(ind_0[i], "get")
                    else ind_0[i].item()
                )
                inj_V = xp.zeros(
                    (self.num_orbitals, i0), dtype=xp.complex128, order="F"
                )  # Set the injection vector as a zero matrix
                K_V = xp.zeros(
                    (self.num_orbitals, i0), dtype=xp.complex128, order="F"
                )  # Set the K vector as a zero matrix

                ind1 = []
                ind2 = []
                sig_flat = []
                # Iterate over contacts
                for contact, sigma_obc, inj, inj_ind, K in zip(
                    self.device.contacts, sigma_obcs, injs, inj_inds, Ks
                ):
                    ind1.append(
                        xp.repeat(
                            contact.orbitals_contact.squeeze(),
                            contact.orbitals_contact.shape[1],
                        )
                    )
                    ind2.append(
                        xp.tile(
                            contact.orbitals_contact.squeeze(),
                            contact.orbitals_contact.shape[1],
                        )
                    )
                    sig_flat.append(sigma_obc[i, :, :].flatten())
                    # Add the injection vector in the contact elements of the rhs
                    inj_V[contact.orbitals_contact.T, inj_ind[i]] = inj[i]
                    # Add the K vector in the contact elements of the rhs
                    K_V[contact.orbitals_contact.T, inj_ind[i]] = K[i]

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
                        energy * self.overlap_phase - self.hamiltonian_phase - upd_0
                    )
                else:
                    self.system_matrix.data[:] = (
                        energy * self.overlap_phase - self.hamiltonian_phase - upd_0
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
                if comm.rank == 0:
                    print(f"Time to set up system of eq.: {t_solve:.2f} s", flush=True)

                times.append(time.perf_counter())
                # Solve for the wavefunction

                if inj_V.size != 0:
                    if CUDSS_AVAILABLE and xp.__name__ == "cupy":
                        # USE CUDSS
                        phi = cuDSS.spsolve_with_CUDSS(self.system_matrix, inj_V)
                    else:
                        if MUMPS_AVAILABLE:
                            # USE MUMPS
                            inst = mumps.Context()
                            t_mumps = time.perf_counter()
                            inst.analyze(self.system_matrix)
                            t_analyze = time.perf_counter() - t_mumps
                            if comm.rank == 0:
                                print(
                                    f"Time for MUMPS analyze: {t_analyze:.2f} s",
                                    flush=True,
                                )

                            t_mumps = time.perf_counter()
                            inst.factor(self.system_matrix)
                            t_factor = time.perf_counter() - t_mumps

                            if comm.rank == 0:
                                print(
                                    f"Time for MUMPS factor: {t_factor:.2f} s",
                                    flush=True,
                                )

                            t_mumps = time.perf_counter()
                            phi = inst.solve(inj_V)
                            t_solve = time.perf_counter() - t_mumps
                            if comm.rank == 0:
                                print(
                                    f"Time for MUMPS solve: {t_solve:.2f} s", flush=True
                                )

                        else:
                            lu = splu(self.system_matrix)
                            phi = lu.solve(inj_V)

                t_solve = time.perf_counter() - times.pop()
                if comm.rank == 0:
                    print(f"Time for electron solver: {t_solve:.2f} s", flush=True)
                times.append(time.perf_counter())
                # Get the bare system matrix back, needed for transmission calculation
                upd_0.data[:] = (
                    1e-15  # Set a small value to the self-energy matrix to avoid numerical issues
                )
                self.system_matrix.data[:] = (
                    energy * self.overlap_phase - self.hamiltonian_phase - upd_0
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
                    for nn in range(self.num_contacts):
                        sigma_b_t.append(sigma_obcs[nn][i, :, :])
                        inj_ind_t.append(inj_inds[nn][i])
                        T_t.append(Ts[nn][i, :, :])
                    self.compute_observables(
                        phi, inj_ind_t, batch_start + i, sigma_b_t, K_V, T_t, energy
                    )

                t_iteration = time.perf_counter() - times.pop()
                if comm.rank == 0:
                    print(
                        f"Time for computing observables: {t_iteration:.2f} s",
                        flush=True,
                    )

            t_iteration = time.perf_counter() - times.pop()
            if comm.rank == 0:
                print(f"Time for iteration: {t_iteration:.2f} s", flush=True)

        t_iteration = time.perf_counter() - times.pop()
        if comm.rank == 0:
            print(f"Time for QTBM: {t_iteration:.2f} s", flush=True)

        # Gather the observables
        comm.Barrier()
        self.observables.electron_transmission_x_slabs = xp.concatenate(
            comm.allgather(self.observables.electron_transmission_x_slabs), axis=-1
        )
        self.observables.electron_transmission_contacts = xp.hstack(
            comm.allgather(self.observables.electron_transmission_contacts)
        )
        self.observables.electron_dos_x_slabs = xp.concatenate(
            comm.allgather(self.observables.electron_dos_x_slabs), axis=-1
        )
