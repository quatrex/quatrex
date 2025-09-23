# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time
from dataclasses import dataclass, field

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, sparse, xp
from qttools.utils.mpi_utils import get_local_slice
from qttools.wave_function_solver import MUMPS, SuperLU, WFSolver, cuDSS
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.device import Device
from quatrex.core.energies import get_electron_energies
from quatrex.core.quatrex_config import QuatrexConfig, SolverConfig

preferred_matrix_type = {
    "mumps": sparse.coo_matrix,
    "superlu": sparse.csc_matrix,
    "cudss": sparse.csr_matrix,
}


def monkhorst_pack(size: tuple[int]) -> NDArray:
    """Constructs a Monkhorst-Pack grid of k-points.

    Parameters
    ----------
    size : tuple[int]
        Grid dimensions as (nx, ny, nz) specifying the number of
        k-points along each reciprocal lattice direction.

    Returns
    -------
    kpts : NDArray
        Array of k-points with shape (nx*ny*nz, 3). Each row contains
        the (kx, ky, kz) coordinates of a k-point in reduced units.


    """
    kpts = np.indices(size).transpose((1, 2, 3, 0)).reshape((-1, 3))
    return (kpts + 0.5) / size - 0.5


def compute_slab_vector_x(
    atom_coords: NDArray, num_slabs: int, orbital_offsets: NDArray, dx: float = 1e-3
) -> tuple[NDArray, NDArray]:
    """Computes spatial slabs for current/DOS along the x-direction.

    Divides the device into spatial slabs perpendicular to the x-axis
    and determines which atoms and orbitals belong to each slab.

    Parameters
    ----------
    atom_coords : NDArray
        Atomic coordinates. The x-coordinates are used for slab
        assignment.
    num_slabs : int
        Number of slabs to create along the x-direction.
    orbital_offsets : NDArray
        Cumulative orbital count array. Used to map from atoms to the
        corresponding orbitals.
    dx : float, optional
        Small offset applied to slab boundaries to ensure all atoms are
        included in exactly one slab, by default 1e-3.

    Returns
    -------
    atom_inds : list[np.ndarray]
        List of arrays, where each array contains indices of atoms in
        the corresponding slab. Length is `num_slabs`.
    orbital_inds : list[np.ndarray]
        List of arrays, where each array contains indices of orbitals in
        the corresponding slab. Length is `num_slabs`.

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
    """Container for transport observables from QTBM calculations.

    Attributes
    ----------
    electron_ldos : NDArray, optional
        Local density of states (LDOS) for electrons with shape
        (n_atoms, n_energies). Provides site-resolved DOS information.
    electron_density : NDArray, optional
        Electron density distribution with shape (n_atoms,).
    hole_density : NDArray, optional
        Hole density distribution with shape (n_atoms,).
    electron_current : dict, optional
        Dictionary containing current density information. Keys may
        include directional components and spatial distributions.
    spill_over_error : NDArray, optional
        Error metric for boundary condition accuracy with shape
        (n_energies,). Quantifies how well the open boundary conditions
        are satisfied.
    electron_transmission_contacts : NDArray, optional
        Contact-to-contact transmission coefficients with shape
        (n_contact_pairs, n_energies). Each element T_ij(E) gives the
        transmission probability from contact i to contact j at energy
        E.
    electron_transmission_contacts_labels : list[str]
        String labels for each contact pair in the format
        "source->drain" corresponding to the transmission matrix rows.
    electron_transmission_x_slabs : NDArray, optional
        Spatial transmission between adjacent slabs with shape
        (n_contacts, n_slabs-1, n_energies). Shows current flow as a
        function of position for each injection contact.
    electron_dos_x_slabs : NDArray, optional
        Position-resolved density of states with shape (n_contacts,
        n_slabs, n_energies). Provides spatial distribution of DOS for
        each injection contact.
    valence_band_edges : NDArray, optional
        Valence band edge energies with shape (n_atoms,).
    conduction_band_edges : NDArray, optional
        Conduction band edge energies with shape (n_atoms,).
    excess_charge_density : NDArray, optional
        Excess charge density distribution with shape (n_atoms,).

    """

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
    """Quantum Transmitting Boundary Method solver.

    Parameters
    ----------
    device : Device
        The quantum device object containing Hamiltonian, atomic
        structure, and attached contacts.
    k : tuple
        k-point for the calculation as (kx, ky, kz). For gamma-only
        calculations, this should be (0, 0, 0).
    quatrex_config : QuatrexConfig
        Configuration object containing calculation parameters, energy
        grid, and numerical settings.
    compute_config : ComputeConfig, optional
        Computational configuration specifying solver options and
        parallelization parameters. If None, default settings are used.

    Attributes
    ----------
    device : Device
        Reference to the device object.
    num_contacts : int
        Number of contacts attached to the device.
    k : tuple
        k-point for the calculation.
    observables : Observables
        Container for computed transport observables including
        transmission matrices, density of states, and current
        distributions.
    electron_energies : NDArray
        Full energy grid for the calculation.
    local_energies : NDArray
        Local portion of energy grid for MPI parallelization.
    hamiltonian_phase : sparse matrix
        Device Hamiltonian with k-point phase factors applied.
    overlap_phase : sparse matrix
        Device overlap matrix with k-point phase factors applied.
    system_matrix : sparse matrix
        System matrix (E*S - H - Σ) for the linear solve.

    """

    def __init__(
        self,
        device: Device,
        k: tuple,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig | None = None,
    ) -> None:
        """Initializes the QTBM solver."""

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

        # Get the electron energies.
        self.electron_energies = get_electron_energies(quatrex_config)

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

        self.solver = self._configure_solver(quatrex_config.electron.solver)
        matrix_type = preferred_matrix_type[
            quatrex_config.electron.solver.direct_solver
        ]

        # TODO: Hamiltonian should be assembled for each k-point. (This
        # can easily be vectorized)
        self.hamiltonian_phase = matrix_type(
            self.device.hamiltonian[0, 0, 0].shape, dtype=xp.complex128
        )
        for r, h_r in self.device.hamiltonian.items():
            self.hamiltonian_phase += h_r * xp.exp(
                1j * self.k[0] * r[0] + 1j * self.k[1] * r[1] + 1j * self.k[2] * r[2]
            )
        self.overlap_phase = matrix_type(
            self.device.overlap[0, 0, 0].shape, dtype=xp.complex128
        )
        for r, s_r in self.device.overlap.items():
            self.overlap_phase += s_r * xp.exp(
                1j * self.k[0] * r[0] + 1j * self.k[1] * r[1] + 1j * self.k[2] * r[2]
            )

    def _configure_solver(self, solver_config: SolverConfig) -> WFSolver:
        """Configures the wavefunction solver based on the config.

        Parameters
        ----------
        solver_config : SolverConfig
            The solver configuration containing solver type and options.

        Returns
        -------
        WFSolver
            The configured wavefunction solver instance.

        """
        if solver_config.direct_solver == "mumps":
            return MUMPS()
        if solver_config.direct_solver == "superlu":
            return SuperLU()
        if solver_config.direct_solver == "cudss":
            return cuDSS()

        raise ValueError(f"Unknown solver: {solver_config.direct_solver}")

    def compute_observables(
        self, phi: NDArray, inj_ind: list, i: int, S: list, K, T, E
    ):
        """Computes transport observables.

        Calculates transmission coefficients, density of states, and
        current distributions from the QTBM wavefunctions. This method
        processes the solution at a single energy point and updates the
        observable arrays.

        Parameters
        ----------
        phi : NDArray
            Wavefunction solution matrix. Each column represents a
            wavefunction for a specific injection mode.
        inj_ind : list
            List of arrays containing injection mode indices for each
            contact.
        i : int
            Energy index in the local energy array for storing results.
        S : list
            Self-energy matrices for each contact, used for transmission
            calculations.
        K : list or NDArray
            Bloch injection vectors for each contact.
        T : list or NDArray
            Bloch transmission matrices for each contact.
        E : float
            Energy value for the current calculation (used for error
            checking).

        """

        if phi.size == 0:
            return
        # Compute transmissions for all the possible contact couples

        cont_1 = 0
        cont_2 = 1
        for n in range(self.num_transmissions):

            # Get the all the wavefunctions injected from contact 1 and
            # extract the elements inside contact 2
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

                # For every slab, get the wavefunction injected from the
                # contact
                phi_1 = phi[self.slab_vec_x_orb[s].T, inj_ind[n]]
                phi_2 = phi[self.slab_vec_x_orb[s + 1].T, inj_ind[n]]

                # Get the transmission matrix between the slab and the
                # next one
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
        """Runs the complete QTBM transport calculation."""
        if comm.rank == 0:
            print("Entering QTBM calculation", flush=True)

        times = []
        comm.Barrier()
        OBC_batch_size = 1
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
                        np.repeat(
                            contact.orbitals_contact.squeeze(),
                            contact.orbitals_contact.shape[1],
                        )
                    )
                    ind2.append(
                        np.tile(
                            contact.orbitals_contact.squeeze(),
                            contact.orbitals_contact.shape[1],
                        )
                    )
                    sig_flat.append(sigma_obc[i, :, :].flatten())
                    # Add the injection vector in the contact elements
                    # of the rhs
                    inj_V[contact.orbitals_contact.T, inj_ind[i]] = inj[i]
                    # Add the K vector in the contact elements of the
                    # rhs
                    K_V[contact.orbitals_contact.T, inj_ind[i]] = K[i]

                # Concatenate the indices and the self-energies
                ind1 = xp.array(np.concatenate(ind1))
                ind2 = xp.array(np.concatenate(ind2))

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
                    phi = self.solver.solve(self.system_matrix, inj_V)

                t_solve = time.perf_counter() - times.pop()
                if comm.rank == 0:
                    print(f"Time for electron solver: {t_solve:.2f} s", flush=True)
                times.append(time.perf_counter())
                # Get the bare system matrix back, needed for
                # transmission calculation
                upd_0.data[:] = (
                    1e-15  # Set a small value to the self-energy matrix to avoid numerical issues
                )
                self.system_matrix.data[:] = (
                    energy * self.overlap_phase - self.hamiltonian_phase - upd_0
                ).data
                # LL = upd_0.tocsr()[rows, cols] if hasattr(LL, 'A'):
                # self.system_matrix.data[:] += LL.A.ravel() else:
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
