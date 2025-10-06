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
from quatrex.core.statistics import fermi_dirac

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
    electron_transmission_indices = []

    electron_dos_orb: NDArray = None

    electron_charge_orb: NDArray = None
    electron_charge_at: NDArray = None

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
    k_grid : tuple
        k-point for the calculation.
    observables : Observables
        Container for computed transport observables including
        transmission matrices, density of states, and current
        distributions.
    electron_energies : NDArray
        Full energy grid for the calculation.
    local_energies : NDArray
        Local portion of energy grid for MPI parallelization.
    """

    def __init__(
        self,
        device: Device,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig | None = None,
        k_grid: tuple = (1, 1, 1),
        k_shift: tuple = (0, 0, 0),
    ) -> None:
        """Initializes the QTBM solver."""

        self.device = device
        self.num_orbitals = device.hamiltonian[0, 0, 0].shape[0]
        self.num_contacts = len(device.contacts)
        self.quatrex_config = quatrex_config

        if self.device.gamma_only and k_grid != (1, 1, 1):
            raise ValueError(
                "The device only has a Gamma point Hamiltonian, "
                "but the wavevector is not (0,0,0)."
            )

        self.k_vector = monkhorst_pack(k_grid)  # The wavevector for the QTBM
        self.k_vector += k_shift
        self.num_kpoints = self.k_vector.shape[0]

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
        self.num_transmissions = int((self.num_contacts**2 - self.num_contacts))

        for cont_1 in range(self.num_contacts):
            for cont_2 in range(self.num_contacts):
                if cont_2 != cont_1:
                    self.observables.electron_transmission_contacts_labels.append(
                        f"{self.device.contacts[cont_1].name[0]}->{self.device.contacts[cont_2].name[0]}"
                    )
                    self.observables.electron_transmission_indices.append(
                        (cont_1, cont_2)
                    )

        # Initialize the observables
        self.observables.electron_transmission_contacts = xp.zeros(
            (self.num_kpoints, self.num_transmissions, self.local_energies.shape[0]),
            dtype=xp.float64,
        )

        self.observables.electron_dos_orb = xp.zeros(
            (
                self.num_kpoints,
                self.num_contacts,
                self.num_orbitals,
                self.local_energies.shape[0],
            ),
            dtype=xp.float64,
        )

        self.solver = self._configure_solver(quatrex_config.electron.solver)
        self.matrix_type = preferred_matrix_type[
            quatrex_config.electron.solver.direct_solver
        ]

        self.observables.electron_current["contact_current"] = xp.zeros(
            self.num_transmissions
        )

        self.observables.electron_charge_orb = xp.zeros(
            (self.num_orbitals,), dtype=xp.float64
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
        self,
        phi: NDArray,
        inj_ind: list,
        i_batch: int,
        i_en: int,
        S: list,
        K,
        T,
        system_matrix,
        overlap_phase,
        K_ind,
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

        for n_t in range(self.num_transmissions):
            # Get the all the wavefunctions injected from contact 1 and
            # extract the elements inside contact 2
            cont_1, cont_2 = self.observables.electron_transmission_indices[n_t]
            phi_n = phi[
                self.device.contacts[cont_2].orbitals_contact.T,
                inj_ind[cont_1][i_batch],
            ]
            # Compute the transmission
            if phi_n.size != 0:
                self.observables.electron_transmission_contacts[K_ind, n_t, i_en] = (
                    xp.trace(
                        xp.real(
                            1j
                            * phi_n.T.conj()
                            @ (
                                S[cont_2][i_batch, :, :]
                                - S[cont_2][i_batch, :, :].T.conj()
                            )
                            @ phi_n
                        )
                    )
                )
                if comm.rank == 0:
                    print(
                        f"Transmission {self.observables.electron_transmission_contacts_labels[n_t]}: {self.observables.electron_transmission_contacts[K_ind, n_t, i_en]}",
                        flush=True,
                    )

        phi_ortho = overlap_phase @ phi  # "Orthogonalize" the wavefunction
        for n, contact in enumerate(self.device.contacts):

            phi_cont = (
                K[contact.orbitals_contact.squeeze(), :]
                + T[n][i_batch] @ phi[contact.orbitals_contact.squeeze(), :]
            )
            # TODO Add the spill over contribution
            phi_ortho[contact.orbitals_contact.squeeze(), :] += (
                contact.get_10(overlap_phase) @ phi_cont
            )
            # CHECK SPILL OVER ERROR (DEBUG)
            error = xp.linalg.norm(
                contact.get_10(system_matrix) @ phi_cont
                + system_matrix[contact.orbitals_contact.squeeze(), :] @ phi
            )
            print(f"    Spill over error for contact {contact.name[0]}: {error}")

        # Compute the DOS for every injected wavefunction
        for n in range(self.num_contacts):
            phi_D = phi[
                :, inj_ind[n][i_batch]
            ].squeeze()  # Get the wavefunction in the slab
            phi_D_ortho = phi_ortho[
                :, inj_ind[n][i_batch]
            ].squeeze()  # Get the "orthogonalized" wavefunction in the slab

            if phi_D.size != 0:

                # Ensure phi_D is always 2D
                if phi_D.ndim == 1:
                    phi_D = phi_D[:, xp.newaxis]
                    phi_D_ortho = phi_D_ortho[:, xp.newaxis]

                self.observables.electron_dos_orb[K_ind, n, :, i_en] = xp.real(
                    xp.sum(xp.multiply(phi_D.conj(), phi_D_ortho), axis=1) / (2 * xp.pi)
                )  # Compute the DOS

    def run(self) -> None:
        """Runs the complete QTBM transport calculation."""
        if comm.rank == 0:
            print("Entering QTBM calculation", flush=True)

        times = []
        comm.Barrier()
        system_matrix = None  # Initialize the system matrix

        print(self.k_vector)

        for k_ind in range(self.num_kpoints):

            if comm.rank == 0:
                print(f"Processing k-point {k_ind+1} of {self.num_kpoints}", flush=True)
            k = self.k_vector[k_ind, :]
            print(k)

            times.append(time.perf_counter())

            # Precompute the Hamiltonian and overlap with k-point
            # phase factors applied.
            hamiltonian_phase = self.matrix_type(
                self.device.hamiltonian[0, 0, 0].shape, dtype=xp.complex128
            )
            for r, h_r in self.device.hamiltonian.items():
                hamiltonian_phase += h_r * xp.exp(
                    1j * 2 * np.pi * (k[0] * r[0] + k[1] * r[1] + k[2] * r[2])
                )

            overlap_phase = self.matrix_type(
                self.device.overlap[0, 0, 0].shape, dtype=xp.complex128
            )
            for r, s_r in self.device.overlap.items():
                overlap_phase += s_r * xp.exp(
                    1j * 2 * np.pi * (k[0] * r[0] + k[1] * r[1] + k[2] * r[2])
                )

            times.append(time.perf_counter())

            for batch_start in range(
                0, len(self.local_energies), self.quatrex_config.electron.obc_batch_size
            ):

                energy_batch = self.local_energies[
                    batch_start : batch_start
                    + self.quatrex_config.electron.obc_batch_size
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
                        k * 2 * np.pi, energy_batch
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
                        (sig_flat, (ind1, ind2)), shape=hamiltonian_phase.shape
                    ).tocsr()
                    upd_0.eliminate_zeros()  # Remove zeros from the self-energy matrix

                    if i == 0 and batch_start == 0 and k_ind == 0:
                        system_matrix = (
                            (energy) * overlap_phase - hamiltonian_phase - upd_0
                        )
                    else:
                        system_matrix.data[:] = (
                            (energy) * overlap_phase - hamiltonian_phase - upd_0
                        ).data

                    t_solve = time.perf_counter() - times.pop()
                    if comm.rank == 0:
                        print(
                            f"Time to set up system of eq.: {t_solve:.2f} s", flush=True
                        )

                    times.append(time.perf_counter())

                    # Solve for the wavefunction
                    if inj_V.size != 0:
                        phi = self.solver.solve(system_matrix, inj_V)

                    t_solve = time.perf_counter() - times.pop()
                    if comm.rank == 0:
                        print(f"Time for electron solver: {t_solve:.2f} s", flush=True)
                    times.append(time.perf_counter())
                    # Get the bare system matrix back, needed for
                    # transmission calculation
                    upd_0.data[:] = (
                        1e-15  # Set a small value to the self-energy matrix to avoid numerical issues
                    )
                    system_matrix.data[:] = (
                        (energy) * overlap_phase - hamiltonian_phase - upd_0
                    ).data
                    # LL = upd_0.tocsr()[rows, cols] if hasattr(LL, 'A'):
                    # self.system_matrix.data[:] += LL.A.ravel() else:
                    #    self.system_matrix.data[:] += LL.ravel()

                    if inj_V.size != 0:
                        self.compute_observables(
                            phi,
                            inj_inds,
                            i,
                            batch_start + i,
                            sigma_obcs,
                            K_V,
                            Ts,
                            system_matrix,
                            overlap_phase,
                            k_ind,
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
        self.observables.electron_transmission_contacts = xp.concatenate(
            comm.allgather(self.observables.electron_transmission_contacts), axis=-1
        )
        self.observables.electron_dos_orb = xp.concatenate(
            comm.allgather(self.observables.electron_dos_orb), axis=-1
        )

        if comm.rank == 0:
            print(self.observables.electron_transmission_contacts.shape)
            print(self.observables.electron_dos_orb.shape)

        # Compute the current from all the k dependent transmissions
        for n_t in range(self.num_transmissions):
            cont_1, cont_2 = self.observables.electron_transmission_indices[n_t]
            Fermi_factor = fermi_dirac(
                self.electron_energies - self.device.contacts[cont_1].fermi_level,
                self.quatrex_config.electron.temperature,
            ) - fermi_dirac(
                self.electron_energies - self.device.contacts[cont_2].fermi_level,
                self.quatrex_config.electron.temperature,
            )

            self.observables.electron_current["contact_current"][n_t] = (
                xp.sum(
                    xp.trapz(
                        Fermi_factor
                        * self.observables.electron_transmission_contacts[:, n_t, :],
                        self.electron_energies,
                        axis=1,
                    )
                )
                / self.num_kpoints
                * 7.7480917310e-5
            )

        # Compute the orbital charge density per orbital
        for n in range(self.num_contacts):
            Fermi_factor = fermi_dirac(
                self.electron_energies - self.device.contacts[n].fermi_level,
                self.quatrex_config.electron.temperature,
            )
            self.observables.electron_charge_orb += (
                2
                * xp.sum(
                    xp.trapz(
                        self.observables.electron_dos_orb[:, n, :, :] * Fermi_factor,
                        self.electron_energies,
                        axis=2,
                    ),
                    axis=0,
                )
                / self.num_kpoints
            )

        self.observables.electron_charge_at = xp.zeros(
            self.device.orbital_offsets.shape[0] - 1
        )
        self.observables.electron_charge_at = xp.add.reduceat(
            self.observables.electron_charge_orb, self.device.orbital_offsets[:-1]
        )
