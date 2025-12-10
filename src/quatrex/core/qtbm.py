# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import os
import time
from dataclasses import dataclass, field

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, sparse, xp
from qttools.utils.inplace_utils import (
    add_inplace,
    add_inplace_OBC,
    sub_inplace,
    sub_inplace_OBC,
)
from qttools.utils.mpi_utils import get_local_slice
from qttools.wave_function_solver import MUMPS, SuperLU, WFSolver, cuDSS
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.constants import e, h
from quatrex.core.device import Device
from quatrex.core.energies import get_electron_energies
from quatrex.core.quatrex_config import QuatrexConfig, SolverConfig
from quatrex.core.statistics import fermi_dirac

_preferred_matrix_type = {
    "mumps": sparse.coo_matrix,
    "superlu": sparse.csc_matrix,
    "cudss": sparse.csr_matrix,
}

if xp.__name__ == "cupy":
    mempool = xp.get_default_memory_pool()
    pinned_mempool = xp.get_default_pinned_memory_pool()


def kr_mat_mul(m: NDArray, a: NDArray, vect: NDArray) -> NDArray:
    """Performs Kronecker matrix multiplication.

    Computes the product of a Kronecker product of matrices with a vector:
    (m ⊗ a) @ vect.

    Parameters
    ----------
    a : NDArray
        First matrix in the Kronecker product.
    m : NDArray
        Second matrix in the Kronecker product.
    vect : NDArray
        Vector to be multiplied.

    Returns
    -------
    result : NDArray
        Resulting vector from the multiplication.

    """
    vect_3d = vect.reshape(a.shape[0], m.shape[0], -1, order="F")

    # 2. Apply 'a' to the first dimension (axis 0)
    # tensordot(a, phi, axes=1) is like a @ phi along the first axis
    temp = np.tensordot(a, vect_3d, axes=1)

    # 3. Apply 'm' to the second dimension (axis 1 of temp)
    # We contract axis 1 of m with axis 1 of temp
    res_simple = np.tensordot(temp, m, axes=(1, 1))

    res_simple = res_simple.transpose(0, 2, 1).reshape(-1, vect.shape[1], order="F")

    return res_simple


def allocate_sys_mat(
    ham: dict, ovl: dict, boundary_SE_indexes: list[NDArray]
) -> list[sparse.csr_matrix]:
    """Allocates the system matrix with the correct sparsity pattern.

    Parameters
    ----------
    ham : dict
        Hopping hamiltonian sparse matrices.
    ovl : dict
        Hopping hamiltonian sparse matrices.
    boundary_SE_indexes : list[NDArray]
        List of destination indices for each boundary self-energy.

    Returns
    -------
    system_matrix : sparse.csr_matrix
        The allocated system matrix with the correct sparsity pattern.
    """

    ham_cpu = {}
    ovl_cpu = {}

    for key, value in ham.items():
        ham_cpu[key] = value.get() if hasattr(value, "get") else value
    for key, value in ovl.items():
        ovl_cpu[key] = value.get() if hasattr(value, "get") else value

    # System matrix size
    mat_size = ham_cpu[(0, 0, 0)].shape[0]

    # Given a row i for the system matrix, check if it is affected by each boundary self-energy
    boundary_SE_mask = []
    for indexes in boundary_SE_indexes:
        boundary_SE_mask.append(np.zeros(mat_size, dtype=np.bool))
        boundary_SE_mask[-1][:] = False
        boundary_SE_mask[-1][indexes] = True

    # List containing all System Matrix col indices for each row
    SM_indices_list = []

    # CSR System-matrix indptr array
    SM_indptr = np.zeros(mat_size + 1, dtype=np.int64)

    for i_row in range(mat_size):

        # Compute the union of all column indices affecting this row
        row_union = np.array([], dtype=np.int64)

        # Add Hamiltonian contributions
        for r, h_r in ham_cpu.items():
            row_start = h_r.indptr[i_row]
            row_end = h_r.indptr[i_row + 1]
            row_union = np.union1d(row_union, h_r.indices[row_start:row_end])

        # Add Overlap contributions
        for r, s_r in ovl_cpu.items():
            row_start = s_r.indptr[i_row]
            row_end = s_r.indptr[i_row + 1]
            row_union = np.union1d(row_union, s_r.indices[row_start:row_end])

        # Add Boundary self-energy contributions
        for i, s_ind in enumerate(boundary_SE_mask):
            if s_ind[i_row]:
                row_union = np.union1d(row_union, boundary_SE_indexes[i])

        # Store the column indices for this row
        SM_indices_list.append(row_union)
        SM_indptr[i_row + 1] = SM_indptr[i_row] + len(row_union)

    total_SM_nnz = SM_indptr[-1]

    # Allocate data and indices arrays
    SM_data = np.zeros(total_SM_nnz, dtype=xp.complex128)
    SM_indices = np.concatenate(SM_indices_list)

    SM_data_gpu = xp.asarray(SM_data)
    SM_indices_gpu = xp.asarray(SM_indices)
    SM_indptr_gpu = xp.asarray(SM_indptr)

    system_matrix = sparse.csr_matrix(
        (SM_data_gpu, SM_indices_gpu, SM_indptr_gpu),
        shape=(mat_size, mat_size),
        dtype=xp.complex128,
    )

    # Check if SM has canonical format
    if not system_matrix.has_canonical_format:
        raise ValueError("System matrix is not in canonical format after allocation.")

    return system_matrix


def compute_update_indeces_sparse(
    M: sparse.csr_matrix, U: sparse.csr_matrix, destination_indexes: NDArray = None
) -> NDArray:
    """Computes the indices for updating the system matrix.

    Parameters
    ----------
    M : sparse.csr_matrix
        The original system matrix.
    U : sparse.csr_matrix
        The update matrix to be applied.
    destination_indexes : NDArray
        The indices in the system matrix where the update should be applied.

    Returns
    -------
    target_indices : NDArray
        The indices in the flattened system matrix corresponding to the
        update positions.

    """

    # Get the CPU versions of M and U
    M = M.get() if hasattr(M, "get") else M
    U = U.get() if hasattr(U, "get") else U

    # Default destination indexes to identity mapping
    if destination_indexes is None:
        destination_indexes = np.arange(M.shape[0], dtype=xp.int64)

    if np.unique(destination_indexes).size != destination_indexes.size:
        raise ValueError(
            "The destination indexes have duplicate entries, cannot compute update indices."
        )

    update_indices = np.zeros_like(U.data, dtype=xp.int64)

    # Iterate over rows of U
    for U_row in range(U.shape[0]):

        # Get the column indices for the current row of U
        row_start = U.indptr[U_row]
        row_end = U.indptr[U_row + 1]
        U_cols = U.indices[row_start:row_end]

        # Get the corresponding row in M
        M_row = destination_indexes[U_row]

        # Get the column indices for the current row of M
        M_row_start = M.indptr[M_row]
        M_row_end = M.indptr[M_row + 1]
        M_cols = M.indices[M_row_start:M_row_end]

        # Check for duplicate column indices in the system matrix row
        if np.unique(M_cols).size != M_cols.size:
            raise ValueError(
                "The system matrix has duplicate column indices in a row, cannot compute update indices."
            )

        # Map U column indices to destination indexes in M
        U_cols_dest = destination_indexes[U_cols]

        # Map U column indices to M column indices
        M_ind_map = np.searchsorted(M_cols, U_cols_dest)
        if (M_cols[M_ind_map] != U_cols_dest).any():
            raise ValueError(
                "Some destination indexes do not exist in the system matrix row, cannot compute update indices."
            )
        if np.unique(M_ind_map).size != U_cols_dest.size:
            raise ValueError(
                "Some destination indexes do not exist in the system matrix row, cannot compute update indices."
            )

        update_indices[row_start:row_end] = M_ind_map + M_row_start

    return xp.array(update_indices)


def compute_update_indeces_dense(
    M: sparse.csr_matrix, destination_indexes: NDArray = None
) -> NDArray:
    """Computes the indices for updating the system matrix.

    Parameters
    ----------
    M : sparse.csr_matrix
        The original system matrix.
    U : NDArray
        The update matrix to be applied.
    destination_indexes : NDArray
        The indices in the system matrix where the update should be applied.

    Returns
    -------
    target_indices : NDArray
        The indices in the flattened system matrix corresponding to the
        update positions.

    """

    # Get the CPU version of M
    M = M.get() if hasattr(M, "get") else M

    # Default destination indexes to identity mapping
    if destination_indexes is None:
        destination_indexes = np.arange(M.shape[0], dtype=xp.int64)

    if np.unique(destination_indexes).size != destination_indexes.size:
        raise ValueError(
            "The destination indexes have duplicate entries, cannot compute update indices."
        )

    U_size = destination_indexes.shape[0]

    update_indices = np.zeros((U_size**2,), dtype=xp.int64)

    for U_row in range(U_size):

        # Get the corresponding row in M
        M_row = destination_indexes[U_row]
        M_row_start = M.indptr[M_row]
        M_row_end = M.indptr[M_row + 1]
        M_cols = M.indices[M_row_start:M_row_end]

        if np.unique(M_cols).size != M_cols.size:
            raise ValueError(
                "The system matrix has duplicate column indices in a row, cannot compute update indices."
            )

        M_ind_map = np.searchsorted(M_cols, destination_indexes)
        if np.unique(M_ind_map).size != destination_indexes.size:
            raise ValueError(
                "Some destination indexes do not exist in the system matrix row, cannot compute update indices."
            )
        if (M_cols[M_ind_map] != destination_indexes).any():
            raise ValueError(
                "Some destination indexes do not exist in the system matrix row, cannot compute update indices."
            )

        update_indices[U_row * U_size : (U_row + 1) * U_size] = M_ind_map + M_row_start

    return xp.array(update_indices)


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

    hole_charge_orb: NDArray = None
    hole_charge_at: NDArray = None


class QTBM:
    """Quantum Transmitting Boundary Method solver.

    Parameters
    ----------
    device : Device
        The quantum device object containing Hamiltonian, atomic
        structure, and attached contacts.
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
    neutrality_level : float
        Charge neutrality level for the device.
    """

    def __init__(
        self,
        device: Device,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig | None = None,
    ) -> None:
        """Initializes the QTBM solver."""

        self.device = device
        self.num_orbitals = device.hamiltonian[0, 0, 0].shape[0]
        self.num_contacts = len(device.contacts)

        self.quatrex_config = quatrex_config
        if compute_config is None:
            compute_config = ComputeConfig()

        self.compute_config = compute_config

        kpoint_grid = quatrex_config.device.kpoint_grid
        if self.device.gamma_only and kpoint_grid != (1, 1, 1):
            raise ValueError(
                "The device only has a Gamma point Hamiltonian, "
                "but more than one k-point is configured."
            )

        # Generate the Monkhorst-Pack k-point grid.
        self.kpoints = monkhorst_pack(kpoint_grid)
        # Shift the k-points.
        self.kpoints += np.array(quatrex_config.device.kpoint_shift)
        self.num_kpoints = self.kpoints.shape[0]

        self.max_batch_size = self.quatrex_config.qtbm.max_batch_size

        self.observables = Observables()

        self.flatband = quatrex_config.electron.flatband
        self.eta_obc = quatrex_config.electron.eta_obc
        self.block_sections = quatrex_config.electron.obc.block_sections

        # Get the electron energies.
        self.electron_energies = get_electron_energies(quatrex_config)

        # Get the local slice of the electron energies
        self.local_energies = get_local_slice(self.electron_energies)

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
        self.matrix_type = _preferred_matrix_type[
            quatrex_config.electron.solver.direct_solver
        ]

        self.observables.electron_current["contact_current"] = xp.zeros(
            self.num_transmissions
        )

        self.observables.electron_charge_orb = xp.zeros(
            (self.num_orbitals,), dtype=xp.float64
        )

        self.observables.hole_charge_orb = xp.zeros(
            (self.num_orbitals,), dtype=xp.float64
        )

        if (
            self.quatrex_config.electron.conduction_band_edge is None
            or self.quatrex_config.electron.valence_band_edge is None
        ):
            if comm.rank == 0:
                print(
                    "WARNING: No band edges provided, only electron charge will be computed."
                )
            self.neutrality_level = -np.inf
        else:
            self.neutrality_level = 0.5 * (
                self.quatrex_config.electron.conduction_band_edge
                + self.quatrex_config.electron.valence_band_edge
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
        overlap,
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

                S_P = xp.zeros_like(phi_n)
                index1 = (
                    -xp.arange(self.device.contacts[cont_2].n_rep_1)[:, None]
                    + xp.arange(self.device.contacts[cont_2].n_rep_1)[None, :]
                )
                index2 = (
                    -xp.arange(self.device.contacts[cont_2].n_rep_2)[:, None]
                    + xp.arange(self.device.contacts[cont_2].n_rep_2)[None, :]
                )

                index1 = xp.kron(
                    index1,
                    xp.ones(
                        (
                            self.device.contacts[cont_2].n_rep_2,
                            self.device.contacts[cont_2].n_rep_2,
                        )
                    ),
                )
                index2 = xp.tile(
                    index2,
                    (
                        self.device.contacts[cont_2].n_rep_1,
                        self.device.contacts[cont_2].n_rep_1,
                    ),
                )

                for key, value in S[cont_2].items():
                    S_P += kr_mat_mul(
                        xp.exp(-1j * key[0] * index1 - 1j * key[1] * index2),
                        value[i_batch, :, :],
                        phi_n,
                    )

                self.observables.electron_transmission_contacts[K_ind, n_t, i_en] = (
                    xp.trace(-2 * xp.imag(phi_n.T.conj() @ S_P))
                )

        phi_ortho = xp.zeros_like(phi)
        for k in overlap.keys():
            phi_ortho += overlap[k] @ phi

        for n, contact in enumerate(self.device.contacts):

            phi_cont = K[contact.orbitals_contact.squeeze(), :]
            index1 = (
                -xp.arange(contact.n_rep_1)[:, None]
                + xp.arange(contact.n_rep_1)[None, :]
            )
            index2 = (
                -xp.arange(contact.n_rep_2)[:, None]
                + xp.arange(contact.n_rep_2)[None, :]
            )

            index1 = xp.kron(index1, xp.ones((contact.n_rep_2, contact.n_rep_2)))
            index2 = xp.tile(index2, (contact.n_rep_1, contact.n_rep_1))

            for key, value in T[n].items():
                phi_cont += kr_mat_mul(
                    xp.exp(-1j * key[0] * index1 - 1j * key[1] * index2),
                    value[i_batch, :, :],
                    phi[contact.orbitals_contact.squeeze(), :],
                )

            # Add the spill over from the overlap
            for o_r in overlap.values():
                phi_ortho[contact.orbitals_contact.squeeze(), :] += (
                    contact.get_10(o_r) @ phi_cont
                )
            # CHECK SPILL OVER ERROR (DEBUG)
            error = xp.linalg.norm(
                contact.get_10(system_matrix) @ phi_cont
                + system_matrix[contact.orbitals_contact.squeeze(), :] @ phi
            )
            if comm.rank == 0:
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

    def _write_outputs(self):
        if comm.rank == 0:

            output_dir = self.quatrex_config.output_dir
            if not os.path.exists(self.quatrex_config.output_dir):
                os.mkdir(self.quatrex_config.output_dir)

            for n in range(self.num_transmissions):
                np.save(
                    f"{output_dir}/transmission_{self.observables.electron_transmission_contacts_labels[n]}.npy",
                    self.observables.electron_transmission_contacts[:, n, :],
                )

                np.save(
                    f"{output_dir}/current_{self.observables.electron_transmission_contacts_labels[n]}.npy",
                    self.observables.electron_current["contact_current"][n],
                )

            for n in range(self.num_contacts):
                np.save(
                    f"{output_dir}/band_{self.device.contacts[n].name[0]}.npy",
                    self.device.contacts[n].band_structure,
                )
                np.save(
                    f"{output_dir}/dos_{self.device.contacts[n].name[0]}.npy",
                    self.observables.electron_dos_orb[:, n, :, :],
                )

            np.save(
                f"{output_dir}/el_charge_orb.npy",
                self.observables.electron_charge_orb,
            )

            np.save(
                f"{output_dir}/el_charge_at.npy",
                self.observables.electron_charge_at,
            )

            np.save(f"{output_dir}/orb.npy", self.device.orbital_offsets)

            np.save(
                f"{output_dir}/ho_charge_orb.npy",
                self.observables.hole_charge_orb,
            )

            np.save(
                f"{output_dir}/ho_charge_at.npy",
                self.observables.hole_charge_at,
            )

    def run(self) -> None:
        """Runs the complete QTBM transport calculation."""
        if comm.rank == 0:
            print("Entering QTBM calculation", flush=True)

        times = []
        comm.Barrier()

        cont_ind_list = []
        for contact in self.device.contacts:
            cont_ind_list.append(contact.orbitals_contact.squeeze())

        # Allocate indices to update the system matrix in-place
        system_matrix = allocate_sys_mat(
            self.device.hamiltonian, self.device.overlap, cont_ind_list
        )  # Initialize the system matrix

        ham_update_ind = []
        for r, h_r in self.device.hamiltonian.items():
            ham_update_ind.append(compute_update_indeces_sparse(system_matrix, h_r))
        overlap_update_ind = []
        for r, s_r in self.device.overlap.items():
            overlap_update_ind.append(compute_update_indeces_sparse(system_matrix, s_r))
        sigma_SM_indexes = []
        for contact in self.device.contacts:
            sigma_SM_indexes.append(
                compute_update_indeces_dense(
                    system_matrix, contact.orbitals_contact.squeeze()
                )
            )

        for k_ind in range(self.num_kpoints):

            if comm.rank == 0:
                print(f"Processing k-point {k_ind+1} of {self.num_kpoints}", flush=True)
            k = self.kpoints[k_ind, :]

            times.append(time.perf_counter())

            # Apply the k-point phase factors to the Hamiltonian and Overlap
            for r, h_r in self.device.hamiltonian.items():
                if r == (0, 0, 0):
                    continue
                h_r.data *= xp.exp(
                    1j * 2 * np.pi * (k[0] * r[0] + k[1] * r[1] + k[2] * r[2])
                )

            for r, s_r in self.device.overlap.items():
                if r == (0, 0, 0):
                    continue
                s_r.data *= xp.exp(
                    1j * 2 * np.pi * (k[0] * r[0] + k[1] * r[1] + k[2] * r[2])
                )

            times.append(time.perf_counter())

            for batch_start in range(0, len(self.local_energies), self.max_batch_size):

                energy_batch = self.local_energies[
                    batch_start : batch_start + self.max_batch_size
                ]

                if comm.rank == 0:
                    print(
                        f"Processing energies {batch_start} to {batch_start + len(energy_batch) - 1}",
                        flush=True,
                    )

                # append for iteration time
                times.append(time.perf_counter())

                times.append(time.perf_counter())

                sigma_obcs_K = []
                injs = []
                inj_inds = []
                Ks = []
                Ts = []
                Ts_K = []
                # Compute the boundary self-energy and the injection vector.
                ind_0 = np.zeros(len(energy_batch), dtype=np.int32)
                for contact in self.device.contacts:
                    times.append(time.perf_counter())

                    inj, num_inj, K, sigma_obc_K, T_K = contact.compute_boundary(
                        k * 2 * np.pi, energy_batch
                    )
                    injs.append(inj)
                    Ks.append(K)

                    Ts_K.append(T_K)
                    sigma_obcs_K.append(sigma_obc_K)

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

                    for r in self.device.overlap.keys():
                        self.device.overlap[r].data *= energy

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

                    # Iterate over contacts
                    for contact, inj, inj_ind, K in zip(
                        self.device.contacts, injs, inj_inds, Ks
                    ):
                        # Add the injection vector in the contact elements
                        # of the rhs
                        inj_V[contact.orbitals_contact.T, inj_ind[i]] = inj[i]
                        # Add the K vector in the contact elements of the
                        # rhs
                        K_V[contact.orbitals_contact.T, inj_ind[i]] = K[i]

                    system_matrix.data[:] = 0

                    # Add the Hamiltonian and overlap contributions
                    for r_idx, (r_key, h_r) in enumerate(
                        self.device.hamiltonian.items()
                    ):

                        sub_inplace(
                            system_matrix.data,
                            h_r.data,
                            ham_update_ind[r_idx],
                        )

                    for r_idx, (r_key, s_r) in enumerate(self.device.overlap.items()):

                        add_inplace(
                            system_matrix.data,
                            s_r.data,
                            overlap_update_ind[r_idx],
                        )

                    # Add the boundary self-energy contributions
                    for c, s_K in enumerate(sigma_obcs_K):

                        for key, value in s_K.items():

                            sub_inplace_OBC(
                                system_matrix.data,
                                value[i, :, :],
                                sigma_SM_indexes[c],
                                key[0],
                                key[1],
                                self.device.contacts[c].n_rep_1,
                                self.device.contacts[c].n_rep_2,
                            )

                    t_solve = time.perf_counter() - times.pop()
                    if comm.rank == 0:
                        print(
                            f"Time to set up system of eq.: {t_solve:.2f} s", flush=True
                        )

                    times.append(time.perf_counter())

                    # Solve for the wavefunction
                    if inj_V.size != 0:
                        phi = self.solver.solve(system_matrix, inj_V)
                    # phi = xp.zeros((self.num_orbitals, inj_V.shape[1]), dtype=xp.complex128)

                    t_solve = time.perf_counter() - times.pop()
                    if comm.rank == 0:
                        print(f"Time for electron solver: {t_solve:.2f} s", flush=True)
                    times.append(time.perf_counter())

                    # Get the bare system matrix back, needed for
                    # transmission calculation

                    # Subtract the open boundary conditions
                    for c, s_K in enumerate(sigma_obcs_K):

                        for key, value in s_K.items():

                            add_inplace_OBC(
                                system_matrix.data,
                                value[i, :, :],
                                sigma_SM_indexes[c],
                                key[0],
                                key[1],
                                self.device.contacts[c].n_rep_1,
                                self.device.contacts[c].n_rep_2,
                            )

                    for r in self.device.overlap.keys():
                        self.device.overlap[r].data *= 1 / energy

                    if inj_V.size != 0:
                        self.compute_observables(
                            phi,
                            inj_inds,
                            i,
                            batch_start + i,
                            sigma_obcs_K,
                            K_V,
                            Ts_K,
                            system_matrix,
                            self.device.overlap,
                            k_ind,
                        )

                    t_iteration = time.perf_counter() - times.pop()
                    if comm.rank == 0:
                        print(
                            f"Time for computing observables: {t_iteration:.2f} s",
                            flush=True,
                        )

                del injs
                del Ks
                del Ts

                del inj
                del K

                t_iteration = time.perf_counter() - times.pop()
                if comm.rank == 0:
                    print(f"Time for iteration: {t_iteration:.2f} s", flush=True)

            for r, h_r in self.device.hamiltonian.items():
                if r == (0, 0, 0):
                    continue
                h_r.data /= xp.exp(
                    1j * 2 * np.pi * (k[0] * r[0] + k[1] * r[1] + k[2] * r[2])
                )

            for r, s_r in self.device.overlap.items():
                if r == (0, 0, 0):
                    continue
                s_r.data /= xp.exp(
                    1j * 2 * np.pi * (k[0] * r[0] + k[1] * r[1] + k[2] * r[2])
                )

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

            self.observables.electron_current["contact_current"][n_t] = -(
                xp.sum(
                    xp.trapz(
                        Fermi_factor
                        * self.observables.electron_transmission_contacts[:, n_t, :],
                        self.electron_energies,
                        axis=1,
                    )
                )
                / self.num_kpoints
                * (2 * e / h)
            )

        # Compute the orbital electron charge density per orbital
        for n in range(self.num_contacts):
            Fermi_factor = fermi_dirac(
                self.electron_energies - self.device.contacts[n].fermi_level,
                self.quatrex_config.electron.temperature,
            )
            Fermi_factor[self.electron_energies < self.neutrality_level] = 0.0
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

        # Compute the orbital hole charge density per orbital
        for n in range(self.num_contacts):
            Fermi_factor = fermi_dirac(
                self.electron_energies - self.device.contacts[n].fermi_level,
                self.quatrex_config.electron.temperature,
            )
            Fermi_factor[self.electron_energies > self.neutrality_level] = 1
            self.observables.hole_charge_orb += (
                2
                * xp.sum(
                    xp.trapz(
                        self.observables.electron_dos_orb[:, n, :, :]
                        * (1 - Fermi_factor),
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

        self.observables.hole_charge_at = xp.zeros(
            self.device.orbital_offsets.shape[0] - 1
        )
        self.observables.hole_charge_at = xp.add.reduceat(
            self.observables.hole_charge_orb, self.device.orbital_offsets[:-1]
        )

        self._write_outputs()
