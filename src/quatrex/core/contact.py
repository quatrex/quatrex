# Copyright (c) 2025-2026 ETH Zurich and the authors of the quatrex package.

import itertools

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, obc, sparse, xp
from qttools.nevp import NEVP, Beyn, Full
from quatrex.core.compute_config import NEVPConfig
from quatrex.core.kpoints import monkhorst_pack
from quatrex.core.quatrex_config import OBCConfig


class Contact:
    """Class representing a contact for QTBM calculations.

    Parameters
    ----------
    device : Device
        The device object to which this contact is attached. Contains
        the Hamiltonian, overlap matrices, and atomic structure
        information.
    name : str
        A unique identifier for this contact.
    origin : NDArray
        The origin coordinates of the contact cell in device
        coordinates.
    lattice_vectors : NDArray
        The lattice vectors defining the unit cell of the contact.
    direction : str
        The transport direction of the contact, specified as 'a', 'b',
        or 'c' corresponding to the lattice axes.
    fermi_level : float
        The Fermi level of the contact in eV.

    Attributes
    ----------
    name : str
        The contact identifier.
    device : Device
        Reference to the parent device.
    lattice_vectors : NDArray
        Contact unit cell lattice vectors.
    origin : NDArray
        Contact origin coordinates.
    direction : int
        Transport direction index (0, 1, or 2).
    transverse_axis : list[int]
        Indices of the two transverse directions.
    obc : obc.Spectral
        Configured open boundary condition solver.
    UC_hamiltonian : dict
        Unit cell Hamiltonian matrices indexed by (i, j, k) tuples.
    UC_overlap : dict
        Unit cell overlap matrices indexed by (i, j, k) tuples.
    orbitals_per_repetition : list[NDArray]
        List of orbital indices for each contact cell repetition.
    transverse_repetition_grid: NDArray
        Number of periodic repetitions in the two transverse directions.
    transverse_rep : int
        Number of repetitions needed in transport direction for
        convergence.
    band_structure: NDArray
        Cached band structure data for the contact.

    """

    def __init__(
        self,
        device,
        name: str,
        origin: NDArray,
        lattice_vectors: NDArray,
        direction: str,
        fermi_level: float,
    ):
        """Initializes the contact object."""

        if len(origin) != 3:
            raise ValueError("Origin must be a 3D coordinate.")
        if lattice_vectors.shape != (3, 3):
            raise ValueError("Vectors must be a 3x3 array.")
        if direction not in ["a", "b", "c"]:
            raise ValueError("Direction must be one of 'a', 'b', or 'c'.")

        self.name = name
        self.device = device

        self.fermi_level = fermi_level

        self.lattice_vectors = lattice_vectors
        self.origin = origin

        self.direction = "abc".index(direction)
        self.transverse_axis = [0, 1, 2]
        self.transverse_axis.remove(self.direction)

        self.UC_hamiltonian = {}
        self.UC_overlap = {}

        self.S10_contact = {}
        self.H10_contact = {}

        # Get the atoms inside the origin cell (defined by the user)
        self.origin_atom_indices = self._get_atom_indices_in_cell(0, 0, 0)
        self.origin_orbital_indices = self._atom_to_orbital_indices(
            self.origin_atom_indices
        )

        self.origin_number_of_orbitals = len(self.origin_orbital_indices)

        if self.origin_number_of_orbitals == 0:
            raise ValueError(
                f"Error in contact {self.name}: No atoms found inside the origin cell."
            )

        if comm.rank == 0:
            print(f"Contact {self.name}:", flush=True)
            print(
                f"    Number of atoms inside the origin cell: {self.origin_number_of_orbitals}",
                flush=True,
            )

        # Check how many periodic repetitions are in the transverse
        # directions
        self._init_periodic_transverse_repetitions()
        if comm.rank == 0:
            print(
                f"    Number of periodic repetitions in the transverse directions: {self.transverse_repetition_grid[0]} x {self.transverse_repetition_grid[1]}",
                flush=True,
            )

        # TODO Check if the contact transverse UC vectors are in the
        # same direction as the device vectors

        number_of_transport_cells = self._init_orbitals()
        self.number_of_transport_cells = number_of_transport_cells

        # self._init_hamiltonian_overlap_matrices()

        if comm.rank == 0:
            print(
                f"    Number of repetitions in transport direction: {number_of_transport_cells-1}",
                flush=True,
            )

        # Orbitals for contact (where to apply the OBC)
        # Sorted first in transport direction, then in transverse directions
        self.orbitals_contact = []
        for i in range(self.transverse_repetition_grid[0]):
            for j in range(self.transverse_repetition_grid[1]):
                # The last contact cell is not included
                self.orbitals_contact.extend(self.orbitals_per_repetition[i][j][:-1])

        self.orbitals_contact = np.concatenate(self.orbitals_contact)[None, :]

        # When getting the 10 matrix (for spill over), it is more efficient to have it sorted first in transverse, then in transport
        # The orbital list is then different. We keep it separated over slice over transport direction.
        self.orbitals_get_10 = []
        for i in range(0, number_of_transport_cells):
            orbitals_slice_x = []
            for j in range(self.transverse_repetition_grid[0]):
                for k in range(self.transverse_repetition_grid[1]):
                    orbitals_slice_x.append(self.orbitals_per_repetition[j][k][i])
            self.orbitals_get_10.append(np.concatenate(orbitals_slice_x))

        # We then need to sort the 10 matrix to have the same ordering as the contact OBCs
        self.sort_orbitals_get_10 = []
        for i in range(
            self.transverse_repetition_grid[0] * self.transverse_repetition_grid[1]
        ):
            for k in range(0, number_of_transport_cells - 1):
                self.sort_orbitals_get_10.append(
                    np.arange(self.origin_number_of_orbitals)
                    + i * self.origin_number_of_orbitals
                    + k
                    * self.origin_number_of_orbitals
                    * self.transverse_repetition_grid[0]
                    * self.transverse_repetition_grid[1]
                )
        self.sort_orbitals_get_10 = np.concatenate(
            self.sort_orbitals_get_10, dtype=int
        )[None, :]

        self.transverse_rep = number_of_transport_cells - 1

        self.obc_solver = self._configure_obc(
            device.quatrex_config.electron.obc, device.compute_config.nevp
        )

        self._compute_band_structure(device)

    def _get_atom_indices_in_cell(self, nx: int, ny: int, nz: int) -> NDArray:
        """Gets the indices of atoms inside a specific periodic repetition.

        This method finds all device atoms that fall within the
        specified periodic repetition of the contact unit cell.

        Parameters
        ----------
        nx : int
            The x-coordinate of the periodic repetition.
        ny : int
            The y-coordinate of the periodic repetition.
        nz : int
            The z-coordinate of the periodic repetition.

        Returns
        -------
        NDArray
            1D array of atom indices that fall within the specified
            periodic repetition.

        """

        # Shift the coordinates of the device atoms to the origin of the
        # contact
        relative_coordinates = self.device.atom_coordinates - self.origin

        # Compute the coefficients relative to the contact cell
        fractional_coordinates = relative_coordinates @ np.linalg.inv(
            self.lattice_vectors
        )

        # Get the indices of the atoms inside the periodic repetition
        indices_inside = np.nonzero(
            (fractional_coordinates[:, 0] >= nx)
            & (fractional_coordinates[:, 0] <= nx + 1)
            & (fractional_coordinates[:, 1] >= ny)
            & (fractional_coordinates[:, 1] <= ny + 1)
            & (fractional_coordinates[:, 2] >= nz)
            & (fractional_coordinates[:, 2] <= nz + 1)
        )[0]

        return indices_inside

    def _reorder_atoms(
        self, atom_indices: NDArray, idx: tuple[int, int, int], tol: float = 0.3
    ) -> NDArray:
        """Reorders atoms to match the ordering in the origin cell.

        This method ensures consistent atom ordering across different
        periodic repetitions of the contact unit cell.

        Parameters
        ----------
        atom_indices : NDArray
            Indices of atoms inside the periodic repetition to be
            reordered.
        idx : tuple[int, int, int]
            The coordinates of the periodic repetition.
        tol : float, optional
            Distance tolerance for atom matching, by default 0.3.

        Returns
        -------
        NDArray
            Reordered array of atom indices that correspond to the same
            ordering as the origin cell atoms.

        """

        sorted_atom_indices = []
        # Tolerance for the distance check

        # Shift the coordinates of the atoms inside the periodic
        # repetition to match the origin cell
        shifted_atom_coordinates = self.device.atom_coordinates[
            atom_indices, :
        ] - self.lattice_vectors @ np.array(idx)

        atoms_type = self.device.atoms_type[atom_indices]

        for origin_atom_index in self.origin_atom_indices:

            delta = (
                shifted_atom_coordinates
                - self.device.atom_coordinates[origin_atom_index, :]
            )

            # Find the atoms in the periodic repetition that are close
            # to the atom in the origin cell and have the same element
            found_atoms = np.nonzero(
                (np.linalg.norm(delta, axis=1) < tol)
                & (self.device.atoms_type[origin_atom_index] == atoms_type)
            )[0]
            if found_atoms.size == 0:
                raise ValueError(
                    f"Error in contact {self.name}: "
                    f"Atom {origin_atom_index} not found in the periodic repetition at ({idx})."
                )
            elif found_atoms.size > 1:
                raise ValueError(
                    f"Error in contact {self.name}: "
                    f"Multiple atoms found in the periodic repetition at ({idx}) "
                    f"matching atom {origin_atom_index} from the origin cell."
                )

            sorted_atom_indices.append(atom_indices[found_atoms[0]])

        return np.array(sorted_atom_indices, dtype=int)

    def _count_repetitions(self, axis: int, sign: int) -> int:
        """Counts periodic repetitions in a given direction.

        Parameters
        ----------
        axis : int
            The axis along which to count the repetitions (0, 1, or
            2).
        sign : int
            The sign of the direction to count the repetitions (1
            for positive, -1 for negative).

        Returns
        -------
        int
            The number of periodic repetitions in the given
            direction.

        """

        assert axis in [0, 1, 2], "Axis must be 0, 1, or 2."
        assert sign in [1, -1], "Sign must be 1 or -1."

        for repetition in itertools.count(start=1):
            idx = [0, 0, 0]
            idx[axis] = sign * repetition

            # Get the atoms inside the periodic repetition
            atom_indices = self._get_atom_indices_in_cell(*idx)

            if atom_indices.shape[0] == 0:
                break

            # Number of atoms inside the periodic repetition
            # does not match the origin cell
            if len(atom_indices) != len(self.origin_atom_indices):
                raise ValueError(
                    f"Atom mismatch at {tuple(idx)} in contact {self.name} for axis {axis} and sign {sign}."
                )

        # Minus one because the last repetition had no atoms
        return repetition - 1

    def _init_periodic_transverse_repetitions(self):
        """Determines number of periodic repetitions in transverse directions."""

        # Count the number of periodic repetitions in each transverse direction
        # (y+, y-, z+, z- )
        repetitions_y_pos = self._count_repetitions(self.transverse_axis[0], 1)
        repetitions_y_neg = self._count_repetitions(self.transverse_axis[0], -1)
        repetitions_z_pos = self._count_repetitions(self.transverse_axis[1], 1)
        repetitions_z_neg = self._count_repetitions(self.transverse_axis[1], -1)

        # Store the number of periodic repetitions in the contact object
        # and the coordinates of the origin cell
        self.origin_cell_offset = np.array((repetitions_y_neg, repetitions_z_neg))
        self.transverse_repetition_grid = np.array(
            [
                repetitions_y_pos + repetitions_y_neg + 1,
                repetitions_z_pos + repetitions_z_neg + 1,
            ]
        )

    def _init_orbitals(self):
        # Initialize the orbitals
        # for each periodic repetition in transverse directions
        ny, nz = self.transverse_repetition_grid
        self.orbitals_per_repetition = [[[] for _ in range(nz)] for _ in range(ny)]

        self.residual_orbitals = np.arange(self.device.hamiltonians[(0, 0, 0)].shape[0])

        residual_orbitals_old = self.residual_orbitals.copy()

        for transport_index in itertools.count(0):
            self._init_orbitals_transverse(transport_index)

            self._init_hamiltonian_overlap_matrices(transport_index)

            if self._residual_coupling() == 0:
                return transport_index + 1

            # The residual orbitals did not change
            # but there are still residual couplings
            # then some orbitals got missed
            if np.array_equal(residual_orbitals_old, self.residual_orbitals):
                raise ValueError(
                    f"Error in contact {self.name}: "
                    f"Could not find all orbitals in the contact unit cell. "
                )

            residual_orbitals_old = self.residual_orbitals.copy()

    def _init_orbitals_transverse(self, transport_index: int):
        """Initialize the orbitals for a given transport cell
        for all transverse periodic repetitions. Additionally,
        this method updates the residual orbitals.

        Parameters
        ----------
        transport_index : int
            The index of the periodic repetition in the transport
            direction.

        """

        # Iterate over all (x, y) combinations
        for idx, idy in itertools.product(
            range(self.transverse_repetition_grid[0]),
            range(self.transverse_repetition_grid[1]),
        ):
            index = [idx - self.origin_cell_offset[0], idy - self.origin_cell_offset[0]]
            index.insert(self.direction, transport_index)

            # Process atom and orbital indices
            atom_indices = self._get_atom_indices_in_cell(*index)
            atom_indices = self._reorder_atoms(atom_indices, index)
            orbital_indices = self._atom_to_orbital_indices(atom_indices)

            self.orbitals_per_repetition[idx][idy].append(orbital_indices)

            self.residual_orbitals = self.residual_orbitals[
                ~np.isin(self.residual_orbitals, orbital_indices)
            ]

    def _atom_to_orbital_indices(self, atom_indices: NDArray) -> NDArray:
        """Gets the orbital indices corresponding to the atoms

        Parameters
        ----------
        atom_indices : NDArray
            The indices of the atoms.

        Returns
        -------
        NDArray
            The indices of the orbitals corresponding to the atoms.

        """

        orbital_offsets = self.device.orbital_offsets
        starts = orbital_offsets[atom_indices]
        ends = orbital_offsets[atom_indices + 1]
        counts = ends - starts

        orbital_indices = np.repeat(starts, counts) + np.concatenate(
            [np.arange(c) for c in counts]
        )

        return orbital_indices

    def _get_circumference_coordinates(self, radius: int) -> list:
        """Gets coordinates only on the circumference of the grid.

        Parameters
        ----------
        radius : int
            The radius of the circumference.

        Returns
        -------
        list
            A list of tuples representing the coordinates on the
            circumference.

        """
        coordinates = []

        for y in range(-radius, radius + 1):
            for z in range(-radius, radius + 1):
                if max(abs(y), abs(z)) == radius:
                    coordinates.append((y, z))

        return coordinates

    def _init_hamiltonian_overlap_matrices(self, transport_index):
        """Initializes the hamiltonian matrix for the transverse contact cell


        Parameters
        ----------
        transport_index : int
            The index of the periodic repetition in the transport
            direction.

        """

        # The hamiltonian and overlap matrices for a given transverse
        # slice are obtained around the origin cell increasing radius
        # until no more hamiltonian or overlap is found.

        # Start with radius 0
        radius = 0

        while True:
            # Stop if no hamilonian or overlap is found
            found = False
            # Get the coordinates on the circumference of the grid
            coords_list = self._get_circumference_coordinates(radius)
            for atom_coordinates in coords_list:
                # Add the coordinates to the origin cell
                a = atom_coordinates[0] + self.origin_cell_offset[0]
                b = atom_coordinates[1] + self.origin_cell_offset[1]

                if self.device.gamma_only and (
                    (
                        self.transverse_repetition_grid[0] == 1
                        and atom_coordinates[0] != 0
                    )
                    or (
                        self.transverse_repetition_grid[1] == 1
                        and atom_coordinates[1] != 0
                    )
                ):
                    continue
                # The coupling is defined in the in the device
                # hamiltonian at (H_1, H_2) (it can be in any hopping
                # hamiltonian). Here we compute in which hopping
                # hamiltonian it is.
                H_1 = int((a + 0.0001) / self.transverse_repetition_grid[0])
                if a < 0:
                    H_1 -= 1
                H_2 = int((b + 0.0001) / self.transverse_repetition_grid[1])
                if b < 0:
                    H_2 -= 1

                if self.device.gamma_only and (
                    self.transverse_repetition_grid[0] > 1
                    or self.transverse_repetition_grid[1] > 1
                ):
                    H_1 = 0
                    H_2 = 0
                    if (
                        (2 * radius + 1) > self.transverse_repetition_grid[0]
                        and self.transverse_repetition_grid[0] > 1
                    ) or (
                        (2 * radius + 1) > self.transverse_repetition_grid[1]
                        and self.transverse_repetition_grid[1] > 1
                    ):
                        raise ValueError(
                            f"Error in contact {self.name}: \n"
                            f"I cannot obtain the UC matrices from the Gamma-point device matrix, probably because the basis decay is not enough.\n"
                            f"Possible solutions:\n"
                            f"  - Increase the UC to include the entire cross-section (1x1 contact UC)\n"
                            f"  - Provide all the hopping Hamiltonians in the device, not only the Gamma point."
                            f"Error encountered with radius {radius}"
                        )

                # These are the orbitals where to look for the coupling
                o_1 = a % self.transverse_repetition_grid[0]
                o_2 = b % self.transverse_repetition_grid[1]
                orb_coup = self.orbitals_per_repetition[o_1][o_2][transport_index]

                ham_tu = [H_1, H_2]
                ham_tu.insert(self.direction, 0)
                ham_tu = tuple(ham_tu)
                # Now get the hamiltonian and overlap matrices for the
                # current coordinates
                if ham_tu in self.device.hamiltonians:
                    ham_read = self.device.hamiltonians[ham_tu][
                        self.origin_orbital_indices, :
                    ][:, orb_coup]
                    if ham_read.nnz != 0:
                        self.UC_hamiltonian[
                            (transport_index, atom_coordinates[0], atom_coordinates[1])
                        ] = ham_read
                        if transport_index == 0:
                            # FORCE THE HAMILTONIAN TO BE HERMITIAN
                            self.UC_hamiltonian[
                                (
                                    transport_index,
                                    -atom_coordinates[0],
                                    -atom_coordinates[1],
                                )
                            ] = ham_read.T.conj()
                        found = True
                if ham_tu in self.device.overlap_matrices:
                    overlap_read = self.device.overlap_matrices[ham_tu][
                        self.origin_orbital_indices, :
                    ][:, orb_coup]
                    if overlap_read.nnz != 0:
                        self.UC_overlap[
                            (transport_index, atom_coordinates[0], atom_coordinates[1])
                        ] = overlap_read
                        if transport_index == 0:
                            # FORCE THE OVERLAP TO BE HERMITIAN
                            self.UC_overlap[
                                (
                                    transport_index,
                                    -atom_coordinates[0],
                                    -atom_coordinates[1],
                                )
                            ] = overlap_read.T.conj()
                        found = True

            if not found:
                if comm.rank == 0:
                    print(f"        Maximum coupling radius: {radius-1}")
                # if self.transverse_repetition_grid[0] == 1 and self.transverse_repetition_grid[1] == 1 and (radius
                #    - 1) > 0: raise ValueError("1x1 UC but more than
                #    1x1 coupling!")
                break

            radius += 1

    def _residual_coupling(self) -> bool:
        """Checks if there is residual coupling between the orbitals in
        the contact and the full device.

        Returns
        -------
        bool
            True if there is residual coupling, False otherwise.

        """

        return self.device.hamiltonians[0, 0, 0][self.origin_orbital_indices, :][
            :, self.residual_orbitals
        ].nnz

    def _configure_obc(
        self, obc_config: OBCConfig, nevp_config: NEVPConfig
    ) -> obc.Spectral:
        """Configures the OBC solver.

        Parameters
        ----------
        obc_config : OBCConfig
            Configuration object containing OBC algorithm settings
            including solver type, convergence parameters, and numerical
            options.
        nevp_config : NEVPConfig
            Configuration object containing NEVP solver settings
            including solver type and algorithm-specific parameters.

        Returns
        -------
        obc_solver: obc.Spectral
            Configured spectral OBC solver ready for boundary condition
            calculations.

        """
        if obc_config.algorithm == "sancho-rubio":
            raise NotImplementedError(
                "Sancho-rubio OBC algorithm does not work with QTBM, please use spectral OBC solver."
            )

        elif obc_config.algorithm == "spectral":
            nevp = self._configure_nevp(obc_config, nevp_config)
            obc_solver = obc.Spectral(
                nevp=nevp,
                block_sections=obc_config.block_sections,
                min_decay=obc_config.min_decay,
                max_decay=obc_config.max_decay,
                num_ref_iterations=obc_config.num_ref_iterations,
                min_propagation=obc_config.min_propagation,
                residual_tolerance=obc_config.residual_tolerance,
                residual_normalization=obc_config.residual_normalization,
                warning_threshold=obc_config.warning_threshold,
                eta_decay=obc_config.eta_decay,
            )

        else:
            raise NotImplementedError(
                f"OBC algorithm '{obc_config.algorithm}' not implemented."
            )

        return obc_solver

    def _configure_nevp(self, obc_config: OBCConfig, nevp_config: NEVPConfig) -> NEVP:
        """Configures the Nonlinear Eigenvalue Problem (NEVP) solver.

        Parameters
        ----------
        obc_config : OBCConfig
            Configuration object containing NEVP solver settings
            including solver type and algorithm-specific parameters.
        nevp_config : NEVPConfig
            Configuration object containing NEVP solver settings
            including solver type and algorithm-specific parameters.

        Returns
        -------
        NEVP
            Configured NEVP solver ready for eigenvalue calculations.

        """
        if obc_config.nevp_solver == "beyn":
            return Beyn(
                r_o=obc_config.r_o,
                r_i=obc_config.r_i,
                m_0=obc_config.m_0,
                num_quad_points=obc_config.num_quad_points,
                num_threads_contour=nevp_config.num_threads_contour,
                eig_compute_location=nevp_config.eig_compute_location,
                project_compute_location=nevp_config.project_compute_location,
                use_qr=nevp_config.use_qr,
                contour_batch_size=nevp_config.contour_batch_size,
                use_pinned_memory=nevp_config.use_pinned_memory,
            )
        if obc_config.nevp_solver == "full":

            a_sparsity = None
            if nevp_config.reduce_sparsity:

                a_sparsity = [
                    xp.zeros_like(self.UC_hamiltonian[0, 0, 0].toarray())
                    for _ in range(2 * self.transverse_rep + 1)
                ]

                for key, values in self.UC_hamiltonian.items():
                    values = values.toarray()
                    a_sparsity[self.transverse_rep + key[0]] += values != 0
                    a_sparsity[self.transverse_rep - key[0]] += values.T != 0

                for key, values in self.UC_overlap.items():
                    values = values.toarray()
                    a_sparsity[self.transverse_rep + key[0]] += values != 0
                    a_sparsity[self.transverse_rep - key[0]] += values.T != 0

                a_sparsity = tuple(a_sparsity)

            return Full(
                eig_compute_location=nevp_config.eig_compute_location,
                a_sparsity=a_sparsity,
                reduce=nevp_config.reduce_sparsity,
            )

        raise NotImplementedError(
            f"NEVP solver '{obc_config.nevp_solver}' not implemented."
        )

    def get_10(self, M: sparse.spmatrix) -> NDArray:
        """Extracts coupling matrix between device and contact.

        This method constructs the matrix that couples the device region
        to the contact.

        Parameters
        ----------
        M : sparse.spmatrix
            The matrix (Hamiltonian or overlap) from which to extract
            coupling elements. Should have dimensions
            (n_device_orbitals, n_device_orbitals).

        Returns
        -------
        NDArray
            Dense matrix representing the coupling between device and
            contact. The matrix has the block structure needed for QTBM
            boundary conditions, with dimensions determined by the
            contact's transverse repetitions.

        """

        mat_list = []
        n = self.orbitals_get_10[0].shape[0]
        for i in range(1, self.transverse_rep + 1):
            # Get the hamiltonian matrix for the current key and index
            mat_list.append(M[self.orbitals_get_10[i], :][:, self.orbitals_get_10[0]])
        h10_temp = sparse.vstack(mat_list, format="csr")
        for i in range(self.transverse_rep - 1):
            mat_list.pop(0)
            mat_list.append(sparse.csr_matrix((n, n), dtype=xp.complex128))
            h10_temp = sparse.hstack(
                [
                    sparse.vstack(mat_list, format="csr"),
                    h10_temp,
                ],
                format="csr",
            )

        return h10_temp[self.sort_orbitals_get_10.T, self.sort_orbitals_get_10]

    def _get_list_mat_phase(self, k1: float, k2: float) -> NDArray:
        """Gets the list of hopping matrices in transport direction with
        the corresponding phase factors (for the transverse direction).

        Parameters
        ----------
        k1 : float
            The k1 value for the phase factor.
        k2 : float
            The k2 value for the phase factor.

        Returns
        -------
        tuple
            A tuple containing two lists: the list of hamiltonian
            matrices and the list of overlap matrices.

        """
        # Size of the hamiltonian and overlap matrices
        n = self.UC_hamiltonian[(0, 0, 0)].shape[0]

        # Initialize the lists of hamiltonian and overlap matrices
        H_coup = []
        S_coup = []

        # Create empty matrices for each repetion in the transport
        # direction
        for ii in range(self.transverse_rep + 1):
            H_coup.append(sparse.csr_matrix((n, n), dtype=xp.complex128))
            S_coup.append(sparse.csr_matrix((n, n), dtype=xp.complex128))

        # Fill the matrices with the hamiltonian and overlap matrices
        # from the unit cell and apply the phase factors for the
        # transverse direction (H^0, H^1, H^2)
        for index, ham in self.UC_hamiltonian.items():
            H_coup[index[0]] += ham * xp.exp(1j * (k1 * index[1] + k2 * index[2]))
        for index, overlap in self.UC_overlap.items():
            S_coup[index[0]] += overlap * xp.exp(1j * (k1 * index[1] + k2 * index[2]))

        # Add the conjugate transpose, for example (H^-2, H^-1, H^0,
        # H^1, H^2)
        for ii in range(self.transverse_rep):
            H_coup.insert(0, H_coup[ii * 2 + 1].conj().T)
            S_coup.insert(0, S_coup[ii * 2 + 1].conj().T)

        # Augment with emtpy matrices (needed for the OBC solver) (H^-2,
        # H^-1, H^0, H^1, H^2, H^3)
        for ii in range(self.transverse_rep - 1):
            H_coup.append(sparse.csr_matrix((n, n), dtype=xp.complex128))
            S_coup.append(sparse.csr_matrix((n, n), dtype=xp.complex128))

        return H_coup, S_coup

    def _construct_circulant_matrix(self, list_mat: list, phase: float) -> NDArray:
        """Constructs circulant matrix from list of matrices with a
        given phase factor.

        Parameters
        ----------
        list_mat : list
            A list of matrices to construct the circulant matrix from.
        phase : float
            The phase factor to apply to the matrices.

        Returns
        -------
        NDArray
            The constructed circulant matrix.

        """
        # Number of matrices in the list
        n = len(list_mat)
        size_mat_batch = list_mat[0].shape[0]
        size_mat_orb = list_mat[0].shape[1]
        # Initialize the circulant matrix (first row)
        mat = xp.zeros(
            (size_mat_batch, size_mat_orb * n, size_mat_orb * n), dtype=xp.complex128
        )
        # Iterate over the number of matrices and apply the phase factor
        for i in range(n):
            mat[:, size_mat_orb * i : size_mat_orb * (i + 1), :] = xp.concatenate(
                list_mat, axis=2
            )
            list_mat.insert(0, list_mat.pop() * phase)

        return mat

    def _upscale_self_energy(
        self, SE_k: dict, k_1_list: NDArray, k_2_list: NDArray, k1: float, k2: float
    ) -> NDArray:
        """Upscales self-energy matrices using circulant matrix.

        Parameters
        ----------
        SE_k : dict
            A dictionary containing self-energy matrices indexed by (k1,
            k2) tuples.
        k_1_list : NDArray
            A list of k1 values for the phase factor.
        k_2_list : NDArray
            A list of k2 values for the phase factor.
        k1 : float
            The k1 value for the lower diagonal phase factor.
        k2 : float
            The k2 value for the lower diagonal phase factor.

        Returns
        -------
        NDArray
            The upscaled self-energy matrix.

        """
        SE_1 = []
        for i in range(self.transverse_repetition_grid[0]):
            SE_2 = []
            for j in range(self.transverse_repetition_grid[1]):
                # Initialize the self-energy matrix sub block
                SE_2.append(
                    xp.zeros_like(SE_k[(k_1_list[0].item(), k_2_list[0].item())])
                )
                # Constuct the self-energy subblock
                for key, value in SE_k.items():
                    SE_2[-1] += value * xp.exp(-1j * (key[0] * i + key[1] * j))
            # Construct the circulant matrix sublock
            SE_1.append(self._construct_circulant_matrix(SE_2, xp.exp(1j * k2)))

        # Construct the final self-energy matrix
        SE_1 = self._construct_circulant_matrix(SE_1, xp.exp(1j * k1)) / (
            self.transverse_repetition_grid[0] * self.transverse_repetition_grid[1]
        )

        return SE_1

    def _upscale_injection_modes(self, modes_k: dict, E: NDArray) -> NDArray:
        """Upscales injection vectors.

        Parameters
        ----------
        modes_k : dict
            A dictionary containing injection vectors indexed by (k1,
            k2) tuples.
        E : NDArray
            The batch of energies for which to compute the total
            inhection vectors.

        Returns
        -------
        NDArray
            The upscaled and concatenated injection vectors.

        """
        # Upscale the k-space modes Iterate over the wavevector keys
        for key, value in modes_k.items():
            # Iterate over the energies in the batch
            for i_E in range(len(value)):
                I_1 = []
                I_2 = []
                # Upscale in 2nd direction first
                for j in range(self.transverse_repetition_grid[1]):
                    I_2.append(modes_k[key][i_E] * xp.exp(1j * (key[1] * j)))
                I_2 = xp.concatenate(I_2, axis=0)
                # Upscale in 1st direction
                for i in range(self.transverse_repetition_grid[0]):
                    I_1.append(I_2 * xp.exp(1j * (key[0] * i)))
                I_1 = xp.concatenate(I_1, axis=0)
                # Store the upscaled modes
                modes_k[key][i_E] = I_1

        # Concatenate all the wavevector (transverse)
        modes = []
        # Iterate over the energies in the batch
        for i_E in range(E.shape[0]):
            modes_E = []
            # Iterate over the wavevector keys
            for key, value in modes_k.items():
                modes_E.append(value[i_E])
            modes_E = xp.concatenate(modes_E, axis=1) / xp.sqrt(
                self.transverse_repetition_grid[0] * self.transverse_repetition_grid[1]
            )
            modes.append(modes_E)

            # SORTING AND NORMALIZATION IS ONLY FOR DEBUG (IT IS NOT
            # REALLY NEEDED) modes[-1] /=
            # xp.exp(1j*xp.angle(modes[-1][0,:])) sort_indices =
            # xp.argsort(modes[-1][0, :]) modes[-1] = modes[-1][:,
            # sort_indices]

        return modes

    def compute_boundary(
        self, k_outer: tuple[float, float, float], E: NDArray
    ) -> tuple:
        """Computes OBC for the contact at given k-points and energies.

        Parameters
        ----------
        k_outer : tuple[float, float, float]
            Wavevector. Captures periodicity in transverse directions.
        E : NDArray
            Batch of energy values for which to compute the boundary
            conditions.

        Returns
        -------
        tuple
            A tuple containing the computed self-energy, injection
            vectors, number of injected modes, transmission matrices, and
            Bloch injection matrices.

        """

        # FOR NOW, ONLY E BATCHING IS SUPPORTED
        k_outer = list(k_outer)

        if k_outer[self.direction] != 0:
            raise ValueError(
                f"Error in contact {self.name}: "
                f"You can't compute the OBC for a non-zero k-point in the transport direction ({self.direction}). "
            )
        # Remove the k-point in the transport direction
        k_outer.pop(self.direction)

        # Create the k-space list needed to upscale the self-energy and
        # injection modes in the transverse directions
        k_inner = [
            np.linspace(0, np.pi * 2, n_rep, endpoint=False) + k_outer[i] / n_rep
            for i, n_rep in enumerate(self.transverse_repetition_grid)
        ]

        sigma_obc_k = {}
        inj_k = {}
        num_inj_k = {}
        K_k = {}
        T_k = {}

        for k_i, k_j in itertools.product(k_inner[0], k_inner[1]):
            # Construct the hamiltonian and overlap matrices for the
            # given ki and kj
            H_list, S_list = self._get_list_mat_phase(k_i, k_j)
            H_tot = sparse.hstack(H_list, format="csr")
            S_tot = sparse.hstack(S_list, format="csr")

            # Create the toeplitz structure for the hamiltonian and
            # overlap matrices (in transport direction)
            for ii in range(self.transverse_rep - 1):
                H_list.insert(0, H_list.pop())
                S_list.insert(0, S_list.pop())
                H_tot = sparse.vstack(
                    [H_tot, sparse.hstack(H_list, format="csr")], format="csr"
                )
                S_tot = sparse.vstack(
                    [S_tot, sparse.hstack(S_list, format="csr")], format="csr"
                )

            S_dense = xp.array(S_tot.todense())
            H_dense = xp.array(H_tot.todense())

            # Construct the system matrices for the OBC solver
            A_tot = xp.split((E[:, None, None] * S_dense - H_dense), 3, axis=2)

            # Solve the OBC for the given ki and kj and store the
            # results in dictionaries
            self.obc_solver.block_sections = self.transverse_rep

            x_ii, phi_surface = self.obc_solver(
                A_tot[1], A_tot[2], A_tot[0], "left", return_injected=True
            )
            sigma_obc_k[k_i, k_j] = (
                A_tot[0]
                @ x_ii
                @ A_tot[2]
                / (
                    self.transverse_repetition_grid[1]
                    * self.transverse_repetition_grid[0]
                )
            )
            inj_k[k_i, k_j] = []

            for i, phi in enumerate(phi_surface):
                inj_k[k_i, k_j].append(-A_tot[0][i] @ phi)

            num_inj_k[k_i, k_j] = []
            for phi in phi_surface:
                num_inj_k[k_i, k_j].append(phi.shape[1])

            K_k[k_i, k_j] = phi_surface
            T_k[k_i, k_j] = (
                -x_ii
                @ A_tot[2]
                / (
                    self.transverse_repetition_grid[1]
                    * self.transverse_repetition_grid[0]
                )
            )

            if comm.rank == 0:
                print(f"    Computed OBC for k1={k_i}, k2={k_j}", flush=True)

        # Upscale self-energy and Bloch matrices
        # sigma_obc = self._upscale_self_energy(
        #    sigma_obc_k, k_inner[0], k_inner[1], k_outer[0], k_outer[1]
        # )
        # T = self._upscale_self_energy(
        #    T_k, k_inner[0], k_inner[1], k_outer[0], k_outer[1]
        # )

        # Upscale injection and Bloch injection matrices
        inj = self._upscale_injection_modes(inj_k, E)
        K = self._upscale_injection_modes(K_k, E)

        # Calculate total number of injected modes
        num_inj = np.zeros(E.shape[0], dtype=np.int32)
        for i_E in range(E.shape[0]):
            for value in num_inj_k.values():
                num_inj[i_E] += value[i_E]

        return inj, num_inj, K, sigma_obc_k, T_k

    def _compute_band_structure(self, device):

        if comm.rank == 0:
            print(
                f"    Computing band structure for contact {self.name}...", flush=True
            )

        # Generate k-points in the transverse directions
        kpoints = monkhorst_pack(device.quatrex_config.device.kpoint_grid)
        kpoints += np.array(device.quatrex_config.device.kpoint_shift)
        num_kpoints = kpoints.shape[0]

        if (
            device.quatrex_config.device.kpoint_grid[self.direction] != 1
            or device.quatrex_config.device.kpoint_shift[self.direction] != 0
        ):
            raise ValueError(
                f"Error in contact {self.name}: "
                f"Band structure calculation requires k-point grid of 1 and shift of 0 in transport direction ({self.direction}). "
            )

        # TODO move to CONFIG
        n_points_BAND = 51

        # Generate k-points in the transport direction
        k_transport = xp.linspace(
            -xp.pi,
            xp.pi,
            n_points_BAND,
        )

        # Initialize band structure array
        self.band_structure = xp.zeros(
            (num_kpoints, n_points_BAND, self.origin_number_of_orbitals),
            dtype=xp.float64,
        )

        for i, k_perp in enumerate(kpoints):
            for j, k_par in enumerate(k_transport):

                # Reconstruct full k-vector
                k_vec = list(k_perp)
                k_vec.pop(self.direction)
                k_vec.insert(0, k_par)

                # Construct Hamiltonian and Overlap at k-point
                H_tot = sparse.csr_matrix(
                    (self.origin_number_of_orbitals, self.origin_number_of_orbitals),
                    dtype=xp.complex128,
                )
                S_tot = sparse.csr_matrix(
                    (self.origin_number_of_orbitals, self.origin_number_of_orbitals),
                    dtype=xp.complex128,
                )
                # Sum over all the hoppings in the UC with the
                # corresponding phase factors
                for index, ham in self.UC_hamiltonian.items():
                    phase = xp.exp(
                        1j
                        * (
                            k_vec[0] * index[0]
                            + k_vec[1] * index[1]
                            + k_vec[2] * index[2]
                        )
                    )
                    H_tot += ham * phase
                    if index[0] > 0:
                        H_tot += ham.T.conj() * phase.conj()
                for index, overlap in self.UC_overlap.items():
                    phase = xp.exp(
                        1j
                        * (
                            k_vec[0] * index[0]
                            + k_vec[1] * index[1]
                            + k_vec[2] * index[2]
                        )
                    )
                    S_tot += overlap * phase
                    if index[0] > 0:
                        S_tot += overlap.T.conj() * phase.conj()

                # DENSIFY (for now)
                H_tot = xp.array(H_tot.todense())
                S_tot = xp.array(S_tot.todense())
                # Solve generalized eigenvalue problem
                chol = xp.linalg.cholesky(S_tot)
                chol_inv = xp.linalg.inv(chol)
                H_ortho = chol_inv @ H_tot @ chol_inv.conj().T

                self.band_structure[i, j, :] = xp.linalg.eigvalsh(H_ortho)
