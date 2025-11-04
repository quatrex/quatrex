import itertools

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, obc, sparse, xp
from qttools.nevp import NEVP, Beyn, Full
from quatrex.core.compute_config import NEVPConfig
from quatrex.core.quatrex_config import OBCConfig


# TODO move to some kid of utils
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
    vectors : NDArray
        The lattice vectors defining the unit cell of the contact.
    direction : str
        The transport direction of the contact, specified as 'a', 'b',
        or 'c' corresponding to the lattice axes.

    Attributes
    ----------
    name : str
        The contact identifier.
    device : Device
        Reference to the parent device.
    vectors : NDArray
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
    atoms : list[NDArray]
        List of atom indices for each contact cell repetition.
    orbitals : list[NDArray]
        List of orbital indices for each contact cell repetition.
    n_rep_1, n_rep_2 : int
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
        vectors: NDArray,
        direction: str,
        fermi_level: float,
        temperature: float,
    ):
        """Initializes the contact object."""

        self.name = name
        self.device = device

        self.fermi_level = fermi_level
        self.temperature = temperature

        self.vectors = vectors
        self.origin = origin

        self.direction = "abc".index(direction)
        self.transverse_axis = [0, 1, 2]
        self.transverse_axis.remove(self.direction)

        self.UC_hamiltonian = {}
        self.UC_overlap = {}

        self.S10_contact = {}
        self.H10_contact = {}

        if len(self.transverse_axis) != 2:
            raise ValueError("Direction must be one of the three axes (0, 1, or 2).")

        relative_coords = self.device.coords - self.origin
        self.coeffs = relative_coords @ np.linalg.inv(self.vectors)

        # Get the atoms inside the origin cell (defined by the user)
        self.at_origin_cell = self._get_atoms_inside_cell(0, 0, 0)
        self.orb_origin_cell = self._get_orbitals(self.at_origin_cell)
        self.n_at_origin_cell = self.at_origin_cell.shape[0]
        self.n_orb_origin_cell = self.orb_origin_cell.shape[0]

        if comm.rank == 0:
            print(f"Contact {self.name}:", flush=True)
            print(
                f"    Number of atoms inside the origin cell: {self.at_origin_cell.shape[0]}",
                flush=True,
            )

        # Check how many periodic repetitions are in the transverse
        # directions
        self._get_periodic_number_transverse()
        if comm.rank == 0:
            print(
                f"    Number of periodic repetitions in the transverse directions: {self.n_rep_1} x {self.n_rep_2}",
                flush=True,
            )

        # TODO Check if the contact transverse UC vectors are in the
        # same direction as the device vectors

        # Get the atoms (and orbtals) in the first contact cell
        self.atoms = []
        self.orbitals = []
        self.atoms.append(self._get_atoms_transverse_sorted(0))
        self.orbitals.append(self._get_orbitals(self.atoms[0]))

        # Get the hamiltonian and overlap matrices for the first contact
        # cell
        if comm.rank == 0:
            print("    Getting matrices for contact cell in repetition=0", flush=True)
        self._get_matrix(0)

        x = 1
        # Iterate over the transport direction until there is no more
        # residual coupling in the contact cell
        while self._residual_coupling(self.orbitals) > 0:
            if comm.rank == 0:
                print(
                    f"        Residual coupling={self._residual_coupling(self.orbitals)}",
                    flush=True,
                )

            # Get atoms, orbitals and matrices for the next contact cell
            self.atoms.append(self._get_atoms_transverse_sorted(x))
            self.orbitals.append(self._get_orbitals(self.atoms[x]))
            if comm.rank == 0:
                print(
                    f"    Getting matrices for contact cell in repetition={x}",
                    flush=True,
                )
            self._get_matrix(x)

            x = x + 1

        if comm.rank == 0:
            print(
                f"    Maximum number of repetitions in transport direction: {x-1}",
                flush=True,
            )

        # In case of multiple contact cells in the transport direction,
        # and in case of multiple unit cells in the transverse
        # direction, we will obtain a the self energy sorted in
        # different way. atoms_2 will be used to sort the atoms in a
        # consistent way (sorted over the transport direction).
        self.atoms_2 = []
        self.orbitals_2 = []

        for i in range(self.n_rep_1):
            for j in range(self.n_rep_2):
                for k in range(x - 1):
                    # start = i*self.n_at_origin_cell*self.n_rep_2 +
                    # j*self.n_at_origin_cell end = start +
                    # self.n_at_origin_cell
                    # self.atoms_2.append(self.atoms[k][start:end])
                    start = self.n_orb_origin_cell * (
                        k * self.n_rep_1 * self.n_rep_2 + i * self.n_rep_2 + j
                    )
                    end = start + self.n_orb_origin_cell
                    self.orbitals_2.append(np.arange(start, end))

        # Concatenate the atoms_2 list and get the orbitals order for
        # the unsorted cell self.atoms_2 = xp.concatenate(self.atoms_2,
        # dtype=int) self.orbitals_2 = self._get_orbitals(self.atoms_2)
        self.orbitals_2 = np.concatenate(self.orbitals_2, dtype=int)

        self.orbitals_contact = np.concatenate(self.orbitals[:-1])[None, :]

        self.transverse_rep = x - 1
        self.obc_solver = self._configure_obc(
            device.quatrex_config.electron.obc, device.compute_config.nevp
        )

        self._compute_band_structure(device)

    def _get_atoms_inside_cell(self, nx: int, ny: int, nz: int) -> NDArray:
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
        relative_coords = self.device.coords - self.origin

        # Compute the coefficients relative to the contact cell
        coeffs = relative_coords @ np.linalg.inv(self.vectors)

        # Get the indices of the atoms inside the periodic repetition
        inside_mask = np.nonzero(
            (coeffs[:, 0] >= nx)
            & (coeffs[:, 0] <= nx + 1)
            & (coeffs[:, 1] >= ny)
            & (coeffs[:, 1] <= ny + 1)
            & (coeffs[:, 2] >= nz)
            & (coeffs[:, 2] <= nz + 1)
        )[0]

        return inside_mask

    def _reorder_atoms(
        self, at_inside_rep: NDArray, a: int, b: int, c: int, tol: float = 0.3
    ) -> NDArray:
        """Reorders atoms to match the ordering in the origin cell.

        This method ensures consistent atom ordering across different
        periodic repetitions of the contact unit cell.

        Parameters
        ----------
        at_inside_rep : NDArray
            Indices of atoms inside the periodic repetition to be
            reordered.
        a : int
            The x-coordinate of the periodic repetition.
        b : int
            The y-coordinate of the periodic repetition.
        c : int
            The z-coordinate of the periodic repetition.
        tol : float, optional
            Distance tolerance for atom matching, by default 0.1.

        Returns
        -------
        NDArray
            Reordered array of atom indices that correspond to the same
            ordering as the origin cell atoms.

        """

        sorted = []
        # Tolerance for the distance check

        a = int(a)  # Ensure a, b, c are integers
        b = int(b)
        c = int(c)
        # Shift the coordinates of the atoms inside the periodic
        # repetition to match the origin cell
        list_vec = np.array([a, b, c], dtype=np.float64)
        coords_rep = self.device.coords[at_inside_rep, :] - self.vectors @ list_vec
        element_rep = self.device.atom_type[at_inside_rep]
        for at in self.at_origin_cell:
            # Find the atoms in the periodic repetition that are close
            # to the atom in the origin cell and have the same element
            delta = coords_rep - self.device.coords[at, :]
            found = np.nonzero(
                (np.linalg.norm(delta, axis=1) < tol)
                & (self.device.atom_type[at] == element_rep)
            )[0]
            if found.size == 0:
                raise ValueError(
                    f"Error in contact {self.name}: "
                    f"Atom {at} not found in the periodic repetition at ({a}, {b}, {c})."
                    f"Min distance found: {np.min(np.linalg.norm(delta, axis=1))}"
                )
            if found.size > 1:
                raise ValueError(
                    f"Error in contact {self.name}: "
                    f"Multiple atoms found in the periodic repetition at ({a}, {b}, {c}) "
                    f"matching atom {at} from the origin cell."
                )
            # Append the index of the found atom to the sorted list
            sorted.append(at_inside_rep[found[0]])

        return np.array(sorted, dtype=int)

    def _get_periodic_number_transverse(self):
        """Determines number of periodic repetitions in transverse directions."""

        def count_reps(axis: int, sign: int) -> int:
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
            # Start counting from the origin cell
            n_rep = 0
            while True:
                # Get the index of the periodic repetition
                idx = [0, 0, 0]
                idx[axis] = sign * (n_rep + 1)
                # Get the atoms inside the periodic repetition
                at_inside_rep = self._get_atoms_inside_cell(*idx)
                # If no atoms are found, break the loop
                if at_inside_rep.shape[0] == 0:
                    break
                # If the number of atoms inside the periodic repetition
                # does not match the origin cell, raise an error
                if at_inside_rep.shape[0] != self.at_origin_cell.shape[0]:
                    pos = tuple(idx)
                    raise ValueError(
                        f"Error in contact {self.name}: "
                        f"Number of atoms inside the cell at {pos} "
                        f"does not match the number of atoms inside the origin cell."
                    )
                # Increment the count of periodic repetitions
                n_rep += 1
            return n_rep

        # 4 transverse directions: y+, y-, z+, z- Count the number of
        # periodic repetitions in each transverse direction
        n_rep_1_plus = count_reps(axis=self.transverse_axis[0], sign=1)
        n_rep_1_minus = count_reps(axis=self.transverse_axis[0], sign=-1)
        n_rep_2_plus = count_reps(axis=self.transverse_axis[1], sign=1)
        n_rep_2_minus = count_reps(axis=self.transverse_axis[1], sign=-1)

        # Store the number of periodic repetitions in the contact object
        # and the coordinates of the origin cell
        self.origin_cell = np.array((n_rep_1_minus, n_rep_2_minus))
        self.n_rep_1 = n_rep_1_plus + n_rep_1_minus + 1
        self.n_rep_2 = n_rep_2_plus + n_rep_2_minus + 1
        self.n_reps = [self.n_rep_1, self.n_rep_2]

    def _get_atoms_transverse_sorted(self, x: int) -> NDArray:
        """Gets the indices of the atoms inside the periodic repetition.

        Parameters
        ----------
        x : int
            The index of the periodic repetition in the transport
            direction.

        Returns
        -------
        NDArray
            The indices of the atoms inside the periodic repetition.

        """

        atom_list = []

        # Start from the (0,0) cell and look for periodic repetitions in
        # the transverse directions The origin cell is defined by the
        # user, so we start from -origin_cell (that is a tuple! (x,y))
        # and go up to (n_rep_1, n_rep_2)
        curr_cell = -self.origin_cell
        max_cell = curr_cell + np.array([self.n_rep_1, self.n_rep_2])

        while curr_cell[0] < max_cell[0]:
            while curr_cell[1] < max_cell[1]:
                # Get the indices of the atoms inside the periodic
                # repetition
                idx = [curr_cell[0], curr_cell[1]]
                idx.insert(self.direction, x)
                at_inside_rep = self._get_atoms_inside_cell(*idx)
                # Reorder the atoms to match the order in the origin
                # cell
                c_list = [curr_cell[0], curr_cell[1]]
                c_list.insert(self.direction, x)
                at_inside_rep = self._reorder_atoms(
                    at_inside_rep, c_list[0], c_list[1], c_list[2]
                )
                atom_list.append(at_inside_rep)
                curr_cell[1] += 1

            curr_cell[0] += 1
            curr_cell[1] = -self.origin_cell[1]

        return np.concatenate(atom_list, dtype=int)

    def _get_orbitals(self, atom_inds: NDArray) -> NDArray:
        """Gets the orbital indices corresponding to the atoms

        Parameters
        ----------
        atoms : NDArray
            The indices of the atoms.

        Returns
        -------
        NDArray
            The indices of the orbitals corresponding to the atoms.

        """

        orbital_ind = np.array([], dtype=np.int32)
        for atom_ind in atom_inds:
            # Starting orbitals for the current atom
            k1 = self.device.orbital_offsets[atom_ind]
            # Ending orbitals for the current atom (can be computed from
            # orbital_offsets)
            k2 = self.device.orbital_offsets[atom_ind + 1]
            orbital_ind = np.concatenate((orbital_ind, np.arange(k1, k2)))

        return orbital_ind

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
                # Check if point is on the boundary (max distance =
                # radius)
                if max(abs(y), abs(z)) == radius:
                    coordinates.append((y, z))

        return coordinates

    def _get_matrix(self, x: int) -> None:
        """Gets the hamiltonian matrix for the transverse contact cell
        at some distance x.

        Parameters
        ----------
        x : int
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
            for coords in coords_list:
                # Add the coordinates to the origin cell
                a = coords[0] + self.origin_cell[0]
                b = coords[1] + self.origin_cell[1]

                if self.device.gamma_only and (
                    (self.n_rep_1 == 1 and coords[0] != 0)
                    or (self.n_rep_2 == 1 and coords[1] != 0)
                ):
                    continue
                # The coupling is defined in the in the device
                # hamiltonian at (H_1, H_2) (it can be in any hopping
                # hamiltonian). Here we compute in which hopping
                # hamiltonian it is.
                H_1 = int((a + 0.0001) / self.n_rep_1)
                if a < 0:
                    H_1 -= 1
                H_2 = int((b + 0.0001) / self.n_rep_2)
                if b < 0:
                    H_2 -= 1

                if self.device.gamma_only and (self.n_rep_1 > 1 or self.n_rep_2 > 1):
                    H_1 = 0
                    H_2 = 0
                    if ((2 * radius + 1) > self.n_rep_1 and self.n_rep_1 > 1) or (
                        (2 * radius + 1) > self.n_rep_2 and self.n_rep_2 > 1
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
                o_1 = a % self.n_rep_1
                o_2 = b % self.n_rep_2
                d1 = (
                    self.n_orb_origin_cell * self.n_rep_2 * o_1
                    + self.n_orb_origin_cell * o_2
                )
                d2 = d1 + self.n_orb_origin_cell
                orb_coup = self.orbitals[x][d1:d2]

                ham_tu = [H_1, H_2]
                ham_tu.insert(self.direction, 0)
                ham_tu = tuple(ham_tu)
                # Now get the hamiltonian and overlap matrices for the
                # current coordinates
                if ham_tu in self.device.hamiltonian:
                    ham_read = self.device.hamiltonian[ham_tu][self.orb_origin_cell, :][
                        :, orb_coup
                    ]
                    if ham_read.nnz != 0:
                        self.UC_hamiltonian[(x, coords[0], coords[1])] = ham_read
                        if x == 0:
                            # FORCE THE HAMILTONIAN TO BE HERMITIAN
                            self.UC_hamiltonian[(x, -coords[0], -coords[1])] = (
                                ham_read.T.conj()
                            )
                        found = True
                if ham_tu in self.device.overlap:
                    overlap_read = self.device.overlap[ham_tu][self.orb_origin_cell, :][
                        :, orb_coup
                    ]
                    if overlap_read.nnz != 0:
                        self.UC_overlap[(x, coords[0], coords[1])] = overlap_read
                        if x == 0:
                            # FORCE THE OVERLAP TO BE HERMITIAN
                            self.UC_overlap[(x, -coords[0], -coords[1])] = (
                                overlap_read.T.conj()
                            )
                        found = True

            if not found:
                if comm.rank == 0:
                    print(f"        Maximum coupling radius: {radius-1}")
                # if self.n_rep_1 == 1 and self.n_rep_2 == 1 and (radius
                #    - 1) > 0: raise ValueError("1x1 UC but more than
                #    1x1 coupling!")
                break

            radius += 1

    def _residual_coupling(self, orbitals: list) -> bool:
        """Checks if there is residual coupling between the orbitals in
        the contact and the full device.

        Parameters
        ----------
        orbitals : list
            A list of orbital indices for which to check the residual
            coupling.

        Returns
        -------
        bool
            True if there is residual coupling, False otherwise.

        """

        tot_orb = np.arange(self.device.hamiltonian[(0, 0, 0)].shape[0])
        tot_orb = tot_orb[~np.isin(tot_orb, np.concatenate(orbitals))]

        return self.device.hamiltonian[0, 0, 0][orbitals[0], :][:, tot_orb].nnz

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

        hamiltonian_list = []
        n = self.orbitals[0].shape[0]
        for i in range(self.transverse_rep + 1):
            # Get the hamiltonian matrix for the current key and index
            hamiltonian_list.append(M[self.orbitals[0], :][:, self.orbitals[i]])
        for i in range(self.transverse_rep - 1):
            hamiltonian_list.append(sparse.csr_matrix((n, n), dtype=xp.complex128))
        h10_temp = sparse.hstack(hamiltonian_list[-self.transverse_rep :])
        for i in range(self.transverse_rep - 1):
            hamiltonian_list.pop()
            h10_temp = sparse.vstack(
                [h10_temp, sparse.hstack(hamiltonian_list[-self.transverse_rep :])]
            )
        return h10_temp.T.conj().todense()

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
        # Initialize the circulant matrix (first row)
        mat = xp.concatenate(list_mat, axis=2)
        # Iterate over the number of matrices and apply the phase factor
        for i in range(n - 1):
            list_mat.insert(0, list_mat.pop() * phase)
            mat = xp.concatenate([mat, xp.concatenate(list_mat, axis=2)], axis=1)

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
        for i in range(self.n_rep_1):
            SE_2 = []
            for j in range(self.n_rep_2):
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
            self.n_rep_1 * self.n_rep_2
        )
        # Reorder the self-energy matrix to match the orbitals of the
        # contact SE_1[:, self.orbitals_2[None,:].T,
        # self.orbitals_2[None,:]] = SE_1
        SE_temp = xp.zeros_like(SE_1)
        SE_temp[:, self.orbitals_2[None, :].T, self.orbitals_2[None, :]] = SE_1

        return SE_temp

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
                for j in range(self.n_rep_2):
                    I_2.append(modes_k[key][i_E] * xp.exp(1j * (key[1] * j)))
                I_2 = xp.concatenate(I_2, axis=0)
                # Upscale in 1st direction
                for i in range(self.n_rep_1):
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
                self.n_rep_1 * self.n_rep_2
            )
            modes_temp = xp.zeros_like(modes_E)
            modes_temp[self.orbitals_2, :] = modes_E
            modes.append(modes_temp)

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
            for i, n_rep in enumerate(self.n_reps)
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
            sigma_obc_k[k_i, k_j] = A_tot[0] @ x_ii @ A_tot[2]
            inj_k[k_i, k_j] = []

            for i, phi in enumerate(phi_surface):
                inj_k[k_i, k_j].append(-A_tot[0][i] @ phi)

            num_inj_k[k_i, k_j] = []
            for phi in phi_surface:
                num_inj_k[k_i, k_j].append(phi.shape[1])

            K_k[k_i, k_j] = phi_surface
            T_k[k_i, k_j] = -x_ii @ A_tot[2]

            if comm.rank == 0:
                print(f"    Computed OBC for k1={k_i}, k2={k_j}", flush=True)

        # Upscale self-energy and Bloch matrices
        sigma_obc = self._upscale_self_energy(
            sigma_obc_k, k_inner[0], k_inner[1], k_outer[0], k_outer[1]
        )
        T = self._upscale_self_energy(
            T_k, k_inner[0], k_inner[1], k_outer[0], k_outer[1]
        )

        # Upscale injection and Bloch injection matrices
        inj = self._upscale_injection_modes(inj_k, E)
        K = self._upscale_injection_modes(K_k, E)

        # Calculate total number of injected modes
        num_inj = np.zeros(E.shape[0], dtype=np.int32)
        for i_E in range(E.shape[0]):
            for value in num_inj_k.values():
                num_inj[i_E] += value[i_E]

        return sigma_obc, inj, num_inj, T, K

    def _compute_band_structure(self, device):

        if comm.rank == 0:
            print(
                f"    Computing band structure for contact {self.name}...", flush=True
            )
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

        n_points_BAND = 51

        k_transport = xp.linspace(
            -xp.pi,
            xp.pi,
            n_points_BAND,
        )

        self.band_structure = xp.zeros(
            (num_kpoints, n_points_BAND, self.n_orb_origin_cell), dtype=xp.float64
        )
        for i, k_perp in enumerate(kpoints):
            for j, k_par in enumerate(k_transport):

                k_vec = list(k_perp)
                k_vec.pop(self.direction)
                k_vec.insert(0, k_par)

                H_tot = sparse.csr_matrix(
                    (self.n_orb_origin_cell, self.n_orb_origin_cell),
                    dtype=xp.complex128,
                )
                S_tot = sparse.csr_matrix(
                    (self.n_orb_origin_cell, self.n_orb_origin_cell),
                    dtype=xp.complex128,
                )
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

                chol = xp.linalg.cholesky(S_tot)
                chol_inv = xp.linalg.inv(chol)

                H_ortho = chol_inv @ H_tot @ chol_inv.conj().T

                self.band_structure[i, j, :] = xp.linalg.eigvalsh(H_ortho)
