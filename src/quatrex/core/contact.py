from qttools import NDArray, obc, sparse, xp
from qttools.nevp import NEVP, Beyn, Full

from quatrex.core.quatrex_config import OBCConfig


class Contact:

    def _get_atoms_inside_cell(self, nx: int, ny: int, nz: int) -> NDArray:
        """Get the indices of atoms inside the repetition of the peridic cell.

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
            The indices of the atoms inside the periodic repetition.
        """

        # Shift the coordinates of the device atoms to the origin of the contact
        relative_coords = self.device.coords - self.origin

        # Compute the coefficients relative to the contact cell
        coeffs = relative_coords @ xp.linalg.inv(self.vectors)

        # Get the indices of the atoms inside the periodic repetition
        inside_mask = xp.nonzero(
            (coeffs[:, 0] >= nx)
            & (coeffs[:, 0] <= nx + 1)
            & (coeffs[:, 1] >= ny)
            & (coeffs[:, 1] <= ny + 1)
            & (coeffs[:, 2] >= nz)
            & (coeffs[:, 2] <= nz + 1)
        )[0]

        return inside_mask

    def _reorder_atoms(self, at_inside_rep: NDArray, a: int, b: int, c: int) -> NDArray:
        """Reorder the atoms inside the cell to match the order in the origin cell.

        Parameters
        ----------
        at_inside_rep : NDArray
            The indices of the atoms inside the periodic repetition.
        a : int
            The x-coordinate of the periodic repetition.
        b : int
            The y-coordinate of the periodic repetition.
        c : int
            The z-coordinate of the periodic repetition.

        Returns
        -------
        NDArray
            The reordered indices of the atoms inside the periodic repetition.
        """

        sorted = []
        # Tolerance for the distance check
        tolerance = 0.1

        a = int(a)  # Ensure a, b, c are integers
        b = int(b)
        c = int(c)
        # Shift the coordinates of the atoms inside the periodic repetition to match the origin cell
        list_vec = xp.array([a, b, c], dtype=xp.float64)
        coords_rep = self.device.coords[at_inside_rep, :] - self.vectors @ list_vec
        element_rep = self.device.atom_type[at_inside_rep]
        for at in self.at_origin_cell:
            # Find the atoms in the periodic repetition that are close to the atom in the origin cell and have the same element
            delta = coords_rep - self.device.coords[at, :]
            found = xp.nonzero(
                (xp.linalg.norm(delta, axis=1) < tolerance)
                & (self.device.atom_type[at] == element_rep)
            )[0]
            if found.size == 0:
                raise ValueError(
                    f"Error in contact {self.name}: "
                    f"Atom {at} not found in the periodic repetition at ({a}, {b}, {c})."
                )
            if found.size > 1:
                raise ValueError(
                    f"Error in contact {self.name}: "
                    f"Multiple atoms found in the periodic repetition at ({a}, {b}, {c}) "
                    f"matching atom {at} from the origin cell."
                )
            # Append the index of the found atom to the sorted list
            sorted.append(at_inside_rep[found[0]])

        return xp.array(sorted, dtype=int)

    def _get_periodic_number_transverse(self) -> int:
        """Count the number of periodic repetitions in one of the transverse directions."""

        def count_reps(axis: int, sign: int) -> int:
            """Count the number of periodic repetitions in a given transverse direction.
            Parameters
            ----------
            axis : int
                The axis along which to count the repetitions (0, 1, or 2).
            sign : int
                The sign of the direction to count the repetitions (1 for positive, -1 for negative).
            Returns
            -------
            int
                The number of periodic repetitions in the given direction.
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
                # If the number of atoms inside the periodic repetition does not match the origin cell, raise an error
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

        # 4 transverse directions: y+, y-, z+, z-
        # Count the number of periodic repetitions in each transverse direction
        n_rep_1_plus = count_reps(axis=self.transverse_axis[0], sign=1)
        n_rep_1_minus = count_reps(axis=self.transverse_axis[0], sign=-1)
        n_rep_2_plus = count_reps(axis=self.transverse_axis[1], sign=1)
        n_rep_2_minus = count_reps(axis=self.transverse_axis[1], sign=-1)

        # Store the number of periodic repetitions in the contact object and the coordinates of the origin cell
        self.origin_cell = xp.array((n_rep_1_minus, n_rep_2_minus))
        self.n_rep_1 = n_rep_1_plus + n_rep_1_minus + 1
        self.n_rep_2 = n_rep_2_plus + n_rep_2_minus + 1

    def _get_atoms_transverse_sorted(self, x: int) -> NDArray:
        """Get the indices of the atoms inside the periodic repetition (in transport direction) of the full contact cell.
        Parameters
        ----------
        x : int
            The index of the periodic repetition in the transport direction.
        Returns
        -------
        NDArray
            The indices of the atoms inside the periodic repetition.
        """

        atom_list = []

        # Start from the (0,0) cell and look for periodic repetitions in the transverse directions
        # The origin cell is defined by the user, so we start from -origin_cell (that is a tuple! (x,y))
        # and go up to (n_rep_1, n_rep_2)
        curr_cell = -self.origin_cell
        max_cell = curr_cell + xp.array([self.n_rep_1, self.n_rep_2])

        while curr_cell[0] < max_cell[0]:
            while curr_cell[1] < max_cell[1]:
                print(f"    Periodic repetition at: {curr_cell[0]}, {curr_cell[1]}")
                # Get the indices of the atoms inside the periodic repetition
                idx = [curr_cell[0], curr_cell[1]]
                idx.insert(self.direction, x)
                at_inside_rep = self._get_atoms_inside_cell(*idx)
                # Reorder the atoms to match the order in the origin cell
                c_list = [curr_cell[0], curr_cell[1]]
                c_list.insert(self.direction, x)
                at_inside_rep = self._reorder_atoms(
                    at_inside_rep, c_list[0], c_list[1], c_list[2]
                )
                print("    " + str(at_inside_rep))
                atom_list.append(at_inside_rep)
                curr_cell[1] += 1

            curr_cell[0] += 1
            curr_cell[1] = -self.origin_cell[1]

        return xp.concatenate(atom_list, dtype=int)

    def _get_orbitals(self, atoms: NDArray) -> NDArray:
        """Get the orbital indices corresponding to the atoms
        Parameters
        ----------
        atoms : NDArray
            The indices of the atoms.
        Returns
        -------
        NDArray
            The indices of the orbitals corresponding to the atoms.
        """

        vec_orb = xp.array([], dtype=xp.int32)
        for i in range(atoms.shape[0]):
            # NEED TO MOVE THE INDEX ON THE CPU
            # I USED A QUICK WORKAROUND FOR NOW
            index = int(atoms[i].get() if hasattr(atoms[i], "get") else atoms[i])
            # Starting orbitals for the current atom
            k1 = int(
                self.device.orbitals_vec[index].get()
                if hasattr(self.device.orbitals_vec[index], "get")
                else self.device.orbitals_vec[index]
            )
            # Ending orbitals for the current atom (can be computed from orbitals_vec)
            k2 = int(
                self.device.orbitals_vec[index + 1].get()
                if hasattr(self.device.orbitals_vec[index + 1], "get")
                else self.device.orbitals_vec[index + 1]
            )
            vec_orb = xp.concatenate((vec_orb, xp.arange(k1, k2)))

        return vec_orb

    def _get_circumference_coordinates(self, radius: int) -> list:
        """Get coordinates only on the circumference (perimeter) of the grid, given a radius.
        Parameters
        ----------
        radius : int
            The radius of the circumference.
        Returns
        -------
        list
            A list of tuples representing the coordinates on the circumference.
        """
        coordinates = []

        for y in range(-radius, radius + 1):
            for z in range(-radius, radius + 1):
                # Check if point is on the boundary (max distance = radius)
                if max(abs(y), abs(z)) == radius:
                    coordinates.append((y, z))

        return coordinates

    def _get_matrix(self, x: int) -> None:
        """Get the hamiltonian matrix for the transverse contact cell at a given distance x in transport direction.
        Parameters
        ----------
        x : int
            The index of the periodic repetition in the transport direction.
        """

        # The hamiltonian and overlap matrices for a given transverse slice are obtained around the origin cell
        # increasing radius until no more hamiltonian or overlap is found.

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
                # The coupling is defined in the in the device hamiltonian at (H_1, H_2)
                # (it can be in any hopping hamiltonian). Here we compute in which hopping hamiltonian it is.
                H_1 = int((a + 0.0001) / self.n_rep_1)
                if a < 0:
                    H_1 -= 1
                H_2 = int((b + 0.0001) / self.n_rep_2)
                if b < 0:
                    H_2 -= 1

                if self.device.gamma_only and (self.n_rep_1 > 1 or self.n_rep_2 > 1):
                    H_1 = 0
                    H_2 = 0
                    if (2 * radius + 1) > self.n_rep_1 or (
                        2 * radius + 1
                    ) > self.n_rep_2:
                        raise ValueError(
                            f"Error in contact {self.name}: \n"
                            f"I cannot obtain the UC matrices from the Gamma-point device matrix, probably because the basis decay is not enough.\n"
                            f"Possible solutions:\n"
                            f"  - Increase the UC to include the entire cross-section (1x1 contact UC)\n"
                            f"  - Provide all the hopping Hamiltonians in the device, not only the Gamma point."
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

                # Now get the hamiltonian and overlap matrices for the current coordinates
                if ham_tu in self.device.hamiltonian:
                    ham_read = self.device.hamiltonian[ham_tu][self.orb_origin_cell, :][
                        :, orb_coup
                    ]
                    if ham_read.nnz != 0:
                        self.UC_hamiltonian[(x, coords[0], coords[1])] = ham_read
                        found = True
                if ham_tu in self.device.overlap:
                    overlap_read = self.device.overlap[ham_tu][self.orb_origin_cell, :][
                        :, orb_coup
                    ]
                    if overlap_read.nnz != 0:
                        self.UC_overlap[(x, coords[0], coords[1])] = overlap_read
                        found = True

            if not found:
                print(f"    Maximum radius: {radius-1}")
                if self.n_rep_1 == 1 and self.n_rep_2 == 1 and (radius - 1) > 0:
                    raise ValueError("1x1 UC but more than 1x1 coupling!")
                break

            radius += 1

    def _residual_coupling(self, orbitals: list) -> bool:
        """Check if there is residual coupling between the orbitals in the contact and the full device.
        Parameters
        ----------
        orbitals : list
            A list of orbital indices for which to check the residual coupling.
        Returns
        -------
        bool
            True if there is residual coupling, False otherwise.
        """

        tot_orb = xp.arange(self.device.hamiltonian[(0, 0, 0)].shape[0])
        tot_orb = tot_orb[~xp.isin(tot_orb, xp.concatenate(orbitals))]

        return self.device.hamiltonian[(0, 0, 0)][orbitals[0], :][:, tot_orb].nnz

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
                "Sancho-rubio OBC algorithm does not work with QTBM, please use spectral OBC solver."
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

    def get_10(self, M):

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

    def __init__(
        self, name, device, vectors: NDArray, origin: NDArray, direction: int, config
    ):
        """Initialize the contact object.
        Parameters
        ----------
        name : str
            The name of the contact.
        device : Device
            The device to which the contact is attached.
        vectors : NDArray
            The vectors defining the unit cell of the contact.
        origin : NDArray
            The origin of the contact cell in the device coordinates.
        direction : int
            The direction of the contact (0, 1, or 2).
        config : QuatrexConfig
            The configuration object containing the OBC settings.
        """

        self.name = name
        self.device = device
        self.vectors = vectors
        self.origin = origin
        self.direction = direction
        self.transverse_axis = [0, 1, 2]
        self.transverse_axis.remove(direction)

        self.obc = self._configure_obc(getattr(config, "electron").obc)

        self.UC_hamiltonian = {}
        self.UC_overlap = {}

        self.S10_contact = {}
        self.H10_contact = {}

        if len(self.transverse_axis) != 2:
            raise ValueError("Direction must be one of the three axes (0, 1, or 2).")

        relative_coords = self.device.coords - self.origin
        self.coeffs = relative_coords @ xp.linalg.inv(self.vectors)

        # Get the atoms inside the origin cell (defined by the user)
        self.at_origin_cell = self._get_atoms_inside_cell(0, 0, 0)
        self.orb_origin_cell = self._get_orbitals(self.at_origin_cell)
        self.n_at_origin_cell = self.at_origin_cell.shape[0]
        self.n_orb_origin_cell = self.orb_origin_cell.shape[0]

        # Check how many periodic repetitions are in the transverse directions
        self._get_periodic_number_transverse()
        print(f"Contact {self.name}:")
        print(
            f"    Number of atoms inside the origin cell: {self.at_origin_cell.shape[0]}"
        )
        print(
            f"    Number of periodic repetitions in the transverse directions: {self.n_rep_1} x {self.n_rep_2}"
        )
        print(f"    Atoms in the origin cell: {self.at_origin_cell}")

        # TODO Check if the contact transverse UC vectors are in the same direction as the device vectors

        # Get the atoms (and orbtals) in the first contact cell
        self.atoms = []
        self.orbitals = []
        self.atoms.append(self._get_atoms_transverse_sorted(0))
        self.orbitals.append(self._get_orbitals(self.atoms[0]))

        # Get the hamiltonian and overlap matrices for the first contact cell
        self._get_matrix(0)

        x = 1
        # Iterate over the transport direction until there is no more residual coupling in the contact cell
        while self._residual_coupling(self.orbitals) > 0:
            print(f"    Residual coupling={self._residual_coupling(self.orbitals)}")

            # Get atoms, orbitals and matrices for the next contact cell
            self.atoms.append(self._get_atoms_transverse_sorted(x))
            self.orbitals.append(self._get_orbitals(self.atoms[x]))
            self._get_matrix(x)

            x = x + 1

        print(f"    Maximum number of transverse repetitions: {x-1}")

        # In case of multiple contact cells in the transport direction,
        # and in case of multiple unit cells in the transverse direction,
        # we will obtain a the self energy sorted in different way.
        # atoms_2 will be used to sort the atoms in a consistent way (sorted over the transport direction).
        self.atoms_2 = []
        self.orbitals_2 = []

        for i in range(self.n_rep_1):
            for j in range(self.n_rep_2):
                for k in range(x - 1):
                    # start = i*self.n_at_origin_cell*self.n_rep_2 + j*self.n_at_origin_cell
                    # end = start + self.n_at_origin_cell
                    # self.atoms_2.append(self.atoms[k][start:end])
                    start = self.n_orb_origin_cell * (
                        k * self.n_rep_1 * self.n_rep_2 + i * self.n_rep_2 + j
                    )
                    end = start + self.n_orb_origin_cell
                    self.orbitals_2.append(xp.arange(start, end))

        # Concatenate the atoms_2 list and get the orbitals order for the unsorted cell
        # self.atoms_2 = xp.concatenate(self.atoms_2, dtype=int)
        # self.orbitals_2 = self._get_orbitals(self.atoms_2)
        self.orbitals_2 = xp.concatenate(self.orbitals_2, dtype=int)

        self.orbitals_contact = xp.concatenate(self.orbitals[:-1])[None, :]

        self.transverse_rep = x - 1

    def _get_list_mat_phase(self, k1: float, k2: float) -> NDArray:
        """Get the list of hopping matrices in transport direction with the corresponding phase factors (for the transverse direction).
        Parameters
        ----------
        k1 : float
            The k1 value for the phase factor.
        k2 : float
            The k2 value for the phase factor.
        Returns
        -------
        tuple
            A tuple containing two lists: the list of hamiltonian matrices and the list of overlap matrices.
        """
        # Size of the hamiltonian and overlap matrices
        n = self.UC_hamiltonian[(0, 0, 0)].shape[0]

        # Initialize the lists of hamiltonian and overlap matrices
        H_coup = []
        S_coup = []

        # Create empty matrices for each repetion in the transport direction
        for ii in range(self.transverse_rep + 1):
            H_coup.append(sparse.csr_matrix((n, n), dtype=xp.complex128))
            S_coup.append(sparse.csr_matrix((n, n), dtype=xp.complex128))

        # Fill the matrices with the hamiltonian and overlap matrices from the unit cell
        # and apply the phase factors for the transverse direction
        # (H^0, H^1, H^2)
        for index, ham in self.UC_hamiltonian.items():
            H_coup[index[0]] += ham * xp.exp(1j * (k1 * index[1] + k2 * index[2]))
        for index, overlap in self.UC_overlap.items():
            S_coup[index[0]] += overlap * xp.exp(1j * (k1 * index[1] + k2 * index[2]))

        # Add the conjugate transpose, for example (H^-2, H^-1, H^0, H^1, H^2)
        for ii in range(self.transverse_rep):
            H_coup.insert(0, H_coup[ii * 2 + 1].conj().T)
            S_coup.insert(0, S_coup[ii * 2 + 1].conj().T)

        # Augment with emtpy matrices (needed for the OBC solver) (H^-2, H^-1, H^0, H^1, H^2, H^3)
        for ii in range(self.transverse_rep - 1):
            H_coup.append(sparse.csr_matrix((n, n), dtype=xp.complex128))
            S_coup.append(sparse.csr_matrix((n, n), dtype=xp.complex128))

        return H_coup, S_coup

    def _construct_circulant_matrix(self, list_mat: list, phase: float) -> NDArray:
        """Construct a circulant matrix from a list of matrices with a given phase factor.
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
        """Upscale self-energy matrices using circulant matrix construction.
        Parameters
        ----------
        SE_k : dict
            A dictionary containing self-energy matrices indexed by (k1, k2) tuples.
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
        # Reorder the self-energy matrix to match the orbitals of the contact
        # SE_1[:, self.orbitals_2[None,:].T, self.orbitals_2[None,:]] = SE_1
        SE_temp = xp.zeros_like(SE_1)
        SE_temp[:, self.orbitals_2[None, :].T, self.orbitals_2[None, :]] = SE_1

        return SE_temp

    def _upscale_injection_modes(self, modes_k: dict, E: NDArray) -> NDArray:
        """Upscale injection vectors using phase multiplication and concatenation.
        Parameters
        ----------
        modes_k : dict
            A dictionary containing injection vectors indexed by (k1, k2) tuples.
        E : NDArray
            The batch of energies for which to compute the total inhection vectors.
        Returns
        -------
        NDArray
            The upscaled and concatenated injection vectors.
        """
        # Upscale the k-space modes
        # Iterate over the wavevector keys
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

            # SORTING AND NORMALIZATION IS ONLY FOR DEBUG (IT IS NOT REALLY NEEDED)
            # modes[-1] /= xp.exp(1j*xp.angle(modes[-1][0,:]))
            # sort_indices = xp.argsort(modes[-1][0, :])
            # modes[-1] = modes[-1][:, sort_indices]

        return modes

    def compute_boundary(self, ka: float, kb: float, kc: float, E) -> None:
        """Compute the open boundary conditions for the contact at a given k1 and k2 and energy Batch E.
        Parameters
        ----------
        k1 : float
            K poiint in the first transverse direction.
        k2 : float
            K point in the second transverse direction.
        E : NDArray
            A batch of energies for which to compute the open boundary conditions.
        Returns
        -------
        tuple
            A tuple containing the self-energy, injection modes, number of injected modes, Bloch matrix, and Bloch injection matrix.
        """

        # FOR NOW, ONLY E BATCHING IS SUPPORTED

        kl = [ka, kb, kc]
        if kl[self.direction] != 0:
            raise ValueError(
                f"Error in contact {self.name}: "
                f"You can't compute the OBC for a non-zero k-point in the transport direction ({self.direction}). "
            )
        # Remove the k-point in the transport direction
        kl.pop(self.direction)
        k1 = kl[0]
        k2 = kl[1]

        # Create the k-space list needed to upscale the self-energy and injection modes in the transverse directions
        k_1_list = (
            xp.linspace(0, xp.pi * 2, self.n_rep_1, endpoint=False) + k1 / self.n_rep_1
        )
        k_2_list = (
            xp.linspace(0, xp.pi * 2, self.n_rep_2, endpoint=False) + k2 / self.n_rep_2
        )

        SE_k = {}
        inj_k = {}
        num_inj_k = {}
        K_k = {}
        T_k = {}

        for ki in k_1_list:
            for kj in k_2_list:
                # Construct the hamiltonian and overlap matrices for the given ki and kj
                H_list, S_list = self._get_list_mat_phase(ki, kj)
                H_tot = sparse.hstack(H_list, format="csr")
                S_tot = sparse.hstack(S_list, format="csr")

                # Create the toeplitz structure for the hamiltonian and overlap matrices (in transport direction)
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

                # Solve the OBC for the given ki and kj and store the results in dictionaries
                self.obc.block_sections = self.transverse_rep
                (
                    _,
                    SE_k[(ki.item(), kj.item())],
                    inj_k[(ki.item(), kj.item())],
                    num_inj_k[(ki.item(), kj.item())],
                    K_k[(ki.item(), kj.item())],
                    T_k[(ki.item(), kj.item())],
                ) = self.obc(
                    A_tot[1],
                    A_tot[2],
                    A_tot[0],
                    "left",
                    return_injected=True,
                )
                print(f"    Computed OBC for k1={ki}, k2={kj}")

        # Upscale self-energy and Bloch matrices
        SE_1 = self._upscale_self_energy(SE_k, k_1_list, k_2_list, k1, k2)
        T_1 = self._upscale_self_energy(T_k, k_1_list, k_2_list, k1, k2)

        # Upscale injection and Bloch injection matrices
        inj = self._upscale_injection_modes(inj_k, E)
        K = self._upscale_injection_modes(K_k, E)

        # Calculate total number of injected modes
        inj_n = xp.zeros(E.shape[0], dtype=xp.int32)
        for i_E in range(E.shape[0]):
            for key, value in num_inj_k.items():
                inj_n[i_E] += value[i_E]

        return SE_1, inj, inj_n, T_1, K
