# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the contact class."""

import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.boundary_conditions import obc
from qttools.comm import comm
from qttools.nevp import NEVP, Beyn, Full
from qttools.profiling import Profiler
from quatrex.core.config import ContactConfig, NEVPConfig, OBCConfig
from quatrex.device.contact_discovery import real_space_discovery, simplified_discovery

profiler = Profiler()


@dataclass
class OBCResult:
    """Data class to hold the results of the contact's OBC calculation.

    Attributes
    ----------
    injection : NDArray
        The injection vectors for the contact at the given k-points and
        energies.
    b_injected : NDArray
        The injection vectors before applying the contact coupling.
    sigma_obc_k : dict
        A dictionary containing the computed self-energy for each
        transverse k-point, indexed by (ky, kz) tuples. Only returned if
        `return_modes_only` is False.
    bloch_k : dict
        A dictionary containing the computed Bloch injection matrices
        for each transverse k-point, indexed by (ky, kz) tuples. Only
        returned if `return_modes_only` is False.
    reflection : NDArray
        The reflection vectors for the contact at the given k-points and
        energies. Only returned if `return_modes_only` is True.
    phi_reflected : NDArray
        The reflected modes for the contact at the given k-points and
        energies. Only returned if `return_modes_only` is True.
    eig_reflected : NDArray
        The reflected eigenvalues for the contact at the given k-points
        and energies. Only returned if `return_modes_only` is True.
    phi_inv_reflected : NDArray
        The pseudoinverse of the reflected modes for the contact at the
        given k-points and energies. Only returned if
        `return_modes_only` is True.

    """

    # These two are always returned.
    injection: NDArray
    b_injected: NDArray

    # These k-dependent quantities are returned when not using low-rank.
    sigma_obc_k: dict[str, NDArray] | None = None
    bloch_k: dict[str, NDArray] | None = None

    # This is returned if we are in the context of low-rank OBCs.
    reflection: NDArray | None = None
    phi_reflected: NDArray | None = None
    eig_reflected: NDArray | None = None
    phi_inv_reflected: NDArray | None = None

    def __getitem__(self, key: int) -> "OBCResult":
        """Allows accessing a single energy index from the OBCResult."""
        if not isinstance(key, int):
            raise TypeError("OBCResult can only be indexed with an integer.")

        # Go through all attributes and index them by the given key if
        # they are not None.
        kwargs = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is None:
                continue
            if isinstance(value, dict):
                kwargs[field] = {k: v[key] for k, v in value.items()}
            else:
                kwargs[field] = value[key]

        return OBCResult(**kwargs)


def order_vector(
    vector: NDArray,
    order: str | NDArray | None,
):
    """Reorders the elements of the given vector according to the
    specified order.

    Parameters
    ----------
    vector : NDArray
        The vector to reorder.
    order : str | NDArray | None
        The order in which to reorder the elements. The only supported
        string is "reverse", which reverses the order of the elements.

    Returns
    -------
    NDArray
        The reordered vector.

    """

    if isinstance(order, str) and order not in ["reverse"]:
        raise ValueError(f"Invalid order string: {order}. Must be 'reverse' or None.")
    if isinstance(order, xp.ndarray) and order.ndim != 1:
        raise ValueError(f"Order array must be 1-dimensional, got shape {order.shape}.")

    if order is None:
        return vector
    if order == "reverse":
        return xp.flip(vector, axis=-1)
    return vector[..., order]


def order_block(
    block: NDArray,
    order: str | NDArray | None,
) -> NDArray:
    """Reorders the blocks of the given matrix according to the
    specified order.

    Parameters
    ----------
    block : NDArray
        The matrix block to reorder.
    order : str | NDArray | None
        The order in which to reorder the blocks. The only supported
        string is "reverse", which reverses the order of the blocks.

    Returns
    -------
    NDArray
        The reordered matrix block.

    """

    if isinstance(order, str) and order not in ["reverse"]:
        raise ValueError(f"Invalid order string: {order}. Must be 'reverse' or None.")
    if isinstance(order, xp.ndarray) and order.ndim != 1:
        raise ValueError(f"Order array must be 1-dimensional, got shape {order.shape}.")

    if order is None:
        return block
    if order == "reverse":
        return xp.flip(block, axis=(-2, -1))
    return block[..., :, order][..., order, :]


def get_inverse_order(
    order: str | NDArray | None,
) -> str | NDArray | None:
    """Computes the inverse of the given order.

    Parameters
    ----------
    order : str | NDArray | None

    Returns
    -------
    str | NDArray | None
        The inverse order, or None if the input order is None.

    """
    # TODO: This should be only called once inside
    # the contact.

    if isinstance(order, str) and order not in ["reverse"]:
        raise ValueError(f"Invalid order string: {order}. Must be 'reverse' or None.")
    if isinstance(order, xp.ndarray) and order.ndim != 1:
        raise ValueError(f"Order array must be 1-dimensional, got shape {order.shape}.")

    if order is None:
        return None
    if order == "reverse":
        return "reverse"
    return xp.argsort(order)


class Contact:
    """Class representing a contact for QTBM calculations.

    Parameters
    ----------
    device : Device
        The device object to which this contact is attached. Contains
        the Hamiltonian, overlap matrices, and atomic structure
        information.
    contact_config : ContactConfig
        The configuration object containing the contact settings such as
        lattice vectors, origin, transport direction, and Fermi level
        information.

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
    transport_direction : int
        Transport direction index (0, 1, or 2).
    transverse_axes : list[int]
        Indices of the two transverse directions.
    obc : obc.Spectral
        Configured open boundary condition solver.
    unit_cell_hamiltonian : dict
        Unit cell Hamiltonian matrices indexed by (i, j, k) tuples.
    unit_cell_overlap : dict
        Unit cell overlap matrices indexed by (i, j, k) tuples.
    unit_cell_orbital_indices : dict
        Dict of orbital indices for each contact cell indexed by (i, j, k) tuples.
    transverse_repetition_grid: NDArray
        Number of periodic repetitions in the two transverse directions.
    num_transport_cells : int
        Number of repetitions needed in transport direction for
        convergence.

    """

    def __init__(self, device, contact_config: ContactConfig):
        """Initializes the contact object."""

        if contact_config.transport_direction not in ["a", "b", "c"]:
            raise ValueError("Direction must be one of 'a', 'b', or 'c'.")

        self.device = device
        self.name = contact_config.name
        self.transport_direction = "abc".index(contact_config.transport_direction)

        if contact_config.contact_finder_method == "real_space":
            self.unit_cell_orbital_indices, repetition_grid, self.origin_key = (
                real_space_discovery(
                    hamiltonian=device.hamiltonians[0, 0, 0],
                    atomic_species=device.atomic_species,
                    atom_coordinates=device.atom_coordinates,
                    orbital_offsets=device.orbital_offsets,
                    contact_config=contact_config,
                )
            )
        elif contact_config.contact_finder_method in ["from_unit", "slice"]:
            self.unit_cell_orbital_indices, repetition_grid, self.origin_key = (
                simplified_discovery(
                    contact_name=contact_config.name,
                    num_orbitals=len(device.orbital_coordinates),
                    device_config=device.config.device,
                    contact_config=contact_config,
                )
            )
        else:
            raise NotImplementedError(
                f"Contact finder method '{contact_config.contact_finder_method}' not implemented."
            )

        self.transport_repetitions = repetition_grid[self.transport_direction]
        self.transverse_repetition_grid = (
            repetition_grid[: self.transport_direction]
            + repetition_grid[self.transport_direction + 1 :]
        )
        self.origin_num_orbitals = len(self.unit_cell_orbital_indices[self.origin_key])
        self.origin_orbital_indices = self.unit_cell_orbital_indices[self.origin_key]

        if comm.rank == 0:
            print(
                f"    Number of repetitions in transport direction: {self.transport_repetitions}",
                flush=True,
            )

        ny, nz = self.transverse_repetition_grid

        # Orbitals for contact (where to apply the OBC)
        # Sorted first in transport direction, then in transverse directions
        self.orbital_indices = np.concatenate(
            [
                self.unit_cell_orbital_indices[i, j, k]
                for j, k, i in np.ndindex(ny, nz, self.transport_repetitions)
            ]
        )
        # When getting the coupling matrix (01) for spill over,
        # it is more efficient to have it sorted first in transverse, then in transport
        # The orbital list is then different.
        # We keep it separated over slice over transport direction.
        self.orbital_indices_per_layer = [
            np.concatenate(
                [self.unit_cell_orbital_indices[i, j, k] for j, k in np.ndindex(ny, nz)]
            )
            for i in range(self.transport_repetitions + 1)
        ]

        # We then need to sort the 10 matrix to have the same ordering as the contact OBCs
        self.transverse_to_transport_indices = np.concatenate(
            [
                np.arange(self.origin_num_orbitals)
                + i * self.origin_num_orbitals
                + k * self.origin_num_orbitals * ny * nz
                for i in range(ny * nz)
                for k in range(self.transport_repetitions)
            ],
            dtype=int,
        )[None, :]

        self.obc_solver = self._configure_obc(
            device.config.electron.obc, device.config.compute.nevp
        )

        self.fermi_level = contact_config.fermi_level
        self.mid_gap_energy = contact_config.mid_gap_energy
        self.delta_fermi_level_conduction_band = (
            contact_config.delta_fermi_level_conduction_band
        )
        self.voltage = contact_config.voltage
        self.temperature = contact_config.temperature

        if comm.rank == 0:
            print(f"    Fermi level: {self.fermi_level} eV", flush=True)
            print(f"    Mid-gap energy: {self.mid_gap_energy} eV", flush=True)
            print(
                f"    Delta Fermi level to conduction band: {self.delta_fermi_level_conduction_band} eV",
                flush=True,
            )
            print(f"    Voltage: {self.voltage} V", flush=True)
            print(f"    Temperature: {self.temperature} K", flush=True)

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

        if obc_config.algorithm == "spectral":
            nevp = self._configure_nevp(obc_config, nevp_config)
            obc_solver = obc.Spectral(
                nevp=nevp,
                block_sections=self.transport_repetitions,  # WARNING: overrides config
                min_decay=obc_config.min_decay,
                max_decay=obc_config.max_decay,
                num_ref_iterations=obc_config.num_ref_iterations,
                min_propagation=obc_config.min_propagation,
                residual_tolerance=obc_config.residual_tolerance,
                residual_normalization=obc_config.residual_normalization,
                eta_decay=obc_config.eta_decay,
                warning_threshold=obc_config.warning_threshold,
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
            return Full(
                eig_compute_location=nevp_config.eig_compute_location,
                use_pinned_memory=nevp_config.use_pinned_memory,
                reduce=nevp_config.reduce_sparsity,
            )

        raise NotImplementedError(
            f"NEVP solver '{obc_config.nevp_solver}' not implemented."
        )

    def get_coupling_matrix(
        self, M: sparse.spmatrix, transpose: bool = False
    ) -> NDArray:
        """Extracts coupling matrix between device and contact.

        This method constructs the matrix that couples the device region
        to the contact.

        Example:
            Given a contact layers |0 1 2 3|,
            the resulting coupling matrix is
            |3 2 1|
            |0 3 2|
            |0 0 3|


        Parameters
        ----------
        M : sparse.spmatrix
            The matrix (Hamiltonian or overlap) from which to extract
            coupling elements. Should have dimensions
            (n_device_orbitals, n_device_orbitals).
        transpose : bool, optional
            If True, the method extracts the transpose of the coupling
            matrix, by default False.

        Returns
        -------
        NDArray
            Dense matrix representing the coupling between device and
            contact. The matrix has the block structure needed for QTBM
            boundary conditions, with dimensions determined by the
            contact's transverse repetitions.

        """

        n = self.orbital_indices_per_layer[0].shape[0]

        indices_zero = self.orbital_indices_per_layer[0]

        # Slice block column of the matrix
        # Thus, no conjugation and transpose is needed
        if not transpose:
            layers = [
                M[indices, :][:, indices_zero]
                for indices in self.orbital_indices_per_layer[1:]
            ]
        else:
            layers = [
                M[:, indices][indices_zero, :].T.conj()
                for indices in self.orbital_indices_per_layer[1:]
            ]

        # NOTE: Stacking sparse matrix is slow
        coupling_matrix = []
        zero = sparse.csr_matrix((n, n), dtype=xp.complex128)
        # Assemble column by column
        for shift in range(self.transport_repetitions):
            layer = layers[shift:] + [zero] * shift
            coupling_matrix.append(sparse.vstack(layer, format="csr"))

        coupling_matrix = sparse.hstack(coupling_matrix[::-1], format="csr")

        indices = self.transverse_to_transport_indices
        return coupling_matrix[indices.T, indices]

    def _construct_contact_matrix(self, UC_matrix: list):
        """Constructs the full contact matrix for the contact at given
        transverse k-points.

        Parameters
        ----------
        UC_matrix : list

        Returns
        -------
        sparse.spmatrix
            The constructed contact matrix in sparse format.

        """

        n = UC_matrix[0].shape[0]
        num_cells = self.transport_repetitions
        zero = sparse.csr_matrix((n, n), dtype=xp.complex128)

        uc_left = [h.conj().T for h in UC_matrix[1:][::-1]]

        # Pad with zeros for the OBCs
        padding = [zero] * (num_cells - 1)
        first_row_blocks = uc_left + UC_matrix + padding

        contact_matrix = []
        for ii in range(num_cells):
            contact_matrix.append(sparse.hstack(first_row_blocks, format="csr"))
            first_row_blocks.insert(0, first_row_blocks.pop())

        contact_matrix = sparse.vstack(contact_matrix, format="csr")

        return contact_matrix

    def _concatenate_eig(self, eig_k: dict, num_energies: int) -> NDArray:
        """Concatenates eigenvectors for different k-points.

        Parameters
        ----------
        eig_k : dict
            A dictionary containing eigenvalues indexed by (k1, k2)
            tuples.
        num_energies : int
            The number of energies for which to compute the total
            eigenvectors.

        Returns
        -------
        NDArray
            The concatenated eigenvectors for all k-points.

        """

        eig = [
            xp.concatenate([eig_k[key][i_E] for key in eig_k.keys()])
            for i_E in range(num_energies)
        ]

        return eig

    def _upscale_pseudo_inverse(self, pseudo_k: dict, num_energies: int) -> NDArray:
        """Upscales injection vectors.

        Parameters
        ----------
        pseudo_k : dict
            A dictionary containing pseudo-inverse vectors indexed by
            (k1, k2) tuples.
        num_energies : int
            The number of energies for which to compute the total
            pseudo-inverse vectors.

        Returns
        -------
        NDArray
            The upscaled and concatenated pseudo-inverse vectors.

        """
        # Upscale the k-space modes Iterate over the wavevector keys
        ny, nz = self.transverse_repetition_grid
        norm = xp.sqrt(ny * nz)

        modes_upscaled = defaultdict(list)
        for key, value in pseudo_k.items():

            assert (
                len(value) == num_energies
            ), "Mismatch in number of energies when upscaling pseudo-inverse vectors."

            # Iterate over the energies in the batch
            for i_E in range(num_energies):

                # Upscale in 2nd direction first
                I_2 = xp.concatenate(
                    [
                        pseudo_k[key][i_E] * xp.exp(-1j * (key[1] * j))
                        for j in range(nz)
                    ],
                    axis=1,
                )

                # Upscale in 1st direction
                I_1 = xp.concatenate(
                    [I_2 * xp.exp(-1j * (key[0] * i)) for i in range(ny)], axis=1
                )

                modes_upscaled[key].append(I_1)

        # Concatenate all the wavevector (transverse)
        modes = [
            xp.concatenate(
                [value[i_E] for value in modes_upscaled.values()],
                axis=0,
            )
            / norm
            for i_E in range(num_energies)
        ]

        return modes

    def _upscale_injection_modes(self, modes_k: dict, num_energies: int) -> NDArray:
        """Upscales injection vectors.

        Parameters
        ----------
        modes_k : dict
            A dictionary containing injection vectors indexed by (k1,
            k2) tuples.
        num_energies : int
            The number of energies for which to compute the total
            injection vectors.

        Returns
        -------
        NDArray
            The upscaled and concatenated injection vectors.

        """
        # Upscale the k-space modes Iterate over the wavevector keys
        ny, nz = self.transverse_repetition_grid
        norm = xp.sqrt(ny * nz)

        modes_upscaled = defaultdict(list)
        for key, value in modes_k.items():

            assert (
                len(value) == num_energies
            ), "Mismatch in number of energies when upscaling injection modes."

            # Iterate over the energies in the batch
            for i_E in range(num_energies):

                # Upscale in 2nd direction first
                I_2 = xp.concatenate(
                    [modes_k[key][i_E] * xp.exp(1j * (key[1] * j)) for j in range(nz)],
                    axis=0,
                )

                # Upscale in 1st direction
                I_1 = xp.concatenate(
                    [I_2 * xp.exp(1j * (key[0] * i)) for i in range(ny)], axis=0
                )

                modes_upscaled[key].append(I_1)

        # Concatenate all the wavevector (transverse)
        modes = [
            xp.concatenate(
                [value[i_E] for value in modes_upscaled.values()],
                axis=1,
            )
            / norm
            for i_E in range(num_energies)
        ]

        return modes

    def slice_matrix(
        self,
        M: sparse.spmatrix,
        upper: bool = False,
    ):
        """Slices the given matrix into a dictionary of submatrices
        corresponding to the unit cell orbital indices.

        Parameters
        ----------
        M : sparse.spmatrix
            The matrix to slice.
        upper : bool, optional
            Whether M is only upper triangular.

        Returns
        -------
        dict
            A dictionary mapping (i, j, k) tuples to the sliced
            submatrices corresponding to the unit cell orbital indices.

        """

        grid = (self.transport_repetitions + 1,) + self.transverse_repetition_grid

        M_slice = {}
        M_origin = M[self.origin_orbital_indices, :]

        if upper:
            M_origin += M[:, self.origin_orbital_indices].T.conj()
            M_origin[:, self.origin_orbital_indices] -= (
                sparse.diags(M_origin[:, self.origin_orbital_indices].diagonal()) / 2
            )

        for index in np.ndindex(*grid):
            M_slice[index] = M_origin[:, self.unit_cell_orbital_indices[index]]

        return M_slice

    def get_contact_blocks(
        self,
        matrices: dict,
        kpoint: NDArray,
        upper: bool = False,
    ) -> dict:
        """Constructs the contact blocks for the given k-point.

        Parameters
        ----------
        matrices : dict
            A dictionary of matrices (Hamiltonian or overlap) indexed by
            the spatial index.
        kpoint : NDArray
            The k-point for which to construct the contact blocks.
        upper : bool, optional
            Whether M is only upper triangular.

        Returns
        -------
        dict
            A dictionary mapping (i, j, k) tuples to the contact blocks
            corresponding to the unit cell orbital indices.

        """
        M_origin = None

        # NOTE: Needs to slice and multiply the phase at once using
        # `slice_matrix` would be wrong with `upper=True` since not each
        # k-point hamiltonian is hermitian, but only the sum over all
        # k-points is hermitian.

        # Assemble the contact layer for the full summed k-point matrix
        for r, matrix in matrices.items():
            phase = np.exp(2j * np.pi * np.dot(kpoint, r))
            term = phase * matrix[self.origin_orbital_indices, :]
            M_origin = term if M_origin is None else M_origin + term

        if upper:
            # NOTE: We could potentially optimize this by only slicing
            # the origin orbital indices since the rest is zero due to
            # being upper triangular.

            M_col = None
            for r, matrix in matrices.items():
                phase = np.exp(2j * np.pi * np.dot(kpoint, r))
                term = phase * matrix[:, self.origin_orbital_indices]
                M_col = term if M_col is None else M_col + term

            M_origin = M_origin + M_col.T.conj()
            M_origin[:, self.origin_orbital_indices] -= (
                sparse.diags(M_origin[:, self.origin_orbital_indices].diagonal()) / 2
            )

        m_xx = {}
        grid = (self.transport_repetitions + 1,) + self.transverse_repetition_grid
        for index in np.ndindex(*grid):
            m_xx[index] = M_origin[:, self.unit_cell_orbital_indices[index]]

        return m_xx

    @profiler.profile("Contact: Compute Boundary", level="default")
    def compute_boundary(
        self,
        M: sparse.spmatrix,
        upper_M: bool,
        k_outer: tuple[float, float, float],
        return_modes_only: bool = False,
    ) -> OBCResult:
        """Computes OBC for the contact at given k-points and energies.

        Parameters
        ----------
        M : sparse.spmatrix
            The system matrix from which to extract coupling elements.
            It should have dimensions (n_device_orbitals, n_device_orbitals).
        upper_M : bool
            Whether to use the upper triangle of the system matrix.
        k_outer : tuple[float, float, float]
            The k-point in the transport direction.
        return_modes_only : bool, optional
            Whether to return only the injection and surface modes
            without computing the full self-energy and Bloch matrices.

        Returns
        -------
        ContactOBCResult
            An object containing the computed OBC results, including
            injection modes, self-energy, and Bloch modes as applicable.

        """
        if k_outer[self.transport_direction] != 0:
            raise ValueError(
                f"Error in contact {self.name}: "
                f"You can't compute the OBC for a non-zero k-point in the transport direction ({self.transport_direction}). "
            )
        # Remove the k-point in the transport direction
        k_outer.pop(self.transport_direction)

        num_energies = 1

        ny, nz = self.transverse_repetition_grid
        M_slice = self.slice_matrix(
            M=M,
            upper=upper_M,
        )

        # Create the k-space list needed to upscale the self-energy and
        # injection modes in the transverse directions
        k_inner = [
            np.linspace(0, np.pi * 2, n_rep, endpoint=False) + k_outer[i] / n_rep
            for i, n_rep in enumerate(self.transverse_repetition_grid)
        ]

        injection_k = {}
        b_injected_k = {}
        if return_modes_only:
            reflection_k = {}
            phi_reflected_k = {}
            eig_reflected_k = {}
            phi_inv_reflected_k = {}
        else:
            sigma_obc_k = {}
            bloch_k = {}

        for ky, kz in itertools.product(k_inner[0], k_inner[1]):
            # Construct the system matrices for the OBC solver

            reduced_M = []

            for i in range(self.transport_repetitions + 1):
                temp = sparse.csr_matrix(
                    (M_slice[i, 0, 0].shape[0], M_slice[i, 0, 0].shape[1]),
                    dtype=xp.complex128,
                )
                for j, k in np.ndindex(ny, nz):
                    if M_slice[i, j, k].nnz > 0:
                        temp += M_slice[i, j, k] * xp.exp(
                            1j
                            * (
                                (ky) * (j - self.origin_key[1])
                                + (kz) * (k - self.origin_key[2])
                            )
                        )
                reduced_M.append(temp)

            temp = self._construct_contact_matrix(reduced_M).toarray()[xp.newaxis, :, :]
            A_tot = xp.split(temp, 3, axis=2)

            # HACK
            A_tot[1][0, :, :] = (
                A_tot[1][0, :, :] + A_tot[1][0, :, :].conj().T
            ) / 2  # Ensure Hermitian

            if return_modes_only:
                _, b_injected, phi_reflected, eig_reflected, phi_inv_reflected = (
                    self.obc_solver(
                        A_tot,
                        "",
                        return_injected=True,
                        return_modes_only=True,
                    )
                )
                reflection_k[ky, kz] = [
                    -A_tot[0][i] @ b for i, b in enumerate(phi_reflected)
                ]
                phi_reflected_k[ky, kz] = phi_reflected.copy()
                eig_reflected_k[ky, kz] = eig_reflected.copy()
                phi_inv_reflected_k[ky, kz] = phi_inv_reflected.copy()
            else:
                # Solve the OBC for the given ki and kj and store the
                # results in dictionaries
                x_ii, b_injected = self.obc_solver(A_tot, "", return_injected=True)
                sigma_obc_k[ky, kz] = A_tot[0] @ x_ii @ A_tot[2] / (ny * nz)
                bloch_k[ky, kz] = -x_ii @ A_tot[2] / (ny * nz)

            injection_k[ky, kz] = [-A_tot[0][i] @ b for i, b in enumerate(b_injected)]
            b_injected_k[ky, kz] = b_injected

        # Upscale injection and Bloch injection matrices
        injection = self._upscale_injection_modes(injection_k, num_energies)
        b_injected = self._upscale_injection_modes(b_injected_k, num_energies)

        obc_result = OBCResult(injection, b_injected)

        if return_modes_only:
            obc_result.reflection = self._upscale_injection_modes(
                reflection_k, num_energies
            )
            obc_result.phi_reflected = self._upscale_injection_modes(
                phi_reflected_k, num_energies
            )
            obc_result.eig_reflected = self._concatenate_eig(
                eig_reflected_k, num_energies
            )
            obc_result.phi_inv_reflected = self._upscale_pseudo_inverse(
                phi_inv_reflected_k, num_energies
            )

        else:
            obc_result.sigma_obc_k = sigma_obc_k
            obc_result.bloch_k = bloch_k

        return obc_result
