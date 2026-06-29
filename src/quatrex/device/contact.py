# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.boundary_conditions import obc
from qttools.comm import comm
from qttools.nevp import NEVP, Beyn, Full
from qttools.profiling import Profiler
from quatrex.bandstructure.contact import (
    contact_band_edges,
    contact_band_structure,
    contact_doping_density,
    contact_fermi_level,
)
from quatrex.core.config import ContactConfig, NEVPConfig, OBCConfig
from quatrex.grid.kpoints import monkhorst_pack

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
    if isinstance(order, str) and order not in ["reverse"]:
        raise ValueError(f"Invalid order string: {order}. Must be 'reverse' or None.")
    elif isinstance(order, xp.ndarray) and order.ndim != 1:
        raise ValueError(f"Order array must be 1-dimensional, got shape {order.shape}.")

    if order is None:
        return vector
    elif order == "reverse":
        return xp.flip(vector, axis=-1)
    else:
        return vector[..., order]


def order_block(
    block: NDArray,
    order: str | NDArray | None,
) -> NDArray:
    """Reorders the blocks of the given matrix according to the specified order.

    Parameters
    ----------
    block : NDArray
        The matrix block to reorder.
    order : str | NDArray | None
        The order in which to reorder the blocks.
        The only supported string is "reverse",
        which reverses the order of the blocks.

    Returns
    -------
    NDArray
        The reordered matrix block.

    """

    if isinstance(order, str) and order not in ["reverse"]:
        raise ValueError(f"Invalid order string: {order}. Must be 'reverse' or None.")
    elif isinstance(order, xp.ndarray) and order.ndim != 1:
        raise ValueError(f"Order array must be 1-dimensional, got shape {order.shape}.")

    if order is None:
        return block
    elif order == "reverse":
        return xp.flip(block, axis=(-2, -1))
    else:
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
    elif isinstance(order, xp.ndarray) and order.ndim != 1:
        raise ValueError(f"Order array must be 1-dimensional, got shape {order.shape}.")

    if order is None:
        return None
    elif order == "reverse":
        return "reverse"
    else:
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
    direction : int
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

        if len(contact_config.origin) != 3:
            raise ValueError("Origin must be a 3D coordinate.")
        if contact_config.lattice_vectors.shape != (3, 3):
            raise ValueError("Vectors must be a 3x3 array.")
        if contact_config.direction not in ["a", "b", "c"]:
            raise ValueError("Direction must be one of 'a', 'b', or 'c'.")

        self.device = device
        self.name = contact_config.name

        self.lattice_vectors = contact_config.lattice_vectors
        self.origin = contact_config.origin

        self.direction = "abc".index(contact_config.direction)
        self.transverse_axes = [0, 1, 2]
        self.transverse_axes.remove(self.direction)

        self.unit_cell_hamiltonian = {}
        self.unit_cell_overlap = {}

        # Get the atoms inside the origin cell (defined by the user)
        self.origin_atom_indices = self._get_atom_indices_in_cell(0, 0, 0)
        self.origin_orbital_indices = self._atom_to_orbital_indices(
            self.origin_atom_indices
        )

        self.origin_num_orbitals = len(self.origin_orbital_indices)

        if self.origin_num_orbitals == 0:
            raise ValueError(
                f"Error in contact {self.name}: No atoms found inside the origin cell."
            )

        if comm.rank == 0:
            print(f"Contact {self.name}:", flush=True)
            print(
                f"    Number of orbitals inside the origin cell: {self.origin_num_orbitals}",
                flush=True,
            )

        # Check how many periodic repetitions are in the transverse
        # directions
        self._init_periodic_transverse_repetitions()
        ny, nz = self.transverse_repetition_grid
        if comm.rank == 0:
            print(
                f"    Number of periodic repetitions in the transverse directions: {ny} x {nz}",
                flush=True,
            )

        # TODO Check if the contact transverse UC vectors are in the
        # same direction as the device vectors

        # +-1 difference because when building the supercells,
        # the last connection is part of the bigger connection block
        self.num_transport_cells = self._init_orbital_indices()

        # Initialize the hamiltonian and overlap matrices
        radius = self._init_hamiltonian_overlap_matrices()

        self._hermitianize_unit_cell_matrices()

        if comm.rank == 0:
            print(
                f"    Number of repetitions in transport direction: {self.num_transport_cells}",
                flush=True,
            )
            print(f"    Maximum coupling radius: {radius}")

        # Orbitals for contact (where to apply the OBC)
        # Sorted first in transport direction, then in transverse directions
        self.orbital_indices = np.concatenate(
            [
                self.unit_cell_orbital_indices[i, j, k]
                for j, k, i in np.ndindex(ny, nz, self.num_transport_cells)
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
            for i in range(self.num_transport_cells + 1)
        ]

        # We then need to sort the 10 matrix to have the same ordering as the contact OBCs
        self.transverse_to_transport_indices = np.concatenate(
            [
                np.arange(self.origin_num_orbitals)
                + i * self.origin_num_orbitals
                + k * self.origin_num_orbitals * ny * nz
                for i in range(ny * nz)
                for k in range(self.num_transport_cells)
            ],
            dtype=int,
        )[None, :]

        self.obc_solver = self._configure_obc(
            device.config.electron.obc, device.config.compute.nevp
        )

        # NOTE: We can either explicitly set the Fermi level in the
        # contact config, or compute it from the doping density and a
        # mid-gap energy.
        if contact_config.fermi_level is not None and device.config.scsp is None:
            self.fermi_level = contact_config.fermi_level
            self.mid_gap_energy = contact_config.mid_gap_energy
        else:
            # TODO: The kpoint grid info is not in the contact config,
            # but we need it to compute a Fermi level. Would be nice if
            # this could be accessed via a sort of general, global
            # simulation context.
            kpoints_transverse = (2 * np.pi) * monkhorst_pack(
                device.config.device.kpoint_grid, device.config.device.kpoint_shift
            )[:, self.transverse_axes]

            doping_density = contact_doping_density(
                coordinates=device.atom_coordinates[self.origin_atom_indices],
                geometry_regions=device.config.device.geometry.regions,
            )

            if comm.rank == 0:
                print(f"    Doping density: {doping_density} Å^-3", flush=True)

            # HACK: If there is a potential present in the input folder,
            # it will have already been baked into the unit cell
            # Hamiltonian by the time we get here, so it now needs to be
            # removed to correctly compute the Fermi level and band
            # edges and then it has to be added back again immediately
            # after to not affect the rest of the calculation. Because
            # the potential is added in-place rather than just set, we
            # have to in-place add its negative to effectively remove
            # it.
            potential = self.device.potential.copy()

            self.device.potential = -potential
            self.device.apply_potential()
            self._init_hamiltonian_overlap_matrices()

            (
                self.fermi_level,
                self.mid_gap_energy,
                self.delta_fermi_level_conduction_band,
            ) = self._compute_fermi_level(
                num_kpoints_transport=contact_config.num_kpoints_transport,
                kpoints_transverse=kpoints_transverse,
                mid_gap_energy=contact_config.mid_gap_energy,
                temperature=contact_config.temperature,
                doping_density=doping_density,
                cell_volume=np.abs(np.linalg.det(self.lattice_vectors)),
            )

            self.device.potential = potential
            self.device.apply_potential()
            self._init_hamiltonian_overlap_matrices()

        self.voltage = contact_config.voltage
        self.temperature = contact_config.temperature

        if comm.rank == 0:
            print(f"    Fermi level: {self.fermi_level} eV", flush=True)
            print(f"    Mid-gap energy: {self.mid_gap_energy} eV", flush=True)
            if contact_config.fermi_level is None or device.config.scsp is not None:
                print(
                    f"    Delta Fermi level: {self.delta_fermi_level_conduction_band} eV",
                    flush=True,
                )
            print(f"    Voltage: {self.voltage} V", flush=True)
            print(f"    Temperature: {self.temperature} K", flush=True)

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

        atomic_species = self.device.atomic_species[atom_indices]

        for origin_atom_index in self.origin_atom_indices:

            delta = (
                shifted_atom_coordinates
                - self.device.atom_coordinates[origin_atom_index, :]
            )

            # Find the atoms in the periodic repetition that are close
            # to the atom in the origin cell and have the same element
            found_atoms = np.nonzero(
                (np.linalg.norm(delta, axis=1) < tol)
                & (self.device.atomic_species[origin_atom_index] == atomic_species)
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
        repetitions_y_pos = self._count_repetitions(self.transverse_axes[0], 1)
        repetitions_y_neg = self._count_repetitions(self.transverse_axes[0], -1)
        repetitions_z_pos = self._count_repetitions(self.transverse_axes[1], 1)
        repetitions_z_neg = self._count_repetitions(self.transverse_axes[1], -1)

        # Store the number of periodic repetitions in the contact object
        # and the coordinates of the origin cell
        self.origin_cell_offset = np.array((repetitions_y_neg, repetitions_z_neg))
        self.transverse_repetition_grid = np.array(
            [
                repetitions_y_pos + repetitions_y_neg + 1,
                repetitions_z_pos + repetitions_z_neg + 1,
            ]
        )

    def _init_orbital_indices(self) -> int:
        """Initializes orbital indices for all periodic repetitions
        in transverse directions and counts number of transport cells.

        Returns
        -------
        int
            The number of periodic repetitions in the transport
            direction needed for convergence.

        """

        # Initialize empty orbitals indices
        # for each periodic repetition in transverse directions
        # list[ny][nz][transport_index] -> orbital indices
        ny, nz = self.transverse_repetition_grid
        self.unit_cell_orbital_indices = {}

        residual_orbitals = np.arange(self.device.hamiltonians[(0, 0, 0)].shape[0])

        residual_orbitals_old = residual_orbitals.copy()

        # First initialize all orbital indices
        # NOTE: This is basically a while True loop with a return inside.
        for transport_index in itertools.count(0):
            residual_orbitals = self._init_orbitals_transverse(
                transport_index, residual_orbitals
            )
            if self._residual_coupling(residual_orbitals) == 0:
                return transport_index

            # The residual orbitals did not change
            # but there are still residual couplings
            # then some orbitals got missed
            if np.array_equal(residual_orbitals_old, residual_orbitals):
                raise ValueError(
                    f"Error in contact {self.name}: "
                    f"Could not find all orbitals in the contact unit cell. "
                )
            residual_orbitals_old = residual_orbitals.copy()

    def _init_orbitals_transverse(
        self, transport_index: int, residual_orbitals: NDArray
    ) -> NDArray:
        """Initialize the orbitals for a given transport cell
        for all transverse periodic repetitions. Additionally,
        this method updates the residual orbitals.

        Parameters
        ----------
        transport_index : int
            The index of the periodic repetition in the transport
            direction.
        residual_orbitals : NDArray
            The orbital indices that have not yet been included in
            the contact unit cell.

        Returns
        -------
        residual_orbitals : NDArray
            The updated residual orbital indices after including
            the orbitals from this transport cell.

        """

        # Iterate over all (x, y) combinations
        ny, nz = self.transverse_repetition_grid
        for idy, idz in itertools.product(
            range(ny),
            range(nz),
        ):
            index = [idy - self.origin_cell_offset[0], idz - self.origin_cell_offset[1]]
            index.insert(self.direction, transport_index)

            # Process atom and orbital indices
            atom_indices = self._get_atom_indices_in_cell(*index)
            atom_indices = self._reorder_atoms(atom_indices, index)
            orbital_indices = self._atom_to_orbital_indices(atom_indices)

            self.unit_cell_orbital_indices[transport_index, idy, idz] = orbital_indices

            residual_orbitals = residual_orbitals[
                ~np.isin(residual_orbitals, orbital_indices)
            ]

        return residual_orbitals

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
                    coordinates.append(np.array([y, z]))

        return coordinates

    def _hermitianize_unit_cell_matrices(self):
        """Ensures that the unit cell Hamiltonian and overlap matrices
        are the Hermitian conjugate of the one with the opposite index.
        """

        for key in list(self.unit_cell_hamiltonian.keys()):
            if key[0] == 0:

                key_opp = (key[0], -key[1], -key[2])
                self.unit_cell_hamiltonian[key_opp] = self.unit_cell_hamiltonian[
                    key
                ].T.conj()

        for key in list(self.unit_cell_overlap.keys()):

            if key[0] == 0:
                key_opp = (key[0], -key[1], -key[2])
                self.unit_cell_overlap[key_opp] = self.unit_cell_overlap[key].T.conj()

    def _init_hamiltonian_overlap_matrices(self) -> int:
        """Initializes the hamiltonian and overlap matrices.

        Returns
        -------
        int
            The maximum coupling radius found.

        """

        # The hamiltonian and overlap matrices for a given transverse
        # slice are obtained around the origin cell increasing radius
        # until no more hamiltonian or overlap is found.

        for transport_index in range(self.num_transport_cells + 1):
            for radius in itertools.count(0):  # While True loop

                found_any_at_radius = False

                # Get the coordinates on the circumference of the grid
                circumference_coordinates = self._get_circumference_coordinates(radius)
                for cell_coordinates in circumference_coordinates:

                    if self.device.gamma_only and (
                        np.any(
                            (self.transverse_repetition_grid == 1)
                            & (cell_coordinates != 0)
                        )
                    ):
                        continue

                    # The coupling is defined in the in the device
                    # hamiltonian at (H_1, H_2)
                    shifted_coordinates = cell_coordinates + self.origin_cell_offset
                    hopping_indices = np.array(
                        (shifted_coordinates + 0.0001)
                        / self.transverse_repetition_grid,
                        dtype=int,
                    )
                    hopping_indices += np.array(
                        [-1 if i < 0 else 0 for i in shifted_coordinates], dtype=int
                    )

                    # Edge case for periodic devices,
                    # when the interactions loop
                    if self.device.gamma_only and np.any(
                        self.transverse_repetition_grid > 1
                    ):
                        diameter = 2 * radius + 1
                        hopping_indices = np.array([0, 0])

                        if np.any(
                            (diameter > self.transverse_repetition_grid)
                            & (self.transverse_repetition_grid > 1)
                        ):
                            raise ValueError(
                                f"Error in contact {self.name}: \n"
                                f"Cannot obtain the UC matrices from the Gamma-point device matrix, probably because the basis decay is not enough.\n"
                                f"Possible solutions:\n"
                                f"  - Increase the UC to include the entire cross-section (1x1 contact UC)\n"
                                f"  - Provide all the hopping Hamiltonians in the device, not only the Gamma point."
                                f"Error encountered with radius {radius}"
                            )

                    # These are the orbitals where to look for the coupling
                    idy, idz = shifted_coordinates % self.transverse_repetition_grid
                    orbital_indices = self.unit_cell_orbital_indices[
                        transport_index, idy, idz
                    ]

                    found_hamiltonian = self._update_unit_cell_matrices(
                        self.device.hamiltonians,
                        self.unit_cell_hamiltonian,
                        cell_coordinates,
                        hopping_indices,
                        transport_index,
                        orbital_indices,
                    )
                    found_overlap = self._update_unit_cell_matrices(
                        self.device.overlap_matrices,
                        self.unit_cell_overlap,
                        cell_coordinates,
                        hopping_indices,
                        transport_index,
                        orbital_indices,
                    )

                    if found_overlap and not found_hamiltonian:
                        raise ValueError(
                            f"Error in contact {self.name}: \n"
                            f"Overlap matrix found without corresponding Hamiltonian at transport index {transport_index} "
                            f"and transverse coordinates {cell_coordinates}."
                        )

                    if found_hamiltonian or found_overlap:
                        found_any_at_radius = True

                if not found_any_at_radius:
                    break

        return radius - 1

    def _update_unit_cell_matrices(
        self,
        quantity: dict,
        output_dict: dict,
        cell_coordinates: NDArray,
        hopping_indices: NDArray,
        transport_index: int,
        orbital_indices: NDArray,
    ) -> bool:
        """Updates the unit cell matrices for a given quantity (hamiltonian or overlap).

        Parameters
        ----------
        quantity : dict
            The device quantity (hamiltonian or overlap) to extract
            the hopping matrix from.
        output_dict : dict
            The output dictionary to store the unit cell matrices.
        cell_coordinates : NDArray
            The transverse cell coordinates.
        hopping_indices : NDArray
            The hopping indices in the device quantity.
        transport_index : int
            The transport index of the periodic repetition.
        orbital_indices : NDArray
            The orbital indices for the periodic repetition.

        Returns
        -------
        bool
            True if a non-zero matrix was found and added, False
            otherwise.

        """

        opposite_hopping_indices = (hopping_indices * -1).tolist()
        hopping_indices = hopping_indices.copy().tolist()

        hopping_indices.insert(self.direction, 0)
        opposite_hopping_indices.insert(self.direction, 0)

        hopping_indices = tuple(hopping_indices)
        opposite_hopping_indices = tuple(opposite_hopping_indices)

        hopping_matrix = quantity.get(hopping_indices)
        if hopping_matrix is None:
            return False

        y, z = cell_coordinates

        # TODO: The hopping matrix sits on the GPU. It seems that there
        # is some strange fancy indexing bug that makes it necessary to
        # handle slicing on the CPU. (cupy-13.5.1)
        # TODO: Change how to check where the data resides (GPU or CPU)
        hopping_matrix = (
            hopping_matrix.get() if hasattr(hopping_matrix, "get") else hopping_matrix
        )

        opposite_hopping_matrix = quantity.get(opposite_hopping_indices)

        if opposite_hopping_matrix is None:
            raise ValueError(
                f"Error in contact {self.name}: \n"
                f"Hopping matrix found at {hopping_indices} without corresponding opposite hopping at {opposite_hopping_indices}."
            )

        opposite_hopping_matrix = (
            opposite_hopping_matrix.get()
            if hasattr(opposite_hopping_matrix, "get")
            else opposite_hopping_matrix
        )

        # In reduced, the coupling is only given by the upper triangular part of the Hamiltonian.
        # We need to add the lower part to get the full coupling.
        unit = (
            sparse.csr_matrix(
                hopping_matrix[self.origin_orbital_indices, :][:, orbital_indices]
            )
            + sparse.csr_matrix(
                opposite_hopping_matrix[orbital_indices, :][
                    :, self.origin_orbital_indices
                ]
            ).T.conj()
        )

        if np.array_equal(self.origin_orbital_indices, orbital_indices):
            unit -= sparse.diags(
                hopping_matrix[self.origin_orbital_indices, :][
                    :, orbital_indices
                ].diagonal(),
                format="csr",
            )

        if unit.nnz == 0:
            return False

        output_dict[(transport_index, y, z)] = unit

        return True

    def _residual_coupling(self, residual_orbitals: NDArray) -> bool:
        """Checks if there is residual coupling between the orbitals in
        the contact and the full device.

        Parameters
        ----------
        residual_orbitals : NDArray
            The orbital indices that have not yet been included in
            the contact unit cell.

        Returns
        -------
        bool
            True if there is residual coupling, False otherwise.

        """

        return (
            self.device.hamiltonians[0, 0, 0][self.origin_orbital_indices, :][
                :, residual_orbitals
            ].nnz
            + self.device.hamiltonians[0, 0, 0][residual_orbitals, :][
                :, self.origin_orbital_indices
            ].nnz
        )

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
                block_sections=self.num_transport_cells,  # WARNING: overrides config
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

            a_xx = None
            if nevp_config.reduce_sparsity:
                # For QTBM, we can precompute the sparsity pattern of
                # the matrix polynomial coefficients here.

                a_xx = [None] * (2 * self.num_transport_cells + 1)
                for r, h_r in self.unit_cell_hamiltonian.items():
                    s_r = self.unit_cell_overlap.get(r, 0)
                    a_r = sparse.csc_matrix(s_r + h_r)

                    a_xx[self.num_transport_cells + r[0]] = a_r
                    a_xx[self.num_transport_cells - r[0]] = a_r.T

                a_xx = tuple(a_xx)

            return Full(
                eig_compute_location=nevp_config.eig_compute_location,
                use_pinned_memory=nevp_config.use_pinned_memory,
                reduce=nevp_config.reduce_sparsity,
                a_xx_sparsity=a_xx,
            )

        raise NotImplementedError(
            f"NEVP solver '{obc_config.nevp_solver}' not implemented."
        )

    def _compute_fermi_level(
        self,
        num_kpoints_transport: int,
        kpoints_transverse: NDArray,
        mid_gap_energy: float,
        temperature: float,
        doping_density: float,
        cell_volume: float,
    ) -> tuple[float, float, float]:
        """Computes the Fermi level for the contact.

        Parameters
        ----------
        num_kpoints_transport : int
            The number of k-points to sample in the transport direction
            for the band structure calculation.
        kpoints_transverse : NDArray
            The array of transverse k-points at which to compute the
            band structure.
        mid_gap_energy : float
            The mid-gap energy of the contact material.

        temperature : float
            The temperature of the contact in Kelvin.
        doping_density : float
            The doping density of the contact. This has to have the
            inverse units of the cell volume, e.g. cm^-3 if the cell
            volume is in cm^3.
        cell_volume : float
            The volume of the contact unit cell in the same units as the
            inverse of the doping density, e.g. cm^3 if the doping
            density is in cm^-3.

        Returns
        -------
        fermi_level : float
            The computed Fermi level in eV.
        mid_gap_energy : float
            The recomputed mid-gap energy based on the band structure.
        delta_fermi_level_conduction_band : float
            The energy difference between the Fermi level and the
            conduction band edge.

        """
        kpoints_transport = xp.linspace(
            -xp.pi, xp.pi, num_kpoints_transport, endpoint=False
        )

        e_k = xp.zeros(
            (
                num_kpoints_transport,
                kpoints_transverse.shape[0],
                self.num_transport_cells * self.origin_num_orbitals,
            ),
            dtype=float,
        )

        for n, (ki, kj) in enumerate(kpoints_transverse):
            hamiltonian_layer = self._construct_contact_matrix(
                self.unit_cell_hamiltonian, ki, kj
            )
            overlap_layer = self._construct_contact_matrix(
                self.unit_cell_overlap, ki, kj
            )

            h_xx = xp.split(hamiltonian_layer.toarray(), 3, axis=-1)
            s_xx = xp.split(overlap_layer.toarray(), 3, axis=-1)

            e_k[:, n, :] = contact_band_structure(kpoints_transport, h_xx, s_xx)

        # Average over transverse k-points.
        e_k = xp.mean(e_k, axis=1)

        fermi_level = contact_fermi_level(
            e_k=e_k,
            kpoints=kpoints_transport,
            mid_gap_energy=mid_gap_energy,
            cell_volume=cell_volume,
            doping_density=doping_density,
            temperature=temperature,
        )

        # Recompute the actual mid-gap energy from the band structure.
        valence_band_edge, conduction_band_edge = contact_band_edges(
            e_k, mid_gap_energy
        )
        if comm.rank == 0:
            print(f"    Conduction band minimum: {conduction_band_edge} eV", flush=True)
            print(f"    Valence band maximum: {valence_band_edge} eV", flush=True)
        mid_gap_energy = 0.5 * (conduction_band_edge + valence_band_edge)
        delta_fermi_level_conduction_band = conduction_band_edge - fermi_level

        return fermi_level, mid_gap_energy, delta_fermi_level_conduction_band

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
        for shift in range(self.num_transport_cells):
            layer = layers[shift:] + [zero] * shift
            coupling_matrix.append(sparse.vstack(layer, format="csr"))

        coupling_matrix = sparse.hstack(coupling_matrix[::-1], format="csr")

        indices = self.transverse_to_transport_indices
        return coupling_matrix[indices.T, indices]

    def _construct_contact_matrix(self, UC_matrix: dict, ky: float, kz: float):
        """Constructs the full contact matrix for the contact at given
        transverse k-points.
        Parameters
        ----------
        UC_matrix : dict
            A dictionary containing the unit cell matrices indexed by
            (i, j, k) tuples.
        ky : float
            The transverse wavevector in the y-direction.
        kz : float
            The transverse wavevector in the z-direction.

        Returns
        -------
        sparse.spmatrix
            The constructed contact matrix in sparse format.

        """

        n = UC_matrix[(0, 0, 0)].shape[0]
        num_cells = self.num_transport_cells
        zero = sparse.csr_matrix((n, n), dtype=xp.complex128)

        uc_right = [zero for _ in range(num_cells + 1)]
        for (x, y, z), ham in UC_matrix.items():
            if 0 <= x <= num_cells:
                uc_right[x] += ham * xp.exp(1j * (ky * y + kz * z))

        uc_left = [h.conj().T for h in uc_right[1:][::-1]]

        # Pad with zeros for the OBCs
        padding = [zero] * (num_cells - 1)
        first_row_blocks = uc_left + uc_right + padding

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

    @profiler.profile("Contact: Compute Boundary", level="default")
    def compute_boundary(
        self,
        k_outer: tuple[float, float, float],
        energies: NDArray,
        return_modes_only: bool = False,
    ) -> OBCResult:
        """Computes OBC for the contact at given k-points and energies.

        Parameters
        ----------
        k_outer : tuple[float, float, float]
            Wavevector. Captures periodicity in transverse directions.
        energies : NDArray
            Batch of energy values for which to compute the boundary
            conditions.
        return_modes_only : bool, optional
            Whether to return only the injection and surface modes
            without computing the full self-energy and Bloch matrices.

        Returns
        -------
        ContactOBCResult
            An object containing the computed OBC results, including
            injection modes, self-energy, and Bloch modes as applicable.

        """

        num_energies = energies.shape[0]
        ny, nz = self.transverse_repetition_grid

        # TODO: Batching over k-points can be implemented here
        # and not only over energies
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

            # Construct the hamiltonian and overlap matrices for the
            # given ki and kj
            H_tot = self._construct_contact_matrix(self.unit_cell_hamiltonian, ky, kz)
            S_tot = self._construct_contact_matrix(self.unit_cell_overlap, ky, kz)

            S_dense = xp.array(S_tot.todense())
            H_dense = xp.array(H_tot.todense())

            # Construct the system matrices for the OBC solver
            A_tot = xp.split((energies[:, None, None] * S_dense - H_dense), 3, axis=2)

            if return_modes_only:
                _, b_injected, phi_reflected, eig_reflected, phi_inv_reflected = (
                    self.obc_solver(
                        A_tot[1],
                        A_tot[2],
                        A_tot[0],
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
                x_ii, b_injected = self.obc_solver(
                    A_tot[1], A_tot[2], A_tot[0], "", return_injected=True
                )
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
