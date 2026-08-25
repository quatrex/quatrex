# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Contact discovery module for identifying contacts in the device."""

import itertools

import numpy as np

from qttools import NDArray, sparse
from qttools.comm import comm
from quatrex.core.config import ContactConfig, DeviceConfig


def _get_atom_indices_in_cell(
    atom_coordinates: NDArray,
    origin: NDArray,
    lattice_vectors: NDArray,
    repetition_inds: tuple[int, int, int],
) -> NDArray:
    """Gets the indices of atoms inside a specific periodic repetition.

    This method finds all device atoms that fall within the
    specified periodic repetition of the contact unit cell.

    Parameters
    ----------
    atom_coordinates : NDArray
        2D array of shape (N, 3) containing the coordinates of the
        device atoms.
    origin : NDArray
        1D array of shape (3,) representing the origin of the contact
        unit cell.
    lattice_vectors : NDArray
        2D array of shape (3, 3) representing the lattice vectors of
        the contact unit cell.
    repetition_inds : tuple[int, int, int]
        The periodic repetition coordinates (nx, ny, nz) of the contact
        unit cell.

    Returns
    -------
    NDArray
        1D array of atom indices that fall within the specified
        periodic repetition.

    """

    nx, ny, nz = repetition_inds

    # Shift the coordinates of the device atoms to the origin of the
    # contact
    relative_coordinates = atom_coordinates - origin

    # Compute the coefficients relative to the contact cell
    fractional_coordinates = relative_coordinates @ np.linalg.inv(lattice_vectors)

    # Get the indices of the atoms inside the periodic repetition
    indices_inside = np.nonzero(
        (fractional_coordinates[:, 0] >= nx)
        & (fractional_coordinates[:, 0] < nx + 1)
        & (fractional_coordinates[:, 1] >= ny)
        & (fractional_coordinates[:, 1] < ny + 1)
        & (fractional_coordinates[:, 2] >= nz)
        & (fractional_coordinates[:, 2] < nz + 1)
    )[0]

    return indices_inside


def _atom_to_orbital_indices(
    orbital_offsets: NDArray,
    atom_indices: NDArray,
) -> NDArray:
    """Gets the orbital indices corresponding to the atoms

    Parameters
    ----------
    orbital_offsets : NDArray
        The offsets of the orbitals for each atom.
    atom_indices : NDArray
        The indices of the atoms.

    Returns
    -------
    NDArray
        The indices of the orbitals corresponding to the atoms.

    """

    starts = orbital_offsets[atom_indices]
    ends = orbital_offsets[atom_indices + 1]
    counts = ends - starts

    orbital_indices = np.repeat(starts, counts) + np.concatenate(
        [np.arange(c) for c in counts]
    )

    return orbital_indices


def _count_repetitions(
    atom_coordinates: NDArray,
    origin: NDArray,
    lattice_vectors: NDArray,
    axis: int,
    sign: int,
    reference_length: int | None = None,
) -> int:
    """Counts periodic repetitions in a given direction.

    Parameters
    ----------
    atom_coordinates : NDArray
        2D array of shape (N, 3) containing the coordinates of the
        device atoms.
    origin : NDArray
        1D array of shape (3,) representing the origin of the contact
        unit cell.
    lattice_vectors : NDArray
        2D array of shape (3, 3) representing the lattice vectors of
        the contact unit cell.
    axis : int
        The axis along which to count the repetitions (0, 1, or
        2).
    sign : int
        The sign of the direction to count the repetitions (1
        for positive, -1 for negative).
    reference_length : int, optional
        The number of atoms in the origin cell, used to check for
        consistency across repetitions. If provided, raises an error
        if the number of atoms in a repetition does not match this
        length.

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
        atom_indices = _get_atom_indices_in_cell(
            atom_coordinates=atom_coordinates,
            origin=origin,
            lattice_vectors=lattice_vectors,
            repetition_inds=idx,
        )

        if atom_indices.shape[0] == 0:
            break

        # Number of atoms inside the periodic repetition
        # does not match the origin cell
        if reference_length is not None and len(atom_indices) != reference_length:
            raise ValueError(
                f"Atom mismatch at {tuple(idx)} for axis {axis} and sign {sign}."
            )

    # Minus one because the last repetition had no atoms
    return repetition - 1


def _init_periodic_transverse_repetitions(
    atom_coordinates: NDArray,
    origin: NDArray,
    lattice_vectors: NDArray,
    transport_direction: int,
    reference_length: int | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Determines number of periodic repetitions in transverse directions.

    Parameters
    ----------
    atom_coordinates : NDArray
        2D array of shape (N, 3) containing the coordinates of the
        device atoms.
    origin : NDArray
        1D array of shape (3,) representing the origin of the contact
        unit cell.
    lattice_vectors : NDArray
        2D array of shape (3, 3) representing the lattice vectors of
        the contact unit cell.
    transport_direction : int
        The axis along which transport occurs (0, 1, or 2).
    reference_length : int, optional
        The number of atoms in the origin cell, used to check for
        consistency across repetitions.

    Returns
    -------
    tuple[int, int]:
        The number of periodic repetitions in the transverse directions.
    tuple[int, int]:
        The offset of the origin cell in the transverse directions.

    """

    transverse_axes = [0, 1, 2]
    transverse_axes.remove(transport_direction)

    # Count the number of periodic repetitions in each transverse direction
    # (y+, y-, z+, z- )
    repetitions_y_pos = _count_repetitions(
        atom_coordinates,
        origin,
        lattice_vectors,
        transverse_axes[0],
        1,
        reference_length,
    )
    repetitions_y_neg = _count_repetitions(
        atom_coordinates,
        origin,
        lattice_vectors,
        transverse_axes[0],
        -1,
        reference_length,
    )
    repetitions_z_pos = _count_repetitions(
        atom_coordinates,
        origin,
        lattice_vectors,
        transverse_axes[1],
        1,
        reference_length,
    )
    repetitions_z_neg = _count_repetitions(
        atom_coordinates,
        origin,
        lattice_vectors,
        transverse_axes[1],
        -1,
        reference_length,
    )

    # Store the number of periodic repetitions in the contact object
    # and the coordinates of the origin cell
    origin_cell_offset = np.array((repetitions_y_neg, repetitions_z_neg))
    transverse_repetition_grid = np.array(
        [
            repetitions_y_pos + repetitions_y_neg + 1,
            repetitions_z_pos + repetitions_z_neg + 1,
        ]
    )

    return transverse_repetition_grid, origin_cell_offset


def _reorder_atoms(
    origin_atom_indices: NDArray,
    atom_coordinates: NDArray,
    lattice_vectors: NDArray,
    atomic_species: NDArray,
    atom_indices: NDArray,
    index: tuple[int, int, int],
    tol: float = 0.3,
) -> NDArray:
    """Reorders atoms to match the ordering in the origin cell.

    This method ensures consistent atom ordering across different
    periodic repetitions of the contact unit cell.

    Parameters
    ----------
    origin_atom_indices : NDArray
        1D array of shape (M,) containing the indices of the atoms
        in the origin cell.
    atom_coordinates : NDArray
        2D array of shape (N, 3) containing the coordinates of the
        device atoms.
    lattice_vectors : NDArray
        2D array of shape (3, 3) representing the lattice vectors of
        the contact unit cell.
    atomic_species : NDArray
        1D array of shape (N,) containing the atomic species of the
        device atoms.
    atom_indices : NDArray
        Indices of atoms inside the periodic repetition to be
        reordered.
    index : tuple[int, int, int]
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

    # Shift the coordinates of the atoms inside the periodic
    # repetition to match the origin cell
    shifted_atom_coordinates = atom_coordinates[
        atom_indices, :
    ] - lattice_vectors @ np.array(index)

    test_atomic_species = atomic_species[atom_indices]

    for origin_atom_index in origin_atom_indices:

        delta = shifted_atom_coordinates - atom_coordinates[origin_atom_index, :]

        # Find the atoms in the periodic repetition that are close
        # to the atom in the origin cell and have the same element
        found_atoms = np.nonzero(
            (np.linalg.norm(delta, axis=1) < tol)
            & (atomic_species[origin_atom_index] == test_atomic_species)
        )[0]
        if found_atoms.size == 0:
            raise ValueError(
                f"Atom {origin_atom_index} not found in the periodic repetition at ({index})."
            )
        if found_atoms.size > 1:
            raise ValueError(
                f"Multiple atoms found in the periodic repetition at ({index}) "
                f"matching atom {origin_atom_index} from the origin cell."
            )

        sorted_atom_indices.append(atom_indices[found_atoms[0]])

    return np.array(sorted_atom_indices, dtype=int)


def _init_orbitals_transverse(
    atom_coordinates: NDArray,
    atomic_species: NDArray,
    orbital_offsets: NDArray,
    origin: NDArray,
    origin_atom_indices: NDArray,
    lattice_vectors: NDArray,
    transverse_repetition_grid: tuple[int, int],
    origin_cell_offset: tuple[int, int],
    transport_direction: int,
    transport_index: int,
) -> tuple[NDArray, dict]:
    """Initialize the orbitals for a given transport cell
    for all transverse periodic repetitions. Additionally, this
    method updates the residual orbitals.

    Parameters
    ----------
    atom_coordinates : NDArray
        2D array of shape (N, 3) containing the coordinates of the
        device atoms.
    atomic_species : NDArray
        1D array of shape (N,) containing the atomic species of the
        device atoms.
    orbital_offsets : NDArray
        1D array of shape (N+1,) containing the offsets of the orbitals
        for each atom.
    origin : NDArray
        1D array of shape (3,) representing the origin of the contact
        unit cell.
    origin_atom_indices : NDArray
        1D array of shape (M,) containing the indices of the atoms
        in the origin cell.
    lattice_vectors : NDArray
        2D array of shape (3, 3) representing the lattice vectors of
        the contact unit cell.
    transverse_repetition_grid : tuple[int, int]
        The number of periodic repetitions in the transverse directions.
    origin_cell_offset : tuple[int, int]
        The offset of the origin cell in the transverse directions.
    transport_direction : int
        The axis along which transport occurs (0, 1, or 2).
    transport_index : int
        The index of the periodic repetition in the transport
        direction.

    Returns
    -------
    transverse_indices : dict
        Updated dictionary containing the orbital indices for each
        periodic repetition in the transverse directions.

    """

    transverse_indices = {}

    # Iterate over all (x, y) combinations
    ny, nz = transverse_repetition_grid
    for idy, idz in itertools.product(
        range(ny),
        range(nz),
    ):
        index = [idy - origin_cell_offset[0], idz - origin_cell_offset[1]]
        index.insert(transport_direction, transport_index)

        # Process atom and orbital indices
        atom_indices = _get_atom_indices_in_cell(
            atom_coordinates=atom_coordinates,
            origin=origin,
            lattice_vectors=lattice_vectors,
            repetition_inds=index,
        )
        atom_indices = _reorder_atoms(
            origin_atom_indices=origin_atom_indices,
            atom_coordinates=atom_coordinates,
            lattice_vectors=lattice_vectors,
            atomic_species=atomic_species,
            atom_indices=atom_indices,
            index=index,
        )
        orbital_indices = _atom_to_orbital_indices(
            orbital_offsets=orbital_offsets,
            atom_indices=atom_indices,
        )

        transverse_indices[transport_index, idy, idz] = orbital_indices

    return transverse_indices


def _residual_coupling(
    hamiltonian: sparse.spmatrix,
    origin_orbital_indices: NDArray,
    residual_orbitals: NDArray,
) -> bool:
    """Checks if there is residual coupling between the orbitals in
    the contact and the full device.

    Parameters
    ----------
    hamiltonian : sparse.spmatrix
        The Hamiltonian matrix of the device.
    origin_orbital_indices : NDArray
        The orbital indices of the contact unit cell.
    residual_orbitals : NDArray
        The orbital indices that have not yet been included in
        the contact unit cell.

    Returns
    -------
    bool
        True if there is residual coupling, False otherwise.

    """

    return (
        hamiltonian[origin_orbital_indices, :][:, residual_orbitals].nnz
        + hamiltonian[residual_orbitals, :][:, origin_orbital_indices].nnz
    )


def _init_orbital_indices(
    hamiltonian: sparse.spmatrix,
    atom_coordinates: NDArray,
    atomic_species: NDArray,
    orbital_offsets: NDArray,
    origin_atom_indices: NDArray,
    origin_orbital_indices: NDArray,
    origin: NDArray,
    lattice_vectors: NDArray,
    transverse_repetition_grid: tuple[int, int],
    origin_cell_offset: tuple[int, int],
    transport_direction: int,
) -> tuple[int, dict]:
    """Initializes orbital indices for all periodic repetitions
    in transverse directions and counts number of transport cells.

    Parameters
    ----------
    hamiltonian : sparse.spmatrix
        The Hamiltonian matrix of the device.
    atom_coordinates : NDArray
        2D array of shape (N, 3) containing the coordinates of the
        device atoms.
    atomic_species : NDArray
        1D array of shape (N,) containing the atomic species of the
        device atoms.
    orbital_offsets : NDArray
        1D array of shape (N+1,) containing the offsets of the orbitals
        for each atom.
    origin_atom_indices : NDArray
        1D array of shape (M,) containing the indices of the atoms in
        the origin cell.
    origin_orbital_indices : NDArray
        1D array of shape (K,) containing the indices of the orbitals in
        the origin cell.
    origin : NDArray
        1D array of shape (3,) representing the origin of the contact
        unit cell.
    lattice_vectors : NDArray
        2D array of shape (3, 3) representing the lattice vectors of the
        contact unit cell.
    transverse_repetition_grid : tuple[int, int]
        The number of periodic repetitions in the transverse directions.
    origin_cell_offset : tuple[int, int]
        The offset of the origin cell in the transverse directions.
    transport_direction : int
        The axis along which transport occurs (0, 1, or 2).

    Returns
    -------
    int
        The number of periodic repetitions in the transport direction
        needed for convergence.
    dict
        A dictionary containing the orbital indices for each periodic
        repetition. The keys are tuples of the form (transport_index,
        idy, idz), where transport_index is the index of the periodic
        repetition in the transport direction, and idy, idz are the
        indices of the periodic repetitions in the transverse
        directions.

    """

    # Initialize empty orbitals indices
    # for each periodic repetition in transverse directions
    unit_cell_orbital_indices = {}

    residual_orbitals = np.arange(hamiltonian.shape[0])
    residual_orbitals_old = residual_orbitals.copy()

    # First initialize all orbital indices
    # NOTE: This is basically a while True loop with a return inside.
    for transport_index in itertools.count(0):
        transverse_indices = _init_orbitals_transverse(
            atom_coordinates=atom_coordinates,
            atomic_species=atomic_species,
            orbital_offsets=orbital_offsets,
            origin=origin,
            origin_atom_indices=origin_atom_indices,
            lattice_vectors=lattice_vectors,
            transverse_repetition_grid=transverse_repetition_grid,
            origin_cell_offset=origin_cell_offset,
            transport_direction=transport_direction,
            transport_index=transport_index,
        )
        unit_cell_orbital_indices.update(transverse_indices)
        residual_orbitals = residual_orbitals[
            ~np.isin(
                residual_orbitals, np.concatenate(list(transverse_indices.values()))
            )
        ]

        if (
            _residual_coupling(
                hamiltonian=hamiltonian,
                origin_orbital_indices=origin_orbital_indices,
                residual_orbitals=residual_orbitals,
            )
            == 0
        ):
            return transport_index, unit_cell_orbital_indices

        # The residual orbitals did not change
        # but there are still residual couplings
        # then some orbitals got missed
        if np.array_equal(residual_orbitals_old, residual_orbitals):
            raise ValueError("Could not find all orbitals in the contact unit cell. ")
        residual_orbitals_old = residual_orbitals.copy()


def real_space_discovery(
    hamiltonian: sparse.spmatrix,
    atomic_species: NDArray,
    atom_coordinates: NDArray,
    orbital_offsets: NDArray,
    contact_config: ContactConfig,
) -> tuple[dict, tuple[int, int, int], tuple[int, int, int]]:
    """Discovers the contact unit cell in real space.

    Note
    ----
    The Hamiltonian is used to check when the coupling of the contact to
    the device stops. Not the real hamiltonian would be needed, but only
    the sparsity pattern.

    Parameters
    ----------
    hamiltonian : sparse.spmatrix
        The Hamiltonian matrix of the device.
    atomic_species : NDArray
        1D array of shape (N,) containing the atomic species of the
        device atoms.
    atom_coordinates : NDArray
        2D array of shape (N, 3) containing the coordinates of the
        device atoms.
    orbital_offsets : NDArray
        1D array of shape (N+1,) containing the offsets of the orbitals
        for each atom.
    contact_config : ContactConfig
        The configuration of the contact, including its name, origin,
        lattice vectors, and transport direction.

    Returns
    -------
    dict
        A dictionary containing the orbital indices for each periodic
        repetition in both transport and transverse directions. The keys
        are tuples of the form (transport_index, idy, idz), where
        transport_index is the index of the periodic repetition in the
        transport direction, and idy, idz are the indices of the
        periodic repetitions in the transverse directions.
    tuple[int, int, int]
        The number of periodic repetitions in the transport and
        transverse directions.
    tuple[int, int, int]
        The offset of the origin cell in the transport and transverse
        directions.

    """
    if comm.rank == 0:
        print(f"Discovering contact {contact_config.name} in real space...", flush=True)

    if contact_config.origin is None or contact_config.lattice_vectors is None:
        raise ValueError(
            "Origin and lattice vectors must be specified for real-space contact."
        )

    if len(contact_config.origin) != 3:
        raise ValueError("Origin must be a 3D coordinate.")
    if contact_config.lattice_vectors.shape != (3, 3):
        raise ValueError("Vectors must be a 3x3 array.")

    lattice_vectors = contact_config.lattice_vectors
    origin = contact_config.origin

    # Get the atoms inside the origin cell (defined by the user)
    origin_atom_indices = _get_atom_indices_in_cell(
        atom_coordinates=atom_coordinates,
        origin=origin,
        lattice_vectors=lattice_vectors,
        repetition_inds=(0, 0, 0),
    )
    origin_orbital_indices = _atom_to_orbital_indices(
        orbital_offsets=orbital_offsets,
        atom_indices=origin_atom_indices,
    )

    origin_num_orbitals = len(origin_orbital_indices)

    if origin_num_orbitals == 0:
        raise ValueError(
            f"Error in contact {contact_config.name}: No atoms found inside the origin cell."
        )

    if comm.rank == 0:
        print(f"Contact {contact_config.name}:", flush=True)
        print(
            f"    Number of orbitals inside the origin cell: {origin_num_orbitals}",
            flush=True,
        )

    # Check how many periodic repetitions are in the transverse
    # directions
    transport_direction = "abc".index(contact_config.transport_direction)
    transverse_repetition_grid, origin_cell_offset = (
        _init_periodic_transverse_repetitions(
            atom_coordinates=atom_coordinates,
            origin=origin,
            lattice_vectors=lattice_vectors,
            transport_direction=transport_direction,
            reference_length=len(origin_atom_indices),
        )
    )
    if comm.rank == 0:
        ny, nz = transverse_repetition_grid
        print(
            f"    Number of periodic repetitions in the transverse directions: {ny} x {nz}",
            flush=True,
        )

    # TODO Check if the contact transverse UC vectors are in the
    # same direction as the device vectors
    num_transport_cells, unit_cell_orbital_indices = _init_orbital_indices(
        hamiltonian=hamiltonian,
        atom_coordinates=atom_coordinates,
        atomic_species=atomic_species,
        orbital_offsets=orbital_offsets,
        origin_atom_indices=origin_atom_indices,
        origin_orbital_indices=origin_orbital_indices,
        origin=origin,
        lattice_vectors=lattice_vectors,
        transverse_repetition_grid=transverse_repetition_grid,
        origin_cell_offset=origin_cell_offset,
        transport_direction=transport_direction,
    )

    repetition_grid = list(transverse_repetition_grid)
    repetition_grid.insert(transport_direction, num_transport_cells)

    origin_key = (0, *origin_cell_offset)

    return unit_cell_orbital_indices, tuple(repetition_grid), origin_key


def simplified_discovery(
    contact_name: str,
    num_orbitals: int,
    device_config: DeviceConfig,
    contact_config: ContactConfig,
) -> tuple[dict, tuple[int, int, int], tuple[int, int, int]]:
    """Discovers the contact unit cell in a simplified manner.

    Parameters
    ----------
    method : str
        The method to use for contact discovery. Can be either
        'from_unit' or 'slice'.
    contact_name : str
        The name of the contact. Can be either 'left' or 'right'.
    num_orbitals : int
        The total number of orbitals in the device.
    device_config : DeviceConfig
        The configuration of the device, including its construction
        method and neighbor cell cutoff.
    contact_config : ContactConfig
        The configuration of the contact, including its direction,
        sections, and contact slice.

    Returns
    -------
    dict
        A dictionary containing the orbital indices for each periodic
        repetition in both transport and transverse directions. The keys
        are tuples of the form (transport_index, idy, idz), where
        transport_index is the index of the periodic repetition in the
        transport direction, and idy, idz are the indices of the
        periodic repetitions in the transverse directions.
    tuple[int, int, int]
        The number of periodic repetitions in the transport and
        transverse directions.
    tuple[int, int, int]
        The offset of the origin cell in the transport and transverse
        directions.

    """
    if contact_config.contact_finder_method not in ["from_unit", "slice"]:
        raise ValueError(
            f"Contact finder method '{contact_config.contact_finder_method}' is not valid. Must be 'from_unit' or 'slice'."
        )

    transport_direction = "abc".index(contact_config.transport_direction)
    transverse_repetition_grid = (
        contact_config.sections[:transport_direction]
        + contact_config.sections[transport_direction + 1 :]
    )

    if contact_config.contact_finder_method == "from_unit":
        if not device_config.construct_from_unit_cell:
            raise ValueError(
                "Contact finder method 'from_unit' requires the device to be constructed from a unit cell."
            )
        transport_repetitions = device_config.neighbor_cell_cutoff[transport_direction]

        block_size = num_orbitals // device_config.num_transport_cells
    else:
        transport_repetitions = contact_config.sections[transport_direction]
        block_size = (
            contact_config.contact_slice[1] - contact_config.contact_slice[0]
        ) // 2

    block_size_hat = (block_size // transport_repetitions) // np.prod(
        transverse_repetition_grid
    )

    if contact_name == "left":
        indices = np.arange(block_size + block_size // transport_repetitions)
        if contact_config.contact_finder_method == "slice":
            indices += contact_config.contact_slice[0]
    elif contact_name == "right":
        indices = np.arange(
            0,
            (block_size + block_size // transport_repetitions),
        )
        if contact_config.contact_finder_method == "slice":
            indices = contact_config.contact_slice[1] - indices - 1
        else:
            indices = num_orbitals - indices - 1
    else:
        raise ValueError(
            f"Contact name '{contact_name}' is not valid for 'from_unit' method. Must be 'left' or 'right'."
        )

    unit_cell_orbital_indices = {
        (i, j, k): indices[
            block_size_hat
            * (
                j
                + k * transverse_repetition_grid[0]
                + i * np.prod(transverse_repetition_grid)
            ) : block_size_hat
            * (
                j
                + k * transverse_repetition_grid[0]
                + i * np.prod(transverse_repetition_grid)
                + 1
            )
        ]
        for i in range(transport_repetitions + 1)
        for j in range(transverse_repetition_grid[0])
        for k in range(transverse_repetition_grid[1])
    }
    origin_key = (0, 0, 0)
    repetition_grid = list(contact_config.sections)
    repetition_grid[transport_direction] = transport_repetitions

    return unit_cell_orbital_indices, tuple(repetition_grid), origin_key
