# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes function to construct a Monkhorst-Pack grid of k-points."""

from dataclasses import dataclass

import ase
import numpy as np
import spglib

from qttools import NDArray, xp

# Real orbitals as functions of unit vectors, shape (num_points, 3) -> (num_points,).
# The normalization is common within a shell, so that the rotation is orthogonal.
ORBITAL_FUNCTIONS = {
    "s": lambda p: np.ones(len(p)),
    "s*": lambda p: np.ones(len(p)),
    "px": lambda p: np.sqrt(3.0) * p[:, 0],
    "py": lambda p: np.sqrt(3.0) * p[:, 1],
    "pz": lambda p: np.sqrt(3.0) * p[:, 2],
    "dxy": lambda p: np.sqrt(15.0) * p[:, 0] * p[:, 1],
    "dyz": lambda p: np.sqrt(15.0) * p[:, 1] * p[:, 2],
    "dzx": lambda p: np.sqrt(15.0) * p[:, 2] * p[:, 0],
    "dx2y2": lambda p: 0.5 * np.sqrt(15.0) * (p[:, 0] ** 2 - p[:, 1] ** 2),
    "dz2": lambda p: 0.5 * np.sqrt(5.0) * (3.0 * p[:, 2] ** 2 - 1.0),
}

# Orbitals that mix under rotations belong to the same shell.
ORBITAL_SHELL = {
    "s": "s",
    "s*": "s*",
    "px": "p",
    "py": "p",
    "pz": "p",
    "dxy": "d",
    "dyz": "d",
    "dzx": "d",
    "dx2y2": "d",
    "dz2": "d",
}


def _random_unit_vectors(num_points: int = 64, seed: int = 0) -> NDArray:
    """Draws random points on the unit sphere used to fit rotation matrices."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(num_points, 3))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


SAMPLE_DIRECTIONS = _random_unit_vectors()


@dataclass(frozen=True)
class SymmetryOperation:
    """A space-group operation, optionally combined with time reversal.

    Attributes
    ----------
    rotation : NDArray
        Rotation W in fractional coordinates (spglib), shape (3, 3).
    translation : NDArray
        Translation t in fractional coordinates, shape (3,).
    is_time_reversed : bool
        If True, the operation is combined with time reversal, so that
        k -> -W^-T k and matrices are transposed.
    """

    rotation: NDArray
    translation: NDArray
    is_time_reversed: bool

    @property
    def is_identity(self) -> bool:
        """Whether this is the plain identity operation."""
        return (
            not self.is_time_reversed
            and np.array_equal(self.rotation, np.eye(3, dtype=int))
            and np.allclose(self.translation, 0.0)
        )

    def map_kpoints(self, kpoints: NDArray) -> NDArray:
        """Maps k-points (rows, reduced coordinates) to k' = +-W^-T k."""
        mapped_kpoints = kpoints @ np.linalg.inv(self.rotation)
        return -mapped_kpoints if self.is_time_reversed else mapped_kpoints


def collect_symmetry_operations(
    structure: ase.Atoms,
    transport_axis: int = 0,
    time_reversal: bool = True,
    symprec: float = 1e-5,
) -> list[SymmetryOperation]:
    """Collects the symmetry operations that are valid for the device.

    Only operations that leave the transport axis untouched are kept.
    The identity comes first, then the other proper operations, then the
    time-reversed ones. This ordering makes the identity the preferred
    operation for the irreducible points themselves.

    Parameters
    ----------
    structure : ase.Atoms
        Structure given to spglib. Its cell and scaled positions must be
        the ones used for `fractional_positions` and `lattice_vectors`
        in `DeviceSymmetry`.
    transport_axis : int, optional
        Index of the transport axis. Default is 0.
    time_reversal : bool, optional
        Whether to add time-reversed versions of all operations.
        Default is True.
    symprec : float, optional
        Symmetry tolerance passed to spglib.

    Returns
    -------
    operations : list[SymmetryOperation]
        The filtered and ordered operations.
    """
    cell = (
        structure.cell.array,
        structure.get_scaled_positions(),
        structure.get_atomic_numbers(),
    )
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    transport_unit_vector = np.eye(3, dtype=int)[transport_axis]

    operations: list[SymmetryOperation] = []
    for rotation, translation in zip(dataset.rotations, dataset.translations):
        translation_along_transport = translation[transport_axis]
        leaves_transport_axis_alone = (
            np.array_equal(rotation[transport_axis], transport_unit_vector)
            and np.array_equal(rotation[:, transport_axis], transport_unit_vector)
            and abs(translation_along_transport - np.round(translation_along_transport))
            < 1e-8
        )
        if not leaves_transport_axis_alone:
            continue
        operations.append(SymmetryOperation(rotation, translation, False))
        if time_reversal:
            operations.append(SymmetryOperation(rotation, translation, True))

    operations.sort(key=lambda op: (op.is_time_reversed, not op.is_identity))
    return operations


def compute_orbital_rotation(
    orbital_labels: list[str],
    rotation_cartesian: NDArray,
) -> NDArray:
    """Computes the rotation matrix of the orbitals on a single atom.

    The matrix ``d`` is defined by

        phi_alpha(R^-1 r) = sum_beta phi_beta(r) d[beta, alpha],

    and is obtained by a least-squares fit of the real orbital functions
    sampled on random directions. This works for improper rotations as
    well.

    Parameters
    ----------
    orbital_labels : list[str]
        Orbital labels on the atom, in the order used by the
        Hamiltonian, e.g. ``["s", "px", "py", "pz", ...]``.
    rotation_cartesian : NDArray
        Cartesian rotation matrix R, shape (3, 3), acting on column
        vectors.

    Returns
    -------
    orbital_rotation : NDArray
        Real orthogonal matrix of shape (num_orbitals, num_orbitals).

    """
    num_orbitals = len(orbital_labels)
    orbital_rotation = np.zeros((num_orbitals, num_orbitals))
    # Row-vector convention: p @ R equals (R^-1 p) for orthogonal R.
    rotated_directions = SAMPLE_DIRECTIONS @ rotation_cartesian

    shells = dict.fromkeys(ORBITAL_SHELL[label] for label in orbital_labels)
    for shell in shells:
        shell_indices = [
            i for i, label in enumerate(orbital_labels) if ORBITAL_SHELL[label] == shell
        ]
        original_values = np.stack(
            [
                ORBITAL_FUNCTIONS[orbital_labels[i]](SAMPLE_DIRECTIONS)
                for i in shell_indices
            ],
            axis=1,
        )
        rotated_values = np.stack(
            [
                ORBITAL_FUNCTIONS[orbital_labels[i]](rotated_directions)
                for i in shell_indices
            ],
            axis=1,
        )
        shell_rotation, *__ = np.linalg.lstsq(
            original_values, rotated_values, rcond=None
        )
        identity = np.eye(len(shell_indices))
        if not np.allclose(shell_rotation.T @ shell_rotation, identity, atol=1e-10):
            raise ValueError(f"Rotation of shell '{shell}' is not orthogonal.")
        orbital_rotation[np.ix_(shell_indices, shell_indices)] = shell_rotation

    return orbital_rotation


def find_atom_permutation(
    fractional_positions: NDArray,
    lattice_vectors: NDArray,
    rotation_fractional: NDArray,
    translation_fractional: NDArray,
    periodic_axes: tuple[int, ...] = (1, 2),
    tolerance: float = 1e-6,
) -> NDArray:
    """Finds the atom permutation induced by a symmetry operation.

    Parameters
    ----------
    fractional_positions : NDArray
        Fractional positions of all device atoms, shape (num_atoms, 3).
    lattice_vectors : NDArray
        Lattice vectors as rows, shape (3, 3).
    rotation_fractional : NDArray
        Rotation W in fractional coordinates (as returned by spglib).
    translation_fractional : NDArray
        Translation t in fractional coordinates.
    periodic_axes : tuple[int, ...], optional
        Axes along which positions are identified modulo a lattice
        vector. The transport axis must not be included. Default is
        (1, 2).
    tolerance : float, optional
        Maximum Cartesian distance for two atoms to count as identical.

    Returns
    -------
    permutation : NDArray
        Integer array of shape (num_atoms,). Atom ``a`` is mapped onto
        atom ``permutation[a]``.

    """
    num_atoms = len(fractional_positions)
    mapped_positions = (
        fractional_positions @ rotation_fractional.T + translation_fractional
    )
    permutation = np.empty(num_atoms, dtype=int)
    axes = list(periodic_axes)

    for atom_ind in range(num_atoms):
        difference = fractional_positions - mapped_positions[atom_ind]
        difference[:, axes] -= np.round(difference[:, axes])
        distances = np.linalg.norm(difference @ lattice_vectors, axis=1)
        matches = np.where(distances < tolerance)[0]
        if len(matches) != 1:
            raise ValueError("Operation is not a symmetry of this structure.")
        permutation[atom_ind] = matches[0]

    return permutation


def compute_gauge_phases(
    kpoint_fractional: NDArray,
    fractional_positions: NDArray,
    num_orbitals: int,
) -> NDArray:
    """Computes the diagonal cell-gauge to atom-gauge phase factors.

    Parameters
    ----------
    kpoint_fractional : NDArray
        k-point in reduced reciprocal coordinates, shape (3,). The
        transport component must be zero.
    fractional_positions : NDArray
        Fractional atom positions exactly as used when assembling H(k).
        They must not be wrapped into the unit cell.
    num_orbitals : int
        Number of orbitals per atom.

    Returns
    -------
    phases : NDArray
        Complex vector of shape (num_atoms * num_orbitals,).

    """
    atom_phases = np.exp(2j * np.pi * (fractional_positions @ kpoint_fractional))
    return np.repeat(atom_phases, num_orbitals)


class DeviceSymmetry:
    """Applies symmetry operations to device matrices in cell gauge.

    Parameters
    ----------
    fractional_positions : NDArray
        Unwrapped fractional positions of all device atoms, shape
        (num_atoms, 3), exactly as used to assemble H(k).
    lattice_vectors : NDArray
        Lattice vectors as rows, shape (3, 3).
    orbital_labels : list[str]
        Orbital labels per atom, in Hamiltonian order.
    periodic_axes : tuple[int, ...], optional
        Periodic (transverse) axes. Default is (1, 2).
    """

    def __init__(
        self,
        fractional_positions: NDArray,
        lattice_vectors: NDArray,
        orbital_labels: list[str],
        periodic_axes: tuple[int, ...] = (1, 2),
    ) -> None:
        self.fractional_positions = fractional_positions
        self.lattice_vectors = lattice_vectors
        self.orbital_labels = orbital_labels
        self.periodic_axes = periodic_axes
        self.num_atoms = len(fractional_positions)
        self.num_orbitals = len(orbital_labels)
        self._cache: dict[tuple[bytes, bytes], tuple[NDArray, NDArray]] = {}

    def _atom_permutation_and_orbital_rotation(
        self, operation: SymmetryOperation
    ) -> tuple[NDArray, NDArray]:
        """Returns the cached atom permutation and orbital rotation."""
        key = (
            operation.rotation.tobytes(),
            (np.round(operation.translation, 8) + 0.0).tobytes(),
        )
        if key not in self._cache:
            rotation_cartesian = (
                self.lattice_vectors.T
                @ operation.rotation
                @ np.linalg.inv(self.lattice_vectors.T)
            )
            permutation = find_atom_permutation(
                self.fractional_positions,
                self.lattice_vectors,
                operation.rotation,
                operation.translation,
                self.periodic_axes,
            )
            orbital_rotation = compute_orbital_rotation(
                self.orbital_labels, rotation_cartesian
            )
            self._cache[key] = (permutation, orbital_rotation)
        return self._cache[key]

    def transform(
        self,
        matrices: NDArray,
        operation: SymmetryOperation,
        kpoint: NDArray,
    ) -> NDArray:
        """Maps device matrices from ``kpoint`` to the equivalent k-point.

        Implements M(k') = D M(k) D^dagger with D = Phi(k') D0 Phi(k)^dagger,
        and for time-reversed operations M(-k') = D^* M(k)^T D^T.

        Parameters
        ----------
        matrices : NDArray
            Matrices at ``kpoint``, shape (..., n, n) with
            n = num_atoms * num_orbitals. Works for H, S, G^R, G^<,
            G^>, and the self-energies, in cell gauge.
        operation : SymmetryOperation
            Operation mapping ``kpoint`` onto the target k-point.
        kpoint : NDArray
            Source k-point in reduced coordinates, shape (3,). Use the
            same (unwrapped) value that was used to build the matrices.

        Returns
        -------
        transformed_matrices : NDArray
            Matrices at the target k-point, shape (..., n, n).
        """
        permutation, orbital_rotation = self._atom_permutation_and_orbital_rotation(
            operation
        )
        # k' = W^-T k, deliberately NOT wrapped into the grid.
        rotated_kpoint = np.linalg.inv(operation.rotation).T @ kpoint
        source_phases = compute_gauge_phases(
            kpoint, self.fractional_positions, self.num_orbitals
        )
        target_phases = compute_gauge_phases(
            rotated_kpoint, self.fractional_positions, self.num_orbitals
        )

        if operation.is_time_reversed:
            # M(-k') = conj(Phi') D0 [Phi M^T Phi^*] D0^T Phi'
            matrices = np.swapaxes(matrices, -1, -2)
            source_phases = source_phases.conj()
            target_phases = target_phases.conj()

        scaled = source_phases.conj()[:, None] * matrices * source_phases[None, :]

        # Rotate the orbitals on each atom: d @ block @ d^T.
        batch_shape = scaled.shape[:-2]
        blocks = scaled.reshape(
            *batch_shape,
            self.num_atoms,
            self.num_orbitals,
            self.num_atoms,
            self.num_orbitals,
        )
        rotated = np.einsum(
            "pi,...aibj,qj->...apbq",
            orbital_rotation,
            blocks,
            orbital_rotation,
            optimize=True,
        )

        # Atom a moves to permutation[a]: new[permutation[a]] = old[a].
        inverse_permutation = np.argsort(permutation)
        permuted = rotated[..., inverse_permutation, :, :, :][
            ..., inverse_permutation, :
        ]

        permuted = permuted.reshape(*batch_shape, *scaled.shape[-2:])
        return target_phases[:, None] * permuted * target_phases.conj()[None, :]


def _monkhorst_pack(
    size: tuple[int] | NDArray,
    shift: tuple[float] | NDArray | None = None,
    gamma_centered: bool = False,
    symmetry_reduction: bool = False,
    structure: ase.Atoms | None = None,
    time_reversal: bool = True,
    operations: list[SymmetryOperation] | None = None,
) -> NDArray:
    """Constructs a Monkhorst-Pack grid of k-points.

    Parameters
    ----------
    size : tuple[int] | NDArray
        Grid dimensions specifying the number of k-points along each
        reciprocal lattice direction.
    shift : tuple[float] | NDArray, optional
        Shift of the grid in each direction. If None, no shift is
        applied. Must not be provided if `gamma_centered` is True.
        Default is None.
    gamma_centered : bool, optional
        If True, the grid is centered around the Gamma point. Default is
        False.
    symmetry_reduction : bool, optional
        If True, the kpoint grid is reduced to the irreducible part of
        the Brillouin zone using the symmetry of the provided structure.
        If False, the full Brillouin zone is sampled. Default is False.
    structure : ase.Atoms, optional
        The atomic structure of the system. This has to be provided if
        `irreducible` is True, as the irreducible k-points depend on the
        symmetry of the structure. Default is None.
    time_reversal : bool, optional
        If True, time-reversal symmetry is considered when determining
        the irreducible k-points. Default is True.
    operations : list[SymmetryOperation], optional
        The symmetry operations to use instead of the ones determined
        from the structure. The identity must be the first one. Default
        is None.

    Returns
    -------
    kpoints : NDArray
        Array of k-points with shape (nx*ny*nz, 3). Each row contains
        the (kx, ky, kz) coordinates of a k-point in reduced units.
        These have to be multiplied by (2 * pi / a) to get the actual
        k-points in reciprocal space, where a is the lattice constant.
    reduce_mapping : NDArray, optional
        If `irreducible` is True, this array maps each k-point in the
        full grid to its corresponding irreducible k-point. It has shape
        (nx*ny*nz,) and contains the indices of the irreducible
        k-points.
    expand_mapping : NDArray, optional
        If `irreducible` is True, this array maps each irreducible
        k-point to its corresponding k-points in the full grid. It has
        shape (n_irreducible,) and contains the indices of the k-points
        in the full grid that correspond to each irreducible k-point.
    multiplicities : NDArray, optional
        If `irreducible` is True, this array contains the multiplicities
        of the irreducible k-points. It has shape (n_irreducible,) and
        indicates how many k-points in the full grid correspond to each
        irreducible k-point.

    """
    size = np.asarray(size)

    if shift is None:
        shift = (0.0, 0.0, 0.0)
    shift = np.asarray(shift, dtype=float)

    if gamma_centered and np.any(shift):
        raise ValueError("A shift must not be provided when `gamma_centered` is True.")

    n = np.moveaxis(np.indices(size), 0, -1).reshape(-1, len(size))
    offset = np.zeros(len(size)) if gamma_centered else 0.5 - size / 2
    kpoints = (n + offset) / size + shift
    if not symmetry_reduction:
        return kpoints

    if operations is None:
        if structure is None:
            raise ValueError(
                "A `structure` is required when `irreducible` is set to True."
            )
        operations = collect_symmetry_operations(structure, 0, time_reversal)

    # image_maps[o, i] is the flat grid index of operation o applied to k-point i.
    grid_operations: list[SymmetryOperation] = []
    image_maps: list[NDArray] = []
    for operation in operations:
        mapped_kpoints = operation.map_kpoints(kpoints)
        grid_indices = (mapped_kpoints - shift) * size - offset
        rounded_indices = np.round(grid_indices)
        if not np.allclose(grid_indices, rounded_indices):
            continue  # This operation does not preserve the grid.
        image_maps.append(
            np.ravel_multi_index(tuple((rounded_indices.astype(int) % size).T), size)
        )
        grid_operations.append(operation)

    image_maps = np.stack(image_maps)
    mapping = np.min(image_maps, axis=0)  # Identity is included, so mapping <= arange.

    reduce_mapping, expand_mapping, multiplicities = np.unique(
        mapping, return_inverse=True, return_counts=True
    )

    # For every full point, the first operation that maps its representative onto it.
    operation_indices = np.array(
        [
            np.flatnonzero(image_maps[:, mapping[full_ind]] == full_ind)[0]
            for full_ind in range(len(kpoints))
        ]
    )

    return (
        kpoints,
        reduce_mapping,
        expand_mapping,
        multiplicities,
        grid_operations,
        operation_indices,
    )


class KGrid:
    """Class to handle Monkhorst-Pack grids and symmetries.

    This class handles reductions from full grids to irreducible grids
    and expansions from irreducible grids to full grids, as well as
    integrating quantities over the Brillouin zone using the
    multiplicities of the irreducible k-points.

    Parameters
    ----------
    size : tuple[int] | NDArray
        Grid dimensions specifying the number of k-points along each
        reciprocal lattice direction.
    shift : tuple[float] | NDArray, optional
        Shift of the grid in each direction. If None, no shift is
        applied. Must not be provided if `gamma_centered` is True.
        Default is None.
    gamma_centered : bool, optional
        If True, the grid is centered around the Gamma point. Default is
        False.
    symmetry_reduction : bool, optional
        If enabled, the kpoint grid can be reduced to the irreducible
        part of the Brillouin zone using the symmetry of the provided
        structure. If disabled, the full Brillouin is always sampled
        explicitely and the `expand` and `reduce` methods do nothing.
        Default is False.
    structure : ase.Atoms, optional
        The atomic structure of the system. This has to be provided if
        `irreducible` is True, as the irreducible k-points depend on the
        symmetry of the structure. Default is None.
    time_reversal : bool, optional
        If True, time-reversal symmetry is considered when determining
        the irreducible k-points. Default is True.
    operations : list[SymmetryOperation], optional
        The symmetry operations to use instead of the ones determined
        from the structure, e.g. if only some of them are supported. The
        identity must be the first one. Default is None.

    """

    def __init__(
        self,
        size: tuple[int] | NDArray,
        shift: tuple[float] | NDArray | None = None,
        gamma_centered: bool = False,
        symmetry_reduction: bool = False,
        structure: ase.Atoms | None = None,
        time_reversal: bool = True,
        operations: list[SymmetryOperation] | None = None,
    ) -> None:
        """Initializes the k-points."""
        if symmetry_reduction:
            (
                full_points,
                reduce_mapping,
                expand_mapping,
                multiplicities,
                grid_operations,
                operation_indices,
            ) = _monkhorst_pack(
                size=size,
                shift=shift,
                gamma_centered=gamma_centered,
                symmetry_reduction=True,
                structure=structure,
                time_reversal=time_reversal,
                operations=operations,
            )
            irreducible_points = full_points[reduce_mapping]
        else:
            full_points = _monkhorst_pack(
                size=size,
                shift=shift,
                gamma_centered=gamma_centered,
                symmetry_reduction=False,
                structure=structure,
                time_reversal=time_reversal,
            )
            reduce_mapping = expand_mapping = irreducible_points = grid_operations = (
                operation_indices
            ) = None
            multiplicities = np.ones(len(full_points), dtype=int)

        self.full_points = full_points
        self.irreducible_points = irreducible_points
        self._reduce_mapping = reduce_mapping
        self._expand_mapping = expand_mapping
        self._multiplicities = multiplicities
        self._operations = grid_operations
        self._operation_indices = operation_indices

        self.num_full_points = len(full_points)
        self.num_irreducible_points = (
            len(irreducible_points)
            if irreducible_points is not None
            else len(full_points)
        )

        self.size = tuple(size)
        self.shift = tuple(shift) if shift is not None else (0.0,) * len(size)
        self.gamma_centered = gamma_centered
        self.symmetry_reduction = symmetry_reduction
        self.time_reversal = time_reversal

    @property
    def operations(self) -> list[SymmetryOperation] | None:
        """Returns the symmetry operations that preserve the grid.

        This is None if symmetry reduction is disabled.

        """
        return self._operations

    @property
    def points(self) -> NDArray:
        """Returns the k-points in the grid.

        This is either the irreducible k-points if symmetry reduction is
        enabled, or the full k-points in the grid if symmetry reduction
        is disabled.

        """
        return self.irreducible_points if self.symmetry_reduction else self.full_points

    @property
    def num_points(self) -> int:
        """Returns the number of k-points in the grid.

        This is either the number of irreducible k-points if symmetry
        reduction is enabled, or the total number of k-points in the
        full grid if symmetry reduction is disabled.

        """
        return (
            self.num_irreducible_points
            if self.symmetry_reduction
            else self.num_full_points
        )

    def reduce(self, values: NDArray, axis: int = 1) -> NDArray:
        """Reduces values from the full grid to the irreducible points.

        The value array can be multi-dimensional. It will be reduced
        along the specified axis.

        This does nothing if the irreducible points are switched off.

        Parameters
        ----------
        values : NDArray
            Values defined on the full k-point grid. The requested axis
            must have the same length as the number of k-points in the
            full grid.
        axis : int, optional
            The axis along which the k-points are defined. If None, the
            input array is flattened before indexing. Default is 1,
            which is the default axis for k-points in the quatrex data
            structures.

        Returns
        -------
        reduced_values : NDArray
            Values reduced to the irreducible k-points.

        """
        if not self.symmetry_reduction:
            return values

        return xp.take(values, self._reduce_mapping, axis=axis)

    def expand(self, values: NDArray, axis: int = 1) -> NDArray:
        """Expands values from the irreducible points to the full grid.

        The values can be multi-dimensional. They are then expanded
        along the specified axis. The resulting array will have
        fftn-compatible shape and ordering in the k-dimensions.

        This does nothing if the irreducible points are switched off.

        Parameters
        ----------
        values : NDArray
            Values defined on the irreducible k-points. The requested
            axis must have the same length as the number of irreducible
            k-points.
        axis : int, optional
            The axis along which the k-points are defined. If None, the
            input array is flattened before indexing. Default is 1,
            which is the default axis for k-points in the quatrex data
            structures.

        Returns
        -------
        expanded_values : NDArray
            Values expanded to the full grid. The resulting array will have
            fftn-compatible shape and ordering in the k-dimensions.
        """
        if not self.symmetry_reduction:
            return values

        return xp.take(values, self._expand_mapping, axis=axis)

    def integrate(self, values: NDArray, axis: int = 1) -> NDArray:
        """Integrates values over the Brillouin zone.

        Due to the periodicity of the Brillouin zone, a zero-order
        integration is performed by summing the values over the
        irreducible k-points, weighted by their multiplicities.

        Parameters
        ----------
        values : NDArray
            Values defined on the irreducible k-points. The requested
            axis must have the same length as the number of irreducible
            k-points if `symmetry_reduction` is True, or the same length
            as the number of k-points in the full grid if
            `symmetry_reduction` is False.
        axis : int, optional
            The axis along which the k-points are defined. Flattened
            array is used if None. Default is 1, which is the default
            axis for k-points in the quatrex data structures.

        Returns
        -------
        integral : NDArray
            The integral of the values over the Brillouin zone.

        """

        if not self.symmetry_reduction:
            return xp.average(values, axis=axis)

        return xp.average(values, axis=axis, weights=self._multiplicities)

    def expand_matrices(
        self,
        matrices: NDArray,
        symmetry: DeviceSymmetry,
        axis: int = 1,
    ) -> NDArray:
        """Expands device matrices from the irreducible points to the full grid.

        Unlike `expand`, this applies the orbital rotation, atom
        permutation, gauge phases, and transposition for time-reversed
        operations. Use it for G^R, G^<, G^>, Sigma, P, W, etc. Use
        `expand` only for scalars such as traced quantities.

        Parameters
        ----------
        matrices : NDArray
            Matrices on the irreducible k-points. The last two axes are
            the device matrix, ``axis`` is the k-point axis (not one of
            the last two).
        symmetry : DeviceSymmetry
            Provides the atom/orbital structure of the device.
        axis : int, optional
            The k-point axis. Default is 1.

        Returns
        -------
        expanded_matrices : NDArray
            Matrices on the full grid, in the same ordering as
            `full_points` (fftn-compatible).
        """
        if not self.symmetry_reduction:
            return matrices

        expanded_slices = []
        for full_ind in range(self.num_full_points):
            irreducible_ind = self._expand_mapping[full_ind]
            operation = self._operations[self._operation_indices[full_ind]]
            source_matrices = np.take(matrices, irreducible_ind, axis=axis)
            if operation.is_identity:
                expanded_slices.append(source_matrices)
                continue
            expanded_slices.append(
                symmetry.transform(
                    source_matrices,
                    operation,
                    self.irreducible_points[irreducible_ind],
                )
            )
        return np.stack(expanded_slices, axis=axis)
