# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the symmetry-packed real-space representation of operators.

A matrix on a transverse k-point grid is equivalent to the matrix in
real space,

    M_ij(R) = 1/N_k sum_k exp(-2 pi i k R) M_ij(k),

where R runs over the N_k cells of the Born-von-Karman supercell.
Elementwise products on the full Brillouin zone, i.e. the k-point
correlations/convolutions in P = GG and Sigma = GW, become pointwise in
(i, j, R). All symmetry-related triples (i, j, R) carry the same
information, so that only one representative per orbit is needed.

"""

from collections.abc import Callable
from dataclasses import dataclass

import ase
import numpy as np

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.greens_function_solver.solver import BackSubstitutionContext
from qttools.utils.gpu_utils import get_host
from qttools.utils.mpi_utils import get_section_sizes
from quatrex.grid.kpoints import (
    DeviceSymmetry,
    KGrid,
    SymmetryOperation,
    collect_symmetry_operations,
    compute_orbital_rotation,
    find_atom_permutation,
)

# The factor eta in M_ji(-R) = eta * conj(M_ij(R)).
HERMITICITY_FACTORS = {"hermitian": 1, "skew-hermitian": -1}


@dataclass(frozen=True)
class OrbitalOperation:
    """A symmetry operation acting as a signed permutation of orbitals.

    The orbital `a` in the home cell is mapped onto `signs[a]` times the
    orbital `targets[a]` in the cell `shifts[a]`.

    Attributes
    ----------
    rotation : NDArray
        Rotation W in fractional coordinates, shape (3, 3).
    translation : NDArray
        Translation t in fractional coordinates, shape (3,).
    targets : NDArray
        Target orbital of each orbital, shape (num_orbitals,).
    signs : NDArray
        Sign (+1 or -1) of each orbital, shape (num_orbitals,).
    shifts : NDArray
        Cell of the target orbital, shape (num_orbitals, 3).

    """

    rotation: NDArray
    translation: NDArray
    targets: NDArray
    signs: NDArray
    shifts: NDArray

    @property
    def inverse_targets(self) -> NDArray:
        """The orbital that is mapped onto each orbital."""
        return np.argsort(self.targets)


def _operation_key(operation: SymmetryOperation | OrbitalOperation) -> tuple:
    """Hashable identifier of an operation, modulo lattice translations."""
    translation = np.round(np.asarray(operation.translation), 6) % 1.0
    translation = np.round(translation, 6) % 1.0
    return (
        tuple(np.asarray(operation.rotation, dtype=int).ravel().tolist()),
        tuple((translation + 0.0).tolist()),
    )


def create_orbital_operation(
    symmetry: DeviceSymmetry,
    operation: SymmetryOperation,
    tolerance: float = 1e-8,
) -> OrbitalOperation:
    """Creates the signed orbital permutation of a symmetry operation.

    Parameters
    ----------
    symmetry : DeviceSymmetry
        The atom and orbital structure of the device.
    operation : SymmetryOperation
        The symmetry operation.
    tolerance : float, optional
        Tolerance for recognizing a signed permutation.

    Returns
    -------
    orbital_operation : OrbitalOperation
        The signed permutation.

    Raises
    ------
    ValueError
        If the operation includes time reversal (which does not hold for
        lesser/greater quantities out of equilibrium) or if it mixes
        orbitals. In the latter case, neither P_ij = G_ij G_ji nor
        Sigma_ij = G_ij W_ij transform covariantly.

    """
    if operation.is_time_reversed:
        raise ValueError("Time-reversed operations are not supported.")

    permutation, orbital_rotation = symmetry._atom_permutation_and_orbital_rotation(
        operation
    )
    local_targets, local_signs = _get_signed_permutation(orbital_rotation, tolerance)
    num_orbitals = symmetry.num_orbitals

    positions = symmetry.fractional_positions
    atom_shifts = _get_atom_shifts(positions, operation, permutation)

    return OrbitalOperation(
        rotation=np.asarray(operation.rotation, dtype=int),
        translation=np.asarray(operation.translation, dtype=float),
        targets=(permutation[:, None] * num_orbitals + local_targets).ravel(),
        signs=np.tile(local_signs, symmetry.num_atoms).astype(int),
        shifts=np.repeat(atom_shifts, num_orbitals, axis=0),
    )


def _get_signed_permutation(
    orbital_rotation: NDArray, tolerance: float
) -> tuple[NDArray, NDArray]:
    """Gets the targets and signs of the orbitals on a single atom."""
    is_unit = np.abs(np.abs(orbital_rotation) - 1.0) < tolerance
    is_zero = np.abs(orbital_rotation) < tolerance
    if not (
        np.all(is_unit | is_zero)
        and np.all(is_unit.sum(axis=0) == 1)
        and np.all(is_unit.sum(axis=1) == 1)
    ):
        raise ValueError("Operation mixes orbitals, it is not a signed permutation.")

    local_targets = np.argmax(is_unit, axis=0)
    local_signs = np.sign(orbital_rotation[local_targets, np.arange(len(is_unit))])
    return local_targets, local_signs.astype(int)


def _get_atom_shifts(
    positions: NDArray, operation: SymmetryOperation, permutation: NDArray
) -> NDArray:
    """Gets the cells in which the images of the atoms are."""
    return np.round(
        positions @ np.asarray(operation.rotation).T
        + operation.translation
        - positions[permutation]
    ).astype(int)


def find_orbital_operations(
    structure: ase.Atoms,
    orbital_labels: dict[str, list[str]],
    transport_axis: int,
    tolerance: float = 1e-4,
) -> tuple[list[SymmetryOperation], list[OrbitalOperation]]:
    """Finds the signed permutations of the orbitals in a structure.

    These are the symmetry operations that leave the transport axis
    untouched, without time reversal, and that do not mix orbitals.

    Parameters
    ----------
    structure : ase.Atoms
        The structure.
    orbital_labels : dict[str, list[str]]
        The orbitals of each species.
    transport_axis : int
        Index of the transport axis.
    tolerance : float, optional
        Maximum Cartesian distance for two atoms to count as identical.

    Returns
    -------
    symmetry_operations : list[SymmetryOperation]
        The symmetry operations. The identity is the first one.
    orbital_operations : list[OrbitalOperation]
        The corresponding signed permutations of the orbitals.

    """
    species = structure.get_chemical_symbols()
    missing = sorted(set(species) - set(orbital_labels))
    if missing:
        raise ValueError(f"The orbital labels of {missing} are missing.")

    labels = [tuple(orbital_labels[s]) for s in species]
    num_orbitals = np.array([len(label) for label in labels])
    offsets = np.hstack(([0], np.cumsum(num_orbitals)))[:-1]

    # NOTE: These must not be wrapped, the matrices are in cell gauge.
    positions = structure.get_scaled_positions(wrap=False)
    lattice_vectors = structure.cell.array
    periodic_axes = tuple(axis for axis in range(3) if axis != transport_axis)

    symmetry_operations, orbital_operations = [], []
    for operation in collect_symmetry_operations(
        structure, transport_axis, time_reversal=False
    ):
        rotation = (
            lattice_vectors.T @ operation.rotation @ np.linalg.inv(lattice_vectors.T)
        )
        try:
            permutation = find_atom_permutation(
                positions,
                lattice_vectors,
                operation.rotation,
                operation.translation,
                periodic_axes,
                tolerance,
            )
            local = {
                label: _get_signed_permutation(
                    compute_orbital_rotation(list(label), rotation), 1e-8
                )
                for label in set(labels)
            }
        except ValueError:
            continue

        if any(labels[a] != labels[b] for a, b in enumerate(permutation)):
            continue

        symmetry_operations.append(operation)
        orbital_operations.append(
            OrbitalOperation(
                rotation=np.asarray(operation.rotation, dtype=int),
                translation=np.asarray(operation.translation, dtype=float),
                targets=np.concatenate(
                    [
                        offsets[b] + local[labels[a]][0]
                        for a, b in enumerate(permutation)
                    ]
                ),
                signs=np.concatenate([local[label][1] for label in labels]),
                shifts=np.repeat(
                    _get_atom_shifts(positions, operation, permutation),
                    num_orbitals,
                    axis=0,
                ),
            )
        )

    return symmetry_operations, orbital_operations


def repeat_orbital_operation(
    operation: OrbitalOperation, num_repetitions: int
) -> OrbitalOperation:
    """Repeats the operation of a unit cell along transport direction.

    Parameters
    ----------
    operation : OrbitalOperation
        The operation, which has to leave the transport axis untouched.
    num_repetitions : int
        The number of unit cells.

    Returns
    -------
    repeated_operation : OrbitalOperation
        The operation of the device.

    """
    num_orbitals = len(operation.targets)
    offsets = np.repeat(np.arange(num_repetitions) * num_orbitals, num_orbitals)
    return OrbitalOperation(
        rotation=operation.rotation,
        translation=operation.translation,
        targets=np.tile(operation.targets, num_repetitions) + offsets,
        signs=np.tile(operation.signs, num_repetitions),
        shifts=np.tile(operation.shifts, (num_repetitions, 1)),
    )


def measure_symmetry_breaking(
    data: NDArray,
    rows: NDArray,
    cols: NDArray,
    kpoints: NDArray,
    operations: list[OrbitalOperation],
    signed: bool,
) -> float:
    """Measures how much a Hermitian matrix breaks the symmetry.

    Parameters
    ----------
    data : NDArray
        The elements at the image of a single k-point under each of the
        operations, shape (num_operations, nnz).
    rows : NDArray
        The row indices of the elements. Only the elements on and above
        the diagonal are needed.
    cols : NDArray
        The column indices of the elements.
    kpoints : NDArray
        The images of the k-point, shape (num_operations, 3). The first
        one has to be the k-point itself.
    operations : list[OrbitalOperation]
        The operations. The identity has to be the first one.
    signed : bool
        Whether the matrix transforms with the signs of the orbitals.

    Returns
    -------
    deviation : float
        The largest absolute deviation.

    """
    data, rows, cols = get_host(data), get_host(rows), get_host(cols)
    rows, cols = rows.astype(np.int64), cols.astype(np.int64)
    num_orbitals = len(operations[0].targets)

    upper = rows <= cols
    keys = rows[upper] * num_orbitals + cols[upper]
    order = np.argsort(keys)
    keys, data = keys[order], data[:, upper][:, order]
    rows, cols = np.divmod(keys, num_orbitals)

    deviation = 0.0
    for values, kpoint, operation in zip(data, kpoints, operations):
        image_rows, image_cols = operation.targets[rows], operation.targets[cols]
        shifts = operation.shifts[cols] - operation.shifts[rows]
        expected = data[0] * np.exp(2j * np.pi * (shifts @ kpoint))
        if signed:
            expected = expected * operation.signs[rows] * operation.signs[cols]

        # The images in the lower triangle are the conjugates.
        flipped = image_rows > image_cols
        image_keys = np.where(
            flipped,
            image_cols * num_orbitals + image_rows,
            image_rows * num_orbitals + image_cols,
        )
        inds = np.minimum(np.searchsorted(keys, image_keys), len(keys) - 1)
        images = np.where(keys[inds] == image_keys, values[inds], 0.0)
        images = np.where(flipped, images.conj(), images)
        deviation = max(deviation, float(np.abs(images - expected).max(initial=0.0)))

    return deviation


def filter_packable_operations(
    symmetry: DeviceSymmetry, operations: list[SymmetryOperation]
) -> list[SymmetryOperation]:
    """Keeps the operations that are signed permutations of the orbitals.

    Parameters
    ----------
    symmetry : DeviceSymmetry
        The atom and orbital structure of the device.
    operations : list[SymmetryOperation]
        The candidate operations.

    Returns
    -------
    packable_operations : list[SymmetryOperation]
        Operations without time reversal that do not mix orbitals.

    """
    packable_operations = []
    for operation in operations:
        try:
            create_orbital_operation(symmetry, operation)
        except ValueError:
            continue
        packable_operations.append(operation)
    return packable_operations


def _check_group(operations: list[OrbitalOperation]) -> None:
    """Checks that the operations are closed under multiplication."""
    elements = {
        (op.rotation.tobytes(), np.asarray(op.targets, dtype=np.int64).tobytes())
        for op in operations
    }
    for a in operations:
        for b in operations:
            product = (
                (a.rotation @ b.rotation).tobytes(),
                np.asarray(a.targets[b.targets], dtype=np.int64).tobytes(),
            )
            if product not in elements:
                raise ValueError("The operations do not form a group.")


def _grid_twist(grid: KGrid) -> NDArray:
    """The signs in M(R + N e_x) = twist_x M(R) due to the grid offset."""
    phases = np.exp(-2j * np.pi * grid.full_points[0] * np.asarray(grid.size))
    twist = np.round(phases.real).astype(int)
    if not np.allclose(phases, twist):
        raise ValueError("Only grids with zero or half-step offset are supported.")
    return twist


class OrbitPacking:
    r"""Packs matrices into symmetry-unique elements (i, j, R).

    Packing maps matrices on the irreducible k-points to the real-space
    representatives. All steps are local in energy, such that this can
    be done in stack distribution. The packed matrices have no k-point
    axis and their last axis can be distributed arbitrarily: Products
    P(t) = G(t) G(t) and Sigma(t) = G(t) W(t) only need a single
    representative t.

    The electronic quantities (G and Sigma) are `signed`: they transform
    with the signs of the orbitals and inherit the offset of the k-point
    grid. The interactions (P and W) are not `signed`. This is because
    the signs cancel in P_ij = G_ij G_ji.

    Representatives for which the signed quantities vanish by symmetry
    are dropped. P also vanishes there. W does not, but it is only ever
    needed as a factor of G, so W should not be unpacked again.

    Parameters
    ----------
    rows : NDArray
        Row indices of the sparsity pattern. This can be the full
        pattern or only its upper triangular part.
    cols : NDArray
        Column indices of the sparsity pattern.
    operations : list[OrbitalOperation]
        The symmetry group. This has to contain all operations used by
        the grids, and all operations have to map both grids onto
        themselves.
    kgrid : KGrid
        The grid of the signed quantities.
    qgrid : KGrid
        The grid of the unsigned quantities. Needs to be gamma-centered.

    """

    def __init__(
        self,
        rows: NDArray,
        cols: NDArray,
        operations: list[OrbitalOperation],
        kgrid: KGrid,
        qgrid: KGrid,
    ) -> None:
        """Builds the orbit table."""
        self.rows = np.asarray(get_host(rows)).astype(np.int64)
        self.cols = np.asarray(get_host(cols)).astype(np.int64)
        self.operations = operations
        self.kgrid = kgrid
        self.qgrid = qgrid

        if tuple(kgrid.size) != tuple(qgrid.size):
            raise ValueError("The k-point and q-point grids must have the same size.")
        self.size = np.asarray(kgrid.size, dtype=int)
        self.num_cells = int(np.prod(self.size))
        self.cells = np.moveaxis(np.indices(self.size), 0, -1).reshape(-1, 3)

        self._twist = _grid_twist(kgrid)
        if np.any(_grid_twist(qgrid) != 1):
            raise ValueError("The q-point grid must be gamma-centered.")

        _check_group(operations)
        self._check_operations()
        self._set_pairs()
        self._build_table()

        self._star_cache: dict[int, tuple] = {}
        self._block_cache: dict[tuple, NDArray] = {}

    def _check_operations(self) -> None:
        """Checks the operations against the grids."""
        for grid in (self.kgrid, self.qgrid):
            for operation in self.operations:
                points = grid.full_points @ np.linalg.inv(operation.rotation)
                steps = (points - grid.full_points[0]) * self.size
                if not np.allclose(steps, np.round(steps)):
                    raise ValueError("Operation does not preserve the k-point grid.")

        self._operation_lookup = {
            _operation_key(op): i for i, op in enumerate(self.operations)
        }
        for grid in (self.kgrid, self.qgrid):
            for operation in grid.operations or []:
                if (
                    operation.is_time_reversed
                    or _operation_key(operation) not in self._operation_lookup
                ):
                    raise ValueError("Grid uses an operation outside of the group.")

    def _set_pairs(self) -> None:
        """Sets the unique (i, j) pairs with i <= j."""
        self._num_orbitals = len(self.operations[0].targets)
        upper = np.minimum(self.rows, self.cols)
        lower = np.maximum(self.rows, self.cols)
        # Every stored element belongs to a pair. The elements below the
        # diagonal are the conjugates of the pair.
        self._pair_keys, self._element_to_pair = np.unique(
            upper * self._num_orbitals + lower, return_inverse=True
        )
        self._element_is_lower = self.rows > self.cols
        self._pair_rows, self._pair_cols = np.divmod(
            self._pair_keys, self._num_orbitals
        )
        # An element from which to get the pair. Prefers the upper one.
        order = np.argsort(self._element_is_lower, kind="stable")[::-1]
        self._pair_to_element = np.empty(len(self._pair_keys), dtype=np.int64)
        self._pair_to_element[self._element_to_pair[order]] = order

    def _find_pairs(self, rows: NDArray, cols: NDArray) -> NDArray:
        """Finds the pair index of the given rows <= cols."""
        keys = rows * self._num_orbitals + cols
        inds = np.searchsorted(self._pair_keys, keys)
        inds = np.minimum(inds, len(self._pair_keys) - 1)
        missing = self._pair_keys[inds] != keys
        if np.any(missing):
            first = np.flatnonzero(missing)[0]
            raise ValueError(
                "The sparsity pattern is not closed under the symmetry operations. "
                f"{missing.sum()} elements are missing, e.g. "
                f"({rows[first]}, {cols[first]})."
            )
        return inds

    def _wrap_cells(self, cells: NDArray) -> tuple[NDArray, NDArray]:
        """Wraps cells into the supercell.

        Returns the flat index of the wrapped cells and the sign picked
        up by the signed quantities.

        """
        windings = np.floor_divide(cells, self.size)
        wrapped = cells - windings * self.size
        inds = np.ravel_multi_index(tuple(np.moveaxis(wrapped, -1, 0)), self.size)
        signs = np.prod(np.where(windings % 2 == 0, 1, self._twist), axis=-1)
        return inds, signs

    def _build_table(self) -> None:
        """Finds the representative of each (i, j, R)."""
        num_pairs = len(self._pair_keys)
        shape = (num_pairs, self.num_cells)
        pair_inds = np.arange(num_pairs, dtype=np.int64)
        is_diagonal = self._pair_rows == self._pair_cols

        representative = np.full(shape, np.iinfo(np.int64).max, dtype=np.int64)
        sign = np.ones(shape, dtype=np.int8)
        conjugate = np.zeros(shape, dtype=bool)
        is_zero = np.zeros(shape, dtype=bool)

        def update(cell_ind, selected, images, image_signs, image_conjugates):
            """Updates the representatives with the given images."""
            own = selected * self.num_cells + cell_ind
            # Stabilizers that flip the sign enforce a zero.
            is_zero[selected, cell_ind] |= (
                (images == own) & (image_signs < 0) & ~image_conjugates
            )
            better = images < representative[selected, cell_ind]
            inds = selected[better]
            representative[inds, cell_ind] = images[better]
            sign[inds, cell_ind] = image_signs[better]
            conjugate[inds, cell_ind] = image_conjugates[better]

        for operation in self.operations:
            image_rows = operation.targets[self._pair_rows]
            image_cols = operation.targets[self._pair_cols]
            orbital_signs = (
                operation.signs[self._pair_rows] * operation.signs[self._pair_cols]
            )
            relative_shifts = (
                operation.shifts[self._pair_cols] - operation.shifts[self._pair_rows]
            )
            # Images in the lower triangle are represented by the
            # conjugate in the upper triangle.
            flipped = image_rows > image_cols
            image_pairs = self._find_pairs(
                np.minimum(image_rows, image_cols), np.maximum(image_rows, image_cols)
            )
            for cell_ind, cell in enumerate(self.cells):
                image_cells = operation.rotation @ cell + relative_shifts
                image_cells[flipped] *= -1
                inds, twists = self._wrap_cells(image_cells)
                update(
                    cell_ind,
                    pair_inds,
                    image_pairs * self.num_cells + inds,
                    orbital_signs * twists,
                    flipped,
                )
                # (i, i, R) and (i, i, -R) are also conjugates.
                inds, twists = self._wrap_cells(-image_cells[is_diagonal])
                update(
                    cell_ind,
                    pair_inds[is_diagonal],
                    image_pairs[is_diagonal] * self.num_cells + inds,
                    orbital_signs[is_diagonal] * twists,
                    np.ones(is_diagonal.sum(), dtype=bool),
                )

        flat_representative = representative.ravel()
        if not np.array_equal(
            flat_representative[flat_representative], flat_representative
        ):
            raise ValueError("Inconsistent orbits.")

        is_zero = is_zero.ravel()
        unique = np.unique(flat_representative)
        if not np.array_equal(is_zero, is_zero[flat_representative]):
            raise ValueError("Inconsistent zeros.")
        self.num_orbits = len(unique)
        kept = unique[~is_zero[unique]]
        self.num_representatives = len(kept)

        rep_pairs, rep_cells = np.divmod(kept, self.num_cells)
        self.rep_rows = self._pair_rows[rep_pairs]
        self.rep_cols = self._pair_cols[rep_pairs]
        self.rep_cells = rep_cells

        # Pair table -> table of the stored elements.
        packed_inds = np.searchsorted(kept, flat_representative).reshape(shape)
        packed_inds = np.minimum(packed_inds, max(len(kept) - 1, 0))
        sign[is_zero.reshape(shape)] = 0

        index_type = np.int32 if len(kept) < np.iinfo(np.int32).max else np.int64
        orbit_index = packed_inds[self._element_to_pair].astype(index_type)
        orbit_sign = sign[self._element_to_pair]
        orbit_conjugate = conjugate[self._element_to_pair]

        # M_ij(R) = eta * conj(M_ji(-R)) for elements in the lower part.
        lower = np.flatnonzero(self._element_is_lower)
        if len(lower):
            negative, twists = self._wrap_cells(-self.cells)
            lower_pairs = self._element_to_pair[lower]
            orbit_index[lower] = packed_inds[lower_pairs][:, negative]
            orbit_sign[lower] = sign[lower_pairs][:, negative] * twists
            orbit_conjugate[lower] = ~conjugate[lower_pairs][:, negative]

        self.orbit_index = xp.asarray(orbit_index)
        self.orbit_sign = xp.asarray(orbit_sign)
        self.orbit_conjugate = xp.asarray(orbit_conjugate)

    @property
    def table_nbytes(self) -> int:
        """The size of the orbit table in bytes."""
        return sum(
            table.nbytes
            for table in (self.orbit_index, self.orbit_sign, self.orbit_conjugate)
        )

    def _get_star(self, operation_ind: int) -> tuple:
        """Gets the preimages (a, b) of the representatives (i, j)."""
        if operation_ind not in self._star_cache:
            operation = self.operations[operation_ind]
            inverse = operation.inverse_targets
            rows, cols = inverse[self.rep_rows], inverse[self.rep_cols]
            self._star_cache[operation_ind] = (
                rows,
                cols,
                operation.signs[rows] * operation.signs[cols],
                # The cell in the phase factor exp(-2 pi i k' R).
                self.cells[self.rep_cells]
                - (operation.shifts[cols] - operation.shifts[rows]),
            )
        return self._star_cache[operation_ind]

    def _iterate_star(self, grid: KGrid):
        """Iterates over the points in the full grid."""
        identity = self._operation_lookup[
            _operation_key(SymmetryOperation(np.eye(3, dtype=int), np.zeros(3), False))
        ]
        for full_ind, kpoint in enumerate(grid.full_points):
            if grid.symmetry_reduction:
                irreducible_ind = int(grid._expand_mapping[full_ind])
                operation = grid.operations[grid._operation_indices[full_ind]]
                operation_ind = self._operation_lookup[_operation_key(operation)]
            else:
                irreducible_ind, operation_ind = full_ind, identity
            yield kpoint, irreducible_ind, operation_ind

    def _get_grid(self, signed: bool) -> KGrid:
        """Gets the grid on which a quantity is defined."""
        return self.kgrid if signed else self.qgrid

    def pack(self, data: NDArray, signed: bool, hermiticity: str) -> NDArray:
        """Packs the stored elements on the irreducible points.

        Parameters
        ----------
        data : NDArray
            The elements in the order of `rows` and `cols`, with shape
            (..., num_irreducible_points, nnz).
        signed : bool
            Whether this is an electronic quantity (G, Sigma) or an
            interaction (P, W).
        hermiticity : str
            Either "hermitian" or "skew-hermitian".

        Returns
        -------
        packed : NDArray
            The representatives, shape (..., num_representatives).

        """
        eta = HERMITICITY_FACTORS[hermiticity]
        packed = xp.zeros(
            data.shape[:-2] + (self.num_representatives,), dtype=xp.complex128
        )
        for kpoint, irreducible_ind, operation_ind in self._iterate_star(
            self._get_grid(signed)
        ):
            rows, cols, signs, cells = self._get_star(operation_ind)
            pairs = self._find_pairs(np.minimum(rows, cols), np.maximum(rows, cols))
            elements = self._pair_to_element[pairs]
            # The stored element might be on the other side.
            conjugate = (rows > cols) != self._element_is_lower[elements]

            factors = np.exp(-2j * np.pi * (cells @ kpoint)) / self.num_cells
            if signed:
                factors = factors * signs
            factors = np.where(conjugate, eta, 1) * factors

            values = data[..., irreducible_ind, :][..., xp.asarray(elements)]
            values = xp.where(xp.asarray(conjugate), values.conj(), values)
            packed += xp.asarray(factors) * values

        return packed

    def accumulate_block(
        self,
        packed: NDArray,
        block: NDArray,
        row_offset: int,
        col_offset: int,
        signed: bool,
    ) -> None:
        """Adds the contribution of a dense block to the packed matrix.

        This applies the symmetry operations to the dense block. Only
        blocks on and above the diagonal are needed. All blocks have to
        be added once to get the packed matrix.

        Note
        ----
        Blocks on the diagonal have to be (skew-)Hermitian.

        Parameters
        ----------
        packed : NDArray
            The packed matrix, shape (..., num_representatives).
        block : NDArray
            The dense block at the irreducible points, shape
            (..., num_irreducible_points, block_rows, block_cols).
        row_offset : int
            The first row of the block.
        col_offset : int
            The first column of the block.
        signed : bool
            Whether this is an electronic quantity (G, Sigma) or an
            interaction (P, W).

        """
        num_rows, num_cols = block.shape[-2:]
        inds, plan = self._get_block_plan(
            row_offset, col_offset, num_rows, num_cols, signed
        )
        if plan is None:
            return

        flat_block = block.reshape(block.shape[:-2] + (num_rows * num_cols,))
        values = 0.0
        for elements, cells, signs, irreducible_inds, phases in plan:
            # All points that are reached with the same operation.
            values = values + signs * xp.sum(
                phases[:, cells]
                * flat_block[..., irreducible_inds[:, None], elements[None, :]],
                axis=-2,
            )
        packed[..., inds] += values

    def _get_block_plan(
        self,
        row_offset: int,
        col_offset: int,
        num_rows: int,
        num_cols: int,
        signed: bool,
    ) -> tuple:
        """Gets the (cached) indices to add a block to the packed matrix."""
        key = (row_offset, col_offset, num_rows, num_cols, signed)
        if key in self._block_cache:
            return self._block_cache[key]

        inds = np.flatnonzero(
            (self.rep_rows >= row_offset)
            & (self.rep_rows < row_offset + num_rows)
            & (self.rep_cols >= col_offset)
            & (self.rep_cols < col_offset + num_cols)
        )
        if len(inds) == 0:
            self._block_cache[key] = (None, None)
            return self._block_cache[key]

        stars: dict[int, list] = {}
        for kpoint, irreducible_ind, operation_ind in self._iterate_star(
            self._get_grid(signed)
        ):
            stars.setdefault(operation_ind, []).append((kpoint, irreducible_ind))

        twist = self._twist if signed else np.ones(3, dtype=int)
        plan = []
        for operation_ind, points in stars.items():
            rows, cols, signs, cells = self._get_star(operation_ind)
            local_rows = rows[inds] - row_offset
            local_cols = cols[inds] - col_offset
            if (
                local_rows.min() < 0
                or local_cols.min() < 0
                or local_rows.max() >= num_rows
                or local_cols.max() >= num_cols
            ):
                raise ValueError("A symmetry operation connects different blocks.")

            # The phases only need to be tabulated for the wrapped cells.
            windings = np.floor_divide(cells[inds], self.size)
            wrapped = cells[inds] - windings * self.size
            factors = np.prod(np.where(windings % 2 == 0, 1, twist), axis=-1)
            if signed:
                factors = factors * signs[inds]
            kpoints = np.array([kpoint for kpoint, __ in points])
            plan.append(
                (
                    xp.asarray(local_rows * num_cols + local_cols),
                    xp.asarray(np.ravel_multi_index(tuple(wrapped.T), self.size)),
                    xp.asarray(factors.astype(np.int8)),
                    xp.asarray(np.array([ind for __, ind in points])),
                    xp.asarray(
                        np.exp(-2j * np.pi * (kpoints @ self.cells.T)) / self.num_cells
                    ),
                )
            )

        self._block_cache[key] = (xp.asarray(inds), plan)
        return self._block_cache[key]

    def accumulate_matrix(
        self, packed: NDArray, matrix: DSDBSparse, signed: bool, mask: NDArray
    ) -> None:
        """Packs the masked energies of a block-tridiagonal matrix again.

        This is used if the matrix was modified after its blocks were
        packed.

        Parameters
        ----------
        packed : NDArray
            The packed matrix, shape (num_energies, num_representatives).
        matrix : DSDBSparse
            The matrix in stack distribution.
        signed : bool
            Whether this is an electronic quantity (G, Sigma) or an
            interaction (P, W).
        mask : NDArray
            Boolean mask of the local energies.

        """
        if not xp.any(mask):
            return
        repacked = xp.zeros((int(mask.sum()), packed.shape[-1]), dtype=packed.dtype)
        offsets = get_host(matrix.block_offsets)
        for i in range(matrix.num_blocks):
            for j in range(i, min(i + 2, matrix.num_blocks)):
                self.accumulate_block(
                    repacked,
                    matrix.blocks[i, j][mask],
                    int(offsets[i]),
                    int(offsets[j]),
                    signed,
                )
        packed[mask] = repacked

    def unpack(self, packed: NDArray, signed: bool, hermiticity: str) -> NDArray:
        """Unpacks a packed matrix to the irreducible points.

        Parameters
        ----------
        packed : NDArray
            The representatives, shape (..., num_representatives).
        signed : bool
            Whether this is an electronic quantity (G, Sigma) or an
            interaction (P).
        hermiticity : str
            Either "hermitian" or "skew-hermitian".

        Returns
        -------
        data : NDArray
            The elements in the order of `rows` and `cols`, with shape
            (..., num_irreducible_points, nnz).

        """
        eta = HERMITICITY_FACTORS[hermiticity]
        points = xp.asarray(self._get_grid(signed).points)
        phases = xp.exp(2j * np.pi * (points @ xp.asarray(self.cells).T))

        data = xp.zeros(
            packed.shape[:-1] + (len(points), len(self.rows)), dtype=xp.complex128
        )
        for cell_ind in range(self.num_cells):
            conjugate = self.orbit_conjugate[:, cell_ind]
            sign = self.orbit_sign[:, cell_ind]
            factors = sign if signed else xp.abs(sign)
            factors = factors * xp.where(conjugate, eta, 1)

            values = packed[..., self.orbit_index[:, cell_ind]]
            values = xp.where(conjugate, values.conj(), values) * factors
            data += phases[:, cell_ind, None] * values[..., None, :]

        return data

    def expand_orbitals(self, values: NDArray, grid: KGrid) -> NDArray:
        """Expands orbital-resolved values to the full grid.

        This is for values that do not depend on the phase of the
        orbitals, e.g. the diagonal elements of a matrix.

        Parameters
        ----------
        values : NDArray
            The values, shape (..., num_irreducible_points,
            num_orbitals).
        grid : KGrid
            The grid of the values.

        Returns
        -------
        expanded_values : NDArray
            The values, shape (..., num_full_points, num_orbitals).

        """
        return xp.stack(
            [
                values[..., irreducible_ind, :][
                    ..., xp.asarray(self.operations[operation_ind].inverse_targets)
                ]
                for __, irreducible_ind, operation_ind in self._iterate_star(grid)
            ],
            axis=-2,
        )

    def callback(
        self,
        packed_lesser: NDArray,
        packed_greater: NDArray,
        block_offsets: NDArray,
        signed: bool,
    ) -> Callable[[BackSubstitutionContext], None]:
        """Creates a callback that packs the blocks in the backward sweep.

        Parameters
        ----------
        packed_lesser : NDArray
            The packed lesser matrix that gets accumulated. Its first
            axis is the stack that is solved for.
        packed_greater : NDArray
            The packed greater matrix that gets accumulated.
        block_offsets : NDArray
            The offsets of the blocks in the solver.
        signed : bool
            Whether these are Green's functions (G) or interactions (W).

        Returns
        -------
        callback : Callable
            The callback to be passed to the selected solver.

        """
        if comm.block.size > 1:
            raise NotImplementedError(
                "Packing is not implemented for domain-distributed solvers."
            )
        offsets = [int(offset) for offset in get_host(block_offsets)]

        def callback(ctx: BackSubstitutionContext) -> None:
            """Packs the diagonal block j and the block (i, j)."""
            for packed, x_jj, x_ij in (
                (packed_lesser, ctx.xl_jj, ctx.xl_ij),
                (packed_greater, ctx.xg_jj, ctx.xg_ij),
            ):
                packed = packed[ctx.stack_slice]
                j_offset = offsets[ctx.j]
                # The solver only makes the output skew-Hermitian.
                x_jj = 0.5 * (x_jj - x_jj.conj().swapaxes(-2, -1))
                self.accumulate_block(packed, x_jj, j_offset, j_offset, signed)
                if ctx.i >= 0:
                    self.accumulate_block(
                        packed, x_ij, offsets[ctx.i], j_offset, signed
                    )

        return callback

    def create_matrix(
        self,
        num_energies: int,
        dsdbsparse_type: type[DSDBSparse],
        symmetry: str = "skew-hermitian",
    ) -> DSDBSparse:
        """Creates a distributed packed matrix.

        This behaves as a matrix with a single k-point, for which the
        representatives are the stored elements. It can be transposed
        between stack and nnz distribution, and the elementwise products
        with other packed matrices (P and Sigma) are the ones of a
        calculation with a single k-point. It has no block structure.

        Parameters
        ----------
        num_energies : int
            The global number of energies.
        dsdbsparse_type : type[DSDBSparse]
            The type of the matrix.
        symmetry : str, optional
            The symmetry of the matrix.

        Returns
        -------
        matrix : DSDBSparse
            The zero-initialized matrix in stack distribution.

        """
        local_size = get_section_sizes(num_energies, comm.stack.size)[0][
            comm.stack.rank
        ]
        inds = xp.arange(self.num_representatives, dtype=xp.int32)
        matrix = dsdbsparse_type(
            dtype=xp.complex128,
            rows=xp.zeros_like(inds),
            cols=inds,
            block_sizes=np.array([self.num_representatives]),
            local_stack_shape=(int(local_size), 1),
            global_stack_shape=(num_energies, 1),
            symmetry=symmetry,
        )
        matrix.allocate_data()
        matrix.data[:] = 0.0
        return matrix
