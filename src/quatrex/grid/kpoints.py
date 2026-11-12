# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes function to construct a Monkhorst-Pack grid of k-points."""

import ase
import numpy as np
import spglib

from qttools import NDArray, xp


def _monkhorst_pack(
    size: tuple[int] | NDArray,
    shift: tuple[float] | NDArray | None = None,
    gamma_centered: bool = False,
    symmetry_reduction: bool = False,
    structure: ase.Atoms | None = None,
    time_reversal: bool = True,
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

    if structure is None:
        raise ValueError("A `structure` is required when `irreducible` is set to True.")

    cell = (
        structure.cell.array,
        structure.get_scaled_positions(),
        structure.get_atomic_numbers(),
    )

    # NOTE: Instead of using spglib.get_ir_reciprocal_mesh, kinda do
    # irreducible k-point search by hand. This is because we want to
    # support arbitrary shifts and gamma-centered grids, which are not
    # directly exposed through spglib. The code here is a bit clanked
    # together and iterated on. I did my best to make it a bit clearer
    # what actually happens.
    symmetry_dataset = spglib.get_symmetry_dataset(cell)
    # spglib pairs rotation operations with their corresponding
    # translations. We only need the rotations here.
    rotations = np.unique(symmetry_dataset.rotations, axis=0)
    if time_reversal:
        rotations = np.unique(np.concatenate([rotations, -rotations]), axis=0)

    mapping = np.arange(len(kpoints))
    for rotation in rotations:
        # Rotate and map back from k-points to the grid indices.
        m = (kpoints @ rotation - shift) * size - offset
        m_rounded = np.round(m)
        if not np.allclose(m, m_rounded):
            # This symmetry operation does not preserve the grid, so we
            # skip it.
            continue

        # Since we construct the grid using np.indices before, we can
        # use np.ravel_multi_index to map the rounded indices back to
        # the flattened grid. Using modulo to ensure that the indices
        # are in bounds.
        flat = np.ravel_multi_index(tuple((m_rounded.astype(int) % size).T), size)
        # Map each k-point to the minimum index of its
        # symmetry-equivalent k-points.
        mapping = np.minimum(mapping, flat)

    reduce_mapping, expand_mapping, multiplicities = np.unique(
        mapping, return_inverse=True, return_counts=True
    )

    return kpoints, reduce_mapping, expand_mapping, multiplicities


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

    """

    def __init__(
        self,
        size: tuple[int] | NDArray,
        shift: tuple[float] | NDArray | None = None,
        gamma_centered: bool = False,
        symmetry_reduction: bool = False,
        structure: ase.Atoms | None = None,
        time_reversal: bool = True,
    ) -> None:
        """Initializes the k-points."""
        if symmetry_reduction:
            full_points, reduce_mapping, expand_mapping, multiplicities = (
                _monkhorst_pack(
                    size=size,
                    shift=shift,
                    gamma_centered=gamma_centered,
                    symmetry_reduction=True,
                    structure=structure,
                    time_reversal=time_reversal,
                )
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
            reduce_mapping = expand_mapping = irreducible_points = None
            multiplicities = np.ones(len(full_points), dtype=int)

        self.full_points = full_points
        self.irreducible_points = irreducible_points
        self._reduce_mapping = reduce_mapping
        self._expand_mapping = expand_mapping
        self._multiplicities = multiplicities

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
