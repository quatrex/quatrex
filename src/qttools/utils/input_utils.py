# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import warnings
from copy import copy

import numpy as np

from qttools import NDArray, sparse, xp


def _trim_zeros_nd(arr: NDArray) -> NDArray:
    """Implementation of trim_zeros over all dimensions

    This function removes all-zero slices from a multi-dimensional array.

    Parameters
    ----------
    arr : NDArray
        The input array.

    Returns
    -------
    NDArray
        The trimmed array.

    """

    nz = xp.nonzero(arr)

    if len(nz[0]) == 0:
        return xp.array([])

    slices = tuple(slice(xp.min(i), xp.max(i) + 1) for i in nz)
    return arr[slices]


def trim_tight_binding_matrix(
    tight_binding_matrix: NDArray,
    value_cutoff: float | None = None,
    neighbor_cell_cutoff: tuple[int, int, int] | None = None,
) -> NDArray:
    """Applies cutoffs to tight-binding matrix elements/blocks.

    Elements are selected based on value cutoff or their distance to the
    home cell. Cells that end up being all zeros after applying the
    cutoffs are removed.

    Parameters
    ----------
    tight_binding_matrix : NDArray
        A tight-binding matrix.
    value_cutoff : float, optional
        Cutoff value for the matrix. Defaults to `None`.
    neighbor_cell_cutoff: tuple, optional
        How many neighboring cells to consider along each lattice
        vector. A cutoff of (1, 1, 0) would consider the home cell and
        the first neighbor cells along the first two lattice vectors,
        but not along the third lattice vector. Defaults to `None`,
        which means to consider all the neighbor cells present in `hr`.

    Returns
    -------
    NDArray
        The remaining matrix after applying the cutoffs.

    """
    trimmed_matrix = tight_binding_matrix.copy()

    if neighbor_cell_cutoff is not None:
        neighbor_cell_cutoff = np.array(neighbor_cell_cutoff)
        # Make sure that we don't ask for more neighbor cells than
        # available in the matrix.
        if any(tight_binding_matrix.shape[:3] < 2 * neighbor_cell_cutoff + 1):
            raise ValueError(
                "matrix contains fewer neighbor cells than requested."
                f"({tight_binding_matrix.shape[:3]=}, {neighbor_cell_cutoff=})"
            )

        for ind in np.ndindex(tight_binding_matrix.shape[:3]):
            # Center the indices around zero.
            cell_index = (
                np.asarray(ind) - np.asarray(tight_binding_matrix.shape[:3]) // 2
            )
            if any(abs(cell_index) > neighbor_cell_cutoff):
                trimmed_matrix[*cell_index] = 0.0

    if value_cutoff is not None:
        trimmed_matrix[xp.abs(trimmed_matrix) < value_cutoff] = 0

    # Rotate such that 0,0,0 is in the center
    trimmed_matrix = xp.roll(
        trimmed_matrix, shift=(trimmed_matrix.shape[0] // 2), axis=0
    )
    trimmed_matrix = xp.roll(
        trimmed_matrix, shift=(trimmed_matrix.shape[1] // 2), axis=1
    )
    trimmed_matrix = xp.roll(
        trimmed_matrix, shift=(trimmed_matrix.shape[2] // 2), axis=2
    )
    # Remove cells that end up being all zeros.
    trimmed_matrix = _trim_zeros_nd(trimmed_matrix)
    # Rotate back
    trimmed_matrix = xp.roll(
        trimmed_matrix, shift=-(trimmed_matrix.shape[0] // 2), axis=0
    )
    trimmed_matrix = xp.roll(
        trimmed_matrix, shift=-(trimmed_matrix.shape[1] // 2), axis=1
    )
    trimmed_matrix = xp.roll(
        trimmed_matrix, shift=-(trimmed_matrix.shape[2] // 2), axis=2
    )

    return trimmed_matrix


def create_coordinate_grid(
    wannier_centers: NDArray,
    supercell_size: tuple,
    lattice_vectors: NDArray,
) -> NDArray:
    """Creates a grid of coordinates for Wannier centers in a supercell.

    Parameters
    ----------
    wannier_centers : NDArray
        Coordinates of the Wannier centers in a unit cell.
    supercell_size : tuple
        Size of the supercell. E.g. (2, 2, 1) for a 2x2 xy-supercell.
    lattice_vectors : NDArray
        Lattice vectors of the system.

    Returns
    -------
    NDArray
        The grid of coordinates for the Wannier centers in the
        supercell.

    """
    num_wann = wannier_centers.shape[0]
    grid = xp.zeros(
        (int(xp.prod(xp.asarray(supercell_size)) * num_wann), 3), dtype=xp.float64
    )
    for i, cell_ind in enumerate(np.ndindex(supercell_size)):
        grid[i * num_wann : (i + 1) * num_wann, :] = (
            wannier_centers + xp.asarray(cell_ind) @ lattice_vectors
        )
    return grid


def _get_transport_block(
    tight_binding_matrix: NDArray,
    supercell_size: tuple,
    global_shift: tuple,
) -> NDArray:
    """Constructs a supercell block from the unit cell.

    Parameters
    ----------
    tight_binding_matrix : NDArray
        Wannier unit cells.
    supercell_size : tuple
        Size of the supercell. E.g. (2, 2, 1) for a 2x2 xy-supercell.
    global_shift : tuple
        Shift in the supercell system. If you want a
        R-shift of 1 cell in x direction, you would pass (1, 0,
        0). NOTE: this is for the supercell and NOT the unit cell.

    Returns
    -------
    NDArray
        The supercell hamiltonian block.

    """
    local_shifts = np.asarray(list(np.ndindex(supercell_size)))
    supercell_size = np.asarray(supercell_size)
    global_shift = np.asarray(global_shift)
    global_shift = np.multiply(global_shift, supercell_size)

    rows = []
    for r_i in local_shifts:
        row = []
        for r_j in local_shifts:
            ind = tuple(r_j - r_i + global_shift)
            try:
                if any(
                    abs(i) > tight_binding_matrix.shape[j] // 2
                    for j, i in enumerate(ind)
                ):
                    raise IndexError
                block = tight_binding_matrix[ind]
            except IndexError:
                block = xp.zeros(
                    tight_binding_matrix.shape[-2:], dtype=tight_binding_matrix.dtype
                )
            row.append(block)
        rows.append(xp.hstack(row))
    return xp.vstack(rows)


def expand_tight_binding_matrix(
    tight_binding_matrix: NDArray,
    num_transport_cells: int,
    transport_direction: int | str,
    block_start: int | None = None,
    block_end: int | None = None,
    periodic_shift: tuple = (0, 0),
) -> tuple[sparse.csr_matrix, NDArray]:
    """Creates a full block-tridiagonal matrix from tight-binding matrix / Wannier Centers.

    The transport cell (same as supercell) is the cell that is repeated
    in the transport direction, and is only connected to
    nearest-neighboring cells. NOTE: interactions outside nearest
    neighbors are not included in the block-tridiagonal Hamiltonian (see
    below).

    Example for a tight-binding matrix with 3 cells in transport direction,

      ------- -------
     | o o o | o x x | x
     | o o o | o o x | x x
     | o o o | o o o | x x x
      ------- ------- -------
     | o o o | o o o | o x x |
     | x o o | o o o | o o x |
     | x x o | o o o | o o o |
      ------- ------- -------
       x x x | o o o | o o o |
         x x | x o o | o o o |
           x | x x o | o o o |
              ------- -------

    Parameters
    ----------
    tight_binding_matrix : NDArray
        Wannier unit cells.
    num_transport_cells : int
        Number of transport cells.
    transport_direction : int or str
        Direction of transport. Can be 0, 1, 2, 'x', 'y', or 'z'.
    block_start : int | None, optional
        Starting block index for arrow shape partition. Defaults to
        `None`.
    block_end : int | None, optional
        Ending block index for arrow shape partition. Defaults to
        `None`.
    periodic_shift : tuple, optional
        Incase the system is periodic in non-transport directions, the
        periodic shift can be used to get interactions between the
        transport cell and the periodic cells. E.g. (0, 1) for one of
        the periodic shifts in the z-direction.

    Returns
    -------
    tuple[sparse.csr_matrix, NDArray]
        The block-tridiagonal Hamiltonian matrix as either a tuple of
        arrays or a sparse matrix and block sizes.

    """

    if isinstance(transport_direction, str):
        transport_direction = "xyz".index(transport_direction)

    supercell_size = tuple(
        [
            shape // 2 if i == transport_direction else 1
            for i, shape in enumerate(tight_binding_matrix.shape[:3])
        ]
    )

    block_start = block_start or 0
    block_end = block_end or num_transport_cells
    if block_start >= block_end:
        raise ValueError("block_start must be smaller than block_end.")
    if block_end > num_transport_cells:
        raise ValueError("block_end must be smaller than num_transport_cells.")
    if block_start < 0:
        raise ValueError("block_start must be greater than or equal to 0.")

    if len(periodic_shift) != 2:
        raise ValueError("periodic_shift must have length 2.")

    block_inds = [
        list(copy(periodic_shift))[:transport_direction]
        + [b]
        + list(copy(periodic_shift))[transport_direction:]
        for b in [-1, 0, 1]
    ]

    if (np.abs(block_inds[1]) > np.array(tight_binding_matrix.shape[:3]) // 2).any():
        warnings.warn(
            "Periodic shift is outside the available range. Interaction will be zero."
        )

    # Create sparse matrices of the blocks.
    blocks = [
        sparse.coo_matrix(
            _get_transport_block(tight_binding_matrix, supercell_size, ind)
        )
        for ind in block_inds
    ]

    # Canoncialize the sparse matrices.
    for block in blocks:
        if block.has_canonical_format is False:
            block.sum_duplicates()

    # Create the block-tridiagonal matrix.
    num_blocks = block_end - block_start
    block_size = blocks[0].shape[0]
    offsets = xp.arange(block_start, block_end) * blocks[0].shape[0]

    def _tile_sparse_blocks(block, num_blocks, offsets):
        return (
            xp.tile(block.row, num_blocks) + xp.repeat(offsets, block.nnz),
            xp.tile(block.col, num_blocks) + xp.repeat(offsets, block.nnz),
            xp.tile(block.data, num_blocks),
        )

    full_rows = []
    full_cols = []
    full_data = []
    shifts = [(1, 0), (0, 0), (0, 1)]
    for block, (row_shift, col_shift) in zip(blocks, shifts):
        rows, cols, data = _tile_sparse_blocks(block, num_blocks, offsets)

        # Shift rows and columns for off-diagonal blocks
        rows += row_shift * block_size
        cols += col_shift * block_size

        full_rows.append(rows)
        full_cols.append(cols)
        full_data.append(data)

    full_rows = xp.hstack(full_rows)
    full_cols = xp.hstack(full_cols)
    full_data = xp.hstack(full_data)

    # Remove the fishtail at the end of the matrix.
    matrix_shape = num_transport_cells * block_size
    valid_mask = (full_cols < matrix_shape) & (full_rows < matrix_shape)
    full_rows = full_rows[valid_mask]
    full_cols = full_cols[valid_mask]
    full_data = full_data[valid_mask]
    # Also return the block sizes.
    block_sizes = np.ones(num_blocks, dtype=int) * block_size
    return (
        sparse.csr_matrix(
            (full_data, (full_rows, full_cols)),
            shape=(matrix_shape, matrix_shape),
        ),
        block_sizes,
    )
