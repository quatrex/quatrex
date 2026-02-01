# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import re
import warnings
from pathlib import Path

import numpy as np

from qttools import NDArray, _DType, sparse, xp


def read_hr_dat(
    path: Path, return_all: bool = False, dtype: _DType = xp.complex128, read_fast=False
) -> tuple[NDArray, ...]:
    """Parses the contents of a `seedname_hr.dat` file.

    The first line gives the date and time at which the file was
    created. The second line states the number of Wannier functions
    `num_wann`. The third line gives the number of Wigner-Seitz
    grid-points.

    The next block of integers gives the degeneracy of each Wigner-Seitz
    grid point, arranged into 15 values per line.

    Finally, the remaining lines each contain, respectively, the
    components of the Wigner-Seitz cell index, the Wannier center
    indices m and n, and and the real and imaginary parts of the
    Hamiltonian matrix element `HRmn` in the localized basis.

    Parameters
    ----------
    path : Path
        Path to a `seedname_hr.dat` file.
    return_all : bool, optional
        Whether to return all the data or just the Hamiltonian in the
        localized basis. When `True`, the degeneracies and the
        Wigner-Seitz cell indices are also returned. Defaults to
        `False`.
    dtype : dtype, optional
        The data type of the Hamiltonian matrix elements. Defaults to
        `numpy.complex128`.
    read_fast : bool, optional
        Whether to assume that the file is well-formatted and all the
        data is sorted correctly. Defaults to `False`.

    Returns
    -------
    hr : ndarray
        The Hamiltonian matrix elements in the localized basis.
    degeneracies : ndarray, optional
        The degeneracies of the Wigner-Seitz grid points.
    R : ndarray, optional
        The Wigner-Seitz cell indices.

    """

    # Strip info from header.
    num_wann, nrpts = xp.loadtxt(path, skiprows=1, max_rows=2, dtype=int)
    num_wann, nrpts = int(num_wann), int(nrpts)

    # Read wannier data (skipping degeneracy info).
    deg_rows = int(xp.ceil(nrpts / 15.0))
    wann_dat = xp.loadtxt(path, skiprows=3 + deg_rows)

    # Assign R
    if read_fast:
        R = wann_dat[:: num_wann**2, :3].astype(int)
    else:
        R = wann_dat[:, :3].astype(int)
    Rs = xp.subtract(R, R.min(axis=0))
    N1, N2, N3 = Rs.max(axis=0) + 1
    N1, N2, N3 = int(N1), int(N2), int(N3)

    # Obtain Hamiltonian elements.
    if read_fast:
        hR = wann_dat[:, 5] + 1j * wann_dat[:, 6]
        hR = hR.reshape(N1, N2, N3, num_wann, num_wann).swapaxes(-2, -1)
        hR = xp.roll(hR, shift=(N1 // 2 + 1, N2 // 2 + 1, N3 // 2 + 1), axis=(0, 1, 2))
    else:
        hR = xp.zeros((N1, N2, N3, num_wann, num_wann), dtype=dtype)
        for line in wann_dat:
            R1, R2, R3 = line[:3].astype(int)
            m, n = line[3:5].astype(int)
            hR_mn_real, hR_mn_imag = line[5:]
            hR[R1, R2, R3, m - 1, n - 1] = hR_mn_real + 1j * hR_mn_imag

    if return_all:
        return hR, xp.unique(R, axis=0)
    return hR


def read_wannier_wout(
    path: Path, transform_home_cell: bool = True
) -> tuple[NDArray, NDArray]:
    """Parses the contents of a `seedname.wout` file and returns the Wannier centers and lattice vectors.

    TODO: Add tests.

    Parameters
    ----------
    path : Path
        Path to a `seedname.wout` file.
    transform_home_cell : bool, optional
        Whether to transform the Wannier centers to the home cell. Defaults to `True`.

    Returns
    -------
    wannier_centers : ndarray
        The Wannier centers.
    lattice_vectors : ndarray
        The lattice vectors.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    num_lines = len(lines)

    # Find the line with the lattice vectors.
    for i, line in enumerate(lines):
        if "Lattice Vectors" in line:
            lattice_vectors = xp.asarray(
                [list(map(float, lines[i + j + 1].split()[1:])) for j in range(3)]
            )
        if "Number of Wannier Functions" in line:
            num_wann = int(line.split()[-2])
            break

    # Find the line with the Wannier centers. Start from the end of the file.
    for i, line in enumerate(lines[::-1]):
        if "Final State" in line:
            # The Wannier centers are enclosed by parantheses, so we have to extract them.
            wannier_centers = xp.asarray(
                [
                    list(
                        map(
                            float,
                            re.findall(r"\((.*?)\)", lines[num_lines - i + j])[0].split(
                                ","
                            ),
                        )
                    )
                    for j in range(num_wann)
                ]
            )
            break

    if transform_home_cell:
        # Get the transformation that diagonalize the lattice vectors
        transformation = xp.linalg.inv(lattice_vectors)
        # Appy it to the wannier centers
        wannier_centers = xp.dot(wannier_centers, transformation)
        # Translate the Wannier centers to the home cell
        wannier_centers = xp.mod(wannier_centers, 1)
        # Transform the Wannier centers back to the original basis
        wannier_centers = xp.dot(wannier_centers, lattice_vectors)

    return wannier_centers, lattice_vectors


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


def get_hamiltonian_block(
    hr: NDArray,
    supercell_size: tuple,
    global_shift: tuple,
) -> NDArray:
    """Constructs a supercell Hamiltonian block from the unit cell.

    Parameters
    ----------
    hr : NDArray
        Wannier Hamiltonian.
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
                if any(abs(i) > hr.shape[j] // 2 for j, i in enumerate(ind)):
                    raise IndexError
                block = hr[ind]
            except IndexError:
                block = xp.zeros(hr.shape[-2:], dtype=hr.dtype)
            row.append(block)
        rows.append(xp.hstack(row))
    return xp.vstack(rows)


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


def create_hamiltonian(
    hr: NDArray,
    num_transport_cells: int,
    transport_dir: int | str = "x",
    supercell_size: tuple = None,
    block_start: int = None,
    block_end: int = None,
    periodic_shift: tuple = (0, 0, 0),
    return_sparse: bool = True,
    cutoff: float = xp.inf,
    coords: NDArray = None,
    lattice_vectors: NDArray = None,
) -> list[NDArray]:
    """Creates a block-tridiagonal Hamiltonian matrix from a Wannier Hamiltonian.

    The transport cell (same as supercell) is the cell that is repeated
    in the transport direction, and is only connected to
    nearest-neighboring cells. NOTE: interactions outside nearest
    neighbors are not included in the block-tridiagonal Hamiltonian (see
    below). It can therefore be important to make sure that the
    transport cell is large enough, such that each row have the same
    number of neighbouring cells. Not setting a transport cell will
    default to a cell that includes all interactions of hr.

      ------- -------
     | o o o | o o o | x
     | o o o | o o o | x x  <- cells outside nearest neighbors are not included
     | o o o | o o o | x x x
      ------- ------- -------
     | o o o | o o o | o o o |
     | o o o | o o o | o o o |
     | o o o | o o o | o o o |
      ------- ------- -------
       x x x | o o o | o o o |
         x x | o o o | o o o |
           x | o o o | o o o |
              ------- -------

    Parameters
    ----------
    hr : NDArray
        Wannier Hamiltonian.
    num_transport_cells : int
        Number of transport cells.
    transport_dir : int or str, optional
        Direction of transport. Can be 0, 1, 2, 'x', 'y', or 'z'.
    transport_cell : tuple, optional
        Size of the transport cell. E.g. (2, 2, 1) for a 2x2
        xy-transport cell.
    block_start : int, optional
        Starting block index for arrow shape partition. Defaults to
        `None`.
    block_end : int, optional
        Ending block index for arrow shape partition. Defaults to
        `None`.
    periodic_shift : tuple, optional
        Incase the system is periodic in non-transport directions, the
        periodic shift can be used to get interactions between the
        transport cell and the periodic cells. E.g. (0, 0, 1) for one of
        the periodic shifts in the z-direction.
    return_sparse : bool, optional
        Whether to return the block-tridiagonal Hamiltonian as a sparse
        matrix. Defaults to `False`.
    cutoff : float, optional
        Cutoff distance for connections between wannier functions.
        Defaults to `np.inf`.
    coords : NDArray, optional
        Coordinates of the Wannier functions in a unit cell. Defaults to
        `None`.
    lattice_vectors : NDArray, optional
        Lattice vectors of the system. Defaults to `None`.

    Returns
    -------
    list[NDArray] or tuple[sparse.coo_matrix, NDArray]
        The block-tridiagonal Hamiltonian matrix as either a tuple of
        arrays or a sparse matrix and block sizes.

    """
    if cutoff is not xp.inf and coords is None and lattice_vectors is None:
        print(
            "Cutoff is set but coords and lattice_vectors are not provided. No cutoff will be applied.",
            flush=True,
        )

    if isinstance(transport_dir, str):
        transport_dir = "xyz".index(transport_dir)

    if supercell_size is None:
        # NOTE: Can also do without the + 1.
        supercell_size = tuple(
            [
                shape // 2 + 1 if i == transport_dir else 1
                for i, shape in enumerate(hr.shape[:3])
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

    if (np.abs(periodic_shift) > np.array(hr.shape[:3]) // 2).any():
        warnings.warn(
            "Periodic shift is outside the available range. Interaction will be zero."
        )
    if periodic_shift[transport_dir] != 0:
        warnings.warn(
            "Periodic shift in the transport direction. This does not make sense and will be ignored."
        )

    upper_ind = tuple(
        [1 if i == transport_dir else periodic_shift[i] for i in range(3)]
    )
    lower_ind = tuple(
        [-1 if i == transport_dir else periodic_shift[i] for i in range(3)]
    )
    diag_ind = tuple([0 if i == transport_dir else periodic_shift[i] for i in range(3)])

    diag_block = get_hamiltonian_block(hr, supercell_size, diag_ind)
    upper_block = get_hamiltonian_block(hr, supercell_size, upper_ind)
    lower_block = get_hamiltonian_block(hr, supercell_size, lower_ind)

    # Enforce cutoff.
    if coords is not None and cutoff < xp.inf and lattice_vectors is not None:
        super_cell_coords = create_coordinate_grid(
            coords, supercell_size, lattice_vectors
        )
        distance_matrix = xp.diagonal(
            xp.subtract.outer(super_cell_coords, super_cell_coords), axis1=1, axis2=3
        )
        diag_dist = xp.linalg.norm(distance_matrix, axis=-1)
        upper_dist = xp.linalg.norm(
            distance_matrix + xp.asarray(upper_ind) @ lattice_vectors, axis=-1
        )
        lower_dist = xp.linalg.norm(
            distance_matrix + xp.asarray(lower_ind) @ lattice_vectors, axis=-1
        )
        diag_block[diag_dist > cutoff] = 0
        upper_block[upper_dist > cutoff] = 0
        lower_block[lower_dist > cutoff] = 0

    if return_sparse:
        # Create sparse matrices of the blocks.
        diag_block = sparse.coo_matrix(diag_block)
        upper_block = sparse.coo_matrix(upper_block)
        lower_block = sparse.coo_matrix(lower_block)
        # Canoncialize the sparse matrices.
        # NOTE: Not sure if this is necessary.
        for mat in [diag_block, upper_block, lower_block]:
            if mat.has_canonical_format is False:
                mat.sum_duplicates()
        # Create the block-tridiagonal matrix.
        num_blocks = block_end - block_start
        offsets = xp.arange(block_start, block_end) * diag_block.shape[0]

        def _tile_sparse_blocks(block, num_blocks, offsets):
            return (
                xp.tile(block.row, num_blocks) + xp.repeat(offsets, block.nnz),
                xp.tile(block.col, num_blocks) + xp.repeat(offsets, block.nnz),
                xp.tile(block.data, num_blocks),
            )

        diag_rows, diag_cols, diag_data = _tile_sparse_blocks(
            diag_block, num_blocks, offsets
        )
        upper_rows, upper_cols, upper_data = _tile_sparse_blocks(
            upper_block, num_blocks, offsets
        )
        lower_rows, lower_cols, lower_data = _tile_sparse_blocks(
            lower_block, num_blocks, offsets
        )
        upper_cols += diag_block.shape[0]
        lower_rows += diag_block.shape[0]

        full_rows = xp.hstack([diag_rows, upper_rows, lower_rows])
        full_cols = xp.hstack([diag_cols, upper_cols, lower_cols])
        full_data = xp.hstack([diag_data, upper_data, lower_data])
        # Remove the fishtail at the end of the matrix.
        matrix_shape = num_transport_cells * diag_block.shape[0]
        valid_mask = (full_cols < matrix_shape) & (full_rows < matrix_shape)
        full_rows = full_rows[valid_mask]
        full_cols = full_cols[valid_mask]
        full_data = full_data[valid_mask]
        # Also return the block sizes.
        block_sizes = np.ones(num_blocks, dtype=int) * diag_block.shape[0]
        return (
            sparse.coo_matrix(
                (full_data, (full_rows, full_cols)),
                shape=(matrix_shape, matrix_shape),
            ),
            block_sizes,
        )
    else:
        # Returns the block-tridiagonal Hamiltonian matrix as a tuple of arrays.
        diag = xp.tile(diag_block, (block_end - block_start, 1))
        upper = xp.tile(
            upper_block,
            (min(block_end + 1, num_transport_cells) - (block_start + 1), 1),
        )
        lower = xp.tile(
            lower_block,
            (min(block_end + 1, num_transport_cells) - (block_start + 1), 1),
        )

        return diag, upper, lower
