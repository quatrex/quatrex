# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import re
from pathlib import Path

from scipy import sparse

from qttools import NDArray, _DType, xp
from qttools.utils.gpu_utils import get_host


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


def cutoff_hr(
    hr: NDArray,
    value_cutoff: float | None = None,
    R_cutoff: int | tuple[int, int, int] | None = None,
    remove_zeros: bool = False,
) -> NDArray:
    """Cutoffs the Hamiltonian matrix elements based on their values and/or the wigner-seitz cell indices.

    TODO: Add tests.

    Parameters
    ----------
    hr : ndarray
        Wannier Hamiltonian.
    value_cutoff : float, optional
        Cutoff value for the Hamiltonian. Defaults to `None`.
    R_cutoff : int or tuple, optional
        Cutoff distance for the Hamiltonian. Defaults to `None`.
    remove_zeros : bool, optional
        Whether to remove cell planes with only zeros. Defaults to `False`.

    Returns
    -------
    ndarray
        The cutoff Hamiltonian.
    """
    hr_cut = None
    if value_cutoff is None and R_cutoff is None:
        return hr.copy()
    if R_cutoff is not None:
        if isinstance(R_cutoff, int):
            R_cutoff = (R_cutoff, R_cutoff, R_cutoff)
        cut_shape = [
            r * 2 + 1 if hr.shape[i] > r * 2 + 1 else hr.shape[i]
            for i, r in enumerate(R_cutoff)
        ] + list(hr.shape[3:])
        hr_cut = xp.zeros(cut_shape, dtype=hr.dtype)
        for ind in xp.ndindex(hr.shape[:3]):
            ind = xp.asarray(ind) - xp.asarray(hr.shape[:3]) // 2
            if (abs(ind) <= xp.asarray(R_cutoff)).all():
                hr_cut[*ind] = hr[*ind]
    if value_cutoff is not None:
        if hr_cut is None:
            hr_cut = hr.copy()
        hr_cut[xp.abs(hr_cut) > value_cutoff] = 0

    # Remove eventual cell planes with only zeros, except the center.
    if remove_zeros:
        zero_mask = hr_cut.any(axis=(-2, -1))
        zero_mask[0, 0, 0] = True

        for ax in range(3):  # Loop through axes (0, 1, 2)
            # Loop backwards through the axis (from the edge to the center).
            for idx in range(hr_cut.shape[ax] // 2, 0, -1):
                axes_to_remove = []
                # Check if all elements are False in the cell plane.
                if not zero_mask.take(idx, axis=ax).any():
                    # If so, remove it.
                    axes_to_remove.append(idx)
                elif not zero_mask.take(-idx, axis=ax).any():
                    axes_to_remove.append(-idx)
                else:
                    # If not, break the loop (to not mess with ordering incase zero planes are not at the edge).
                    break
                hr_cut = xp.delete(hr_cut, axes_to_remove, axis=ax)

    return hr_cut


def get_hamiltonian_block(
    hr: NDArray,
    supercell_size: tuple,
    global_shift: tuple,
) -> xp.ndarray:
    """Constructs a supercell hamiltonian block from an hr array.

    Parameters
    ----------
    hr : ndarray
        Wannier Hamiltonian.
    supercell_size : tuple
        Size of the supercell. E.g. (2, 2, 1) for a 2x2 xy-supercell.
    global_shift : tuple
        Shift in the supercell system. If you want a
        R-shift of 1 cell in x direction, you would pass (1, 0,
        0). NOTE: this is for the supercell and NOT the unit cell.

    Returns
    -------
    ndarray
        The supercell hamiltonian block.

    """
    local_shifts = xp.asarray(list(xp.ndindex(supercell_size)))
    # Transform to NDArrays (because of cupy multiply).
    if not isinstance(supercell_size, xp.ndarray):
        supercell_size = xp.asarray(supercell_size)
    if not isinstance(global_shift, xp.ndarray):
        global_shift = xp.asarray(global_shift)
    global_shift = xp.multiply(global_shift, supercell_size)

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
    wannier_centers: NDArray, super_cell: tuple, lattice_vectors: NDArray
) -> NDArray:
    """Creates a grid of coordinates for Wannier functions in a supercell."""
    num_wann = wannier_centers.shape[0]
    grid = xp.zeros(
        (int(xp.prod(xp.asarray(super_cell)) * num_wann), 3), dtype=xp.float64
    )
    for i, cell_ind in enumerate(xp.ndindex(super_cell)):
        grid[i * num_wann : (i + 1) * num_wann, :] = (
            wannier_centers + xp.asarray(cell_ind) @ lattice_vectors
        )
    return grid


def create_hamiltonian(
    hR: NDArray,
    num_transport_cells: int,
    transport_dir: int | str = "x",
    transport_cell: list = None,
    cutoff: float = xp.inf,
    coords: NDArray = None,
    lattice_vectors: NDArray = None,
    return_sparse: bool = True,
) -> list[NDArray]:
    """Creates a block-tridiagonal Hamiltonian matrix from a Wannier Hamiltonian.
    The transport cell (same as supercell) is the cell that is repeated in the transport direction,
    and is only connected to nearest-neighboring cells. NOTE: interactions outside
    nearest neighbors are not included in the block-tridiagonal Hamiltonian (see below).
    It can therefore be important to make sure that the transport cell is large enough, such that
    each row have the same number of neighbouring cells. Not setting a transport cell will default
    to a cell that includes all interactions of hR.

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
    hR : ndarray
        Wannier Hamiltonian.
    num_transport_cells : int
        Number of transport cells.
    transport_dir : int or str, optional
        Direction of transport. Can be 0, 1, 2, 'x', 'y', or 'z'.
    transport_cell : tuple, optional
        Size of the transport cell. E.g. [2, 2, 1] for a 2x2 xy-transport cell.
    cutoff : float, optional
        Cutoff distance for connections between wannier functions. Defaults to `np.inf`.
    coords : ndarray, optional
        Coordinates of the Wannier functions in a unit cell. Defaults to `None`.
    lattice_vectors : ndarray, optional
        Lattice vectors of the system. Defaults to `None`.
    return_sparse : bool, optional
        Whether to return the block-tridiagonal Hamiltonian as a sparse matrix. Defaults to `False`.

    Returns
    -------
    list[ndarray]
        The block-tridiagonal Hamiltonian matrix as either a tuple of arrays or a sparse matrix and block sizes.
    """
    if cutoff is not xp.inf and coords is None and lattice_vectors is None:
        print(
            "Cutoff is set but coords and lattice_vectors are not provided. No cutoff will be applied.",
            flush=True,
        )

    if isinstance(transport_dir, str):
        transport_dir = "xyz".index(transport_dir)

    if transport_cell is None:
        # NOTE: Can also do without the + 1.
        transport_cell = tuple(
            [
                shape // 2 + 1 if i == transport_dir else 1
                for i, shape in enumerate(hR.shape[:3])
            ]
        )

    upper_ind = tuple([1 if i == transport_dir else 0 for i in range(3)])
    lower_ind = tuple([-1 if i == transport_dir else 0 for i in range(3)])

    diag_block = get_hamiltonian_block(hR, transport_cell, (0, 0, 0))
    upper_block = get_hamiltonian_block(hR, transport_cell, upper_ind)
    lower_block = get_hamiltonian_block(hR, transport_cell, lower_ind)

    # Enforce cutoff.
    if coords is not None and cutoff < xp.inf and lattice_vectors is not None:
        super_cell_coords = create_coordinate_grid(
            coords, transport_cell, lattice_vectors
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
        # Create a sparse matrix of a block row.
        coo_mat = sparse.coo_matrix(
            get_host(xp.hstack([lower_block, diag_block, upper_block]))
        )
        if coo_mat.has_canonical_format is False:
            coo_mat.sum_duplicates()
        # Tile the block row to create the full block-tridiagonal matrix.
        offsets = xp.arange(num_transport_cells) * diag_block.shape[0]
        full_rows = xp.tile(coo_mat.row, num_transport_cells) + xp.repeat(
            offsets, coo_mat.nnz
        )
        full_cols = xp.tile(coo_mat.col, num_transport_cells) + xp.repeat(
            offsets, coo_mat.nnz
        )
        full_data = xp.tile(coo_mat.data, num_transport_cells)
        # Remove the coupling to the leads to make it square.
        valid_mask = (full_cols >= lower_block.shape[1]) & (
            full_cols < diag_block.shape[0] * num_transport_cells + lower_block.shape[1]
        )
        full_rows = full_rows[valid_mask]
        full_cols = full_cols[valid_mask] - lower_block.shape[1]
        full_data = full_data[valid_mask]
        # Also return the block sizes.
        block_sizes = xp.ones(num_transport_cells, dtype=int) * diag_block.shape[0]
        return (
            sparse.coo_matrix(
                (get_host(full_data), (get_host(full_rows), get_host(full_cols)))
            ),
            block_sizes,
        )
    else:
        # Returns the block-tridiagonal Hamiltonian matrix as a tuple of arrays.
        # Create the block-tridiagonal matrix.
        diag = xp.tile(diag_block, (num_transport_cells, 1))
        upper = xp.tile(upper_block, (num_transport_cells - 1, 1))
        lower = xp.tile(lower_block, (num_transport_cells - 1, 1))

        return diag, upper, lower
