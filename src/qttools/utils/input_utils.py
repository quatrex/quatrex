# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from pathlib import Path

from qttools import NDArray, _DType, xp


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
        Whether to asume that the file is well-formed and all the
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


def get_hamiltonian_block(
    hr: NDArray,
    supercell_size: tuple,
    global_shift: tuple,
) -> xp.ndarray:
    """Constructs a supercell hamiltonian block from an hr array.

    Parameters
    ----------
    hr : np.ndarray
        Wannier Hamiltonian.
    supercell_size : tuple
        Size of the supercell. E.g. (2, 2, 1) for a 2x2 xy-supercell.
    global_shift : tuple
        Shift in the supercell system. If you want a
        R-shift of 1 cell in x direction, you would pass (1, 0,
        0).

    Returns
    -------
    np.ndarray
        The supercell hamiltonian block.

    """
    local_shifts = xp.array(list(xp.ndindex(supercell_size)))
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
        rows.append(row)
    return xp.block(rows)


def create_coordinate_grid(
    wannier_centers: NDArray, super_cell: tuple, lattice_vectors: NDArray
) -> NDArray:
    """Creates a grid of coordinates for Wannier functions in a supercell."""
    num_wann = wannier_centers.shape[0]
    grid = xp.zeros((xp.prod(super_cell) * num_wann, 3), dtype=xp.float64)
    for i, cell_ind in enumerate(xp.ndindex(*super_cell)):
        grid[i * num_wann : (i + 1) * num_wann, :] = (
            wannier_centers + xp.array(cell_ind) @ lattice_vectors
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
) -> list[NDArray]:
    """Creates a block-tridiagonal Hamiltonian matrix from a Wannier Hamiltonian.
    The transport cell (same as supercell) is the cell that is repeated in the transport direction,
    and is only connected to nearest-neighboring cells. Note therefore that interactions outside
    nearest neighbors are not included in the block-tridiagonal Hamiltonian.

    Parameters
    ----------
    hR : np.ndarray
        Wannier Hamiltonian.
    num_transport_cells : int
        Number of transport cells.
    transport_dir : int or str, optional
        Direction of transport. Can be 0, 1, 2, 'x', 'y', or 'z'.
    transport_cell : tuple, optional
        Size of the transport cell. E.g. [2, 2, 1] for a 2x2 xy-transport cell.
    cutoff : float, optional
        Cutoff distance for connections between wannier functions. Defaults to `np.inf`.
    coords : np.ndarray, optional
        Coordinates of the Wannier functions in a unit cell. Defaults to `None`.
    lattice_vectors : np.ndarray, optional
        Lattice vectors of the system. Defaults to `None`.

    Returns
    -------
    list[np.ndarray]
        The block-tridiagonal Hamiltonian matrix.
    """
    if cutoff is not None and coords is None and lattice_vectors is None:
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
            distance_matrix + xp.array(upper_ind) @ lattice_vectors, axis=-1
        )
        lower_dist = xp.linalg.norm(
            distance_matrix + xp.array(lower_ind) @ lattice_vectors, axis=-1
        )
        diag_block[diag_dist > cutoff] = 0
        upper_block[upper_dist > cutoff] = 0
        lower_block[lower_dist > cutoff] = 0

    # Create the block-tridiagonal matrix.
    diag = xp.tile(diag_block, (num_transport_cells, 1))
    upper = xp.tile(upper_block, (num_transport_cells - 1, 1))
    lower = xp.tile(lower_block, (num_transport_cells - 1, 1))

    return diag, upper, lower
