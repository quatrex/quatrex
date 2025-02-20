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


def create_hamiltonian(hR: NDArray, num_transport_cells: int, cutoff: float = xp.inf, coords: NDArray = None) -> list[NDArray]:

    connections = hR.shape[2] // 2
    num_unit_cells_per_supercell = connections + 1
    wann_rows = hR.shape[3]
    wann_cols = hR.shape[4]
    block_rows = num_unit_cells_per_supercell * wann_rows
    block_cols = num_unit_cells_per_supercell * wann_cols

    # Create the diag, upper, and lower blocks.
    diag_block = xp.zeros((block_rows, block_cols), dtype=hR.dtype)
    upper_block = xp.zeros((block_rows, block_cols), dtype=hR.dtype)
    lower_block = xp.zeros((block_rows, block_cols), dtype=hR.dtype)
    for i in range(num_unit_cells_per_supercell):
        sl_i = slice(i * wann_rows, (i + 1) * wann_rows)
        for j in range(i - connections, 0):
            sl_j = slice(j * wann_cols, (j + 1) * wann_cols)
            lower_block[sl_i, sl_j] = hR[0, 0, j - i]
        for j in range(0, num_unit_cells_per_supercell):
            sl_j = slice(j * wann_cols, (j + 1) * wann_cols)
            diag_block[sl_i, sl_j] = hR[0, 0, j - i]
        for j in range(num_unit_cells_per_supercell, i + connections + 1):
            sl_j = slice(j * wann_cols, (j + 1) * wann_cols)
            upper_block[sl_i, sl_j] = hR[0, 0, j - i]
    
    # Enforce cutoff.
    # NOTE: Assuming single transport direction and coordinate.
    if coords is not None and cutoff < xp.inf:
        unit_cell_dist = xp.abs(xp.subtract.outer(coords, coords))
        unit_cell_width = unit_cell_dist.max()  # NOTE: Can be constant.
        diag_dist = xp.empty((block_rows, block_cols), dtype=unit_cell_dist.dtype)
        upper_dist = xp.empty((block_rows, block_cols), dtype=unit_cell_dist.dtype)
        lower_dist = xp.empty((block_rows, block_cols), dtype=unit_cell_dist.dtype)
        for i in range(num_unit_cells_per_supercell):
            sl_i = slice(i * wann_rows, (i + 1) * wann_rows)
            for j in range(i - connections, 0):
                sl_j = slice(j * wann_cols, (j + 1) * wann_cols)
                lower_dist[sl_i, sl_j] = unit_cell_dist + abs(i - j) * unit_cell_width
            for j in range(0, num_unit_cells_per_supercell):
                sl_j = slice(j * wann_cols, (j + 1) * wann_cols)
                diag_dist[sl_i, sl_j] = unit_cell_dist + abs(i - j) * unit_cell_width
            for j in range(num_unit_cells_per_supercell, i + connections + 1):
                sl_j = slice(j * wann_cols, (j + 1) * wann_cols)
                upper_dist[sl_i, sl_j] = unit_cell_dist + abs(i - j) * unit_cell_width
        diag_block[diag_dist > cutoff] = 0
        upper_block[upper_dist > cutoff] = 0
        lower_block[lower_dist > cutoff] = 0
    
    # Create the block-tridiagonal matrix.
    diag = xp.repeat(diag_block, num_transport_cells, axis=0)
    upper = xp.repeat(upper_block, num_transport_cells - 1, axis=0)
    lower = xp.repeat(lower_block, num_transport_cells - 1, axis=0)

    return diag, upper, lower
            

