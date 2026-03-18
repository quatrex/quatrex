# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sps
from mpi4py.MPI import COMM_WORLD as comm_world

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.utils.mpi_utils import distributed_load, get_section_sizes
from quatrex.core.config import QuatrexConfig
from quatrex.grid.kpoints import monkhorst_pack


def create_coordinate_grid(
    unit_cell_coords: NDArray,
    transport_cell_size: int,
    transport_ind: int,
    lattice_vectors: NDArray,
) -> NDArray:
    """Creates a grid of coordinates for orbital centers in a transport cell.
    Only expands in the transport direction, and assumes that the unit cell is
    repeated in the transport direction.

    Parameters
    ----------
    unit_cell_coords : NDArray
        Coordinates of the orbital centers in a unit cell.
    transport_cell_size : int
        Number of unit cells in the transport direction that make up the transport cell.
    transport_ind : int
        Index of the transport direction (0, 1, or 2).
    lattice_vectors : NDArray
        Lattice vectors of the system.

    Returns
    -------
    NDArray
        The grid of coordinates for the orbital centers in the
        transport cell.

    """
    num_coords = unit_cell_coords.shape[0]
    grid = xp.zeros((transport_cell_size * num_coords, 3), dtype=xp.float64)
    for i in range(transport_cell_size):
        grid[i * num_coords : (i + 1) * num_coords, :] = (
            unit_cell_coords + i * lattice_vectors[transport_ind]
        )
    return grid


def _construct_transport_cell(
    matrix_dict: dict,
    transport_cell_size: int,
    transport_ind: int,
    shift: tuple,
) -> NDArray:
    """Constructs a transport block from the unit cell.
    This expand the unit cell matrix into a block matrix for the transport cell,
    which is repeated in the transport direction.

    Parameters
    ----------
    matrix_dict : dict
        The dictionary of matrices corresponding to different periodic
        repetitions. It is assumed that only the upper part of the (0,0,0) cell
        and half the keys are present.
    transport_cell_size : int
        Size of the transport cell.
    transport_ind : int
        Direction of transport. Can be 0, 1, 2.
    shift : tuple
        Shift in the transport cell system.
        It is expected to be 0 / transport_cell_size / -transport_cell_size
        in the transport direction and arbitrary in the other directions.
        Shift of (0,0,0) constructs the diagonal transport cell.
        Shift of (2,2,2) constructs the second off-diagonal transport cell
        for the connection (2,2) between the center matrix and the periodic image.

    Returns
    -------
    NDArray
        The transport cell hamiltonian block.

    """

    if shift[transport_ind] not in [0, transport_cell_size, -transport_cell_size]:
        raise ValueError(
            f"Shift in the transport direction must be 0, transport_cell_size, or -transport_cell_size. "
            f"Got shift={shift} and transport_cell_size={transport_cell_size}."
        )

    unit_cell_shape = matrix_dict[(0, 0, 0)].shape
    unit_cell_dtype = matrix_dict[(0, 0, 0)].dtype

    rows = []
    for r_i in range(transport_cell_size):
        row = []
        for r_j in range(transport_cell_size):

            coord = list(shift)
            coord[transport_ind] += r_j - r_i
            coord = tuple(int(i) for i in coord)

            fliped_coord = tuple(-int(i) for i in coord)

            if coord in matrix_dict:
                block = matrix_dict[coord]
            elif fliped_coord in matrix_dict and not np.all(np.array(shift) == 0):
                block = matrix_dict[fliped_coord].conj().T
            else:
                block = xp.zeros(unit_cell_shape, unit_cell_dtype)

            row.append(block)
        rows.append(xp.hstack(row))
    return xp.vstack(rows)


def _expand_tight_binding_matrix(
    matrix_dict: dict,
    num_transport_cells: int,
    transport_ind: int,
    block_start: int | None = None,
    block_end: int | None = None,
    periodic_shift: tuple = (0, 0, 0),
) -> sparse.csr_matrix:
    """Creates a full block-tridiagonal matrix from tight-binding matrix / Wannier Centers.

    The transport cell (same as supercell) is the cell that is repeated
    in the transport direction, and is only connected to nearest-neighboring cells.
    NOTE: interactions outside nearest neighbors are not included
    in the block-tridiagonal Hamiltonian (see below).

    Example for a tight-binding matrix with 3 cells in transport direction for (0,0,0) and (0,0,1),

      ------- -------           |   ------- -------
     | o o o | o x x | x        |  | o o o | o x x | x
     | x o o | o o x | x x      |  | o o o | o o x | x x
     | x x o | o o o | x x x    |  | o o o | o o o | x x x
      ------- ------- -------   |   ------- ------- -------
     | x x x | o o o | o x x |  |  | o o o | o o o | o x x |
     | x x x | x o o | o o x |  |  | x o o | o o o | o o x |
     | x x x | x x o | o o o |  |  | x x o | o o o | o o o |
      ------- ------- -------   |   ------- ------- -------
       x x x | x x x | o o o |  |    x x x | o o o | o o o |
         x x | x x x | x o o |  |      x x | x o o | o o o |
           x | x x x | x x o |  |        x | x x o | o o o |
              ------- -------   |           ------- -------

    for (0,0,0) only the upper diagonal part is expanded
    while for other shifts also the lower diagonal half of the matrix
    is expanded.

    Parameters
    ----------
    matrix_dict : dict
        Wannier unit cells.
    num_transport_cells : int
        Number of transport cells.
    transport_ind : int or str
        Direction of transport. Can be 0, 1, 2.
    block_start : int | None, optional
        Starting block index for arrow shape partition. Defaults to
        `None`.
    block_end : int | None, optional
        Ending block index for arrow shape partition. Defaults to
        `None`.
    periodic_shift : tuple, optional
        Incase the system is periodic in non-transport directions, the
        periodic shift can be used to get interactions between the
        transport cell and the periodic cells. E.g. (0, 0, 1) for one of
        the periodic shifts in the z-direction.

    Returns
    -------
    sparse.csr_matrix
        The block-tridiagonal Hamiltonian matrix.

    """

    if isinstance(transport_ind, str):
        transport_ind = "xyz".index(transport_ind)

    # NOTE: Max alone is not enough since only half the keys are expected to be present.
    transport_keys = np.array(list(matrix_dict.keys()))[:, transport_ind]
    transport_cell_size = np.max(np.abs(transport_keys))

    block_start = block_start or 0
    block_end = block_end or num_transport_cells
    if block_start >= block_end:
        raise ValueError("block_start must be smaller than block_end.")
    if block_end > num_transport_cells:
        raise ValueError("block_end must be smaller than num_transport_cells.")
    if block_start < 0:
        raise ValueError("block_start must be greater than or equal to 0.")

    if len(periodic_shift) != 3:
        raise ValueError("periodic_shift must have length 3.")

    if all(np.array(periodic_shift) == 0):
        transport_blocks_inds = [0, 1]
    else:
        transport_blocks_inds = [-1, 0, 1]

    block_inds = []
    for b in transport_blocks_inds:
        temp_list = list(periodic_shift)
        temp_list[transport_ind] = b * transport_cell_size
        block_inds.append(tuple(int(i) for i in temp_list))

    if block_inds[-1] not in matrix_dict.keys():
        warnings.warn(
            "Periodic shift is outside the available range. Interaction will be zero."
        )

    # Expand and convert to sparse matrices.
    # TODO: assumes matrices are dense for now
    blocks = [
        sparse.coo_matrix(
            _construct_transport_cell(
                matrix_dict=matrix_dict,
                transport_cell_size=transport_cell_size,
                transport_ind=transport_ind,
                shift=shift,
            )
        )
        for shift in block_inds
    ]

    # Canoncialize the sparse matrices.
    for block in blocks:
        if block.has_canonical_format is False:
            block.sum_duplicates()

    # Create the block-tridiagonal matrix.
    num_blocks = block_end - block_start
    block_size = blocks[0].shape[0]
    offsets = xp.arange(block_start, block_end) * blocks[0].shape[0]

    if all(np.array(periodic_shift) == 0):
        block_shifts = [(0, 0), (0, 1)]
    else:
        block_shifts = [(1, 0), (0, 0), (0, 1)]

    full_rows = []
    full_cols = []
    full_data = []
    for block, (row_shift, col_shift) in zip(blocks, block_shifts):
        rows = xp.tile(block.row, num_blocks) + xp.repeat(offsets, block.nnz)
        cols = xp.tile(block.col, num_blocks) + xp.repeat(offsets, block.nnz)
        data = xp.tile(block.data, num_blocks)

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

    return sparse.csr_matrix(
        (full_data, (full_rows, full_cols)),
        shape=(matrix_shape, matrix_shape),
    )


def _sum_operator(
    matrix_dict: dict[tuple, sparse.csr_matrix | NDArray],
    symmetric: bool,
    phases: dict | None = None,
):
    """Sums the contributions from different periodic repetitions
    of a hermitian operator (e.g., Hamiltonian, overlap) to construct the matrix
    for a specific k-point.

    Parameters
    ----------
    matrix_dict : dict[tuple, sparse.csr_matrix | NDArray]
        The dictionary of matrices corresponding to different periodic
        repetitions. It is assumed that only the upper part of the (0,0,0) cell
        and half the keys are present.
    symmetric : bool
        Whether the resulting matrix should be symmetric. If `True`, only
        construct the upper triangular part.
    phases :

    """

    if phases is None:
        phases = {coord: 1.0 for coord in matrix_dict.keys()}

    # NOTE: Sparse matrix addition is slow
    # but unavoidable due to memory constraints.
    # TODO: Could still be optimized
    summed_matrix = matrix_dict[(0, 0, 0)]
    for coord, matrix in matrix_dict.items():
        if coord == (0, 0, 0):
            continue
        phase = phases[coord]
        summed_matrix += (
            phase * sparse.triu(matrix) + (phase * sparse.tril(matrix)).conj().T
        )

    if not symmetric:
        summed_matrix = summed_matrix + summed_matrix.T.conj()
        summed_matrix.setdiag(summed_matrix.diagonal() / 2)

    return summed_matrix


def _assemble_kpoint(
    out_matrix: DSDBSparse,
    matrix_dict: dict[tuple, sparse.csr_matrix | NDArray],
    kpoint_grid: NDArray,
    kpoint_shift: NDArray,
    kshift: int | NDArray,
) -> None:
    """Assembles a DSBSparse from a dictionary of sparse matrices
    corresponding to different periodic repetitions.

    Parameters
    ----------
    out_matrix : DSDBSparse
        The matrix to assemble into.
    matrix_dict : dict[tuple, sparse.csr_matrix | NDArray]
        The dictionary of matrices corresponding to different periodic
        repetitions. It is assumed that only the upper part of the (0,0,0) cell
        and half the keys are present.
    kpoint_grid : NDArray
        The k-point grid.
    kshift : int | NDArray
        The k-point shift to apply.

    """

    num_dimensions = len(kpoint_grid)

    if isinstance(kshift, int):
        kshift = np.array([kshift for _ in range(num_dimensions)])

    if not matrix_dict:
        raise ValueError("No matrices found in matrix_dict.")

    for cell in matrix_dict.keys():
        if len(cell) != num_dimensions:
            raise ValueError(
                f"Cell {cell} has incorrect dimensionality. "
                f"Expected {num_dimensions}, got {len(cell)}."
            )

    if all(kpoint_grid == 1):
        out_matrix.stack[(...,)] += _sum_operator(matrix_dict, out_matrix.symmetry)
    else:

        kpoints = monkhorst_pack(kpoint_grid, kpoint_shift).reshape(
            tuple(kpoint_grid) + (-1,)
        )
        kpoints = np.roll(kpoints, shift=kshift, axis=tuple(range(num_dimensions)))

        index = np.argwhere(kpoint_grid > 1)[0]
        for stack_index in np.ndindex(kpoints.shape[:-1]):
            kpoint = kpoints[stack_index]
            stack_index = np.array(stack_index)
            stack_index = tuple(stack_index[index])

            phases = {
                coord: xp.exp(2j * np.pi * (np.asarray(coord) @ kpoint))
                for coord in matrix_dict.keys()
            }

            out_matrix.stack[(...,) + stack_index] += _sum_operator(
                matrix_dict, out_matrix.symmetry, phases=phases
            )


def _create_matrix_from_unit_cells(
    config: QuatrexConfig,
    matrix_dict: dict,
) -> dict:
    """Creates a matrix from unit cells with periodic shifts.

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex simulation configuration.
    matrix_dict : dict[tuple, sparse.csr_matrix | NDArray]
        The dictionary of matrices corresponding to different periodic
        repetitions. It is assumed that only the upper part of the (0,0,0) cell
        and half the keys are present.

    Returns
    -------
    dict
        The expanded matrices

    """
    # Determine the local slice of the data.
    # NOTE: This is arrow-wise partitioning.
    # TODO: Allow more options, e.g., block row-wise partitioning.
    section_sizes, __ = get_section_sizes(
        config.device.num_transport_cells, comm.block.size
    )
    section_offsets = np.hstack(([0], np.cumsum(section_sizes)))
    start_block = section_offsets[comm.block.rank]
    end_block = section_offsets[comm.block.rank + 1]

    # Create a matrix for each connecting layer along the transverse
    # directions.
    out_matrix_dict = {}

    transport_ind = "xyz".index(config.device.transport_direction)
    for coord in matrix_dict.keys():

        # Do not expand multiple time in
        # transport direction
        if coord[transport_ind] > 0:
            continue

        matrix_sparray = _expand_tight_binding_matrix(
            matrix_dict=matrix_dict,
            num_transport_cells=config.device.num_transport_cells,
            transport_ind=transport_ind,
            block_start=start_block,
            block_end=end_block,
            periodic_shift=coord,
        )
        out_matrix_dict[coord] = matrix_sparray.astype(xp.complex128)

    return out_matrix_dict


def load_matrices(
    config: QuatrexConfig,
    matrix_name: str,
):
    """Loads a Hermitian matrix from file

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex configuration.
    matrix_name : str
        The name of the matrix ('hamiltonian', 'overlap', etc.).


    Returns
    -------
    dict
        The dict of sparse matrices corresponding to different periodic repetitions.
        It is assumed that only the upper part of the (0,0,0) cell and half
        the keys are stored.

    """

    # load the matrices
    matrix_dict = distributed_load(config.input_dir / f"{matrix_name}.mat")

    if (0, 0, 0) not in matrix_dict.keys():
        raise ValueError(
            f"Expected to find a key [0,0,0] in the matrix file, but it was not found. "
            f"Available keys: {list(matrix_dict.keys())}"
        )

    # assert that the keys form a complete grid,
    keys = np.array(list(matrix_dict.keys()))
    max_coords = keys.max(axis=0)
    min_coords = keys.min(axis=0)

    if not np.all(np.abs(min_coords) == max_coords):
        raise ValueError(
            f"Expected the keys to form a complete grid with symmetric positive and negative coordinates, "
            f"but found min_coords={min_coords} and max_coords={max_coords}."
        )

    expected_size = np.prod(max_coords - min_coords + 1)
    actual_size = len(matrix_dict)
    if expected_size != actual_size:
        raise ValueError(
            f"Expected {expected_size} unit cells based on the detected grid shape, "
            f"but found {actual_size} unit cells in the matrix file."
        )

    # assert that more than the neighbor cell cutoff is available if the cutoff is requested
    if config.device.neighbor_cell_cutoff is not None:
        if any(max_coords[i] < config.device.neighbor_cell_cutoff[i] for i in range(3)):
            raise ValueError(
                "Matrix contains fewer neighbor cells than requested."
                f"({max_coords=}, {config.device.neighbor_cell_cutoff=})"
            )

    # drop half the keys
    matrix_dict = {
        coord: matrix
        for coord, matrix in matrix_dict.items()
        if coord > (0, 0, 0) or (coord == (0, 0, 0))
    }

    # assert that the matrix_dict have the same shape
    matrix_shape = matrix_dict[(0, 0, 0)].shape
    matrix_type = type(matrix_dict[(0, 0, 0)])
    for coord, matrix in matrix_dict.items():
        if matrix.shape != matrix_shape:
            raise ValueError(
                f"Matrix at coordinate {coord} has shape {matrix.shape}, "
                f"but expected shape is {matrix_shape}."
            )
        if not isinstance(matrix, matrix_type):
            raise ValueError(
                f"Matrix at coordinate {coord} has type {type(matrix)}, "
                f"but expected type is {matrix_type}."
            )

    # only keep the upper part of the (0,0,0) matrix
    # NOTE: this is done on the CPU
    if isinstance(matrix_dict[(0, 0, 0)], np.ndarray):
        matrix_dict[(0, 0, 0)] = np.triu(matrix_dict[(0, 0, 0)])
    elif isinstance(matrix_dict[(0, 0, 0)], sps.spmatrix):
        matrix_dict[(0, 0, 0)] = sps.triu(matrix_dict[(0, 0, 0)])
    else:
        raise ValueError(
            f"Matrix at coordinate (0,0,0) has unsupported type {type(matrix_dict[(0, 0, 0)])}."
        )

    # drop keys out side the neighbor cell cutoff if requested
    if config.device.neighbor_cell_cutoff is not None:
        matrix_dict = {
            coord: matrix
            for coord, matrix in matrix_dict.items()
            if all(
                abs(c) <= config.device.neighbor_cell_cutoff[i]
                for i, c in enumerate(coord)
            )
        }

    # transfer the matrix_dict to the GPU
    if isinstance(matrix_dict[(0, 0, 0)], np.ndarray):
        matrix_dict = {
            coord: xp.asarray(matrix).astype(xp.complex128)
            for coord, matrix in matrix_dict.items()
        }
    elif isinstance(matrix_dict[(0, 0, 0)], sps.spmatrix):
        matrix_dict = {
            coord: sparse.csr_matrix(matrix).astype(xp.complex128)
            for coord, matrix in matrix_dict.items()
        }

    # expand potentially if the system is periodic
    # and given bz unit cell matrix_dict
    if config.device.construct_from_unit_cell:
        # raise NotImplementedError("Constructing from unit cell is not implemented yet.")
        matrix_dict = _create_matrix_from_unit_cells(config, matrix_dict)

    # NOTE: for closed systems,
    # transport direction will be None
    transport_ind = "xyz".index(config.device.transport_direction)

    # drop keys which are bigger than zero in the transport direction
    matrix_dict = {
        coord: matrix
        for coord, matrix in matrix_dict.items()
        if coord[transport_ind] == 0
    }

    # make sure that the matrices are canonical
    for matrix in matrix_dict.values():
        if not matrix.has_canonical_format:
            matrix.sum_duplicates()
            matrix.sort_indices()

    return matrix_dict


def assemble_matrix(
    config: QuatrexConfig,
    matrix_name: str,
    sparsity_pattern: sparse.coo_matrix | None = None,
    shift_kpoints: bool = False,
) -> tuple[DSDBSparse, sparse.coo_matrix]:
    """Loads a Hermitian matrix from file and optionally
    applies a provided sparsity pattern.

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex configuration.
    matrix_name : str
        The name of the matrix ('hamiltonian', 'overlap', etc.).
    sparsity_pattern : sparse.coo_matrix | None
        The sparsity pattern to enforce. If None, the sparsity of the
        loaded matrix is used.
    shift_kpoints : bool
        Whether to "shift"/"center" the kpoints in the allocated
        DSDBSparse.

    Returns
    -------
    matrix : DSDBSparse
        The loaded matrix.
    sparsity_pattern : sparse.coo_matrix
        The sparsity pattern of the returned matrix.

    """

    matrix_dict = load_matrices(config, matrix_name)

    # load or construct the block sizes for the DSDBSparse
    if config.device.construct_from_unit_cell:
        expanded_shape = matrix_dict[(0, 0, 0)].shape
        block_sizes = [
            expanded_shape[0] // config.device.num_transport_cells
        ] * config.device.num_transport_cells
    else:
        block_sizes = config.device.block_size
        if isinstance(block_sizes, int):
            num_blocks, remainder = divmod(matrix_dict[(0, 0, 0)].shape[0], block_sizes)
            if remainder != 0:
                raise ValueError(
                    f"Block size {block_sizes} does not evenly divide the number of orbitals {matrix_dict[(0,0,0)].shape[0]}."
                )
            block_sizes = [block_sizes] * num_blocks

    block_sizes = np.array(block_sizes)

    if sparsity_pattern is None:
        # TODO: This could lead to cancelations
        # and then the sparsity pattern is not the true union
        matrix_sparray = _sum_operator(matrix_dict, config.scba.symmetric)
        sparsity_pattern = matrix_sparray.copy()
        sparsity_pattern.data[:] = 1
        sparsity_pattern = sparsity_pattern + sparsity_pattern.T

    matrix = config.compute.dsdbsparse_type.from_sparray(
        sparsity_pattern.astype(xp.complex128),
        block_sizes=block_sizes,
        global_stack_shape=(comm.stack.size,)
        + tuple([k for k in config.device.kpoint_grid if k > 1]),
        symmetry=config.scba.symmetric,
        symmetry_op=xp.conj,
        bits=config.compute.num_bits,
    )
    matrix.data[:] = 0.0  # Initialize to zero.

    # Shift the k-points if requested
    # Needed for the coulomb matrix
    _assemble_kpoint(
        out_matrix=matrix,
        matrix_dict=matrix_dict,
        kpoint_grid=np.array(config.device.kpoint_grid),
        kpoint_shift=np.array(config.device.kpoint_shift),
        kshift=-(np.array(config.device.kpoint_grid) // 2) if shift_kpoints else 0,
    )

    return matrix, sparsity_pattern


def distributed_read_xyz(filename: Path) -> tuple[NDArray, NDArray, NDArray]:
    """Reads atomic structure data from an XYZ file.

    Parameters
    ----------
    filename : Path
        Path to the XYZ file containing the atomic structure. The file
        should have the standard XYZ format with lattice parameters on
        the second line.

    Returns
    -------
    lattice : NDArray
        3x3 array containing the lattice vectors (in rows).
    atom_coordinates : NDArray
        (N_atoms, 3) array containing atomic coordinates.
    atom_types : NDArray
        (N_atoms,) array containing atom symbol for each atom.

    """

    lattice_vectors = None
    atom_coordinates = None
    atom_types = None

    if comm_world.rank == 0:
        # Read only the second line of the file (this contains the
        # lattice parameters)
        with open(filename, "r") as f:
            __ = f.readline()
            lattice_line = f.readline().strip()

        if not lattice_line.startswith("Lattice="):
            raise ValueError(
                f"Invalid lattice line in {filename}. Expected 'Lattice=', got '{lattice_line}'"
            )

        lattice_vectors = lattice_line.split("=")[1].strip().split('"')[1]
        lattice_vectors = np.fromstring(
            lattice_vectors, dtype=np.float64, sep=" "
        ).reshape(3, 3)
        atom_coordinates = np.loadtxt(filename, skiprows=2, usecols=(1, 2, 3))
        atom_types = np.loadtxt(filename, skiprows=2, usecols=(0,), dtype=str)

    # Broadcast the data to all the ranks
    lattice_vectors = comm_world.bcast(lattice_vectors, root=0)
    atom_coordinates = comm_world.bcast(atom_coordinates, root=0)
    atom_types = comm_world.bcast(atom_types, root=0)

    return lattice_vectors, atom_coordinates, atom_types
