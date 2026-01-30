# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
from typing import Callable

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.datastructures.dsdbsparse import _block_view
from qttools.utils.gpu_utils import get_host
from qttools.utils.input_utils import create_hamiltonian, trim_tight_binding_matrix
from qttools.utils.mpi_utils import distributed_load, get_section_sizes
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig


def homogenize(matrix: DSDBSparse) -> None:
    """Homogenizes a matrix in stack distribution.

    Parameters
    ----------
    matrix : DSDBSparse
        The matrix to homogenize.
    """

    raise NotImplementedError()
    # assert xp.all(matrix.block_sizes == matrix.block_sizes[0])
    # if matrix.distribution_state != "stack":
    #     raise ValueError("Matrix must be in stack distribution")

    # for i in range(len(matrix.block_sizes) - 2):
    #     matrix.blocks[i + 1, i + 1] = matrix.blocks[0, 0]
    #     matrix.blocks[i + 1, i + 2] = matrix.blocks[0, 1]
    #     matrix.blocks[i + 2, i + 1] = matrix.blocks[1, 0]

    # matrix.blocks[-1, -1] = matrix.blocks[0, 0]
    # matrix.blocks[-1, -2] = matrix.blocks[1, 0]


def assemble_kpoint_dsb(
    buffer: DSDBSparse,
    lattice_matrix: dict[tuple, sparse.csr_matrix | NDArray],
    number_of_kpoints: NDArray,
    roll_index: int | NDArray,
    transport_direction: str | None = None,
) -> DSDBSparse:
    """Assembles a DSBSparse with the k-point distribution."""
    if isinstance(roll_index, int):
        roll_index = xp.array([roll_index, roll_index, roll_index])

    # Pre-filter cells based on transport direction
    if transport_direction is not None:
        transport_idx = "xyz".index(transport_direction)
        # Interacting cells in transport direction should not be included.
        valid_cells = [
            cell for cell in lattice_matrix.keys() if cell[transport_idx] == 0
        ]
    else:
        valid_cells = list(lattice_matrix.keys())

    if not valid_cells:
        return buffer

    # Convert valid_cells to array for vectorization
    cell_array = xp.array(valid_cells)

    # Pre-compute rolled indices
    rolled_indices = [
        xp.roll(xp.arange(number_of_kpoints[dim]), roll_index[dim]) for dim in range(3)
    ]

    # Pre-compute k-point values
    k_values = [
        (rolled_indices[dim] - number_of_kpoints[dim] // 2) / number_of_kpoints[dim]
        for dim in range(3)
    ]

    # Convert lattice matrices to a list for faster indexing
    if isinstance(lattice_matrix[valid_cells[0]], xp.ndarray):
        lattice_matrices_array = xp.array(
            [lattice_matrix[cell] for cell in valid_cells]
        )
    # Check if it is sparse.csr_matrix or sparse.coo_matrix
    elif isinstance(lattice_matrix[valid_cells[0]], sparse.csr_matrix):
        lattice_matrices_array = np.array(
            [lattice_matrix[cell] for cell in valid_cells]
        )
    elif isinstance(lattice_matrix[valid_cells[0]], sparse.coo_matrix):
        lattice_matrices_array = np.array(
            [lattice_matrix[cell].tocsr() for cell in valid_cells]
        )
    else:
        raise TypeError("Unsupported lattice matrix type.")

    # Loop over k-points
    for i in rolled_indices[0]:
        for j in rolled_indices[1]:
            for k in rolled_indices[2]:
                stack_index = tuple(
                    [i]
                    if number_of_kpoints[0] > 1
                    else (
                        [] + [j]
                        if number_of_kpoints[1] > 1
                        else [] + [k] if number_of_kpoints[2] > 1 else []
                    )
                )
                ik = k_values[0][i]
                jk = k_values[1][j]
                kk = k_values[2][k]

                phases = xp.exp(
                    2
                    * xp.pi
                    * 1j
                    * (
                        ik * cell_array[:, 0]
                        + jk * cell_array[:, 1]
                        + kk * cell_array[:, 2]
                    )
                )
                if isinstance(lattice_matrices_array[0], xp.ndarray):
                    phases = phases[:, None, None]  # Reshape for broadcasting
                elif isinstance(lattice_matrices_array[0], sparse.csr_matrix):
                    phases = get_host(phases)
                # This sum is extremely slow when the lattice matrices are sparse.csr_matrix
                matrix_contribution = xp.sum(phases * lattice_matrices_array, axis=0)
                buffer.stack[(...,) + stack_index] += sparse.csr_matrix(
                    matrix_contribution
                )


def _create_matrix_from_unit_cells(
    quatrex_config: QuatrexConfig,
    unit_cells: NDArray,
) -> tuple[sparse.coo_matrix, dict | None, NDArray | None]:
    """Creates a matrix from unit cells with periodic shifts.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The quatrex simulation configuration.
    unit_cells : NDArray
        The unit cell data.

    Returns
    -------
    tuple[sparse.coo_matrix, dict | None, NDArray | None]
        The matrix, optional k-point dictionary, and optional block
        sizes.

    """
    # Determine the local slice of the data.
    # NOTE: This is arrow-wise partitioning.
    # TODO: Allow more options, e.g., block row-wise partitioning.
    section_sizes, __ = get_section_sizes(
        quatrex_config.device.num_transport_cells, comm.block.size
    )
    section_offsets = np.hstack(([0], np.cumsum(section_sizes)))
    start_block = section_offsets[comm.block.rank]
    end_block = section_offsets[comm.block.rank + 1]
    transport_ind = "xyz".index(quatrex_config.device.transport_direction)

    # The neighbor cell cutoff along the transport direction determines
    # the size of the transport cell.
    unit_cells_per_transport_cell = [1, 1, 1]
    unit_cells_per_transport_cell[transport_ind] = (
        quatrex_config.device.neighbor_cell_cutoff[transport_ind]
    )

    matrix_dict = {}
    # Create a matrix for each connecting layer along the transverse
    # directions. The number of periodic cells is determined by the
    # shape of the unit cell data.
    for periodic_shift in xp.ndindex(unit_cells.shape[:3]):
        # Center the periodic shift around zero.
        periodic_shift = tuple(
            ps - (us // 2) for ps, us in zip(periodic_shift, unit_cells.shape[:3])
        )
        matrix_sparray, block_sizes = create_hamiltonian(
            hr=unit_cells,
            num_transport_cells=quatrex_config.device.num_transport_cells,
            transport_dir=quatrex_config.device.transport_direction,
            supercell_size=tuple(unit_cells_per_transport_cell),
            block_start=start_block,
            block_end=end_block,
            periodic_shift=periodic_shift,
            return_sparse=True,
        )
        matrix_dict[periodic_shift] = matrix_sparray.astype(xp.complex128)

    matrix_sparray = sum(matrix_dict.values())
    matrix_sparray.sum_duplicates()
    block_sizes = get_host(block_sizes)
    block_sizes_array = np.asarray(
        [block_sizes[0]] * quatrex_config.device.num_transport_cells
    )

    return matrix_sparray, matrix_dict, block_sizes_array


def _load_matrix_from_unit_cell(
    quatrex_config: QuatrexConfig,
    matrix_name: str,
) -> tuple[sparse.coo_matrix, dict | None, NDArray | None]:
    """Loads a matrix from unit cell data.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The quatrex simulation configuration.
    matrix_name : str
        Name of the matrix ('hamiltonian','overlap' or
        'coulomb_matrix').

    Returns
    -------
    tuple[sparse.coo_matrix, dict | None, NDArray | None]
        The matrix, optional k-point dictionary, and optional block sizes.

    """
    unit_cells = distributed_load(
        quatrex_config.input_dir / f"{matrix_name}_unit_cells.npy"
    ).astype(xp.complex128)

    # Apply cutoff if requested and available
    trimmed_unit_cells = trim_tight_binding_matrix(
        tight_binding_matrix=unit_cells,
        neighbor_cell_cutoff=quatrex_config.device.neighbor_cell_cutoff,
    )

    return _create_matrix_from_unit_cells(quatrex_config, trimmed_unit_cells)


def load_matrix(
    quatrex_config: QuatrexConfig,
    compute_config: ComputeConfig,
    matrix_name: str,
    sparsity_pattern: sparse.coo_matrix | None = None,
    shift_kpoints: bool = False,
    symmetry_op: Callable = xp.conj,
) -> tuple[DSDBSparse, sparse.coo_matrix]:
    """Loads a matrix from file, applying symmetrization and optionally
    using a provided sparsity pattern.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.
    matrix_name : str
        The name of the matrix ('hamiltonian', 'overlap', etc.).
    sparsity_pattern : sparse.coo_matrix | None
        The sparsity pattern to enforce. If None, the sparsity of the
        loaded matrix is used.
    shift_kpoints : bool
        Whether to "shift"/"center" the kpoints in the allocated
        DSDBSparse.
    symmetry_op : Callable, optional
        The symmetry operation to apply, by default xp.conj

    Returns
    -------
    matrix : DSDBSparse
        The loaded matrix.
    sparsity_pattern : sparse.coo_matrix
        The sparsity pattern of the returned matrix.

    """

    if quatrex_config.device.construct_from_unit_cell:
        matrix_sparray, matrix_dict, block_sizes = _load_matrix_from_unit_cell(
            quatrex_config, matrix_name
        )
    else:
        matrix_sparray = distributed_load(
            quatrex_config.input_dir / f"{matrix_name}.npz"
        ).astype(xp.complex128)
        block_sizes = get_host(
            distributed_load(quatrex_config.input_dir / "block_sizes.npy")
        )
        matrix_dict = None

    # TODO: This is not efficient and will be refactored when the inputs
    # are unified in (issue #214).
    if sparsity_pattern is None:
        sparsity_pattern = matrix_sparray.copy()
        sparsity_pattern.data[:] = 1
        # Make sure that the sparsity pattern is symmetric.
        sparsity_pattern = sparsity_pattern + sparsity_pattern.T

    # Symmetrize the data.
    matrix_sparray = 0.5 * (matrix_sparray + symmetry_op(matrix_sparray).T)

    matrix = compute_config.dsdbsparse_type.from_sparray(
        sparsity_pattern.astype(xp.complex128),
        block_sizes=block_sizes,
        global_stack_shape=(comm.stack.size,)
        + tuple([k for k in quatrex_config.electron.number_of_kpoints if k > 1]),
        symmetry=quatrex_config.scba.symmetric,
        symmetry_op=symmetry_op,
    )
    matrix.data[:] = 0.0  # Initialize to zero.
    if matrix_dict is None:
        matrix += matrix_sparray
    else:
        number_of_kpoints = xp.array(quatrex_config.electron.number_of_kpoints)
        roll_index = -(number_of_kpoints // 2) if shift_kpoints else 0
        assemble_kpoint_dsb(
            buffer=matrix,
            lattice_matrix=matrix_dict,
            number_of_kpoints=number_of_kpoints,
            roll_index=roll_index,
            transport_direction=quatrex_config.device.transport_direction,
        )
        # Explicitely try to free the memory
        del matrix_dict

    return matrix, sparsity_pattern


def compute_sparsity_pattern(
    positions: NDArray,
    cutoff_distance: float,
    transport_direction: str = "x",
    strategy: str = "box",
    start_idx: int = 0,
    end_idx: int = None,
    batch_size: int = 1000,
) -> sparse.coo_matrix:
    """Computes the sparsity pattern for the interaction matrix.

    Parameters
    ----------
    positions : NDArray
        The grid points.
    cutoff_distance : float
        The interaction cutoff.
    transport_direction : str, optional
        The transport direction, by default 'x'.
    strategy : str, optional
        The strategy to use, by default "box", where only the distance
        along the transport direction is considered. The other option is
        "sphere", where the usual Euclidean distance between points
        matters.
    start_idx : int, optional
        The start index for which to compute the sparsity pattern, by
        default 0.
    end_idx : int, optional
        The end index for which to compute the sparsity pattern, by
        default None.
    batch_size : int, optional
        The batch size for distance computations, by default 1000.

    Returns
    -------
    sparse.coo_matrix
        The sparsity pattern.

    """
    if strategy == "sphere":

        def distance(x, y):
            """Euclidean distance."""
            return xp.linalg.norm(x[..., xp.newaxis, :] - y[xp.newaxis, ...], axis=-1)

    elif strategy == "box":

        idx = {"x": 0, "y": 1, "z": 2}[transport_direction]

        def distance(x, y):
            """Distance along transport direction."""
            return xp.abs(x[..., idx][..., xp.newaxis] - y[..., idx][xp.newaxis, ...])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    end_idx = end_idx or len(positions)

    num_diags = end_idx - start_idx

    rows, cols = [], []

    for i in range(
        start_idx, max(start_idx + 2 * num_diags, len(positions)), batch_size
    ):
        positions_batch = positions[i : i + batch_size]
        distances = distance(positions, positions_batch)

        batch_cols, batch_rows = xp.where(distances < cutoff_distance)
        local_mask = (((batch_rows + i) >= start_idx) & (batch_cols >= start_idx)) & (
            ((batch_rows + i) < end_idx) | (batch_cols < end_idx)
        )

        cols.append(batch_cols[local_mask])
        rows.append(i + batch_rows[local_mask])

    rows, cols = xp.hstack(rows), xp.hstack(cols)
    return sparse.coo_matrix(
        (xp.ones_like(rows, dtype=xp.float32), (rows, cols)),
        shape=(len(positions), len(positions)),
    )


def compute_num_connected_blocks(
    sparsity_pattern: sparse.coo_matrix, block_sizes: NDArray
) -> int:
    """Computes the number of "connected" blocks in the sparsity pattern.

    This number of "connected" blocks is the number of blocks that need
    to be merged together to arrive at a block-tridiagonal matrix after
    multiplying the sparsity pattern with itself twice (s @ s @ s).

    This is a heuristic used to determine the block size for the
    screened Coulomb interaction.

    Parameters
    ----------
    sparsity_pattern : sparse.coo_matrix
        The sparsity pattern.
    block_sizes : NDArray
        The block sizes.

    Returns
    -------
    int
        The number of connected blocks.

    """

    s_01 = sparsity_pattern.tocsr()[
        : block_sizes[0], block_sizes[0] : int(sum(block_sizes[:2]))
    ]
    __, cols, __ = sparse.find(s_01)

    bandwidth = cols.max()
    triple_bandwidth = 3 * bandwidth

    if triple_bandwidth <= block_sizes[0]:
        return 1

    if triple_bandwidth <= sum(block_sizes[:2]):
        return 2

    return 3


def get_periodic_superblocks(
    a_ii: NDArray, a_ij: NDArray, a_ji: NDArray, block_sections: int
) -> NDArray:
    """Constructs a periodic superblock structure from the given blocks.

    The periodic superblock structure will repeat the left- and
    upper-most subblocks of the input block layer.

    Parameters
    ----------
    a_ii : NDArray
        The diagonal block made up of smaller subblocks.
    a_ij : NDArray
        The superdiagonal block made up of smaller subblocks.
    a_ji : NDArray
        The subdiagonal block made up of smaller subblocks.
    block_sections : int
        The number of subblocks each block is divided into. So if the
        block is of shape (n, n), the subblocks each have a shape of
        (n // block_sections, n // block_sections).

    Returns
    -------
    NDArray
        The periodic superblock structure.

    """
    # Stack the diagonal and superdiagonal blocks and divide them into
    # sublayers. We are interested in the first, outermost sublayer.
    view_ij = _block_view(xp.concatenate((a_ii, a_ij), -1), -2, block_sections)
    # Divide the sublayer into sublayers along the remaining axis.
    view_ij = _block_view(view_ij[0], -1, 2 * block_sections)

    # Stack the diagonal and subdiagonal blocks and divide them into
    # sublayers. Like before we are interested in the first, outermost
    # sublayer.
    view_ji = _block_view(xp.concatenate((a_ii, a_ji), -2), -1, block_sections)
    # Divide the sublayer into sublayers along the remaining axis.
    view_ji = _block_view(view_ji[0], -2, 2 * block_sections)

    # Stack the sublayers to form a periodic layer from the outermost
    # subblocks.
    periodic_layer = xp.vstack((view_ji[block_sections::-1], view_ij[1:]))

    # Stack the periodic layer to form a periodic superblock structure.
    subblock_shape = a_ii.shape[:-2] + (a_ii.shape[-1] // block_sections,) * 2
    periodic_blocks = xp.zeros(
        (block_sections, 3 * block_sections, *subblock_shape),
        dtype=a_ii.dtype,
    )
    for i in range(block_sections):
        periodic_blocks[i, :] = xp.roll(periodic_layer, i, axis=0)

    # Recover the correct superbblock structure form the subblocks.
    periodic_blocks = xp.concatenate(xp.concatenate(periodic_blocks, -2), -1)
    return _block_view(periodic_blocks, -1, 3)
