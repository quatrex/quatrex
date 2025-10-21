# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.
import numpy as np
from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.datastructures.dsdbsparse import _block_view
from qttools.utils.gpu_utils import get_host, get_device
from qttools.utils.input_utils import create_hamiltonian, cutoff_hr
from qttools.utils.mpi_utils import distributed_load, get_section_sizes


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
    lattice_matrix: dict[tuple, sparse.csr_matrix | xp.ndarray],
    number_of_kpoints: xp.ndarray,
    roll_index: int | xp.ndarray,
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
        np.roll(np.arange(number_of_kpoints[dim]), roll_index[dim])
        for dim in range(3)
    ]
    
    # Pre-compute k-point values
    k_values = [
        (rolled_indices[dim] - number_of_kpoints[dim] // 2) / number_of_kpoints[dim]
        for dim in range(3)
    ]

    # Convert lattice matrices to a list for faster indexing
    if isinstance(lattice_matrix[valid_cells[0]], xp.ndarray):
        lattice_matrices_array = xp.array([lattice_matrix[cell] for cell in valid_cells])
    elif isinstance(lattice_matrix[valid_cells[0]], sparse.csr_matrix):
        lattice_matrices_array = np.array([lattice_matrix[cell] for cell in valid_cells])
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
                    2 * xp.pi * 1j * (
                        ik * cell_array[:, 0] +
                        jk * cell_array[:, 1] +
                        kk * cell_array[:, 2]
                    )
                )
                if isinstance(lattice_matrices_array[0], xp.ndarray):
                    phases = phases[:, None, None]  # Reshape for broadcasting
                elif isinstance(lattice_matrices_array[0], sparse.csr_matrix):
                    phases = get_host(phases)
                
                # This sum is extremely slow when the lattice matrices are sparse.csr_matrix
                matrix_contribution = xp.sum(
                    phases * lattice_matrices_array, axis=0
                )
                buffer.stack[(...,) + stack_index] += sparse.csr_matrix(matrix_contribution)


def load_matrix_from_unit_cell(
    quatrex_config, matrix_name: str, use_r_cutoff: bool = True
) -> tuple[sparse.coo_matrix, dict | None, NDArray | None]:
    """Generic method to load a matrix from unit cell data.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The quatrex simulation configuration.
    matrix_name : str
        Name of the matrix ('hamiltonian' or 'overlap').
    use_r_cutoff : bool
        Whether to apply R_cutoff to the unit cells.

    Returns
    -------
    tuple[sparse.coo_matrix, dict | None, NDArray | None]
        The matrix, optional k-point dictionary, and optional block sizes.
    """
    unit_cells = distributed_load(
        quatrex_config.input_dir / f"{matrix_name}_unit_cells.npy"
    ).astype(xp.complex128)

    # Apply cutoff if requested and available
    if use_r_cutoff and quatrex_config.device.R_cutoff is not None:
        unit_cells = cutoff_hr(
            unit_cells,
            R_cutoff=quatrex_config.device.R_cutoff,
        )
    elif matrix_name == "overlap":
        # For overlap, use unit_cell_per_supercell as R_cutoff
        unit_cells = cutoff_hr(
            unit_cells,
            R_cutoff=quatrex_config.device.unit_cell_per_supercell,
        )

    return _create_matrix_from_unit_cells(quatrex_config, unit_cells)


def _create_matrix_from_unit_cells(
    quatrex_config, unit_cells
) -> tuple[sparse.coo_matrix, dict | None, NDArray | None]:
    """Generic method to create a matrix from unit cells with periodic shifts.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The quatrex simulation configuration.
    unit_cells : NDArray
        The unit cell data.

    Returns
    -------
    tuple[sparse.coo_matrix, dict | None, NDArray | None]
        The matrix, optional k-point dictionary, and optional block sizes.
    """
    # Determine the local slice of the data.
    # NOTE: This is arrow-wise partitioning.
    # TODO: Allow more options, e.g., block row-wise partitioning.
    section_sizes, __ = get_section_sizes(
        quatrex_config.device.number_of_supercells, comm.block.size
    )
    section_offsets = np.hstack(([0], np.cumsum(section_sizes)))
    start_block = section_offsets[comm.block.rank]
    end_block = section_offsets[comm.block.rank + 1]

    matrix_dict = {}
    # Create the matrix for each periodic shift
    for periodic_shift in xp.ndindex(
        tuple(2 * ps - 1 for ps in quatrex_config.device.cells_in_periodic_directions)
    ):
        periodic_shift = tuple(
            [
                ps - quatrex_config.device.cells_in_periodic_directions[i] + 1
                for i, ps in enumerate(periodic_shift)
            ]
        )
        matrix_sparray, block_sizes = create_hamiltonian(
            unit_cells,
            quatrex_config.device.number_of_supercells,
            quatrex_config.device.transport_direction,
            quatrex_config.device.unit_cell_per_supercell,
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
        [block_sizes[0]] * quatrex_config.device.number_of_supercells
    )

    return matrix_sparray, matrix_dict, block_sizes_array


def load_matrix_from_files(
    quatrex_config, matrix_name: str
) -> tuple[sparse.coo_matrix, dict | None, NDArray | None]:
    """Generic method to load a matrix from pre-computed files.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The quatrex simulation configuration.
    matrix_name : str
        Name of the matrix ('hamiltonian' or 'overlap' or 'coulomb_matrix').

    Returns
    -------
    tuple[sparse.coo_matrix, dict | None, NDArray | None]
        The matrix, optional k-point dictionary, and optional block sizes.
    """
    # Define file loading priority for each matrix type
    if matrix_name == "hamiltonian":
        file_patterns = [
            ("hamiltonian.npz", "npz"),
            ("hamiltonian.pkl", "pkl"),
        ]
    elif matrix_name == "overlap":  # overlap
        file_patterns = [
            ("overlap_matrix.pkl", "pkl"),
            ("overlap.npz", "npz"),
        ]
    elif matrix_name == "coulomb_matrix":  # coulomb_matrix
        file_patterns = [
            ("coulomb_matrix.pkl", "pkl"),
            ("coulomb_matrix.npz", "npz"),
        ]

    # Try loading files in priority order
    for filename, file_type in file_patterns:
        try:
            if file_type == "pkl":
                matrix_dict = distributed_load(quatrex_config.input_dir / filename)
                matrix_sparray = sum(matrix_dict.values())
                matrix_sparray.sum_duplicates()
                block_sizes = get_host(
                    distributed_load(quatrex_config.input_dir / "block_sizes.npy")
                )
                return matrix_sparray, matrix_dict, block_sizes

            else:  # npz
                matrix_sparray = distributed_load(
                    quatrex_config.input_dir / filename
                ).astype(xp.complex128)
                block_sizes = get_host(
                    distributed_load(quatrex_config.input_dir / "block_sizes.npy")
                )
                return matrix_sparray, None, block_sizes

        except FileNotFoundError:
            continue

    # If no files found, raise an error
    raise FileNotFoundError(
        f"No {matrix_name} files found in {quatrex_config.input_dir}"
    )


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
    grid : NDArray
        The grid points.
    interaction_cutoff : float
        The interaction cutoff.
    transport_direction : str, optional
        The transport direction, by default 'x'.
    strategy : str, optional
        The strategy to use, by default "box", where only the distance
        along the transport direction is considered. The other option is
        "sphere", where the usual Euclidean distance between points
        matters.

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
    block_sizes : list
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
