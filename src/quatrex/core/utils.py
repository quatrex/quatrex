# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
from qttools import NDArray, sparse, xp
from qttools.datastructures import DSDBSparse
from qttools.datastructures.dsdbsparse import _block_view


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
    lattice_matrix: dict[tuple, sparse.csr_matrix],
    number_of_kpoints: xp.ndarray,
    roll_index: int | xp.ndarray,
) -> DSDBSparse:
    """Assembles a DSBSparse with the k-point distribution."""
    if isinstance(roll_index, int):
        roll_index = xp.array([roll_index, roll_index, roll_index])
    for i, ii in enumerate(xp.roll(xp.arange(number_of_kpoints[0]), roll_index[0])):
        for j, jj in enumerate(xp.roll(xp.arange(number_of_kpoints[1]), roll_index[1])):
            for k, kk in enumerate(
                xp.roll(xp.arange(number_of_kpoints[2]), roll_index[2])
            ):
                stack_index = tuple(
                    [i]
                    if number_of_kpoints[0] > 1
                    else (
                        [] + [j]
                        if number_of_kpoints[1] > 1
                        else [] + [k] if number_of_kpoints[2] > 1 else []
                    )
                )
                ik = (ii - number_of_kpoints[0] // 2) / number_of_kpoints[0]
                jk = (jj - number_of_kpoints[1] // 2) / number_of_kpoints[1]
                kk = (kk - number_of_kpoints[2] // 2) / number_of_kpoints[2]
                for cell_index in lattice_matrix.keys():
                    buffer.stack[(...,) + stack_index] += (
                        xp.exp(
                            2
                            * xp.pi
                            * 1j
                            * (
                                ik * cell_index[0]
                                + jk * cell_index[1]
                                + kk * cell_index[2]
                            )
                        )
                        * lattice_matrix[cell_index]
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
