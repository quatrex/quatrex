# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.
from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBSparse


def homogenize(matrix: DSBSparse) -> None:
    """Homogenizes a matrix in stack distribution.

    Parameters
    ----------
    matrix : DSBSparse
        The matrix to homogenize.
    """

    assert xp.all(matrix.block_sizes == matrix.block_sizes[0])
    if matrix.distribution_state != "stack":
        raise ValueError("Matrix must be in stack distribution")

    for i in range(len(matrix.block_sizes) - 2):
        matrix.blocks[i + 1, i + 1] = matrix.blocks[0, 0]
        matrix.blocks[i + 1, i + 2] = matrix.blocks[0, 1]
        matrix.blocks[i + 2, i + 1] = matrix.blocks[1, 0]

    matrix.blocks[-1, -1] = matrix.blocks[0, 0]
    matrix.blocks[-1, -2] = matrix.blocks[1, 0]


def compute_sparsity_pattern(
    positions: NDArray,
    cutoff_distance: float,
    strategy: str = "box",
) -> sparse.coo_matrix:
    """Computes the sparsity pattern for the interaction matrix.

    Parameters
    ----------
    grid : NDArray
        The grid points.
    interaction_cutoff : float
        The interaction cutoff.
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
            return xp.linalg.norm(x - y, axis=-1)

    elif strategy == "box":

        def distance(x, y):
            """Distance along transport direction."""
            return xp.abs(x[..., 0] - y[..., 0])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    rows, cols = [], []
    for i, position in enumerate(positions):
        distances = distance(positions, position)
        interacting = xp.where(distances < cutoff_distance)[0]
        cols.extend(interacting)
        rows.extend([i] * len(interacting))

    rows, cols = xp.array(rows), xp.array(cols)
    return sparse.coo_matrix((xp.ones_like(rows, dtype=xp.float32), (rows, cols)))


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
