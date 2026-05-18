# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from qttools import NDArray, sparse, xp


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
