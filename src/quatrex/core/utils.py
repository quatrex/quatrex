# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes utility methods. TODO will be refactored"""

from qttools import NDArray, sparse, xp


def compute_sparsity_pattern(
    positions: NDArray,
    cutoff_distance: float,
    transport_direction: str = "a",
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
        The transport direction, by default "a".
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

        idx = {"a": 0, "b": 1, "c": 2}[transport_direction]

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


def split_trapezoid(
    y: NDArray, x: NDArray, split: NDArray | float
) -> tuple[NDArray, NDArray]:
    r"""Integrates below and above a split point with the trapezoidal rule.

    The integrand is taken as piecewise linear on the grid, and the grid
    cell containing the split point is divided exactly at it, i.e.

    $$
    \int_{x_0}^{s} y(x) \mathrm{d}x \quad \text{and} \quad \int_{s}^{x_{N-1}}
    y(x) \mathrm{d}x,
    $$

    which add up to the trapezoidal integral over the full grid. Masking
    the integrand at the split point instead introduces an error of
    order $y(s) \Delta x / 2$, which matters when the integrand does
    not vanish there (e.g. the density of states of graphene at the
    charge neutrality level).

    Parameters
    ----------
    y : NDArray
        The integrand with the grid along the last axis.
    x : NDArray
        The (ascending) grid points.
    split : NDArray or float
        The split points, broadcastable to `y.shape[:-1]`. Points outside
        the grid are clamped to it.

    Returns
    -------
    below : NDArray
        The integral from the first grid point to the split point.
    above : NDArray
        The integral from the split point to the last grid point.

    """
    dx = xp.diff(x)
    cells = 0.5 * (y[..., 1:] + y[..., :-1]) * dx
    cumulative = xp.concatenate(
        [xp.zeros(y.shape[:-1] + (1,), dtype=cells.dtype), xp.cumsum(cells, axis=-1)],
        axis=-1,
    )

    split = xp.broadcast_to(xp.asarray(split, dtype=x.dtype), y.shape[:-1])
    cell = xp.clip(xp.searchsorted(x, split, side="right") - 1, 0, x.size - 2)
    fraction = xp.clip((split - x[cell]) / dx[cell], 0.0, 1.0)

    def _take(a: NDArray, indices: NDArray) -> NDArray:
        return xp.take_along_axis(a, indices[..., xp.newaxis], axis=-1)[..., 0]

    y_left = _take(y, cell)
    y_split = y_left + fraction * (_take(y, cell + 1) - y_left)
    below = _take(cumulative, cell) + 0.5 * fraction * dx[cell] * (y_left + y_split)

    return below, cumulative[..., -1] - below
