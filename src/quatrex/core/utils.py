# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.datastructures.dsdbsparse import _block_view
from qttools.utils.mpi_utils import get_section_sizes


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


def _one_sided_gradient(y, x=None, axis=0, direction="forward"):
    if x is None:
        x = np.arange(y.shape[axis])
    if not len(x) == y.shape[axis]:
        raise ValueError(
            "Length of x must match the size of y along the specified axis."
        )

    if direction == "forward":
        append_value = np.take(y, -1, axis=axis)
        append_value = append_value.reshape(
            [y.shape[i] if i != axis else 1 for i in range(y.ndim)]
        )
        y_diff = np.diff(y, append=append_value, axis=axis)
    elif direction == "backward":
        prepend_value = np.take(y, 0, axis=axis)
        prepend_value = prepend_value.reshape(
            [y.shape[i] if i != axis else 1 for i in range(y.ndim)]
        )
        y_diff = np.diff(y, prepend=prepend_value, axis=axis)

    dx = x[1] - x[0]
    if direction == "forward":
        x_diff = np.diff(x, append=x[-1] + dx)
    elif direction == "backward":
        x_diff = np.diff(x, prepend=x[0] - dx)
    # Reshape x_diff to broadcast correctly along the specified axis
    shape = [1] * y.ndim
    shape[axis] = len(x_diff)
    x_diff = x_diff.reshape(shape)
    gradient = y_diff / x_diff
    return gradient


def filtering_peaks_mask(
    matrix: DSDBSparse,
    energies: NDArray,
    peak_limit: float,
) -> DSDBSparse:
    """Calculates a mask for filtering peaks in the DSDBSparse matrix.

    Parameters
    ----------
    matrix : DSDBSparse
        The DSDBSparse matrix to filter.
    energies : NDArray
        The energies corresponding to the matrix.
    peak_limit : float
        The peak limit. Peaks above this limit will be filtered out.

    Returns
    -------
    DSDBSparse
        The filtered DSDBSparse data.
    """

    matrix_diag = matrix.diagonal()
    block_sizes = matrix.block_sizes
    block_offsets = matrix.block_offsets
    local_dos = []
    for i, (bsz, boff) in enumerate(zip(block_sizes, block_offsets)):
        matrix_density = matrix_diag[..., boff : boff + bsz].imag.mean(axis=-1)
        local_dos.append(xp.abs(matrix_density))

    local_dos = xp.array(local_dos)
    dos = comm.stack.all_gather_v(local_dos, axis=1, mask=matrix._stack_padding_mask)

    forward_gradient = _one_sided_gradient(dos, x=energies, axis=1, direction="forward")
    backward_gradient = _one_sided_gradient(
        dos, x=energies, axis=1, direction="backward"
    )
    mask = (
        (xp.min(forward_gradient, axis=0) < -peak_limit)
        | (xp.max(backward_gradient, axis=0) > peak_limit)
        | (xp.max(dos, axis=0) > 20)
    )

    section_sizes, __ = get_section_sizes(energies.size, comm.stack.size)
    section_offsets = np.hstack(([0], np.cumsum(section_sizes)))
    local_mask = mask[
        section_offsets[comm.stack.rank] : section_offsets[comm.stack.rank + 1]
    ]

    return local_mask
