# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from qttools import NDArray, xp
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


def get_periodic_superblocks(
    a_ij: NDArray, a_ii: NDArray, a_ji: NDArray, block_sections: int
) -> NDArray:
    """Constructs a periodic superblock structure from the given blocks.

    The periodic superblock structure will repeat the left- and
    upper-most subblocks of the input block layer.

    This does the following:
     a_ii  a_ij
     a_ji
    | c d | e 0 |
    | b j | k l |
    -------------
    | a i |
    | 0 h |
    then the periodic layer would be
    | a b c d e |
    with block_sections = 2
    leading to periodic superblocks
    | c d | e 0 |
    | b c | d e |
    -------------
    | a b |
    | 0 a |

    If there are interactions g/f like
     a_ii  a_ij
     a_ji
    | c d | e g |
    | b j | k l |
    -------------
    | a i |
    | f h |
    they will be ignored without any warning

    Parameters
    ----------
    a_ji : NDArray
        The subdiagonal block made up of smaller subblocks.
    a_ii : NDArray
        The diagonal block made up of smaller subblocks.
    a_ij : NDArray
        The superdiagonal block made up of smaller subblocks.
    block_sections : int
        The number of subblocks each block is divided into. So if the
        block is of shape (n, n), the subblocks each have a shape of
        (n // block_sections, n // block_sections).

    Returns
    -------
    NDArray
        The periodic superblock structure.

    """

    if block_sections == 1:
        return a_ji, a_ii, a_ij

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
    periodic_layer = xp.vstack(
        (view_ji[block_sections::-1], view_ij[1 : 1 + block_sections])
    )

    # Stack the periodic layer to form a periodic superblock structure.
    subblock_shape = a_ii.shape[:-2] + (a_ii.shape[-1] // block_sections,) * 2
    periodic_blocks = xp.zeros(
        (block_sections, 3 * block_sections, *subblock_shape),
        dtype=a_ii.dtype,
    )
    for i in range(block_sections):
        periodic_blocks[i, i : 3 * block_sections - (block_sections - 1) + i] = (
            periodic_layer
        )

    # Recover the correct superblock structure form the subblocks.
    periodic_blocks = xp.concatenate(xp.concatenate(periodic_blocks, -2), -1)
    return _block_view(periodic_blocks, -1, 3)


def expand_periodic_superblocks(
    a_ji: NDArray,
    a_ii: NDArray,
    a_ij: NDArray,
    block_sections: int,
    repetitions: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Expands the periodic superblocks to a larger block structure.

    The periodic superblocks are constructed from the outermost subblocks of the input blocks.
    This function calls `get_periodic_superblocks` to construct the periodic superblocks,
    and then repeats the resulting structure.

    This does the following:
     a_ii  a_ij
     a_ji
    | c d | e 0 |
    | b j | k l |
    -------------
    | a i |
    | 0 h |
    then the periodic layer would be
    | a b c d e |
    with block_sections = 2
    leading to periodic superblocks
    | c d | e 0 |
    | b c | d e |
    -------------
    | a b |
    | 0 a |

    if we want to double, this would give us
    || c d | e 0 || 0 0 | 0 0 ||
    || b c | d e || 0 0 | 0 0 ||
    || a b | c d || e 0 | 0 0 ||
    || 0 a | b c || d e | 0 0 ||
    ----------------------------
    || 0 0 | a b ||
    || 0 0 | 0 a ||
    || 0 0 | 0 0 ||
    || 0 0 | 0 0 ||

    Similar to `get_periodic_superblocks`,
    extra interactions are ignored without any warning.

    NOTE: Similarly feature could be achieved
    by repeating the periodic layer

    Parameters
    ----------
    a_ji : NDArray
        The subdiagonal block made up of smaller subblocks.
    a_ii : NDArray
        The diagonal block made up of smaller subblocks.
    a_ij : NDArray
        The superdiagonal block made up of smaller subblocks.
    block_sections : int
        The number of subblocks each block is divided into. So if the
        block is of shape (n, n), the subblocks each have a shape of
        (n // block_sections, n // block_sections).
    repetitions : int
        The number of times to repeat the periodic superblock structure.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        The expanded subdiagonal, diagonal, and superdiagonal blocks.

    """

    if repetitions == 1 and block_sections == 1:
        return a_ji, a_ii, a_ij

    new_shape = list(a_ii.shape)
    new_shape[-1] = new_shape[-1] * repetitions
    new_shape[-2] = new_shape[-2] * repetitions

    a_ji_out = xp.zeros_like(a_ji, shape=new_shape)
    a_ii_out = xp.zeros_like(a_ii, shape=new_shape)
    a_ij_out = xp.zeros_like(a_ij, shape=new_shape)

    a_ji_tmp, a_ii_tmp, a_ij_tmp = get_periodic_superblocks(
        a_ji=a_ji,
        a_ii=a_ii,
        a_ij=a_ij,
        block_sections=block_sections,
    )

    if repetitions == 1:
        return a_ji_tmp, a_ii_tmp, a_ij_tmp

    n = a_ii.shape[-1]
    for i in range(repetitions):
        a_ii_out[
            ...,
            i * n : (i + 1) * n,
            i * n : (i + 1) * n,
        ] = a_ii_tmp

    for i in range(repetitions - 1):
        a_ii_out[
            ...,
            i * n : (i + 1) * n,
            (i + 1) * n : (i + 2) * n,
        ] = a_ij_tmp
        a_ii_out[
            ...,
            (i + 1) * n : (i + 2) * n,
            i * n : (i + 1) * n,
        ] = a_ji_tmp

    a_ij_out[..., -n:, :n] = a_ij_tmp
    a_ji_out[..., :n, -n:] = a_ji_tmp

    return a_ji_out, a_ii_out, a_ij_out
