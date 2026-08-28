# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the methods to acclerate operations with Toeplitz matrices."""

import warnings

import numpy as np

from qttools import NDArray, xp
from qttools.datastructures import DSDBSparse
from qttools.datastructures.dsdbsparse import _block_view
from qttools.utils.gpu_utils import get_host


def homogenize(matrix: DSDBSparse) -> None:
    """Homogenizes a matrix in stack distribution.

    Parameters
    ----------
    matrix : DSDBSparse
        The matrix to homogenize.
    """

    raise NotImplementedError()


def periodize_layer(
    a_xx: tuple[NDArray, NDArray, NDArray],
    block_sections: int,
) -> NDArray:
    """Constructs a periodic superblock structure from the given blocks
    where the block are given in the order (a_ji, a_ii, a_ij).

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
        return a_xx

    a_ji, a_ii, a_ij = a_xx
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


def periodize_repeat_layer(
    a_xx: tuple[NDArray, NDArray, NDArray],
    block_sections: int,
    repetitions: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Expands the periodic superblocks to a larger block structure
    where the block are given in the order (a_ji, a_ii, a_ij).

    The periodic superblocks are constructed from the outermost subblocks of the input blocks.
    This function calls `periodize_layer` to construct the periodic superblocks,
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

    Similar to `periodize_layer`,
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
        return a_xx

    a_ji, a_ii, a_ij = a_xx

    new_shape = list(a_ii.shape)
    new_shape[-1] = new_shape[-1] * repetitions
    new_shape[-2] = new_shape[-2] * repetitions

    a_ji_out = xp.zeros_like(a_ji, shape=new_shape)
    a_ii_out = xp.zeros_like(a_ii, shape=new_shape)
    a_ij_out = xp.zeros_like(a_ij, shape=new_shape)

    a_xx_tmp = periodize_layer(
        a_xx=a_xx,
        block_sections=block_sections,
    )

    if repetitions == 1:
        return a_xx_tmp

    a_ji_tmp, a_ii_tmp, a_ij_tmp = a_xx_tmp

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


def construct_transport_cell(
    matrix_dict: dict,
    transport_cell_size: int,
    transport_ind: int,
    block_index: int,
    transverse_shift: tuple = (0, 0),
    key_assumption: str | None = None,
) -> NDArray:
    """Constructs a transport block from the unit cell.
    This expand the unit cell matrix into a block matrix for the
    transport cell, which is repeated in the transport direction.

    This function targets the case of real space matrices which are
    toeplitz in the transport direction.

    Parameters
    ----------
    matrix_dict : dict
        The dictionary of matrices corresponding to different periodic
        repetitions. It is assumed that only the upper parts are
        present.
    transport_cell_size : int
        Size of the transport cell.
    transport_ind : int
        Direction of transport. Can be 0, 1, 2.
    block_index : int
        The index of the block to expand. Can be either -1, 0, 1
        representing either the lower, diagonal, or upper block in the
        transport direction.
    transverse_shift : tuple, optional
        Shift in the transverse directions. The shift means for which
        real space coordinate the block should be constructed. The
        default is (0, 0).
    key_assumption : str or None
        Assumption on the keys in the matrix_dict. If it is None, it is
        assumed that all keys are present. If it is "upper", it is
        assumed that only the upper triangular part of the matrices are
        present, and the lower triangular part can be obtained by
        conjugate transpose. If it is "half", it is assumed that the
        full matrix is present for half the keys.

    Returns
    -------
    NDArray
        The transport cell hamiltonian block.

    """

    if block_index not in [-1, 0, 1]:
        raise ValueError(f"Index must be -1, 0, or 1. Got {block_index}.")

    if key_assumption not in [None, "upper", "half"]:
        raise ValueError(
            f"key_assumption must be None, 'upper', or 'half'. Got {key_assumption}."
        )

    unit_cell_shape = next(iter(matrix_dict.values())).shape
    unit_cell_dtype = next(iter(matrix_dict.values())).dtype
    zero_block = xp.zeros(unit_cell_shape, unit_cell_dtype)

    rows = []
    for r_i in range(transport_cell_size):
        row = []
        for r_j in range(transport_cell_size):

            coord = list(transverse_shift)
            coord.insert(transport_ind, block_index * transport_cell_size + r_j - r_i)

            coord = tuple(int(i) for i in coord)
            coord_flipped = tuple(-int(i) for i in coord)

            block = matrix_dict.get(coord)
            block_flipped = matrix_dict.get(coord_flipped)

            if block is not None:
                if key_assumption == "upper" and block_flipped is not None:
                    block = block + xp.triu(block_flipped, k=1).conj().swapaxes(-2, -1)
            elif key_assumption == "half" and block_flipped is not None:
                block = block_flipped.conj().swapaxes(-2, -1)
            else:
                block = zero_block

            row.append(block)

        rows.append(xp.concatenate(row, axis=-1))

    return xp.concatenate(rows, axis=-2)


def extract_subblocks(
    a_xx: tuple[NDArray, ...],
    block_sections: int,
) -> tuple[NDArray, ...]:
    """Extracts the smallest periodic layer from the largest periodic
    layer.

    Note
    ----
    This function checks the periodicity of the blocks and will warn if
    the periodicity is not satisfied.

    TODO This function could be reused in `periodize_layer` after
    checking the periodic layer logic.

    Parameters
    ----------
    a_xx : tuple[NDArray, ...]
        The largest periodic blocks.
    block_sections : int
        The number of sections to split the periodic layer into.

    Returns
    -------
    blocks : tuple[NDArray, ...]
        The non-zero blocks making up the matrix layer.

    """
    # Construct layer of periodic matrix in semi-infinite lead.
    if block_sections == 1:
        return a_xx

    # Get a nested block view of the layer.
    # TODO double check the logic here since it is different than in the next function.
    # It seems that the wrong part of `a_ji` is taken.
    view = _block_view(xp.concatenate(a_xx, axis=-1), -1, 3 * block_sections)
    view = _block_view(view, -2, block_sections)

    # Make sure that the reduction leads to periodic sublayers.
    relative_errors = xp.zeros(block_sections - 1)
    first_block_norm = xp.linalg.norm(view[0, :])
    for i in range(1, block_sections):
        relative_errors[i - 1] = (
            xp.linalg.norm(view[0, :] - xp.roll(view[i, :], -i, axis=0))
            / first_block_norm
        )

    if xp.max(relative_errors) > 1e-3:
        warnings.warn(
            f"Requested block sectioning is not periodic. ({xp.max(relative_errors):.2e})",
            RuntimeWarning,
        )

    # Select relevant blocks and remove empty ones.
    blocks = view[0, : -block_sections + 1]
    indices = np.where([get_host(xp.any(b)) for b in blocks])[0]

    if indices.size == 0 or len(blocks) <= 3:
        return tuple(blocks)

    n_data = min(indices[0], len(blocks) - 1 - indices[-1])

    # keep at least 3 central blocks
    n_limit = (len(blocks) - 3) // 2

    n = min(n_data, n_limit)

    return tuple(blocks[n:-n]) if n > 0 else tuple(blocks)


def upscale_subblocks(
    blocks: tuple[NDArray, ...],
    block_sections: int,
) -> tuple[NDArray, ...]:
    """Upscales the full blocks from the periodic layer.

    TODO: This function could be replaced by `periodize_repeat_layer`
    after checking the periodic layer logic.

    Parameters
    ----------
    blocks : tuple[NDArray, ...]
        The blocks of the periodic matrix.
    block_sections : int
        The number of sections to split the periodic matrix layer into.

    Returns
    -------
    blocks : tuple[NDArray, ...]
        The non-zero blocks making up the matrix layer.

    """
    if block_sections == 1:
        return blocks

    n_blocks = len(blocks)
    if n_blocks % 2 == 0:
        raise ValueError("Expected an odd number of coefficient blocks.")

    max_len = 2 * block_sections + 1
    if n_blocks > max_len:
        raise ValueError(
            f"Too many coefficient blocks ({n_blocks}) for the requested "
            f"block_sections ({block_sections}); expected at most {max_len}."
        )

    zero_block = xp.zeros_like(blocks[0])

    # Undo the symmetric trimming
    n_pad = (max_len - n_blocks) // 2
    band = (zero_block,) * n_pad + tuple(blocks) + (zero_block,) * n_pad

    # Undo the truncation
    row0 = band + (zero_block,) * (block_sections - 1)

    # Rebuild all `block_sections` periodic rows via cyclic shifts of row 0.
    row0_stack = xp.stack(row0, axis=0)
    rows = [xp.roll(row0_stack, i, axis=0) for i in range(block_sections)]

    # Re-tile the nested block grid back into a flat (N, 3N) matrix.
    row_arrays = [
        xp.concatenate([row[j] for j in range(3 * block_sections)], axis=-1)
        for row in rows
    ]
    matrix = xp.concatenate(row_arrays, axis=-2)

    # Split back into the 3 macro blocks.
    n = matrix.shape[-1] // 3
    return tuple(matrix[..., k * n : (k + 1) * n] for k in range(3))
