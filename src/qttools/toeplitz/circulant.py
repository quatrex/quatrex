# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes functions for block phi circulant matrices."""

from qttools import NDArray, xp
from qttools.toeplitz.toeplitz import construct_transport_cell


def _canonical_offset(diff: int, n: int) -> int:
    """Maps a raw index difference to its canonical (smallest-magnitude)
    representative modulo `n`, matching how offsets are keyed in the
    original matrix_dict (e.g. -1, 0, 1 rather than 0..n-1).

    Parameters
    ----------
    diff : int
        The raw index difference.
    n : int
        The number of sections in the block circulant structure.

    Returns
    -------
    int
        The canonical index difference modulo `n`.

    """
    r = diff % n
    if r > n // 2:
        r -= n
    return r


def _upscale_single_dimension(
    matrix_dict: dict,
    sections: int,
    dim: int,
    phase_shift: complex = 1.0,
) -> dict:
    """Upscales a single dimension of the tight binding matrix to a
    block phi circulant matrix with the given number of sections.

    The matrix is upscaled in the transverse directions, while the
    transport direction is kept unchanged.

    For example, given a `matrix_dict` with size `[3, 2, 2]`, the
    resulting dict will have size `[3, 2, 1]` if `dim=1` and size `[3,
    1, 2]` if `dim=2`. It is assumed that all nessecary keys are present
    in the `matrix_dict`.

    Parameters
    ----------
    matrix_dict : dict
        The tight binding matrix to upscale.
    sections : int
        The number of sections in the block phi circulant structure.
    dim : int
        The dimension to upscale.
    phase_shift : complex, optional
        The phase factor to apply to the blocks, by default 1.0.

    Returns
    -------
    dict
        The upscaled block phi circulant matrix.

    """
    example = next(iter(matrix_dict.values()))
    norb = example.shape[0]

    # Ensure dtype supports complex numbers if phase is complex
    orig_dtype = next(iter(matrix_dict.values())).dtype
    dtype = xp.result_type(orig_dtype, phase_shift)
    zero_block = xp.zeros((norb, norb), dtype=dtype)

    # Group keys by every component except the one on `dim`.
    groups: dict[tuple, list[tuple]] = {}
    for key in matrix_dict:
        other = key[:dim] + key[dim + 1 :]
        groups.setdefault(other, []).append(key)

    new_dict = {}
    for other in groups:
        big = xp.zeros((sections * norb, sections * norb), dtype=dtype)
        for p in range(sections):
            for q in range(sections):
                raw = q - p
                r = _canonical_offset(raw, sections)
                lookup_key = other[:dim] + (r,) + other[dim:]
                block = matrix_dict.get(lookup_key, zero_block)

                if p > q:
                    factor = phase_shift
                else:
                    factor = 1.0

                big[p * norb : (p + 1) * norb, q * norb : (q + 1) * norb] = (
                    factor * block
                )

        new_key = other[:dim] + (0,) + other[dim:]
        new_dict[new_key] = big

    return new_dict


def expand_transverse(
    matrix_dict: dict,
    transport_ind: int,
    sections: tuple[int, int],
    phases: tuple[complex, complex] = (1.0, 1.0),
) -> dict:
    """Upscales a layer of the tight binding matrix to a block circulant
    matrix with the given number of sections.

    The matrix is upscaled in the transverse directions, while the
    transport direction is kept unchanged.

    For example, given a `matrix_dict` with size `[3, 2, 2]`, the
    resulting dict will have size `[3, 1, 1]`. It is assumed that all
    nessecary keys are present in the `matrix_dict`.

    Parameters
    ----------
    matrix_dict : dict
        The tight binding matrix to upscale.
    transport_ind : int
        The index of the transport direction.
    sections : tuple[int, int]
        The number of sections in the block circulant structure.
    phases : tuple[complex, complex], optional
        The phase factors to apply to the blocks in each transverse
        direction, by default (1.0, 1.0).

    Returns
    -------
    dict
        The upscaled block circulant matrix.

    """
    transverse_inds = [i for i in range(3) if i != transport_ind]
    upscaled = _upscale_single_dimension(
        matrix_dict=matrix_dict,
        sections=sections[0],
        dim=transverse_inds[0],
        phase_shift=phases[0],
    )
    upscaled = _upscale_single_dimension(
        matrix_dict=upscaled,
        sections=sections[1],
        dim=transverse_inds[1],
        phase_shift=phases[1],
    )
    return upscaled


def construct_circulant_cell(
    matrix_dict: dict,
    transport_cell_size: int,
    transport_ind: int,
    block_index: int,
    sections: tuple[int, int],
    phases: tuple[complex, complex] = (1.0, 1.0),
    key_assumption: str | None = None,
) -> NDArray:
    """Expands a unit cell matrix into a block matrix.
    This function first expands in transverse directions, and then
    constructs the transport cell block. To expand in transverse
    direction, a block circulant structure is assumed

    The function assumes that all necessary keys are present in the
    `matrix_dict`.

    Parameters
    ----------
    matrix_dict : dict
        The dictionary of matrices corresponding to different periodic
        repetitions.
    transport_cell_size : int
        Size of the transport cell.
    transport_ind : int
        Direction of transport. Can be 0, 1, 2.
    block_index : int
        The index of the block to expand. Can be either -1, 0, 1
        representing either the lower, diagonal, or upper block in the
        transport direction.
    sections : tuple[int, int]
        The number of sections in the transverse directions.
    phases : tuple[complex, complex], optional
        The phase shifts to apply in the transverse directions.
    key_assumption : str | None, optional
        Assumption on the keys in the matrix_dict. It must be either
        None, or "half". The assumption is only for the transport
        direction while for the transverse directions, it is assumed all
        keys are present.

    Returns
    -------
    NDArray
        The expanded block matrix.

    """
    if block_index not in [-1, 0, 1]:
        raise ValueError(f"Index must be -1, 0, or 1. Got {block_index}.")

    if key_assumption not in [None, "half"]:
        raise ValueError(
            f"key_assumption must be None, or 'half'. Got {key_assumption}."
        )

    # upscale first in transverse directions
    matrix_dict = expand_transverse(
        matrix_dict=matrix_dict,
        transport_ind=transport_ind,
        sections=sections,
        phases=phases,
    )

    return construct_transport_cell(
        matrix_dict=matrix_dict,
        transport_cell_size=transport_cell_size,
        transport_ind=transport_ind,
        block_index=block_index,
        key_assumption=key_assumption,
    )
