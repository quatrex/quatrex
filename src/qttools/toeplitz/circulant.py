# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes functions for block phi circulant matrices."""

from qttools import NDArray, xp


def check_phi_circulant(
    a: NDArray,
    sections: int,
    phase: complex = 1.0,
) -> bool:
    """Check if a matrix is block phi circulant with the given number of sections.

    Parameters
    ----------
    a : NDArray
        The matrix to check.
    sections : int
        The number of sections in the block circulant structure.
    phase : complex, optional
        The phase factor to apply to the blocks, by default 1.0.

    Returns
    -------
    bool
        True if a is block circulant with the given number of sections, False otherwise.

    """
    if a.shape[-1] % sections != 0:
        raise ValueError("The last dimension of a must be divisible by sections.")

    if a.shape[-2] != a.shape[-1]:
        raise ValueError(
            "The second to last dimension of a must be equal to the last dimension of a."
        )

    block_size = a.shape[-1] // sections
    # Take the first block-row (top n rows)
    block_layer = a[..., :block_size, :]
    blocks = xp.split(block_layer, sections, axis=-1)

    for i in range(sections):
        if i == 0:
            shifted_blocks = blocks
        else:
            # Scale wrapped blocks by phase, leave remaining blocks unscaled
            shifted_blocks = [phase * b for b in blocks[-i:]] + blocks[:-i]

        expected = xp.concatenate(shifted_blocks, axis=-1)
        actual = a[..., i * block_size : (i + 1) * block_size, :]

        if not xp.allclose(actual, expected):
            return False

    return True


def check_nested_phi_circulant(
    a: NDArray,
    sections: tuple[int, int],
    phases: tuple[complex, complex] = (1.0, 1.0),
) -> bool:
    """Check if a matrix is block phi circulant in 2D with the given
    number of sections.

    Parameters
    ----------
    a : NDArray
        The matrix to check.
    sections : tuple[int, int]
        The number of sections in the block circulant structure for each
        dimension.
    phases : tuple[complex, complex], optional
        The phase factors to apply to the blocks for each dimension, by
        default (1.0, 1.0).

    Returns
    -------
    bool
        True if a is block circulant in 2D with the given number of
        sections, False otherwise.

    """
    if not check_phi_circulant(a, sections[1], phases[1]):
        return False

    for i in range(sections[1]):
        for j in range(sections[1]):
            block_size = a.shape[-1] // sections[1]
            slice_i = slice(i * block_size, (i + 1) * block_size)
            slice_j = slice(j * block_size, (j + 1) * block_size)
            if not check_phi_circulant(a[slice_i, slice_j], sections[0], phases[0]):
                return False

    return True


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
