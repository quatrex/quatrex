from qttools.utils.gpu_utils import xp


def operator_inverse(
    a_xx: list[xp.ndarray],
    z: xp.ndarray,
    contour_type: xp.dtype,
    in_type: xp.dtype,
) -> xp.ndarray:
    """Computes the inverse of a matrix polynomial at sample points.

    Parameters
    ----------
    a_xx : list[xp.ndarray]
        The coefficients of the matrix polynomial.
    z : xp.ndarray
        The sample points at which to compute the inverse.
    contour_type : xp.dtype
        The data type for the contour integration.
    in_type : xp.dtype
        The data type for the input matrices.

    Returns
    -------
    inv_sum : xp.ndarray
        The inverse of the matrix polynomial.

    """
    half_num_blocks = len(a_xx) // 2
    tmp = [z ** (i - half_num_blocks) * a_x for i, a_x in enumerate(a_xx)]
    sum = tmp[0]
    for i in range(1, len(tmp)):
        sum += tmp[i]

    return xp.linalg.inv(sum.astype(contour_type)).astype(in_type)
