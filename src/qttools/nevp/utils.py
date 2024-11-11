# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, xp


def operator_inverse(
    a_xx: tuple[NDArray, ...],
    z: NDArray,
    contour_type: xp.dtype,
    in_type: xp.dtype,
) -> NDArray:
    """Computes the inverse of a matrix polynomial at sample points.

    Parameters
    ----------
    a_xx : tuple[NDArray, ...]
        The coefficients of the matrix polynomial.
    z : NDArray
        The sample points at which to compute the inverse.
    contour_type : xp.dtype
        The data type for the contour integration.
    in_type : xp.dtype
        The data type for the input matrices.

    Returns
    -------
    NDArray
        The inverse of the matrix polynomial.

    """
    b = len(a_xx) // 2
    operator = sum(z**n * a_xn for a_xn, n in zip(a_xx, range(-b, b + 1)))

    return xp.linalg.inv(operator.astype(contour_type)).astype(in_type)
