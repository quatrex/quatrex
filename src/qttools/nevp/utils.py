from qttools.utils.gpu_utils import xp


def contour_quadrature(radius: float, num_quad_points: int):
    """Computes quadrature points and weights for contour integration.

    This function computes the quadrature points and weights for
    contour integration on a circular contour of a given radius using
    the trapezoidal rule.

    Parameters
    ----------
    radius : float
        The radius of the circular contour.
    num_quad_points : int
        The number of quadrature points to use.

    Returns
    -------
    z, w : tuple[xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray]
        The quadrature points and weights.

    """
    zeta = xp.arange(num_quad_points) + 1
    z = radius * xp.exp(2j * xp.pi * (zeta + 0.5) / num_quad_points)

    w = xp.ones(num_quad_points) / num_quad_points

    # Reshape to broadcast over batch dimension.
    z = z.reshape(1, -1, 1, 1)
    w = w.reshape(1, -1, 1, 1)

    return z, w


def operator_inverse(
    a_xx: list[xp.ndarray],
    z,
    contour_type: xp.dtype,
    in_type: xp.dtype,
):
    half_num_blocks = len(a_xx) // 2
    tmp = [z ** (i - half_num_blocks) * a_x for i, a_x in enumerate(a_xx)]
    sum = tmp[0]
    for i in range(1, len(tmp)):
        sum += tmp[i]

    return xp.linalg.inv(sum.astype(contour_type)).astype(in_type)
