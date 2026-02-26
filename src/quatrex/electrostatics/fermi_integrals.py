# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
import numpy as np
import scipy as sp
from scipy.optimize import minimize

from qttools import NDArray
from quatrex.electrostatics._params import (
    PARAMS_MINUS_ONE_HALF,
    PARAMS_PLUS_ONE_HALF,
    C,
)

# A few special values of the Gamma function, that do not need to be
# evaluated numerically.
_gamma_values = {
    -1 / 2: -2 * np.sqrt(np.pi),
    1 / 2: np.sqrt(np.pi),
    3 / 2: np.sqrt(np.pi) / 2,
}


def _gamma(n: float) -> float:
    """Computes the gamma function of n.

    Parameters
    ----------
    n : float
        The input value.

    Returns
    -------
    gamma_n : float
        The value of the gamma function at n.

    """
    if n in _gamma_values:
        return _gamma_values[n]

    return sp.special.gamma(n)


def fermi_integral(k: int, eta: NDArray, num_quad_points: int = 500) -> float:
    """Computes the Fermi integral of order k by quadrature.[^1].

    Extended (composite) trapezoidal quadrature rule with a variable
    transformation, x = exp( t - exp( t ) ). This method should be
    accurate with more than 500 points for an eta < 15.

    For k = -3/2, a different variable transformation is used to avoid
    numerical issues with the integrand in the other transform at x=0.

    For k = 0 and k = -1 analytical solutions exist, and those are
    returned.

    [^1]: W. H. Press et al., "Numerical recipies: The art of scientific
    computing", Cambridge University Press, 2007.

    Parameters
    ----------
    k : int
        The order of the Fermi integral. If k is 0 or -1, the analytical
        solution is returned.
    eta : NDArray
        The parameter eta. The method should be very accurate for
        eta < 15.
    num_quad_points : int
        The number of points. Default is 500.

    Returns
    -------
    integral : float
        The value of the Fermi integral.

    """
    if k == 0:
        # Analytical solution for k = 0.
        return np.log(1 + np.exp(eta))

    if k == -1:
        # Analytical solution for k = -1.
        return 1 / (1 + np.exp(-eta))

    # Make sure eta has the right shape for broadcasting.
    eta = np.atleast_1d(eta)

    if k == -3 / 2:
        # NOTE: For k = -3/2, the usual transformation produces wrong
        # values. Another option is to apply a different transformation
        # to the integral of order -1/2, compute the analytical
        # derivative of this integral with respect to eta, and use the
        # fact that the derivative of the Fermi integral of order k is
        # the Fermi integral of order k-1.
        a, b = 0, 7.0  # Found to be sufficient for eta < 15.
        t = np.linspace(a, b, num_quad_points)
        x = np.exp(t**2 - eta[..., np.newaxis])
        f = x / (1 + x) ** 2
        # NOTE: Factor two from variable transformation, and the gamma
        # function in the denominator of the -1/2 order Fermi
        # integral, which is needed to compute the derivative for
        # k=-3/2.
        return np.trapezoid(f, t, axis=1) * 2 / _gamma((k + 1) + 1)

    a, b = -4.5, 5.0
    t = np.linspace(a, b, num_quad_points)
    x = np.exp(t - np.exp(-t))
    f = x * (1 + np.exp(-t)) * x**k / (1 + np.exp(x - eta[..., np.newaxis]))
    return np.trapezoid(f, t) / _gamma(k + 1)


def R_j(t: float, P: NDArray, Q: NDArray) -> float:
    """Computes a rational polynomial function R_j(t)"""
    numerator = sum(P_n * t**n for n, P_n in enumerate(P))
    denominator = sum(Q_n * t**n for n, Q_n in enumerate(Q))
    return numerator / denominator


def _inverse_fermi_integral_numerical(
    k: float, u: NDArray, num_quad_points: int = 500
) -> float:
    """Numerically computes the inverse Fermi integral of order k.

    Parameters
    ----------
    k : float
        The order of the Fermi integral.
    u : NDArray
        The values for which we want to find the corresponding eta.
    num_quad_points : int
        The number of points to use in the quadrature for the Fermi
        integral.

    Returns
    -------
    eta : NDArray
        The values of eta such that fermi_integral(k, eta) = u.


    Raises
    ------
    RuntimeError
        If the optimization fails to converge.

    """

    def cost_function(eta: NDArray, k: float, u: float, num_quad_points: int):
        return np.linalg.norm(
            (fermi_integral(k, eta[..., np.newaxis], num_quad_points) - u) ** 2
        )

    # Minimize the cost function.
    result = minimize(cost_function, x0=np.zeros_like(u), args=(k, u, num_quad_points))

    if result.success:
        return result.x

    raise RuntimeError("Quadrature optimization failed to converge: " + result.message)


def _inverse_fermi_integral_approximate_plus_one_half(u: NDArray) -> NDArray:
    """Inverse of the Fermi integral of order 1/2.

    This function uses a piecewise rational approximation scheme to
    compute the inverse of the Fermi integral of order 1/2.

    Parameters
    ----------
    u : NDArray
        The value(s) for which we want to find the corresponding eta.
        Must be non-negative.

    Returns
    -------
    eta : NDArray
        The value(s) of eta such that fermi_integral(1/2, eta) = u.

    """
    u = np.atleast_1d(np.asarray(u))
    result = np.zeros_like(u, dtype=float)
    params = PARAMS_PLUS_ONE_HALF

    # Region 0
    mask_0 = u < params["R_0"].u_max
    if np.any(mask_0):
        u_0 = u[mask_0]
        R_0 = R_j(u_0, params["R_0"].P, params["R_0"].Q)
        result[mask_0] = np.log(u_0 * R_0)

    # Region 1
    mask_1 = (u >= params["R_0"].u_max) & (u < params["R_1"].u_max)
    if np.any(mask_1):
        t = params["R_1"].alpha + params["R_1"].beta * u[mask_1]
        result[mask_1] = R_j(t, params["R_1"].P, params["R_1"].Q)

    # Region 2
    mask_2 = (u >= params["R_1"].u_max) & (u < params["R_2"].u_max)
    if np.any(mask_2):
        t = params["R_2"].alpha + params["R_2"].beta * u[mask_2]
        result[mask_2] = R_j(t, params["R_2"].P, params["R_2"].Q)

    # Region 3
    mask_3 = (u >= params["R_2"].u_max) & (u < params["R_3"].u_max)
    if np.any(mask_3):
        t = params["R_3"].alpha + params["R_3"].beta * u[mask_3]
        result[mask_3] = R_j(t, params["R_3"].P, params["R_3"].Q)

    # Region 4
    mask_4 = (u >= params["R_3"].u_max) & (u < params["R_4"].u_max)
    if np.any(mask_4):
        t = params["R_4"].alpha + params["R_4"].beta * u[mask_4]
        result[mask_4] = R_j(t, params["R_4"].P, params["R_4"].Q)

    # Region S (u >= params["R_4"].u_max)
    mask_s = u >= params["R_4"].u_max
    if np.any(mask_s):
        s = 1.0 + params["R_S"].beta * u[mask_s] ** (4.0 / 3.0)
        R_S = R_j(s, params["R_S"].P, params["R_S"].Q)
        result[mask_s] = (R_S / (1 - s)) ** 0.5

    return result


def _inverse_fermi_integral_approximate_minus_one_half(u: NDArray) -> NDArray:
    """Inverse of the Fermi integral of order -1/2.

    This function uses a piecewise rational approximation scheme to
    compute the inverse of the Fermi integral of order -1/2.

    Parameters
    ----------
    u : NDArray
        The value(s) for which we want to find the corresponding eta.
        Must be non-negative.

    Returns
    -------
    eta : NDArray
        The value(s) of eta such that fermi_integral(-1/2, eta) = u.

    """
    u = np.atleast_1d(np.asarray(u))
    result = np.zeros_like(u, dtype=float)
    params = PARAMS_MINUS_ONE_HALF

    # Region 0
    mask_0 = u < params["R_0"].u_max
    if np.any(mask_0):
        y = params["R_0"].u_max - u[mask_0]
        R_0 = R_j(y, params["R_0"].P, params["R_0"].Q)
        result[mask_0] = np.log(u[mask_0] * R_0)

    # Region 1
    mask_1 = (u >= params["R_0"].u_max) & (u < params["R_1"].u_max)
    if np.any(mask_1):
        y = u[mask_1] - params["R_0"].u_max
        result[mask_1] = R_j(y, params["R_1"].P, params["R_1"].Q)

    # Region 2
    mask_2 = (u >= params["R_1"].u_max) & (u < params["R_2"].u_max)
    if np.any(mask_2):
        y = u[mask_2] - params["R_1"].u_max
        result[mask_2] = R_j(y, params["R_2"].P, params["R_2"].Q)

    # Region 3
    mask_3 = (u >= params["R_2"].u_max) & (u < params["R_3"].u_max)
    if np.any(mask_3):
        y = u[mask_3] - params["R_2"].u_max
        result[mask_3] = R_j(y, params["R_3"].P, params["R_3"].Q)

    # Region 4
    mask_4 = (u >= params["R_3"].u_max) & (u < params["R_4"].u_max)
    if np.any(mask_4):
        y = u[mask_4] - params["R_3"].u_max
        result[mask_4] = R_j(y, params["R_4"].P, params["R_4"].Q)

    # Region 5 (u >= params["R_4"].u_max)
    mask_5 = u >= params["R_4"].u_max
    if np.any(mask_5):
        t = C / u[mask_5] ** 4
        y = 1 - t
        R_5 = R_j(y, params["R_5"].P, params["R_5"].Q)
        result[mask_5] = (R_5 / t) ** 0.5

    return result


def inverse_fermi_integral(
    k: float,
    u: float,
    method: str = "numerical",
    num_quad_points: int = 500,
):
    """Computes the inverse Fermi integral of order k.

    This function finds the value of eta such that the Fermi integral
    of order k equals u, i.e., it solves the equation
    fermi_integral(k, eta) = u.

    Parameters
    ----------
    k : float
        The order of the Fermi integral. If k is 0, the analytical
        solution is always used, irrespective of the specified method.
    u : float
        The value for which we want to find the corresponding eta.
    method : str, optional
        The method to use to determine the inverse Fermi integral. If
        "numerical" (the default), it uses an optimization method to
        minimize the difference between the Fermi integral and u. For k
        = -1/2 and k = 1/2 minimax rational approximation schemes
        [^1][^2] are implemented and can be used by setting method to
        "approximate".
    num_quad_points : int
        The number of points to use in the quadrature for the Fermi
        integral.

    [^1]: T. Fukushima, "Analytical computation of inverse Fermi-Dirac
    integral of order -1/2 by piecewise rational function
    approximation", 2020.
    [^2]: T. Fukushima, "Precise and fast computation of inverse
    Fermi-Dirac integral of order 1/2 by minimax rational function
    approximation", 2015.

    Returns
    -------
    eta : float
        The value of eta such that fermi_integral(k, eta) = u.


    """
    if np.any(u < 0):
        raise ValueError("u must be non-negative")

    if k == 0:
        # Analytical solution for k = 0.
        return np.log(np.exp(u) - 1)

    if k == -1:
        # Analytical solution for k = -1.
        return -np.log(1 / u - 1)

    if method == "numerical":
        return _inverse_fermi_integral_numerical(k, u, num_quad_points)

    if method == "approximate":
        if k == -0.5:
            return _inverse_fermi_integral_approximate_minus_one_half(u)
        if k == 0.5:
            return _inverse_fermi_integral_approximate_plus_one_half(u)

        raise ValueError(
            "Approximate method only implemented for k = -1/2 and k = 1/2."
        )

    raise ValueError("Invalid method specified. Use 'numerical' or 'approximate'.")
