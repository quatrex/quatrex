# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
import numpy as np
from scipy.optimize import minimize

from qttools import NDArray
from quatrex.electrostatics._params import (
    PARAMS_MINUS_ONE_HALF,
    PARAMS_PLUS_ONE_HALF,
    C,
)


def fermi_integral(k: int, eta: float, num_quad_points: int = 500) -> float:
    """Computes the Fermi integral of order k by quadrature.[^1].

    Extended (composite) trapezoidal quadrature rule with a variable
    transformation, x = exp( t - exp( t ) ). This method should be
    accurate with more than 500 points for an eta < 15.

    [^1]: W. H. Press et al., "Numerical recipies: The art of scientific
    computing", Cambridge University Press, 2007.

    Parameters
    ----------
    k : int
        The order of the Fermi integral. If k is 0 or -1, the analytical
        solution is returned.
    eta : float
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

    a, b = -4.5, 5.0
    t = np.linspace(a, b, num_quad_points)
    x = np.exp(t - np.exp(-t))
    f = x * (1 + np.exp(-t)) * x**k / (1 + np.exp(x - eta))
    return np.trapezoid(f, t)


def R_j(t: float, P: NDArray, Q: NDArray) -> float:
    """Computes a rational polynomial function R_j(t)"""
    numerator = sum(P_n * t**n for n, P_n in enumerate(P))
    denominator = sum(Q_n * t**n for n, Q_n in enumerate(Q))
    return numerator / denominator


def _inverse_fermi_integral_numerical(
    k: float, u: float, num_quad_points: int = 500
) -> float:
    """Numerically computes the inverse Fermi integral of order k.

    Parameters
    ----------
    k : float
        The order of the Fermi integral.
    u : float
        The value for which we want to find the corresponding eta.
    num_quad_points : int
        The number of points to use in the quadrature for the Fermi
        integral.

    Returns
    -------
    eta : float
        The value of eta such that fermi_integral(k, eta) = u.


    Raises
    ------
    RuntimeError
        If the optimization fails to converge.

    """

    def cost_function(eta: float, k: float, u: float, num_quad_points: int):
        return (fermi_integral(k, eta, num_quad_points) - u) ** 2

    # Minimize the cost function.
    result = minimize(cost_function, x0=0.0, args=(k, u, num_quad_points))

    if result.success:
        return result.x[0]

    raise RuntimeError("Quadrature optimization failed to converge: " + result.message)


def _inverse_fermi_integral_approximate_plus_one_half(u: float):
    """Inverse of the Fermi integral of order 1/2.

    This function uses a piecewise rational approximation scheme to
    compute the inverse of the Fermi integral of order 1/2.

    Parameters
    ----------
    u : float
        The value for which we want to find the corresponding eta.
        Must be non-negative.

    """
    params = PARAMS_PLUS_ONE_HALF

    if u < params["R_0"].u_max:
        R_0 = R_j(u, params["R_0"].P, params["R_0"].Q)
        return np.log(u * R_0)

    if u < params["R_1"].u_max:
        t = params["R_1"].alpha + params["R_1"].beta * u
        return R_j(t, params["R_1"].P, params["R_1"].Q)

    if u < params["R_2"].u_max:
        t = params["R_2"].alpha + params["R_2"].beta * u
        return R_j(t, params["R_2"].P, params["R_2"].Q)

    if u < params["R_3"].u_max:
        t = params["R_3"].alpha + params["R_3"].beta * u
        return R_j(t, params["R_3"].P, params["R_3"].Q)

    if u < params["R_4"].u_max:
        t = params["R_4"].alpha + params["R_4"].beta * u
        return R_j(t, params["R_4"].P, params["R_4"].Q)

    s = 1.0 + params["R_S"].beta * u ** (4.0 / 3.0)
    R_S = R_j(s, params["R_S"].P, params["R_S"].Q)
    return (R_S / (1 - s)) ** 0.5


def _inverse_fermi_integral_approximate_minus_one_half(u: float):
    """Inverse of the Fermi integral of order -1/2.

    This function uses a piecewise rational approximation scheme to
    compute the inverse of the Fermi integral of order -1/2.

    Parameters
    ----------
    u : float
        The value for which we want to find the corresponding eta.
        Must be non-negative.

    """
    params = PARAMS_MINUS_ONE_HALF

    if u < params["R_0"].u_max:
        y = params["R_0"].u_max - u
        R_0 = R_j(y, params["R_0"].P, params["R_0"].Q)
        return np.log(u * R_0)

    if u < params["R_1"].u_max:
        y = u - params["R_0"].u_max
        return R_j(y, params["R_1"].P, params["R_1"].Q)

    if u < params["R_2"].u_max:
        y = u - params["R_1"].u_max
        return R_j(y, params["R_2"].P, params["R_2"].Q)

    if u < params["R_3"].u_max:
        y = u - params["R_2"].u_max
        return R_j(y, params["R_3"].P, params["R_3"].Q)

    if u < params["R_4"].u_max:
        y = u - params["R_3"].u_max
        return R_j(y, params["R_4"].P, params["R_4"].Q)

    t = C / u**4
    y = 1 - t
    R_5 = R_j(y, params["R_5"].P, params["R_5"].Q)
    return (R_5 / t) ** 0.5


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
    if u < 0:
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
