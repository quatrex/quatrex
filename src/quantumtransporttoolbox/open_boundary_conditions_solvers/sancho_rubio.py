# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors. All rights reserved.
import logging
import time

import numpy as np
import numpy.linalg as npla
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def sancho_rubio(
    a_ii: ArrayLike,
    a_ij: ArrayLike,
    a_ji: ArrayLike,
    max_iterations: int = 5000,
    max_delta: float = 1e-8,
) -> np.ndarray:
    """Calculates the surface Green's function iteratively.

    This function generalizes the iterative scheme for the calculation
    of surface Green's functions given in [1]_ in the sense that it can
    be applied to arbitrary periodic system matrices.

    Parameters
    ----------
    a_ii : array_like
        On-diagonal block of the system matrix.
    a_ij : array_like
        Super-diagonal block of the system matrix.
    a_ji : array_like, optional
        Sub-diagonal block of the system matrix.
    max_iterations : int, optional
        Maximum number of iterations, by default 5000.
    max_delta : float, optional
        Maximum relative change in the surface greens function, by
        default 1e-8.

    Returns
    -------
    x_ii : np.ndarray
        The surface Green's function.

    References
    ----------
    .. [1] M.P. López-Sancho, Jose Lopez Sancho, Jessy Rubio. (2000).
       Highly convergent schemes for the calculation of bulk and surface
       Green-Functions. Journal of Physics F: Metal Physics. 15. 851.

    """
    a_ii = np.asarray(a_ii)
    a_ij = np.asarray(a_ij)
    a_ji = np.asarray(a_ji)

    epsilon = a_ii.copy()
    epsilon_s = a_ii.copy()
    alpha = a_ji.copy()
    beta = a_ij.copy()

    delta = float("inf")
    t = time.perf_counter()
    for __ in range(max_iterations):
        inverse = npla.inv(epsilon)

        epsilon = epsilon - alpha @ inverse @ beta - beta @ inverse @ alpha
        epsilon_s = epsilon_s - alpha @ inverse @ beta

        alpha = alpha @ inverse @ alpha
        beta = beta @ inverse @ beta

        delta = np.sum(np.abs(alpha) + np.abs(beta)) / 2

        if delta < max_delta:
            logger.debug(
                f"Surface Green's function converged after {__} iterations "
                f"({time.perf_counter() - t:.2f} s)."
            )
            break

    else:  # Did not break, i.e. max_iterations reached.
        raise RuntimeError("Surface Green's function did not converge.")

    return npla.inv(epsilon_s)
