# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import warnings

import numpy as np
import numpy.linalg as npla

from qttools.obc.obc import OBC


class SanchoRubio(OBC):
    """Calculates the surface Green's function iteratively.

    Parameters
    ----------
    max_iterations : int, optional
        The maximum number of iterations to perform.
    convergence_tol : float, optional
        The convergence tolerance for the iterative scheme. The
        criterion for convergence is that the average Frobenius norm of
        the update matrices `alpha` and `beta` is less than this value.

    References
    ----------
    .. [1] M.P. López-Sancho, Jose Lopez Sancho, Jessy Rubio. (2000).
       Highly convergent schemes for the calculation of bulk and surface
       Green-Functions. Journal of Physics F: Metal Physics. 15. 851.

    """

    def __init__(self, max_iterations: int = 1000, convergence_tol: float = 1e-7):
        """Initializes the Sancho-Rubio OBC."""
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol

    def __call__(
        self,
        a_ii: np.ndarray,
        a_ij: np.ndarray,
        a_ji: np.ndarray,
        contact: str,
        out: None | np.ndarray = None,
    ) -> np.ndarray | None:

        epsilon = a_ii.copy()
        epsilon_s = a_ii.copy()
        alpha = a_ji.copy()
        beta = a_ij.copy()

        delta = float("inf")
        for __ in range(self.max_iterations):
            inverse = npla.inv(epsilon)

            epsilon = epsilon - alpha @ inverse @ beta - beta @ inverse @ alpha
            epsilon_s = epsilon_s - alpha @ inverse @ beta

            alpha = alpha @ inverse @ alpha
            beta = beta @ inverse @ beta

            delta = (
                np.linalg.norm(np.abs(alpha) + np.abs(beta), axis=(-2, -1)).max() / 2
            )

            if delta < self.convergence_tol:
                break

        else:  # Did not break, i.e. max_iterations reached.
            warnings.warn("Surface Green's function did not converge.", RuntimeWarning)

        x_ii = npla.inv(epsilon_s)

        if out is not None:
            out[...] = x_ii
            return

        return x_ii
