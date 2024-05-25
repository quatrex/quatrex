# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import numpy.linalg as npla

from qttools.obc.obc import OBC


class SanchoRubio(OBC):
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
    contact : str
        The contact side.

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
        """Returns the surface Green's function."""

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

            delta = np.sum(np.abs(alpha) + np.abs(beta)) / 2

            if delta < self.convergence_tol:
                break

        else:  # Did not break, i.e. max_iterations reached.
            raise RuntimeError("Surface Green's function did not converge.")

        x_ii = npla.inv(epsilon_s)

        if out is not None:
            out[...] = x_ii
            return

        return x_ii
