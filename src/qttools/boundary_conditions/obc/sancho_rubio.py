# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes the Sancho-Rubio open boundary condition solver."""

import warnings

from qttools import NDArray, xp
from qttools.boundary_conditions.obc.obc import OBCSolver
from qttools.kernels import linalg


class SanchoRubio(OBCSolver):
    """Calculates the surface Green's function iteratively.[^1].

    [^1]: M P Lopez Sancho et al., "Highly convergent schemes for the
    calculation of bulk and surface Green functions", 1985 J. Phys. F:
    Met. Phys. 15 851

    Parameters
    ----------
    max_iterations : int, optional
        The maximum number of iterations to perform.
    convergence_tol : float, optional
        The convergence tolerance for the iterative scheme. The
        criterion for convergence is that the average Frobenius norm of
        the update matrices `alpha` and `beta` is less than this value.

    """

    def __init__(self, max_iterations: int = 100, convergence_tol: float = 1e-6):
        """Initializes the Sancho-Rubio OBC."""
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol

    def __call__(
        self,
        a_xx: tuple[NDArray, ...],
        contact: str,
    ) -> NDArray:
        """Returns the surface Green's function.

        Parameters
        ----------
        a_xx : tuple[NDArray, ...]
            The boundary blocks of the system matrix.
        contact : str
            The contact to which the boundary blocks belong.

        Returns
        -------
        x_ii : NDArray
            The system's surface Green's function.

        """
        if len(a_xx) != 3:
            raise ValueError(
                f"Sancho-Rubio OBC requires exactly 3 boundary blocks, "
                f"but {len(a_xx)} were provided."
            )
        a_ji, a_ii, a_ij = a_xx

        epsilon = a_ii.copy()
        epsilon_s = a_ii.copy()
        alpha = a_ji.copy()
        beta = a_ij.copy()

        for __ in range(self.max_iterations):
            inverse = linalg.inv(epsilon)

            epsilon = epsilon - alpha @ inverse @ beta - beta @ inverse @ alpha
            epsilon_s = epsilon_s - alpha @ inverse @ beta

            alpha = alpha @ inverse @ alpha
            beta = beta @ inverse @ beta

            delta = (
                xp.linalg.norm(xp.abs(alpha) + xp.abs(beta), axis=(-2, -1)).max() / 2
            )

            if delta < self.convergence_tol:
                break

        else:  # Did not break, i.e. max_iterations reached.
            warnings.warn("Surface Green's function did not converge.", RuntimeWarning)

        x_ii = linalg.inv(epsilon_s)

        return x_ii
