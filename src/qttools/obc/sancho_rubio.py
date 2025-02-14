# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import warnings

from qttools import NDArray, xp
from qttools.obc.obc import OBCSolver


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
        a_ii: NDArray,
        a_ij: NDArray,
        a_ji: NDArray,
        contact: str,
        out: None | NDArray = None,
    ) -> NDArray | None:
        """Returns the surface Green's function.

        Parameters
        ----------
        a_ii : NDArray
            Diagonal boundary block of a system matrix.
        a_ij : NDArray
            Superdiagonal boundary block of a system matrix.
        a_ji : NDArray
            Subdiagonal boundary block of a system matrix.
        contact : str
            The contact to which the boundary blocks belong.
        out : NDArray, optional
            The array to store the result in. If not provided, a new
            array is returned.

        Returns
        -------
        x_ii : NDArray
            The system's surface Green's function.

        """
        epsilon = a_ii.copy()
        epsilon_s = a_ii.copy()
        alpha = a_ji.copy()
        beta = a_ij.copy()

        delta = float("inf")
        for __ in range(self.max_iterations):
            inverse = xp.linalg.inv(epsilon)

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

        x_ii = xp.linalg.inv(epsilon_s)

        if out is not None:
            out[...] = x_ii
            return

        return x_ii
