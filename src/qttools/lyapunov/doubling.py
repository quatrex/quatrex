# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import warnings

from qttools import NDArray, xp
from qttools.lyapunov.lyapunov import LyapunovSolver


class Doubling(LyapunovSolver):
    """A solver for the Lyapunov equation using iterative doubling.

    Parameters
    ----------
    max_iterations : int, optional
        The maximum number of iterations to perform.
    convergence_tol : float, optional
        The required accuracy for convergence.

    """

    def __init__(
        self, max_iterations: int = 100, convergence_tol: float = 1e-6
    ) -> None:
        """Initializes the solver."""
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol

    def __call__(
        self,
        a: NDArray,
        q: NDArray,
        contact: str,
        out: None | NDArray = None,
    ) -> NDArray | None:
        """Computes the solution of the discrete-time Lyapunov equation.

        Parameters
        ----------
        a : NDArray
            The system matrix.
        q : NDArray
            The right-hand side matrix.
        contact : str
            The contact to which the boundary blocks belong.
        out : NDArray, optional
            The array to store the result in. If not provided, a new
            array is returned.

        Returns
        -------
        x : NDArray | None
            The solution of the discrete-time Lyapunov equation.

        """

        if a.ndim == 2:
            a = a[xp.newaxis, :, :]
            q = q[xp.newaxis, :, :]

        a_i = a.copy()
        x = q.copy()

        for __ in range(self.max_iterations):
            x_i = x + a_i @ x @ a_i.conj().swapaxes(-1, -2)

            if xp.linalg.norm(x_i - x, axis=(-2, -1)).max() < self.convergence_tol:
                break

            a_i = a_i @ a_i
            x = x_i

        else:  # Did not break, i.e. max_iterations reached.
            warnings.warn("Lyapunov equation did not converge.", RuntimeWarning)

        if out is not None:
            out[...] = x
            return

        return x
