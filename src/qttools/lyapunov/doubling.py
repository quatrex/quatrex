# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import warnings

from qttools import NDArray, xp
from qttools.lyapunov.lyapunov import LyapunovSolver
from qttools.lyapunov.utils import system_reduction
from qttools.profiling import Profiler

profiler = Profiler()


class Doubling(LyapunovSolver):
    """A solver for the Lyapunov equation using iterative doubling.

    Parameters
    ----------
    max_iterations : int, optional
        The maximum number of iterations to perform.
    convergence_rel_tol : float, optional
        The required relative accuracy for convergence.
    convergence_abs_tol : float, optional
        The required absolute accuracy for convergence.
        Either convergence_rel_tol or convergence_abs_tol must be satisfied.
    reduce_sparsity : bool, optional
        Whether to reduce the sparsity of the system matrix

    """

    def __init__(
        self,
        max_iterations: int = 10,
        convergence_rel_tol: float = 1e-5,
        convergence_abs_tol: float = 1e-8,
        reduce_sparsity: bool = True,
    ) -> None:
        """Initializes the solver."""
        self.max_iterations = max_iterations
        self.convergence_rel_tol = convergence_rel_tol
        self.convergence_abs_tol = convergence_abs_tol
        self.reduce_sparsity = reduce_sparsity

    @profiler.profile(level="debug")
    def _solve(
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

        a = xp.broadcast_to(a, q.shape)
        a_i = a.copy()
        x = q.copy()

        for __ in range(self.max_iterations):
            x_i = x + a_i @ x @ a_i.conj().swapaxes(-1, -2)

            absolute_recursion_errors = xp.linalg.norm(x_i - x, axis=(-2, -1))
            relative_recursion_errors = absolute_recursion_errors / xp.linalg.norm(
                x_i, axis=(-2, -1)
            )
            x = x_i

            if xp.all(
                (relative_recursion_errors < self.convergence_rel_tol)
                | (absolute_recursion_errors < self.convergence_abs_tol)
            ):
                break

            a_i = a_i @ a_i

        else:  # Did not break, i.e. max_iterations reached.
            warnings.warn("Lyapunov equation did not converge.", RuntimeWarning)

        if out is not None:
            out[...] = x
            return

        return x

    @profiler.profile(level="api")
    def __call__(
        self,
        a: NDArray,
        q: NDArray,
        contact: str,
        out: None | NDArray = None,
    ) -> NDArray | None:
        """Computes the solution of the discrete-time Lyapunov equation.

        The matrices a and q can have different ndims with q.ndim >= a.ndim (will broadcast)

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

        assert q.shape[-2:] == a.shape[-2:]
        assert q.ndim >= a.ndim

        # NOTE: possible to cache the sparsity reduction
        if self.reduce_sparsity:
            return system_reduction(a, q, contact, self._solve, out=out)

        return self._solve(a, q, contact, out=out)
