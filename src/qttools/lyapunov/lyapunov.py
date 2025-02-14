# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from abc import ABC, abstractmethod

from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, xp


class LyapunovSolver(ABC):
    r"""Solver interface for the discrete-time Lyapunov equation.

    The discrete-time Lyapunov equation is defined as:

    \[
        X - A X A^H = Q
    \]

    """

    @abstractmethod
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
        ...


class LyapunovMemoizer:
    """Memoization class to reuse the result of an Lyapunov evaluation.

    Parameters
    ----------
    lyapunov_solver : LyapunovSolver
        The Lyapunov solver to wrap.
    num_ref_iterations : int, optional
        The number of refinement iterations to do.
    convergence_tol : float, optional
        The required accuracy for convergence.

    """

    def __init__(
        self,
        lyapunov_solver: LyapunovSolver,
        num_ref_iterations: int = 10,
        convergence_tol: float = 1e-4,
    ) -> None:
        """Initializes the memoizer."""
        self.lyapunov_solver = lyapunov_solver
        self.num_ref_iterations = num_ref_iterations
        self.convergence_tol = convergence_tol
        self._cache = {}

    def _call_with_cache(
        self,
        a: NDArray,
        q: NDArray,
        contact: str,
        out: None | NDArray = None,
    ) -> NDArray | None:
        """Calls the wrapped Lyapunov function with cache handling.

        Parameters
        ----------
        a : NDArray
            The system matrix.
        q : NDArray
            The right-hand side matrix.
        contact : str
            The contact to which the boundary blocks belong. Used as a
            key for the cache.
        out : NDArray, optional
            The array to store the result in. If not provided, a new
            array is returned.

        Returns
        -------
        x : NDArray | None
            The solution of the discrete-time Lyapunov equation.

        """
        x = self.lyapunov_solver(a, q, contact, out=out)
        if out is None:
            self._cache[contact] = x.copy()
            return x

        self._cache[contact] = out.copy()
        return None

    def __call__(
        self,
        a: NDArray,
        q: NDArray,
        contact: str,
        out: None | NDArray = None,
    ) -> NDArray | None:
        """Computes the solution of the discrete-time Lyapunov equation.

        This is a memoized wrapper around a Lyapunov solver.

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
        # Try to reuse the result from the cache.
        x = self._cache.get(contact, None)

        if x is None:
            return self._call_with_cache(a, q, contact, out=out)

        # Do refinement iterations.
        for __ in range(self.num_ref_iterations - 1):
            x = q + a @ x @ a.conj().swapaxes(-2, -1)

        x_ref = q + a @ x @ a.conj().swapaxes(-2, -1)

        # Check for convergence accross all MPI ranks.
        recursion_error = xp.mean(
            xp.linalg.norm(x_ref - x, axis=(-2, -1))
            / xp.linalg.norm(x_ref, axis=(-2, -1))
        )
        converged = recursion_error < self.convergence_tol
        converged = comm.allreduce(converged, op=MPI.LAND)

        if converged:
            self._cache[contact] = x_ref.copy()
            if out is None:
                return x_ref
            out[:] = x_ref
            return None

        # If the result did not converge, recompute it from scratch.
        return self._call_with_cache(a, q, contact, out=out)
