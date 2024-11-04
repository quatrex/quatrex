from abc import ABC, abstractmethod

from qttools import xp


class LyapunovSolver(ABC):
    """Solver interface for the discrete-time Lyapunov equation.

    The discrete-time Lyapunov equation is defined as:
    `x - a @ x a.conj().T = q`.

    """

    @abstractmethod
    def __call__(
        self,
        a: xp.ndarray,
        q: xp.ndarray,
        contact: str,
        out: None | xp.ndarray = None,
    ) -> xp.ndarray | None:
        """Computes the solution of the discrete-time Lyapunov equation.

        Parameters
        ----------
        a : array_like
            The system matrix.
        b : array_like
            The right-hand side matrix.
        contact : str
            The contact to which the boundary blocks belong.
        out : array_like, optional
            The array to store the result in. If not provided, a new
            array is returned.

        Returns
        -------
        x : array_like, optional
            The solution of the discrete-time Lyapunov equation.

        """
        ...


class LyapunovMemoizer:
    """Memoization class to reuse the result of an Lyapunov evaluation.

    Parameters
    ----------
    lyapunov : Lyapunov
        The Lyapunov solver to wrap.
    num_ref_iterations : int, optional
        The number of refinement iterations to do.
    convergence_tol : float, optional
        The required accuracy for convergence.

    """

    def __init__(
        self,
        lyapunov: LyapunovSolver,
        num_ref_iterations: int = 10,
        convergence_tol: float = 1e-6,
    ) -> None:
        """Initializes the memoizer."""
        self.lyapunov = lyapunov
        self.num_ref_iterations = num_ref_iterations
        self.convergence_tol = convergence_tol
        self._cache = {}

    def _call_with_cache(
        self,
        a: xp.ndarray,
        q: xp.ndarray,
        contact: str,
        out: None | xp.ndarray = None,
    ) -> xp.ndarray | None:
        """Calls the wrapped Lyapunov function with cache handling."""
        x = self.lyapunov(a, q, contact, out=out)
        if out is None:
            self._cache[contact] = x.copy()
            return x

        self._cache[contact] = out.copy()
        return None

    def __call__(
        self,
        a: xp.ndarray,
        q: xp.ndarray,
        contact: str,
        out: None | xp.ndarray = None,
    ) -> xp.ndarray | None:
        """Computes the solution of the discrete-time Lyapunov equation."""
        # Try to reuse the result from the cache.
        x = self._cache.get(contact, None)

        if x is None:
            return self._call_with_cache(a, q, contact, out=out)

        # Do refinement iterations.
        for __ in range(self.num_ref_iterations - 1):
            x = q + a @ x @ a.conj().swapaxes(-2, -1)

        x_ref = q + a @ x @ a.conj().swapaxes(-2, -1)

        # Check for convergence.
        rel_diff = xp.linalg.norm(x_ref - x, axis=(-2, -1)) / xp.linalg.norm(
            x, axis=(-2, -1)
        )
        mean_rel_diff = xp.mean(rel_diff)
        if mean_rel_diff < self.convergence_tol:
            self._cache[contact] = x_ref.copy()
            if out is None:
                return x_ref
            out[:] = x_ref
            return None

        # If the result did not converge, recompute it from scratch.
        return self._call_with_cache(a, q, contact, out=out)
