# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from abc import ABC, abstractmethod

import numpy as np

from qttools.utils.gpu_utils import xp


class OBC(ABC):
    """Abstract base class for the open-boundary condition solver."""

    @abstractmethod
    def __call__(
        self,
        a_ii: xp.ndarray,
        a_ij: xp.ndarray,
        a_ji: xp.ndarray,
        contact: str,
        out: None | xp.ndarray = None,
    ) -> np.ndarray | None:
        """Returns the surface Green's function.

        Parameters
        ----------
        a_ii : array_like
            Diagonal boundary block of a system matrix.
        a_ij : array_like
            Off-diagonal boundary block of a system matrix, connecting
            lead (i) to system (j).
        a_ji : array_like
            Off-diagonal boundary block of a system matrix, connecting
            system (j) to lead (i).
        contact : str
            The contact to which the boundary blocks belong.
        out : array_like, optional
            The array to store the result in. If not provided, a new
            array is returned.

        Returns
        -------
        x_ii : array_like, optional
            The system's surface Green's function.

        """
        ...


class Memoizer:
    """Memoization class to reuse the result of an OBC function call.

    Parameters
    ----------
    obc : OBC
        The OBC solver to wrap.
    num_ref_iterations : int, optional
        The maximum number of refinement iterations to try before giving
        up.
    convergence_tol : float, optional
        The required accuracy for convergence.

    """

    def __init__(
        self,
        obc: "OBC",
        num_ref_iterations: int = 2,
        convergence_tol: float = 1e-6,
    ) -> None:
        """Initalizes the memoizer."""
        self.obc = obc
        self.num_ref_iterations = num_ref_iterations
        self.convergence_tol = convergence_tol
        self._cache = {}

    def _call_with_cache(
        self,
        a_ii: np.ndarray,
        a_ij: np.ndarray,
        a_ji: np.ndarray,
        contact: str,
        out: None | np.ndarray = None,
    ) -> np.ndarray | None:
        """Calls the wrapped obc function with cache handling."""
        x_ii = self.obc(a_ii, a_ij, a_ji, contact, out=out)
        if out is None:
            self._cache[contact] = x_ii.copy()
            return x_ii

        self._cache[contact] = out.copy()
        return None

    def __call__(
        self,
        a_ii: np.ndarray,
        a_ij: np.ndarray,
        a_ji: np.ndarray,
        contact: str,
        out: None | np.ndarray = None,
    ) -> np.ndarray | None:
        """Calls the wrapped function."""
        # Try to reuse the result from the cache.
        x_ii = self._cache.get(contact, None)

        if x_ii is None:
            return self._call_with_cache(a_ii, a_ij, a_ji, contact, out=out)

        # Do refinement iterations.
        for __ in range(self.num_ref_iterations - 1):
            x_ii = xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij)

        x_ii_ref = xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij)

        # Check for convergence.
        rel_diff = xp.linalg.norm(x_ii_ref - x_ii, axis=(-2, -1)) / xp.linalg.norm(
            x_ii, axis=(-2, -1)
        )
        mean_rel_diff = xp.mean(rel_diff)
        if mean_rel_diff < self.convergence_tol:
            self._cache[contact] = x_ii_ref.copy()
            if out is None:
                return x_ii_ref
            out[:] = x_ii_ref
            return None

        # If the result did not converge, recompute it from scratch.
        return self._call_with_cache(a_ii, a_ij, a_ji, contact, out=out)
