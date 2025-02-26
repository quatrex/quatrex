# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from abc import ABC, abstractmethod

from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, xp


class OBCSolver(ABC):
    r"""Abstract base class for the open-boundary condition solver.

    The recursion relation for the surface Green's function is given by:

    \[
        x_{ii} = (a_{ii} - a_{ji} x_{ii} a_{ij})^{-1}
    \]

    """

    @abstractmethod
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
        ...


class OBCMemoizer:
    """Memoization class to reuse the result of an OBC function call.

    Parameters
    ----------
    obc_solver : OBCSolver
        The OBC solver to wrap.
    num_ref_iterations : int, optional
        The maximum number of refinement iterations to do.
    convergence_tol : float, optional
        The required accuracy for convergence.

    """

    def __init__(
        self,
        obc_solver: "OBCSolver",
        num_ref_iterations: int = 3,
        convergence_tol: float = 1e-4,
    ) -> None:
        """Initalizes the memoizer."""
        self.obc_solver = obc_solver
        self.num_ref_iterations = num_ref_iterations
        self.convergence_tol = convergence_tol
        self._cache = {}

    def _call_with_cache(
        self,
        a_ii: NDArray,
        a_ij: NDArray,
        a_ji: NDArray,
        contact: str,
        out: None | NDArray = None,
    ) -> NDArray | None:
        """Calls the wrapped OBC solver with cache handling.

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
        x_ii = self.obc_solver(a_ii, a_ij, a_ji, contact, out=out)
        if out is None:
            self._cache[contact] = x_ii.copy()
            return x_ii

        self._cache[contact] = out.copy()
        return None

    def __call__(
        self,
        a_ii: NDArray,
        a_ij: NDArray,
        a_ji: NDArray,
        contact: str,
        out: None | NDArray = None,
    ) -> NDArray | None:
        """Returns the surface Green's function.

        This is a memoized wrapper around an OBC solver.

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
        # Try to reuse the result from the cache.
        x_ii = self._cache.get(contact, None)

        if x_ii is None:
            return self._call_with_cache(a_ii, a_ij, a_ji, contact, out=out)

        # Do refinement iterations.
        for __ in range(self.num_ref_iterations - 1):
            x_ii = xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij)

        x_ii_ref = xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij)

        # Check for convergence accross all MPI ranks.
        recursion_error = xp.max(
            xp.linalg.norm(x_ii_ref - x_ii, axis=(-2, -1))
            / xp.linalg.norm(x_ii_ref, axis=(-2, -1))
        )
        converged = recursion_error < self.convergence_tol
        converged = comm.allreduce(converged, op=MPI.LAND)

        if converged:
            self._cache[contact] = x_ii_ref.copy()
            if out is None:
                return x_ii_ref
            out[:] = x_ii_ref
            return None

        # If the result did not converge, recompute it from scratch.
        return self._call_with_cache(a_ii, a_ij, a_ji, contact, out=out)
