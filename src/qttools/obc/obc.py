# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import warnings
from abc import ABC, abstractmethod

from qttools import NDArray, global_comm, stack_comm, xp
from qttools.kernels.linalg import inv
from qttools.profiling import Profiler
from qttools.utils.mpi_utils import check_gpu_aware_mpi

profiler = Profiler()

GPU_AWARE_MPI = check_gpu_aware_mpi()
comm = global_comm if stack_comm is None else stack_comm


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
    memoize_rel_tol : float, optional
        The required relative accuracy to only memoize.
    memoize_abs_tol : float, optional
        The required absolute accuracy to only memoize.
        If either of the tolerances is met, the result is memoized.
    warning_threshold : float, optional
        The threshold for the relative recursion error to issue a warning.
    force_memoizing: bool, optionak
        Force memoizing using q as the initial guess.

    """

    def __init__(
        self,
        obc_solver: "OBCSolver",
        num_ref_iterations: int = 3,
        memoize_rel_tol: float = 1e-2,
        memoize_abs_tol: float = 1e-8,
        warning_threshold: float = 1e-4,
        force_memoizing: bool = False,
    ) -> None:
        """Initalizes the memoizer."""
        self.obc_solver = obc_solver
        self.num_ref_iterations = num_ref_iterations
        self.memoize_rel_tol = memoize_rel_tol
        self.memoize_abs_tol = memoize_abs_tol
        self.warning_threshold = warning_threshold
        self.force_memoizing = force_memoizing
        self._cache = {}

        if num_ref_iterations < 2:
            warnings.warn(
                "The number of refinement iterations should be at least 2. Defaulting to 2.",
                RuntimeWarning,
            )
            self.num_ref_iterations = 2

    @profiler.profile(level="debug")
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

    @profiler.profile(level="api")
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
        # TODO: merge with Lyapunov memoizer
        # since there is code duplication

        # Try to reuse the result from the cache.
        x_ii = self._cache.get(contact, None)

        if x_ii is None:
            return self._call_with_cache(a_ii, a_ij, a_ji, contact, out=out)

        x_ii_ref = inv(a_ii - a_ji @ x_ii @ a_ij)

        if not self.force_memoizing:

            # Check for convergence accross all MPI ranks.
            absolute_recursion_errors = xp.linalg.norm(x_ii_ref - x_ii, axis=(-2, -1))
            relative_recursion_errors = absolute_recursion_errors / xp.linalg.norm(
                x_ii_ref, axis=(-2, -1)
            )

            local_memoizing = xp.all(
                (absolute_recursion_errors < self.memoize_abs_tol)
                | (relative_recursion_errors < self.memoize_rel_tol)
            )

            # NOTE: it would be possible to memoize even if few energies did not converge
            if not local_memoizing:
                # If the result did not converge, recompute it from scratch.
                return self._call_with_cache(a_ii, a_ij, a_ji, contact, out=out)

        x_ii = x_ii_ref

        # Do refinement iterations.
        for __ in range(self.num_ref_iterations - 2):
            x_ii = inv(a_ii - a_ji @ x_ii @ a_ij)

        x_ii_ref = inv(a_ii - a_ji @ x_ii @ a_ij)

        absolute_recursion_errors = xp.linalg.norm(x_ii_ref - x_ii, axis=(-2, -1))
        relative_recursion_errors = absolute_recursion_errors / xp.linalg.norm(
            x_ii_ref, axis=(-2, -1)
        )
        x_ii = x_ii_ref

        # TODO: we should allow data gathering of the final recursion error
        if xp.any(
            (relative_recursion_errors > self.warning_threshold)
            & (absolute_recursion_errors > self.memoize_abs_tol)
        ):
            warnings.warn(
                f"High relative recursion error: {xp.max(relative_recursion_errors):.2e} "
                + f"at rank {comm.rank} for {contact} OBC",
                RuntimeWarning,
            )

        self._cache[contact] = x_ii.copy()
        if out is None:
            return x_ii
        out[:] = x_ii
        return None
