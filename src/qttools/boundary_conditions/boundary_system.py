# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import warnings
from abc import ABC, abstractmethod

from qttools import NDArray, xp
from qttools.boundary_conditions.system_reduction import IdentityReducer, SystemReducer
from qttools.comm import comm


class IdentityCompressor:
    """Identity cache compressor that does not perform any compression."""

    def compress(self, x: NDArray):
        """Compress the data to be cached.
        In this case, it simply returns the input as is.

        Parameters
        ----------
        x : NDArray
            The data to be compressed.
        Returns
        -------
        compressed_x : NDArray
            The compressed data, which is the same as the input in this case, but
            potentially copied to ensure immutability.

        """
        return x.copy() if hasattr(x, "copy") else x

    def decompress(self, x: NDArray):
        """Decompress the cached data.
        In this case, it simply returns the input as is.

        Parameters
        ----------
        x : NDArray
            The data to be decompressed.
        Returns
        -------
        decompressed_x : NDArray
            The decompressed data, which is the same as the input in this case.

        """
        return x


class BaseBoundarySystem(ABC):
    """Abstract base class to solve boundary systems with memoization and system reduction.

    Parameters
    ----------
    boundary_solver : callable
        The boundary system solver to be memoized.
    cache_compressor : object, optional
        An object with 'compress' and 'decompress' methods to handle
        cache compression. If None, no compression is applied.
    num_ref_iterations : int, optional
        The number of fixed-point iterations to refine the solution. Default is 2.
    relative_tol : float, optional
        The relative tolerance for convergence. Default is 0.2.
    absolute_tol : float, optional
        The absolute tolerance for convergence. Default is 1e-6.
    warning_threshold : float, optional
        The threshold for issuing a warning about high residuals. Default is 0.1.
    memoization_mode : str, optional
        The memoization mode. Can be 'off', 'auto', 'force-after-first', or 'force'.
        Default is 'auto'.
    agreement_threshold : float, optional
        The threshold for agreement across MPI ranks to consider a memoized solution valid.
        Default is 0.999.

    """

    def __init__(
        self,
        boundary_solver,
        cache_compressor: None = None,
        system_reducer: SystemReducer | None = None,
        num_ref_iterations: int = 2,
        relative_tol: float = 2e-1,
        absolute_tol: float = 1e-6,
        warning_threshold: float = 1e-1,
        memoization_mode: str = "auto",
        agreement_threshold: float = 0.999,
    ) -> None:
        """Initializes the boundary method."""

        self.boundary_solver = boundary_solver
        self.num_ref_iterations = num_ref_iterations
        self.relative_tol = relative_tol
        self.absolute_tol = absolute_tol
        self.warning_threshold = warning_threshold
        self.memoization_mode = memoization_mode
        self.agreement_threshold = agreement_threshold

        self.cache_compressor = cache_compressor or IdentityCompressor()

        if not hasattr(self.cache_compressor, "compress"):
            raise ValueError("Cache compressor must have a 'compress' method.")
        if not hasattr(self.cache_compressor, "decompress"):
            raise ValueError("Cache compressor must have a 'decompress' method.")

        if not (0.0 <= self.agreement_threshold <= 1.0):
            raise ValueError("Memoizing tolerance must be between 0 and 1.")

        self.system_reducer = system_reducer or IdentityReducer()

        if not hasattr(self.system_reducer, "contract_system"):
            raise ValueError("System reducer must have a 'contract_system' method.")
        if not hasattr(self.system_reducer, "expand_solution"):
            raise ValueError("System reducer must have an 'expand_solution' method.")
        if not hasattr(self.system_reducer, "expand_residuals"):
            raise ValueError("System reducer must have an 'expand_residuals' method.")

        self._cache = {}

        if self.num_ref_iterations < 2:
            warnings.warn(
                "The number of refinement iterations should be at least 2. Defaulting to 2.",
                RuntimeWarning,
            )
            self.num_ref_iterations = 2

    @abstractmethod
    def _fixed_point_step(
        self,
        boundary_system: tuple[NDArray, ...],
        solution: NDArray,
    ) -> NDArray:
        """Perform a fixed-point iteration step to refine the solution.

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The boundary system to solve.
        solution : NDArray
            The current solution to refine.

        Returns
        -------
        refined_solution : NDArray
            The refined solution after one fixed-point iteration step.

        """
        ...

    @abstractmethod
    def _get_starting_guess(
        self,
        boundary_system: tuple[NDArray, ...],
    ) -> NDArray:
        """Get a starting guess for the boundary system.

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The boundary system to solve.

        Returns
        -------
        starting_guess : NDArray
            The starting guess for the boundary system.

        """
        ...

    @abstractmethod
    def _get_residuals(
        self,
        boundary_system: tuple[NDArray, ...],
        test_solution: NDArray,
    ) -> tuple[NDArray, ...]:
        """Compute the residuals of a test solution.

        Computing the residuals of a test solution involves performing
        an additional fixed-point iteration step to refine the solution,
        and then comparing the refined solution to the test solution.

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The boundary system to solve.
        test_solution : NDArray
            The test solution to evaluate.

        Returns
        -------
        rel_residuals : NDArray
            The relative residuals of the test solution.
        abs_residuals : NDArray
            The absolute residuals of the test solution.
        solution : tuple[NDArray, ...]
            The (possibly refined) solution of the boundary system.

        """

        ...

    def _solve(
        self,
        boundary_system: tuple[NDArray, ...],
        contact: str,
        **kwargs,
    ) -> NDArray | tuple[NDArray, ...]:
        """Solve the boundary system without memoization.

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The boundary system to solve.
        contact : str
            The contact to which the boundary system belongs.
        **kwargs
            Additional arguments to pass to the boundary system solver.
            Mostly for the injection vectors in case of OBCs / QTBM.

        Returns
        -------
        solution : NDArray | tuple[NDArray, ...]
            The solution of the boundary system.

        """
        solution = self.boundary_solver(*boundary_system, contact, **kwargs)

        if self.memoization_mode != "off":
            if type(solution) is not xp.ndarray:
                raise NotImplementedError(
                    "Memoizing on multiple solution not supported."
                )
            self._cache[contact] = self.cache_compressor.compress(solution)

        return solution

    def _memoized_solve(
        self,
        boundary_system: tuple[NDArray, ...],
        contact: str,
        **kwargs,
    ) -> NDArray | tuple[NDArray, ...]:
        """Solve the boundary system with memoization.

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The boundary system to solve.
        contact : str
            The contact to which the boundary system belongs.
        **kwargs
            Additional arguments to pass to the boundary system solver.
            Mostly for the injection vectors in case of OBCs / QTBM.

        Returns
        -------
        solution : NDArray | tuple[NDArray, ...]
            The solution of the boundary system.

        """

        if self.memoization_mode == "off":
            return self._solve(boundary_system, contact, **kwargs)

        # Try to reuse the result from the cache.
        solution = self.cache_compressor.decompress(self._cache.get(contact, None))

        if solution is None:
            if self.memoization_mode in ["auto", "force-after-first"]:
                return self._solve(boundary_system, contact, **kwargs)

            elif self.memoization_mode == "force":
                solution = self._get_starting_guess(boundary_system)

            else:
                raise RuntimeError(f"Invalid memoizing mode: {self.memoization_mode}")

        rel_residuals, abs_residuals, solution = self._get_residuals(
            boundary_system, solution
        )

        if self.memoization_mode == "auto":
            # Check for convergence accross all MPI ranks.
            local_memoizing = xp.sum(
                (abs_residuals < self.absolute_tol)
                | (rel_residuals < self.relative_tol)
            )
            local_memoizing = xp.array(local_memoizing, dtype=xp.int32)
            global_memoizing = xp.array([0], dtype=xp.int32)

            comm.stack.all_reduce(local_memoizing, global_memoizing)

            global_memoizing = global_memoizing / comm.stack.size

            # allow a few ranks to not converge
            if global_memoizing > self.agreement_threshold:
                # If the result did not converge, recompute it from scratch.
                return self._solve(boundary_system, contact, **kwargs)

        for __ in range(self.num_ref_iterations - 2):
            solution = self._fixed_point_step(boundary_system, solution)

        return solution

    def __call__(
        self,
        boundary_system: tuple[NDArray, ...],
        contact: str,
        **kwargs,
    ) -> tuple[NDArray, NDArray, NDArray | tuple[NDArray, ...]]:
        """Solve the boundary system with memoization and system reduction.

        This is a wrapper around the boundary system solver

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The boundary system to solve.
        contact : str
            The contact to which the boundary system belongs.
        **kwargs
            Additional arguments to pass to the boundary system solver.
            Mostly for the injection vectors in case of OBCs / QTBM.

        Returns
        -------
        solution : NDArray | tuple[NDArray, ...]
            The solution of the boundary system.
        rel_residuals : NDArray
            The relative residuals of the solution.
        abs_residuals : NDArray
            The absolute residuals of the solution.

        """

        # First, deflate the system
        reduced_system = self.system_reducer.contract_system(boundary_system)

        # memoize and cache on the reduced system
        reduced_solution = self._memoized_solve(reduced_system, contact, **kwargs)

        rel_residuals, abs_residuals, reduced_solution = self._get_residuals(
            reduced_system, reduced_solution
        )

        if xp.any(
            (rel_residuals > self.warning_threshold)
            & (abs_residuals > self.absolute_tol)
        ):
            warnings.warn(
                f"High error at rank {comm.stack.rank} for {contact} of "
                + f"{self.boundary_solver.__class__.__bases__[0].__name__} {self.boundary_solver.__class__.__name__}:\n"
                + f"  Relative recursion error: {xp.max(rel_residuals):.3e}\n"
                + f"  Absolute recursion error: {xp.max(abs_residuals):.3e}\n",
                RuntimeWarning,
            )

        if self.memoization_mode != "off":
            if type(reduced_solution) is not xp.ndarray:
                raise NotImplementedError(
                    "Memoizing on multiple solution not supported."
                )
            self._cache[contact] = self.cache_compressor.compress(reduced_solution)

        # Expand the solution back to the full system
        solution = self.system_reducer.expand_solution(
            boundary_system, reduced_system, reduced_solution
        )
        # Residual need to be expanded as well
        # since the reduced system can be on a different space
        rel_residuals, abs_residuals = self.system_reducer.expand_residuals(
            boundary_system, reduced_system, rel_residuals, abs_residuals
        )

        return solution, rel_residuals, abs_residuals
