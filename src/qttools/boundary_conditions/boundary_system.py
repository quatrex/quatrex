# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.

import warnings
from abc import ABC, abstractmethod

from qttools import NDArray, xp
from qttools.comm import comm
from quatrex.core.quatrex_config import MemoizerConfig


class BaseBoundarySystem(ABC):
    """Abstract base class to solve boundary systems with memoization and system reduction.

    Parameters
    ----------
    boundary_solver : callable
        The boundary system solver to be memoized.
    cache_compressor : object, optional
        An object with 'compress' and 'decompress' methods to handle
        cache compression. If None, no compression is applied.
    config : MemoizerConfig, optional
        Configuration for the memoizer.

    """

    def __init__(
        self,
        boundary_solver,
        cache_compressor: None = None,
        config: MemoizerConfig = MemoizerConfig(),
    ) -> None:
        """Initializes the boundary method."""

        self.boundary_solver = boundary_solver
        self.num_ref_iterations = config.num_ref_iterations
        self.relative_tol = config.relative_tol
        self.absolute_tol = config.absolute_tol
        self.warning_threshold = config.warning_threshold
        self.mode = config.mode
        self.agreement_threshold = config.agreement_threshold

        if cache_compressor is None:
            self.compress = lambda x: x.copy()
            self.decompress = lambda x: x
        else:
            assert hasattr(
                cache_compressor, "compress"
            ), "Cache compressor must have a 'compress' method."
            assert hasattr(
                cache_compressor, "decompress"
            ), "Cache compressor must have a 'decompress' method."

            self.compress = cache_compressor.compress
            self.decompress = cache_compressor.decompress

        if not (0.0 <= self.agreement_threshold <= 1.0):
            raise ValueError("Memoizing tolerance must be between 0 and 1.")

        self._cache = {}

        if self.num_ref_iterations < 2:
            warnings.warn(
                "The number of refinement iterations should be at least 2. Defaulting to 2.",
                RuntimeWarning,
            )
            self.num_ref_iterations = 2

    @abstractmethod
    def _contract_system(
        self,
        boundary_system: tuple[NDArray, ...],
    ) -> tuple[NDArray, ...]:
        """Contract the boundary system to a reduced system.

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The full boundary system.

        Returns
        -------
        reduced_system : tuple[NDArray, ...]
            The reduced boundary system.

        """
        ...

    @abstractmethod
    def _expand_solution(
        self,
        boundary_system: tuple[NDArray, ...],
        reduced_system: tuple[NDArray, ...],
        reduced_solution: NDArray | tuple[NDArray, ...],
    ) -> NDArray | tuple[NDArray, ...]:
        """Expand the solution from the reduced system to the full system.

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The full boundary system.
        reduced_system : tuple[NDArray, ...]
            The reduced boundary system.
        reduced_solution : NDArray | tuple[NDArray, ...]
            The solution of the reduced system.

        Returns
        -------
        full_solution : NDArray | tuple[NDArray, ...]
            The solution of the full system.

        """
        ...

    @abstractmethod
    def _expand_residuals(
        self,
        boundary_system: tuple[NDArray, ...],
        reduced_system: tuple[NDArray, ...],
        rel_residuals: NDArray,
        abs_residuals: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Expand the residuals from the reduced system to the full system.

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The full boundary system.
        reduced_system : tuple[NDArray, ...]
            The reduced boundary system.
        rel_residuals : NDArray
            The relative residuals of the reduced system.
        abs_residuals : NDArray
            The absolute residuals of the reduced system.

        Returns
        -------
        full_rel_residuals : NDArray
            The relative residuals of the full system.
        full_abs_residuals : NDArray
            The absolute residuals of the full system.

        """
        ...

    @abstractmethod
    def _fix_point_step(
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
    ) -> tuple[NDArray, NDArray, NDArray | tuple[NDArray, ...]]:
        """Compute the residuals of a test solution.

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
        solution : NDArray | tuple[NDArray, ...]
            The (possibly refined) solution of the boundary system.

        """

        ...

    def _solve(
        self,
        boundary_system: tuple[NDArray, ...],
        contact: str,
    ) -> NDArray | tuple[NDArray, ...]:
        """Solve the boundary system without memoization.

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The boundary system to solve.
        contact : str
            The contact to which the boundary system belongs.

        Returns
        -------
        solution : NDArray | tuple[NDArray, ...]
            The solution of the boundary system.

        """
        solution = self.boundary_solver(*boundary_system, contact)

        if self.mode != "off":
            if type(solution) is not xp.ndarray:
                raise NotImplementedError(
                    "Memoizing on multiple solution not supported."
                )
            self._cache[contact] = self.compress(solution)

        return solution

    def _memoized_solve(
        self,
        boundary_system: tuple[NDArray, ...],
        contact: str,
    ) -> NDArray | tuple[NDArray, ...]:
        """Solve the boundary system with memoization.

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The boundary system to solve.
        contact : str
            The contact to which the boundary system belongs.

        Returns
        -------
        solution : NDArray | tuple[NDArray, ...]
            The solution of the boundary system.

        """

        if self.mode == "off":
            return self._solve(boundary_system, contact)

        # Try to reuse the result from the cache.
        solution = self.decompress(self._cache.get(contact, None))

        if solution is None:
            if self.mode in ["auto", "force-after-first"]:
                return self._solve(boundary_system, contact)

            elif self.mode == "force":
                solution = self._get_starting_guess(boundary_system)

            else:
                raise RuntimeError(f"Invalid memoizing mode: {self.mode}")

        rel_residuals, abs_residuals, solution = self._get_residuals(
            boundary_system, solution
        )

        if self.mode == "auto":
            # Check for convergence accross all MPI ranks.
            local_memoizing = xp.sum(
                (abs_residuals < self.absolute_tol)
                | (rel_residuals < self.relative_tol)
            )
            local_memoizing = xp.array(local_memoizing, dtype=xp.int32)
            global_memoizing = xp.array(0, dtype=xp.int32)

            comm.stack.all_reduce(local_memoizing, global_memoizing)

            global_memoizing = global_memoizing / comm.stack.size

            # allow a few ranks to not converge
            if global_memoizing > self.agreement_threshold:
                # If the result did not converge, recompute it from scratch.
                return self._solve(boundary_system, contact)

        for __ in range(self.num_ref_iterations - 2):
            solution = self._fix_point_step(boundary_system, solution)

        return solution

    def __call__(
        self,
        boundary_system: tuple[NDArray, ...],
        contact: str,
    ) -> tuple[NDArray, NDArray, NDArray | tuple[NDArray, ...]]:
        """Solve the boundary system with memoization and system reduction.

        This is a wrapper around the boundary system solver

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The boundary system to solve.
        contact : str
            The contact to which the boundary system belongs.

        Returns
        -------
        rel_residuals : NDArray
            The relative residuals of the solution.
        abs_residuals : NDArray
            The absolute residuals of the solution.
        solution : NDArray | tuple[NDArray, ...]
            The solution of the boundary system.

        """

        # First, deflate the system
        reduced_system = self._contract_system(boundary_system)

        # memoize and cache on the reduced system
        reduced_solution = self._memoized_solve(reduced_system, contact)

        rel_residuals, abs_residuals, reduced_solution = self._get_residuals(
            reduced_system, reduced_solution
        )

        if xp.any(
            (rel_residuals > self.warning_threshold)
            & (abs_residuals > self.absolute_tol)
        ):
            warnings.warn(
                f"High error at rank {comm.stack.rank} for {contact} of {self.boundary_solver.__class__.__name__}:\n"
                + f"  Relative recursion error: {xp.max(rel_residuals):.3e}\n"
                + f"  Absolute recursion error: {xp.max(abs_residuals):.3e}\n",
                RuntimeWarning,
            )

        if self.mode != "off":
            if type(reduced_solution) is not xp.ndarray:
                raise NotImplementedError(
                    "Memoizing on multiple solution not supported."
                )
            self._cache[contact] = self.compress(reduced_solution)

        # Expand the solution back to the full system
        solution = self._expand_solution(
            boundary_system, reduced_system, reduced_solution
        )
        # Residual need to be expanded as well
        # since the reduced system can be on a different space
        rel_residuals, abs_residuals = self._expand_residuals(
            boundary_system, reduced_system, rel_residuals, abs_residuals
        )

        return rel_residuals, abs_residuals, solution
