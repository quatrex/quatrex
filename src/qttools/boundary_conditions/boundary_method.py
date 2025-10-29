# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.

import warnings
from abc import ABC, abstractmethod

from qttools import NDArray, xp
from qttools.comm import comm
from quatrex.core.quatrex_config import MemoizerConfig


class BoundaryMethod(ABC):

    def __init__(
        self,
        boundary_solver,
        cache_compressor: None = None,
        config: MemoizerConfig = MemoizerConfig(),
    ) -> None:

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

        assert self.mode in [
            "auto",
            "force",
            "force-after-first",
            "off",
        ], f"Invalid memoizing mode: {self.mode}"
        if self.mode == "off":
            warnings.warn(
                "Memoizing mode is set to 'off'. The memoizer will not cache any results.",
                RuntimeWarning,
            )

    @abstractmethod
    def _contract_system(
        self, boundary_system: tuple[NDArray, ...]
    ) -> tuple[NDArray, ...]: ...

    @abstractmethod
    def _expand_solution(
        self,
        boundary_system: tuple[NDArray, ...],
        reduced_system: tuple[NDArray, ...],
        reduced_solution: NDArray | tuple[NDArray, ...],
    ) -> NDArray | tuple[NDArray, ...]: ...

    @abstractmethod
    def _expand_residuals(
        self,
        boundary_system: tuple[NDArray, ...],
        reduced_system: tuple[NDArray, ...],
        rel_residuals: NDArray,
        abs_residuals: NDArray,
    ) -> tuple[NDArray, NDArray]: ...

    @abstractmethod
    def _fix_point_step(
        self, boundary_system: tuple[NDArray, ...], solution: NDArray
    ): ...

    @abstractmethod
    def _get_starting_guess(self, boundary_system: tuple[NDArray, ...]): ...

    @abstractmethod
    def _get_residuals(
        self, boundary_system: tuple[NDArray, ...], test_solution: NDArray
    ) -> tuple[NDArray, NDArray, NDArray | tuple[NDArray, ...]]: ...

    def _solve(
        self,
        boundary_system: tuple[NDArray, ...],
        contact: str,
    ) -> NDArray | tuple[NDArray, ...]:

        solution = self.boundary_solver(boundary_system, contact)

        if self.mode != "off":
            if type(solution) is not xp.ndarray:
                raise NotImplementedError(
                    "Memoizing on multiple solution not supported."
                )
            self._cache[contact] = self.compress(solution)

        return solution

    def _memoized_solve(self, boundary_system: tuple[NDArray, ...], contact: str):
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

    def __call__(self, boundary_system: tuple[NDArray, ...], contact: str) -> NDArray:

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
                f"High relative recursion error: {xp.max(rel_residuals):.3e} "
                + f"at rank {comm.stack.rank} for {contact} of {self.boundary_solver.__class__.__name__}",
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
        rel_residuals, abs_residuals = self._expand_residuals(
            boundary_system, reduced_system, rel_residuals, abs_residuals
        )

        return solution, rel_residuals, abs_residuals
