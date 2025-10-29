# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, xp
from qttools.boundary_conditions.boundary_system import BaseBoundarySystem
from qttools.lyapunov.lyapunov import LyapunovSolver
from quatrex.core.quatrex_config import MemoizerConfig


class LyapunovMethod(BaseBoundarySystem):
    """A lyapunov system solver with memoization and system reduction.

        The lyapyunov equation to be solved is given by:

        $$\mathbf{x} = \mathbf{q} + \mathbf{a} \mathbf{x}  \mathbf{a}^H$$

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
        boundary_solver: LyapunovSolver,
        cache_compressor: None = None,
        config: MemoizerConfig = MemoizerConfig(),
    ) -> None:
        """Initializes the lyapunov system."""

        super().__init__(
            boundary_solver,
            cache_compressor,
            config,
        )

    def _contract_system(
        self,
        boundary_system: tuple[NDArray, NDArray],
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

    def _expand_solution(
        self,
        boundary_system: tuple[NDArray, NDArray],
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

    def _expand_residuals(
        self,
        boundary_system: tuple[NDArray, NDArray],
        reduced_system: tuple[NDArray, NDArray],
        rel_residuals: NDArray,
        abs_residuals: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Expand the residuals from the reduced system to the full system.

        The reduced lyapynov system has the same residuals as the full system.

        Parameters
        ----------
        boundary_system : tuple[NDArray, NDArray]
            The full boundary system to solve.
            It is expected to be a tuple (a, q) where 'a' is the system matrix
            and 'q' is the right-hand side matrix.
        reduced_system : tuple[NDArray, ...]
            The reduced boundary system to solve.
            It is expected to be a tuple (a, q) where 'a' is the reduced system matrix
            and 'q' is the reduced right-hand side matrix.
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
        return rel_residuals, abs_residuals

    def _fix_point_step(
        self,
        boundary_system: tuple[NDArray, NDArray],
        solution: NDArray,
    ) -> NDArray:
        """Perform a fixed-point iteration step to refine the solution.

            The fix-point iteration is given by:
            $$\mathbf{x}_{n+1} = \mathbf{q} + \mathbf{a} \mathbf{x}_{n}  \mathbf{a}^H$$

        Parameters
        ----------
        boundary_system : tuple[NDArray, NDArray]
            The boundary system to solve.
            It is expected to be a tuple (a, q) where 'a' is the system matrix
            and 'q' is the right-hand side matrix.
        solution : NDArray
            The current solution to refine.

        Returns
        -------
        refined_solution : NDArray
            The refined solution after one fixed-point iteration step.

        """
        a, q = boundary_system
        return q + a @ solution @ a.conj().swapaxes(-2, -1)

    def _get_starting_guess(
        self,
        boundary_system: tuple[NDArray, NDArray],
    ) -> NDArray:
        """Get a starting guess for the boundary system.

        For the lyapunov equation, a good starting guess is simply the right-hand side 'q'.

        Parameters
        ----------
        boundary_system : tuple[NDArray, NDArray]
            The boundary system to solve.
            It is expected to be a tuple (a, q) where 'a' is the system matrix
            and 'q' is the right-hand side matrix.

        Returns
        -------
        starting_guess : NDArray
            The starting guess for the boundary system.

        """
        _, q = boundary_system
        return q

    def _get_residuals(
        self,
        boundary_system: tuple[NDArray, NDArray],
        test_solution: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Compute the residuals of a test solution.

            They are computed as follows:
            $$\mathbf{x}_{ref} = \mathbf{q} + \mathbf{a} \mathbf{x}_{test}  \mathbf{a}^H$$
            $$\mathbf{residual_{abs}} = \lvert \mathbf{x}_{ref} - \mathbf{x}_{test} \rvert$$
            $$\mathbf{residual_{rel}} = \frac{\mathbf{residual_{abs}}}{\lvert \mathbf{x}_{ref} \rvert}$$

        Parameters
        ----------
        boundary_system : tuple[NDArray, NDArray]
            The boundary system to solve.
            It is expected to be a tuple (a, q) where 'a' is the system matrix
            and 'q' is the right-hand side matrix.
        test_solution : NDArray
            The test solution to evaluate.

        Returns
        -------
        rel_residuals : NDArray
            The relative residuals of the test solution.
        abs_residuals : NDArray
            The absolute residuals of the test solution.
        solution : NDArray
            The (possibly refined) solution of the boundary system.

        """
        solution_ref = self._fix_point_step(boundary_system, test_solution)

        abs_residuals = xp.linalg.norm(solution_ref - test_solution, axis=(-2, -1))
        rel_residuals = abs_residuals / xp.linalg.norm(solution_ref, axis=(-2, -1))

        return rel_residuals, abs_residuals, solution_ref
