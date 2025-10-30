# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from abc import ABC, abstractmethod

from qttools import NDArray, xp
from qttools.boundary_conditions.boundary_system import BaseBoundarySystem
from qttools.kernels import linalg
from quatrex.core.quatrex_config import MemoizerConfig


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
    ) -> NDArray:
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

        Returns
        -------
        x_ii : NDArray
            The system's surface Green's function.

        """
        ...


class OBCSystem(BaseBoundarySystem):
    """An obc system solver with memoization and system reduction.

    Parameters
    ----------
    boundary_solver : OBCSolver
        The obc solver to be memoized.
    cache_compressor : object, optional
        An object with 'compress' and 'decompress' methods to handle
        cache compression. If None, no compression is applied.
    config : MemoizerConfig, optional
        Configuration for the memoizer.

    """

    def __init__(
        self,
        boundary_solver: OBCSolver,
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
        boundary_system: tuple[NDArray, NDArray, NDArray],
    ) -> tuple[NDArray, ...]:
        """Contract the boundary system to a reduced system.

        Parameters
        ----------
        boundary_system : tuple[NDArray, NDArray, NDArray]
            The full boundary system.
            It is expected to be a tuple (a_ii, a_ij, a_ji)

        Returns
        -------
        reduced_system : tuple[NDArray, NDArray, NDArray]
            The reduced boundary system.
            It is expected to be a tuple (a_ii, a_ij, a_ji)

        """
        # TODO: implement system reduction
        # by using the periodicity in non transport directions
        return boundary_system

    def _expand_solution(
        self,
        boundary_system: tuple[NDArray, NDArray, NDArray],
        reduced_system: tuple[NDArray, ...],
        reduced_solution: NDArray | tuple[NDArray, ...],
    ) -> NDArray | tuple[NDArray, ...]:
        """Expand the solution from the reduced system to the full system.

        Parameters
        ----------
        boundary_system : tuple[NDArray, NDArray, NDArray]
            The full boundary system.
            It is expected to be a tuple (a_ii, a_ij, a_ji)
        reduced_system : tuple[NDArray, NDArray, NDArray]
            The reduced boundary system.
            It is expected to be a tuple (a_ii, a_ij, a_ji)
        reduced_solution : NDArray | tuple[NDArray, ...]
            The solution of the reduced system.

        Returns
        -------
        full_solution : NDArray | tuple[NDArray, ...]
            The solution of the full system.

        """
        # TODO: implement system reduction
        # by using the periodicity in non transport directions
        return reduced_solution

    def _expand_residuals(
        self,
        boundary_system: tuple[NDArray, NDArray, NDArray],
        reduced_system: tuple[NDArray, ...],
        rel_residuals: NDArray,
        abs_residuals: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Expand the residuals from the reduced system to the full system.

        TODO: If system reduction is implemented,
        the residuals would need to be combined of the different subsystems.

        Parameters
        ----------
        boundary_system : tuple[NDArray, NDArray, NDArray]
            The full boundary system.
            It is expected to be a tuple (a_ii, a_ij, a_ji)
        reduced_system : tuple[NDArray, NDArray, NDArray]
            The reduced boundary system.
            It is expected to be a tuple (a_ii, a_ij, a_ji)
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
        # TODO: implement system reduction
        # by using the periodicity in non transport directions
        return rel_residuals, abs_residuals

    def _fix_point_step(
        self,
        boundary_system: tuple[NDArray, NDArray, NDArray],
        solution: NDArray,
    ) -> NDArray:
        r"""Perform a fixed-point iteration step to refine the solution.

        The fix-point iteration is given by:
        $$\mathbf{x}_{n+1} = [\mathbf{a}_{ii} - \mathbf{a}_{ji} \mathbf{x}_{n} \mathbf{a}_{ij}]^{-1}$$

        Parameters
        ----------
        boundary_system : tuple[NDArray, NDArray, NDArray]
            The boundary system to solve.
            It is expected to be a tuple (a_ii, a_ij, a_ji)
        solution : NDArray
            The current solution to refine.

        Returns
        -------
        refined_solution : NDArray
            The refined solution after one fixed-point iteration step.

        """
        a_ii, a_ij, a_ji = boundary_system
        return linalg.inv(a_ii - a_ji @ solution @ a_ij)

    def _get_starting_guess(
        self,
        boundary_system: tuple[NDArray, NDArray, NDArray],
    ) -> NDArray:
        r"""Get a starting guess for the obc system.

        For the obc, a good starting guess is the inverse of the
        diagonal block $\mathbf{a}_{ii}$.

        Parameters
        ----------
        boundary_system : tuple[NDArray, NDArray, NDArray]
            The boundary system to solve.
            It is expected to be a tuple (a_ii, a_ij, a_ji)

        Returns
        -------
        starting_guess : NDArray
            The starting guess for the boundary system.

        """
        a_ii, _, _ = boundary_system
        return linalg.inv(a_ii)

    def _get_residuals(
        self,
        boundary_system: tuple[NDArray, NDArray, NDArray],
        test_solution: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray | tuple[NDArray, ...]]:
        r"""Compute the residuals of a test solution.

            They are computed as follows:
            $$\mathbf{x}_{ref} = [\mathbf{a}_{ii} - \mathbf{a}_{ji} \mathbf{x}_{test} \mathbf{a}_{ij}]^{-1}$$
            $$\mathbf{residual_{abs}} = \lvert \mathbf{x}_{ref} - \mathbf{x}_{test} \rvert$$
            $$\mathbf{residual_{rel}} = \frac{\mathbf{residual_{abs}}}{\lvert \mathbf{x}_{ref} \rvert}$$

        Parameters
        ----------
        boundary_system : tuple[NDArray, NDArray, NDArray]
            The boundary system to solve.
            It is expected to be a tuple (a_ii, a_ij, a_ji)
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
        if type(test_solution) in [tuple, list]:
            # first output is the surface Green's function
            solution_ref = self._fix_point_step(boundary_system, test_solution[0])

            abs_residuals = xp.linalg.norm(
                solution_ref - test_solution[0], axis=(-2, -1)
            )
            rel_residuals = abs_residuals / xp.linalg.norm(solution_ref, axis=(-2, -1))

            return rel_residuals, abs_residuals, (solution_ref, *test_solution[1:])

        elif type(test_solution) is xp.ndarray:
            solution_ref = self._fix_point_step(boundary_system, test_solution)

            abs_residuals = xp.linalg.norm(solution_ref - test_solution, axis=(-2, -1))
            rel_residuals = abs_residuals / xp.linalg.norm(solution_ref, axis=(-2, -1))

            return rel_residuals, abs_residuals, solution_ref
        else:
            raise TypeError(
                f"Expected test_solution to be of type "
                f"xp.ndarray, tuple or list, got {type(test_solution)}"
            )
