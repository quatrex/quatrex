# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from abc import ABC, abstractmethod

from qttools import NDArray


class SystemReducer(ABC):
    """Interface for reducing the boundary system to a smaller system.
    The boundary system can exhibit sparsity or symmetries, which can be
    exploited to reduce the size of the system.

    """

    @abstractmethod
    def contract_system(
        self,
        boundary_system: tuple[NDArray, ...],
    ) -> tuple[NDArray, ...]:
        """Contract the boundary system to a reduced system.
        This method should identify and remove redundancies in the boundary system,
        resulting in a smaller system that is easier to solve.

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
    def expand_solution(
        self,
        boundary_system: tuple[NDArray, ...],
        reduced_system: tuple[NDArray, ...],
        reduced_solution: NDArray | tuple[NDArray, ...],
    ) -> NDArray | tuple[NDArray, ...]:
        """Expand the solution from the reduced system to the full system.
        The method should be called after `contract_system` and solving the system.
        It should take the solution of the reduced system and map it back to the full system,
        filling in any missing values as necessary.

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
    def expand_residuals(
        self,
        boundary_system: tuple[NDArray, ...],
        reduced_system: tuple[NDArray, ...],
        rel_residuals: NDArray,
        abs_residuals: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Expand the residuals from the reduced system to the full system.
        The method should be called after `contract_system` and solving the system.
        It should take the residuals of the reduced system and map them back to the full system,
        filling in any missing values as necessary.

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


class IdentityReducer(SystemReducer):
    """A system reducer that does not perform any reduction. It simply returns the input as output.
    This can be used as a default reducer when no reduction is desired."""

    def contract_system(
        self,
        boundary_system: tuple[NDArray, ...],
    ) -> tuple[NDArray, ...]:
        """Do not contract the boundary system, return it as is.
        The retured system alias the input system.

        Parameters
        ----------
        boundary_system : tuple[NDArray, ...]
            The full boundary system.

        Returns
        -------
        reduced_system : tuple[NDArray, ...]
            The reduced boundary system.

        """
        return boundary_system

    def expand_solution(
        self,
        boundary_system: tuple[NDArray, ...],
        reduced_system: tuple[NDArray, ...],
        reduced_solution: NDArray | tuple[NDArray, ...],
    ) -> NDArray | tuple[NDArray, ...]:
        """Do not expand the solution, return it as is.
        The returned solution alias the input solution.

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
        return reduced_solution

    def expand_residuals(
        self,
        boundary_system: tuple[NDArray, ...],
        reduced_system: tuple[NDArray, ...],
        rel_residuals: NDArray,
        abs_residuals: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Do not expand the residuals, return them as is.
        The returned residuals alias the input residuals.

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
        return rel_residuals, abs_residuals
