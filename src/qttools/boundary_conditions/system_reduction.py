# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from abc import ABC, abstractmethod

from qttools import NDArray


class SystemReducer(ABC):

    @abstractmethod
    def contract_system(
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
    def expand_solution(
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
    def expand_residuals(
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


class IdentityReducer(SystemReducer):
    def contract_system(
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
        return boundary_system

    def expand_solution(
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
        return reduced_solution

    def expand_residuals(
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
        return rel_residuals, abs_residuals
