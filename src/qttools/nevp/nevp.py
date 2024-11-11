# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from abc import ABC, abstractmethod

from qttools import NDArray


class NEVP(ABC):
    """Abstract base class for the non-linear eigenvalue solvers."""

    @abstractmethod
    def __call__(self, a_xx: tuple[NDArray, ...]) -> tuple[NDArray, NDArray]:
        """Solves the plynomial eigenvalue problem.

        This method solves the non-linear eigenvalue problem defined by
        the coefficient blocks `a_xx` from lowest to highest order.

        \\[
            \\left( \sum_{n=-b}^{b} a_n w^n \\right) v = 0
        \\]

        Parameters
        ----------
        a_xx : tuple[NDArray, ...]
            The coefficient blocks of the non-linear eigenvalue problem
            from lowest to highest order.

        Returns
        -------
        ws : NDArray
            The eigenvalues.
        vs : NDArray
            The eigenvectors.

        """
        ...
