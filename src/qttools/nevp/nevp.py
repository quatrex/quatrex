# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from abc import ABC, abstractmethod

from qttools import xp


class NEVP(ABC):
    """Abstract base class for the non-linear eigenvalue solvers."""

    @abstractmethod
    def __call__(self, a_xx: list[xp.ndarray]) -> tuple[xp.ndarray, xp.ndarray]:
        """Solves the eigenvalue problem.

        This method solves the non-linear eigenvalue problem defined by
        the coefficient blocks `a_xx` from lowest to highest order.


        Parameters
        ----------
        a_xx : xp.ndarray
            The coefficient blocks of the non-linear eigenvalue problem
            from lowest to highest order.

        Returns
        -------
        ws : xp.ndarray
            The eigenvalues.
        vs : xp.ndarray
            The eigenvectors.

        """
        ...
