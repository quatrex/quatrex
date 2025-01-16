# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from abc import ABC, abstractmethod

from qttools import NDArray


class NEVP(ABC):
    """Abstract base class for the non-linear eigenvalue solvers."""

    @abstractmethod
    def __call__(self, a_xx: tuple[NDArray, ...], left: bool = False) -> tuple:
        r"""Solves the plynomial eigenvalue problem.

        This method solves the non-linear eigenvalue problem defined by
        the coefficient blocks `a_xx` from lowest to highest order.

        \[
            \left( \sum_{n=-b}^{b} a_n w^n \right) v = 0
        \]

        Parameters
        ----------
        a_xx : tuple[NDArray, ...]
            The coefficient blocks of the non-linear eigenvalue problem
            from lowest to highest order.
        left : bool, optional
            Whether to solve additionally for the left eigenvectors.

        Returns
        -------
        ws : NDArray
            The right eigenvalues.
        vs : NDArray
            The right eigenvectors.
        wl : NDArray, optional
            The left eigenvalues.
            Returned only if `left` is `True`.
        vl : NDArray, optional
            The left eigenvectors.
            Returned only if `left` is `True`.

        """
        ...
