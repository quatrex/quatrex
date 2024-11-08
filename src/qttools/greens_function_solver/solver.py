# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from abc import ABC, abstractmethod

from qttools.datastructures import DSBSparse


class GFSolver(ABC):
    @abstractmethod
    def selected_inv(
        self, a: DSBSparse, out: DSBSparse = None, max_batch_size: int = 1
    ) -> None | DSBSparse:
        """Perform the selected inversion of a matrix in block-tridiagonal form.

        Parameters
        ----------
        a : DSBSparse
            Matrix to invert.
        out : DSBSparse, optional
            Output matrix, by default None.
        max_batch_size : int, optional
            Maximum batch size to use when inverting the matrix, by default 1.

        Returns
        -------
        None | DSBSparse
            If `out` is None, returns None. Otherwise, returns the inverted matrix
            as a DSBSparse object.
        """
        ...

    @abstractmethod
    def selected_solve(
        self,
        a: DSBSparse,
        sigma_lesser: DSBSparse,
        sigma_greater: DSBSparse,
        out: tuple | None = None,
        return_retarded: bool = False,
        max_batch_size: int = 1,
    ) -> None | tuple:
        """Perform a selected-solve of the congruence matrix equation: A * X * A^T = B.

        Parameters
        ----------
        a : DSBSparse
            Matrix to invert.
        sigma_lesser : DSBSparse
            Lesser matrix. This matrix is expected to be skewed-hermitian.
        sigma_greater : DSBSparse
            Greater matrix. This matrix is expected to be skewed-hermitian.
        out : tuple | None, optional
            Output matrix, by default None
        return_retarded : bool, optional
            Weither the retarded Green's functioln should be returned, by default False
        max_batch_size : int, optional
            Maximum batch size to use when inverting the matrix, by default 1

        Returns
        -------
        None | tuple
            If `out` is None, returns None. Otherwise, returns the inverted matrix
            as a DSBSparse object. If `return_retarded` is True, returns a tuple with
            the retarded Green's function as the last element.
        """
        ...
