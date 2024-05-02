from abc import ABC, abstractmethod
from qttools.datastructures.dbsparse import DBSparse


class Solver(ABC):

    def __init__(self, config) -> None:
        pass

    @abstractmethod
    def selected_inv(a: DBSparse, out=None, **kwargs) -> None | DBSparse:
        """
        Perform the selected inversion of a matrix in block-tridiagonal form.

        Parameters
        ----------
        a : DBSparse
            Matrix to invert.
        out : _type_, optional
            Output matrix, by default None.

        Returns
        -------
        None | DBSparse
            If `out` is None, returns None. Otherwise, returns the inverted matrix
            as a DBSparse object.
        """
        ...

    @abstractmethod
    def selected_solve(
        a: DBSparse,
        sigma_lesser: DBSparse,
        sigma_greater: DBSparse,
        out: tuple | None = None,
        return_retarded: bool = False,
        **kwargs,
    ) -> None | tuple:
        """Solve the selected quadratic matrix equation and compute only selected
        elements of it's inverse.

        Parameters
        ----------
        a : DBSparse
            Matrix to invert.
        sigma_lesser : DBSparse
            Lesser matrix.
        sigma_greater : DBSparse
            Greater matrix.
        out : tuple | None, optional
            Output matrix, by default None
        return_retarded : bool, optional
            Weither the retarded Green's functioln should be returned, by default False

        Returns
        -------
        None | tuple
            If `out` is None, returns None. Otherwise, returns the inverted matrix
            as a DBSparse object. If `return_retarded` is True, returns a tuple with
            the retarded Green's function as the second element.
        """
        ...
