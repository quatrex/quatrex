# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from abc import ABC, abstractmethod

from qttools import NDArray
from qttools.datastructures import DSBSparse


# NOTE: Maybe it's overkill to have a class for this, but makes the
# grouping a bit easier. Also thinking forward to the possibility of
# adding more contacts in the future.
class OBCBlocks:
    """Class to hold the OBC blocks used in the GF solvers.

    This class holds the OBC blocks for lesser, greater and retarded
    Green's functions. These are lists of NDArray objects.

    Parameters
    ----------
    num_blocks : int
        Number of blocks in the structure.

    """

    def __init__(self, num_blocks: int):
        self.retarded: list[NDArray | None] = [None] * num_blocks
        self.lesser: list[NDArray | None] = [None] * num_blocks
        self.greater: list[NDArray | None] = [None] * num_blocks


class GFSolver(ABC):
    """Abstract base class for the Green's function solvers."""

    @abstractmethod
    def selected_inv(
        self,
        a: DSBSparse,
        obc_blocks: OBCBlocks | None = None,
        out: DSBSparse = None,
    ) -> None | DSBSparse:
        """Performs selected inversion of a block-tridiagonal matrix.

        Parameters
        ----------
        a : DSBSparse
            Matrix to invert.
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.
        out : DSBSparse, optional
            Preallocated output matrix, by default None.

        Returns
        -------
        None | DSBSparse
            If `out` is None, returns None. Otherwise, returns the
            inverted matrix as a DSBSparse object.

        """
        ...

    @abstractmethod
    def selected_solve(
        self,
        a: DSBSparse,
        sigma_lesser: DSBSparse,
        sigma_greater: DSBSparse,
        obc_blocks: OBCBlocks | None = None,
        out: tuple | None = None,
        return_retarded: bool = False,
        return_current: bool = False,
    ) -> None | tuple:
        r"""Produces elements of the solution to the congruence equation.

        This method produces selected elements of the solution to the
        relation:

        \[
            X^{\lessgtr} = A^{-1} \Sigma^{\lessgtr} A^{-\dagger}
        \]

        Parameters
        ----------
        a : DSBSparse
            Matrix to invert.
        sigma_lesser : DSBSparse
            Lesser matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        sigma_greater : DSBSparse
            Greater matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        obc_blocks : dict[int, OBCBlocks], optional
            OBC blocks for lesser, greater and retarded Green's
            functions, by default None.
        out : tuple[DSBSparse, ...] | None, optional
            Preallocated output matrices, by default None
        return_retarded : bool, optional
            Wether the retarded Green's function should be returned
            along with lesser and greater, by default False
        return_current : bool, optional
            Whether to compute and return the current for each layer via
            the Meir-Wingreen formula. By default False.

        Returns
        -------
        None | tuple
            If `out` is None, returns None. Otherwise, the solutions are
            returned as DSBParse matrices. If `return_retarded` is True,
            returns a tuple with the retarded Green's function as the
            last element.

        """
        ...
