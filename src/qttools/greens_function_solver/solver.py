# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes the abstract base class for the Green's function solvers."""

from abc import ABC, abstractmethod

from qttools import NDArray
from qttools.datastructures import DSDBSparse


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
        a: DSDBSparse,
        out: DSDBSparse,
        obc_blocks: OBCBlocks | None = None,
    ) -> None:
        """Performs selected inversion of a block-tridiagonal matrix.

        Parameters
        ----------
        a : DSDBSparse
            Matrix to invert.
        out : DSDBSparse
            Preallocated output matrix.
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.

        """
        ...

    @abstractmethod
    def selected_solve(
        self,
        a: DSDBSparse,
        sigma_lesser: DSDBSparse,
        sigma_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
        obc_blocks: OBCBlocks | None = None,
        return_retarded: bool = False,
        return_meir_wingreen_current: bool = False,
        return_device_current: bool = False,
    ) -> None | NDArray:
        r"""Produces elements of the solution to the congruence equation.

        This method produces selected elements of the solution to the
        relation:

        \[
            X^{\lessgtr} = A^{-1} \Sigma^{\lessgtr} A^{-\dagger}
        \]

        Parameters
        ----------
        a : DSDBSparse
            Matrix to invert.
        sigma_lesser : DSDBSparse
            Lesser matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        sigma_greater : DSDBSparse
            Greater matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        out : tuple[DSDBSparse, ...]
            Preallocated output matrices
        obc_blocks : dict[int, OBCBlocks], optional
            OBC blocks for lesser, greater and retarded Green's
            functions, by default None.
        return_retarded : bool, optional
            Wether the retarded Green's function should be returned
            along with lesser and greater, by default False
        return_meir_wingreen_current : bool, optional
            Whether to compute and return the current for each layer via
            the Meir-Wingreen formula. By default False.
        return_device_current : bool, optional
            Whether to additionally compute and return the coherent bond
            current between adjacent blocks, evaluated from the *dense*
            off-diagonal Green's function blocks. Only supported
            together with `return_meir_wingreen_current`. By default
            False.

        Returns
        -------
        None | tuple | NDArray
            If `return_meir_wingreen_current` is True, returns the
            current for each layer. If `return_device_current` is True,
            the bond current as well.

        """
        ...
