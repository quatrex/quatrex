# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes the abstract base class for the Green's function solvers."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass
class BackSubstitutionContext:
    """Context for the back substitution step of the selected solve.

    This is used to compute observables such as the Meir-Wingreen
    current and the device current during the back substitution step of
    the selected solve. This is needed because we discard the dense
    off-diagonal blocks after they are used to update the diagonal
    blocks.

    Attributes
    ----------
    i : int
        Current layer index.
    j : int
        Previous layer index.
    stack_slice : slice
        Stack slice for the current batch.
    a_ij : NDArray
        Off-diagonal system matrix block from layer i to layer j.
    a_ji : NDArray
        Off-diagonal system matrix block from layer j to layer i.
    xr_hat_ii : NDArray
        Dense Retarded Green's function diagonal block **before** back
        substitution update.
    xl_hat_ii : NDArray
        Dense lesser GF diagonal block **before** back substitution
        update.
    xl_ij : NDArray
        Dense lesser GF off-diagonal block **after** back substitution
        update.
    xl_jj : NDArray
        Dense lesser GF diagonal block **after** back substitution
        update.
    xg_hat_ii : NDArray
        Dense greater GF diagonal block **before** back substitution
        update.
    xg_ij : NDArray
        Dense greater GF off-diagonal block **after** back substitution
        update.
    xg_jj : NDArray
        Dense greater GF diagonal block **after** back substitution
        update.
    sigma_lesser_ij : NDArray
        Off-diagonal lesser self-energy block from layer i to layer j.
    sigma_greater_ij : NDArray
        Off-diagonal greater self-energy block from layer i to layer j.

    """

    i: int | None = None
    j: int | None = None

    stack_slice: slice | None = None

    xl_ij: NDArray | None = None

    a_ij: NDArray | None = None
    a_ji: NDArray | None = None

    obc_blocks: OBCBlocks | None = None

    xr_hat_ii: NDArray | None = None

    xl_hat_ii: NDArray | None = None
    xl_jj: NDArray | None = None

    xg_hat_ii: NDArray | None = None
    xg_ij: NDArray | None = None
    xg_jj: NDArray | None = None

    sigma_lesser_ij: NDArray | None = None
    sigma_greater_ij: NDArray | None = None


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
        callbacks: list[Callable[[BackSubstitutionContext], None]] | None = None,
    ) -> None:
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
        callbacks : list[Callable], optional
            List of callback functions to be called during the back
            substitution step. Each callback function should accept a
            single argument of type `BackwardSubstitutionContext`, by
            default None.

        """
        ...
