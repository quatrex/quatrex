# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from abc import ABC, abstractmethod

from qttools import NDArray, sparse


class WFSolver(ABC):
    """Abstract base class for wave function solvers."""

    def __init__(self, matrix_type: str | None, matrix_view: str | None) -> None:
        """Initializes the wave function solver.

        Parameters
        ----------
        matrix_type : str, optional
            The type of the system matrix. This describes properties
            like symmetry and definiteness, which can be used by solvers
            to optimize the solution process. Can be None if the solver
            does not require this information or if it can be inferred
            from the matrix itself.
        matrix_view : str, optional
            The view of the system matrix sparsity. This is a hint to
            the solver about which part of the matrix to use, which can
            be relevant for symmetric matrices where only the upper or
            lower part is needed. Can be None if the solver does not
            require this information or if it can be inferred from the
            matrix itself.

        """
        ...

    @abstractmethod
    def solve(
        self,
        a: sparse.spmatrix,
        b: NDArray,
        reuse_analysis: bool = False,
        reuse_factorization: bool = False,
    ) -> NDArray:
        """Solves the sparse linear system a @ x = b.

        Parameters
        ----------
        a : sparse.spmatrix
            The sparse system matrix.
        b : NDArray
            The right-hand side vector.
        reuse_analysis : bool, optional
            Whether to reuse the analysis phase from a previous solve,
            by default False. This typically includes symbolic
            factorization and ordering but can vary between solvers.
            This is useful when solving multiple linear systems with the
            same sparsity pattern but different numerical values.
        reuse_factorization : bool, optional
            Whether to reuse the numerical factorization from a previous
            solve, by default False. This can only be True if
            reuse_analysis is also True. Note that this must only be
            True if the matrix values have not changed since the last
            factorization.

        Returns
        -------
        x : NDArray
            The solution vector.

        """
        ...
