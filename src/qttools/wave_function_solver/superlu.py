# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, sparse
from qttools.wave_function_solver.solver import WFSolver

if "cupy" in sparse.__name__:
    from cupyx.scipy.sparse import linalg
else:
    from scipy.sparse import linalg


class SuperLU(WFSolver):
    """Wave function solver using LU decomposition for solving.

    This solver uses the SuperLU on the CPU for facorization. Depending
    on the chosen array module, the solution phase is computed on the
    CPU or GPU.

    """

    def __init__(
        self,
        matrix_type: str = "complex_nonsymmetric",
        view: str = "default",
    ) -> None:
        """Initializes the SuperLU wave function solver."""

        # Matrix_type is currently not used in the SuperLU solver

        if view != "default":
            raise ValueError("SuperLU solver currently only supports 'default' view.")

    def solve(
        self,
        a: sparse.spmatrix,
        b: NDArray,
        reuse_sym_fact: bool = False,
        reuse_fact: bool = False,
    ) -> NDArray:
        """Solves the sparse system a @ x = b using LU decomposition.

        Parameters
        ----------
        a : sparse.spmatrix
            The sparse system matrix.
        b : NDArray
            The right-hand side vector.

        Returns
        -------
        x : NDArray
            The solution vector.

        """

        if reuse_sym_fact or reuse_fact:
            print(
                "Warning: SuperLU solver does not support factorization or sym. factorization reuse. Matrix will be refactorized."
            )

        lu = linalg.splu(a)
        return lu.solve(b)
