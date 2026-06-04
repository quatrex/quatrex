# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, sparse
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

profiler = Profiler()


if sparse.__name__ == "cupyx.scipy.sparse":
    from cupyx.scipy.sparse import linalg
else:
    from scipy.sparse import linalg


class SuperLU(WFSolver):
    """Wave function solver using LU decomposition for solving.

    This solver uses the SuperLU on the CPU for facorization. Depending
    on the chosen array module, the solution phase is computed on the
    CPU or GPU.

    Parameters
    ----------
    matrix_type : str, optional
        The type of matrix to be solved. The only valid option is
        'complex_nonsymmetric', which is the default. This is a
        placeholder for future support of other matrix types.
    matrix_view : str, optional
        The view of the matrix. The only valid option is 'full', which
        means the full matrix is used. Default is 'full'. This is a
        placeholder for future support of other matrix views.

    """

    def __init__(
        self,
        matrix_type: str = "complex_nonsymmetric",
        matrix_view: str = "full",
    ) -> None:
        """Initializes the SuperLU wave function solver."""

        if matrix_type != "complex_nonsymmetric" or matrix_view != "full":
            raise ValueError(
                "SuperLU solver currently only supports 'complex_nonsymmetric' "
                "matrix type and 'full' matrix view."
            )

        self._lu = None

    @profiler.profile("SuperLU solve", level="default")
    def solve(
        self,
        a: sparse.spmatrix,
        b: NDArray,
        reuse_analysis: bool = False,
        reuse_factorization: bool = False,
    ) -> NDArray:
        """Solves the sparse system a @ x = b using LU decomposition.

        Parameters
        ----------
        a : sparse.spmatrix
            The sparse system matrix.
        b : NDArray
            The right-hand side vector.
        reuse_analysis : bool, optional
            Unused for this solver since it does not involve an analysis
            phase.
        reuse_factorization : bool, optional
            Whether to reuse the numerical factorization from a previous
            solve, by default False. Note that this must only be
            True if the matrix values have not changed since the last
            factorization.

        Returns
        -------
        x : NDArray
            The solution vector.

        """
        if reuse_analysis:
            raise ValueError("SuperLU solver does not support reuse of analysis.")

        if self._lu is None or not reuse_factorization:
            with profiler.profile_range("SuperLU: factorization", level="default"):
                self._lu = linalg.splu(a)

        return self._lu.solve(b)
