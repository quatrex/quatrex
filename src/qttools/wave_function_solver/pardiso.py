# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

try:
    from pydiso.mkl_solver import MATRIX_TYPES, MKLPardisoSolver

    pardiso_available = True

except ImportError:
    pardiso_available = False

from qttools import NDArray, sparse, xp
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

profiler = Profiler()


class PARDISO(WFSolver):
    """Wave function solver using PARDISO for sparse matrix solving.

    Parameters
    ----------
    matrix_type : str, optional
        The type of the system matrix. Must be one of the valid PARDISO
        matrix types. Default is 'complex_nonsymmetric'.
    matrix_view : str, optional
        The view of the matrix. Valid options are 'default' and 'up'.
        The 'up' view is a hint to the user to use the upper triangular
        part of the matrix, which is required for symmetric matrices.
        Default is 'default', meaning the full matrix is used.
    verbose : bool, optional
        If True, enable verbose output from PARDISO. Default is False.

    """

    def __init__(
        self,
        matrix_type: str | None = None,
        matrix_view: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Initializes the PARDISO wave function solver."""
        if not pardiso_available:
            raise ImportError(
                "python-pardiso is not available. "
                "Please install it to use this solver."
            )
        if xp.__name__ != "numpy":
            raise ValueError("PARDISO solver requires numpy backend.")

        if matrix_type is not None and matrix_type not in MATRIX_TYPES:
            raise ValueError(
                f"Invalid matrix type '{matrix_type}'. "
                f"Valid options are: {list(MATRIX_TYPES.keys())}"
            )
        self.matrix_type = matrix_type

        if matrix_view is not None and matrix_view not in ["full", "upper"]:
            # NOTE: The matrix view option is not explicitely passed to
            # PARDISO. For matrices with symmetric structure, the solver
            # always uses the upper triangular part, so this is just a
            # hint to the user.
            raise ValueError(
                f"Invalid view '{matrix_view}'. "
                f"Valid options are 'full' or 'upper'."
            )

        self.verbose = verbose

        self._context = None

    @profiler.profile("PARDISO solve", level="default")
    def solve(
        self,
        a: sparse.spmatrix,
        b: NDArray,
        reuse_analysis: bool = False,
        reuse_factorization: bool = False,
    ) -> NDArray:
        """Solves the sparse linear system a @ x = b using PARDISO.

        Parameters
        ----------
        a : sparse.spmatrix
            The sparse system matrix.
        b : NDArray
            The right-hand side vector.
        reuse_analysis : bool, optional
            Whether to reuse the symbolic factorization from a previous
            solve, by default False. This is useful when solving
            multiple linear systems with the same sparsity pattern but
            different numerical values.
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
        if reuse_factorization and not reuse_analysis:
            raise ValueError(
                "Cannot reuse total factorization without reusing symbolic factorization."
            )

        if self._context is None or not reuse_analysis:
            with profiler.profile_range("PARDISO: analysis", level="default"):
                self._context = MKLPardisoSolver(
                    a, matrix_type=self.matrix_type, factor=False, verbose=self.verbose
                )

        if not self._context._factored or not reuse_factorization:
            with profiler.profile_range("PARDISO: factorization", level="default"):
                self._context.refactor(a)

        return self._context.solve(b)
