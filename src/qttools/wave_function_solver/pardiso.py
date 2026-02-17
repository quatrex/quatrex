# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, sparse, xp
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

try:
    from pydiso.mkl_solver import MATRIX_TYPES, MKLPardisoSolver

    pardiso_available = True

except ImportError:
    pardiso_available = False

profiler = Profiler()


class PARDISO(WFSolver):
    """Wave function solver using PARDISO for sparse matrix solving.

    This solver uses PARDISO to solve sparse linear systems on the CPU. It
    can reuse the analysis phase if configured to do so, which can speed
    up repeated solves with the same matrix structure.

    Parameters
    ----------
    reuse_analysis : bool, optional
        If True, reuse the analysis phase for subsequent solves with the
        same matrix structure. Default is True.
    verbose : bool, optional
        If True, enable verbose output from MUMPS. Default is False.

    """

    def __init__(
        self,
        reuse_factorization: bool = True,
        hermitian_matrix: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initializes the PARDISO wave function solver."""
        if not pardiso_available:
            raise ImportError(
                "python-pardiso is not available. Please install it to use this solver."
            )
        if xp.__name__ != "numpy":
            raise ValueError("PARDISO solver requires numpy backend.")

        self.reuse_factorization = reuse_factorization
        self.hermitian_matrix = hermitian_matrix
        self.context = None
        self.verbose = verbose

    @profiler.profile(level="api")
    def solve(self, a: sparse.spmatrix, b: NDArray) -> NDArray:
        """Solves the sparse linear system a @ x = b using PARDISO.

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
        if self.context is None or not self.reuse_factorization:
            if self.hermitian_matrix:
                type = MATRIX_TYPES["complex_hermitian_indefinite"]
            else:
                type = MATRIX_TYPES["complex_nonsymmetric"]
            self.context = MKLPardisoSolver(
                a, matrix_type=type, factor=True, verbose=self.verbose
            )

        else:
            print("Reusing PARDISO factorization.")
            self.context.refactor(a)

        return self.context.solve(b)
