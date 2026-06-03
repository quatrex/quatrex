# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, sparse, xp
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

try:
    from pydiso.mkl_solver import MATRIX_TYPES, MKLPardisoSolver

    _pardiso_available = True

except ImportError:
    _pardiso_available = False

profiler = Profiler()

valid_matrix_types = [
    "real_structurally_symmetric",
    "real_symmetric_positive_definite",
    "real_symmetric_indefinite",
    "complex_structurally_symmetric",
    "complex_hermitian_positive_definite",
    "complex_hermitian_indefinite",
    "complex_symmetric",
    "real_nonsymmetric",
    "complex_nonsymmetric",
]


def pardiso_available():
    """Checks if the PARDISO solver is available."""
    return _pardiso_available


class PARDISO(WFSolver):
    """Wave function solver using PARDISO for sparse matrix solving.

    This solver uses PARDISO to solve sparse linear systems on the CPU.
    It can reuse the analysis phase if configured to do so, which can
    speed up repeated solves with the same matrix structure.

    Parameters
    ----------
    matrix_type : str, optional
        The type of matrix to be solved. Must be one of the valid matrix
        types. Default is 'complex_nonsymmetric'.
    view : str, optional
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
        view: str | None = "default",
        verbose: bool = False,
    ) -> None:
        """Initializes the PARDISO wave function solver."""
        if not _pardiso_available:
            raise ImportError(
                "python-pardiso is not available. Please install it to use this solver."
            )
        if xp.__name__ != "numpy":
            raise ValueError("PARDISO solver requires numpy backend.")

        if view not in ["default", "up"]:
            raise ValueError(
                f"Invalid view '{view}'. Valid options are: ['default', 'up']"
            )  # The wiew option is not used internally. It is just a hint for the user to not use the down view, which is not supported by PARDISO.

        if matrix_type not in valid_matrix_types and matrix_type is not None:
            raise ValueError(
                f"Invalid matrix type '{matrix_type}'. Valid options are: {valid_matrix_types}"
            )
        self.matrix_type = matrix_type
        self.context = None
        self.verbose = verbose

    def solve(
        self,
        a: sparse.spmatrix,
        b: NDArray,
        reuse_sym_fact: bool = False,
        reuse_fact: bool = False,
    ) -> NDArray:
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

        if not reuse_sym_fact and reuse_fact:
            raise ValueError(
                "Cannot reuse total factorization without reusing symbolic factorization."
            )

        if not (reuse_sym_fact and self.context is not None):
            if self.matrix_type is not None:
                type = MATRIX_TYPES[self.matrix_type]
            else:
                type = None
            self.context = MKLPardisoSolver(
                a, matrix_type=type, factor=True, verbose=self.verbose
            )

        elif not (reuse_fact and self.context._factored):
            self.context.refactor(a)

        return self.context.solve(b)
