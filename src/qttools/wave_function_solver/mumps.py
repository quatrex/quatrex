# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

try:
    import mumps

    mumps_available = True

except ImportError:
    mumps_available = False

from qttools import NDArray, sparse, xp
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

profiler = Profiler()


class MUMPS(WFSolver):
    """Wave function solver using MUMPS for sparse matrix solving.

    This solver uses MUMPS to solve sparse linear systems on the CPU. It
    can reuse the analysis phase if configured to do so, which can speed
    up repeated solves with the same matrix structure.

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
    ordering : str, optional
        The ordering method to use for the matrix factorization. Valid
        options are 'amd', 'amf', 'scotch', 'pord', 'metis', 'qamd', and
        'auto'. The 'metis' and 'scotch' orderings are apparently
        usually pretty good. The 'auto' option will let MUMPS choose the
        "best" ordering. Default is 'metis'.
    verbose : bool, optional
        If True, enable verbose output from MUMPS. Default is False.

    """

    def __init__(
        self,
        matrix_type: str = "complex_nonsymmetric",
        matrix_view: str = "full",
        ordering: str = "metis",
        verbose: bool = False,
    ) -> None:
        """Initializes the MUMPS wave function solver."""
        if not mumps_available:
            raise ImportError(
                "python-mumps is not available. "
                "Please install it to use this solver."
            )
        if xp.__name__ != "numpy":
            raise ValueError("MUMPS solver requires numpy backend.")

        if matrix_type != "complex_nonsymmetric" or matrix_view != "full":
            raise ValueError(
                "MUMPS solver currently only supports 'complex_nonsymmetric' "
                "matrix type and 'full' matrix view."
            )

        if ordering not in mumps.orderings:
            raise ValueError(
                f"Invalid ordering '{ordering}'. "
                f"Valid options are: {mumps.orderings}"
            )

        self.ordering = ordering
        self._context = mumps.Context(verbose=verbose)

    @profiler.profile("MUMPS solve", level="default")
    def solve(
        self,
        a: sparse.spmatrix,
        b: NDArray,
        reuse_analysis: bool = False,
        reuse_factorization: bool = False,
    ) -> NDArray:
        """Solves the sparse linear system a @ x = b using MUMPS.

        Parameters
        ----------
        a : sparse.spmatrix
            The sparse system matrix.
        b : NDArray
            The dense right-hand side vector.
        reuse_analysis : bool, optional
            Whether to reuse the analysis phase from a previous solve,
            by by default False. This is useful when solving multiple
            linear systems with the same sparsity pattern but different
            numerical values.
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

        if not self._context.analyzed or not reuse_analysis:
            with profiler.profile_range("MUMPS: analysis", level="default"):
                self._context.analyze(a, ordering=self.ordering)

        if not self._context.factored or not reuse_factorization:
            with profiler.profile_range("MUMPS: factorization", level="default"):
                self._context.factor(a, ordering=self.ordering, reuse_analysis=True)

        return self._context.solve(b)
