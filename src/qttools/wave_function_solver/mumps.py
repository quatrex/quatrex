# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, sparse, xp
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

try:
    import mumps

    mumps_available = True

except ImportError:
    mumps_available = False

profiler = Profiler()

valid_orderings = ["amd", "amf", "scotch", "pord", "metis", "qamd", "auto"]


class MUMPS(WFSolver):
    """Wave function solver using MUMPS for sparse matrix solving.

    This solver uses MUMPS to solve sparse linear systems on the CPU. It
    can reuse the analysis phase if configured to do so, which can speed
    up repeated solves with the same matrix structure.

    Parameters
    ----------
    reuse_analysis : bool, optional
        If True, reuse the analysis phase for subsequent solves with the
        same matrix structure. Default is True.
    ordering : str, optional
        The ordering method to use for the matrix factorization. Valid
        options are 'amd', 'amf', 'scotch', 'pord', 'metis', 'qamd', and
        'auto'. Default is 'auto'. The 'metis' and 'scotch' orderings
        are apparently usually pretty good. The 'auto' option will
        let MUMPS choose the "best" ordering. Default is 'metis'.
    verbose : bool, optional
        If True, enable verbose output from MUMPS. Default is False.

    """

    def __init__(
        self,
        reuse_analysis: bool = True,
        ordering: str = "metis",
        verbose: bool = False,
    ) -> None:
        """Initializes the MUMPS wave function solver."""
        if not mumps_available:
            raise ImportError(
                "python-mumps is not available. Please install it to use this solver."
            )
        if xp.__name__ != "numpy":
            raise ValueError("MUMPS solver requires numpy backend.")

        self.reuse_analysis = reuse_analysis
        if ordering not in valid_orderings:
            raise ValueError(
                f"Invalid ordering '{ordering}'. Valid options are: {valid_orderings}"
            )
        self.ordering = ordering
        self.context = mumps.Context(verbose=verbose)

    def solve(self, a: sparse.spmatrix, b: NDArray) -> NDArray:
        """Solves the sparse linear system a @ x = b using MUMPS.

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
        if not (self.reuse_analysis and self.context.analyzed):
            self.context.analyze(a, ordering=self.ordering)

        self.context.factor(
            a=a,
            ordering=self.ordering,
            reuse_analysis=self.reuse_analysis,
        )
        return self.context.solve(b)
