# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, sparse, xp
from qttools.wave_function_solver.solver import WFSolver

try:
    import mumps

    _mumps_available = True

except ImportError:
    _mumps_available = False

valid_orderings = ["amd", "amf", "scotch", "pord", "metis", "qamd", "auto"]


def mumps_available() -> bool:
    """Checks if the MUMPS solver is available.

    Returns
    -------
    bool
        True if MUMPS is available, False otherwise.
    """
    return _mumps_available


class MUMPS(WFSolver):
    """Wave function solver using MUMPS for sparse matrix solving.

    This solver uses MUMPS to solve sparse linear systems on the CPU. It
    can reuse the analysis phase if configured to do so, which can speed
    up repeated solves with the same matrix structure.

    Parameters
    ----------
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
        matrix_type: str = "complex_nonsymmetric",
        view: str = "default",
        ordering: str = "metis",
        verbose: bool = False,
    ) -> None:
        """Initializes the MUMPS wave function solver."""
        if not mumps_available():
            raise ImportError(
                "python-mumps is not available. Please install it to use this solver."
            )
        if xp.__name__ != "numpy":
            raise ValueError("MUMPS solver requires numpy backend.")

        if matrix_type != "complex_nonsymmetric" or view != "default":
            raise ValueError(
                "MUMPS solver currently only supports 'complex_nonsymmetric' matrix type and 'default' view."
            )

        if ordering not in valid_orderings:
            raise ValueError(
                f"Invalid ordering '{ordering}'. Valid options are: {valid_orderings}"
            )
        self.ordering = ordering
        self.context = mumps.Context(verbose=verbose)

    def solve(
        self,
        a: sparse.spmatrix,
        b: NDArray,
        reuse_sym_fact: bool = False,
        reuse_fact: bool = False,
    ) -> NDArray:
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

        if not reuse_sym_fact and reuse_fact:
            raise ValueError(
                "Cannot reuse total factorization without reusing symbolic factorization."
            )

        if not (reuse_sym_fact and self.context.analyzed):
            self.context.analyze(a, ordering=self.ordering)

        if not (reuse_fact and self.context.factored):
            self.context.factor(
                a=a,
                ordering=self.ordering,
                reuse_analysis=True,
            )

        return self.context.solve(b)
