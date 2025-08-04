# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, sparse
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

try:
    from nvmath.sparse.advanced import DirectSolver

    nvmath_available = True

except ImportError:
    nvmath_available = False


profiler = Profiler()


class cuDSS(WFSolver):
    """Wave function solver using cuDSS for sparse matrix solving.

    This solver uses the cuDSS library to solve sparse linear systems
    on NVIDIA GPUs.

    Parameters
    ----------
    explicitely_reset_operands : bool, optional
        If True, explicitly reset the operands (matrix and right-hand
        side vector) using the reset_operands() method before each
        solve. This can be useful if the matrix structure changes
        frequently, but may incur additional overhead. If False, the
        caller is responsible for ensuring that the matrix and
        right-hand side vector are correctly set before each solve.
        Default is True, meaning that the operands are reset explicitly
        before each solve.

    """

    def __init__(self, explicitely_reset_operands: bool = True) -> None:
        """Initializes the cuDSS wave function solver."""
        if not nvmath_available:
            raise ImportError(
                "cuDSS is not available. Please install it to use this solver."
            )

        self.direct_solver = None
        self.plan_info = None
        self.explicitely_reset_operands = explicitely_reset_operands

    def __del__(self) -> None:
        """Cleans up the cuDSS solver context."""
        if self.direct_solver is not None:
            self.direct_solver.free()
            self.direct_solver = None

    @profiler.profile(level="api")
    def solve(self, a: sparse.spmatrix, b: NDArray) -> NDArray:
        """Solves the sparse linear system a @ x = b using cuDSS.

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
        if self.direct_solver is None or self.plan_info is None:
            self.direct_solver = DirectSolver(a, b)
            # NOTE: By default this uses a custom nested dissectioning
            # scheme based on METIS. Other options could in principle be
            # exposed as a parameter.
            self.plan_info = self.direct_solver.plan()

        if self.explicitely_reset_operands:
            self.direct_solver.reset_operands(a=a, b=b)

        self.direct_solver.factorize()
        return self.direct_solver.solve(b)
