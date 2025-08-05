# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.

import ctypes.util

from qttools import NDArray, sparse
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

try:
    from nvmath.sparse.advanced import DirectSolver

    nvmath_available = True

except ImportError:
    nvmath_available = False


profiler = Profiler()

# Possible multithreading libraries for cuDSS in descending order of
# preference.
_mtlayer_libs = [
    "libcudss_mtlayer_gomp.so.0",
    "libcudss_mtlayer_gomp.so.0.5.0",
]


class cuDSS(WFSolver):
    """Wave function solver using cuDSS for sparse matrix solving.

    This solver uses the cuDSS library to solve sparse linear systems
    on NVIDIA GPUs.

    Parameters
    ----------
    explicitely_reset_operands : str, optional
        String indicating which operands to reset explicitly. If "a" is
        in the string, the system matrix `a` will be reset before
        solving. If "b" is in the string, the right-hand side vector `b`
        will be reset before solving. Default is "b", meaning only the
        right-hand side vector will be reset.
    use_multithreading : bool, optional
        Whether to use multithreading for the solver. If True, it will
        attempt to find a suitable multithreading library. Default is
        True.

    """

    def __init__(
        self,
        explicitely_reset_operands: str = "b",
        use_multithreading: bool = True,
    ) -> None:
        """Initializes the cuDSS wave function solver."""
        if not nvmath_available:
            raise ImportError(
                "cuDSS is not available. Please install it to use this solver."
            )

        self.direct_solver = None
        self.plan_info = None
        self.explicitely_reset_a = "a" in explicitely_reset_operands
        self.explicitely_reset_b = "b" in explicitely_reset_operands

        self.solver_options = {}

        if use_multithreading:
            # Try to find a multithreading library for cuDSS.
            multithreading_lib = [
                ctypes.util.find_library(lib) for lib in _mtlayer_libs
            ].pop(0)

            if multithreading_lib is None:
                raise ImportError("No suitable multithreading library found for cuDSS.")

            self.solver_options["multithreading_lib"] = multithreading_lib

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
            self.direct_solver = DirectSolver(a, b, options=self.solver_options)
            # NOTE: By default this uses a custom nested dissectioning
            # scheme based on METIS. Other options could in principle be
            # exposed as a parameter.
            self.plan_info = self.direct_solver.plan()

        if self.explicitely_reset_a:
            self.direct_solver.reset_operands(a=a)
            # After resetting a, we need to re-plan.
            self.plan_info = self.direct_solver.plan()

        # TODO: This does not support setting a right-hand side that has
        # a different shape or strides than the original one. This makes
        # nvmath-python a bad fit for us.
        if self.explicitely_reset_b:
            self.direct_solver.reset_operands(b=b)

        self.direct_solver.factorize()
        return self.direct_solver.solve()
