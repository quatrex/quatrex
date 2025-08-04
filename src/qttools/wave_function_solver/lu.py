# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, sparse
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

if "cupy" in sparse.__name__:
    from cupyx.scipy.sparse import linalg
else:
    from scipy.sparse import linalg

profiler = Profiler()


class LU(WFSolver):
    """Wave function solver using LU decomposition for solving.

    This solver uses the SuperLU on the CPU for facorization. Depending
    on the chosen array module, the solution phase is computed on the
    CPU or GPU.

    """

    @profiler.profile(level="api")
    def solve(self, a: sparse.spmatrix, b: NDArray) -> NDArray:
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
        lu = linalg.splu(a)
        return lu.solve(b)
