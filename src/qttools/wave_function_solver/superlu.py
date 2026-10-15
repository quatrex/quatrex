# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes the SuperLU wave function solver."""

import warnings

from qttools import NDArray, sparse, xp
from qttools.comm import comm as quatrex_comm
from qttools.comm.comm import _SubCommunicator
from qttools.profiling import Profiler
from qttools.wave_function_solver.solver import WFSolver

profiler = Profiler()


if sparse.__name__ == "cupyx.scipy.sparse":
    from cupyx.scipy.sparse import linalg
else:
    from scipy.sparse import linalg

_ALLOWED_MATRIX_VIEWS = ["full"]


class SuperLU(WFSolver):
    """Wave function solver using LU decomposition for solving.

    This solver uses the SuperLU on the CPU for facorization. Depending
    on the chosen array module, the solution phase is computed on the
    CPU or GPU.

    Parameters
    ----------
    matrix_dtype : xp.dtype[xp.inexact]
        The data type of the system matrix. This needs to be
        provided to correctly initialize the solver.
    matrix_property : str
        The property of the system matrix. This describes properties
        like symmetry and definiteness, which can be used by solvers
        to optimize the solution process. Can be None if the solver
        does not require this information or if it can be inferred
        from the matrix itself.
    matrix_view : str
        The view of the system matrix sparsity. Not all solvers are
        compatible with all views.
    reuse_analysis : bool, optional
        Whether to reuse the analysis phase when solve is called
        multiple times, by default False. This typically includes
        symbolic factorization and ordering but can vary between
        solvers. This is useful when solving multiple linear systems
        with the same sparsity pattern but different numerical
        values.
    reuse_factorization : bool, optional
        Whether to reuse the numerical factorization when solve is
        called multiple times, by default False. This can only be
        True if reuse_analysis is also True. Note that this must
        only be True if the matrix values have not changed since the
        last factorization.

    """

    @staticmethod
    def valid_views() -> list[str]:
        """Returns the valid matrix views.

        Returns
        -------
        list[str]
            The valid matrix views.

        """
        return _ALLOWED_MATRIX_VIEWS

    def __init__(
        self,
        matrix_dtype: xp.dtype[xp.inexact],
        matrix_property: str,
        matrix_view: str,
        reuse_analysis: bool = False,
        reuse_factorization: bool = False,
        comm: _SubCommunicator | None = None,
        local_rows: tuple | None = None,
    ) -> None:
        """Initializes the SuperLU wave function solver."""

        if comm is not None and comm.size != 1:
            raise ValueError(
                "SuperLU solver does not support distributed solves. "
                "Please use a different solver for distributed solves."
            )

        # NOTE: This is a bit of hack to not need to modify the logic
        # when allocating the solver in QTBM.
        if matrix_view not in self.valid_views():
            if quatrex_comm.rank == 0:
                warnings.warn(
                    f"Matrix view '{matrix_view}' is not supported by the SuperLU solver. "
                    f"Using 'full' instead."
                )
            matrix_view = "full"

        if reuse_analysis:
            if quatrex_comm.rank == 0:
                warnings.warn(
                    "SuperLU solver does not support an analysis phase, so "
                    "the reuse_analysis flag is ignored."
                )
            reuse_analysis = False

        super().__init__(
            matrix_dtype=matrix_dtype,
            matrix_property=matrix_property,
            matrix_view=matrix_view,
            reuse_analysis=reuse_analysis,
            reuse_factorization=reuse_factorization,
            comm=comm,
            local_rows=local_rows,
        )

        self._lu = None

    @profiler.profile("SuperLU solve", level="default")
    def solve(
        self,
        a: sparse.spmatrix,
        b: NDArray,
    ) -> NDArray:
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
        if self._lu is None or not self.reuse_factorization:
            with profiler.profile_range("SuperLU: factorization", level="default"):
                self._lu = linalg.splu(a)

        return self._lu.solve(b)
