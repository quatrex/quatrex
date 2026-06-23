# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.


from qttools import xp
from qttools.comm import comm
from qttools.wave_function_solver.cudss import cuDSS, cudss_available
from qttools.wave_function_solver.mumps import MUMPS, mumps_available
from qttools.wave_function_solver.pardiso import PARDISO, pardiso_available
from qttools.wave_function_solver.solver import WFSolver
from qttools.wave_function_solver.superlu import SuperLU


def auto_select_solver(matrix_type: str, matrix_view: str) -> WFSolver:
    """Auto-selects the solver based on the matrix type.

    On GPU, cuDSS is the preferred solver if available. If cuDSS is not
    available, SuperLU is used as a fallback.

    On CPU, PARDISO is the preferred solver if available. If PARDISO is
    not available, MUMPS is used as a fallback. If MUMPS is also not
    available, SuperLU is used as a final fallback.

    If the matrix type is symmetric or Hermitian, only cuDSS on GPU and
    PARDISO on CPU are supported. If these solvers are not available, an
    error is raised.

    Parameters
    ----------
    matrix_type : str
        The type of the matrix.
    matrix_view : str
        The view of the matrix.

    Returns
    -------
    WFSolver
        The selected wavefunction solver instance.

    """
    if xp.__name__ == "cupy":
        if cudss_available:
            if comm.rank == 0:
                print("Auto-selecting cuDSS solver.", flush=True)
            return cuDSS(matrix_type=matrix_type, matrix_view=matrix_view)

        if matrix_type in ["real_symmetric_indefinite", "complex_hermitian_indefinite"]:
            raise ValueError(
                "On GPU, cuDSS is the only general solver that supports symmetric matrices"
            )

        if comm.rank == 0:
            print("Auto-selecting SuperLU solver as fallback.", flush=True)
        return SuperLU(matrix_type=matrix_type, matrix_view=matrix_view)

    if pardiso_available:
        if comm.rank == 0:
            print("Auto-selecting PARDISO solver.", flush=True)
        return PARDISO(matrix_type=matrix_type, matrix_view=matrix_view)

    if matrix_type in ["real_symmetric_indefinite", "complex_hermitian_indefinite"]:
        raise ValueError(
            "On CPU, PARDISO is the only general solver that supports symmetric matrices"
        )

    if mumps_available:
        if comm.rank == 0:
            print("Auto-selecting MUMPS solver as fallback.", flush=True)
        return MUMPS(matrix_type=matrix_type, matrix_view=matrix_view)

    if comm.rank == 0:
        print("Auto-selecting SuperLU solver as fallback.", flush=True)
    return SuperLU(matrix_type=matrix_type, matrix_view=matrix_view)
