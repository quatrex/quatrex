# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from mpi4py.MPI import COMM_WORLD as comm

from qttools import xp
from qttools.wave_function_solver.cudss import cuDSS, cudss_available
from qttools.wave_function_solver.mumps import MUMPS, mumps_available
from qttools.wave_function_solver.pardiso import PARDISO, pardiso_available
from qttools.wave_function_solver.solver import WFSolver
from qttools.wave_function_solver.superlu import SuperLU


def auto_select_solver(matrix_type: str, view: str) -> WFSolver:
    """Auto-selects the solver based on the matrix type and view.

    Parameters
    ----------
    matrix_type : str
        The type of the matrix.
    view : str
        The view of the matrix.

    Returns
    -------
    WFSolver
        The selected wavefunction solver.

    """
    if xp.__name__ == "cupy":
        if matrix_type in [
            "real_symmetric_indefinite",
            "complex_hermitian_indefinite",
        ]:
            if cudss_available():
                if comm.rank == 0:
                    print("Auto-selecting cuDSS solver.")
                return cuDSS(matrix_type=matrix_type, view=view)
            else:
                raise ValueError(
                    "On GPU, cuDSS is the only general solver that supports symmetric matrices"
                )
        else:
            if cudss_available():
                if comm.rank == 0:
                    print("Auto-selecting cuDSS solver.")
                return cuDSS(matrix_type=matrix_type, view=view)
            else:
                if comm.rank == 0:
                    print("Auto-selecting SuperLU solver as fallback.")
                return SuperLU(matrix_type=matrix_type, view=view)
    else:
        if matrix_type in [
            "real_symmetric_indefinite",
            "complex_hermitian_indefinite",
        ]:
            if pardiso_available():
                if comm.rank == 0:
                    print("Auto-selecting PARDISO solver.")
                return PARDISO(matrix_type=matrix_type, view=view)
            else:
                raise ValueError(
                    "On CPU, PARDISO is the only general solver that supports symmetric matrices"
                )
        else:
            if pardiso_available():
                if comm.rank == 0:
                    print("Auto-selecting PARDISO solver.")
                return PARDISO(matrix_type=matrix_type, view=view)
            elif mumps_available():
                if comm.rank == 0:
                    print("Auto-selecting MUMPS solver as fallback.")
                return MUMPS(matrix_type=matrix_type, view=view)
            else:
                if comm.rank == 0:
                    print("Auto-selecting SuperLU solver as fallback.")
                return SuperLU(matrix_type=matrix_type, view=view)
