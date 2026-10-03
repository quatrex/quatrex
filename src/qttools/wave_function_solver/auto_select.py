# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes method to auto-select the best wave function solver."""

from qttools import xp
from qttools.comm import comm as quatrex_comm
from qttools.comm.comm import _SubCommunicator
from qttools.wave_function_solver.cudss import cuDSS, cudss_available
from qttools.wave_function_solver.mumps import MUMPS, mumps_available
from qttools.wave_function_solver.pardiso import PARDISO, pardiso_available
from qttools.wave_function_solver.petsc import PETSc, petsc_available
from qttools.wave_function_solver.solver import WFSolver
from qttools.wave_function_solver.superlu import SuperLU


def _select_distributed_solver(
    matrix_type: str,
    matrix_view: str,
    comm: _SubCommunicator,
    local_rows: tuple,
    **kwargs,
) -> WFSolver:
    """Selects the best wave function solver for distributed execution.

    Parameters
    ----------
    matrix_type : str
        The type of the matrix.
    matrix_view : str
        The view of the matrix.
    comm : _SubCommunicator
        The communicator for parallel execution.
    local_rows : tuple
        The range of local rows for the current process.
    kwargs : dict
        Additional keyword arguments to pass to the solver constructor.

    Returns
    -------
    WFSolver
        The selected wavefunction solver instance.

    """
    if not (cudss_available or petsc_available):
        raise ValueError("Parallel execution requires cuDSS or PETSc to be available.")

    if xp.__name__ == "cupy" and cudss_available:
        if quatrex_comm.rank == 0:
            print("Auto-selecting cuDSS solver for distributed execution.", flush=True)
        return cuDSS(
            matrix_type=matrix_type,
            matrix_view=matrix_view,
            comm=comm,
            local_rows=local_rows,
        )

    if quatrex_comm.rank == 0:
        print("Auto-selecting PETSc solver for distributed execution.", flush=True)

    if matrix_type in ["real_symmetric_indefinite", "complex_hermitian_indefinite"]:
        raise ValueError("PETSc does only support general matrices.")

    return PETSc(
        matrix_type=matrix_type,
        matrix_view=matrix_view,
        comm=comm,
        local_rows=local_rows,
        petsc_options=kwargs.get("petsc_options", {}),
    )


def _select_non_distributed_solver(
    matrix_type: str,
    matrix_view: str,
) -> WFSolver:
    """Selects the best wave function solver for non-distributed
    execution.

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
            if quatrex_comm.rank == 0:
                print("Auto-selecting cuDSS solver.", flush=True)
            return cuDSS(
                matrix_type=matrix_type,
                matrix_view=matrix_view,
            )

        if matrix_type in ["real_symmetric_indefinite", "complex_hermitian_indefinite"]:
            raise ValueError(
                "On GPU, cuDSS is the only general solver that supports symmetric matrices"
            )

        if quatrex_comm.rank == 0:
            print("Auto-selecting SuperLU solver as fallback.", flush=True)
        return SuperLU(matrix_type=matrix_type, matrix_view=matrix_view)

    if pardiso_available:
        if quatrex_comm.rank == 0:
            print("Auto-selecting PARDISO solver.", flush=True)
        return PARDISO(matrix_type=matrix_type, matrix_view=matrix_view)

    if matrix_type in ["real_symmetric_indefinite", "complex_hermitian_indefinite"]:
        raise ValueError(
            "On CPU, PARDISO is the only general solver that supports symmetric matrices"
        )

    if mumps_available:
        if quatrex_comm.rank == 0:
            print("Auto-selecting MUMPS solver as fallback.", flush=True)
        return MUMPS(matrix_type=matrix_type, matrix_view=matrix_view)

    if quatrex_comm.rank == 0:
        print("Auto-selecting SuperLU solver as fallback.", flush=True)
    return SuperLU(matrix_type=matrix_type, matrix_view=matrix_view)


def auto_select_solver(
    matrix_type: str,
    matrix_view: str,
    comm: _SubCommunicator,
    local_rows: tuple,
    **kwargs,
) -> WFSolver:
    """Auto-selects the solver based on the matrix type.

    On GPU, cuDSS is the preferred solver if available. If cuDSS is not
    available, SuperLU is used as a fallback.

    On CPU, PARDISO is the preferred solver if available. If PARDISO is
    not available, MUMPS is used as a fallback. If MUMPS is also not
    available, SuperLU is used as a final fallback.

    If the matrix type is symmetric or Hermitian, only cuDSS on GPU and
    PARDISO on CPU are supported. If these solvers are not available, an
    error is raised.

    In the distributed case, cuDSS is preferred on GPU and PETSc is used
    on CPU. If neither is available, an error is raised.

    Parameters
    ----------
    matrix_type : str
        The type of the matrix.
    matrix_view : str
        The view of the matrix.
    comm : _SubCommunicator
        The communicator for parallel execution.
    local_rows : tuple
        The range of local rows for the current process.
    kwargs : dict
        Additional keyword arguments to pass to the solver constructor.

    Returns
    -------
    WFSolver
        The selected wavefunction solver instance.

    """
    if comm.size > 1:
        return _select_distributed_solver(
            matrix_type=matrix_type,
            matrix_view=matrix_view,
            comm=comm,
            local_rows=local_rows,
            **kwargs,
        )

    return _select_non_distributed_solver(
        matrix_type=matrix_type,
        matrix_view=matrix_view,
    )
