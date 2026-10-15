# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes method to auto-select the best wave function solver."""

from typing import Any

from qttools import NDArray, xp
from qttools.comm import comm as quatrex_comm
from qttools.wave_function_solver.cudss import cuDSS, cudss_available
from qttools.wave_function_solver.petsc import PETSc, petsc_available
from qttools.wave_function_solver.solver import WFSolver
from qttools.wave_function_solver.superlu import SuperLU
from qttools.wave_function_solver.thomas import Thomas


def _select_distributed_solver(
    inputs: dict[str, Any],
    optional_inputs: dict[str, Any] = {},
) -> WFSolver:
    """Selects the best wave function solver for distributed execution.

    Parameters
    ----------
    inputs : dict
        dictionary of input parameters for the solver.
    optional_inputs : dict
        dictionary of optional input parameters for the solver.

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
            **inputs,
        )

    if quatrex_comm.rank == 0:
        print("Auto-selecting PETSc solver for distributed execution.", flush=True)

    if "petsc_options" not in optional_inputs:
        raise ValueError(
            "PETSc options must be provided for distributed execution with PETSc."
        )

    return PETSc(**inputs, petsc_options=optional_inputs["petsc_options"])


def _select_non_distributed_solver(
    inputs: dict[str, Any],
    optional_inputs: dict[str, Any],
) -> WFSolver:
    """Selects the best wave function solver for non-distributed
    execution.

    Parameters
    ----------
    inputs : dict
        dictionary of input parameters for the solver.
    optional_inputs : dict
        dictionary of optional input parameters for the solver.

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
                **inputs,
            )

        if quatrex_comm.rank == 0:
            print("Auto-selecting SuperLU solver as fallback.", flush=True)
        return SuperLU(
            **inputs,
        )

    if quatrex_comm.rank == 0:
        print("Auto-selecting SuperLU solver as fallback.", flush=True)
    return SuperLU(
        **inputs,
    )


def _auto_select_solver(
    inputs: dict[str, Any],
    optional_inputs: dict[str, Any],
) -> WFSolver:
    """Auto-selects the solver based on the matrix type.

    On GPU, cuDSS is the preferred solver if available. If cuDSS is not
    available, SuperLU is used as a fallback.

    In the distributed case, cuDSS is preferred on GPU and PETSc is used
    on CPU. If neither is available, an error is raised.

    Parameters
    ----------
    inputs : dict
         dictionary of input parameters for the solver.
    optional_inputs : dict
        dictionary of optional input parameters for the solver.

    Returns
    -------
    WFSolver
        The selected wavefunction solver class.

    """
    if "comm" in inputs and inputs["comm"] is not None and inputs["comm"].size > 1:
        return _select_distributed_solver(
            inputs,
            optional_inputs,
        )

    return _select_non_distributed_solver(
        inputs,
        optional_inputs,
    )


def configure_solver(
    solver_name: str,
    matrix_dtype: xp.dtype[xp.inexact],
    matrix_property: str,
    matrix_view: str,
    row_offsets: NDArray,
    petsc_options: dict[str, Any] | None = None,
) -> WFSolver:
    """Configures the wavefunction solver based on the config.

    Parameters
    ----------
    solver_name : str
        The name of the solver to use for solving the linear system.
    matrix_dtype : xp.dtype[xp.inexact]
        The data type of the system matrix.
    matrix_property : str
        The type of the system matrix, describing properties like
        symmetry and definiteness.
    matrix_view : str
        The view of the system matrix sparsity, indicating which
        part of the matrix to use for symmetric matrices.
    row_offsets : NDArray
        The row offsets partitioning the system matrix across MPI
        ranks.
    petsc_options : dict[str, Any], optional
        Dictionary of PETSc options to configure the solver.

    Returns
    -------
    WFSolver
        The configured wavefunction solver instance.

    """

    local_rows = (
        row_offsets[quatrex_comm.block.rank],
        row_offsets[quatrex_comm.block.rank + 1],
    )

    distributed = {
        "cudss": cuDSS,
        "petsc": PETSc,
    }
    serial = {
        "superlu": SuperLU,
        "thomas": Thomas,
    }

    inputs: dict[str, Any] = {
        "matrix_dtype": matrix_dtype,
        "matrix_property": matrix_property,
        "matrix_view": matrix_view,
    }

    inputs["reuse_analysis"] = True
    inputs["reuse_factorization"] = False
    inputs["comm"] = quatrex_comm.block
    inputs["local_rows"] = local_rows

    optional_inputs: dict[str, Any] = {}

    if solver_name in ["petsc", "auto"]:
        if petsc_options is None:
            raise ValueError("PETSc options must be provided for PETSc solver.")
        optional_inputs["petsc_options"] = petsc_options

    if solver_name in serial:
        return serial[solver_name](
            **inputs,
            **optional_inputs,
        )

    if solver_name in distributed:
        return distributed[solver_name](
            **inputs,
            **optional_inputs,
        )

    if solver_name == "auto":
        return _auto_select_solver(
            inputs,
            optional_inputs,
        )

    raise ValueError(f"Unknown direct solver: {solver_name}")
