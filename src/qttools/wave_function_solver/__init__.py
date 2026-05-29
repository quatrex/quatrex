# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from qttools import sparse
from qttools.wave_function_solver.auto_select import auto_select_solver
from qttools.wave_function_solver.cudss import cuDSS
from qttools.wave_function_solver.mumps import MUMPS
from qttools.wave_function_solver.pardiso import PARDISO
from qttools.wave_function_solver.solver import WFSolver
from qttools.wave_function_solver.superlu import SuperLU
from qttools.wave_function_solver.thomas import Thomas

preferred_matrix_type = {
    "mumps": sparse.coo_matrix,
    "superlu": sparse.csc_matrix,
    "cudss": sparse.csr_matrix,
    "pardiso": sparse.csr_matrix,
    "thomas": sparse.csr_matrix,
    "auto": sparse.csr_matrix,
}

__all__ = [
    "WFSolver",
    "SuperLU",
    "MUMPS",
    "cuDSS",
    "PARDISO",
    "Thomas",
    "preferred_matrix_type",
    "auto_select_solver",
]
