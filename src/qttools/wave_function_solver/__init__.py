# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.

from qttools import sparse
from qttools.wave_function_solver.cudss import cuDSS
from qttools.wave_function_solver.mumps import MUMPS
from qttools.wave_function_solver.solver import WFSolver
from qttools.wave_function_solver.superlu import SuperLU

preferred_matrix_type = {
    "mumps": sparse.coo_matrix,
    "superlu": sparse.csc_matrix,
    "cudss": sparse.csr_matrix,
}

__all__ = ["WFSolver", "SuperLU", "MUMPS", "cuDSS", "preferred_matrix_type"]
