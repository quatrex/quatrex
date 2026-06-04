# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
import importlib.util
from dataclasses import dataclass, field

import pytest

from qttools import xp
from qttools.wave_function_solver import (
    MUMPS,
    PARDISO,
    SuperLU,
    Thomas,
    WFSolver,
    cuDSS,
)

NUM_ROWS = [
    pytest.param(27, id="27-rows"),
    pytest.param(53, id="53-rows"),
]
NUM_RHS = [
    pytest.param(1, id="1-rhs"),
    pytest.param(12, id="12-rhs"),
]


@pytest.fixture(params=NUM_ROWS)
def n(request: pytest.FixtureRequest) -> int:
    """Fixture to provide the number of rows for the sparse matrix."""
    return request.param


@pytest.fixture(params=NUM_RHS)
def m(request: pytest.FixtureRequest) -> int:
    """Fixture to provide the number of right-hand sides for the solver."""
    return request.param


mumps_available = importlib.util.find_spec("mumps") is not None
nvmath_available = importlib.util.find_spec("nvmath") is not None
pardiso_available = importlib.util.find_spec("pydiso") is not None


@dataclass(frozen=True)
class WFSolverSpec:
    solver_type: type[WFSolver]
    sparse_format: str
    order: str = "C"
    use_banded: bool = False
    supports_reuse_analysis: bool = False
    supports_reuse_factorization: bool = False
    factorization_needs_analysis: bool = False
    supports_symmetric: bool = False
    supports_hermitian: bool = False
    solve_kwargs: dict[str, object] = field(default_factory=dict)


SOLVER_SPECS = [
    pytest.param(
        WFSolverSpec(
            solver_type=SuperLU,
            sparse_format="csc",
            supports_reuse_factorization=True,
        ),
        id="superlu",
    ),
    pytest.param(
        WFSolverSpec(
            solver_type=MUMPS,
            sparse_format="coo",
            supports_reuse_analysis=True,
            supports_reuse_factorization=True,
            factorization_needs_analysis=True,
        ),
        id="mumps",
        marks=[
            pytest.mark.skipif(
                not mumps_available, reason="Requires python-mumps package"
            ),
            pytest.mark.skipif(xp.__name__ != "numpy", reason="Requires numpy backend"),
        ],
    ),
    pytest.param(
        WFSolverSpec(
            solver_type=PARDISO,
            sparse_format="csr",
            supports_reuse_analysis=True,
            supports_reuse_factorization=True,
            factorization_needs_analysis=True,
            supports_symmetric=True,
            supports_hermitian=True,
        ),
        id="pardiso",
        marks=[
            pytest.mark.skipif(not pardiso_available, reason="Requires pydiso package"),
            pytest.mark.skipif(xp.__name__ != "numpy", reason="Requires numpy backend"),
        ],
    ),
    pytest.param(
        WFSolverSpec(
            solver_type=cuDSS,
            sparse_format="csr",
            order="F",
            supports_reuse_analysis=True,
            supports_reuse_factorization=True,
            factorization_needs_analysis=True,
            supports_symmetric=True,
            supports_hermitian=True,
        ),
        id="cudss",
        marks=[
            pytest.mark.skipif(
                not nvmath_available, reason="Requires nvmath-python package"
            ),
            pytest.mark.skipif(xp.__name__ != "cupy", reason="Requires cupy backend"),
        ],
    ),
    pytest.param(
        WFSolverSpec(
            solver_type=Thomas,
            sparse_format="csr",
            use_banded=True,
            supports_reuse_analysis=True,
            supports_symmetric=True,
            supports_hermitian=True,
            solve_kwargs={"overwrite_b": False},
        ),
        id="thomas",
    ),
]


@pytest.fixture(params=SOLVER_SPECS)
def solver_spec(request: pytest.FixtureRequest) -> WFSolverSpec:
    return request.param
