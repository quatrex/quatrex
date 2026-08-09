# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from pathlib import Path

import pytest

from qttools import NDArray, xp
from qttools.utils.mpi_utils import distributed_load

EXAMPLES_DIR = Path(__file__).parents[3].resolve() / "examples"
MOS2_EXAMPLE = EXAMPLES_DIR / "w90" / "mos2" / "inputs"


@pytest.fixture(scope="session")
def matrix_dict(request: pytest.FixtureRequest) -> dict[str, NDArray]:
    """Returns the wannier tight binding matrix of the mos2 example"""

    matrix_dict = distributed_load(MOS2_EXAMPLE / "hamiltonian.h5")

    matrix_dict = {key: xp.asarray(matrix) for key, matrix in matrix_dict.items()}

    return matrix_dict
