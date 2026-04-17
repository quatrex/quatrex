# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from pathlib import Path

import numpy as np
import pytest

from qttools import NDArray, xp
from qttools.utils.mpi_utils import distributed_load

EXAMPLES_DIR = Path(__file__).parents[3].resolve() / "examples"
MOS2_EXAMPLE = EXAMPLES_DIR / "w90" / "mos2" / "inputs"


@pytest.fixture(scope="session")
def unit_cells(request: pytest.FixtureRequest) -> NDArray:
    """Returns the wannier tight binding matrix of the mos2 example"""

    matrices = distributed_load(MOS2_EXAMPLE / "hamiltonian.mat")

    keys = np.array(list(matrices.keys()))

    min_coords = keys.min(axis=0)
    max_coords = keys.max(axis=0)
    grid_shape = max_coords - min_coords + 1

    expected_size = np.prod(grid_shape)
    actual_size = len(matrices)
    if expected_size != actual_size:
        raise ValueError(
            f"Expected {expected_size} unit cells based on the detected grid shape, "
            f"but found {actual_size} unit cells in the matrix file."
        )

    first_matrix = next(iter(matrices.values()))
    matrix_shape = first_matrix.shape
    unit_cells = xp.zeros(
        tuple(grid_shape) + tuple(matrix_shape), dtype=first_matrix.dtype
    )

    for coord, matrix in matrices.items():
        unit_cells[coord] = xp.asarray(matrix).astype(xp.complex128)

    return unit_cells
