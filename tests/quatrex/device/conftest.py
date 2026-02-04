# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from pathlib import Path

import pytest

from qttools import NDArray, xp

EXAMPLES_DIR = Path(__file__).parents[3].resolve() / "examples"
MOS2_EXAMPLE = EXAMPLES_DIR / "w90" / "mos2" / "inputs"


@pytest.fixture(autouse=True, scope="session")
def unit_cells(request: pytest.FixtureRequest) -> NDArray:
    """Returns the wannier tight binding matrix of the mos2 example"""

    input_path = MOS2_EXAMPLE / "hamiltonian_unit_cells.npy"
    unit_cells = xp.load(input_path)

    return unit_cells
