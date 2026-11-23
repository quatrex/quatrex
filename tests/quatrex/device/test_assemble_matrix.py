# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Tests the assembly of the device matrices at transverse k-points."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse as sps

from qttools import sparse
from qttools.comm import comm
from qttools.comm.comm import _default_config
from qttools.datastructures import DSDBCOO
from qttools.utils.gpu_utils import get_host
from quatrex.device import inputs
from quatrex.grid.kpoints import monkhorst_pack

BLOCK_SIZE = 4
NUM_BLOCKS = 3


@pytest.fixture(autouse=True, scope="module")
def configure_comm():
    """Configures the communicator for the tests."""
    comm.configure(
        block_comm_size=1,
        block_comm_config=_default_config.copy(),
        stack_comm_config=_default_config.copy(),
        global_comm_config=_default_config.copy(),
        override=True,
    )


def _images() -> dict[tuple, np.ndarray]:
    """Creates a Hermitian operator with images that cancel at Gamma.

    Like an s-p overlap between atoms half a period apart, one element
    has opposite signs in the images 0 and -1.

    """
    n = NUM_BLOCKS * BLOCK_SIZE
    home = 2.0 * np.eye(n) + 0.1 * (np.eye(n, k=1) + np.eye(n, k=-1))
    image = np.zeros((n, n))
    image[1, 0] = 0.2

    home[0, 2] = home[2, 0] = 0.3
    image[2, 0] = -0.3

    return {(0, 0, 0): home, (0, 1, 0): image, (0, -1, 0): image.T}


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("num_kpoints", [1, 3, 4])
def test_assemble_matrix_with_cancelations(
    monkeypatch: pytest.MonkeyPatch, symmetric: bool, num_kpoints: int
):
    """Tests that elements that cancel at Gamma are kept at k-points."""
    images = _images()
    # NOTE: `load_matrices` only keeps the upper triangular parts.
    matrix_dict = {
        coord: sparse.csr_matrix(sps.csr_matrix(np.triu(matrix), dtype=np.complex128))
        for coord, matrix in images.items()
    }
    monkeypatch.setattr(inputs, "load_matrices", lambda *__, **___: matrix_dict)
    config = SimpleNamespace(
        device=SimpleNamespace(
            construct_from_unit_cell=False,
            num_spin_channels=1,
            block_size=BLOCK_SIZE,
            kpoint_grid=(1, num_kpoints, 1),
            kpoint_shift=(0.0, 0.0, 0.0),
        ),
        scba=SimpleNamespace(symmetric=symmetric),
        compute=SimpleNamespace(dsdbsparse_type=DSDBCOO),
    )

    matrix, __ = inputs.assemble_matrix(config, "overlap")

    kpoints = monkhorst_pack((1, num_kpoints, 1), (0.0, 0.0, 0.0))
    for i in range(NUM_BLOCKS):
        for j in range(max(0, i - 1), min(NUM_BLOCKS, i + 2)):
            block = get_host(matrix.blocks[i, j]).reshape(-1, BLOCK_SIZE, BLOCK_SIZE)
            for ik, kpoint in enumerate(kpoints):
                reference = sum(
                    np.exp(2j * np.pi * np.dot(coord, kpoint)) * image
                    for coord, image in images.items()
                )[
                    i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                    j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
                ]
                assert np.allclose(block[ik], reference)
