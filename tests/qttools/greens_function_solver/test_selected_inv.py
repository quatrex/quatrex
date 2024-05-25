# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import pytest
from scipy import sparse

from qttools.greens_function_solver.inv import Inv
from qttools.greens_function_solver.solver import GFSolver

from qttools.datastructures import DSBCSR
from qttools.datastructures import DSBSparse


@pytest.mark.parametrize("gf_solver", [Inv])
@pytest.mark.parametrize(
    "blocksize, n_blocks",
    [(10, 10)],
)
@pytest.mark.parametrize("dbsparse_type", [DSBCSR])
@pytest.mark.parametrize(
    "stackshape",
    [
        pytest.param((1,), id="no-batching"),
        pytest.param((10,), id="batching"),
    ],
)
@pytest.mark.parametrize("out", [True, False])
def test_selected_inv(
    BT_array: np.ndarray,
    BT_block_sizes: np.ndarray,
    cut_dense_to_BT: callable,
    gf_solver: GFSolver,
    dbsparse_type: DSBSparse,
    blocksize: int,
    n_blocks: int,
    stackshape: tuple[int],
    out: bool,
):
    coo_bt = sparse.coo_matrix(BT_array)

    ref_inv = cut_dense_to_BT(np.linalg.inv(coo_bt.toarray()), blocksize, n_blocks)

    dbsparse = dbsparse_type.from_sparray(coo_bt, BT_block_sizes, stackshape)

    if out:
        test_inv = dbsparse_type.zeros_like(dbsparse)
        gf_solver.selected_inv(dbsparse, out=test_inv)
    else:
        test_inv = gf_solver.selected_inv(dbsparse)

    assert np.allclose(test_inv.to_dense()[(0,) * len(stackshape)], ref_inv)
