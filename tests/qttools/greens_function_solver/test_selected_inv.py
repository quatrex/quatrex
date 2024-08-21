# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import pytest
from scipy import sparse

from qttools.datastructures import DSBCSR, DSBSparse
from qttools.greens_function_solver.inv import Inv
from qttools.greens_function_solver.solver import GFSolver


@pytest.mark.parametrize("gf_solver", [Inv])
@pytest.mark.parametrize(
    "blocksize, n_blocks",
    [(10, 10)],
)
@pytest.mark.parametrize("dbsparse_type", [DSBCSR])
@pytest.mark.parametrize(
    "global_stack_shape",
    [
        pytest.param((10,), id="stack"),
    ],
)
@pytest.mark.parametrize("out", [True, False])
@pytest.mark.parametrize(
    "max_batch_size",
    [
        pytest.param(1, id="no-batch"),
        pytest.param(2, id="uniform-batch"),
        pytest.param(3, id="non-uniform-batch"),
        pytest.param(10, id="everything-batch"),
    ],
)
def test_selected_inv(
    BT_array: np.ndarray,
    BT_block_sizes: np.ndarray,
    cut_dense_to_BT: callable,
    gf_solver: GFSolver,
    dbsparse_type: DSBSparse,
    blocksize: int,
    n_blocks: int,
    global_stack_shape: tuple[int],
    out: bool,
    max_batch_size: int,
):
    coo_bt = sparse.coo_matrix(BT_array)

    ref_inv = cut_dense_to_BT(np.linalg.inv(coo_bt.toarray()), blocksize, n_blocks)

    dbsparse = dbsparse_type.from_sparray(coo_bt, BT_block_sizes, global_stack_shape)

    solver = gf_solver()

    if out:
        gf_inv = dbsparse_type.zeros_like(dbsparse)
        solver.selected_inv(dbsparse, out=gf_inv, max_batch_size=max_batch_size)
    else:
        gf_inv = solver.selected_inv(dbsparse, max_batch_size=max_batch_size)

    assert np.allclose(
        np.repeat(ref_inv[np.newaxis, :, :], global_stack_shape[0], axis=0),
        gf_inv.to_dense(),
    )
