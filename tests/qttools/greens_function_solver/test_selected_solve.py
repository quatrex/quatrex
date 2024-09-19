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
@pytest.mark.parametrize("dsbsparse_type", [DSBCSR])
@pytest.mark.parametrize(
    "global_stack_shape",
    [
        pytest.param((10,), id="stack"),
    ],
)
@pytest.mark.parametrize("out", [True, False])
@pytest.mark.parametrize("return_retarded", [True])
@pytest.mark.parametrize(
    "max_batch_size",
    [
        pytest.param(1, id="no_batch"),
        pytest.param(2, id="uniform_batch"),
        pytest.param(3, id="non_uniform_batch"),
        pytest.param(10, id="everything_batch"),
    ],
)
def test_selected_solve(
    BT_array: np.ndarray,
    BT_block_sizes: np.ndarray,
    cut_dense_to_BT: callable,
    gf_solver: GFSolver,
    dsbsparse_type: DSBSparse,
    blocksize: int,
    n_blocks: int,
    global_stack_shape: tuple[int],
    out: bool,
    return_retarded: bool,
    max_batch_size: int,
):
    coo_A = sparse.coo_matrix(BT_array)
    coo_Bl = sparse.coo_matrix(BT_array)
    coo_Bg = sparse.coo_matrix(BT_array)

    # Reference solution of:
    # (1) A * Xr = I
    # (2) A * Xl * A^T = Bl
    # (3) A * Xg * A^T = Bg
    ref_Xr = np.linalg.inv(coo_A.toarray())
    ref_Xl = cut_dense_to_BT(
        ref_Xr @ coo_Bl.toarray() @ ref_Xr.conj().T, blocksize, n_blocks
    )
    ref_Xg = cut_dense_to_BT(
        ref_Xr @ coo_Bg.toarray() @ ref_Xr.conj().T, blocksize, n_blocks
    )

    A = dsbsparse_type.from_sparray(coo_A, BT_block_sizes, global_stack_shape)
    Bl = dsbsparse_type.from_sparray(coo_Bl, BT_block_sizes, global_stack_shape)
    Bg = dsbsparse_type.from_sparray(coo_Bg, BT_block_sizes, global_stack_shape)

    solver = gf_solver()

    if out:
        Xr = dsbsparse_type.zeros_like(A)
        Xl = dsbsparse_type.zeros_like(A)
        Xg = dsbsparse_type.zeros_like(A)

        solver.selected_solve(
            A,
            Bl,
            Bg,
            out=[Xr, Xl, Xg],
            return_retarded=return_retarded,
            max_batch_size=max_batch_size,
        )
    else:
        Xr, Xl, Xg = solver.selected_solve(
            A,
            Bl,
            Bg,
            return_retarded=return_retarded,
            max_batch_size=max_batch_size,
        )

    assert np.allclose(
        np.repeat(ref_Xr[np.newaxis, :, :], global_stack_shape[0], axis=0),
        Xr.to_dense(),
    )

    assert np.allclose(
        np.repeat(ref_Xl[np.newaxis, :, :], global_stack_shape[0], axis=0),
        Xl.to_dense(),
    )

    assert np.allclose(
        np.repeat(ref_Xg[np.newaxis, :, :], global_stack_shape[0], axis=0),
        Xg.to_dense(),
    )
