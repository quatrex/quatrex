# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from numpy.typing import ArrayLike

from qttools import sparse, xp
from qttools.datastructures import DSBSparse
from qttools.greens_function_solver import GFSolver


def test_selected_solve(
    bt_dense: ArrayLike,
    gfsolver_type: GFSolver,
    dsbsparse_type: DSBSparse,
    out: bool,
    return_retarded: bool,
    max_batch_size: int,
    block_sizes: ArrayLike,
    global_stack_shape: int | tuple,
):
    coo_A = sparse.coo_matrix(bt_dense)

    coo_Bl = sparse.coo_matrix(bt_dense)
    coo_Bl += -coo_Bl.conj().T

    coo_Bg = sparse.coo_matrix(bt_dense)
    coo_Bg += -coo_Bg.conj().T

    # Reference solution of:
    # (1) A * Xr = I
    ref_Xr = xp.linalg.inv(bt_dense)

    # (2) A * Xl * A^T = Bl
    ref_Xl = ref_Xr @ xp.asarray(coo_Bl.toarray()) @ ref_Xr.conj().T

    # (3) A * Xg * A^T = Bg
    ref_Xg = ref_Xr @ xp.asarray(coo_Bg.toarray()) @ ref_Xr.conj().T

    block_sizes = block_sizes
    densify_blocks = [(i, i) for i in range(len(block_sizes))]

    A = dsbsparse_type.from_sparray(
        coo_A, block_sizes, global_stack_shape, densify_blocks
    )
    Bl = dsbsparse_type.from_sparray(
        coo_Bl, block_sizes, global_stack_shape, densify_blocks
    )
    Bg = dsbsparse_type.from_sparray(
        coo_Bg, block_sizes, global_stack_shape, densify_blocks
    )

    solver = gfsolver_type()

    if out:
        Xr = dsbsparse_type.zeros_like(A)
        Xl = dsbsparse_type.zeros_like(A)
        Xg = dsbsparse_type.zeros_like(A)

        solver.selected_solve(
            A,
            Bl,
            Bg,
            out=[Xl, Xg, Xr],
            return_retarded=return_retarded,
            max_batch_size=max_batch_size,
        )
    else:
        if return_retarded:
            Xl, Xg, Xr = solver.selected_solve(
                A,
                Bl,
                Bg,
                return_retarded=return_retarded,
                max_batch_size=max_batch_size,
            )
        else:
            Xl, Xg = solver.selected_solve(
                A,
                Bl,
                Bg,
                return_retarded=return_retarded,
                max_batch_size=max_batch_size,
            )

    if return_retarded:
        xr_mask = Xr.to_dense().astype(bool)
        assert xp.allclose(
            xp.broadcast_to(ref_Xr, (*global_stack_shape, *ref_Xr.shape)) * xr_mask,
            Xr.to_dense() * xr_mask,
        )

    xl_mask = Xl.to_dense().astype(bool)
    assert xp.allclose(
        xp.broadcast_to(ref_Xl, (*global_stack_shape, *ref_Xl.shape)) * xl_mask,
        Xl.to_dense() * xl_mask,
    )

    xg_mask = Xg.to_dense().astype(bool)
    assert xp.allclose(
        xp.broadcast_to(ref_Xg, (*global_stack_shape, *ref_Xg.shape)) * xg_mask,
        Xg.to_dense() * xg_mask,
    )
