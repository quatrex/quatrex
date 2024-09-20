# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from numpy.typing import ArrayLike
from scipy import sparse

from qttools.datastructures import DSBSparse
from qttools.greens_function_solver import GFSolver
from qttools.utils.gpu_utils import get_host, xp


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
    bt_mask = bt_dense.astype(bool)

    coo_A = sparse.coo_matrix(get_host(bt_dense))
    coo_Bl = sparse.coo_matrix(get_host(bt_dense))
    coo_Bg = sparse.coo_matrix(get_host(bt_dense))

    # Reference solution of:
    # (1) A * Xr = I
    ref_Xr = xp.linalg.inv(bt_dense)

    # (2) A * Xl * A^T = Bl
    ref_Xl = (ref_Xr @ xp.asarray(coo_Bl.toarray()) @ ref_Xr.conj().T) * bt_mask

    # (3) A * Xg * A^T = Bg
    ref_Xg = (ref_Xr @ xp.asarray(coo_Bg.toarray()) @ ref_Xr.conj().T) * bt_mask

    ref_Xr = ref_Xr * bt_mask

    A = dsbsparse_type.from_sparray(coo_A, block_sizes, global_stack_shape)
    Bl = dsbsparse_type.from_sparray(coo_Bl, block_sizes, global_stack_shape)
    Bg = dsbsparse_type.from_sparray(coo_Bg, block_sizes, global_stack_shape)

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
        assert xp.allclose(
            xp.broadcast_to(ref_Xr, (*global_stack_shape, *ref_Xr.shape)),
            Xr.to_dense(),
        )

    assert xp.allclose(
        xp.broadcast_to(ref_Xl, (*global_stack_shape, *ref_Xl.shape)),
        Xl.to_dense(),
    )

    assert xp.allclose(
        xp.broadcast_to(ref_Xg, (*global_stack_shape, *ref_Xg.shape)),
        Xg.to_dense(),
    )
