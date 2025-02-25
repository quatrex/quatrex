# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBSparse
from qttools.greens_function_solver import GFSolver


def test_selected_inv(
    bt_dense: NDArray,
    gfsolver_type: GFSolver,
    dsbsparse_type: DSBSparse,
    out: bool,
    max_batch_size: int,
    block_sizes: NDArray,
    global_stack_shape: int | tuple,
):
    """Tests the selected inversion method of a Green's function solver."""
    bt_mask = bt_dense.astype(bool)
    ref_inv = xp.linalg.inv(bt_dense) * bt_mask

    coo = sparse.coo_matrix(bt_dense)

    block_sizes = block_sizes
    dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)

    solver = gfsolver_type(max_batch_size=max_batch_size)

    def sinv(input: DSBSparse, out: DSBSparse = None) -> DSBSparse:
        return solver.selected_inv(input, out=out)

    if out:
        gf_inv = dsbsparse_type.zeros_like(dsbsparse)
        # solver.selected_inv(dsbsparse, out=gf_inv)
        sinv(dsbsparse, gf_inv)
    else:
        # gf_inv = solver.selected_inv(dsbsparse)
        gf_inv = sinv(dsbsparse)

    bt_mask_broadcasted = xp.broadcast_to(
        bt_mask, (*global_stack_shape, *bt_mask.shape)
    )

    assert xp.allclose(
        xp.broadcast_to(ref_inv, (*global_stack_shape, *ref_inv.shape)),
        gf_inv.to_dense() * bt_mask_broadcasted,
    )
