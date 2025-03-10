# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import time

from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBSparse, DSBCOO
from qttools.greens_function_solver import GFSolver, RGF


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

    print(f"{max_batch_size=}, {block_sizes=}, {global_stack_shape=}", flush=True)

    original = []
    new = []
    for _ in range(10):
        if out:
            gf_inv = dsbsparse_type.zeros_like(dsbsparse)
            if xp.__name__ == "cupy":
                xp.cuda.get_current_stream().synchronize()
            start = time.perf_counter()
            solver.selected_inv(dsbsparse, out=gf_inv)
        else:
            if xp.__name__ == "cupy":
                xp.cuda.get_current_stream().synchronize()
            start = time.perf_counter()
            gf_inv = solver.selected_inv(dsbsparse)
        if xp.__name__ == "cupy":
            xp.cuda.get_current_stream().synchronize()
        end = time.perf_counter()
        original.append(end - start)


        if dsbsparse_type is DSBCOO and gfsolver_type is RGF:
            if out:
                gf_inv = dsbsparse_type.zeros_like(dsbsparse)
                if xp.__name__ == "cupy":
                    xp.cuda.get_current_stream().synchronize()
                start = time.perf_counter()
                solver.selected_inv_new(dsbsparse, out=gf_inv)
            else:
                if xp.__name__ == "cupy":
                    xp.cuda.get_current_stream().synchronize()
                start = time.perf_counter()
                gf_inv = solver.selected_inv_new(dsbsparse)
            if xp.__name__ == "cupy":
                xp.cuda.get_current_stream().synchronize()
            end = time.perf_counter()
            new.append(end - start)

    original = xp.median(xp.array(original))
    if len(new) > 0:
        new = xp.median(xp.array(new))
        print(f"Original: {original:.6f} s, New: {new:.6f} s", flush=True)
    else:
        print(f"Original: {original:.6f} s", flush=True)

    bt_mask_broadcasted = xp.broadcast_to(
        bt_mask, (*global_stack_shape, *bt_mask.shape)
    )

    assert xp.allclose(
        xp.broadcast_to(ref_inv, (*global_stack_shape, *ref_inv.shape)),
        gf_inv.to_dense() * bt_mask_broadcasted,
    )


if __name__ == "__main__":
    import pytest
    pytest.main(["-s", __file__])
