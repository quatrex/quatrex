# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.greens_function_solver import GFSolver


@pytest.fixture(autouse=True, scope="module")
def configure_comm():
    """setup any state specific to the execution of the given module."""
    if xp.__name__ == "cupy":
        _default_config = {
            "all_to_all": "host_mpi",
            "all_gather": "host_mpi",
            "all_reduce": "host_mpi",
            "bcast": "host_mpi",
        }
    elif xp.__name__ == "numpy":
        _default_config = {
            "all_to_all": "device_mpi",
            "all_gather": "device_mpi",
            "all_reduce": "device_mpi",
            "bcast": "device_mpi",
        }
    # Configure the comm singleton.
    comm.configure(
        block_comm_size=1,
        block_comm_config=_default_config,
        stack_comm_config=_default_config,
        override=True,
    )


def test_selected_inv(
    bt_dense: NDArray,
    gfsolver_type: GFSolver,
    dsdbsparse_type: DSDBSparse,
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
    dsdbsparse = dsdbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)

    solver = gfsolver_type(max_batch_size=max_batch_size)

    if out:
        gf_inv = dsdbsparse_type.zeros_like(dsdbsparse)
        solver.selected_inv(dsdbsparse, out=gf_inv)
    else:
        gf_inv = solver.selected_inv(dsdbsparse)

    bt_mask_broadcasted = xp.broadcast_to(
        bt_mask, (*global_stack_shape, *bt_mask.shape)
    )

    assert xp.allclose(
        xp.broadcast_to(ref_inv, (*global_stack_shape, *ref_inv.shape)),
        gf_inv.to_dense() * bt_mask_broadcasted,
    )
