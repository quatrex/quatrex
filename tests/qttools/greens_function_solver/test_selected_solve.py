# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

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


def test_selected_solve(
    bt_dense: NDArray,
    gfsolver_type: GFSolver,
    dsdbsparse_type: DSDBSparse,
    return_retarded: bool,
    max_batch_size: int,
    block_sizes: NDArray,
    global_stack_shape: tuple,
):
    """Tests the selected solve method of a Green's function solver."""
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

    A = dsdbsparse_type.from_sparray(
        sparray=coo_A, block_sizes=block_sizes, global_stack_shape=global_stack_shape
    )
    Bl = dsdbsparse_type.from_sparray(
        sparray=coo_Bl, block_sizes=block_sizes, global_stack_shape=global_stack_shape
    )
    Bg = dsdbsparse_type.from_sparray(
        sparray=coo_Bg, block_sizes=block_sizes, global_stack_shape=global_stack_shape
    )

    solver = gfsolver_type(max_batch_size=max_batch_size)

    Xr = dsdbsparse_type.empty_like(A)
    Xl = dsdbsparse_type.empty_like(A)
    Xg = dsdbsparse_type.empty_like(A)
    Xr.allocate_data()
    Xl.allocate_data()
    Xg.allocate_data()

    # NOTE: This is set to zero since later the non-zero values are used
    # for the masked. Otherwise, the off-diagonal values would be tested
    # as well while they are not computed.
    Xr.data = 0.0

    solver.selected_solve(
        A,
        Bl,
        Bg,
        out=[Xl, Xg, Xr],
        return_retarded=return_retarded,
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
