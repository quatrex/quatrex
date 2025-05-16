# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBCOO
from qttools.greens_function_solver.rgf_dist import RGFDist

BLOCK_SIZES = [
    pytest.param(np.array([10] * 5), id="constant-block-size"),
    pytest.param(np.array([5] * 3 + [10] * 2 + [5] * 3), id="mixed-block-size"),
]
GLOBAL_STACK_SHAPES = [
    pytest.param((4,), id="1D-stack"),
    pytest.param((5, 2), id="2D-stack"),
]


def setup_module():
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
        block_comm_size=3,
        block_comm_config=_default_config,
        stack_comm_config=_default_config,
        override=True,
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("block_sizes", BLOCK_SIZES)
@pytest.mark.parametrize("global_stack_shape", GLOBAL_STACK_SHAPES)
def test_rgf_dist(block_sizes: NDArray, global_stack_shape: tuple):
    """Tests the selected solve method of a Green's function solver."""

    global_block_sizes = np.tile(block_sizes, comm.block.size)
    shape = (int(global_block_sizes.sum()), int(global_block_sizes.sum()))

    a_sparray = None
    sigma_lesser_sparray = None
    sigma_greater_sparray = None
    if global_comm.rank == 0:
        a_sparray = sparse.diags(
            xp.random.rand(11), xp.arange(-5, 6), shape=shape
        ).tocsr()
        a_sparray.data = xp.random.rand(len(a_sparray.data))
        a_sparray += sparse.diags(
            xp.random.rand(11) * 1j, xp.arange(-5, 6), shape=shape
        ).tocsr()
        a_sparray += sparse.diags(
            [20 * (1 + 1j) * xp.random.rand() * 1j], [0], shape=shape
        ).tocsr()

        sigma_lesser_sparray = sparse.diags(
            xp.random.rand(11), xp.arange(-5, 6), shape=shape
        ).tocsr()
        sigma_lesser_sparray.data = xp.random.rand(len(sigma_lesser_sparray.data))
        sigma_lesser_sparray += sparse.diags(
            xp.random.rand(11) * 1j, xp.arange(-5, 6), shape=shape
        ).tocsr()
        sigma_lesser_sparray = sigma_lesser_sparray - sigma_lesser_sparray.conj().T

        sigma_greater_sparray = sparse.diags(
            xp.random.rand(11), xp.arange(-5, 6), shape=shape
        ).tocsr()
        sigma_greater_sparray.data = xp.random.rand(len(sigma_greater_sparray.data))
        sigma_greater_sparray += sparse.diags(
            xp.random.rand(11) * 1j, xp.arange(-5, 6), shape=shape
        ).tocsr()
        sigma_greater_sparray = sigma_greater_sparray - sigma_greater_sparray.conj().T

    a_sparray = global_comm.bcast(a_sparray, root=0)
    sigma_lesser_sparray = global_comm.bcast(sigma_lesser_sparray, root=0)
    sigma_greater_sparray = global_comm.bcast(sigma_greater_sparray, root=0)

    a_sparray = a_sparray.tocoo()
    a_sparray.sum_duplicates()
    a_sparray.eliminate_zeros()
    a_dsdbcoo = DSDBCOO.from_sparray(a_sparray, global_block_sizes, global_stack_shape)

    sigma_lesser_sparray = sigma_lesser_sparray.tocoo()
    sigma_lesser_sparray.sum_duplicates()
    sigma_lesser_sparray.eliminate_zeros()
    sigma_lesser_dsdbcoo = DSDBCOO.from_sparray(
        sigma_lesser_sparray, global_block_sizes, global_stack_shape
    )

    sigma_greater_sparray = sigma_greater_sparray.tocoo()
    sigma_greater_sparray.sum_duplicates()
    sigma_greater_sparray.eliminate_zeros()
    sigma_greater_dsdbcoo = DSDBCOO.from_sparray(
        sigma_greater_sparray, global_block_sizes, global_stack_shape
    )

    out_sparray = sparse.diags(
        xp.ones(11), xp.arange(-5, 6), shape=shape, dtype=a_sparray.dtype
    ).tocsr()

    xr_out_dsdbcoo = DSDBCOO.from_sparray(
        out_sparray, global_block_sizes, global_stack_shape
    )
    xl_out_dsdbcoo = DSDBCOO.from_sparray(
        out_sparray, global_block_sizes, global_stack_shape
    )
    xg_out_dsdbcoo = DSDBCOO.from_sparray(
        out_sparray, global_block_sizes, global_stack_shape
    )

    xr_out_dsdbcoo.data[:] = 0.0
    xl_out_dsdbcoo.data[:] = 0.0
    xg_out_dsdbcoo.data[:] = 0.0

    solver = RGFDist()

    solver.selected_solve(
        a=a_dsdbcoo,
        sigma_lesser=sigma_lesser_dsdbcoo,
        sigma_greater=sigma_greater_dsdbcoo,
        out=(xl_out_dsdbcoo, xg_out_dsdbcoo, xr_out_dsdbcoo),
        return_retarded=True,
    )

    # Make reference results
    Xr_rgf = xr_out_dsdbcoo.to_dense()
    Xl_rgf = xl_out_dsdbcoo.to_dense()
    Xg_rgf = xg_out_dsdbcoo.to_dense()

    print("Got to reference results", flush=True)

    _Xr_ref = xp.linalg.inv(a_dsdbcoo.to_dense())
    Xr_ref = xp.zeros_like(_Xr_ref)
    Xr_ref[*Xr_rgf.nonzero()] = _Xr_ref[*Xr_rgf.nonzero()]

    _Xl_ref = (
        _Xr_ref @ sigma_lesser_dsdbcoo.to_dense() @ _Xr_ref.conj().swapaxes(-2, -1)
    )
    Xl_ref = xp.zeros_like(_Xl_ref)
    Xl_ref[*Xl_rgf.nonzero()] = _Xl_ref[*Xl_rgf.nonzero()]

    _Xg_ref = (
        _Xr_ref @ sigma_greater_dsdbcoo.to_dense() @ _Xr_ref.conj().swapaxes(-2, -1)
    )
    Xg_ref = xp.zeros_like(_Xg_ref)
    Xg_ref[*Xg_rgf.nonzero()] = _Xg_ref[*Xg_rgf.nonzero()]

    # Check results
    assert xp.allclose(Xr_rgf, Xr_ref)
    assert xp.allclose(Xl_rgf, Xl_ref)
    assert xp.allclose(Xg_rgf, Xg_ref)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("block_sizes", BLOCK_SIZES)
@pytest.mark.parametrize("global_stack_shape", GLOBAL_STACK_SHAPES)
def test_rgf_dist_no_retarded(block_sizes: NDArray, global_stack_shape: tuple):
    """Tests the selected solve method of a Green's function solver."""

    global_block_sizes = np.tile(block_sizes, comm.block.size)
    shape = (int(global_block_sizes.sum()), int(global_block_sizes.sum()))

    a_sparray = None
    sigma_lesser_sparray = None
    sigma_greater_sparray = None
    if global_comm.rank == 0:
        a_sparray = sparse.diags(
            xp.random.rand(11), xp.arange(-5, 6), shape=shape
        ).tocsr()
        a_sparray.data = xp.random.rand(len(a_sparray.data))
        a_sparray += sparse.diags(
            xp.random.rand(11) * 1j, xp.arange(-5, 6), shape=shape
        ).tocsr()
        a_sparray += sparse.diags(
            [20 * (1 + 1j) * xp.random.rand() * 1j], [0], shape=shape
        ).tocsr()

        sigma_lesser_sparray = sparse.diags(
            xp.random.rand(11), xp.arange(-5, 6), shape=shape
        ).tocsr()
        sigma_lesser_sparray.data = xp.random.rand(len(sigma_lesser_sparray.data))
        sigma_lesser_sparray += sparse.diags(
            xp.random.rand(11) * 1j, xp.arange(-5, 6), shape=shape
        ).tocsr()
        sigma_lesser_sparray = sigma_lesser_sparray - sigma_lesser_sparray.conj().T

        sigma_greater_sparray = sparse.diags(
            xp.random.rand(11), xp.arange(-5, 6), shape=shape
        ).tocsr()
        sigma_greater_sparray.data = xp.random.rand(len(sigma_greater_sparray.data))
        sigma_greater_sparray += sparse.diags(
            xp.random.rand(11) * 1j, xp.arange(-5, 6), shape=shape
        ).tocsr()
        sigma_greater_sparray = sigma_greater_sparray - sigma_greater_sparray.conj().T

    a_sparray = global_comm.bcast(a_sparray, root=0)
    sigma_lesser_sparray = global_comm.bcast(sigma_lesser_sparray, root=0)
    sigma_greater_sparray = global_comm.bcast(sigma_greater_sparray, root=0)

    a_sparray = a_sparray.tocoo()
    a_sparray.sum_duplicates()
    a_sparray.eliminate_zeros()
    a_dsdbcoo = DSDBCOO.from_sparray(a_sparray, global_block_sizes, global_stack_shape)

    sigma_lesser_sparray = sigma_lesser_sparray.tocoo()
    sigma_lesser_sparray.sum_duplicates()
    sigma_lesser_sparray.eliminate_zeros()
    sigma_lesser_dsdbcoo = DSDBCOO.from_sparray(
        sigma_lesser_sparray, global_block_sizes, global_stack_shape
    )

    sigma_greater_sparray = sigma_greater_sparray.tocoo()
    sigma_greater_sparray.sum_duplicates()
    sigma_greater_sparray.eliminate_zeros()
    sigma_greater_dsdbcoo = DSDBCOO.from_sparray(
        sigma_greater_sparray, global_block_sizes, global_stack_shape
    )

    out_sparray = sparse.diags(
        xp.ones(11), xp.arange(-5, 6), shape=shape, dtype=a_sparray.dtype
    ).tocsr()

    xl_out_dsdbcoo = DSDBCOO.from_sparray(
        out_sparray, global_block_sizes, global_stack_shape
    )
    xg_out_dsdbcoo = DSDBCOO.from_sparray(
        out_sparray, global_block_sizes, global_stack_shape
    )

    xl_out_dsdbcoo.data[:] = 0.0
    xg_out_dsdbcoo.data[:] = 0.0

    solver = RGFDist()

    solver.selected_solve(
        a=a_dsdbcoo,
        sigma_lesser=sigma_lesser_dsdbcoo,
        sigma_greater=sigma_greater_dsdbcoo,
        out=(xl_out_dsdbcoo, xg_out_dsdbcoo),
        return_retarded=False,
    )

    # Make reference results
    Xl_rgf = xl_out_dsdbcoo.to_dense()
    Xg_rgf = xg_out_dsdbcoo.to_dense()

    print("Got to reference results", flush=True)

    _Xr_ref = xp.linalg.inv(a_dsdbcoo.to_dense())

    _Xl_ref = (
        _Xr_ref @ sigma_lesser_dsdbcoo.to_dense() @ _Xr_ref.conj().swapaxes(-2, -1)
    )
    Xl_ref = xp.zeros_like(_Xl_ref)
    Xl_ref[*Xl_rgf.nonzero()] = _Xl_ref[*Xl_rgf.nonzero()]

    _Xg_ref = (
        _Xr_ref @ sigma_greater_dsdbcoo.to_dense() @ _Xr_ref.conj().swapaxes(-2, -1)
    )
    Xg_ref = xp.zeros_like(_Xg_ref)
    Xg_ref[*Xg_rgf.nonzero()] = _Xg_ref[*Xg_rgf.nonzero()]

    # Check results
    assert xp.allclose(Xl_rgf, Xl_ref)
    assert xp.allclose(Xg_rgf, Xg_ref)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("block_sizes", BLOCK_SIZES)
@pytest.mark.parametrize("global_stack_shape", GLOBAL_STACK_SHAPES)
def test_rgf_dist_batched(block_sizes: NDArray, global_stack_shape: tuple):
    """Tests the selected solve method of a Green's function solver."""

    global_block_sizes = np.tile(block_sizes, comm.block.size)
    shape = (int(global_block_sizes.sum()), int(global_block_sizes.sum()))

    a_sparray = None
    sigma_lesser_sparray = None
    sigma_greater_sparray = None
    if global_comm.rank == 0:
        a_sparray = sparse.diags(
            xp.random.rand(11), xp.arange(-5, 6), shape=shape
        ).tocsr()
        a_sparray.data = xp.random.rand(len(a_sparray.data))
        a_sparray += sparse.diags(
            xp.random.rand(11) * 1j, xp.arange(-5, 6), shape=shape
        ).tocsr()
        a_sparray += sparse.diags(
            [20 * (1 + 1j) * xp.random.rand() * 1j], [0], shape=shape
        ).tocsr()

        sigma_lesser_sparray = sparse.diags(
            xp.random.rand(11), xp.arange(-5, 6), shape=shape
        ).tocsr()
        sigma_lesser_sparray.data = xp.random.rand(len(sigma_lesser_sparray.data))
        sigma_lesser_sparray += sparse.diags(
            xp.random.rand(11) * 1j, xp.arange(-5, 6), shape=shape
        ).tocsr()
        sigma_lesser_sparray = sigma_lesser_sparray - sigma_lesser_sparray.conj().T

        sigma_greater_sparray = sparse.diags(
            xp.random.rand(11), xp.arange(-5, 6), shape=shape
        ).tocsr()
        sigma_greater_sparray.data = xp.random.rand(len(sigma_greater_sparray.data))
        sigma_greater_sparray += sparse.diags(
            xp.random.rand(11) * 1j, xp.arange(-5, 6), shape=shape
        ).tocsr()
        sigma_greater_sparray = sigma_greater_sparray - sigma_greater_sparray.conj().T

    a_sparray = global_comm.bcast(a_sparray, root=0)
    sigma_lesser_sparray = global_comm.bcast(sigma_lesser_sparray, root=0)
    sigma_greater_sparray = global_comm.bcast(sigma_greater_sparray, root=0)

    a_sparray = a_sparray.tocoo()
    a_sparray.sum_duplicates()
    a_sparray.eliminate_zeros()
    a_dsdbcoo = DSDBCOO.from_sparray(a_sparray, global_block_sizes, global_stack_shape)

    sigma_lesser_sparray = sigma_lesser_sparray.tocoo()
    sigma_lesser_sparray.sum_duplicates()
    sigma_lesser_sparray.eliminate_zeros()
    sigma_lesser_dsdbcoo = DSDBCOO.from_sparray(
        sigma_lesser_sparray, global_block_sizes, global_stack_shape
    )

    sigma_greater_sparray = sigma_greater_sparray.tocoo()
    sigma_greater_sparray.sum_duplicates()
    sigma_greater_sparray.eliminate_zeros()
    sigma_greater_dsdbcoo = DSDBCOO.from_sparray(
        sigma_greater_sparray, global_block_sizes, global_stack_shape
    )

    out_sparray = sparse.diags(
        xp.ones(11), xp.arange(-5, 6), shape=shape, dtype=a_sparray.dtype
    ).tocsr()

    xr_out_dsdbcoo = DSDBCOO.from_sparray(
        out_sparray, global_block_sizes, global_stack_shape
    )
    xl_out_dsdbcoo = DSDBCOO.from_sparray(
        out_sparray, global_block_sizes, global_stack_shape
    )
    xg_out_dsdbcoo = DSDBCOO.from_sparray(
        out_sparray, global_block_sizes, global_stack_shape
    )

    xr_out_dsdbcoo.data[:] = 0.0
    xl_out_dsdbcoo.data[:] = 0.0
    xg_out_dsdbcoo.data[:] = 0.0

    solver = RGFDist(max_batch_size=1)

    solver.selected_solve(
        a=a_dsdbcoo,
        sigma_lesser=sigma_lesser_dsdbcoo,
        sigma_greater=sigma_greater_dsdbcoo,
        out=(xl_out_dsdbcoo, xg_out_dsdbcoo, xr_out_dsdbcoo),
        return_retarded=True,
    )

    # Make reference results
    Xr_rgf = xr_out_dsdbcoo.to_dense()
    Xl_rgf = xl_out_dsdbcoo.to_dense()
    Xg_rgf = xg_out_dsdbcoo.to_dense()

    print("Got to reference results", flush=True)

    _Xr_ref = xp.linalg.inv(a_dsdbcoo.to_dense())
    Xr_ref = xp.zeros_like(_Xr_ref)
    Xr_ref[*Xr_rgf.nonzero()] = _Xr_ref[*Xr_rgf.nonzero()]

    _Xl_ref = (
        _Xr_ref @ sigma_lesser_dsdbcoo.to_dense() @ _Xr_ref.conj().swapaxes(-2, -1)
    )
    Xl_ref = xp.zeros_like(_Xl_ref)
    Xl_ref[*Xl_rgf.nonzero()] = _Xl_ref[*Xl_rgf.nonzero()]

    _Xg_ref = (
        _Xr_ref @ sigma_greater_dsdbcoo.to_dense() @ _Xr_ref.conj().swapaxes(-2, -1)
    )
    Xg_ref = xp.zeros_like(_Xg_ref)
    Xg_ref[*Xg_rgf.nonzero()] = _Xg_ref[*Xg_rgf.nonzero()]

    # Check results
    assert xp.allclose(Xr_rgf, Xr_ref)
    assert xp.allclose(Xl_rgf, Xl_ref)
    assert xp.allclose(Xg_rgf, Xg_ref)
