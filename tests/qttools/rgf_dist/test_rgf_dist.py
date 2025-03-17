# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, host_xp, sparse, xp
from qttools.datastructures.dbsparse import DBCOO
from qttools.greens_function_solver.rgf_dist import RGFDist

BLOCK_SIZES = [
    pytest.param(host_xp.array([10] * 5), id="constant-block-size"),
    pytest.param(host_xp.array([5] * 3 + [10] * 2 + [5] * 3), id="mixed-block-size"),
]


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("block_sizes", BLOCK_SIZES)
def test_rgf_dist(
    block_sizes: NDArray,
):
    """Tests the selected solve method of a Green's function solver."""

    global_block_sizes = xp.tile(block_sizes, comm.Get_size())
    shape = (global_block_sizes.sum(), global_block_sizes.sum())

    a_sparray = None
    bl_sparray = None
    bg_sparray = None
    if comm.rank == 0:
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

        bl_sparray = sparse.diags(
            xp.random.rand(11), xp.arange(-5, 6), shape=shape
        ).tocsr()
        bl_sparray.data = xp.random.rand(len(bl_sparray.data))
        bl_sparray += sparse.diags(
            xp.random.rand(11) * 1j, xp.arange(-5, 6), shape=shape
        ).tocsr()
        bl_sparray = bl_sparray - bl_sparray.conj().T

        bg_sparray = sparse.diags(
            xp.random.rand(11), xp.arange(-5, 6), shape=shape
        ).tocsr()
        bg_sparray.data = xp.random.rand(len(bg_sparray.data))
        bg_sparray += sparse.diags(
            xp.random.rand(11) * 1j, xp.arange(-5, 6), shape=shape
        ).tocsr()
        bg_sparray = bg_sparray - bg_sparray.conj().T

    a_sparray = comm.bcast(a_sparray, root=0)
    bl_sparray = comm.bcast(bl_sparray, root=0)
    bg_sparray = comm.bcast(bg_sparray, root=0)

    a_sparray = a_sparray.tocoo()
    a_sparray.sum_duplicates()
    a_sparray.eliminate_zeros()
    a_dbcoo = DBCOO.from_sparray(a_sparray, global_block_sizes)

    bl_sparray = bl_sparray.tocoo()
    bl_sparray.sum_duplicates()
    bl_sparray.eliminate_zeros()
    bl_dbcoo = DBCOO.from_sparray(bl_sparray, global_block_sizes)

    bg_sparray = bg_sparray.tocoo()
    bg_sparray.sum_duplicates()
    bg_sparray.eliminate_zeros()
    bg_dbcoo = DBCOO.from_sparray(bg_sparray, global_block_sizes)

    out_sparray = sparse.diags(
        xp.ones(11), xp.arange(-5, 6), shape=shape, dtype=a_sparray.dtype
    ).tocsr()

    out_dbcoo = DBCOO.from_sparray(out_sparray, global_block_sizes)
    out_dbcoo.local_data[:] = 0.0

    xl_out_dbcoo = DBCOO.from_sparray(out_sparray, global_block_sizes)
    xl_out_dbcoo.local_data[:] = 0.0

    xg_out_dbcoo = DBCOO.from_sparray(out_sparray, global_block_sizes)
    xg_out_dbcoo.local_data[:] = 0.0

    solver = RGFDist(solve_lesser=True, solve_greater=True)

    solver.selected_solve(
        a=a_dbcoo,
        out=out_dbcoo,
        bl=bl_dbcoo,
        xl_out=xl_out_dbcoo,
        bg=bg_dbcoo,
        xg_out=xg_out_dbcoo,
    )

    # Make reference results
    Xr_rgf = out_dbcoo.to_dense()
    Xl_rgf = xl_out_dbcoo.to_dense()
    Xg_rgf = xg_out_dbcoo.to_dense()

    _Xr_ref = xp.linalg.inv(a_sparray.toarray())
    Xr_ref = xp.zeros_like(_Xr_ref)
    Xr_ref[*Xr_rgf.nonzero()] = _Xr_ref[*Xr_rgf.nonzero()]

    _Xl_ref = _Xr_ref @ bl_sparray.toarray() @ _Xr_ref.conj().T
    Xl_ref = xp.zeros_like(_Xl_ref)
    Xl_ref[*Xl_rgf.nonzero()] = _Xl_ref[*Xl_rgf.nonzero()]

    _Xg_ref = _Xr_ref @ bg_sparray.toarray() @ _Xr_ref.conj().T
    Xg_ref = xp.zeros_like(_Xg_ref)
    Xg_ref[*Xg_rgf.nonzero()] = _Xg_ref[*Xg_rgf.nonzero()]

    # Check results
    assert xp.allclose(Xr_rgf, Xr_ref)
    assert xp.allclose(Xl_rgf, Xl_ref)
    assert xp.allclose(Xg_rgf, Xg_ref)
