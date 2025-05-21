# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import functools
from types import ModuleType

import pytest
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import DTypeLike, NDArray, sparse, xp
from qttools.comm import comm
from qttools.comm.comm import GPU_AWARE_MPI
from qttools.datastructures import (
    DSDBSparse,
    bbanded_matmul,
    bd_matmul,
    bd_matmul_distr,
    bd_sandwich,
    bd_sandwich_distr,
    btd_matmul,
    btd_sandwich,
)
from qttools.utils.mpi_utils import get_section_sizes


def _create_btd_coo(
    sizes: NDArray, dtype: DTypeLike = xp.complex128, integer: bool = False
) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    offsets = xp.hstack(([0], xp.cumsum(xp.asarray(sizes))))

    rng = xp.random.default_rng(42)

    def _rvs(size=None, rng=rng):
        if integer:
            return rng.integers(-5, 5, size=size)
        return rng.uniform(size=size)

    is_complex = xp.iscomplexobj(dtype(0))

    arr = xp.zeros((size, size), dtype=dtype)
    for i in range(len(sizes)):
        # Diagonal block.
        block_shape = (int(sizes[i]), int(sizes[i]))
        arr[offsets[i] : offsets[i + 1], offsets[i] : offsets[i + 1]] = _rvs(
            block_shape
        )
        if is_complex:
            arr[offsets[i] : offsets[i + 1], offsets[i] : offsets[i + 1]] += 1j * _rvs(
                block_shape
            )
        # Superdiagonal block.
        if i < len(sizes) - 1:
            block_shape = (int(sizes[i]), int(sizes[i + 1]))
            arr[offsets[i] : offsets[i + 1], offsets[i + 1] : offsets[i + 2]] = _rvs(
                block_shape
            )
            if is_complex:
                arr[
                    offsets[i] : offsets[i + 1], offsets[i + 1] : offsets[i + 2]
                ] += 1j * _rvs(block_shape)
            arr[offsets[i + 1] : offsets[i + 2], offsets[i] : offsets[i + 1]] = _rvs(
                block_shape
            ).T
            if is_complex:
                arr[offsets[i + 1] : offsets[i + 2], offsets[i] : offsets[i + 1]] += (
                    1j * _rvs(block_shape).T
                )

    cutoff = rng.uniform(low=0.1, high=0.4)
    if not integer:
        arr[xp.abs(arr) < cutoff] = 0
    return sparse.coo_matrix(arr)


class TestNotDistr:
    """Tests the non-distributed matrix multiplication and sandwich operations."""

    @classmethod
    def setup_class(cls):
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

    def test_btd_matmul(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place addition of a DSDBSparse matrix."""
        coo = _create_btd_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsdbsparse.to_dense()

        # Initalize the output matrix with the correct sparsity pattern.

        out = dsdbsparse_type.from_sparray(coo @ coo, block_sizes, global_stack_shape)

        btd_matmul(dsdbsparse, dsdbsparse, out)

        assert xp.allclose(dense @ dense, out.to_dense())

    def test_btd_sandwich(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place addition of a DSDBSparse matrix."""
        coo = _create_btd_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsdbsparse.to_dense()

        # Initalize the output matrix with the correct sparsity pattern.
        out = dsdbsparse_type.from_sparray(
            coo @ coo @ coo, block_sizes, global_stack_shape
        )

        btd_sandwich(dsdbsparse, dsdbsparse, out)

        assert xp.allclose(dense @ dense @ dense, out.to_dense())

    def test_bd_sandwich(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place addition of a DSDBSparse matrix."""
        coo = _create_btd_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsdbsparse.to_dense()

        # Initalize the output matrix with the correct sparsity pattern.
        out = dsdbsparse_type.from_sparray(
            coo @ coo @ coo, block_sizes, global_stack_shape
        )

        bd_sandwich(dsdbsparse, dsdbsparse, out)

        assert xp.allclose(dense @ dense @ dense, out.to_dense())

    def test_bd_matmul_spillover(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place addition of a DSDBSparse matrix."""
        coo = _create_btd_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsdbsparse.to_dense()
        dense_shape = list(dense.shape)
        NBC = 1
        left_obc = int(sum(block_sizes[0:NBC]))
        right_obc = int(sum(block_sizes[-NBC:]))
        dense_shape[-2] += left_obc + right_obc
        dense_shape[-1] += left_obc + right_obc

        dense_exp = xp.zeros(tuple(dense_shape), dtype=dense.dtype)
        dense_exp[
            ...,
            left_obc : left_obc + sum(block_sizes),
            left_obc : left_obc + sum(block_sizes),
        ] = dense
        # simply repeat the boundaries slices
        dense_exp[..., :left_obc, :-left_obc] = dense_exp[
            ..., left_obc : 2 * left_obc, left_obc:
        ]
        dense_exp[..., :-left_obc, :left_obc] = dense_exp[
            ..., left_obc:, left_obc : 2 * left_obc
        ]
        dense_exp[..., -right_obc:, right_obc:] = dense_exp[
            ..., -2 * right_obc : -right_obc, :-right_obc
        ]
        dense_exp[..., right_obc:, -right_obc:] = dense_exp[
            ..., :-right_obc, -2 * right_obc : -right_obc
        ]

        expended_product = dense_exp @ dense_exp
        ref = expended_product[
            ...,
            left_obc : left_obc + sum(block_sizes),
            left_obc : left_obc + sum(block_sizes),
        ]

        # Initalize the output matrix with the correct sparsity pattern.

        out = dsdbsparse_type.from_sparray(
            sparse.coo_matrix(_get_last_2d(ref)), block_sizes, global_stack_shape
        )

        bd_matmul(dsdbsparse, dsdbsparse, out, spillover_correction=True)

        assert xp.allclose(ref, out.to_dense())

    def test_bd_sandwich_spillover(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place addition of a DSDBSparse matrix."""
        coo = _create_btd_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsdbsparse.to_dense()
        dense_shape = list(dense.shape)
        NBC = 1
        left_obc = int(sum(block_sizes[0:NBC]))
        right_obc = int(sum(block_sizes[-NBC:]))
        dense_shape[-2] += left_obc + right_obc
        dense_shape[-1] += left_obc + right_obc
        dense_exp = xp.zeros(tuple(dense_shape), dtype=dense.dtype)
        dense_exp[
            ...,
            left_obc : left_obc + sum(block_sizes),
            left_obc : left_obc + sum(block_sizes),
        ] = dense
        # simply repeat the boundaries slices
        dense_exp[..., :left_obc, :-left_obc] = dense_exp[
            ..., left_obc : 2 * left_obc, left_obc:
        ]
        dense_exp[..., :-left_obc, :left_obc] = dense_exp[
            ..., left_obc:, left_obc : 2 * left_obc
        ]
        dense_exp[..., -right_obc:, right_obc:] = dense_exp[
            ..., -2 * right_obc : -right_obc, :-right_obc
        ]
        dense_exp[..., right_obc:, -right_obc:] = dense_exp[
            ..., :-right_obc, -2 * right_obc : -right_obc
        ]

        expended_product = dense_exp @ dense_exp @ dense_exp
        ref = expended_product[
            ...,
            left_obc : left_obc + sum(block_sizes),
            left_obc : left_obc + sum(block_sizes),
        ]

        # Initalize the output matrix with the correct sparsity pattern.

        out = dsdbsparse_type.from_sparray(
            sparse.coo_matrix(_get_last_2d(ref)), block_sizes, global_stack_shape
        )

        bd_sandwich(dsdbsparse, dsdbsparse, out, spillover_correction=True)

        assert xp.allclose(ref, out.to_dense())


@pytest.mark.skipif(
    xp.__name__ != "cupy", reason="DSBanded matmul tests require a GPU."
)
def test_bbanded_matmul(
    dsbanded_matmul_type: tuple[DSDBSparse, DSDBSparse],
    block_sizes: NDArray,
    global_stack_shape: tuple,
    banded_block_size: int,
    datatype: DTypeLike,
    dtype: DTypeLike,
):
    """Tests the in-place addition of a DSBSparse matrix."""

    def _set_torch(dsbsparse: DSDBSparse, mod: ModuleType, dt: DTypeLike):

        if isinstance(dsbsparse, tuple):
            matrices = [dsbsparse[0], dsbsparse[1]]
        else:
            matrices = [dsbsparse]

        for dsbsparse in matrices:
            batch_size = functools.reduce(
                lambda x, y: x * y,
                dsbsparse.data.shape[: len(dsbsparse.global_stack_shape)],
            )
            banded_data = dsbsparse.data.reshape((batch_size, *dsbsparse.banded_shape))
            if mod.__name__ == "cupy":
                import torch

                banded_data = banded_data.astype(dt)
                dsbsparse.torch = torch.asarray(banded_data, device="cuda")
            else:  # mod.__name__ == "torch"
                dsbsparse.torch = mod.asarray(banded_data, dtype=dt, device="cuda")

    coo = _create_btd_coo(block_sizes, dtype=datatype, integer=True)
    dense = coo.toarray()

    dsbanded_type_a, dsbanded_type_b = dsbanded_matmul_type
    mod, dt = dtype

    dsbsparse_a = dsbanded_type_a.from_sparray(
        coo, block_sizes, global_stack_shape, banded_block_size=banded_block_size
    )
    if isinstance(dsbsparse_a, tuple):
        val = dsbsparse_a[0].to_dense() + 1j * dsbsparse_a[1].to_dense()
        assert xp.allclose(val, dense)
    else:
        assert xp.allclose(dsbsparse_a.to_dense(), dense)

    dsbsparse_b = dsbanded_type_b.from_sparray(
        coo, block_sizes, global_stack_shape, banded_block_size=banded_block_size
    )
    if isinstance(dsbsparse_b, tuple):
        val = dsbsparse_b[0].to_dense() + 1j * dsbsparse_b[1].to_dense()
        assert xp.allclose(val, dense)
    else:
        assert xp.allclose(dsbsparse_b.to_dense(), dense)

    reference = dense @ dense
    _set_torch(dsbsparse_a, mod, dt)
    _set_torch(dsbsparse_b, mod, dt)

    if isinstance(dsbsparse_a, tuple):
        if isinstance(dsbsparse_b, tuple):
            real = bbanded_matmul(dsbsparse_a[0], dsbsparse_b[0]) - bbanded_matmul(
                dsbsparse_a[1], dsbsparse_b[1]
            )
            imag = bbanded_matmul(dsbsparse_a[0], dsbsparse_b[1]) + bbanded_matmul(
                dsbsparse_a[1], dsbsparse_b[0]
            )
            value = (real, imag)
        else:
            real = bbanded_matmul(dsbsparse_a[0], dsbsparse_b)
            imag = bbanded_matmul(dsbsparse_a[1], dsbsparse_b)
            value = (real, imag)
    else:
        if isinstance(dsbsparse_b, tuple):
            real = bbanded_matmul(dsbsparse_a, dsbsparse_b[0])
            imag = bbanded_matmul(dsbsparse_a, dsbsparse_b[1])
            value = (real, imag)
        else:
            value = bbanded_matmul(dsbsparse_a, dsbsparse_b)
    if isinstance(value, tuple):
        value = value[0].to_dense() + 1j * value[1].to_dense()
    else:
        value = value.to_dense()

    # with xp.printoptions(threshold=xp.inf, linewidth=xp.inf):
    #     print(reference[:block_sizes[0], :sum(block_sizes[:2])])
    #     print()
    #     print(value[0, :block_sizes[0], :sum(block_sizes[:2])])
    #     print()

    relerror = xp.linalg.norm(reference - value) / xp.linalg.norm(reference)
    print(relerror)
    assert xp.allclose(reference, value)


@pytest.mark.skipif(
    xp.__name__ != "cupy", reason="DSBanded matmul tests require a GPU."
)
def test_bbanded_matmul_spillover(
    dsbanded_matmul_type: tuple[DSDBSparse, DSDBSparse],
    block_sizes: NDArray,
    global_stack_shape: tuple,
    banded_block_size: int,
    datatype: DTypeLike,
    dtype: DTypeLike,
):
    """Tests the in-place addition of a DSBSparse matrix."""

    def _set_torch(dsbsparse: DSDBSparse, mod: ModuleType, dt: DTypeLike):

        if isinstance(dsbsparse, tuple):
            matrices = [dsbsparse[0], dsbsparse[1]]
        else:
            matrices = [dsbsparse]

        for dsbsparse in matrices:
            batch_size = functools.reduce(
                lambda x, y: x * y,
                dsbsparse.data.shape[: len(dsbsparse.global_stack_shape)],
            )
            banded_data = dsbsparse.data.reshape((batch_size, *dsbsparse.banded_shape))
            if mod.__name__ == "cupy":
                import torch

                banded_data = banded_data.astype(dt)
                dsbsparse.torch = torch.asarray(banded_data, device="cuda")
            else:  # mod.__name__ == "torch"
                dsbsparse.torch = mod.asarray(banded_data, dtype=dt, device="cuda")

    coo = _create_btd_coo(block_sizes, dtype=datatype, integer=True)
    dense = coo.toarray()

    # with xp.printoptions(threshold=xp.inf, linewidth=xp.inf):
    #     print(dense[:sum(block_sizes[:2]), :sum(block_sizes[:2])])
    #     print()

    dense_shape = list(dense.shape)
    NBC = 1
    left_obc = int(sum(block_sizes[0:NBC]))
    right_obc = int(sum(block_sizes[-NBC:]))
    dense_shape[-2] += left_obc + right_obc
    dense_shape[-1] += left_obc + right_obc

    dense_exp = xp.zeros(tuple(dense_shape), dtype=dense.dtype)
    dense_exp[
        ...,
        left_obc : left_obc + sum(block_sizes),
        left_obc : left_obc + sum(block_sizes),
    ] = dense
    # simply repeat the boundaries slices
    dense_exp[..., :left_obc, :-left_obc] = dense_exp[
        ..., left_obc : 2 * left_obc, left_obc:
    ]
    dense_exp[..., :-left_obc, :left_obc] = dense_exp[
        ..., left_obc:, left_obc : 2 * left_obc
    ]
    dense_exp[..., -right_obc:, right_obc:] = dense_exp[
        ..., -2 * right_obc : -right_obc, :-right_obc
    ]
    dense_exp[..., right_obc:, -right_obc:] = dense_exp[
        ..., :-right_obc, -2 * right_obc : -right_obc
    ]

    # with xp.printoptions(threshold=xp.inf, linewidth=xp.inf):
    #     print(dense_exp[:sum(block_sizes[:3]), :sum(block_sizes[:3])])
    #     print()
    #     print(dense_exp[block_sizes[0], :sum(block_sizes[:3])])
    #     print(dense_exp[:sum(block_sizes[:3]), block_sizes[0]].T)
    #     print(dense_exp[block_sizes[0], :sum(block_sizes[:3])] * dense_exp[:sum(block_sizes[:3]), block_sizes[0]])
    #     print(dense_exp[block_sizes[0], :sum(block_sizes[:3])] @ dense_exp[:sum(block_sizes[:3]), block_sizes[0]])
    #     print()

    expended_product = dense_exp @ dense_exp
    reference = expended_product[
        ...,
        left_obc : left_obc + sum(block_sizes),
        left_obc : left_obc + sum(block_sizes),
    ]

    dsbanded_type_a, dsbanded_type_b = dsbanded_matmul_type
    mod, dt = dtype

    dsbsparse_a = dsbanded_type_a.from_sparray(
        coo, block_sizes, global_stack_shape, banded_block_size=banded_block_size
    )
    if isinstance(dsbsparse_a, tuple):
        val = dsbsparse_a[0].to_dense() + 1j * dsbsparse_a[1].to_dense()
        assert xp.allclose(val, dense)
    else:
        assert xp.allclose(dsbsparse_a.to_dense(), dense)

    dsbsparse_b = dsbanded_type_b.from_sparray(
        coo, block_sizes, global_stack_shape, banded_block_size=banded_block_size
    )
    if isinstance(dsbsparse_b, tuple):
        val = dsbsparse_b[0].to_dense() + 1j * dsbsparse_b[1].to_dense()
        assert xp.allclose(val, dense)
    else:
        assert xp.allclose(dsbsparse_b.to_dense(), dense)

    # _set_torch(dsbsparse_a, mod, dt)
    # _set_torch(dsbsparse_b, mod, dt)

    def _matmul(a, b):
        a.enforce_boundary_conditions()
        _set_torch(a, mod, dt)
        # a_data = a.data.reshape((a.data.shape[:len(a.global_stack_shape)] + a.banded_shape))
        # with xp.printoptions(threshold=xp.inf, linewidth=xp.inf):
        #     print(a_data[0, :sum(block_sizes[:2])])
        #     print()
        b.enforce_boundary_conditions()
        _set_torch(b, mod, dt)
        # b_data = b.data.reshape((b.data.shape[:len(b.global_stack_shape)] + b.banded_shape))
        # with xp.printoptions(threshold=xp.inf, linewidth=xp.inf):
        #     print(b_data[0, :, :sum(block_sizes[:2])])
        #     print()
        out = bbanded_matmul(a, b, spillover_correction=False)
        out_data = out.data.reshape(
            (out.data.shape[: len(out.global_stack_shape)] + out.banded_shape)
        )
        with xp.printoptions(threshold=xp.inf, linewidth=xp.inf):
            print(out_data[0, : sum(block_sizes[:2])])
            print()
        return out

    if isinstance(dsbsparse_a, tuple):
        if isinstance(dsbsparse_b, tuple):
            real = _matmul(dsbsparse_a[0], dsbsparse_b[0]) - _matmul(
                dsbsparse_a[1], dsbsparse_b[1]
            )
            imag = _matmul(dsbsparse_a[0], dsbsparse_b[1]) + _matmul(
                dsbsparse_a[1], dsbsparse_b[0]
            )
            value = (real, imag)
        else:
            real = _matmul(dsbsparse_a[0], dsbsparse_b)
            imag = _matmul(dsbsparse_a[1], dsbsparse_b)
            value = (real, imag)
    else:
        if isinstance(dsbsparse_b, tuple):
            real = _matmul(dsbsparse_a, dsbsparse_b[0])
            imag = _matmul(dsbsparse_a, dsbsparse_b[1])
            value = (real, imag)
        else:
            value = _matmul(dsbsparse_a, dsbsparse_b)
    if isinstance(value, tuple):
        value = value[0].to_dense() + 1j * value[1].to_dense()
    else:
        value = value.to_dense()

    # with xp.printoptions(threshold=xp.inf, linewidth=xp.inf):
    #     print(reference[:block_sizes[0], :sum(block_sizes[:2])])
    #     print()
    #     print(value[0, :block_sizes[0], :sum(block_sizes[:2])])
    #     print()

    relerror = xp.linalg.norm(reference - value) / xp.linalg.norm(reference)
    print(relerror)
    assert xp.allclose(reference, value)


def _get_last_2d(x):
    m, n = x.shape[-2:]
    return x.flat[: m * n].reshape(m, n)


@pytest.mark.mpi(min_size=3)
class TestDistr:
    """Tests the distributed matrix multiplication and sandwich operations."""

    @classmethod
    def setup_class(cls):
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

    def test_bd_matmul_distr(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place addition of a DSDBSparse matrix."""

        if (
            xp.__name__ == "cupy"
            and not GPU_AWARE_MPI
            and not hasattr(comm.block, "_nccl_comm")
            and comm.block.size > 1
        ):
            pytest.skip(
                "Skipping test because GPU-aware MPI is not available and the block communicator does not have an NCCL communicator."
            )

        coo = _create_btd_coo(block_sizes)
        coo = global_comm.bcast(coo, root=0)
        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        dense = dsdbsparse.to_dense()

        # Initalize the output matrix with the correct sparsity pattern.

        out = dsdbsparse_type_dist.from_sparray(
            coo @ coo, block_sizes, global_stack_shape
        )
        out.data[:] = 0

        local_blocks, _ = get_section_sizes(len(block_sizes), comm.block.size)
        start_block = sum(local_blocks[: comm.block.rank])
        end_block = start_block + local_blocks[comm.block.rank]

        bd_matmul_distr(
            dsdbsparse, dsdbsparse, out, start_block=start_block, end_block=end_block
        )

        ref = dense @ dense
        val = out.to_dense()

        assert xp.allclose(val, ref)

    def test_bd_sandwich_distr(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place addition of a DSDBSparse matrix."""

        if (
            xp.__name__ == "cupy"
            and not GPU_AWARE_MPI
            and not hasattr(comm.block, "_nccl_comm")
            and comm.block.size > 1
        ):
            pytest.skip(
                "Skipping test because GPU-aware MPI is not available and the block communicator does not have an NCCL communicator."
            )

        coo = _create_btd_coo(block_sizes)
        coo = global_comm.bcast(coo, root=0)
        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        dense = dsdbsparse.to_dense()

        # Initalize the output matrix with the correct sparsity pattern.

        out = dsdbsparse_type_dist.from_sparray(
            coo @ coo @ coo, block_sizes, global_stack_shape
        )
        out.data[:] = 0

        local_blocks, _ = get_section_sizes(len(block_sizes), comm.block.size)
        start_block = sum(local_blocks[: comm.block.rank])
        end_block = start_block + local_blocks[comm.block.rank]

        bd_sandwich_distr(
            dsdbsparse, dsdbsparse, out, start_block=start_block, end_block=end_block
        )

        assert xp.allclose(dense @ dense @ dense, out.to_dense())

    def test_bd_matmul_distr_spillover(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place addition of a DSDBSparse matrix."""

        if (
            xp.__name__ == "cupy"
            and not GPU_AWARE_MPI
            and not hasattr(comm.block, "_nccl_comm")
            and comm.block.size > 1
        ):
            pytest.skip(
                "Skipping test because GPU-aware MPI is not available and the block communicator does not have an NCCL communicator."
            )

        coo = _create_btd_coo(block_sizes)
        coo = global_comm.bcast(coo, root=0)
        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        dense = dsdbsparse.to_dense()
        dense_shape = list(dense.shape)
        NBC = 1
        left_obc = int(sum(block_sizes[0:NBC]))
        right_obc = int(sum(block_sizes[-NBC:]))
        dense_shape[-2] += left_obc + right_obc
        dense_shape[-1] += left_obc + right_obc

        dense_exp = xp.zeros(tuple(dense_shape), dtype=dense.dtype)
        dense_exp[
            ...,
            left_obc : left_obc + sum(block_sizes),
            left_obc : left_obc + sum(block_sizes),
        ] = dense
        # simply repeat the boundaries slices
        dense_exp[..., :left_obc, :-left_obc] = dense_exp[
            ..., left_obc : 2 * left_obc, left_obc:
        ]
        dense_exp[..., :-left_obc, :left_obc] = dense_exp[
            ..., left_obc:, left_obc : 2 * left_obc
        ]
        dense_exp[..., -right_obc:, right_obc:] = dense_exp[
            ..., -2 * right_obc : -right_obc, :-right_obc
        ]
        dense_exp[..., right_obc:, -right_obc:] = dense_exp[
            ..., :-right_obc, -2 * right_obc : -right_obc
        ]

        expended_product = dense_exp @ dense_exp
        ref = expended_product[
            ...,
            left_obc : left_obc + sum(block_sizes),
            left_obc : left_obc + sum(block_sizes),
        ]

        # Initalize the output matrix with the correct sparsity pattern.

        out = dsdbsparse_type_dist.from_sparray(
            sparse.coo_matrix(_get_last_2d(ref)), block_sizes, global_stack_shape
        )
        out.data[:] = 0

        local_blocks, _ = get_section_sizes(len(block_sizes), comm.block.size)
        start_block = sum(local_blocks[: comm.block.rank])
        end_block = start_block + local_blocks[comm.block.rank]

        bd_matmul_distr(
            dsdbsparse,
            dsdbsparse,
            out,
            start_block=start_block,
            end_block=end_block,
            spillover_correction=True,
        )

        assert xp.allclose(ref, out.to_dense())

    def test_bd_sandwich_distr_spillover(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place addition of a DSDBSparse matrix."""

        if (
            xp.__name__ == "cupy"
            and not GPU_AWARE_MPI
            and not hasattr(comm.block, "_nccl_comm")
            and comm.block.size > 1
        ):
            pytest.skip(
                "Skipping test because GPU-aware MPI is not available and the block communicator does not have an NCCL communicator."
            )

        coo = _create_btd_coo(block_sizes)
        coo = global_comm.bcast(coo, root=0)
        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        dense = dsdbsparse.to_dense()
        dense_shape = list(dense.shape)
        NBC = 1
        left_obc = int(sum(block_sizes[0:NBC]))
        right_obc = int(sum(block_sizes[-NBC:]))
        dense_shape[-2] += left_obc + right_obc
        dense_shape[-1] += left_obc + right_obc
        dense_exp = xp.zeros(tuple(dense_shape), dtype=dense.dtype)
        dense_exp[
            ...,
            left_obc : left_obc + sum(block_sizes),
            left_obc : left_obc + sum(block_sizes),
        ] = dense
        # simply repeat the boundaries slices
        dense_exp[..., :left_obc, :-left_obc] = dense_exp[
            ..., left_obc : 2 * left_obc, left_obc:
        ]
        dense_exp[..., :-left_obc, :left_obc] = dense_exp[
            ..., left_obc:, left_obc : 2 * left_obc
        ]
        dense_exp[..., -right_obc:, right_obc:] = dense_exp[
            ..., -2 * right_obc : -right_obc, :-right_obc
        ]
        dense_exp[..., right_obc:, -right_obc:] = dense_exp[
            ..., :-right_obc, -2 * right_obc : -right_obc
        ]

        expended_product = dense_exp @ dense_exp @ dense_exp
        ref = expended_product[
            ...,
            left_obc : left_obc + sum(block_sizes),
            left_obc : left_obc + sum(block_sizes),
        ]

        # Initalize the output matrix with the correct sparsity pattern.

        out = dsdbsparse_type_dist.from_sparray(
            sparse.coo_matrix(_get_last_2d(ref)), block_sizes, global_stack_shape
        )
        out.data[:] = 0

        local_blocks, _ = get_section_sizes(len(block_sizes), comm.block.size)
        start_block = sum(local_blocks[: comm.block.rank])
        end_block = start_block + local_blocks[comm.block.rank]

        bd_sandwich_distr(
            dsdbsparse,
            dsdbsparse,
            out,
            start_block=start_block,
            end_block=end_block,
            spillover_correction=True,
        )

        assert xp.allclose(ref, out.to_dense())


@pytest.mark.skipif(
    xp.__name__ != "cupy", reason="DSBanded matmul tests require a GPU."
)
def test_bbanded_sandwich_spillover(
    dsbanded_matmul_type: tuple[DSDBSparse, DSDBSparse],
    block_sizes: NDArray,
    global_stack_shape: tuple,
    banded_block_size: int,
    datatype: DTypeLike,
    dtype: DTypeLike,
):
    """Tests the in-place addition of a DSBSparse matrix."""

    def _set_torch(dsbsparse: DSDBSparse, mod: ModuleType, dt: DTypeLike):

        if isinstance(dsbsparse, tuple):
            matrices = [dsbsparse[0], dsbsparse[1]]
        else:
            matrices = [dsbsparse]

        for dsbsparse in matrices:
            batch_size = functools.reduce(
                lambda x, y: x * y,
                dsbsparse.data.shape[: len(dsbsparse.global_stack_shape)],
            )
            banded_data = dsbsparse.data.reshape((batch_size, *dsbsparse.banded_shape))
            if mod.__name__ == "cupy":
                import torch

                banded_data = banded_data.astype(dt)
                dsbsparse.torch = torch.asarray(banded_data, device="cuda")
            else:  # mod.__name__ == "torch"
                dsbsparse.torch = mod.asarray(banded_data, dtype=dt, device="cuda")

    coo = _create_btd_coo(block_sizes, dtype=datatype, integer=True)
    dense = coo.toarray()
    dense_shape = list(dense.shape)
    NBC = 1
    left_obc = int(sum(block_sizes[0:NBC]))
    right_obc = int(sum(block_sizes[-NBC:]))
    dense_shape[-2] += left_obc + right_obc
    dense_shape[-1] += left_obc + right_obc
    dense_exp = xp.zeros(tuple(dense_shape), dtype=dense.dtype)
    dense_exp[
        ...,
        left_obc : left_obc + sum(block_sizes),
        left_obc : left_obc + sum(block_sizes),
    ] = dense
    # simply repeat the boundaries slices
    dense_exp[..., :left_obc, :-left_obc] = dense_exp[
        ..., left_obc : 2 * left_obc, left_obc:
    ]
    dense_exp[..., :-left_obc, :left_obc] = dense_exp[
        ..., left_obc:, left_obc : 2 * left_obc
    ]
    dense_exp[..., -right_obc:, right_obc:] = dense_exp[
        ..., -2 * right_obc : -right_obc, :-right_obc
    ]
    dense_exp[..., right_obc:, -right_obc:] = dense_exp[
        ..., :-right_obc, -2 * right_obc : -right_obc
    ]

    expended_product = dense_exp @ dense_exp @ dense_exp
    reference = expended_product[
        ...,
        left_obc : left_obc + sum(block_sizes),
        left_obc : left_obc + sum(block_sizes),
    ]

    dsbanded_type_a, dsbanded_type_b = dsbanded_matmul_type
    mod, dt = dtype

    dsbsparse_a = dsbanded_type_a.from_sparray(
        coo, block_sizes, global_stack_shape, banded_block_size=banded_block_size
    )
    if isinstance(dsbsparse_a, tuple):
        val = dsbsparse_a[0].to_dense() + 1j * dsbsparse_a[1].to_dense()
        assert xp.allclose(val, dense)
    else:
        assert xp.allclose(dsbsparse_a.to_dense(), dense)

    dsbsparse_b = dsbanded_type_b.from_sparray(
        coo, block_sizes, global_stack_shape, banded_block_size=banded_block_size
    )
    if isinstance(dsbsparse_b, tuple):
        val = dsbsparse_b[0].to_dense() + 1j * dsbsparse_b[1].to_dense()
        assert xp.allclose(val, dense)
    else:
        assert xp.allclose(dsbsparse_b.to_dense(), dense)

    dsbsparse_c = dsbanded_type_b.from_sparray(
        coo, block_sizes, global_stack_shape, banded_block_size=banded_block_size
    )
    if isinstance(dsbsparse_c, tuple):
        val = dsbsparse_c[0].to_dense() + 1j * dsbsparse_c[1].to_dense()
        assert xp.allclose(val, dense)
    else:
        assert xp.allclose(dsbsparse_c.to_dense(), dense)

    # _set_torch(dsbsparse_a, mod, dt)
    # _set_torch(dsbsparse_b, mod, dt)

    def _matmul(a, b, spillover_a: bool = True):
        if spillover_a:
            a.enforce_boundary_conditions()
        _set_torch(a, mod, dt)
        b.enforce_boundary_conditions()
        _set_torch(b, mod, dt)
        out = bbanded_matmul(
            a, b, spillover_correction=False, sandwhich=not spillover_a
        )
        return out

    def _matmul_high(a, b, spillover_a: bool = True):
        if isinstance(a, tuple):
            if isinstance(b, tuple):
                real = _matmul(a[0], b[0]) - _matmul(
                    a[1], b[1], spillover_a=spillover_a
                )
                imag = _matmul(a[0], b[1]) + _matmul(
                    a[1], b[0], spillover_a=spillover_a
                )
                value = (real, imag)
            else:
                real = _matmul(a[0], b, spillover_a=spillover_a)
                imag = _matmul(a[1], b, spillover_a=spillover_a)
                value = (real, imag)
        else:
            if isinstance(b, tuple):
                real = _matmul(a, b[0], spillover_a=spillover_a)
                imag = _matmul(a, b[1], spillover_a=spillover_a)
                value = (real, imag)
            else:
                value = _matmul(a, b, spillover_a=spillover_a)
        return value

    tmp = _matmul_high(dsbsparse_a, dsbsparse_b)
    value = _matmul_high(tmp, dsbsparse_c, spillover_a=False)
    if isinstance(value, tuple):
        value = value[0].to_dense() + 1j * value[1].to_dense()
    else:
        value = value.to_dense()

    # with xp.printoptions(threshold=xp.inf, linewidth=xp.inf):
    #     print(reference[:block_sizes[0], :sum(block_sizes[:2])])
    #     print()
    #     print(value[0, :block_sizes[0], :sum(block_sizes[:2])])
    #     print()

    relerror = xp.linalg.norm(reference - value) / xp.linalg.norm(reference)
    print(relerror)
    assert xp.allclose(reference, value)


def _get_last_2d(x):
    m, n = x.shape[-2:]
    return x.flat[: m * n].reshape(m, n)


if __name__ == "__main__":
    pytest.main(["--only-mpi", __file__])
