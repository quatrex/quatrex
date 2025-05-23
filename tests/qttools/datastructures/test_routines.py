# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.comm.comm import GPU_AWARE_MPI
from qttools.datastructures import (
    DSDBSparse,
    bd_matmul,
    bd_matmul_distr,
    bd_sandwich,
    bd_sandwich_distr,
    btd_matmul,
    btd_sandwich,
)
from qttools.utils.mpi_utils import get_section_sizes


def _create_btd_coo(sizes: NDArray) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    offsets = xp.hstack(([0], xp.cumsum(xp.asarray(sizes))))

    arr = xp.zeros((size, size), dtype=xp.complex128)
    for i in range(len(sizes)):
        # Diagonal block.
        block_shape = (int(sizes[i]), int(sizes[i]))
        arr[offsets[i] : offsets[i + 1], offsets[i] : offsets[i + 1]] = xp.random.rand(
            *block_shape
        ) + 1j * xp.random.rand(*block_shape)
        # Superdiagonal block.
        if i < len(sizes) - 1:
            block_shape = (int(sizes[i]), int(sizes[i + 1]))
            arr[offsets[i] : offsets[i + 1], offsets[i + 1] : offsets[i + 2]] = (
                xp.random.rand(*block_shape) + 1j * xp.random.rand(*block_shape)
            )
            arr[offsets[i + 1] : offsets[i + 2], offsets[i] : offsets[i + 1]] = (
                xp.random.rand(*block_shape).T + 1j * xp.random.rand(*block_shape).T
            )
    rng = xp.random.default_rng()
    cutoff = rng.uniform(low=0.1, high=0.4)
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

    def test_bd_matmul(
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

        bd_matmul(dsdbsparse, dsdbsparse, out)

        assert xp.allclose(dense @ dense, out.to_dense())

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


if __name__ == "__main__":
    pytest.main(["--only-mpi", __file__])
