# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse, bd_matmul, bd_sandwich
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


class TestNonDistr:
    """Tests the non-distributed matrix multiplication and sandwich operations."""

    block_comm_size = 1

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
            block_comm_size=cls.block_comm_size,
            block_comm_config=_default_config,
            stack_comm_config=_default_config,
            override=True,
        )

    def test_bd_matmul(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place addition of a DSDBSparse matrix."""

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

        bd_matmul(
            dsdbsparse, dsdbsparse, out, start_block=start_block, end_block=end_block
        )

        ref = dense @ dense
        val = out.to_dense()

        assert xp.allclose(val, ref)

    def test_bd_sandwich(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place addition of a DSDBSparse matrix."""

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

        bd_sandwich(
            dsdbsparse, dsdbsparse, out, start_block=start_block, end_block=end_block
        )

        assert xp.allclose(dense @ dense @ dense, out.to_dense())


@pytest.mark.mpi(min_size=3)
class TestDistr(TestNonDistr):
    """Tests all tests of TestNotDistr in a distributed setting."""

    pass


@pytest.mark.mpi(min_size=3)
class TestDomainDistr(TestNonDistr):
    """Tests all tests of TestNotDistr in a distributed setting with domain decomposition."""

    block_comm_size = 3
