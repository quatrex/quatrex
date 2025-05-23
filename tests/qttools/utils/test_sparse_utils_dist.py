# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import functools

import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.comm.comm import GPU_AWARE_MPI
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.utils.mpi_utils import get_section_sizes
from qttools.utils.sparse_utils import product_sparsity_pattern_dsdbsparse

GLOBAL_STACK_SHAPES = [
    pytest.param((7,), id="1D-stack"),
    pytest.param((6, 2), id="2D-stack"),
]


# TODO test for block distributed
@pytest.fixture(
    autouse=True,
    scope="module",
    params=[
        pytest.param(1, id="block-comm-size-1"),
        pytest.param(3, id="block-comm-size-3"),
    ],
)
def configure_comm(request):
    """Setup any state specific to the execution of the given module."""
    block_comm_size = request.param

    # Default configuration setup based on the xp module
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

    # Configure the comm singleton with the parameterized block_comm_size
    comm.configure(
        block_comm_size=block_comm_size,
        block_comm_config=_default_config,
        stack_comm_config=_default_config,
        override=True,
    )


def _create_btd_coo(sizes: NDArray) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    offsets = xp.hstack(([0], xp.cumsum(xp.asarray(sizes))))

    arr = xp.zeros((size, size), dtype=xp.float32)
    for i in range(len(sizes)):
        # Diagonal block.
        block_shape = (int(sizes[i]), int(sizes[i]))
        arr[offsets[i] : offsets[i + 1], offsets[i] : offsets[i + 1]] = xp.random.rand(
            *block_shape
        )  # + 1j * xp.random.rand(*block_shape)
        # Superdiagonal block.
        if i < len(sizes) - 1:
            block_shape = (int(sizes[i]), int(sizes[i + 1]))
            arr[offsets[i] : offsets[i + 1], offsets[i + 1] : offsets[i + 2]] = (
                xp.random.rand(*block_shape)  # + 1j * xp.random.rand(*block_shape)
            )
            arr[offsets[i + 1] : offsets[i + 2], offsets[i] : offsets[i + 1]] = (
                xp.random.rand(*block_shape).T  # + 1j * xp.random.rand(*block_shape).T
            )
    rng = xp.random.default_rng()
    cutoff = rng.uniform(low=0.1, high=0.4)
    arr[xp.abs(arr) < cutoff] = 0
    coo = sparse.coo_matrix(arr)
    coo.data[:] = 1
    return coo


def _create_btd_coo_periodic(sizes: NDArray) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    offsets = xp.hstack(([0], xp.cumsum(xp.asarray(sizes))))

    arr = xp.zeros((size, size), dtype=xp.float32)

    rng = xp.random.default_rng()
    block1 = xp.random.rand(*(int(sizes[0]), int(sizes[0])))
    block2 = xp.random.rand(*(int(sizes[0]), int(sizes[0])))
    block3 = xp.random.rand(*(int(sizes[0]), int(sizes[0])))
    cutoff = rng.uniform(low=0.1, high=0.4)

    for i in range(len(sizes)):
        # Diagonal block.
        arr[offsets[i] : offsets[i + 1], offsets[i] : offsets[i + 1]] = (
            block1  # + 1j * xp.random.rand(*block_shape)
        )
        arr[offsets[i] : offsets[i + 1], offsets[i] : offsets[i + 1]][
            xp.abs(block1) < cutoff
        ] = 0
        # Superdiagonal block.
        if i < len(sizes) - 1:
            arr[offsets[i] : offsets[i + 1], offsets[i + 1] : offsets[i + 2]] = block2
            arr[offsets[i + 1] : offsets[i + 2], offsets[i] : offsets[i + 1]] = block3
            arr[offsets[i] : offsets[i + 1], offsets[i + 1] : offsets[i + 2]][
                xp.abs(block2) < cutoff
            ] = 0
            arr[offsets[i + 1] : offsets[i + 2], offsets[i] : offsets[i + 1]][
                xp.abs(block3) < cutoff
            ] = 0

    coo = sparse.coo_matrix(arr)
    coo.data[:] = 1
    return coo


def _expand_matrix(
    matrix: sparse.spmatrix, block_sizes: NDArray, NBC: int = 1
) -> sparse.spmatrix:

    shape = list(matrix.shape)
    left_obc = int(sum(block_sizes[0:NBC]))
    right_obc = int(sum(block_sizes[-NBC:]))
    shape[-2] += left_obc + right_obc
    shape[-1] += left_obc + right_obc

    csr = matrix.tocsr()

    expanded = sparse.csr_matrix(tuple(shape), dtype=matrix.dtype)

    # simply repeat the boundaries slices
    expanded[
        left_obc : left_obc + int(sum(block_sizes)),
        left_obc : left_obc + int(sum(block_sizes)),
    ] = csr
    expanded[:left_obc, :-left_obc] = expanded[left_obc : 2 * left_obc, left_obc:]
    expanded[:-left_obc, :left_obc] = expanded[left_obc:, left_obc : 2 * left_obc]
    expanded[-right_obc:, right_obc:] = expanded[
        -2 * right_obc : -right_obc, :-right_obc
    ]
    expanded[right_obc:, -right_obc:] = expanded[
        :-right_obc, -2 * right_obc : -right_obc
    ]

    return expanded


@pytest.mark.mpi(min_size=3)
@pytest.mark.parametrize("global_stack_shape", GLOBAL_STACK_SHAPES)
def test_product_sparsity_dsdbsparse(
    dsdbsparse_type: DSDBSparse,
    num_matrices: int,
    block_sizes: NDArray,
    global_stack_shape: tuple,
):
    """Tests the computation of the matrix product's sparsity pattern."""

    if (
        xp.__name__ == "cupy"
        and not GPU_AWARE_MPI
        and not hasattr(comm.block, "_nccl_comm")
        and comm.block.size > 1
    ):
        pytest.skip(
            "Skipping test because GPU-aware MPI is not available and the block communicator does not have an NCCL communicator."
        )

    if dsdbsparse_type.__name__ == "DSDBCSR":
        pytest.skip("DSDBCSR does not support this test")

    last_block_sizes = block_sizes[-3:]
    if num_matrices > 3:
        block_sizes = np.hstack(
            (block_sizes, *[last_block_sizes for _ in range(num_matrices - 3)])
        )
    matrices = [_create_btd_coo(block_sizes) for _ in range(num_matrices)]
    matrices = [global_comm.bcast(matrix, root=0) for matrix in matrices]
    dsdbsparse_matrices = [
        dsdbsparse_type.from_sparray(matrix, block_sizes, global_stack_shape)
        for matrix in matrices
    ]
    dense_matrices = [matrix.to_dense() for matrix in dsdbsparse_matrices]
    for i in range(num_matrices):
        assert xp.allclose(dense_matrices[i], matrices[i].toarray())

    product = functools.reduce(lambda x, y: x @ y, matrices)
    product.data[:] = 1
    ref = product.toarray()

    local_blocks, _ = get_section_sizes(len(block_sizes), comm.block.size)
    start_block = sum(local_blocks[: comm.block.rank])
    end_block = start_block + local_blocks[comm.block.rank]

    rows, cols = product_sparsity_pattern_dsdbsparse(
        *dsdbsparse_matrices,
        in_num_diag=3,
        start_block=start_block,
        end_block=end_block,
    )
    val = sparse.coo_matrix(
        (xp.ones(len(rows)), (rows, cols)), shape=product.shape
    ).toarray()

    # Each rank in the block communicator computes its own local sparsity pattern.
    full = sum(comm.block._mpi_comm.allgather(val))

    if comm.rank == 0:
        print(xp.nonzero(ref - full))

    assert xp.allclose(ref, full)


@pytest.mark.mpi(min_size=3)
@pytest.mark.parametrize("global_stack_shape", GLOBAL_STACK_SHAPES)
def test_product_sparsity_dsdbsparse_spillover(
    dsdbsparse_type: DSDBSparse,
    num_matrices: int,
    block_sizes: NDArray,
    global_stack_shape: tuple,
):
    """Tests the computation of the matrix product's sparsity pattern."""

    if not np.all(block_sizes == block_sizes.flat[0]):
        pytest.skip(
            "Skipping test because the block sizes are not all equal."
            + "The construction of the test matrix would need to be changed"
            + "such that the only the boundary layers are periodic"
        )

    if (
        xp.__name__ == "cupy"
        and not GPU_AWARE_MPI
        and not hasattr(comm.block, "_nccl_comm")
        and comm.block.size > 1
    ):
        pytest.skip(
            "Skipping test because GPU-aware MPI is not available and the block communicator does not have an NCCL communicator."
        )

    if dsdbsparse_type.__name__ == "DSDBCSR":
        pytest.skip("DSDBCSR does not support this test")

    last_block_sizes = block_sizes[-3:]
    if num_matrices > 3:
        block_sizes = np.hstack(
            (block_sizes, *[last_block_sizes for _ in range(num_matrices - 3)])
        )
    matrices = [_create_btd_coo_periodic(block_sizes) for _ in range(num_matrices)]
    matrices = [global_comm.bcast(matrix, root=0) for matrix in matrices]
    dsdbsparse_matrices = [
        dsdbsparse_type.from_sparray(matrix, block_sizes, global_stack_shape)
        for matrix in matrices
    ]
    dense_matrices = [matrix.to_dense() for matrix in dsdbsparse_matrices]
    for i in range(num_matrices):
        assert xp.allclose(dense_matrices[i], matrices[i].toarray())

    shape = matrices[0].shape
    expanded_matrices = [_expand_matrix(matrix, block_sizes, 1) for matrix in matrices]
    product = functools.reduce(lambda x, y: x @ y, expanded_matrices)
    product.data[:] = 1
    ref = product.toarray()[
        block_sizes[0] : block_sizes[0] + int(sum(block_sizes)),
        block_sizes[0] : block_sizes[0] + int(sum(block_sizes)),
    ]

    local_blocks, _ = get_section_sizes(len(block_sizes), comm.block.size)
    start_block = sum(local_blocks[: comm.block.rank])
    end_block = start_block + local_blocks[comm.block.rank]

    rows, cols = product_sparsity_pattern_dsdbsparse(
        *dsdbsparse_matrices,
        in_num_diag=3,
        start_block=start_block,
        end_block=end_block,
        spillover=True,
    )
    val = sparse.coo_matrix((xp.ones(len(rows)), (rows, cols)), shape=shape).toarray()

    # Each rank in the block communicator computes its own local sparsity pattern.
    full = sum(comm.block._mpi_comm.allgather(val))

    print(f"{comm.block.rank=}, {start_block=}, {end_block=}", flush=True)
    if comm.rank == 0:
        print(xp.nonzero(ref - full))

    assert xp.allclose(ref, full)


if __name__ == "__main__":
    pytest.main([__file__])
