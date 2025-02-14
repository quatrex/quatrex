# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, sparse, xp
from qttools.datastructures import (
    DSBSparse,
    bd_matmul,
    bd_sandwich,
    btd_matmul,
    btd_sandwich,
)


def _create_btd_coo(sizes: NDArray) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    offsets = xp.hstack(([0], xp.cumsum(sizes)))

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


def test_btd_matmul(
    dsbsparse_type: DSBSparse,
    block_sizes: NDArray,
    global_stack_shape: tuple,
):
    """Tests the in-place addition of a DSBSparse matrix."""
    coo = _create_btd_coo(block_sizes)
    dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
    dense = dsbsparse.to_dense()

    # Initalize the output matrix with the correct sparsity pattern.

    out = dsbsparse_type.from_sparray(coo @ coo, block_sizes, global_stack_shape)

    btd_matmul(dsbsparse, dsbsparse, out)

    assert xp.allclose(dense @ dense, out.to_dense())


def test_btd_sandwich(
    dsbsparse_type: DSBSparse,
    block_sizes: NDArray,
    global_stack_shape: tuple,
):
    """Tests the in-place addition of a DSBSparse matrix."""
    coo = _create_btd_coo(block_sizes)
    dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
    dense = dsbsparse.to_dense()

    # Initalize the output matrix with the correct sparsity pattern.
    out = dsbsparse_type.from_sparray(coo @ coo @ coo, block_sizes, global_stack_shape)

    btd_sandwich(dsbsparse, dsbsparse, out)

    assert xp.allclose(dense @ dense @ dense, out.to_dense())


def test_bd_matmul(
    dsbsparse_type: DSBSparse,
    block_sizes: NDArray,
    global_stack_shape: tuple,
):
    """Tests the in-place addition of a DSBSparse matrix."""
    coo = _create_btd_coo(block_sizes)
    dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
    dense = dsbsparse.to_dense()

    # Initalize the output matrix with the correct sparsity pattern.

    out = dsbsparse_type.from_sparray(coo @ coo, block_sizes, global_stack_shape)

    bd_matmul(dsbsparse, dsbsparse, out)

    assert xp.allclose(dense @ dense, out.to_dense())


def test_bd_sandwich(
    dsbsparse_type: DSBSparse,
    block_sizes: NDArray,
    global_stack_shape: tuple,
):
    """Tests the in-place addition of a DSBSparse matrix."""
    coo = _create_btd_coo(block_sizes)
    dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
    dense = dsbsparse.to_dense()

    # Initalize the output matrix with the correct sparsity pattern.
    out = dsbsparse_type.from_sparray(coo @ coo @ coo, block_sizes, global_stack_shape)

    bd_sandwich(dsbsparse, dsbsparse, out)

    assert xp.allclose(dense @ dense @ dense, out.to_dense())


def test_bd_matmul_spillover(
    dsbsparse_type: DSBSparse,
    block_sizes: NDArray,
    global_stack_shape: tuple,
):
    """Tests the in-place addition of a DSBSparse matrix."""
    coo = _create_btd_coo(block_sizes)
    dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
    dense = dsbsparse.to_dense()
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

    out = dsbsparse_type.from_sparray(
        sparse.coo_matrix(_get_last_2d(ref)), block_sizes, global_stack_shape
    )

    bd_matmul(dsbsparse, dsbsparse, out, spillover_correction=True)

    assert xp.allclose(ref, out.to_dense())


def test_bd_sandwich_spillover(
    dsbsparse_type: DSBSparse,
    block_sizes: NDArray,
    global_stack_shape: tuple,
):
    """Tests the in-place addition of a DSBSparse matrix."""
    coo = _create_btd_coo(block_sizes)
    dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
    dense = dsbsparse.to_dense()
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

    out = dsbsparse_type.from_sparray(
        sparse.coo_matrix(_get_last_2d(ref)), block_sizes, global_stack_shape
    )

    bd_sandwich(dsbsparse, dsbsparse, out, spillover_correction=True)

    assert xp.allclose(ref, out.to_dense())


def _get_last_2d(x):
    m, n = x.shape[-2:]
    return x.flat[: m * n].reshape(m, n)
