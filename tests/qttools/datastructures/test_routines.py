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


def _create_bd_coo(sizes: NDArray, num_diag) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    offsets = xp.hstack(([0], xp.cumsum(sizes)))

    arr = xp.zeros((size, size), dtype=xp.complex128)
    num_diag = min(num_diag, len(sizes))
    for i in range(len(sizes)):
        # Diagonal block.
        block_shape = (int(sizes[i]), int(sizes[i]))
        arr[offsets[i] : offsets[i + 1], offsets[i] : offsets[i + 1]] = xp.random.rand(
            *block_shape
        ) + 1j * xp.random.rand(*block_shape)
        for idiag in range(1, num_diag // 2 + 1):
            # Superdiagonal block.
            if i < len(sizes) - idiag:
                block_shape = (int(sizes[i]), int(sizes[i + idiag]))
                arr[
                    offsets[i] : offsets[i + 1],
                    offsets[i + idiag] : offsets[i + idiag + 1],
                ] = xp.random.rand(*block_shape) + 1j * xp.random.rand(*block_shape)
                arr[
                    offsets[i + idiag] : offsets[i + idiag + 1],
                    offsets[i] : offsets[i + 1],
                ] = (
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
    num_diag: int,
):
    """Tests the block diagonal matmul of two DSBSparse matrix."""
    coo = _create_bd_coo(block_sizes, num_diag)
    dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
    dense = dsbsparse.to_dense()

    # number of diagonal blocks in product should be smaller than the total number of blocks
    num_blocks = len(block_sizes)
    num_diag = min(num_diag, num_blocks - 2)

    # dense blocks returned
    out = bd_matmul(
        dsbsparse,
        dsbsparse,
        out=None,
        in_num_diag=num_diag,
        out_num_diag=num_diag // 2 * 2 * 2 + 1,
    )
    out_dense = xp.zeros_like(dense)
    for row, col in out:
        out_dense[
            ...,
            dsbsparse.block_offsets[row] : dsbsparse.block_offsets[row + 1],
            dsbsparse.block_offsets[col] : dsbsparse.block_offsets[col + 1],
        ] = out[row, col]

    assert xp.allclose(dense @ dense, out_dense)

    # return with out
    # Initalize the output matrix with the correct sparsity pattern.
    out = dsbsparse_type.from_sparray(coo @ coo, block_sizes, global_stack_shape)

    bd_matmul(
        dsbsparse,
        dsbsparse,
        out,
        in_num_diag=num_diag,
        out_num_diag=num_diag // 2 * 2 * 2 + 1,
    )

    assert xp.allclose(dense @ dense, out.to_dense())


def test_bd_sandwich(
    dsbsparse_type: DSBSparse,
    block_sizes: NDArray,
    global_stack_shape: tuple,
    num_diag: int,
):
    """Tests the block diagonal matmul of three 'sandwiched' DSBSparse matrix."""
    coo = _create_btd_coo(block_sizes)
    dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
    dense = dsbsparse.to_dense()

    # number of diagonal blocks in product should be smaller than the total number of blocks
    num_blocks = len(block_sizes)
    num_diag = min(num_diag, num_blocks - 4)

    # dense blocks returned
    out = bd_sandwich(
        dsbsparse,
        dsbsparse,
        out=None,
        in_num_diag=num_diag,
        out_num_diag=num_diag // 2 * 3 * 2 + 1,
    )
    out_dense = xp.zeros_like(dense)
    for row, col in out:
        out_dense[
            ...,
            dsbsparse.block_offsets[row] : dsbsparse.block_offsets[row + 1],
            dsbsparse.block_offsets[col] : dsbsparse.block_offsets[col + 1],
        ] = out[row, col]

    assert xp.allclose(dense @ dense @ dense, out_dense)

    # return with out
    # Initalize the output matrix with the correct sparsity pattern.
    out = dsbsparse_type.from_sparray(coo @ coo @ coo, block_sizes, global_stack_shape)

    bd_sandwich(
        dsbsparse, dsbsparse, out, in_num_diag=num_diag, out_num_diag=num_diag + 4
    )

    assert xp.allclose(dense @ dense @ dense, out.to_dense())


def test_bd_matmul_spillover(
    dsbsparse_type: DSBSparse,
    block_sizes: NDArray,
    global_stack_shape: tuple,
    num_diag: int,
):
    """Tests the block diagonal matmul of two DSBSparse matrix with spillover correction."""
    coo = _create_bd_coo(block_sizes, num_diag)
    dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
    dense = dsbsparse.to_dense()
    dense_shape = list(dense.shape)
    NBC = num_diag // 2
    left_obc = int((block_sizes[0]))
    right_obc = int((block_sizes[-1]))
    dense_shape[-2] += left_obc * NBC + right_obc * NBC
    dense_shape[-1] += left_obc * NBC + right_obc * NBC
    dense_exp = xp.zeros(tuple(dense_shape), dtype=dense.dtype)
    # copy matrix into the center
    size = sum(block_sizes)
    exp_size = size + left_obc * NBC + right_obc * NBC
    dense_exp[
        ...,
        left_obc * NBC : left_obc * NBC + size,
        left_obc * NBC : left_obc * NBC + size,
    ] = dense
    # simply repeat the boundaries slices
    for i in range(NBC):
        dense_exp[
            ..., i * left_obc : (i + 1) * left_obc, i * left_obc : (i * left_obc + size)
        ] = dense[..., :left_obc, :]
        dense_exp[
            ..., i * left_obc : (i * left_obc + size), i * left_obc : (i + 1) * left_obc
        ] = dense[..., :, :left_obc]
        dense_exp[
            ...,
            exp_size - (i + 1) * right_obc : exp_size - i * right_obc,
            (exp_size - i * right_obc - size) : (exp_size - i * right_obc),
        ] = dense[..., -right_obc:, :]
        dense_exp[
            ...,
            (exp_size - i * right_obc - size) : (exp_size - i * right_obc),
            (exp_size - (i + 1) * right_obc) : (exp_size - i * right_obc),
        ] = dense[..., :, -right_obc:]

    expended_product = dense_exp @ dense_exp
    ref = expended_product[
        ...,
        left_obc * NBC : left_obc * NBC + size,
        left_obc * NBC : left_obc * NBC + size,
    ]

    # Initalize the output matrix with the correct sparsity pattern.

    out = dsbsparse_type.from_sparray(
        sparse.coo_matrix(_get_last_2d(ref)), block_sizes, global_stack_shape
    )

    bd_matmul(
        dsbsparse,
        dsbsparse,
        out,
        in_num_diag=num_diag,
        out_num_diag=num_diag // 2 * 2 * 2 + 1,
        spillover_correction=True,
    )

    assert xp.allclose(ref, out.to_dense())


def test_bd_sandwich_spillover(
    dsbsparse_type: DSBSparse,
    block_sizes: NDArray,
    global_stack_shape: tuple,
    num_diag: int,
):
    """Tests the block diagonal matmul of three DSBSparse matrix with spillover correction."""
    coo = _create_bd_coo(block_sizes, num_diag)
    dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
    dense = dsbsparse.to_dense()
    dense_shape = list(dense.shape)
    NBC = num_diag // 2
    left_obc = int(block_sizes[0])
    right_obc = int(block_sizes[-1])
    dense_shape[-2] += left_obc * NBC + right_obc * NBC
    dense_shape[-1] += left_obc * NBC + right_obc * NBC
    size = sum(block_sizes)
    exp_size = size + left_obc * NBC + right_obc * NBC
    dense_exp = xp.zeros(tuple(dense_shape), dtype=dense.dtype)
    # copy data into the center
    dense_exp[
        ...,
        left_obc * NBC : left_obc * NBC + size,
        left_obc * NBC : left_obc * NBC + size,
    ] = dense
    # simply repeat the boundaries slices
    for i in range(NBC):
        dense_exp[
            ..., i * left_obc : (i + 1) * left_obc, i * left_obc : (i * left_obc + size)
        ] = dense[..., :left_obc, :]
        dense_exp[
            ..., i * left_obc : (i * left_obc + size), i * left_obc : (i + 1) * left_obc
        ] = dense[..., :, :left_obc]
        dense_exp[
            ...,
            exp_size - (i + 1) * right_obc : exp_size - i * right_obc,
            (exp_size - i * right_obc - size) : (exp_size - i * right_obc),
        ] = dense[..., -right_obc:, :]
        dense_exp[
            ...,
            (exp_size - i * right_obc - size) : (exp_size - i * right_obc),
            (exp_size - (i + 1) * right_obc) : (exp_size - i * right_obc),
        ] = dense[..., :, -right_obc:]

    expended_product = dense_exp @ dense_exp @ dense_exp
    ref = expended_product[
        ...,
        left_obc * NBC : left_obc * NBC + size,
        left_obc * NBC : left_obc * NBC + size,
    ]

    # Initalize the output matrix with the correct sparsity pattern.

    out = dsbsparse_type.from_sparray(
        sparse.coo_matrix(_get_last_2d(ref)), block_sizes, global_stack_shape
    )

    bd_sandwich(
        dsbsparse,
        dsbsparse,
        out,
        in_num_diag=num_diag,
        out_num_diag=num_diag // 2 * 3 * 2 + 1,
        spillover_correction=True,
    )

    assert xp.allclose(ref, out.to_dense())


def _get_last_2d(x):
    m, n = x.shape[-2:]
    return x.flat[: m * n].reshape(m, n)
