from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBSparse, btd_matmul, btd_sandwich, bd_matmul, bd_sandwich


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