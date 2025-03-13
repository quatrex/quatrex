# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest

from qttools import xp
from qttools.utils.gpu_utils import (
    empty_like_pinned,
    empty_pinned,
    get_any_location,
    get_host,
    zeros_like_pinned,
    zeros_pinned,
)


@pytest.mark.usefixtures(
    "shape", "dtype", "order", "input_module", "output_module", "use_pinned_memory"
)
def test_get_any_location(
    shape: int | tuple[int, ...],
    dtype: type | str,
    order: str,
    input_module: str,
    output_module: str,
    use_pinned_memory: bool,
):
    """Test the empty_pinned function."""

    if xp.__name__ == "numpy" and input_module != "numpy" and output_module != "numpy":
        return

    if xp.__name__ == "numpy":
        arr = xp.array(shape, dtype=dtype, order=order)
        out = get_any_location(arr, output_module, use_pinned_memory=use_pinned_memory)
        assert xp.allclose(out, arr)
        return

    if input_module == "numpy":
        arr = np.array(shape, dtype=dtype, order=order)
    else:
        arr = xp.array(shape, dtype=dtype, order=order)

    out_ref = get_any_location(arr, output_module, use_pinned_memory=False)

    out = get_any_location(arr, output_module, use_pinned_memory=use_pinned_memory)

    xp.allclose(out_ref, out)

    # test that pinned memory was returned
    if output_module == "numpy" and output_module != input_module and use_pinned_memory:
        assert (
            xp.cuda.runtime.pointerGetAttributes(out.ctypes.data).type
            == xp.cuda.runtime.memoryTypeHost
        )
    # Would be cool to test that internally also pinned memory is used for h2d transfers
    # but this is not possible with the current implementation

    if output_module == input_module:
        assert out is arr

    out = get_host(out)
    arr = get_host(arr)
    assert np.allclose(out, arr)


@pytest.mark.usefixtures("shape", "dtype", "order")
def test_empty_pinned(
    shape: int | tuple[int, ...],
    dtype: type | str,
    order: str,
):
    """Test the empty_pinned function."""
    arr = empty_pinned(shape, dtype=dtype, order=order)
    assert arr.shape == shape
    assert arr.dtype == dtype
    assert arr.flags["C_CONTIGUOUS"] if order == "C" else arr.flags["F_CONTIGUOUS"]


@pytest.mark.usefixtures("shape", "dtype", "order")
def test_zeros_pinned(
    shape: int | tuple[int, ...],
    dtype: type | str,
    order: str,
):
    """Test the zeros_pinned function."""
    arr = zeros_pinned(shape, dtype=dtype, order=order)
    assert arr.shape == shape
    assert arr.dtype == dtype
    assert arr.flags["C_CONTIGUOUS"] if order == "C" else arr.flags["F_CONTIGUOUS"]
    assert xp.allclose(arr, np.zeros(shape, dtype=dtype))


@pytest.mark.usefixtures("shape", "dtype", "order")
def test_empty_like_pinned(
    shape: int | tuple[int, ...],
    dtype: type | str,
    order: str,
):
    """Test the empty_like_pinned function."""
    arr = xp.random.rand(*shape)
    arr_like = empty_like_pinned(arr, dtype=dtype, order=order, shape=shape)
    assert arr_like.shape == arr.shape
    assert arr_like.dtype == dtype
    assert (
        arr_like.flags["C_CONTIGUOUS"]
        if order == "C"
        else arr_like.flags["F_CONTIGUOUS"]
    )


@pytest.mark.usefixtures("shape", "dtype", "order")
def test_zeros_like_pinned(
    shape: int | tuple[int, ...],
    dtype: type | str,
    order: str,
):
    """Test the zeros_like_pinned function."""
    arr = xp.random.rand(*shape)
    arr_like = zeros_like_pinned(arr, dtype=dtype, order=order, shape=shape)
    assert arr_like.shape == arr.shape
    assert arr_like.dtype == dtype
    assert (
        arr_like.flags["C_CONTIGUOUS"]
        if order == "C"
        else arr_like.flags["F_CONTIGUOUS"]
    )
    assert xp.allclose(arr_like, np.zeros(shape, dtype=dtype))
