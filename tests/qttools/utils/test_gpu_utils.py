import numpy as np
import pytest

from qttools import xp
from qttools.utils.gpu_utils import (
    empty_like_pinned,
    empty_pinned,
    zeros_like_pinned,
    zeros_pinned,
)


@pytest.mark.usefixtures("shape", "dtype", "order")
def test_empty_pinned(
    shape: int | tuple[int, ...],
    dtype: type | str,
    order: str = "C",
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
    order: str = "C",
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
    order: str = "C",
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
    order: str = "C",
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
