# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import QTX_USE_CUPY_JIT, xp

if xp.__name__ == "cupy":
    if QTX_USE_CUPY_JIT:
        from qttools.kernels.datastructure.cupy import _cupy_jit as cupy_backend
    else:
        from qttools.kernels.datastructure.cupy import _cupy_rawkernel as cupy_backend


@pytest.mark.parametrize("N", [1, 100, 10000])
@pytest.mark.parametrize("input_dtype", [xp.bool_, xp.int32, xp.int64])
def test_reduction(N: int, input_dtype: xp.dtype):
    """Tests the reduction kernel for correctness."""

    if xp.__name__ == "numpy":
        pytest.skip("Skipping test since numpy is used.")

    if QTX_USE_CUPY_JIT:
        pytest.skip("Skipping test since cupy jit is used.")

    a = xp.random.randint(0, 100, size=N).astype(input_dtype)

    # Compute the reference result using numpy
    reference_result = xp.sum(a)
    result = cupy_backend.reduction(a)

    assert result == reference_result
