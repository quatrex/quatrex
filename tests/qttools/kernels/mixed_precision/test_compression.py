# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import xp
from qttools.kernels.mixed_precision import compress, decompress


@pytest.mark.parametrize("bits", [16, 24, 32, 40, 48])
@pytest.mark.parametrize("out_flag", [True, False])
@pytest.mark.usefixtures("n", "batch_shape")
@pytest.mark.skipif(xp.__name__ != "cupy", reason="Requires cupy backend")
def test_compression(n: int, batch_shape: tuple[int, ...], out_flag: bool, bits: int):

    rng = xp.random.default_rng()
    A_ref = rng.random((*batch_shape, n)) + 1j * rng.random((*batch_shape, n))
    A_ref = A_ref.astype(xp.complex128)

    if out_flag:
        out = xp.empty(A_ref.shape + (2 * (bits // 8),), dtype=xp.uint8)
    else:
        out = None

    if out_flag:
        A = xp.empty_like(A_ref)
    else:
        A = None

    out = compress(A_ref, bits, out=out)
    A = decompress(out, bits, out=A)

    # determine tolerance from number of bits
    tol = 10 * 2 ** (-(bits - 9))
    assert (xp.linalg.norm(A_ref - A) / xp.linalg.norm(A_ref)) < tol


@pytest.mark.parametrize("bits", [16, 24, 32, 40, 48])
@pytest.mark.parametrize("out_flag", [True, False])
@pytest.mark.usefixtures("n", "batch_shape")
@pytest.mark.skipif(xp.__name__ != "cupy", reason="Requires cupy backend")
def test_zeroing(n: int, batch_shape: tuple[int, ...], out_flag: bool, bits: int):

    out = xp.empty((*batch_shape, n) + (2 * (bits // 8),), dtype=xp.uint8)

    out[:] = 0

    if out_flag:
        A = xp.empty((*batch_shape, n), dtype=xp.complex128)
    else:
        A = None

    A = decompress(out, bits, out=A)

    assert xp.allclose(A, xp.zeros_like(A))
