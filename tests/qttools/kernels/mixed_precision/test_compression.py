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


@pytest.mark.parametrize("bits", [16, 24, 32, 40, 48])
@pytest.mark.usefixtures("n", "batch_shape")
@pytest.mark.skipif(xp.__name__ != "cupy", reason="Requires cupy backend")
def test_overflow_to_inf(n: int, batch_shape: tuple[int, ...], bits: int):
    large_val = 1.5e39
    A_ref = xp.full((*batch_shape, n), large_val + 1j * large_val, dtype=xp.complex128)

    compressed = compress(A_ref, bits)
    A_res = decompress(compressed, bits)

    assert xp.all(xp.isinf(A_res.real))
    assert xp.all(xp.isinf(A_res.imag))
    assert not xp.any(xp.isnan(A_res))


@pytest.mark.parametrize("bits", [16, 24, 32, 40, 48])
@pytest.mark.usefixtures("n", "batch_shape")
@pytest.mark.skipif(xp.__name__ != "cupy", reason="Requires cupy backend")
def test_underflow_to_zero(n: int, batch_shape: tuple[int, ...], bits: int):
    small_val = 1e-45
    A_ref = xp.full((*batch_shape, n), small_val + 1j * small_val, dtype=xp.complex128)

    compressed = compress(A_ref, bits)
    A_res = decompress(compressed, bits)

    assert xp.all(A_res == 0)


@pytest.mark.parametrize("bits", [16, 24, 32, 40, 48])
@pytest.mark.skipif(xp.__name__ != "cupy", reason="Requires cupy backend")
def test_rounding_to_even(bits: int):
    num_mantissa = bits - 9
    lsb_value = 2 ** (-num_mantissa)

    val_down = 1.0 + (0.5 * lsb_value)
    val_up = 1.0 + (1.5 * lsb_value)

    A_ref = xp.array([val_down + 1j * val_up], dtype=xp.complex128)

    compressed = compress(A_ref, bits)
    A_res = decompress(compressed, bits)

    assert float(A_res[0].real) == 1.0
    assert float(A_res[0].imag) == 1.0 + (2.0 * lsb_value)


@pytest.mark.parametrize("bits", [16, 24, 32, 40, 48])
@pytest.mark.skipif(xp.__name__ != "cupy", reason="Requires cupy backend")
def test_nan_preservation(bits: int):
    A_ref = xp.array([xp.nan + 1j * xp.nan], dtype=xp.complex128)
    compressed = compress(A_ref, bits)
    A_res = decompress(compressed, bits)

    print(A_ref, compressed, A_res)

    assert xp.all(xp.isnan(A_res))


@pytest.mark.parametrize("out_flag", [True, False])
@pytest.mark.usefixtures("n", "batch_shape")
@pytest.mark.skipif(xp.__name__ != "cupy", reason="Requires cupy backend")
def test_float32(n: int, batch_shape: tuple[int, ...], out_flag: bool):

    bits = 32
    batch_shape = list(batch_shape)
    batch_shape[0] += 13
    rng = xp.random.default_rng()
    A_ref = rng.random((*batch_shape, n)) + 1j * rng.random((*batch_shape, n))
    A_ref = A_ref.astype(xp.complex128)
    A_ref[0] = xp.nan + 1j * xp.nan
    A_ref[1] = xp.float64(1.0e-40) + 1j * xp.float64(1.0e-40)
    A_ref[2] = xp.inf + 1j * xp.inf
    A_ref[3] = xp.float64(0.0) + 1j * xp.float64(0.0)
    A_ref[4] = xp.float64(-0.0) + 1j * xp.float64(-0.0)
    A_ref[5] = xp.float64(-1.0e-40) + 1j * xp.float64(-1.0e-40)
    neg_nan_bits = xp.array([0xFFC00000], dtype=xp.uint32).view(xp.float32)
    A_ref[6] = xp.float64(neg_nan_bits) + 1j * xp.float64(neg_nan_bits)
    A_ref[7] = xp.float64(-xp.inf) + 1j * xp.float64(-xp.inf)
    A_ref[8] = xp.float64(1.401298464324817e-45) + 1j * xp.float64(
        1.401298464324817e-45
    )
    smallest_normal = xp.float64(1.1754943508222875e-38)
    largest_subnormal = xp.float64(1.1754942106924411e-38)
    A_ref[9] = smallest_normal + 1j * largest_subnormal
    max_f32 = xp.float64(xp.finfo(xp.float32).max)
    A_ref[10] = max_f32 + 1j * -max_f32
    A_ref[11] = xp.nan + 1j * 0.0
    A_ref[12] = xp.inf + 1j * -1.234e-40

    if out_flag:
        out = xp.empty(A_ref.shape + (2 * (bits // 8),), dtype=xp.uint8)
    else:
        out = None

    out = compress(A_ref, bits, out=out)

    A_test = A_ref.astype(xp.complex64).copy()
    A_test = A_test.view(xp.uint8).reshape(A_ref.shape + (2 * (bits // 8),))

    # check for bitwise equality of compressed data
    assert xp.array_equal(out, A_test)

    if out_flag:
        A = xp.empty_like(A_ref)
    else:
        A = None
    out = compress(A_ref, bits, out=out)
    A = decompress(out, bits, out=A)
    A_test = A_ref.astype(xp.complex64).astype(xp.complex128).copy()

    assert xp.array_equal(xp.isnan(A), xp.isnan(A_test))

    for i in range(batch_shape[0]):
        mask = ~xp.isnan(A_test[i]) & ~xp.isnan(A[i])
        if xp.any(mask):
            assert xp.array_equal(A[i][mask], A_test[i][mask])
