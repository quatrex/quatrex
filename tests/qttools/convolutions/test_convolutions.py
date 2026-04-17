# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import numpy as np

from qttools import NDArray, xp
from qttools.fft import (
    fft_circular_convolve,
    fft_convolve,
    fft_convolve_kpoints,
    fft_correlate_kpoints,
)
from qttools.fft.convolve import _naive_convolve


def naive_correlate(a: NDArray, b: NDArray) -> NDArray:
    """Naive implementation of correlation for testing purposes."""
    ne_a = a.shape[0]
    ne_b = b.shape[0]
    ne = ne_a + ne_b - 1
    result_shape = (ne,) + a.shape[1:]
    result = xp.zeros(result_shape, dtype=a.dtype)
    for i in range(ne_a):
        for j in range(ne_b):
            # Second array is reversed for correlation
            # It is important that it is the second array that is reversed
            result[i + j] += a[i] * b[ne_b - j - 1]
    return result


def naive_circular_convolve(a: NDArray, b: NDArray, axes: tuple[int]) -> NDArray:
    """Naive implementation of circular convolution for testing purposes."""
    result = xp.zeros_like(a)
    shape = tuple(a.shape[i] for i in axes)
    # Precompute mapping from full-dim -> position in axes
    axis_pos = {axis: pos for pos, axis in enumerate(axes)}
    for idx_a in xp.ndindex(*shape):
        i_a = tuple(
            (idx_a[axis_pos[i]] if i in axis_pos else slice(None))
            for i in range(a.ndim)
        )
        for idx_b in xp.ndindex(*shape):
            idx = tuple((idx_a[i] + idx_b[i]) % shape[i] for i in range(len(shape)))
            i = tuple(
                (idx[axis_pos[i]] if i in axis_pos else slice(None))
                for i in range(a.ndim)
            )
            i_b = tuple(
                (idx_b[axis_pos[i]] if i in axis_pos else slice(None))
                for i in range(a.ndim)
            )
            result[i] += a[i_a] * b[i_b]
    return result


def naive_circular_correlate(a: NDArray, b: NDArray, axes: tuple[int]) -> NDArray:
    """Naive implementation of circular correlation for testing purposes."""
    result = xp.zeros_like(a)
    shape = tuple(a.shape[i] for i in axes)
    # Precompute mapping from full-dim -> position in axes
    axis_pos = {axis: pos for pos, axis in enumerate(axes)}
    for idx_a in xp.ndindex(*shape):
        i_a = tuple(
            (idx_a[axis_pos[i]] if i in axis_pos else slice(None))
            for i in range(a.ndim)
        )
        for idx_b in xp.ndindex(*shape):
            idx = tuple((idx_a[i] - idx_b[i]) % shape[i] for i in range(len(shape)))
            i = tuple(
                (idx[axis_pos[i]] if i in axis_pos else slice(None))
                for i in range(a.ndim)
            )
            i_b = tuple(
                (idx_b[axis_pos[i]] if i in axis_pos else slice(None))
                for i in range(a.ndim)
            )
            result[i] += a[i_a] * b[i_b]
    return result


def naive_convolve_kpoints(a: NDArray, b: NDArray) -> NDArray:
    """Naive implementation of convolution over energy axis and circular over k-points."""
    ne_a = a.shape[0]
    ne_b = b.shape[0]
    ne = ne_a + ne_b - 1
    nk = a.shape[1:-1]
    result_shape = (ne,) + a.shape[1:]
    result = xp.zeros(result_shape, dtype=a.dtype)
    for i in range(ne_a):
        for j in range(ne_b):
            for k1 in xp.ndindex(*nk):
                for k2 in xp.ndindex(*nk):
                    k = tuple((k1[d] + k2[d]) % nk[d] for d in range(len(nk)))
                    result[i + j, *k] += a[i, *k1] * b[j, *k2]
    return result


def naive_correlate_kpoints(a: NDArray, b: NDArray) -> NDArray:
    """Naive implementation of correlation over energy axis and circular over k-points."""
    ne_a = a.shape[0]
    ne_b = b.shape[0]
    ne = ne_a + ne_b - 1
    nk = a.shape[1:-1]
    result_shape = (ne,) + a.shape[1:]
    result = xp.zeros(result_shape, dtype=a.dtype)
    for i in range(ne_a):
        for j in range(ne_b):
            for k1 in xp.ndindex(nk):
                for k2 in xp.ndindex(nk):
                    k = tuple((k1[d] - k2[d]) % nk[d] for d in range(len(nk)))
                    # Important: second array is reversed for correlation
                    result[i + j, *k] += a[i, *k1] * b[ne_b - j - 1, *k2]
    return result


class TestNaiveConvolutions:
    """Tests for the naive convolution implementations used for
    testing purposes, ie tests for the testing functions."""

    @staticmethod
    def test_naive_convolve():
        a = xp.array([1, 2, 3], dtype=xp.float64)
        b = xp.array([0, 1, 0.5], dtype=xp.float64)
        result = _naive_convolve(a, b)
        # NOTE: No -1 length here, needed for correct hilbert transform later on.
        expected = xp.array([0, 1, 2.5, 4, 1.5, 0], dtype=xp.float64)
        assert xp.allclose(result, expected)
        na = a.shape[0]
        nb = b.shape[0]
        # NOTE: No na+nb-1 length here, needed for correct hilbert transform later on.
        expected_fft = xp.fft.ifft(xp.fft.fft(a, n=na + nb) * xp.fft.fft(b, n=na + nb))
        assert xp.allclose(result, expected_fft)

    @staticmethod
    def test_naive_correlate():
        a = xp.array([1, 2, 3], dtype=xp.float64)
        b = xp.array([0, 1, 0.5], dtype=xp.float64)
        result = naive_correlate(a, b)
        expected = xp.array([0.5, 2, 3.5, 3, 0], dtype=xp.float64)
        assert xp.allclose(result, expected)
        na = a.shape[0]
        nb = b.shape[0]
        expected_fft = xp.fft.ifft(
            xp.fft.fft(a, n=na + nb - 1) * xp.fft.fft(b[::-1], n=na + nb - 1)
        )
        assert xp.allclose(result, expected_fft)

    @staticmethod
    def test_naive_circular_convolve():
        # This is interesting, the 0th element is 1*0 + 2*0.5 + 3*2 = 7, not
        # 1*0.5 + 2*2 + 3*0 = 4.5 which I would expect for the 0-shift.
        # So if we would convolve with two (-2,-1,0,1,2) ordered arrays,
        # we would get (1,2,-2,-1,0) shift ordering in the result.
        # But fear not, the fft is only used for the calculation for the
        # self-energy, and convolving with the (-2,-1,0,1,2) ordering with
        # the (0,1,2,-2,-1) ordering of W gives the correct (-2,-1,0,1,2) result.
        a = xp.array([1, 2, 3], dtype=xp.float64)
        b = xp.array([0, 2, 0.5], dtype=xp.float64)
        result = naive_circular_convolve(a, b, axes=(0,))
        expected = xp.array([7, 3.5, 4.5], dtype=xp.float64)
        assert xp.allclose(result, expected)
        expected_fft = xp.fft.ifft(xp.fft.fft(a) * xp.fft.fft(b))
        assert xp.allclose(result, expected_fft)

    @staticmethod
    def test_naive_circular_correlate():
        # NOTE: Here the result is as expected, the 0th (no shift) element
        # is 1*0+ 2*2 + 3*0.5 = 5.5. We have to roll the fft by one to get this.
        a = xp.array([1, 2, 3], dtype=xp.complex128)
        b = xp.array([0, 2, 0.5], dtype=xp.complex128)
        result = naive_circular_correlate(a, b, axes=(0,))
        expected = xp.array([5.5, 6.5, 3], dtype=xp.complex128)
        assert xp.allclose(result, expected)
        expected_fft = xp.fft.ifft(xp.fft.fft(a) * (xp.fft.fft(b[::-1])))
        expected_fft = xp.roll(expected_fft, shift=1)
        assert xp.allclose(result, expected_fft)

    @staticmethod
    def test_naive_convolve_kpoints():
        a = xp.array([[1, 2], [3, 4]], dtype=xp.float64)
        b = xp.array([[0, 1], [0.5, 0]], dtype=xp.float64)
        # Add empty orbital dimensions
        a = a.reshape(a.shape + (1,))
        b = b.reshape(b.shape + (1,))
        result = naive_convolve_kpoints(a, b)
        expected = xp.array([[2, 1], [4.5, 4], [1.5, 2]], dtype=xp.float64)
        expected = expected.reshape(expected.shape + (1,))
        assert xp.allclose(result, expected)

    @staticmethod
    def test_naive_correlate_kpoints():
        a = xp.array([[1, 2], [3, 4]], dtype=xp.float64)
        b = xp.array([[0, 1], [0.5, 0]], dtype=xp.float64)
        # Add empty orbital dimensions
        a = a.reshape(a.shape + (1,))
        b = b.reshape(b.shape + (1,))
        result = naive_correlate_kpoints(a, b)
        expected = xp.array([[0.5, 1], [3.5, 3], [4, 3]], dtype=xp.float64)
        expected = expected.reshape(expected.shape + (1,))
        assert xp.allclose(result, expected)


class TestFFTConvolutions:
    def test_fft_convolve(self, array_shape):
        a = xp.random.random(array_shape)
        b = xp.random.random(array_shape)
        result = fft_convolve(a, b)
        # Compare with naive convolve
        expected = _naive_convolve(a, b)
        assert xp.allclose(result, expected)

    def test_fft_circular_convolve(self, array_shape):
        a = xp.random.random(array_shape)
        b = xp.random.random(array_shape)
        # Randomly choose axes to convolve over
        # TODO: Random axes is not a good idea, since the
        # tests are not reproducible and determinstic.
        size = np.random.choice(np.arange(1, len(array_shape) + 1))
        axes = tuple(
            sorted(np.random.choice(len(array_shape), size=size, replace=False))
        )
        result = fft_circular_convolve(a, b, axes)
        # Compare with naive circular convolve
        expected = naive_circular_convolve(a, b, axes)
        assert xp.allclose(result, expected)

    def test_fft_convolve_kpoints(self, array_shape):
        a = xp.random.random(array_shape)
        b = xp.random.random(array_shape)
        result = fft_convolve_kpoints(a, b)
        # Compare with naive convolve
        expected = naive_convolve_kpoints(a, b)
        assert xp.allclose(result, expected)

    def test_fft_correlate_kpoints(self, array_shape):
        a = xp.random.random(array_shape)
        b = xp.random.random(array_shape)
        # Add empty orbital dimensions
        a = a.reshape(a.shape + (1,))
        b = b.reshape(b.shape + (1,))
        result = fft_correlate_kpoints(a, b)
        # Compare with naive correlate
        expected = naive_correlate_kpoints(a, b)
        assert xp.allclose(result, expected)
