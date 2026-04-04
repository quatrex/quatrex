# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import cupy as cp
import numpy as np

from qttools import NDArray

BLOCK_SIZE = 256

cuda_source = f"""
#include <cupy/complex.cuh>

template<int T>
__global__
void _compress(unsigned char* out, const complex<double>* inp, const size_t N) {{
    static const int num_bytes = T / 8;
    __shared__ unsigned char s_out[{BLOCK_SIZE} * num_bytes * 2];

    size_t tile = blockIdx.x;
    size_t idx = tile * {BLOCK_SIZE} + threadIdx.x;

    for (size_t i = 0; i < num_bytes * 2; ++i) {{
        s_out[threadIdx.x * num_bytes * 2 + i] = 0;
    }}

    if (idx < N) {{
        complex<double> r_inp = inp[idx];
        double vals[2] = {{r_inp.real(), r_inp.imag()}};
        unsigned long long packed_vals[2];

        for(int v=0; v<2; ++v) {{
            unsigned long long bits_64 = __double_as_longlong(vals[v]);
            unsigned long long sign = (bits_64 >> 63) & 0x1ULL;
            unsigned long long exp_64 = (bits_64 >> 52) & 0x7FFULL;
            unsigned long long mant_64 = bits_64 & 0xFFFFFFFFFFFFFULL;

            const int num_exponent_bits = 9; 
            const int num_mantissa_bits = T - num_exponent_bits;
            const int shift = 52 - num_mantissa_bits;
            
            unsigned int exp_f;
            unsigned long long mant_trunc = mant_64 >> shift;

            if (exp_64 == 0x7FFULL) {{
                exp_f = 0xFF;
                if (mant_64 != 0) {{
                    mant_trunc = (1ULL << (num_mantissa_bits - 1));
                }}
            }} else {{
                int unbiased_exp = (int)exp_64 - 1023;
                int biased_f = unbiased_exp + 127;

                if (biased_f <= 0) {{
                    exp_f = 0;
                    mant_trunc = 0;
                }} else if (biased_f >= 0xFF) {{
                    exp_f = 0xFF;
                    mant_trunc = 0;
                }} else {{
                    exp_f = (unsigned int)biased_f;
                    // Round to Nearest Even
                    unsigned long long guard_bit = 1ULL << (shift - 1);
                    unsigned long long sticky_mask = guard_bit - 1;
                    if ((bits_64 & guard_bit) && ((bits_64 & sticky_mask) || (mant_trunc & 1))) {{
                        mant_trunc++;
                        if (mant_trunc >= (1ULL << num_mantissa_bits)) {{
                            mant_trunc = 0; exp_f++;
                            if (exp_f >= 0xFF) {{ exp_f = 0xFF; mant_trunc = 0; }}
                        }}
                    }}
                }}
            }}
            unsigned int sgn_exp = (static_cast<unsigned int>(sign) << 8) | (exp_f & 0xFF);
            packed_vals[v] = ((unsigned long long)sgn_exp << num_mantissa_bits) | (mant_trunc & ((1ULL << num_mantissa_bits) - 1));
        }}

        for (size_t i = 0; i < num_bytes; i++) {{
            s_out[threadIdx.x * num_bytes * 2 + i] = (packed_vals[0] >> (8 * i)) & 0xFF;
            s_out[threadIdx.x * num_bytes * 2 + num_bytes + i] = (packed_vals[1] >> (8 * i)) & 0xFF;
        }}
    }}
    __syncthreads();

    size_t start_idx = tile * {BLOCK_SIZE} * num_bytes * 2;
    size_t end_idx = min(N * num_bytes * 2, start_idx + {BLOCK_SIZE} * num_bytes * 2);
    for (size_t i = start_idx + threadIdx.x; i < end_idx; i += {BLOCK_SIZE}) {{
        out[i] = s_out[i - start_idx];
    }}
}}

template<int T>
__global__
void _decompress(complex<double>* out, const unsigned char* inp, const size_t N) {{
    static const int num_bytes = T / 8;
    __shared__ unsigned char s_inp[{BLOCK_SIZE} * num_bytes * 2];

    size_t tile = blockIdx.x;
    size_t idx = tile * {BLOCK_SIZE} + threadIdx.x;
    size_t start_idx = tile * {BLOCK_SIZE} * num_bytes * 2;
    size_t end_idx = min(N * num_bytes * 2, start_idx + {BLOCK_SIZE} * num_bytes * 2);
    
    for (size_t i = threadIdx.x; (start_idx + i) < end_idx; i += {BLOCK_SIZE}) {{
        s_inp[i] = inp[start_idx + i];
    }}
    __syncthreads();

    if (idx < N) {{
        auto unpack = [&](size_t offset) -> double {{
            unsigned long long packed = 0;
            size_t base_s_idx = threadIdx.x * num_bytes * 2 + offset;
            for (size_t i = 0; i < num_bytes; i++) {{
                packed |= (unsigned long long)s_inp[base_s_idx + i] << (8 * i);
            }}

            const int num_exponent_bits = 9;
            const int num_mantissa_bits = T - num_exponent_bits;
            const int shift = 52 - num_mantissa_bits;

            unsigned int sgn_exp_f = (unsigned int)(packed >> num_mantissa_bits);
            unsigned long long sign = (sgn_exp_f >> 8) & 1;
            unsigned long long exp_f = sgn_exp_f & 0xFF;
            unsigned long long mant_bits = packed & ((1ULL << num_mantissa_bits) - 1);

            unsigned long long exp_d;
            if (exp_f == 0xFF) {{
                exp_d = 0x7FF;
                if (mant_bits != 0) {{
                    mant_bits = (1ULL << (num_mantissa_bits - 1));
                }} else {{
                    mant_bits = 0;
                }}
            }}
            else{{
                if (exp_f == 0) exp_d = 0;
                //else if (exp_f == 0xFF) exp_d = 0x7FF;
                else exp_d = exp_f + (1023 - 127);
            }}
            // Shift bits back to the high-order position of the 52-bit mantissa
            unsigned long long mant_d = mant_bits << shift;
            unsigned long long res = (sign << 63) | (exp_d << 52) | mant_d;
            return __longlong_as_double(res);
        }};
        out[idx] = complex<double>(unpack(0), unpack(num_bytes));
    }}
}}

extern "C" {{
    __global__ void _compress_fp16(unsigned char* out, const complex<double>* inp, const size_t N) {{ _compress<16>(out, inp, N); }}
    __global__ void _compress_fp24(unsigned char* out, const complex<double>* inp, const size_t N) {{ _compress<24>(out, inp, N); }}
    __global__ void _compress_fp32(unsigned char* out, const complex<double>* inp, const size_t N) {{ _compress<32>(out, inp, N); }}
    __global__ void _compress_fp40(unsigned char* out, const complex<double>* inp, const size_t N) {{ _compress<40>(out, inp, N); }}
    __global__ void _compress_fp48(unsigned char* out, const complex<double>* inp, const size_t N) {{ _compress<48>(out, inp, N); }}
    __global__ void _compress_fp56(unsigned char* out, const complex<double>* inp, const size_t N) {{ _compress<56>(out, inp, N); }}

    __global__ void _decompress_fp16(complex<double>* out, const unsigned char* inp, const size_t N) {{ _decompress<16>(out, inp, N); }}
    __global__ void _decompress_fp24(complex<double>* out, const unsigned char* inp, const size_t N) {{ _decompress<24>(out, inp, N); }}
    __global__ void _decompress_fp32(complex<double>* out, const unsigned char* inp, const size_t N) {{ _decompress<32>(out, inp, N); }}
    __global__ void _decompress_fp40(complex<double>* out, const unsigned char* inp, const size_t N) {{ _decompress<40>(out, inp, N); }}
    __global__ void _decompress_fp48(complex<double>* out, const unsigned char* inp, const size_t N) {{ _decompress<48>(out, inp, N); }}
    __global__ void _decompress_fp56(complex<double>* out, const unsigned char* inp, const size_t N) {{ _decompress<56>(out, inp, N); }}
}}
"""

module = cp.RawModule(code=cuda_source, options=("--std=c++17",))

_kernels = {
    "compress": {
        b: module.get_function(f"_compress_fp{b}") for b in [16, 24, 32, 40, 48, 56]
    },
    "decompress": {
        b: module.get_function(f"_decompress_fp{b}") for b in [16, 24, 32, 40, 48, 56]
    },
}


def compress(inp: NDArray, bits: int, out: NDArray | None = None) -> NDArray:
    """Compresses complex128 data to a custom floating point format.
    It is specified by the number of bits where
    1 bit is for the sign, 8 bits are for the exponent (same as fp32) and
    the rest of the bits are for the mantissa (taken from fp64).

    Parameters
    ----------
    inp : NDArray
        Input array of complex128 values to be compressed.
    bits : int
        Number of bits for the custom floating point format (e.g., 16, 24
        32, 40, 48).
    out : NDArray, optional
        Pre-allocated output array to store the compressed data. If None, a new array will
        be created. The shape of the output array will be inp.shape + (2*(bits // 8),).
        The dtype of the output array must be cp.uint8.

    Returns
    -------
    NDArray
        The compressed data as an array of unsigned bytes.

    """

    # check input is complex128
    if inp.dtype != cp.complex128:
        raise ValueError(
            f"Input array must have dtype cp.complex128 but got {inp.dtype}."
        )

    if bits not in _kernels["compress"].keys():
        raise ValueError(
            f"Unsupported bit width: {bits}. Supported values are {list(_kernels['compress'].keys())}."
        )

    inp = cp.ascontiguousarray(inp)

    N = np.prod(inp.shape)

    if out is None:
        out = cp.empty(inp.shape + (2 * (bits // 8),), dtype=cp.uint8)
    else:
        if out.shape != inp.shape + (2 * (bits // 8),):
            raise ValueError(
                f"Output array must have shape {inp.shape + (2*(bits // 8),)} but got {out.shape}."
            )
        if out.dtype != cp.uint8:
            raise ValueError(
                f"Output array must have dtype cp.uint8 but got {out.dtype}."
            )

    # check if the output array is contiguous
    if not out.flags["C_CONTIGUOUS"]:
        _out = cp.empty(out.shape, dtype=cp.uint8)
    else:
        _out = out

    if not _out.flags["C_CONTIGUOUS"]:
        raise ValueError("Must be contiguous")

    if not inp.flags["C_CONTIGUOUS"]:
        raise ValueError("Must be contiguous")

    _kernels["compress"][bits](
        ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,), (BLOCK_SIZE,), (_out, inp, N)
    )

    if not out.flags["C_CONTIGUOUS"]:
        out[:] = _out

    return out


def decompress(inp: NDArray, bits: int, out: NDArray | None = None) -> NDArray:
    """Decompresses data from a custom floating point format to complex128.
    The custom floating point format is specified by the number of bits where
    1 bit is for the sign, 8 bits are for the exponent (same as fp32) and
    the rest of the bits are for the mantissa (taken from fp64).

    Parameters
    ----------
    inp : NDArray
        Input array of unsigned bytes representing the compressed data.
        The shape of the input array should be (..., 2*(bits // 8))
        where bits is the bit width of the custom floating point format.
    bits : int
        Number of bits for the custom floating point format (e.g., 16, 24, 32, 40, 48).
    out : NDArray, optional
        Pre-allocated output array to store the decompressed complex128 data.
        If None, a new array will be created.
        The shape of the output array will be inp.shape[:-1] and dtype will be cp.complex128.

    Returns
    -------
    NDArray
        The decompressed data as an array of complex128 values.

    """

    if bits not in _kernels["decompress"].keys():
        raise ValueError(
            f"Unsupported bit width: {bits}. Supported values are {list(_kernels['decompress'].keys())}."
        )

    if inp.dtype != cp.uint8:
        raise ValueError(f"Input array must have dtype cp.uint8 but got {inp.dtype}.")

    inp = cp.ascontiguousarray(inp)

    N = np.prod(inp.shape[:-1])

    if out is None:
        out = cp.empty(inp.shape[:-1], dtype=cp.complex128)
    else:
        if out.shape != inp.shape[:-1]:
            raise ValueError(
                f"Output array must have shape {inp.shape[:-1]} but got {out.shape}."
            )
        if out.dtype != cp.complex128:
            raise ValueError(
                f"Output array must have dtype cp.complex128 but got {out.dtype}."
            )

    if inp.shape[-1] != 2 * (bits // 8):
        raise ValueError(
            f"Last dimension of input array must be {2*(bits // 8)} but got {inp.shape[-1]}."
        )

    if not out.flags["C_CONTIGUOUS"]:
        raise ValueError("Must be contiguous")

    if not inp.flags["C_CONTIGUOUS"]:
        raise ValueError("Must be contiguous")

    _kernels["decompress"][bits](
        ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,), (BLOCK_SIZE,), (out, inp, N)
    )

    return out
