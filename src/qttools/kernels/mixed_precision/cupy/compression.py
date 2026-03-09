# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import cupy as cp
import numpy as np

from qttools import NDArray

BLOCK_SIZE = 256

cuda_source = f"""
#include <cupy/complex.cuh>

template<int T>
__global__
void _compress(unsigned char* out, const complex<double>* inp, const int N) {{
    static const int num_bytes = T / 8;
    __shared__ unsigned char s_out[{BLOCK_SIZE} * num_bytes * 2];

    int tiles = (N + {BLOCK_SIZE} - 1) / {BLOCK_SIZE};
    int tile = blockIdx.x;
    int idx = tile * {BLOCK_SIZE} + threadIdx.x;

    for (int i = 0; i < num_bytes * 2; ++i) {{
        s_out[threadIdx.x * num_bytes * 2 + i] = 0;
    }}


    if (idx < N) {{
        complex<double> r_inp = inp[idx];

        unsigned long long real_bits_64 = __double_as_longlong(r_inp.real());
        unsigned long long imag_bits_64 = __double_as_longlong(r_inp.imag());

        float real_float = static_cast<float>(r_inp.real());
        float imag_float = static_cast<float>(r_inp.imag());
        unsigned int real_bits = __float_as_int(real_float);
        unsigned int imag_bits = __float_as_int(imag_float);

        // T is given in bits
        unsigned char chars_real[num_bytes];
        unsigned char chars_imag[num_bytes];

        // get exponent bits from fp32
        int num_exponent_bits = 9; // fp32 has 8 exponent bits and 1 sign bit
        unsigned int real_sgn_exp = (real_bits >> 23) & 0x1FF;
        unsigned int imag_sgn_exp = (imag_bits >> 23) & 0x1FF;


        // get mantissa bits from fp64
        int num_mantissa_bits = T - num_exponent_bits;
        unsigned long long real_mant = (real_bits_64 & 0xFFFFFFFFFFFFFULL) >> (52 - num_mantissa_bits);
        unsigned long long imag_mant = (imag_bits_64 & 0xFFFFFFFFFFFFFULL) >> (52 - num_mantissa_bits);

        unsigned long long packed_real = ((unsigned long long)real_sgn_exp << num_mantissa_bits) | real_mant;
        unsigned long long packed_imag = ((unsigned long long)imag_sgn_exp << num_mantissa_bits) | imag_mant;

        for (int i = 0; i < num_bytes; i++) {{
            chars_real[i] = (packed_real >> (8 * i)) & 0xFF;
            chars_imag[i] = (packed_imag >> (8 * i)) & 0xFF;
        }}

        for (int i = 0; i < num_bytes; ++i) {{
            s_out[threadIdx.x * num_bytes * 2 + i] = chars_real[i];
            s_out[threadIdx.x * num_bytes * 2 + num_bytes + i] = chars_imag[i];
        }}
    }}
    __syncthreads();

    int start_idx = tile * {BLOCK_SIZE} * num_bytes * 2;
    int end_idx = min(N * num_bytes * 2, start_idx + {BLOCK_SIZE} * num_bytes * 2);
    for (int i = start_idx + threadIdx.x; i < end_idx; i += {BLOCK_SIZE}) {{
        out[i] = s_out[i - start_idx];
    }}
}}


extern "C" __global__ void _compress_fp16(unsigned char* out, const complex<double>* inp, const int N) {{ _compress<16>(out, inp, N); }}
extern "C" __global__ void _compress_fp24(unsigned char* out, const complex<double>* inp, const int N) {{ _compress<24>(out, inp, N); }}
extern "C" __global__ void _compress_fp32(unsigned char* out, const complex<double>* inp, const int N) {{ _compress<32>(out, inp, N); }}
extern "C" __global__ void _compress_fp40(unsigned char* out, const complex<double>* inp, const int N) {{ _compress<40>(out, inp, N); }}
extern "C" __global__ void _compress_fp48(unsigned char* out, const complex<double>* inp, const int N) {{ _compress<48>(out, inp, N); }}

// Use int T for a non-type template parameter
template<int T>
__global__
void _decompress(complex<double>* out, const unsigned char* inp, const int N) {{
    // Since T is a template param, num_bytes is a compile-time constant
    static const int num_bytes = T / 8;
    __shared__ unsigned char s_inp[{BLOCK_SIZE} * num_bytes * 2];

    int tile = blockIdx.x;
    int idx = tile * {BLOCK_SIZE} + threadIdx.x;

    // 1. Coalesced load from global to shared memory
    int start_idx = tile * {BLOCK_SIZE} * num_bytes * 2;
    int end_idx = min(N * num_bytes * 2, start_idx + {BLOCK_SIZE} * num_bytes * 2);
    
    for (int i = threadIdx.x; (start_idx + i) < end_idx; i += {BLOCK_SIZE}) {{
        s_inp[i] = inp[start_idx + i];
    }}
    __syncthreads();

    if (idx < N) {{
        unsigned long long packed_real = 0;
        unsigned long long packed_imag = 0;

        // Reconstruct the packed integers (Little-endian)
        int base_s_idx = threadIdx.x * num_bytes * 2;
        for (int i = 0; i < num_bytes; i++) {{
            packed_real |= (unsigned long long)s_inp[base_s_idx + i] << (8 * i);
            packed_imag |= (unsigned long long)s_inp[base_s_idx + num_bytes + i] << (8 * i);
        }}

         // 1 sign + 8 exponent from fp32
        const int num_exponent_bits = 9;
        const int num_mantissa_bits = T - num_exponent_bits;

        // Lambda to reconstruct double from truncated bits
        auto unpack = [&](unsigned long long packed) -> double {{
            // Extract Sign and Exponent
            unsigned int sgn_exp_f = (unsigned int)(packed >> num_mantissa_bits);
            unsigned long long sign = (sgn_exp_f >> 8) & 1;
            unsigned long long exp_f = sgn_exp_f & 0xFF;

            // Extract Mantissa
            unsigned long long mant_bits = packed & ((1ULL << num_mantissa_bits) - 1);

            // Shift Exponent: Convert fp32 bias (127) to fp64 bias (1023)
            unsigned long long exp_d;
            if (exp_f == 0) exp_d = 0;
            else if (exp_f == 255) exp_d = 2047;
            else exp_d = exp_f + (1023 - 127);

            // Align mantissa to the MSB of the double's 52-bit mantissa field
            unsigned long long mant_d = mant_bits << (52 - num_mantissa_bits);

            unsigned long long res = (sign << 63) | (exp_d << 52) | mant_d;
            return __longlong_as_double(res);
        }};

        out[idx] = complex<double>(unpack(packed_real), unpack(packed_imag));
    }}
}}

extern "C" __global__ void _decompress_fp16(complex<double>* out, const unsigned char* inp, const int N) {{ _decompress<16>(out, inp, N); }}
extern "C" __global__ void _decompress_fp24(complex<double>* out, const unsigned char* inp, const int N) {{ _decompress<24>(out, inp, N); }}
extern "C" __global__ void _decompress_fp32(complex<double>* out, const unsigned char* inp, const int N) {{ _decompress<32>(out, inp, N); }}
extern "C" __global__ void _decompress_fp40(complex<double>* out, const unsigned char* inp, const int N) {{ _decompress<40>(out, inp, N); }}
extern "C" __global__ void _decompress_fp48(complex<double>* out, const unsigned char* inp, const int N) {{ _decompress<48>(out, inp, N); }}
"""

module = cp.RawModule(code=cuda_source, options=("--std=c++11",))

_compress = {
    16: module.get_function("_compress_fp16"),
    24: module.get_function("_compress_fp24"),
    32: module.get_function("_compress_fp32"),
    40: module.get_function("_compress_fp40"),
    48: module.get_function("_compress_fp48"),
}

_decompress = {
    16: module.get_function("_decompress_fp16"),
    24: module.get_function("_decompress_fp24"),
    32: module.get_function("_decompress_fp32"),
    40: module.get_function("_decompress_fp40"),
    48: module.get_function("_decompress_fp48"),
}


def compress(inp: NDArray, bits: int, out: NDArray | None = None):
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

    if bits not in _compress.keys():
        raise ValueError(
            f"Unsupported bit width: {bits}. Supported values are {list(_compress.keys())}."
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

    _compress[bits](((N + BLOCK_SIZE - 1) // BLOCK_SIZE,), (BLOCK_SIZE,), (out, inp, N))

    return out


def decompress(inp: NDArray, bits: int, out: NDArray | None = None):
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

    if bits not in _decompress.keys():
        raise ValueError(
            f"Unsupported bit width: {bits}. Supported values are {list(_decompress.keys())}."
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

    _decompress[bits](
        ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,), (BLOCK_SIZE,), (out, inp, N)
    )

    return out
