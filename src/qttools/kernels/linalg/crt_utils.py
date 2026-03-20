"""
Ozaki-II Light: end-to-end FP64 GEMM using serial single-modulus INT8 kernels.

Pipeline:
  1. Preprocess: scale FP64 A[M,K], B[K,N] → per-modulus INT8 slices + CRT params
  2. GEMM: call crt kernel s times (one per modulus) → uint8 residues[s][M][N]
  3. Reconstruct: FP64 CRT accumulation on GPU using s_{i1}/s_{i2} split
     (Ozaki split-accumulation in PyTorch tensor ops)
  4. Inverse scaling → FP64 result

The reconstruction uses the Ozaki split-accumulation scheme (Section 4.3):
  accum1 = sum(s_{i1} * u_i)   -- exact in FP64 by construction of beta_i
  accum2 = sum(s_{i2} * u_i)   -- remainder
  Q = round(P_inv * accum1)     -- quotient
  C'' = fma(-P_1, Q, fma(-P_2, Q, accum1)) + accum2  -- double-double reduction

Fully vectorized on GPU, no Python loops, no integer overflow.
"""

import math
from functools import reduce

import numpy as np
import torch


def safe_P_half(log2_P, moduli, K):
    """
    Compute P_half with safety cap for rmod_fp64 precision.

    The slice kernel's rmod_fp64 requires |x/p| * 2^{-51} < 0.5 for correctness,
    where x = trunc(mu * A[i,j]) and p is any modulus.  The worst case is the
    smallest modulus p_min and largest scaled element.
    """
    P_half_raw = float(np.float32((log2_P - 1.5) / 2.0))
    p_min = min(int(m) for m in moduli)
    max_elem_bits = 3  # log2(6) ~ 2.6, round up
    P_half_max = math.log2(p_min) + 50 - max_elem_bits + 0.5 * math.log2(max(K, 1))
    return min(P_half_raw, float(np.float32(P_half_max)))


def _build_coprime_set():
    """Build the paper's coprime moduli set from {256, 255, ..., 29}."""
    selected = []
    for candidate in range(256, 28, -1):
        if all(math.gcd(candidate, s) == 1 for s in selected):
            selected.append(candidate)
        if len(selected) >= 20:
            break
    return selected


AVAILABLE_MODULI = _build_coprime_set()


def _extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    g, x, y = _extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


def _mod_inverse(a, m):
    g, x, _ = _extended_gcd(a % m, m)
    assert g == 1, f"No modular inverse for {a} mod {m}"
    return x % m


def compute_crt_params(moduli):
    """
    Compute CRT parameters following the paper's Section 4.1 and 4.3.

    Returns:
        big_P_int, s_i1, s_i2, P_1, P_2, P_inv
    """
    moduli = [int(m) for m in moduli]
    N = len(moduli)
    big_P = reduce(lambda a, b: a * b, moduli)

    P_1 = float(big_P)
    P_2 = float(big_P - int(P_1))
    P_inv = 1.0 / P_1

    s_i1_list = []
    s_i2_list = []

    w_values = []
    for m_t in moduli:
        P_t = big_P // m_t
        q_t = _mod_inverse(P_t, m_t)
        w_values.append(P_t * q_t)

    ceil_log2_N = math.ceil(math.log2(max(N, 2)))
    log2_w = [int(w).bit_length() - 1 for w in w_values]
    max_log2_w = max(log2_w)

    for i, (w, m_t) in enumerate(zip(w_values, moduli)):
        beta_i = 53 - 8 - ceil_log2_N + log2_w[i] - max_log2_w
        full_val = float(w)

        if beta_i <= 0:
            s_i1_list.append(0.0)
            s_i2_list.append(float(w))
        elif full_val == 0.0:
            s_i1_list.append(0.0)
            s_i2_list.append(0.0)
        else:
            exp = math.floor(math.log2(abs(full_val)))
            shift = exp - beta_i + 1
            s1 = (
                math.trunc(full_val / (2.0**shift)) * (2.0**shift)
                if shift > 0
                else full_val
            )
            s2 = float(w - int(s1))
            s_i1_list.append(s1)
            s_i2_list.append(s2)

    return big_P, s_i1_list, s_i2_list, P_1, P_2, P_inv


def get_recommended_moduli(k, num_moduli=None, target_bits=52):
    """Select coprime moduli so that the CRT product P is large enough."""
    if num_moduli is not None:
        assert num_moduli <= len(
            AVAILABLE_MODULI
        ), f"Max {len(AVAILABLE_MODULI)} moduli"
        return AVAILABLE_MODULI[:num_moduli]

    required_log2_P = 1.0 + math.log2(max(k, 1)) + 2.0 * target_bits
    P = 1
    selected = []
    for p in AVAILABLE_MODULI:
        selected.append(p)
        P *= p
        if math.log2(P) > required_log2_P:
            break
    if math.log2(P) <= required_log2_P:
        raise ValueError(
            f"Cannot find enough moduli: need log2(P) > {required_log2_P:.1f} "
            f"but only have log2(P) = {math.log2(P):.1f} with {len(selected)} moduli."
        )
    return selected


def _rmod_fp64(X, p_i, pinv_64=None, pinv_32=None, N_total=None):
    """rmod(x, p_i) = x - round(x/p_i) * p_i, result in [-p_i/2, p_i/2]."""
    p_f = float(p_i)
    p_half = p_f / 2.0
    r = torch.fmod(X, p_f)
    r = torch.where(r > p_half, r - p_f, r)
    r = torch.where(r < -p_half, r + p_f, r)
    return r


def crt_preprocess(A, B, moduli=None, num_moduli=None, target_bits=52):
    """
    Preprocess FP64 matrices for crt serial GEMM.

    Uses Ozaki-II Cauchy-Schwarz scaling (Section 4.2) and CRT split-accumulation
    parameters (Section 4.3). No GROUP_SIZE constraint on num_moduli.

    Args:
        A: [M, K] FP64 tensor
        B: [K, N] FP64 tensor
        moduli: list of coprime moduli, or None for auto
        num_moduli: number of moduli
        target_bits: desired precision bits (default 52)

    Returns:
        dict with per-modulus A/B slices and CRT reconstruction constants
    """
    assert A.dtype == torch.float64 and B.dtype == torch.float64
    M_dim, K = A.shape
    K2, N = B.shape
    assert K == K2

    if moduli is None:
        moduli = get_recommended_moduli(K, num_moduli, target_bits)
    s = len(moduli)

    # Ozaki-II Cauchy-Schwarz scaling (Section 4.2)
    big_P = reduce(lambda a, b: a * b, [int(m) for m in moduli])
    log2_P = math.log2(float(big_P))
    P_half = safe_P_half(log2_P, moduli, K)

    a_sq_sum = (A * A).sum(dim=1).clamp(min=1e-300)
    mu_exp = torch.floor(P_half - 0.5 * torch.log2(a_sq_sum))
    mu = torch.pow(2.0, mu_exp).to(torch.float64)
    mu_inv = 1.0 / mu

    b_sq_sum = (B * B).sum(dim=0).clamp(min=1e-300)
    nu_exp = torch.floor(P_half - 0.5 * torch.log2(b_sq_sum))
    nu = torch.pow(2.0, nu_exp).to(torch.float64)
    nu_inv = 1.0 / nu

    A_prime = torch.trunc(mu.unsqueeze(1) * A)
    B_prime = torch.trunc(B * nu.unsqueeze(0))
    B_prime_t = B_prime.t().contiguous()

    K_aligned = ((K + 127) // 128) * 128
    device = A.device

    # Sequential rmod per modulus to avoid O(s*M*K) FP64 peak memory.
    # At 16384 with s=9, vectorized approach needs ~36GB; sequential needs ~4GB.
    A_slices = []
    B_slices = []
    for t in range(s):
        m_val = float(moduli[t])
        half_m = m_val / 2.0

        a_r_t = torch.fmod(A_prime, m_val)
        a_r_t = torch.where(a_r_t > half_m, a_r_t - m_val, a_r_t)
        a_r_t = torch.where(a_r_t < -half_m, a_r_t + m_val, a_r_t)

        b_r_t = torch.fmod(B_prime_t, m_val)
        b_r_t = torch.where(b_r_t > half_m, b_r_t - m_val, b_r_t)
        b_r_t = torch.where(b_r_t < -half_m, b_r_t + m_val, b_r_t)

        if K == K_aligned:
            A_slices.append(a_r_t.to(torch.int8).contiguous())
            B_slices.append(b_r_t.to(torch.int8).contiguous())
        else:
            a_int8 = torch.zeros(M_dim, K_aligned, dtype=torch.int8, device=device)
            a_int8[:, :K] = a_r_t.to(torch.int8)
            A_slices.append(a_int8)
            b_int8 = torch.zeros(N, K_aligned, dtype=torch.int8, device=device)
            b_int8[:, :K] = b_r_t.to(torch.int8)
            B_slices.append(b_int8)
        del a_r_t, b_r_t

    # CRT reconstruction parameters (s_i1/s_i2 split, P_1/P_2 double-double, P_inv)
    big_P_int, s_i1, s_i2, P_1, P_2, P_inv = compute_crt_params(moduli)

    s_i1_tensor = torch.tensor(s_i1, dtype=torch.float64, device=device)
    s_i2_tensor = torch.tensor(s_i2, dtype=torch.float64, device=device)

    return {
        "A_slices": A_slices,
        "B_slices": B_slices,
        "scale_a": mu_inv,
        "scale_b": nu_inv,
        "moduli_list": moduli,
        "k_aligned": K_aligned,
        "num_moduli": s,
        "s_i1": s_i1_tensor,
        "s_i2": s_i2_tensor,
        "P_1": P_1,
        "P_2": P_2,
        "P_inv": P_inv,
    }


def crt_reconstruct_fp64(residues, s_i1, s_i2, P_1, P_2, P_inv):
    """
    CRT reconstruction from uint8 residues using FP64 split-accumulation.

    NOTE: This method suffers from catastrophic cancellation when P > 2^53.
    Use crt_reconstruct_garner() instead for full precision.
    Kept for reference / kernel epilogue equivalence testing.
    """
    R = torch.stack(residues).to(torch.float64)
    M_dim, N_dim = residues[0].shape
    R_flat = R.reshape(len(residues), -1)
    accum1 = (s_i1.unsqueeze(0) @ R_flat).reshape(M_dim, N_dim)
    accum2 = (s_i2.unsqueeze(0) @ R_flat).reshape(M_dim, N_dim)
    Q = torch.round(P_inv * accum1)
    dev = accum1.device
    inner = torch.addcmul(
        accum1, torch.tensor(-P_2, dtype=torch.float64, device=dev), Q
    )
    C_prime = torch.addcmul(
        inner, torch.tensor(-P_1, dtype=torch.float64, device=dev), Q
    )
    C_prime = C_prime + accum2
    return C_prime


def crt_reconstruct_garner(residues, moduli, P_1, P_2):
    """
    CRT reconstruction using Garner's algorithm + double-double Horner evaluation.

    This avoids the catastrophic cancellation in the split-accumulation method
    when P > 2^53.  Precision is limited only by the final FP64 conversion
    (53 bits of the result), not by intermediate overflow.

    Algorithm:
      1. Garner mixed-radix digits: v[j] computed via modular arithmetic
         (all values < p_j < 256, exact in int16/int64).
      2. Horner evaluation with Dekker double-double arithmetic:
         C'' = v[0] + p[0]*(v[1] + p[1]*(v[2] + ... ))
         Each multiply-by-scalar uses error-free two-product (Dekker),
         each add uses Knuth two-sum → ~106-bit intermediate precision.
      3. Center to [-P/2, P/2) via double-double subtraction of P.

    Args:
        residues: list of s tensors [M, N] uint8, residues U_i in [0, p_i)
        moduli:   list of s coprime moduli (Python ints)
        P_1, P_2: double-double representation of P = prod(moduli)

    Returns:
        C_prime: [M, N] FP64, the reconstructed integer product C''
    """
    s = len(residues)
    moduli_int = [int(m) for m in moduli]

    # --- Step 1: Garner mixed-radix digits ---
    # Precompute c[j][i] = p_i^{-1} mod p_j  for all i < j
    garner_c = []
    for j in range(s):
        row = []
        for i in range(j):
            row.append(_mod_inverse(moduli_int[i], moduli_int[j]))
        garner_c.append(row)

    # Compute v[j] for all elements simultaneously.
    # Memory-efficient: keep v[] as uint8 (each digit < p_j < 256),
    # only use int64 for temporary modular arithmetic within the loop.
    v = [residues[0].clone()]  # uint8, v[0] = U_0 in [0, p_0)

    for j in range(1, s):
        p_j = moduli_int[j]
        temp = residues[j].to(torch.int64)
        for i in range(j):
            temp = ((temp - v[i].to(torch.int64)) * garner_c[j][i]) % p_j
        v.append(temp.to(torch.uint8))
        del temp

    # --- Step 2: Double-double Horner evaluation ---
    # C'' = v[0] + p[0]*(v[1] + p[1]*(v[2] + ... + p[s-2]*v[s-1]))
    # Horner form: start from v[s-1], work backwards.
    SPLIT = 2.0**27 + 1.0  # Veltkamp splitting constant

    hi = v[s - 1].to(torch.float64)
    lo = torch.zeros_like(hi)

    for i in range(s - 2, -1, -1):
        p_f = float(moduli_int[i])

        # --- (hi, lo) *= p_f  using Dekker two-product ---
        prod_hi = hi * p_f
        # Error-free product via Veltkamp split of hi
        c = SPLIT * hi
        hi_h = c - (c - hi)
        hi_l = hi - hi_h
        err_mul = hi_h * p_f - prod_hi + hi_l * p_f
        lo = lo * p_f + err_mul
        hi = prod_hi

        # --- (hi, lo) += v[i]  using Knuth two-sum ---
        v_f = v[i].to(torch.float64)
        s_val = hi + v_f
        err_add = v_f - (s_val - hi)
        lo = lo + err_add
        hi = s_val
        del v_f

    # --- Step 3: Center to [-P/2, P/2) ---
    # P/2 in double-double: (P_1/2, P_2/2)
    P_half_hi = P_1 / 2.0
    mask = hi > P_half_hi
    if mask.any():
        # (hi, lo) -= (P_1, P_2) using two-sum for hi - P_1
        s_val = hi - P_1
        err_sub = -P_1 - (s_val - hi)
        hi = torch.where(mask, s_val, hi)
        lo = torch.where(mask, lo + err_sub - P_2, lo)

    return hi + lo


def crt_zgemm3m_preprocess(
    A_re, A_im, B_re, B_im, moduli=None, num_moduli=None, target_bits=52
):
    """
    Preprocess complex FP64 matrices for crt ZGEMM3M (Gauss 3-GEMM).

    Uses complex-norm Cauchy-Schwarz scaling (shared scales for re/im parts)
    so that modular subtraction/addition preserves anti-cancellation.

    The Gauss identity:
      C_re = A_re*B_re - A_im*B_im = S1 - S2
      C_im = A_re*B_im + A_im*B_re = S3 - S1 - S2
    where S1 = A_re*B_re, S2 = A_im*B_im, S3 = (A_re+A_im)*(B_re+B_im)

    In modular domain, T_A_res = (A_re_res + A_im_res) mod p is exact,
    and all combinations are done via modular arithmetic — zero cancellation.

    Args:
        A_re, A_im: [M, K] FP64 real/imaginary parts of A
        B_re, B_im: [K, N] FP64 real/imaginary parts of B
        moduli: list of coprime moduli, or None for auto
        num_moduli: number of moduli
        target_bits: desired precision bits (default 52)

    Returns:
        dict with per-modulus INT8 slices for all 5 operands + CRT params
    """
    assert A_re.dtype == torch.float64 and A_im.dtype == torch.float64
    assert B_re.dtype == torch.float64 and B_im.dtype == torch.float64
    M_dim, K = A_re.shape
    assert A_im.shape == (M_dim, K)
    K2, N = B_re.shape
    assert B_im.shape == (K2, N)
    assert K == K2

    if moduli is None:
        moduli = get_recommended_moduli(K, num_moduli, target_bits)
    s = len(moduli)

    # Complex-norm Cauchy-Schwarz scaling
    # Use complex row/column norms so same scale applies to re and im parts.
    # This costs 0.5 bits vs real GEMM (factor of 2 in the K dimension).
    big_P = reduce(lambda a, b: a * b, [int(m) for m in moduli])
    log2_P = math.log2(float(big_P))
    P_half = safe_P_half(log2_P, moduli, K)

    # Per-row complex norm: ||A_complex[m,:]||_2^2 = ||A_re||^2 + ||A_im||^2
    a_sq_sum = (A_re * A_re + A_im * A_im).sum(dim=1).clamp(min=1e-300)
    mu_exp = torch.floor(P_half - 0.5 * torch.log2(a_sq_sum))
    mu = torch.pow(2.0, mu_exp).to(torch.float64)
    mu_inv = 1.0 / mu

    # Per-column complex norm: ||B_complex[:,n]||_2^2 = ||B_re||^2 + ||B_im||^2
    b_sq_sum = (B_re * B_re + B_im * B_im).sum(dim=0).clamp(min=1e-300)
    nu_exp = torch.floor(P_half - 0.5 * torch.log2(b_sq_sum))
    nu = torch.pow(2.0, nu_exp).to(torch.float64)
    nu_inv = 1.0 / nu

    # Scale and truncate — shared scales for re and im
    A_re_prime = torch.trunc(mu.unsqueeze(1) * A_re)
    A_im_prime = torch.trunc(mu.unsqueeze(1) * A_im)
    B_re_prime = torch.trunc(B_re * nu.unsqueeze(0))
    B_im_prime = torch.trunc(B_im * nu.unsqueeze(0))

    # T_A = A_re' + A_im', T_B = B_re' + B_im' (exact integer addition in FP64)
    T_A_prime = A_re_prime + A_im_prime
    T_B_prime = B_re_prime + B_im_prime

    # Transpose B matrices for NT layout (kernel expects B as [N, K])
    B_re_prime_t = B_re_prime.t().contiguous()
    B_im_prime_t = B_im_prime.t().contiguous()
    T_B_prime_t = T_B_prime.t().contiguous()

    K_aligned = ((K + 127) // 128) * 128
    device = A_re.device

    # Vectorized rmod for all moduli: [s, 1, 1] broadcast
    moduli_f = torch.tensor(
        [float(m) for m in moduli], dtype=torch.float64, device=device
    )
    moduli_v = moduli_f.view(s, 1, 1)
    half_v = moduli_v / 2.0

    def batch_rmod(X):
        """Compute rmod(X, p_i) for all moduli, return [s, *X.shape] int8."""
        r = torch.fmod(X.unsqueeze(0), moduli_v)
        r = torch.where(r > half_v, r - moduli_v, r)
        r = torch.where(r < -half_v, r + moduli_v, r)
        return r

    # Compute residues for all 5 operands: [s, M, K] or [s, N, K]
    a_re_r = batch_rmod(A_re_prime)  # [s, M, K]
    a_im_r = batch_rmod(A_im_prime)  # [s, M, K]
    b_re_r = batch_rmod(B_re_prime_t)  # [s, N, K]
    b_im_r = batch_rmod(B_im_prime_t)  # [s, N, K]
    t_a_r = batch_rmod(T_A_prime)  # [s, M, K]
    t_b_r = batch_rmod(T_B_prime_t)  # [s, N, K]

    # Pack into per-modulus int8 slices with K padding
    def pack_slices(r_all, dim0):
        slices = []
        if K == K_aligned:
            for t in range(s):
                slices.append(r_all[t].to(torch.int8).contiguous())
        else:
            for t in range(s):
                buf = torch.zeros(dim0, K_aligned, dtype=torch.int8, device=device)
                buf[:, :K] = r_all[t].to(torch.int8)
                slices.append(buf)
        return slices

    A_re_slices = pack_slices(a_re_r, M_dim)
    A_im_slices = pack_slices(a_im_r, M_dim)
    B_re_slices = pack_slices(b_re_r, N)
    B_im_slices = pack_slices(b_im_r, N)
    T_A_slices = pack_slices(t_a_r, M_dim)
    T_B_slices = pack_slices(t_b_r, N)

    # CRT reconstruction parameters
    big_P_int, s_i1, s_i2, P_1, P_2, P_inv = compute_crt_params(moduli)

    s_i1_tensor = torch.tensor(s_i1, dtype=torch.float64, device=device)
    s_i2_tensor = torch.tensor(s_i2, dtype=torch.float64, device=device)

    return {
        "A_re_slices": A_re_slices,
        "A_im_slices": A_im_slices,
        "B_re_slices": B_re_slices,
        "B_im_slices": B_im_slices,
        "T_A_slices": T_A_slices,
        "T_B_slices": T_B_slices,
        "scale_a": mu_inv,
        "scale_b": nu_inv,
        "moduli_list": moduli,
        "k_aligned": K_aligned,
        "num_moduli": s,
        "s_i1": s_i1_tensor,
        "s_i2": s_i2_tensor,
        "P_1": P_1,
        "P_2": P_2,
        "P_inv": P_inv,
    }


def crt_zgemm3m(A_re, A_im, B_re, B_im, moduli=None, num_moduli=None, target_bits=52):
    """
    Complex FP64 GEMM (ZGEMM) using CRT with Gauss 3-GEMM trick.

    Computes C = (A_re + i*A_im) @ (B_re + i*B_im) using only 3s INT8 kernel
    calls (vs 4s for naive approach), with exact modular arithmetic for the
    complex combination — zero catastrophic cancellation.

    Pipeline:
      1. Complex-norm Cauchy-Schwarz scaling (shared for re/im)
      2. For each modulus p_i:
           S1 = GEMM(A_re_res, B_re_res) mod p     (A_re * B_re)
           S2 = GEMM(A_im_res, B_im_res) mod p     (A_im * B_im)
           S3 = GEMM(T_A_res, T_B_res) mod p       ((A_re+A_im)*(B_re+B_im))
           C_re_res = (S1 - S2) mod p               exact modular subtraction
           C_im_res = (S3 - S1 - S2) mod p          exact modular combination
      3. CRT reconstruction for C_re and C_im
      4. Inverse scaling

    Args:
        A_re, A_im: [M, K] FP64 real/imaginary parts of A
        B_re, B_im: [K, N] FP64 real/imaginary parts of B
        moduli: list of coprime moduli, or None for auto
        num_moduli: number of moduli
        target_bits: desired precision bits (default 52)

    Returns:
        C_re, C_im: [M, N] FP64 real/imaginary parts of C = A @ B
    """
    import emu_gemm

    M_dim, K = A_re.shape
    K2, N = B_re.shape
    assert K == K2

    params = crt_zgemm3m_preprocess(
        A_re,
        A_im,
        B_re,
        B_im,
        moduli=moduli,
        num_moduli=num_moduli,
        target_bits=target_bits,
    )

    s = params["num_moduli"]
    moduli_list = params["moduli_list"]
    device = A_re.device

    # For each modulus: 3 GEMMs (Gauss) + modular combination
    C_re_residues = []
    C_im_residues = []

    for i in range(s):
        p = moduli_list[i]
        p_int = int(p)

        # S1 = A_re * B_re mod p
        S1 = torch.zeros(M_dim, N, dtype=torch.uint8, device=device)
        emu_gemm.int8_gemm_crt_nt(
            params["A_re_slices"][i], params["B_re_slices"][i], S1, p
        )

        # S2 = A_im * B_im mod p
        S2 = torch.zeros(M_dim, N, dtype=torch.uint8, device=device)
        emu_gemm.int8_gemm_crt_nt(
            params["A_im_slices"][i], params["B_im_slices"][i], S2, p
        )

        # S3 = (A_re + A_im) * (B_re + B_im) mod p
        S3 = torch.zeros(M_dim, N, dtype=torch.uint8, device=device)
        emu_gemm.int8_gemm_crt_nt(
            params["T_A_slices"][i], params["T_B_slices"][i], S3, p
        )

        # Modular combination — exact integer arithmetic, zero cancellation!
        # C_re = (S1 - S2) mod p
        # C_im = (S3 - S1 - S2) mod p
        # All values are uint8 in [0, p), compute in int32 to avoid underflow
        S1_i32 = S1.to(torch.int32)
        S2_i32 = S2.to(torch.int32)
        S3_i32 = S3.to(torch.int32)

        C_re_mod = (S1_i32 - S2_i32) % p_int
        C_im_mod = (S3_i32 - S1_i32 - S2_i32) % p_int

        C_re_residues.append(C_re_mod.to(torch.uint8))
        C_im_residues.append(C_im_mod.to(torch.uint8))

    # CRT reconstruction on GPU (Garner's algorithm, double-double Horner)
    C_re_prime = crt_reconstruct_garner(
        C_re_residues, moduli_list, params["P_1"], params["P_2"]
    )

    C_im_prime = crt_reconstruct_garner(
        C_im_residues, moduli_list, params["P_1"], params["P_2"]
    )

    # Inverse scaling (shared mu, nu for re and im)
    scale = params["scale_a"].unsqueeze(1) * params["scale_b"].unsqueeze(0)
    C_re = C_re_prime * scale
    C_im = C_im_prime * scale

    return C_re, C_im


def _compute_garner_coefficients(moduli):
    """
    Compute Garner coefficients for CRT reconstruction.

    garner_c[j][i] = p_i^{-1} mod p_j   for i < j
    Stored as flat [s, s] array (upper entries unused, lower triangle filled).
    """
    s = len(moduli)
    moduli_int = [int(m) for m in moduli]
    garner_c = [[0] * s for _ in range(s)]
    for j in range(s):
        for i in range(j):
            garner_c[j][i] = _mod_inverse(moduli_int[i], moduli_int[j])
    return garner_c


def _precompute_barrett_constants(moduli, device):
    """Precompute Barrett constants for the crt_zgemm3m_slice kernel."""
    barrett_m = []
    barrett_r32 = []
    for p in moduli:
        p_int = int(p)
        q = (1 << 32) // p_int
        barrett_m.append(int(q))
        barrett_r32.append(int((1 << 32) - q * p_int))

    return (
        torch.tensor(barrett_m, dtype=torch.int32, device=device),
        torch.tensor(barrett_r32, dtype=torch.int32, device=device),
    )


def _is_cupy(x):
    """Check if x is a cupy ndarray without importing cupy at module level."""
    return type(x).__module__.startswith("cupy")


def _to_torch(x):
    """Zero-copy cupy ndarray → torch CUDA tensor via DLPack."""
    return torch.from_dlpack(x)


def _to_cupy(t):
    """Zero-copy torch CUDA tensor → cupy ndarray via DLPack."""
    import cupy as cp

    return cp.from_dlpack(t)


def zgemm3m_crt(A_re, A_im, B_re, B_im, num_moduli=None, moduli=None, target_bits=52):
    """
    Unified FP64-in / FP64-out complex GEMM using fused 3M-CRT kernels.

        C = (A_re + j·A_im) @ (B_re + j·B_im)^T

    Accepts both **torch.Tensor** (CUDA) and **cupy.ndarray** inputs.
    Returns (C_re, C_im) in the same type as the inputs.

    Pipeline (all on GPU, using fused high-performance kernels):
      1. Scale  — row-wise Cauchy–Schwarz scaling        [crt_zgemm3m_scale_nt]
      2. Slice  — FP64 → INT8 modular residues            [crt_zgemm3m_slice_nt]
      3. GEMM   — fused Gauss 3-GEMM per modulus           [int8_zgemm3m_crt_nt]
      4. CRT    — Garner reconstruction → FP64             [crt_zgemm3m_crt_nt]

    Args:
        A_re: [M, K] FP64, real part of A          (torch.Tensor or cupy.ndarray)
        A_im: [M, K] FP64, imaginary part of A     (torch.Tensor or cupy.ndarray)
        B_re: [N, K] FP64, real part of B  (NT layout, K-contiguous)
        B_im: [N, K] FP64, imaginary part of B  (NT layout, K-contiguous)
        num_moduli: number of CRT moduli (6–16).  None = auto from K & target_bits.
        moduli: explicit list of coprime moduli.  Overrides num_moduli if given.
        target_bits: target mantissa precision (default 52 for full FP64).

    Returns:
        C_re: [M, N] FP64, real part of C
        C_im: [M, N] FP64, imaginary part of C
    """
    import emu_gemm

    # ── detect input type & convert cupy → torch (zero-copy) ──
    use_cupy = _is_cupy(A_re)
    if use_cupy:
        A_re, A_im = _to_torch(A_re), _to_torch(A_im)
        B_re, B_im = _to_torch(B_re), _to_torch(B_im)

    # ── input validation ──
    assert A_re.dtype == torch.float64 and A_im.dtype == torch.float64
    assert B_re.dtype == torch.float64 and B_im.dtype == torch.float64
    M, K = A_re.shape
    assert A_im.shape == (M, K)
    N, K2 = B_re.shape
    assert B_im.shape == (N, K2) and K == K2
    device = A_re.device

    # ── moduli selection & CRT parameters ──
    if moduli is None:
        moduli = get_recommended_moduli(K, num_moduli, target_bits)
    s = len(moduli)

    big_P_int, s_i1_list, s_i2_list, P_1, P_2, P_inv = compute_crt_params(moduli)
    log2_P = math.log2(float(big_P_int))
    P_half = safe_P_half(log2_P, moduli, K)
    K_aligned = ((K + 127) // 128) * 128

    moduli_tensor = torch.tensor(
        [int(m) for m in moduli], dtype=torch.int32, device=device
    )
    barrett_m, barrett_r32 = _precompute_barrett_constants(moduli, device)
    qPi_hi = torch.tensor(s_i1_list, dtype=torch.float64, device=device)
    qPi_lo = torch.tensor(s_i2_list, dtype=torch.float64, device=device)

    # ── 1. Scale ──
    mu = torch.empty(M, dtype=torch.float64, device=device)
    mu_inv = torch.empty(M, dtype=torch.float64, device=device)
    nu = torch.empty(N, dtype=torch.float64, device=device)
    nu_inv = torch.empty(N, dtype=torch.float64, device=device)

    emu_gemm.crt_zgemm3m_scale_nt(A_re, A_im, mu, mu_inv, P_half)
    emu_gemm.crt_zgemm3m_scale_nt(B_re, B_im, nu, nu_inv, P_half)

    # ── 2. Slice  (FP64 → INT8, all moduli at once) ──
    A_re_s = torch.empty(s, M, K_aligned, dtype=torch.int8, device=device)
    A_im_s = torch.empty(s, M, K_aligned, dtype=torch.int8, device=device)
    T_A_s = torch.empty(s, M, K_aligned, dtype=torch.int8, device=device)
    B_re_s = torch.empty(s, N, K_aligned, dtype=torch.int8, device=device)
    B_im_s = torch.empty(s, N, K_aligned, dtype=torch.int8, device=device)
    T_B_s = torch.empty(s, N, K_aligned, dtype=torch.int8, device=device)

    emu_gemm.crt_zgemm3m_slice_nt(
        A_re,
        A_im,
        mu,
        moduli_tensor,
        barrett_m,
        barrett_r32,
        A_re_s,
        A_im_s,
        T_A_s,
        K,
        K_aligned,
    )
    emu_gemm.crt_zgemm3m_slice_nt(
        B_re,
        B_im,
        nu,
        moduli_tensor,
        barrett_m,
        barrett_r32,
        B_re_s,
        B_im_s,
        T_B_s,
        K,
        K_aligned,
    )

    # ── 3. Fused 3M-GEMM  (1 kernel launch per modulus) ──
    D_merged = torch.empty(s, M, 2 * N, dtype=torch.uint8, device=device)
    a_cat = [torch.cat([A_re_s[i], A_im_s[i], T_A_s[i]], dim=1) for i in range(s)]
    b_cat = [torch.cat([B_re_s[i], B_im_s[i], T_B_s[i]], dim=1) for i in range(s)]

    for i in range(s):
        emu_gemm.int8_zgemm3m_crt_nt(a_cat[i], b_cat[i], D_merged[i], moduli[i])

    # ── 4. CRT reconstruction  (Garner's algorithm → FP64) ──
    D_re = D_merged[:, :, :N].contiguous()
    D_im = D_merged[:, :, N:].contiguous()

    # Re-pack into S1/S2/S3 expected by the CRT kernel:
    #   S1 = D_re,  S2 = 0,  S3 = (D_re + D_im) mod p
    # so that CRT gives  C_re = CRT(S1) - CRT(S2) = CRT(D_re)
    #                     C_im = CRT(S3) - CRT(S1) - CRT(S2) = CRT(D_im)
    S1 = D_re
    S2 = torch.zeros_like(S1)
    S3 = torch.empty_like(S1)
    for i in range(s):
        p = int(moduli[i])
        tmp = D_re[i].to(torch.int16) + D_im[i].to(torch.int16)
        S3[i] = torch.where(tmp >= p, tmp - p, tmp).to(torch.uint8)

    C_re = torch.zeros(M, N, dtype=torch.float64, device=device)
    C_im = torch.zeros(M, N, dtype=torch.float64, device=device)

    emu_gemm.crt_zgemm3m_crt_nt(
        S1,
        S2,
        S3,
        moduli_tensor,
        qPi_hi,
        qPi_lo,
        P_1,
        P_2,
        P_inv,
        mu_inv,
        nu_inv,
        C_re,
        C_im,
    )

    # ── convert back to cupy if needed ──
    if use_cupy:
        return _to_cupy(C_re), _to_cupy(C_im)
    return C_re, C_im


def crt_kernel_zgemm3m(
    A_re, A_im, B_re, B_im, moduli=None, num_moduli=None, target_bits=52
):
    """
    Complex FP64 GEMM using 3 GPU kernels:
      1. Slice kernel: FP64 → INT8 slices for all moduli (fused scale+trunc+rmod)
      2. crt kernel: INT8 GEMM per modulus (3 calls per modulus for Gauss trick)
      3. CRT merge kernel: uint8 residues → FP64 (fused CRT + inverse scaling)

    This replaces crt_zgemm3m() with GPU-accelerated preprocessing and CRT merge.

    Args:
        A_re, A_im: [M, K] FP64 real/imaginary parts of A
        B_re, B_im: [N, K] FP64 real/imaginary parts of B (NT layout, K-contiguous)
        moduli: list of coprime moduli, or None for auto
        num_moduli: number of moduli
        target_bits: desired precision bits (default 52)

    Returns:
        C_re, C_im: [M, N] FP64 real/imaginary parts of C = A @ B^T
    """
    import emu_gemm

    assert A_re.dtype == torch.float64 and A_im.dtype == torch.float64
    assert B_re.dtype == torch.float64 and B_im.dtype == torch.float64
    M_dim, K = A_re.shape
    assert A_im.shape == (M_dim, K)
    N, K2 = B_re.shape
    assert B_im.shape == (N, K2)
    assert K == K2

    if moduli is None:
        moduli = get_recommended_moduli(K, num_moduli, target_bits)
    s = len(moduli)
    device = A_re.device

    # --- Scaling (fused CUDA kernel) ---
    big_P = reduce(lambda a, b: a * b, [int(m) for m in moduli])
    log2_P = math.log2(float(big_P))
    P_half = safe_P_half(log2_P, moduli, K)

    # Per-row complex norm for A → mu, mu_inv (1 fused kernel launch)
    mu = torch.empty(M_dim, dtype=torch.float64, device=device)
    mu_inv = torch.empty(M_dim, dtype=torch.float64, device=device)
    emu_gemm.crt_zgemm3m_scale_nt(A_re, A_im, mu, mu_inv, P_half)

    # B is already in NT layout [N, K] — no transpose needed
    # Per-row complex norm for B → nu, nu_inv (1 fused kernel launch)
    nu = torch.empty(N, dtype=torch.float64, device=device)
    nu_inv = torch.empty(N, dtype=torch.float64, device=device)
    emu_gemm.crt_zgemm3m_scale_nt(B_re, B_im, nu, nu_inv, P_half)

    K_aligned = ((K + 127) // 128) * 128

    # Moduli tensor for kernels
    moduli_tensor = torch.tensor(
        [int(m) for m in moduli], dtype=torch.int32, device=device
    )
    barrett_m_tensor, barrett_r32_tensor = _precompute_barrett_constants(moduli, device)

    # ========== Kernel 1: Fused Slice (FP64 → INT8) ==========
    # A-side: [s, M, K_aligned] for A_re, A_im, T_A
    A_re_slices = torch.empty(s, M_dim, K_aligned, dtype=torch.int8, device=device)
    A_im_slices = torch.empty(s, M_dim, K_aligned, dtype=torch.int8, device=device)
    T_A_slices = torch.empty(s, M_dim, K_aligned, dtype=torch.int8, device=device)

    emu_gemm.crt_zgemm3m_slice_nt(
        A_re,
        A_im,
        mu,
        moduli_tensor,
        barrett_m_tensor,
        barrett_r32_tensor,
        A_re_slices,
        A_im_slices,
        T_A_slices,
        K,
        K_aligned,
    )

    # B-side: [s, N, K_aligned] for B_re, B_im, T_B
    B_re_slices = torch.empty(s, N, K_aligned, dtype=torch.int8, device=device)
    B_im_slices = torch.empty(s, N, K_aligned, dtype=torch.int8, device=device)
    T_B_slices = torch.empty(s, N, K_aligned, dtype=torch.int8, device=device)

    emu_gemm.crt_zgemm3m_slice_nt(
        B_re,
        B_im,
        nu,
        moduli_tensor,
        barrett_m_tensor,
        barrett_r32_tensor,
        B_re_slices,
        B_im_slices,
        T_B_slices,
        K,
        K_aligned,
    )

    # ========== Kernel 2: crt GEMM (3 calls per modulus) ==========
    S1_all = torch.zeros(s, M_dim, N, dtype=torch.uint8, device=device)
    S2_all = torch.zeros(s, M_dim, N, dtype=torch.uint8, device=device)
    S3_all = torch.zeros(s, M_dim, N, dtype=torch.uint8, device=device)

    for i in range(s):
        p = moduli[i]
        # S1 = A_re * B_re mod p
        emu_gemm.int8_gemm_crt_nt(A_re_slices[i], B_re_slices[i], S1_all[i], p)
        # S2 = A_im * B_im mod p
        emu_gemm.int8_gemm_crt_nt(A_im_slices[i], B_im_slices[i], S2_all[i], p)
        # S3 = T_A * T_B mod p
        emu_gemm.int8_gemm_crt_nt(T_A_slices[i], T_B_slices[i], S3_all[i], p)

    # ========== Kernel 3: Fused CRT Merge (uint8 → FP64) ==========
    # Direct CRT: precompute qPi = q_i * P_i (double-double split)
    big_P_int, s_i1, s_i2, P_1, P_2, P_inv = compute_crt_params(moduli)
    qPi_hi_tensor = torch.tensor(s_i1, dtype=torch.float64, device=device)
    qPi_lo_tensor = torch.tensor(s_i2, dtype=torch.float64, device=device)

    # Output
    C_re = torch.zeros(M_dim, N, dtype=torch.float64, device=device)
    C_im = torch.zeros(M_dim, N, dtype=torch.float64, device=device)

    emu_gemm.crt_zgemm3m_crt_nt(
        S1_all,
        S2_all,
        S3_all,
        moduli_tensor,
        qPi_hi_tensor,
        qPi_lo_tensor,
        P_1,
        P_2,
        P_inv,
        mu_inv,
        nu_inv,
        C_re,
        C_im,
    )

    return C_re, C_im


def crt_gemm(A, B, moduli=None, num_moduli=None, target_bits=52):
    """
    End-to-end FP64 GEMM using crt serial INT8 kernels + CRT reconstruction.

    C = A @ B   where A: [M, K], B: [K, N], both FP64

    Pipeline:
      1. Preprocess → per-modulus INT8 slices + CRT constants
      2. Serial GEMM: crt_nt called s times → uint8 residues [s][M][N]
      3. CRT reconstruction: FP64 split-accumulation on GPU (no integer overflow)
      4. Inverse scaling → FP64 result

    Returns:
        C: [M, N] FP64 tensor
    """
    import emu_gemm

    M_dim, K = A.shape
    K2, N = B.shape
    assert K == K2

    params = crt_preprocess(
        A, B, moduli=moduli, num_moduli=num_moduli, target_bits=target_bits
    )

    s = params["num_moduli"]
    moduli_list = params["moduli_list"]
    device = A.device

    # Serial GEMM → uint8 residues
    residues = []
    for i in range(s):
        D_i = torch.zeros(M_dim, N, dtype=torch.uint8, device=device)
        emu_gemm.int8_gemm_crt_nt(
            params["A_slices"][i], params["B_slices"][i], D_i, moduli_list[i]
        )
        residues.append(D_i)

    # CRT reconstruction on GPU (Garner's algorithm, double-double Horner)
    C_prime = crt_reconstruct_garner(
        residues, moduli_list, params["P_1"], params["P_2"]
    )

    # Inverse scaling
    C = C_prime * params["scale_a"].unsqueeze(1) * params["scale_b"].unsqueeze(0)
    return C
