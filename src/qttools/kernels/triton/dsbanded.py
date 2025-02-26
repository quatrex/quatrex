# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

TT_AVAIL = False
try:
    import triton
    import triton.language as tl
    import torch
    TT_AVAIL = True
except (ImportError, ModuleNotFoundError):
    pass


def cdiv(a, b):
    return (a + b - 1) // b


def band_gemm_from_band_format(
    a_blk_tall_and_skinny: torch.Tensor,
    b_blk_short_and_fat: torch.Tensor,
    c_blk_tall_and_skinny: torch.Tensor,
    blk_diag_dist_a: int = None,
    blk_diag_dist_b: int = None,
    BLK_M: int = 128,
    BLK_N: int = 128,
    BLK_K: int = 32,
    perform_scaling: bool = False,
    scale_quant: float = 1.0,
    transpose_B: bool = False,
) -> None:
    """
    Matrix multiplication of two banded matrices.

    Parameters
    ----------
    a_blk_tall_and_skinny : torch.Tensor
        A tall and skinny banded matrix of shape (b, m, (2*blk_diag_dist_a + 1)*BLK_M).
    b_blk_short_and_fat : torch.Tensor
        A short and fat banded matrix of shape (b, (2*blk_diag_dist_b + 1)*BLK_N, m).
    c_blk_tall_and_skinny : torch.Tensor
        Output - A tall and skinny banded matrix of shape (b, m, (2*blk_diag_dist_c + 1)*BLK_M).
    blk_diag_dist_a : int, optional
        The half-bandwidth of matrix `a`, by default None, counted
        in the number of blocks of size `BLK_M`.
    blk_diag_dist_b : int, optional
        The half-bandwidth of matrix `b`, by default None, counted
        in the number of blocks of size `BLK_N`.
    BLK_M : int, optional
        The block size, by default 128.
    BLK_N : int, optional
        The block size, by default 128.
    BLK_K : int, optional
        The block size, by default 32.
    """

    b, M, band_a = a_blk_tall_and_skinny.shape
    _, band_b, N = b_blk_short_and_fat.shape
    if blk_diag_dist_a is None:
        blk_diag_dist_a = (band_a // BLK_M - 1) // 2
    if blk_diag_dist_b is None:
        blk_diag_dist_b = (band_b // BLK_N - 1) // 2

    blk_diag_dist_c = blk_diag_dist_a + blk_diag_dist_b
    blk_band_c = 2 * blk_diag_dist_c + 1

    num_block_rows = cdiv(M, BLK_M)
    grid = (
        b,
        num_block_rows * blk_band_c,
    )

    if transpose_B:
        band_gemm_kernel[grid](
            a_ptr=a_blk_tall_and_skinny,
            b_ptr=b_blk_short_and_fat,
            c_ptr=c_blk_tall_and_skinny,
            M=M,
            blk_diag_dist_a=blk_diag_dist_a,
            blk_diag_dist_b=blk_diag_dist_b,
            BLK_M=BLK_M,
            BLK_N=BLK_N,
            BLK_K=BLK_K,
            num_pids_col=blk_band_c,
            stride_ab=a_blk_tall_and_skinny.stride(0),
            stride_am=a_blk_tall_and_skinny.stride(1),
            stride_ak=a_blk_tall_and_skinny.stride(2),
            stride_bb=b_blk_short_and_fat.stride(0),
            stride_bk=b_blk_short_and_fat.stride(2),
            stride_bn=b_blk_short_and_fat.stride(1),
            stride_cb=c_blk_tall_and_skinny.stride(0),
            stride_cm=c_blk_tall_and_skinny.stride(1),
            stride_cn=c_blk_tall_and_skinny.stride(2),
            allow_tf32=True,
            scale_quant=perform_scaling,
            scale_factor=scale_quant,
            transpose_B=transpose_B,
        )
    else:
        band_gemm_kernel[grid](
            a_ptr=a_blk_tall_and_skinny,
            b_ptr=b_blk_short_and_fat,
            c_ptr=c_blk_tall_and_skinny,
            M=M,
            blk_diag_dist_a=blk_diag_dist_a,
            blk_diag_dist_b=blk_diag_dist_b,
            BLK_M=BLK_M,
            BLK_N=BLK_N,
            BLK_K=BLK_K,
            num_pids_col=blk_band_c,
            stride_ab=a_blk_tall_and_skinny.stride(0),
            stride_am=a_blk_tall_and_skinny.stride(1),
            stride_ak=a_blk_tall_and_skinny.stride(2),
            stride_bb=b_blk_short_and_fat.stride(0),
            stride_bk=b_blk_short_and_fat.stride(1),
            stride_bn=b_blk_short_and_fat.stride(2),
            stride_cb=c_blk_tall_and_skinny.stride(0),
            stride_cm=c_blk_tall_and_skinny.stride(1),
            stride_cn=c_blk_tall_and_skinny.stride(2),
            allow_tf32=True,
            scale_quant=perform_scaling,
            scale_factor=scale_quant,
            transpose_B=transpose_B,
        )

    print(
        f"band gemm from band format. perform scaling: {perform_scaling}, scale_quant: {scale_quant}"
    )
    print(
        f"Norms. a_blk_tall_and_skinny: {torch.norm(a_blk_tall_and_skinny)}, b_blk_short_and_fat: {torch.norm(b_blk_short_and_fat)}, c_blk_tall_and_skinny: {torch.norm(c_blk_tall_and_skinny)}, c_blk_tall_and_skinny type: {c_blk_tall_and_skinny.dtype}"
    )

    # # --- DEBUG ONLY ----
    # # calculate:
    # # - nonzeros
    # # - max and min values
    # # for a, b, c
    # nnz_a_blk_tall_and_skinny = torch.count_nonzero(a_blk_tall_and_skinny)
    # nnz_b_blk_short_and_fat = torch.count_nonzero(b_blk_short_and_fat)
    # nnz_c_blk_tall_and_skinny = torch.count_nonzero(c_blk_tall_and_skinny)
    # max_a_blk_tall_and_skinny = torch.max(a_blk_tall_and_skinny)
    # min_a_blk_tall_and_skinny = torch.min(a_blk_tall_and_skinny)
    # max_b_blk_short_and_fat = torch.max(b_blk_short_and_fat)
    # min_b_blk_short_and_fat = torch.min(b_blk_short_and_fat)
    # max_c_blk_tall_and_skinny = torch.max(c_blk_tall_and_skinny)
    # min_c_blk_tall_and_skinny = torch.min(c_blk_tall_and_skinny)
    # # print
    # print(
    #     f"\n\nnnz_a_blk_tall_and_skinny: {nnz_a_blk_tall_and_skinny}, nnz_b_blk_short_and_fat: {nnz_b_blk_short_and_fat}, nnz_c_blk_tall_and_skinny: {nnz_c_blk_tall_and_skinny}"
    # )
    # print(
    #     f"max_a_blk_tall_and_skinny: {max_a_blk_tall_and_skinny}, min_a_blk_tall_and_skinny: {min_a_blk_tall_and_skinny}"
    # )
    # print(
    #     f"max_b_blk_short_and_fat: {max_b_blk_short_and_fat}, min_b_blk_short_and_fat: {min_b_blk_short_and_fat}"
    # )
    # print(
    #     f"max_c_blk_tall_and_skinny: {max_c_blk_tall_and_skinny}, min_c_blk_tall_and_skinny: {min_c_blk_tall_and_skinny}"
    # )
    # # print scaling factor
    # print(f"scale_quant: {scale_quant}, perform_scaling: {perform_scaling}")
    # # ---- END OF DEBUG ----

    # print("Inside band_gemm_from_band_format\n")
    # # count nonzeros:
    # nnz_a_blk_tall_and_skinny = torch.count_nonzero(a_blk_tall_and_skinny)
    # nnz_b_blk_short_and_fat = torch.count_nonzero(b_blk_short_and_fat)
    # nnz_c_blk_tall_and_skinny = torch.count_nonzero(c_blk_tall_and_skinny)
    # # print
    # print(
    #     f"nnz_a_blk_tall_and_skinny: {nnz_a_blk_tall_and_skinny}, nnz_b_blk_short_and_fat: {nnz_b_blk_short_and_fat}, nnz_c_blk_tall_and_skinny: {nnz_c_blk_tall_and_skinny}"
    # )
    # # print shapes
    # print(
    #     f"a_blk_tall_and_skinny.shape: {a_blk_tall_and_skinny.shape}, b_blk_short_and_fat.shape: {b_blk_short_and_fat.shape}, c_blk_tall_and_skinny.shape: {c_blk_tall_and_skinny.shape}"
    # )


@triton.jit
def band_gemm_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    blk_diag_dist_a,
    blk_diag_dist_b,
    num_pids_col,
    stride_ab,
    stride_am,
    stride_ak,  #
    stride_bb,
    stride_bk,
    stride_bn,  #
    stride_cb,
    stride_cm,
    stride_cn,
    allow_tf32: tl.constexpr,
    scale_quant: tl.constexpr,
    scale_factor: tl.constexpr,
    # Meta-parameters
    BLK_M: tl.constexpr,
    BLK_N: tl.constexpr,
    BLK_K: tl.constexpr,  #
    transpose_B: tl.constexpr,
):
    batch_pid = tl.program_id(axis=0)

    # move the pointers of a_ptr, b_ptr, c_ptr to the current batch
    a_ptr += batch_pid * stride_ab
    b_ptr += batch_pid * stride_bb
    c_ptr += batch_pid * stride_cb

    pid = tl.program_id(axis=1)
    bandC_blk_row = pid // num_pids_col
    bandC_blk_col = pid % num_pids_col

    blk_band_a = blk_diag_dist_a * 2 + 1
    blk_band_b = blk_diag_dist_b * 2 + 1
    blk_diag_dist_c = blk_diag_dist_a + blk_diag_dist_b

    # (pid_m, pid_n) is the coordinate of block in bandC that this program is responsible for.
    # the corresponding coordinate of bloack in "dense" matrix C is:
    denseC_blk_row = bandC_blk_row
    denseC_blk_col = bandC_blk_row + bandC_blk_col - blk_diag_dist_c

    num_denseC_blocks = tl.cdiv(M, BLK_M)

    if denseC_blk_col >= num_denseC_blocks:
        return
    if denseC_blk_row >= num_denseC_blocks:
        return
    if denseC_blk_col < 0:
        return
    if denseC_blk_row < 0:
        return

    bandA_col_start = max(
        0, denseC_blk_col - denseC_blk_row + blk_diag_dist_a - blk_diag_dist_b
    )
    bandB_row_start = max(
        0, -denseC_blk_col + denseC_blk_row - blk_diag_dist_a + blk_diag_dist_b
    )

    num_blocks = min(blk_band_a - bandA_col_start, blk_band_b - bandB_row_start)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (denseC_blk_row * BLK_M + tl.arange(0, BLK_M)) % M
    offs_bn = (denseC_blk_col * BLK_N + tl.arange(0, BLK_N)) % M
    offs_k = tl.arange(0, BLK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # # move the pointers to the start of the band
    a_ptrs += bandA_col_start * BLK_N * stride_ak
    b_ptrs += bandB_row_start * BLK_M * stride_bk

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    scale_factor = tl.full([1], scale_factor, dtype=tl.float32)

    accumulator = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(num_blocks * BLK_M, BLK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        if scale_quant:
            accumulator = (
                tl.dot(a, b, allow_tf32=allow_tf32) * scale_factor + accumulator
            )
            # accumulator = tl.dot(a, b, accumulator, allow_tf32=allow_tf32)
        else:
            accumulator = tl.dot(a, b, accumulator, allow_tf32=allow_tf32)
        a_ptrs += BLK_K * stride_ak
        b_ptrs += BLK_K * stride_bk

    if scale_quant:
        c = accumulator
    else:
        c = accumulator.to(tl.float16)

    offs_cm = bandC_blk_row * BLK_M + tl.arange(0, BLK_M)
    offs_cn = bandC_blk_col * BLK_N + tl.arange(0, BLK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    tl.store(c_ptrs, c)