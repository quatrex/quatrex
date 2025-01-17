import torch

import triton
import triton.language as tl
import scipy as sp
import torch.nn.functional as F
from einops import rearrange


has_gpu = torch.cuda.is_available()


def calculate_bandwidth(A: torch.Tensor) -> int:
    """
    Calculate the bandwidth `b` of a given band matrix A using parallelized tensor operations.

    Args:
        A (torch.Tensor): A square matrix of size (n, n).

    Returns:
        int: The bandwidth `b`.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix A must be square.")

    # Get the indices of all non-zero elements
    non_zero_indices = torch.nonzero(A, as_tuple=True)

    # Calculate the row and column differences
    row_indices, col_indices = non_zero_indices
    abs_diff = torch.abs(row_indices - col_indices)

    # Find the maximum difference
    max_band = torch.max(abs_diff).item()

    # Calculate the bandwidth
    return 2 * max_band + 1


def cdiv(a: torch.Tensor, b: torch.Tensor):
    return int(torch.div(a - 1, b, rounding_mode="floor") + 1)


def dense_mixed_precision_band_matrix(a: torch.Tensor, single_distance, half_distance):
    m, n = a.shape
    for i in range(m):
        for j in range(n):
            if abs(i - j) >= single_distance:
                # convert back and forth, just to mimic the loss of precision
                a[i, j] = a[i, j].float().double()
            elif abs(i - j) >= half_distance:
                # convert back and forth, just to mimic the loss of precision
                a[i, j] = a[i, j].half().double()


def get_num_blocks(band, blk_size) -> (int, int):
    dist_from_diag = band // 2
    num_blocks = cdiv(dist_from_diag, blk_size) * 2 + 1
    first_block_offset = blk_size - dist_from_diag % blk_size
    return num_blocks, first_block_offset



def tallNskinny_to_blkTallNSkinny(
    a_band: torch.tensor, blk_size: int = 1, zero_out=False
):
    """
    Convert a tall and skinny matrix to a block tall and skinny matrix.
    """
    B, m, band = a_band.shape

    num_blocks, block_offset = get_num_blocks(band, blk_size)
    num_blocks_offdiag = num_blocks // 2

    # Create a tensor to store the result
    A_blk = torch.zeros(
        (B, m, blk_size * num_blocks),
        device=a_band.device,
        dtype=a_band.dtype,
    )
    # copy the bands
    for b in range(B):
        for blk_row in range(0, m, blk_size):
            for row_in_blk in range(blk_size):
                A_blk[
                    b,
                    blk_row + row_in_blk,
                    (block_offset + row_in_blk) : (block_offset + row_in_blk + band),
                ] = a_band[b, blk_row + row_in_blk, :]

        if zero_out:
            for b_i in range(num_blocks_offdiag):
                for b_j in range(num_blocks_offdiag - b_i):
                    A_blk[
                        b,
                        b_i * blk_size : (b_i + 1) * blk_size,
                        b_j * blk_size : (b_j + 1) * blk_size,
                    ] = 0

                    b_lrow = m // blk_size - b_i - 1
                    b_lcol = num_blocks - b_j - 1
                    A_blk[
                        b,
                        b_lrow * blk_size : (b_lrow + 1) * blk_size,
                        b_lcol * blk_size : (b_lcol + 1) * blk_size,
                    ] = 0

    return A_blk


def shortNfat_to_blkShortNFat(b_band: torch.tensor, blk_size: int = 1, zero_out=False):
    """
    Convert a short and fat matrix to a block short and fat matrix.
    """
    B, band, n = b_band.shape

    num_blocks, block_offset = get_num_blocks(band, blk_size)
    num_blocks_offdiag = num_blocks // 2

    # Create a tensor to store the result
    B_blk = torch.zeros(
        (B, blk_size * num_blocks, n),
        device=b_band.device,
        dtype=b_band.dtype,
    )

    # copy the bands
    for b in range(B):
        for blk_col in range(0, n, blk_size):
            for col_in_blk in range(blk_size):
                B_blk[
                    b,
                    (block_offset + col_in_blk) : (block_offset + col_in_blk + band),
                    blk_col + col_in_blk,
                ] = b_band[b, :, blk_col + col_in_blk]

        # zero out the first and last block cols
        if zero_out:
            for b_i in range(num_blocks_offdiag):
                for b_j in range(num_blocks_offdiag - b_i):
                    B_blk[
                        b,
                        b_i * blk_size : (b_i + 1) * blk_size,
                        b_j * blk_size : (b_j + 1) * blk_size,
                    ] = 0

                    b_lrow = num_blocks - b_i - 1
                    b_lcol = n // blk_size - b_j - 1
                    # print(f"b_i={b_i}, b_j={b_j}, b_lrow={b_lrow}, b_lcol={b_lcol}")
                    B_blk[
                        b,
                        b_lrow * blk_size : (b_lrow + 1) * blk_size,
                        b_lcol * blk_size : (b_lcol + 1) * blk_size,
                    ] = 0

    return B_blk


def blkTallNSkinny_to_tallNskinny(
    a_blk: torch.tensor, blk_size: int = 1, band: int = 1
):
    """
    Convert a block tall and skinny matrix to a tall and skinny matrix.
    """
    batch, m, _ = a_blk.shape
    # assert m % blk_size == 0, "Block size must divide the band size"

    # Create a tensor to store the result
    A_band = torch.zeros(batch, m, band, device=a_blk.device, dtype=a_blk.dtype)

    # copy the bands
    for b in range(batch):
        for blk_row in range(0, m, blk_size):
            for row_in_blk in range(blk_size):
                A_blk_band_row = a_blk[
                    b, blk_row + row_in_blk, row_in_blk : row_in_blk + band
                ]
                A_band[b, blk_row + row_in_blk, :] = F.pad(
                    A_blk_band_row, (0, band - A_blk_band_row.shape[0])
                )

        # remove last blk_size - 1 columns
    A_band = A_band[:, :, : -blk_size + 1]

    return A_band


def blkTallNSkinny_to_dense(
    c_blk: torch.tensor, blk_size: int = 1, band_a: int = 1, band_b: int = 1
):
    band_a, _ = get_num_blocks(band_a, blk_size)
    band_b, _ = get_num_blocks(band_b, blk_size)

    diag_dist_a = band_a // 2
    diag_dist_b = band_b // 2
    diag_dist_c = diag_dist_a + diag_dist_b
    band_c = 2 * diag_dist_c + 1
    
    c_diag = blkTallNSkinny_to_tallNskinny(c_blk, blk_size, blk_size * band_c)
    
    c_dense = tallNskinny_to_dense_banded(c_diag) 
    return c_dense


def extract_last_diagonal(A):
    m, n = A.shape
    mid = n // 2  # Calculate the middle column index
    # Start at the middle of the last row
    row_start = m - 1
    col_start = mid

    # Length of the diagonal
    diag_length = min(mid, m)

    # Generate indices for the diagonal
    rows = torch.arange(row_start, row_start - diag_length, -1)
    cols = torch.arange(col_start, col_start + diag_length)

    # Extract diagonal elements using advanced indexing
    diagonal = A[rows, cols]
    return diagonal


def extract_first_diagonal(A):
    m, n = A.shape
    mid = n // 2  # Calculate the middle column index
    # Start at the middle of the last row
    row_start = 0  # m - 1
    col_start = mid

    # Length of the diagonal
    diag_length = min(mid, m)

    # Generate indices for the diagonal
    rows = torch.arange(row_start, row_start + diag_length)
    cols = torch.arange(col_start, col_start - diag_length, -1)

    # Extract diagonal elements using advanced indexing
    diagonal = A[rows, cols]
    return diagonal


def tallNskinny_to_dense_banded(a_band: torch.tensor, dist_from_diag: int = None):
    """
    Convert a tall and skinny banded matrix back to a dense banded matrix.
    """
    batch, m, band = a_band.shape
    if not dist_from_diag:
        dist_from_diag = band // 2

    # Create indices for rows and columns in the dense matrix
    rows = torch.arange(m).repeat(band, 1).T  # m x band
    cols = torch.arange(-dist_from_diag, dist_from_diag + 1).repeat(m, 1) + rows

    # Clip column indices to ensure they're valid
    cols = cols.clamp(0, m - 1)

    # Mask to handle valid positions
    valid_mask = (cols >= 0) & (cols < m)

    # Initialize dense matrix
    A_dense = torch.zeros((batch, m, m), device=a_band.device, dtype=a_band.dtype)

    # Use scatter_add_ to populate the dense matrix
    for b in range(batch):
        A_dense[b].index_put_(
            (rows[valid_mask], cols[valid_mask]), a_band[b, valid_mask]
        )

        # somehow, the last column of A_dense is not populated. This corresponds to the last diagonal of a_band
        # This diagonal starts at [m-1, dist_from_diag] and ends at [m-1-dist_from_diag, band - 1]
        # We need to populate the last column of A_dense with this diagonal
        last_diag = torch.flip(extract_last_diagonal(a_band[b]), dims=[0])
        # last_diag = last_diag[::-1]
        # reshape last_diag to a column vector
        # last_diag = last_diag.reshape(-1, 1)
        A_dense[b, -dist_from_diag:m, -1] = last_diag

        # It seems that the first column of A_dense is also incorrectly populated.
        first_diag = extract_first_diagonal(a_band[b])
        A_dense[b, :dist_from_diag, 0] = first_diag

    return A_dense


def blkTallNskinny_to_denseBlock(
    a_blk: torch.tensor,
    BLK_M: int,
    band_a: int,
    BLK_N: int = None,
    band_b: int = None,
    align_to_corner=True,
):
    if not BLK_N:
        BLK_N = BLK_M
    band_a, _ = get_num_blocks(band_a, BLK_M)
    if band_b:
        band_b, _ = get_num_blocks(band_b, BLK_N)
    else:
        band_b = 0

    diag_dist_a = band_a // 2
    diag_dist_b = band_b // 2
    diag_dist_c = diag_dist_a + diag_dist_b
    band_c = 2 * diag_dist_c + 1

    M = a_blk.shape[0]

    num_blocks_M = cdiv(M, BLK_M)
    num_blocks_N = band_c
    reshaped_c = rearrange(
        a_blk,
        "(m_b b_m) (n_b b_n) -> m_b n_b b_m b_n",
        m_b=num_blocks_M,
        n_b=num_blocks_N,
        b_m=BLK_M,
        b_n=BLK_N,
    )

    if align_to_corner:
        # in the first row of blocks, we start with diag_dist_c empty blocks that
        # correspond to a "negative" column indices. We shift the first diag_dist_c
        # block rows to the left to remove the empty blocks.
        for i in range(diag_dist_c):
            reshaped_c[i, :] = torch.concat(
                (reshaped_c[i, diag_dist_c - i :], reshaped_c[i, : diag_dist_c - i])
            )
    return reshaped_c


def dense_banded_to_shortAndFat(b: torch.tensor, band: int):
    B, m, n = b.shape
    dist_from_diag = band // 2

    # Calculate indices for bands and columns
    cols = torch.arange(n).repeat(band, 1)  # band x n
    rows = (
        torch.arange(-dist_from_diag, dist_from_diag + 1).unsqueeze(1).repeat(1, n)
        + cols
    )

    # Clip row indices to stay within bounds
    rows = rows.clamp(0, m - 1)

    # Gather elements
    B_band = b[:, rows, cols]
    return B_band


def dense_banded_to_blkTallNSkinny(a: torch.tensor, band: int, blk_size: int = 1):
    # if a is 2D, add the batch dimension b = 1 and convert it to 3D
    if len(a.shape) == 2:
        a = a.unsqueeze(0)
    A_tallNSkinny = dense_banded_to_tallNskinny(a, band)
    A_blk = tallNskinny_to_blkTallNSkinny(A_tallNSkinny, blk_size, zero_out=True)
    return A_blk


def dense_banded_to_blkShortNFat(b: torch.tensor, band: int, blk_size: int = 1):
    # if a is 2D, add the batch dimension b = 1 and convert it to 3D
    if len(b.shape) == 2:
        b = b.unsqueeze(0)
    B_shortNFat = dense_banded_to_shortAndFat(b, band)
    B_blk = shortNfat_to_blkShortNFat(B_shortNFat, blk_size, zero_out=True)
    return B_blk


def csr_banded_to_blkTallNSkinny(
    a_sparse: sp.sparse.csr_matrix, band: int, blk_size: int = 1
):
    A_tallNSkinny = csr_banded_to_tallNskinny(a_sparse, band)
    A_blk = tallNskinny_to_blkTallNSkinny(A_tallNSkinny, blk_size)
    return A_blk


def csr_banded_to_blkShortNFat(
    b_sparse: sp.sparse.csr_matrix, band: int, blk_size: int = 1
):
    B_shortNFat = csr_banded_to_shortAndFat(b_sparse, band)
    B_blk = shortNfat_to_blkShortNFat(B_shortNFat, blk_size)
    return B_blk


def dense_banded_to_tallNskinny(a: torch.tensor, band: int):
    b, m, n = a.shape
    dist_from_diag = band // 2

    # Calculate indices for rows and bands
    rows = torch.arange(m).repeat(band, 1).T  # m x band
    cols = torch.arange(-dist_from_diag, dist_from_diag + 1).repeat(m, 1) + rows

    # Clip column indices to stay within bounds
    cols = cols.clamp(0, n - 1)

    # Gather elements
    A_band = a[:, rows, cols]
    return A_band


def csr_banded_to_tallNskinny(a_csr: sp.sparse.csr_matrix, band: int):
    """
    Convert a banded matrix to a tall and skinny matrix.
    """
    m, n = a_csr.shape
    assert m == n, "Input matrix must be square"

    # Create a tensor to store the result
    A_band = torch.zeros(m, band)

    dist_from_diag = band // 2

    # iterate over the CSR matrix strutcture
    indices, indptr, data = a_csr.indices, a_csr.indptr, a_csr.data
    data = torch.tensor(data)
    for row in range(m):
        for j in range(indptr[row], indptr[row + 1]):
            col = indices[j]
            if abs(row - col) <= band:
                A_band[row, col - row + dist_from_diag] = data[j]
            else:
                # throw an error that the matrix is not banded
                raise ValueError(
                    "Matrix is not banded with the specified band size of {}".format(
                        band
                    )
                )
    return A_band


def csr_banded_to_shortAndFat(b_csr: sp.sparse.csr_matrix, band: int):
    """
    Convert a banded matrix to a tall and skinny matrix.
    """
    m, n = b_csr.shape
    assert m == n, "Input matrix must be square"
    # assert m > 2 * band, "Matrix must be larger than 2*band"

    # Create a tensor to store the result
    B_band = torch.zeros(band, n)

    dist_from_diag = band // 2

    # iterate over the CSR matrix strutcture
    indices, indptr, data = b_csr.indices, b_csr.indptr, b_csr.data
    data = torch.tensor(data)
    for row in range(m):
        for j in range(indptr[row], indptr[row + 1]):
            col = indices[j]
            if abs(row - col) <= band:
                B_band[row - col + dist_from_diag, col] = data[j]
            else:
                # throw an error that the matrix is not banded
                raise ValueError(
                    "Matrix is not banded with the specified band size of {}".format(
                        band
                    )
                )
    return B_band


if has_gpu:
    def is_cuda():
        return triton.runtime.driver.active.get_current_target().backend == "cuda"

    def is_hip_mi200():
        target = triton.runtime.driver.active.get_current_target()
        return target.backend == "hip" and target.arch == "gfx90a"

    def get_cuda_autotune_config():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=5,
                num_warps=2,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=5,
                num_warps=2,
            ),
            # Good config for fp8 inputs.
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
        ]

    def get_hip_autotune_config():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 16,
                    "GROUP_SIZE_M": 1,
                    "waves_per_eu": 2,
                },
                num_warps=4,
                num_stages=2,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 16,
                    "GROUP_SIZE_M": 4,
                    "waves_per_eu": 2,
                },
                num_warps=8,
                num_stages=2,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 1,
                    "waves_per_eu": 2,
                },
                num_warps=8,
                num_stages=2,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                    "waves_per_eu": 3,
                },
                num_warps=4,
                num_stages=2,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 1,
                    "waves_per_eu": 8,
                },
                num_warps=4,
                num_stages=2,
            ),
        ]

    def get_autotune_config():
        if is_cuda():
            return get_cuda_autotune_config()
        else:
            return get_hip_autotune_config()

    @triton.autotune(
        configs=get_autotune_config(),
        key=["M", "N", "K"],
    )
    @triton.jit
    def matmul_kernel(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        # Matrix dimensions
        M,
        N,
        K,
        stride_am,
        stride_ak,  #
        stride_bk,
        stride_bn,  #
        stride_cm,
        stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr,  #
    ):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        # See above `L2 Cache Optimizations` section for details.
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # We accumulate along the K dimension.
            accumulator = tl.dot(a, b, accumulator)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        # You can fuse arbitrary activation functions here
        # while the accumulator is still in FP32!
        if ACTIVATION == "leaky_relu":
            accumulator = leaky_relu(accumulator)
        c = accumulator.to(tl.float16)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    # We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
    @triton.jit
    def leaky_relu(x):
        return tl.where(x >= 0, x, 0.01 * x)

    # %%
    # We can now create a convenience wrapper function that only takes two input tensors,
    # and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.

    def matmul(a, b, activation=""):
        # Check constraints.
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        M, K = a.shape
        K, N = b.shape
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        # print(f"triton default matmul. Params: M={M}, N={N}, K={K}")
        matmul_kernel[grid](
            a,
            b,
            c,  #
            M,
            N,
            K,  #
            a.stride(0),
            a.stride(1),  #
            b.stride(0),
            b.stride(1),  #
            c.stride(0),
            c.stride(1),  #
            ACTIVATION=activation,  #
        )
        # print(f"best config: {matmul_kernel.best_config}")
        return c

    # @triton.jit
    def log_triton(log_ptr, pid, data, offset):
        # offset = data.shape[0] + 1
        for i in range(offset):
            datum = data[i]
            pid_ptr = log_ptr + pid * offset + i + tl.arange(0, 1)
            pid_data = tl.zeros((1,), dtype=tl.int32)
            pid_data += datum
            tl.store(pid_ptr, pid_data)

    @triton.jit
    def matmul_band_mixed_precision_kernel(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        # Matrix dimensions
        M,
        band_a,
        band_b,
        band_c,
        diag_dist_a,
        diag_dist_b,
        diag_dist_c,
        num_pids_row,
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
        variant: tl.constexpr,  #
        allow_tf32: tl.constexpr,
        # Meta-parameters
        BLK_M: tl.constexpr,
        BLK_N: tl.constexpr,
        BLK_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        log,
    ):
        batch_pid = tl.program_id(axis=0)

        # move the pointers of a_ptr, b_ptr, c_ptr to the current batch
        a_ptr += batch_pid * stride_ab
        b_ptr += batch_pid * stride_bb
        c_ptr += batch_pid * stride_cb

        pid = tl.program_id(axis=1)
        bandC_row = pid // num_pids_col
        bandC_col = pid % num_pids_col

        # (bandC_row, bandC_col) is the coordinate of block in bandC that this program is responsible for.
        # the corresponding coordinate of bloack in "dense" matrix C is:
        denseC_row = bandC_row
        denseC_col = bandC_row + bandC_col - diag_dist_c

        num_denseC_blocks = tl.cdiv(M, BLK_N)

        bandA_col_start = max(0, denseC_col - denseC_row + diag_dist_a - diag_dist_b)
        bandB_row_start = max(0, -denseC_col + denseC_row - diag_dist_a + diag_dist_b)

        num_blocks = min(band_a - bandA_col_start, band_b - bandB_row_start)

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_am = (denseC_row * BLK_M + tl.arange(0, BLK_M)) % M
        offs_bn = (denseC_col * BLK_N + tl.arange(0, BLK_N)) % M
        offs_k = tl.arange(0, BLK_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # print(f"Pid: {pid}")
        # cannot use f-string instide triton.jit
        # print("Pid: " + pid)
        # tl.static_print("Pid: " + str(pid))
        # log += "Pid: " + str(pid) + "\n"
        # tl.device_print(
        #     f"Pid: {pid}, ({bandC_row}, {bandC_col}), bandA_col_start={bandA_col_start}, bandB_row_start={bandB_row_start}, num_blocks={num_blocks}"
        # )

        # ################
        # ### LOGGING ####
        # ################
        # doing_work = 0

        # if denseC_col >= num_denseC_blocks:
        #     doing_work = 10
        # if denseC_row >= num_denseC_blocks:
        #     doing_work = 100
        # if denseC_col < 0:
        #     doing_work = -10
        # if denseC_row < 0:
        #     doing_work = -100
        # pid_ptr = log + pid * 16 + tl.arange(0, 1)
        # pid_data = tl.zeros((1,), dtype=tl.int32)
        # pid_data += pid
        # tl.store(pid_ptr, pid_data)

        # bandC_row_ptr = log + pid * 16 + 1 + tl.arange(0, 1)
        # bandC_row_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_row_data += bandC_row
        # tl.store(bandC_row_ptr, bandC_row_data)

        # bandC_col_ptr = log + pid * 16 + 2 + tl.arange(0, 1)
        # bandC_col_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_col_data += bandC_col
        # tl.store(bandC_col_ptr, bandC_col_data)

        # bandA_col_start_ptr = log + pid * 16 + 3 + tl.arange(0, 1)
        # bandA_col_start_data = tl.zeros((1,), dtype=tl.int32)
        # bandA_col_start_data += bandA_col_start
        # tl.store(bandA_col_start_ptr, bandA_col_start_data)

        # bandB_row_start_ptr = log + pid * 16 + 4 + tl.arange(0, 1)
        # bandB_row_start_data = tl.zeros((1,), dtype=tl.int32)
        # bandB_row_start_data += bandB_row_start
        # tl.store(bandB_row_start_ptr, bandB_row_start_data)

        # num_blocks_ptr = log + pid * 16 + 5 + tl.arange(0, 1)
        # num_blocks_data = tl.zeros((1,), dtype=tl.int32)
        # num_blocks_data += num_blocks
        # tl.store(num_blocks_ptr, num_blocks_data)

        # bandC_row_ptr = log + pid * 16 + 6 + tl.arange(0, 1)
        # bandC_row_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_row_data += denseC_row
        # tl.store(bandC_row_ptr, bandC_row_data)

        # bandC_col_ptr = log + pid * 16 + 7 + tl.arange(0, 1)
        # bandC_col_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_col_data += denseC_col
        # tl.store(bandC_col_ptr, bandC_col_data)

        # bandC_col_ptr = log + pid * 16 + 8 + tl.arange(0, 1)
        # bandC_col_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_col_data += doing_work
        # tl.store(bandC_col_ptr, bandC_col_data)

        # bandC_col_ptr = log + pid * 16 + 9 + tl.arange(0, 1)
        # bandC_col_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_col_data += bandC_row * BLK_M * stride_cm
        # tl.store(bandC_col_ptr, bandC_col_data)

        # bandC_col_ptr = log + pid * 16 + 10 + tl.arange(0, 1)
        # bandC_col_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_col_data += bandC_col * BLK_N * stride_cn
        # tl.store(bandC_col_ptr, bandC_col_data)

        # if pid != 2:
        #     return
        # ##################
        # # END OF LOGGING #
        # ##################

        if denseC_col >= num_denseC_blocks:
            return
        if denseC_row >= num_denseC_blocks:
            return
        if denseC_col < 0:
            return
        if denseC_row < 0:
            return

        # log_data = [pid, bandA_col_start, bandB_row_start, num_blocks]
        # log_triton(log, pid, log_data, offset=4)

        # logs_data += torch.tensor(
        #     [
        #         band_a,
        #         band_b,
        #         band_c,
        #         bandA_col_start,
        #         bandB_row_start,
        #         num_blocks,
        #         0,
        #         0,
        #     ],
        #     type=tl.int32,
        # )
        # logs_data += pid
        # concat = tl.cat(logs_data, logs_data)
        # tl.store(log_ptrs, concat)
        # log[pid] =
        # # move the pointers to the start of the band
        a_ptrs += bandA_col_start * BLK_N * stride_ak
        b_ptrs += bandB_row_start * BLK_M * stride_bk

        if variant == 0:
            src_dtype = tl.float16
            dest_dtype = tl.float16
        elif variant == 1:
            src_dtype = tl.float32
            dest_dtype = tl.float32
        elif variant == 2:
            src_dtype = tl.float32
            dest_dtype = tl.float16
        elif variant == 3:
            src_dtype = tl.float16
            dest_dtype = tl.float32
        elif variant == 4:
            src_dtype = tl.float64
            dest_dtype = tl.float32
        elif variant == 5:
            src_dtype = tl.float64
            dest_dtype = tl.float16
        elif variant == 6:
            src_dtype = tl.float64
            dest_dtype = tl.float64
        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)

        if variant > 1:
            for k in range(0, tl.cdiv(num_blocks * BLK_M, BLK_K)):
                # Load the next block of A and B, generate a mask by checking the K dimension.
                # If it is out of bounds, set it to 0.
                a = tl.load(a_ptrs).to(
                    dest_dtype
                )  # , mask=offs_k[None, :] < band_a - k * BLK_K, other=0.0)
                b = tl.load(b_ptrs).to(
                    dest_dtype
                )  # , mask=offs_k[:, None] < band_b - k * BLK_K, other=0.0)
                # We accumulate along the K dimension.
                accumulator = tl.dot(a, b, accumulator, allow_tf32=allow_tf32)
                # accumulator += b
                # Advance the ptrs to the next K block.
                a_ptrs += BLK_K * stride_ak
                b_ptrs += BLK_K * stride_bk
            c = accumulator.to(tl.float16)
        else:
            # a_ptrs += BLK_K * stride_ak
            # b_ptrs += BLK_K * stride_bk
            for k in range(tl.cdiv(num_blocks * BLK_M, BLK_K)):
                # Load the next block of A and B, generate a mask by checking the K dimension.
                # If it is out of bounds, set it to 0.
                a = tl.load(
                    a_ptrs
                )  # , mask=offs_k[None, :] < band_a - k * BLK_K, other=0.0)
                b = tl.load(
                    b_ptrs
                )  # , mask=offs_k[:, None] < band_b - k * BLK_K, other=0.0)
                # We accumulate along the K dimension.
                accumulator = tl.dot(a, b, accumulator)  # , allow_tf32=allow_tf32) + 1
                # accumulator = accumulator + b.to(tl.float32)
                # Advance the ptrs to the next K block.
                a_ptrs += BLK_K * stride_ak
                b_ptrs += BLK_K * stride_bk
            c = accumulator.to(tl.float16)

        # if True:  # denseC_col == 1 and denseC_row == 1:
        #     c += 1
        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = bandC_row * BLK_M + tl.arange(0, BLK_M)
        offs_cn = bandC_col * BLK_N + tl.arange(0, BLK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        # c_ptrs = (
        #     c_ptr
        #     + stride_cm * tl.arange(0, BLK_M)[:, None]
        #     + stride_cn * tl.arange(0, BLK_N)[None, :]
        # )
        # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
        # tl.store(c_ptrs, c, mask=c_mask)
        tl.store(c_ptrs, c)

    @triton.jit
    def matmul_band_FP64_kernel(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        # Matrix dimensions
        M,
        band_a,
        band_b,
        band_c,
        diag_dist_a,
        diag_dist_b,
        diag_dist_c,
        num_pids_row,
        num_pids_col,
        stride_am,
        stride_ak,  #
        stride_bk,
        stride_bn,  #
        stride_cm,
        stride_cn,
        # Meta-parameters
        BLK_M: tl.constexpr,
        BLK_N: tl.constexpr,
        BLK_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        log,
    ):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """

        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        # See above `L2 Cache Optimizations` section for details.
        pid = tl.program_id(axis=0)
        # num_pid_in_group = GROUP_SIZE_M * num_pid_n # 9
        # group_id = pid // num_pid_in_group # 0
        # first_pid_m = group_id * GROUP_SIZE_M # 0
        # group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        # bandC_row = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        # bandC_col = (pid % num_pid_in_group) // group_size_m
        bandC_row = pid // num_pids_col
        bandC_col = pid % num_pids_col

        # (pid_m, pid_n) is the coordinate of block in bandC that this program is responsible for.
        # the corresponding coordinate of bloack in "dense" matrix C is:
        denseC_row = bandC_row
        denseC_col = bandC_row + bandC_col - diag_dist_c

        num_denseC_blocks = tl.cdiv(M, BLK_N)

        bandA_col_start = max(0, denseC_col - denseC_row + diag_dist_a - diag_dist_b)
        bandB_row_start = max(0, -denseC_col + denseC_row - diag_dist_a + diag_dist_b)

        num_blocks = min(band_a - bandA_col_start, band_b - bandB_row_start)

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_am = (denseC_row * BLK_M + tl.arange(0, BLK_M)) % M
        offs_bn = (denseC_col * BLK_N + tl.arange(0, BLK_N)) % M
        offs_k = tl.arange(0, BLK_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # print(f"Pid: {pid}")
        # cannot use f-string instide triton.jit
        # print("Pid: " + pid)
        # tl.static_print("Pid: " + str(pid))
        # log += "Pid: " + str(pid) + "\n"
        # tl.device_print(
        #     f"Pid: {pid}, ({bandC_row}, {bandC_col}), bandA_col_start={bandA_col_start}, bandB_row_start={bandB_row_start}, num_blocks={num_blocks}"
        # )

        # ################
        # ### LOGGING ####
        # ################
        # doing_work = 0

        # if denseC_col >= num_denseC_blocks:
        #     doing_work = 10
        # if denseC_row >= num_denseC_blocks:
        #     doing_work = 100
        # if denseC_col < 0:
        #     doing_work = -10
        # if denseC_row < 0:
        #     doing_work = -100
        # pid_ptr = log + pid * 16 + tl.arange(0, 1)
        # pid_data = tl.zeros((1,), dtype=tl.int32)
        # pid_data += pid
        # tl.store(pid_ptr, pid_data)

        # bandC_row_ptr = log + pid * 16 + 1 + tl.arange(0, 1)
        # bandC_row_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_row_data += bandC_row
        # tl.store(bandC_row_ptr, bandC_row_data)

        # bandC_col_ptr = log + pid * 16 + 2 + tl.arange(0, 1)
        # bandC_col_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_col_data += bandC_col
        # tl.store(bandC_col_ptr, bandC_col_data)

        # bandA_col_start_ptr = log + pid * 16 + 3 + tl.arange(0, 1)
        # bandA_col_start_data = tl.zeros((1,), dtype=tl.int32)
        # bandA_col_start_data += bandA_col_start
        # tl.store(bandA_col_start_ptr, bandA_col_start_data)

        # bandB_row_start_ptr = log + pid * 16 + 4 + tl.arange(0, 1)
        # bandB_row_start_data = tl.zeros((1,), dtype=tl.int32)
        # bandB_row_start_data += bandB_row_start
        # tl.store(bandB_row_start_ptr, bandB_row_start_data)

        # num_blocks_ptr = log + pid * 16 + 5 + tl.arange(0, 1)
        # num_blocks_data = tl.zeros((1,), dtype=tl.int32)
        # num_blocks_data += num_blocks
        # tl.store(num_blocks_ptr, num_blocks_data)

        # bandC_row_ptr = log + pid * 16 + 6 + tl.arange(0, 1)
        # bandC_row_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_row_data += denseC_row
        # tl.store(bandC_row_ptr, bandC_row_data)

        # bandC_col_ptr = log + pid * 16 + 7 + tl.arange(0, 1)
        # bandC_col_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_col_data += denseC_col
        # tl.store(bandC_col_ptr, bandC_col_data)

        # bandC_col_ptr = log + pid * 16 + 8 + tl.arange(0, 1)
        # bandC_col_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_col_data += doing_work
        # tl.store(bandC_col_ptr, bandC_col_data)

        # bandC_col_ptr = log + pid * 16 + 9 + tl.arange(0, 1)
        # bandC_col_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_col_data += bandC_row * BLK_M * stride_cm
        # tl.store(bandC_col_ptr, bandC_col_data)

        # bandC_col_ptr = log + pid * 16 + 10 + tl.arange(0, 1)
        # bandC_col_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_col_data += bandC_col * BLK_N * stride_cn
        # tl.store(bandC_col_ptr, bandC_col_data)

        # # if pid != 5:
        # #     return
        # # ##################
        # # # END OF LOGGING #
        # # ##################

        if denseC_col >= num_denseC_blocks:
            return
        if denseC_row >= num_denseC_blocks:
            return
        if denseC_col < 0:
            return
        if denseC_row < 0:
            return

        # log_data = [pid, bandA_col_start, bandB_row_start, num_blocks]
        # log_triton(log, pid, log_data, offset=4)

        # logs_data += torch.tensor(
        #     [
        #         band_a,
        #         band_b,
        #         band_c,
        #         bandA_col_start,
        #         bandB_row_start,
        #         num_blocks,
        #         0,
        #         0,
        #     ],
        #     type=tl.int32,
        # )
        # logs_data += pid
        # concat = tl.cat(logs_data, logs_data)
        # tl.store(log_ptrs, concat)
        # log[pid] =
        # # move the pointers to the start of the band
        a_ptrs += bandA_col_start * BLK_N * stride_ak
        b_ptrs += bandB_row_start * BLK_M * stride_bk

        a_ptr += bandA_col_start * BLK_N * stride_ak
        b_ptr += bandB_row_start * BLK_M * stride_bk

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.

        accumulator = tl.zeros((BLK_M, BLK_N), dtype=tl.float64)

        for k in range(0, tl.cdiv(num_blocks * BLK_M, BLK_K)):
            # for k in range(0, num_blocks):  # tl.cdiv(num_blocks * BLK_M, BLK_K)):
            # for k in range(0, 2):  # tl.cdiv(num_blocks * BLK_M, BLK_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(
                a_ptrs
            )  # , mask=offs_k[None, :] < band_a - k * BLK_K, other=0.0)
            b = tl.load(
                b_ptrs
            )  # , mask=offs_k[:, None] < band_b - k * BLK_K, other=0.0)
            # We accumulate along the K dimension.
            # accumulator = tl.dot(a, b, accumulator, allow_tf32=allow_tf32)
            for k_idx in range(BLK_K):
                # Calculate pointers for the k_idx-th column of a and row of b
                a_col_ptrs = (
                    a_ptr + offs_am * stride_am + (k_idx + k * BLK_K) * stride_ak
                )
                b_row_ptrs = (
                    b_ptr + (k_idx + k * BLK_K) * stride_bk + offs_bn * stride_bn
                )

                # Load the column and row
                a_col = tl.load(a_col_ptrs)  # [BLOCK_SIZE_M]
                b_row = tl.load(b_row_ptrs)  # [BLOCK_SIZE_N]

                # Outer product accumulation
                accumulator += tl.where(
                    (offs_am[:, None] < M) & (offs_bn[None, :] < M),
                    a_col[:, None] * b_row[None, :],
                    0.0,
                )
                # accumulator += a_col[:, None] * b_row[None, :]

            # accumulator = b
            # accumulator += a * b
            # Advance the ptrs to the next K block.
            a_ptrs += BLK_K * stride_ak
            b_ptrs += BLK_K * stride_bk

        # if True:  # denseC_col == 1 and denseC_row == 1:
        #     c += 1
        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = bandC_row * BLK_M + tl.arange(0, BLK_M)
        offs_cn = bandC_col * BLK_N + tl.arange(0, BLK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        # c_ptrs = (
        #     c_ptr
        #     + stride_cm * tl.arange(0, BLK_M)[:, None]
        #     + stride_cn * tl.arange(0, BLK_N)[None, :]
        # )
        # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
        # tl.store(c_ptrs, c, mask=c_mask)
        tl.store(c_ptrs, accumulator)

    variants = {
        (torch.float16, torch.float16): 0,
        (torch.float32, torch.float32): 1,
        (torch.float32, torch.float16): 2,
        (torch.float16, torch.float32): 3,
        (torch.float64, torch.float32): 4,
        (torch.float64, torch.float16): 5,
        (torch.float64, torch.float64): 6,
    }

    def matmul_band_mixed_precision(
        a,
        b,
        c,
        band_a: int,
        band_b: int = None,
        dest_dtype=torch.float16,
        allow_tf32: bool = True,
        BLK_M: int = 128,
        BLK_N: int = 128,
        BLK_K: int = 32,
    ):
        if band_b is None:
            band_b = band_a
        batch_size, M, _ = a.shape
        # band_b, N = b.shape

        band_a, _ = get_num_blocks(band_a, BLK_M)
        band_b, _ = get_num_blocks(band_b, BLK_N)

        diag_dist_a = band_a // 2
        diag_dist_b = band_b // 2
        diag_dist_c = diag_dist_a + diag_dist_b
        band_c = 2 * diag_dist_c + 1

        num_block_rows = cdiv(M, BLK_M)
        grid = (
            batch_size,
            num_block_rows * band_c,
        )

        # for batch in range(batch_size):
        if True:
            if dest_dtype == torch.float64:
                matmul_band_FP64_kernel[grid](
                    a,
                    b,
                    c,  #
                    M=M,
                    band_a=band_a,
                    band_b=band_b,
                    band_c=band_c,
                    diag_dist_a=diag_dist_a,
                    diag_dist_b=diag_dist_b,
                    diag_dist_c=diag_dist_c,
                    num_pids_row=num_block_rows,
                    num_pids_col=band_c,
                    stride_am=a.stride(0),
                    stride_ak=a.stride(1),  #
                    stride_bk=b.stride(0),
                    stride_bn=b.stride(1),  #
                    stride_cm=c.stride(0),
                    stride_cn=c.stride(1),  #
                    BLK_M=BLK_M,
                    BLK_N=BLK_N,
                    BLK_K=BLK_K,  #
                    GROUP_SIZE_M=1,  #
                    log=None,
                )
            else:
                # print(f"option 2, variant: {variants[(a.dtype, dest_dtype)]}\n\n")
                # print(f"a:\n{a.detach().cpu().numpy()[32:48, :]}")
                # print(f"b:\n{b.detach().cpu().numpy()[:, 0:16]}")
                matmul_band_mixed_precision_kernel[grid](
                    a,
                    b,
                    c,  #
                    # a[batch],
                    # b[batch],
                    # c[batch],  #
                    M=M,
                    band_a=band_a,
                    band_b=band_b,
                    band_c=band_c,
                    diag_dist_a=diag_dist_a,
                    diag_dist_b=diag_dist_b,
                    diag_dist_c=diag_dist_c,
                    num_pids_row=num_block_rows,
                    num_pids_col=band_c,
                    stride_ab=a.stride(0),
                    stride_am=a.stride(1),
                    stride_ak=a.stride(2),  #
                    stride_bb=b.stride(0),
                    stride_bk=b.stride(1),
                    stride_bn=b.stride(2),  #
                    stride_cb=c.stride(0),
                    stride_cm=c.stride(1),
                    stride_cn=c.stride(2),  #
                    variant=variants[(a.dtype, dest_dtype)],
                    allow_tf32=allow_tf32,
                    BLK_M=BLK_M,
                    BLK_N=BLK_N,
                    BLK_K=BLK_K,  #
                    GROUP_SIZE_M=1,  #
                    log=None,
                )

        return c

    def band_mm(A, B, band, band_out=None):
        """
        Matrix multiplication of two banded matrices.

        Parameters
        ----------
        A : torch.Tensor
            A banded matrix of shape (n, m).
        B : torch.Tensor
            A banded matrix of shape (m, p).
        band : int
            The bandwidth of the matrices.

        Returns
        -------
        torch.Tensor
            The product of the two matrices.
        """
        m, k = A.shape
        k, n = B.shape

        if band_out is None:
            band_out = 2 * band

        # "bandify" the matrices
        A_band = torch.zeros(m, 2 * band + 1, device=A.device, dtype=A.dtype)
        B_band = torch.zeros(2 * band + 1, n, device=A.device, dtype=A.dtype)
        C_band = torch.zeros(m, 4 * band + 1, device=A.device, dtype=A.dtype)

        # copy the bands
        for i in range(m):
            for j in range(k):
                if abs(i - j) <= band:
                    A_band[i, i - j + band] = A[i, j]

        # Create a tensor to store the result
        C = torch.zeros(m, m, device=A.device, dtype=A.dtype)

        # print parameters
        # print(f"m={m}, n={n}, k={k}, band={band}")

        # Perform the matrix multiplication
        for i in range(m):
            for j in range(n):
                for k in range(k):
                    if abs(i - k) <= band and abs(k - j) <= band:
                        # print(f"i={i}, j={j}, k={k}, A[i, k]={A[i, k]}, B[k, j]={B[k, j]}")
                        C[i, j] += A[i, k] * B[k, j]

        return C

    @triton.jit
    def matmul_band_kernel(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        # Matrix dimensions
        M,
        band_a,
        band_b,
        band_c,
        diag_dist_a,
        diag_dist_b,
        diag_dist_c,
        num_pids_row,
        num_pids_col,
        stride_am,
        stride_ak,  #
        stride_bk,
        stride_bn,  #
        stride_cm,
        stride_cn,
        # Meta-parameters
        BLK_M: tl.constexpr,
        BLK_N: tl.constexpr,
        BLK_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        log,
    ):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        # See above `L2 Cache Optimizations` section for details.
        pid = tl.program_id(axis=0)
        # num_pid_in_group = GROUP_SIZE_M * num_pid_n # 9
        # group_id = pid // num_pid_in_group # 0
        # first_pid_m = group_id * GROUP_SIZE_M # 0
        # group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        # bandC_row = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        # bandC_col = (pid % num_pid_in_group) // group_size_m
        bandC_row = pid // num_pids_col
        bandC_col = pid % num_pids_col

        # (pid_m, pid_n) is the coordinate of block in bandC that this program is responsible for.
        # the corresponding coordinate of bloack in "dense" matrix C is:
        denseC_row = bandC_row
        denseC_col = bandC_row + bandC_col - diag_dist_c

        num_denseC_blocks = tl.cdiv(M, BLK_N)

        bandA_col_start = max(0, denseC_col - denseC_row + diag_dist_a - diag_dist_b)
        bandB_row_start = max(0, -denseC_col + denseC_row - diag_dist_a + diag_dist_b)

        num_blocks = min(band_a - bandA_col_start, band_b - bandB_row_start)

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_am = (denseC_row * BLK_M + tl.arange(0, BLK_M)) % M
        offs_bn = (denseC_col * BLK_N + tl.arange(0, BLK_N)) % M
        offs_k = tl.arange(0, BLK_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # print(f"Pid: {pid}")
        # cannot use f-string instide triton.jit
        # print("Pid: " + pid)
        # tl.static_print("Pid: " + str(pid))
        # log += "Pid: " + str(pid) + "\n"
        # tl.device_print(
        #     f"Pid: {pid}, ({bandC_row}, {bandC_col}), bandA_col_start={bandA_col_start}, bandB_row_start={bandB_row_start}, num_blocks={num_blocks}"
        # )

        # ################
        # ### LOGGING ####
        # ################
        # doing_work = 0

        # if denseC_col >= num_denseC_blocks:
        #     doing_work = 10
        # if denseC_row >= num_denseC_blocks:
        #     doing_work = 100
        # if denseC_col < 0:
        #     doing_work = -10
        # if denseC_row < 0:
        #     doing_work = -100
        # pid_ptr = log + pid * 16 + tl.arange(0, 1)
        # pid_data = tl.zeros((1,), dtype=tl.int32)
        # pid_data += pid
        # tl.store(pid_ptr, pid_data)

        # bandC_row_ptr = log + pid * 16 + 1 + tl.arange(0, 1)
        # bandC_row_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_row_data += bandC_row
        # tl.store(bandC_row_ptr, bandC_row_data)

        # bandC_col_ptr = log + pid * 16 + 2 + tl.arange(0, 1)
        # bandC_col_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_col_data += bandC_col
        # tl.store(bandC_col_ptr, bandC_col_data)

        # bandA_col_start_ptr = log + pid * 16 + 3 + tl.arange(0, 1)
        # bandA_col_start_data = tl.zeros((1,), dtype=tl.int32)
        # bandA_col_start_data += bandA_col_start
        # tl.store(bandA_col_start_ptr, bandA_col_start_data)

        # bandB_row_start_ptr = log + pid * 16 + 4 + tl.arange(0, 1)
        # bandB_row_start_data = tl.zeros((1,), dtype=tl.int32)
        # bandB_row_start_data += bandB_row_start
        # tl.store(bandB_row_start_ptr, bandB_row_start_data)

        # num_blocks_ptr = log + pid * 16 + 5 + tl.arange(0, 1)
        # num_blocks_data = tl.zeros((1,), dtype=tl.int32)
        # num_blocks_data += num_blocks
        # tl.store(num_blocks_ptr, num_blocks_data)

        # bandC_row_ptr = log + pid * 16 + 6 + tl.arange(0, 1)
        # bandC_row_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_row_data += denseC_row
        # tl.store(bandC_row_ptr, bandC_row_data)

        # bandC_col_ptr = log + pid * 16 + 7 + tl.arange(0, 1)
        # bandC_col_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_col_data += denseC_col
        # tl.store(bandC_col_ptr, bandC_col_data)

        # bandC_col_ptr = log + pid * 16 + 8 + tl.arange(0, 1)
        # bandC_col_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_col_data += doing_work
        # tl.store(bandC_col_ptr, bandC_col_data)

        # bandC_col_ptr = log + pid * 16 + 9 + tl.arange(0, 1)
        # bandC_col_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_col_data += bandC_row * BLK_M * stride_cm
        # tl.store(bandC_col_ptr, bandC_col_data)

        # bandC_col_ptr = log + pid * 16 + 10 + tl.arange(0, 1)
        # bandC_col_data = tl.zeros((1,), dtype=tl.int32)
        # bandC_col_data += bandC_col * BLK_N * stride_cn
        # tl.store(bandC_col_ptr, bandC_col_data)

        # ##################
        # # END OF LOGGING #
        # ##################

        if denseC_col >= num_denseC_blocks:
            return
        if denseC_row >= num_denseC_blocks:
            return
        if denseC_col < 0:
            return
        if denseC_row < 0:
            return

        # log_data = [pid, bandA_col_start, bandB_row_start, num_blocks]
        # log_triton(log, pid, log_data, offset=4)

        # logs_data += torch.tensor(
        #     [
        #         band_a,
        #         band_b,
        #         band_c,
        #         bandA_col_start,
        #         bandB_row_start,
        #         num_blocks,
        #         0,
        #         0,
        #     ],
        #     type=tl.int32,
        # )
        # logs_data += pid
        # concat = tl.cat(logs_data, logs_data)
        # tl.store(log_ptrs, concat)
        # log[pid] =
        # # move the pointers to the start of the band
        a_ptrs += bandA_col_start * BLK_N * stride_ak
        b_ptrs += bandB_row_start * BLK_M * stride_bk

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(num_blocks * BLK_M, BLK_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(
                a_ptrs
            )  # , mask=offs_k[None, :] < band_a - k * BLK_K, other=0.0)
            b = tl.load(
                b_ptrs
            )  # , mask=offs_k[:, None] < band_b - k * BLK_K, other=0.0)
            # We accumulate along the K dimension.
            accumulator = tl.dot(a, b, accumulator)
            # accumulator += b
            # Advance the ptrs to the next K block.
            a_ptrs += BLK_K * stride_ak
            b_ptrs += BLK_K * stride_bk
        c = accumulator.to(tl.float16)

        # if True:  # denseC_col == 1 and denseC_row == 1:
        #     c += 1
        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = bandC_row * BLK_M + tl.arange(0, BLK_M)
        offs_cn = bandC_col * BLK_N + tl.arange(0, BLK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        # c_ptrs = (
        #     c_ptr
        #     + stride_cm * tl.arange(0, BLK_M)[:, None]
        #     + stride_cn * tl.arange(0, BLK_N)[None, :]
        # )
        # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
        # tl.store(c_ptrs, c, mask=c_mask)
        tl.store(c_ptrs, c)

    # %%
    # We can now create a convenience wrapper function that only takes two input tensors,
    # and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.

    def matmul_band(
        a,
        b,
        c,
        band_a: int,
        band_b: int = None,
        BLK_M: int = 128,
        BLK_N: int = 128,
        BLK_K: int = 32,
    ):
        if band_b is None:
            band_b = band_a
        M, _ = a.shape
        # band_b, N = b.shape

        band_a, _ = get_num_blocks(band_a, BLK_M)
        band_b, _ = get_num_blocks(band_b, BLK_N)

        diag_dist_a = band_a // 2
        diag_dist_b = band_b // 2
        diag_dist_c = diag_dist_a + diag_dist_b
        band_c = 2 * diag_dist_c + 1

        num_block_rows = cdiv(M, BLK_M)
        # band_c, _ = get_num_blocks(band_c, BLK_N)

        # # Check constraints.
        # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        # assert a.is_contiguous(), "Matrix A must be contiguous"

        # # Allocates output.
        # # c = torch.empty(
        # #     (M, min(BLK_M * cdiv(band_c + BLK_M - 1, BLK_M), M)),
        # #     device=a.device,
        # #     dtype=a.dtype,
        # # )

        # c = torch.zeros(
        #     (M, BLK_M * band_c),
        #     device=a.device,
        #     dtype=a.dtype,
        # )
        # 1D launch kernel where each block gets its own program.
        # grid = (cdiv(M, BLK_M) * cdiv(band_c, BLK_N),)
        grid = (num_block_rows * band_c,)

        # print(
        #     f"triton matmul_band. Params: M={M}, band_a={band_a}, band_b={band_b}, band_c={band_c}"
        # )
        # print(f"BLK_M={BLK_M}, BLK_N={BLK_N}, BLK_K={BLK_K}")
        # # print(f"Matrices:\na (shape {a.shape})\n{a}\nb (shape {b.shape})\n{b}")
        # print(f"Grid: {grid} = {num_block_rows} * {band_c}")
        # # exit()
        # log = torch.zeros(grid[0] * 16, dtype=torch.int32, device=a.device)
        matmul_band_kernel[grid](
            a,
            b,
            c,  #
            M=M,
            band_a=band_a,
            band_b=band_b,
            band_c=band_c,
            diag_dist_a=diag_dist_a,
            diag_dist_b=diag_dist_b,
            diag_dist_c=diag_dist_c,
            num_pids_row=num_block_rows,
            num_pids_col=band_c,
            stride_am=a.stride(0),
            stride_ak=a.stride(1),  #
            stride_bk=b.stride(0),
            stride_bn=b.stride(1),  #
            stride_cm=c.stride(0),
            stride_cn=c.stride(1),  #
            BLK_M=BLK_M,
            BLK_N=BLK_N,
            BLK_K=BLK_K,  #
            GROUP_SIZE_M=1,  #
            log=None,
        )
        # log = log.reshape(grid[0], 16)
        # log = log[:, :11]
        # print(
        #     [
        #         "pid",
        #         "bandC_row",
        #         "bandC_col",
        #         "bandA_col_start",
        #         "bandB_row_start",
        #         "num_blocks",
        #         "denseC_row",
        #         "denseC_col",
        #     ]
        # )
        # import pandas as pd

        # # set pandas display options, set the maximum number of rows to display
        # pd.set_option("display.max_rows", 1000)

        # df = pd.DataFrame(
        #     log.cpu().numpy(),
        #     columns=[
        #         "pid",
        #         "bandC_row",
        #         "bandC_col",
        #         "bandA_col_start",
        #         "bandB_row_start",
        #         "num_blocks",
        #         "denseC_row",
        #         "denseC_col",
        #         "doing_work",
        #         "c_row_offset",
        #         "c_col_offset",
        #     ],
        # )
        # print(df)
        # exit()

        # num_blocks_M = cdiv(M, BLK_M)
        # num_blocks_N = band_c
        # reshaped_c = rearrange(
        #     c,
        #     "(m_b b_m) (n_b b_n) -> m_b n_b b_m b_n",
        #     m_b=num_blocks_M,
        #     n_b=num_blocks_N,
        #     b_m=BLK_M,
        #     b_n=BLK_N,
        # )

        # c_diag = blkTallNSkinny_to_tallNskinny(c, BLK_M, BLK_M * band_c)
        # c_dense = tallNskinny_to_dense_banded(
        #     c_diag, dist_from_diag=diag_dist_c * BLK_M
        # )
        # print(f"c_diag:\n{c_diag}")
        # print(f"c_dense:\n{c_dense}")
        # print(f"Result:\nc\n{c_dense}?")
        # # print(f"Result shape: {reshaped_c.shape}")
        # for b_m in range(num_blocks_M):
        #     for b_n in range(num_blocks_N):
        #         print(f"Block {b_m}, {b_n}")
        #         print(reshaped_c[b_m, b_n])
        # # print(f"\n\nfull:\n{c}")
        # print(f"\nfirst block cols:\n{c[:, :BLK_M]}")
        # print(f"\nlast block cols:\n{c[:, -BLK_M:]}")
        # exit()
        return c
