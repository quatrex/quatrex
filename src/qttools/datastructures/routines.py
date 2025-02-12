# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

torch_cuda_avail = False
try:
    import torch
    if torch.cuda.is_available():
        from qttools.kernels import banded_kernels
        torch_cuda_avail = True
except (ImportError, ModuleNotFoundError):
    pass

from qttools.datastructures.dsbsparse import DSBSparse


def _dense(arr: DSBSparse, **kwargs) -> torch.Tensor:
    """Returns the dense representation of a DSBSparse array."""

    # Keyword arguments
    BLK_M = kwargs.get("BLK_M", 16)
    BLK_N = kwargs.get("BLK_N", 16)
    BLK_K = kwargs.get("BLK_K", 16)

    arr_dense = torch.tensor(arr.to_dense(), device="cuda")
    # arr_dense is expected to be a 3D tensor: (batch_size, N, N)
    # if there are only 2 dimensions, we add a dummy dimension
    # if there are more than 3 dimensions, we concatenate (flatten) the first batch dimensions
    if len(arr_dense.shape) == 2:
        arr_dense = arr_dense.unsqueeze(0)
    elif len(arr_dense.shape) > 3:
        arr_dense = arr_dense.view(-1, arr_dense.shape[-2], arr_dense.shape[-1])

    # TODO: Is the below needed because the conversion kernels depend on N being divisible by BLK_M?
    # TODO: If yes, we need to amend the kernels to work with any N.
    N = arr_dense.shape[-1]
    # if N is not divisible by BLK_M, we pad the matrix with zeros
    if N % BLK_M != 0:
        arr_dense = torch.nn.functional.pad(arr_dense, (0, BLK_M - N % BLK_M, 0, BLK_M - N % BLK_M))
    return arr_dense


def banded_matmul(a: DSBSparse, b: DSBSparse, out: DSBSparse, **kwargs) -> None:
    """Matrix multiplication between two banded matrices. """
    if not torch_cuda_avail:
        raise NotImplementedError("Banded matrix multiplication requires PyTorch and a CUDA-enabled device.")
    
    # Keyword arguments
    source_dtype = kwargs.get("source_dtype", torch.float32)
    dest_dtype = kwargs.get("dest_dtype", torch.float32)
    allow_tf32 = kwargs.get("allow_tf32", True)
    BLK_M = kwargs.get("BLK_M", 16)
    BLK_N = kwargs.get("BLK_N", 16)
    BLK_K = kwargs.get("BLK_K", 16)

    # Densify arrays
    a_dense = _dense(a, **kwargs)
    b_dense = _dense(b, **kwargs)
    
    batch, N = a.shape[0], a.shape[1]
    bw_a = banded_kernels.calculate_bandwidth(a[0])
    bw_b = banded_kernels.calculate_bandwidth(b[0])

    A_tallNSkinnyBand = (
        banded_kernels.dense_banded_to_blkTallNSkinny(a_dense, bw_a, BLK_M)
        .to("cuda")
        .to(source_dtype)
    )
    B_shortAndFat = (
        banded_kernels.dense_banded_to_blkShortNFat(b_dense, bw_b, BLK_N)
        .to("cuda")
        .to(source_dtype)
    )

    band_a, _ = banded_kernels.get_num_blocks(bw_a, BLK_M)
    band_b, _ = banded_kernels.get_num_blocks(bw_b, BLK_N)

    diag_dist_a = band_a // 2
    diag_dist_b = band_b // 2
    diag_dist_c = diag_dist_a + diag_dist_b
    band_c = 2 * diag_dist_c + 1

    c_blkTallNSkinny = torch.zeros(
        (batch, N, BLK_M * band_c),
        device=A_tallNSkinnyBand.device,
        dtype=dest_dtype,
    )

    # print(f"A_tallNSkinnyBand: \n{A_tallNSkinnyBand[0].detach().cpu().numpy()}")
    # print(f"B_shortAndFat: \n{B_shortAndFat[0].detach().cpu().numpy()}")
    c_blkTallNSkinny = banded_kernels.matmul_band_mixed_precision(
        A_tallNSkinnyBand,
        B_shortAndFat,
        c_blkTallNSkinny,
        band_a=bw_a,
        band_b=bw_b,
        dest_dtype=dest_dtype,
        allow_tf32=allow_tf32,
        BLK_M=BLK_M,
        BLK_N=BLK_N,
        BLK_K=BLK_K,
    )
    # print(f"c_blkTallNSkinny: \n{c_blkTallNSkinny[0].detach().cpu().numpy()}")
    # exit()
    c_dense = banded_kernels.blkTallNSkinny_to_dense(
        c_blkTallNSkinny, BLK_M, band_a=bw_a, band_b=bw_b
    ).to(source_dtype)

    N_not_padded = out.shape[-1]
    out[:] = c_dense[:, :N_not_padded, :N_not_padded]
