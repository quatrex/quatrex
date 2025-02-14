# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools.datastructures import DSBSparse


def correct_out_range_index(i: int, k: int, num_blocks: int):
    # find the index of block in the matrix being repeated into open-end
    # based on the difference of row and col, ie diagonal
    diag = k - i
    k_1 = min(max(k, 0), num_blocks - 1)
    i_1 = k_1 - diag  # keep the same diag
    i_2 = min(max(i_1, 0), num_blocks - 1)
    k_2 = i_2 + diag  # keep the same diag
    return (i_2, k_2)


def bd_matmul(
    a: DSBSparse,
    b: DSBSparse,
    out: DSBSparse,
    in_num_diag: int = 3,
    out_num_diag: int = 5,
    spillover_correction: bool = False,
):
    """Matrix multiplication of two `a @ b` BD DSBSparse matrices.

    Parameters
    ----------
    a : DSBSparse
        The first block diagonal matrix.
    b : DSBSparse
        The second block diagonal matrix.
    out : DSBSparse
        The output matrix. This matrix must have the same block size as
        `a` and `b`. It will compute up to `out_numdiag`.
    in_num_diag: int
        The number of diagonals in input matrices
    out_num_diag: int
        The number of diagonals in output matrices
    spillover_correction : bool, optional
        Whether to apply spillover corrections to the output matrix.
        This is necessary when the matrices represent open-ended
        systems. The default is False.

    """
    if a.distribution_state == "nnz" or b.distribution_state == "nnz":
        raise ValueError(
            "Matrix multiplication is not supported for matrices in nnz distribution state."
        )
    num_blocks = len(a.block_sizes)

    # Make sure the output matrix is initialized to zero.
    out.data = 0

    for i in range(num_blocks):
        for j in range(
            max(i - out_num_diag // 2, 0), min(i + out_num_diag // 2 + 1, num_blocks)
        ):
            tmp = out.blocks[i, j]
            for k in range(i - in_num_diag // 2, i + in_num_diag // 2 + 1):
                if abs(j - k) > in_num_diag // 2:
                    continue
                out_range = (k < 0) or (k >= num_blocks)
                if out_range and (not spillover_correction):
                    continue
                else:
                    if out_range:
                        i_a, k_a = correct_out_range_index(i, k, num_blocks)
                        k_b, j_b = correct_out_range_index(k, j, num_blocks)
                        tmp += a.blocks[i_a, k_a] @ b.blocks[k_b, j_b]
                    else:
                        tmp += a.blocks[i, k] @ b.blocks[k, j]
            out.blocks[i, j] = tmp

    return


def bd_sandwich(
    a: DSBSparse,
    b: DSBSparse,
    out: DSBSparse,
    in_num_diag: int = 3,
    out_num_diag: int = 7,
    spillover_correction: bool = False,
):
    """Compute the sandwich product `a @ b @ a` BTD DSBSparse matrices.

    Parameters
    ----------
    a : DSBSparse
        The first block tridiagonal matrix.
    b : DSBSparse
        The second block tridiagonal matrix.
    out : DSBSparse
        The output matrix. This matrix must have the same block size as
        `a`, and `b`. It will compute up to heptadiagonal.
    in_num_diag: int
        The number of diagonals in input matrices
    out_num_diag: int
        The number of diagonals in output matrices
    spillover_correction : bool, optional
        Whether to apply spillover corrections to the output matrix.
        This is necessary when the matrices represent open-ended
        systems. The default is False.

    """
    if a.distribution_state == "nnz" or b.distribution_state == "nnz":
        raise ValueError(
            "Matrix multiplication is not supported for matrices in nnz distribution state."
        )
    num_blocks = len(a.block_sizes)

    # Make sure the output matrix is initialized to zero.
    out.data = 0

    for i in range(num_blocks):
        for j in range(
            max(i - out_num_diag // 2, 0), min(i + out_num_diag // 2 + 1, num_blocks)
        ):
            tmp = out.blocks[i, j]
            for m in range(i - in_num_diag // 2, i + in_num_diag // 2 + 1):
                for k in range(m - in_num_diag // 2, m + in_num_diag // 2 + 1):
                    if abs(j - k) > in_num_diag // 2:
                        continue
                    out_range = (
                        (k < 0) or (k >= num_blocks) or (m < 0) or (m >= num_blocks)
                    )
                    if out_range and (not spillover_correction):
                        continue
                    else:
                        if out_range:
                            a_i, a_m = correct_out_range_index(i, m, num_blocks)
                            b_m, b_k = correct_out_range_index(m, k, num_blocks)
                            a_k, a_j = correct_out_range_index(k, j, num_blocks)
                            tmp += (
                                a.blocks[a_i, a_m]
                                @ b.blocks[b_m, b_k]
                                @ a.blocks[a_k, a_j]
                            )
                        else:
                            tmp += a.blocks[i, m] @ b.blocks[m, k] @ a.blocks[k, j]
            out.blocks[i, j] = tmp

    return


def btd_matmul(
    a: DSBSparse,
    b: DSBSparse,
    out: DSBSparse,
    spillover_correction: bool = False,
):
    """Matrix multiplication of two `a @ b` BTD DSBSparse matrices.

    Parameters
    ----------
    a : DSBSparse
        The first block tridiagonal matrix.
    b : DSBSparse
        The second block tridiagonal matrix.
    out : DSBSparse
        The output matrix. This matrix must have the same block size as
        `a` and `b`. It will compute up to pentadiagonal.
    spillover_correction : bool, optional
        Whether to apply spillover corrections to the output matrix.
        This is necessary when the matrices represent open-ended
        systems. The default is False.

    """
    if a.distribution_state == "nnz" or b.distribution_state == "nnz":
        raise ValueError(
            "Matrix multiplication is not supported for matrices in nnz distribution state."
        )
    num_blocks = len(a.block_sizes)

    # Make sure the output matrix is initialized to zero.
    out.data = 0

    for i in range(num_blocks):
        for j in range(max(0, i - 2), min(num_blocks, i + 3)):
            out_ij = out.blocks[i, j]
            for k in range(max(0, i - 1), min(num_blocks, i + 2)):
                out_ij += a.blocks[i, k] @ b.blocks[k, j]

            out.blocks[i, j] = out_ij

    if not spillover_correction:
        return

    # Corrections accounting for the fact that the matrices should have
    # open ends.
    out.blocks[0, 0] += a.blocks[1, 0] @ b.blocks[0, 1]
    out.blocks[-1, -1] += a.blocks[-2, -1] @ b.blocks[-1, -2]


def btd_sandwich(
    a: DSBSparse,
    b: DSBSparse,
    out: DSBSparse,
    spillover_correction: bool = False,
):
    """Compute the sandwich product `a @ b @ a` BTD DSBSparse matrices.

    Parameters
    ----------
    a : DSBSparse
        The first block tridiagonal matrix.
    b : DSBSparse
        The second block tridiagonal matrix.
    out : DSBSparse
        The output matrix. This matrix must have the same block size as
        `a`, and `b`. It will compute up to heptadiagonal.
    spillover_correction : bool, optional
        Whether to apply spillover corrections to the output matrix.
        This is necessary when the matrices represent open-ended
        systems. The default is False.

    """
    if a.distribution_state == "nnz" or b.distribution_state == "nnz":
        raise ValueError(
            "Matrix multiplication is not supported for matrices in nnz distribution state."
        )
    num_blocks = len(a.block_sizes)

    # Make sure the output matrix is initialized to zero.
    out.data = 0

    for i in range(num_blocks):
        for j in range(max(0, i - 3), min(num_blocks, i + 4)):
            out_ij = out.blocks[i, j]
            for k in range(max(0, i - 2), min(num_blocks, i + 3)):
                a_kj = a.blocks[k, j]
                for m in range(max(0, i - 1), min(num_blocks, i + 2)):
                    out_ij += a.blocks[i, m] @ b.blocks[m, k] @ a_kj

            out.blocks[i, j] = out_ij

    if not spillover_correction:
        return

    # Corrections accounting for the fact that the matrices should have
    # open ends.
    out.blocks[0, 0] += (
        a.blocks[1, 0] @ b.blocks[0, 1] @ a.blocks[0, 0]
        + a.blocks[0, 0] @ b.blocks[1, 0] @ a.blocks[0, 1]
        + a.blocks[1, 0] @ b.blocks[0, 0] @ a.blocks[0, 1]
    )
    out.blocks[0, 1] += a.blocks[1, 0] @ b.blocks[0, 1] @ a.blocks[0, 1]
    out.blocks[1, 0] += a.blocks[1, 0] @ b.blocks[1, 0] @ a.blocks[0, 1]

    out.blocks[-1, -1] += (
        a.blocks[-2, -1] @ b.blocks[-1, -2] @ a.blocks[-1, -1]
        + a.blocks[-1, -1] @ b.blocks[-2, -1] @ a.blocks[-1, -2]
        + a.blocks[-2, -1] @ b.blocks[-1, -1] @ a.blocks[-1, -2]
    )
    out.blocks[-1, -2] += a.blocks[-2, -1] @ b.blocks[-1, -2] @ a.blocks[-1, -2]
    out.blocks[-2, -1] += a.blocks[-2, -1] @ b.blocks[-2, -1] @ a.blocks[-1, -2]
