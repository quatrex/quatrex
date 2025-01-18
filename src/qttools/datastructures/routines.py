from qttools.datastructures import DSBSparse


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
            for k in range(max(0, i - 1), min(num_blocks, i + 2)):
                out.blocks[i, j] += a.blocks[i, k] @ b.blocks[k, j]

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
            for k in range(max(0, i - 2), min(num_blocks, i + 3)):
                for m in range(max(0, i - 1), min(num_blocks, i + 2)):
                    out.blocks[i, j] += a.blocks[i, m] @ b.blocks[m, k] @ a.blocks[k, j]

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
