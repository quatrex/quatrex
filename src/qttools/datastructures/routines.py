# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from mpi4py.MPI import Intracomm, Request

from qttools import xp
from qttools.datastructures import DSBSparse, DBSparse
from qttools.profiling import Profiler

profiler = Profiler()


@profiler.profile(level="api")
def correct_out_range_index(i: int, k: int, num_blocks: int):
    # find the index of block in the matrix being repeated into open-end
    # based on the difference of row and col, ie diagonal
    diag = k - i
    k_1 = min(max(k, 0), num_blocks - 1)
    i_1 = k_1 - diag  # keep the same diag
    i_2 = min(max(i_1, 0), num_blocks - 1)
    k_2 = i_2 + diag  # keep the same diag
    return (i_2, k_2)


@profiler.profile(level="api")
def bd_matmul(
    a: DSBSparse,
    b: DSBSparse,
    out: DSBSparse | None,
    in_num_diag: int = 3,
    out_num_diag: int = 5,
    spillover_correction: bool = False,
    accumulator_dtype=None,
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
        `a` and `b`. It will compute up to `out_num_diag` diagonals.
    in_num_diag: int
        The number of diagonals in input matrices
    out_num_diag: int
        The number of diagonals in output matrices
    spillover_correction : bool, optional
        Whether to apply spillover corrections to the output matrix.
        This is necessary when the matrices represent open-ended
        systems. The default is False.
    accumulator_dtype : data type, optional
        The data type of the temporary accumulator matrices. The default is complex128.

    TODO: replace @ by appropriate gemm

    """
    if a.distribution_state == "nnz" or b.distribution_state == "nnz":
        raise ValueError(
            "Matrix multiplication is not supported for matrices in nnz distribution state."
        )
    num_blocks = len(a.block_sizes)

    if accumulator_dtype is None:
        accumulator_dtype = a.dtype

    # Make sure the output matrix is initialized to zero.
    if out is not None:
        out.data = 0
        out_block = False
        # NOTE: Using the stack attribute to force caching of the data view.
        out_ = out.stack[...]
    else:
        out_block = True
        out = {}

    a_ = a.stack[...]
    b_ = b.stack[...]

    for i in range(num_blocks):
        for j in range(
            max(i - out_num_diag // 2, 0), min(i + out_num_diag // 2 + 1, num_blocks)
        ):
            if out_block:
                partsum = xp.zeros(
                    (a.block_sizes[i], a.block_sizes[j]), dtype=accumulator_dtype
                )
            else:
                partsum = (out_.blocks[i, j]).astype(accumulator_dtype)

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
                        partsum += a_.blocks[i_a, k_a] @ b_.blocks[k_b, j_b]
                    else:
                        partsum += a_.blocks[i, k] @ b_.blocks[k, j]

            if out_block:
                out[i, j] = partsum
            else:
                out_.blocks[i, j] = partsum

    if out_block:
        return out


@profiler.profile(level="api")
def bd_sandwich(
    a: DSBSparse,
    b: DSBSparse,
    out: DSBSparse | None,
    in_num_diag: int = 3,
    out_num_diag: int = 7,
    spillover_correction: bool = False,
    accumulator_dtype=None,
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
        `a`, and `b`. It will compute up to `out_num_diag` diagonals.
    in_num_diag: int
        The number of diagonals in input matrices
    out_num_diag: int
        The number of diagonals in output matrices
    spillover_correction : bool, optional
        Whether to apply spillover corrections to the output matrix.
        This is necessary when the matrices represent open-ended
        systems. The default is False.
    accumulator_dtype : data type, optional
        The data type of the temporary accumulator matrices. The default is complex128.

    TODO: replace @ by appropriate gemm

    """
    if a.distribution_state == "nnz" or b.distribution_state == "nnz":
        raise ValueError(
            "Matrix multiplication is not supported for matrices in nnz distribution state."
        )
    num_blocks = len(a.block_sizes)

    if accumulator_dtype is None:
        accumulator_dtype = a.dtype

    # Make sure the output matrix is initialized to zero.
    if out is not None:
        out.data = 0
        out_block = False
        # NOTE: Using the stack attribute to force caching of the data view.
        out_ = out.stack[...]
    else:
        out_block = True
        out = {}

    a_ = a.stack[...]
    b_ = b.stack[...]

    for i in range(num_blocks):

        ab_ik = [None] * num_blocks * 2

        for m in range(i - in_num_diag // 2, i + in_num_diag // 2 + 1):

            out_range = (m < 0) or (m >= num_blocks)
            if out_range and (not spillover_correction):
                continue
            else:
                if out_range:
                    a_i, a_m = correct_out_range_index(i, m, num_blocks)
                else:
                    a_i, a_m = i, m

            a_im = a_.blocks[a_i, a_m]

            for k in range(m - in_num_diag // 2, m + in_num_diag // 2 + 1):
                out_range = (k < 0) or (k >= num_blocks) or (m < 0) or (m >= num_blocks)
                if out_range and (not spillover_correction):
                    continue
                else:
                    if out_range:
                        b_m, b_k = correct_out_range_index(m, k, num_blocks)
                    else:
                        b_m, b_k = m, k
                if ab_ik[k] is None:
                    ab_ik[k] = (a_im @ b_.blocks[b_m, b_k]).astype(
                        accumulator_dtype
                    )  # cast data type
                else:
                    ab_ik[k] += (a_im @ b_.blocks[b_m, b_k]).astype(
                        accumulator_dtype
                    )  # cast data type

        for j in range(
            max(i - out_num_diag // 2, 0), min(i + out_num_diag // 2 + 1, num_blocks)
        ):

            if out_block:
                partsum = xp.zeros(
                    (a.block_sizes[i], a.block_sizes[j]), dtype=accumulator_dtype
                )
            else:
                partsum = (out_.blocks[i, j]).astype(
                    accumulator_dtype
                )  # cast data type

            for k in range(j - in_num_diag // 2, j + in_num_diag // 2 + 1):
                out_range = (k < 0) or (k >= num_blocks)
                if out_range and (not spillover_correction):
                    continue
                else:
                    if out_range:
                        a_k, a_j = correct_out_range_index(k, j, num_blocks)
                    else:
                        a_k, a_j = k, j
                if ab_ik[k] is None:
                    continue
                partsum += (ab_ik[k] @ a_.blocks[a_k, a_j]).astype(
                    accumulator_dtype
                )  # cast data type

            if out_block:
                out[i, j] = partsum
            else:
                out_.blocks[i, j] = partsum

    if out_block:
        return out


@profiler.profile(level="api")
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

    # NOTE: Using the stack attribute to force caching of the data view.
    out_ = out.stack[...]
    a_ = a.stack[...]
    b_ = b.stack[...]

    for i in range(num_blocks):
        for j in range(max(0, i - 2), min(num_blocks, i + 3)):
            out_ij = out.blocks[i, j]
            for k in range(max(0, i - 1), min(num_blocks, i + 2)):
                out_ij += a_.blocks[i, k] @ b_.blocks[k, j]

            out_.blocks[i, j] = out_ij

    if not spillover_correction:
        return

    # Corrections accounting for the fact that the matrices should have
    # open ends.
    out_.blocks[0, 0] += a_.blocks[1, 0] @ b_.blocks[0, 1]
    out_.blocks[-1, -1] += a_.blocks[-2, -1] @ b_.blocks[-1, -2]


@profiler.profile(level="api")
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

    # NOTE: Using the stack attribute to force caching of the data view.
    out_ = out.stack[...]
    a_ = a.stack[...]
    b_ = b.stack[...]

    for i in range(num_blocks):
        for j in range(max(0, i - 3), min(num_blocks, i + 4)):
            out_ij = out_.blocks[i, j]
            for k in range(max(0, i - 2), min(num_blocks, i + 3)):
                a_kj = a_.blocks[k, j]
                for m in range(max(0, i - 1), min(num_blocks, i + 2)):
                    out_ij += a_.blocks[i, m] @ b_.blocks[m, k] @ a_kj

            out_.blocks[i, j] = out_ij

    if not spillover_correction:
        return

    # Corrections accounting for the fact that the matrices should have
    # open ends.
    out_.blocks[0, 0] += (
        a_.blocks[1, 0] @ b_.blocks[0, 1] @ a_.blocks[0, 0]
        + a_.blocks[0, 0] @ b_.blocks[1, 0] @ a_.blocks[0, 1]
        + a_.blocks[1, 0] @ b_.blocks[0, 0] @ a_.blocks[0, 1]
    )
    out_.blocks[0, 1] += a_.blocks[1, 0] @ b_.blocks[0, 1] @ a_.blocks[0, 1]
    out_.blocks[1, 0] += a_.blocks[1, 0] @ b_.blocks[1, 0] @ a_.blocks[0, 1]

    out_.blocks[-1, -1] += (
        a_.blocks[-2, -1] @ b_.blocks[-1, -2] @ a_.blocks[-1, -1]
        + a_.blocks[-1, -1] @ b_.blocks[-2, -1] @ a_.blocks[-1, -2]
        + a_.blocks[-2, -1] @ b_.blocks[-1, -1] @ a_.blocks[-1, -2]
    )
    out_.blocks[-1, -2] += a_.blocks[-2, -1] @ b_.blocks[-1, -2] @ a_.blocks[-1, -2]
    out_.blocks[-2, -1] += a_.blocks[-2, -1] @ b_.blocks[-2, -1] @ a_.blocks[-1, -2]


class BlockMatrix(dict):

    def __init__(self, dbsparse: DBSparse, mapping=None):
        self.dbsparse = dbsparse
        mapping = mapping or {}
        super(BlockMatrix, self).__init__(mapping)

    def __getitem__(self, key):
        if super(BlockMatrix, self).__contains__(key):
            return super(BlockMatrix, self).__getitem__(key)
        return self.dbsparse.blocks[key]

    def __setitem__(self, key, val):
        # TODO: Check that we can set this block
        self.dbsparse.blocks[key] = val

    def toarray(self):
        size = sum(self.dbsparse.block_sizes)
        out = xp.zeros((size, size), dtype=self.dbsparse.local_data.dtype)
        for i, (isz, ioff) in enumerate(
            zip(self.dbsparse.block_sizes, self.dbsparse.block_offsets)
        ):
            for j, (jsz, joff) in enumerate(
                zip(self.dbsparse.block_sizes, self.dbsparse.block_offsets)
            ):
                out[ioff : ioff + isz, joff : joff + jsz] = self[i, j]
        return out


def arrow_partition_halo_comm(
    a: BlockMatrix,
    b: BlockMatrix,
    a_num_diag: int,
    b_num_diag: int,
    start_block: int,
    end_block: int,
    comm: Intracomm,
):
    a_off = a_num_diag // 2
    b_off = b_num_diag // 2
    c_off = a_off + b_off
    rank = comm.rank

    reqs = []
    # Send halo blocks to previous rank
    if start_block > 0:
        for i in range(start_block, start_block + c_off):
            for j in range(start_block, min(a.num_blocks, i + a_off + 1)):
                reqs.append(comm.isend(a[i, j], dest=rank - 1, tag=0))
        for j in range(start_block, start_block + c_off):
            for i in range(start_block, min(b.num_blocks, j + b_off + 1)):
                reqs.append(comm.isend(b[i, j], dest=rank - 1, tag=1))
    # Send halo blocks to next rank
    if end_block < a.num_blocks:
        for i in range(end_block, end_block + c_off):
            for j in range(max(0, i - a_off), end_block):
                reqs.append(comm.isend(a[i, j], dest=rank + 1, tag=0))
    if end_block < b.num_blocks:
        for j in range(end_block, end_block + c_off):
            for i in range(max(0, j - b_off), end_block):
                reqs.append(comm.isend(b[i, j], dest=rank + 1, tag=1))
    # Receive halo blocks from previous rank
    if start_block > 0:
        for i in range(start_block, start_block + a_off):
            for j in range(max(0, i - a_off), start_block):
                a[i, j] = comm.recv(source=rank - 1, tag=0)
        for j in range(start_block, start_block + b_off):
            for i in range(max(0, j - b_off), start_block):
                b[i, j] = comm.recv(source=rank - 1, tag=1)
    # Receive halo blocks from next rank
    if end_block < a.num_blocks:
        for i in range(end_block, end_block + c_off):
            for j in range(end_block, min(a.num_blocks, i + a_off + 1)):
                a[i, j] = comm.recv(source=rank + 1, tag=0)
    if end_block < b.num_blocks:
        for j in range(end_block, end_block + c_off):
            for i in range(end_block, min(b.num_blocks, j + b_off + 1)):
                b[i, j] = comm.recv(source=rank + 1, tag=1)
    Request.Waitall(reqs)
