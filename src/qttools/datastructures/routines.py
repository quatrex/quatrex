# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from mpi4py.MPI import COMM_WORLD as comm, Intracomm, Request

from qttools import xp
# from qttools.datastructures import DSBSparse, DBSparse
from qttools.datastructures.dbsparse import DBSparse
from qttools.datastructures.dsbsparse import DSBSparse
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
    else:
        out_block = True
        out = {}

    for i in range(num_blocks):
        for j in range(
            max(i - out_num_diag // 2, 0), min(i + out_num_diag // 2 + 1, num_blocks)
        ):
            if out_block:
                partsum = xp.zeros(
                    (a.block_sizes[i], a.block_sizes[j]), dtype=accumulator_dtype
                )
            else:
                partsum = (out.blocks[i, j]).astype(accumulator_dtype)

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
                        partsum += a.blocks[i_a, k_a] @ b.blocks[k_b, j_b]
                    else:
                        partsum += a.blocks[i, k] @ b.blocks[k, j]

            if out_block:
                out[i, j] = partsum
            else:
                out.blocks[i, j] = partsum

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
    else:
        out_block = True
        out = {}

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

            a_im = a.blocks[a_i, a_m]

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
                    ab_ik[k] = (a_im @ b.blocks[b_m, b_k]).astype(
                        accumulator_dtype
                    )  # cast data type
                else:
                    ab_ik[k] += (a_im @ b.blocks[b_m, b_k]).astype(
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
                partsum = (out.blocks[i, j]).astype(accumulator_dtype)  # cast data type

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
                partsum += (ab_ik[k] @ a.blocks[a_k, a_j]).astype(
                    accumulator_dtype
                )  # cast data type

            if out_block:
                out[i, j] = partsum
            else:
                out.blocks[i, j] = partsum

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


class BlockMatrix(dict):

    def __init__(self, dbsparse: DBSparse, local_keys: set[tuple[int, int]],
                 origin: tuple[int, int], mapping=None):
        self.dbsparse = dbsparse
        self.local_keys = local_keys
        self.origin = origin
        mapping = mapping or {}
        super(BlockMatrix, self).__init__(mapping)

    def __getitem__(self, key):
        if super(BlockMatrix, self).__contains__(key):
            return super(BlockMatrix, self).__getitem__(key)
        if key in self.local_keys:
            key = (key[0] - self.origin[0], key[1] - self.origin[1])
            return self.dbsparse.local_blocks[key]
        print(f"Something bad happened: {comm.rank=}, {key=}, {self.origin=}")
        # return None
        raise KeyError(key)
        # return xp.zeros((int(self.dbsparse.block_sizes[key[0]]),
        #                  int(self.dbsparse.block_sizes[key[1]])),
        #                 dtype=self.dbsparse.local_data.dtype)

    def __setitem__(self, key, val):
        if key in self.local_keys:
            key = (key[0] - self.origin[0], key[1] - self.origin[1])
            self.dbsparse.local_blocks[key] = val
        else:
            return super(BlockMatrix, self).__setitem__(key, val)

    def toarray(self):
        size = int(sum(self.dbsparse.block_sizes))
        out = xp.zeros((size, size), dtype=self.dbsparse.local_data.dtype)
        for i, (isz, ioff) in enumerate(zip(self.dbsparse.block_sizes, self.dbsparse.block_offsets)):
            for j, (jsz, joff) in enumerate(zip(self.dbsparse.block_sizes, self.dbsparse.block_offsets)):
                try:
                    out[ioff:ioff + isz, joff:joff + jsz] = self[i, j]
                except KeyError:
                    pass
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
    num_blocks = a.dbsparse.num_blocks
    a_off = a_num_diag // 2
    b_off = b_num_diag // 2
    c_off = a_off + b_off
    rank = comm.rank

    reqs = []
    # Send halo blocks to previous rank
    if start_block > 0:
        for i in range(start_block, min(num_blocks, start_block + c_off)):
            for j in range(max(start_block, i - a_off), min(a.dbsparse.num_blocks, i + a_off + 1)):
                reqs.append(comm.isend(a[i, j], dest=rank - 1, tag=0))
        for j in range(start_block, min(num_blocks, start_block + c_off)):
            for i in range(max(start_block, j - b_off), min(b.dbsparse.num_blocks, j + b_off + 1)):
                reqs.append(comm.isend(b[i, j], dest=rank - 1, tag=1))
    # Send halo blocks to next rank
    if end_block < a.dbsparse.num_blocks:
        for i in range(end_block, min(num_blocks, end_block + a_off)):
            for j in range(max(0, i - a_off), min(end_block, i + a_off + 1)):
                reqs.append(comm.isend(a[i, j], dest=rank + 1, tag=0))
    if end_block < b.dbsparse.num_blocks:
        for j in range(end_block, min(num_blocks, end_block + b_off)):
            for i in range(max(0, j - b_off), min(end_block, j + b_off + 1)):
                reqs.append(comm.isend(b[i, j], dest=rank + 1, tag=1))
    # Receive halo blocks from next rank
    if end_block < a.dbsparse.num_blocks:
        for i in range(end_block, min(num_blocks, end_block + c_off)):
            for j in range(max(end_block, i - a_off), min(a.dbsparse.num_blocks, i + a_off + 1)):
                a[i, j] = comm.recv(source=rank + 1, tag=0)
    if end_block < b.dbsparse.num_blocks:
        for j in range(end_block, min(num_blocks, end_block + c_off)):
            for i in range(max(end_block, j - b_off), min(b.dbsparse.num_blocks, j + b_off + 1)):
                b[i, j] = comm.recv(source=rank + 1, tag=1)
    # Receive halo blocks from previous rank
    if start_block > 0:
        for i in range(start_block, min(num_blocks, start_block + a_off)):
            for j in range(max(0, i - a_off), min(start_block, i + a_off + 1)):
                a[i, j] = comm.recv(source=rank - 1, tag=0)
        for j in range(start_block, min(num_blocks, start_block + b_off)):
            for i in range(max(0, j - b_off), min(start_block, i + b_off + 1)):
                b[i, j] = comm.recv(source=rank - 1, tag=1)
    Request.Waitall(reqs)


def bd_matmul_distr(
    a: DBSparse | BlockMatrix,
    b: DBSparse | BlockMatrix,
    out: DBSparse | None,
    a_num_diag: int = 3,
    b_num_diag: int = 3,
    out_num_diag: int = 5,
    start_block: int = 0,
    end_block: int = None,
    comm: Intracomm = comm,
    spillover_correction: bool = False,
    accumulator_dtype=None,

):
    """Matrix multiplication of two `a @ b` BD DBSparse matrices.

    Parameters
    ----------
    a : DBSparse
        The first block diagonal matrix.
    b : DBSparse
        The second block diagonal matrix.
    out : DBSparse
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
    # if a.distribution_state == "nnz" or b.distribution_state == "nnz":
    #     raise ValueError(
    #         "Matrix multiplication is not supported for matrices in nnz distribution state."
    #     )

    if isinstance(a, BlockMatrix):
        a_ = a
        num_blocks = len(a.dbsparse.block_sizes)
        end_block = end_block or num_blocks
        accumulator_dtype = accumulator_dtype or a.dbsparse.dtype
    else:
        num_blocks = len(a.block_sizes)
        end_block = end_block or num_blocks
        accumulator_dtype = accumulator_dtype or a.dtype
        local_keys = set()
        for i in range(start_block, end_block):
            for j in range(start_block, min(num_blocks, i + a_num_diag // 2 + 1)):
                local_keys.add((i, j))
        for j in range(start_block, end_block):
            for i in range(end_block, min(num_blocks, j + a_num_diag // 2 + 1)):
                local_keys.add((i, j))
        a_ = BlockMatrix(a, local_keys, (start_block, start_block))

    if isinstance(b, BlockMatrix):
        b_ = b
    else:
        local_keys = set()
        for i in range(start_block, end_block):
            for j in range(start_block, min(num_blocks, i + b_num_diag // 2 + 1)):
                local_keys.add((i, j))
        for j in range(start_block, end_block):
            for i in range(end_block, min(num_blocks, j + b_num_diag // 2 + 1)):
                local_keys.add((i, j))
        b_ = BlockMatrix(b, local_keys, (start_block, start_block))

    arrow_partition_halo_comm(a_, b_, a_num_diag, b_num_diag, start_block, end_block, comm)

    # Make sure the output matrix is initialized to zero.
    if out is not None:
        out.local_data[:] = 0
        local_keys = set()
        for i in range(start_block, end_block):
            for j in range(start_block, min(num_blocks, i + out_num_diag // 2 + 1)):
                local_keys.add((i, j))
        for j in range(start_block, end_block):
            for i in range(end_block, min(num_blocks, j + out_num_diag // 2 + 1)):
                local_keys.add((i, j))
        out_ = BlockMatrix(out, local_keys, (start_block, start_block))
    else:
        out_ = BlockMatrix(a_.dbsparse, set(), (start_block, start_block))
    
    for sector in ((start_block, end_block, start_block, num_blocks),
                   (end_block, num_blocks, start_block, end_block)):
        
        brow_start, brow_end, bcol_start, bcol_end = sector

        for i in range(brow_start, brow_end):
            for j in range(
                max(i - out_num_diag // 2, bcol_start), min(i + out_num_diag // 2 + 1, bcol_end)
            ):
                partsum = None

                for k in range(i - a_num_diag // 2, i + a_num_diag // 2 + 1):
                    if abs(j - k) > b_num_diag // 2:
                        continue
                    out_range = (k < 0) or (k >= num_blocks)
                    if out_range and (not spillover_correction):
                        continue
                    else:
                        if out_range:
                            i_a, k_a = correct_out_range_index(i, k, num_blocks)
                            k_b, j_b = correct_out_range_index(k, j, num_blocks)
                        else:
                            i_a, k_a = i, k
                            k_b, j_b = k, j
                        try:
                            if partsum is None:
                                partsum = (a_[i_a, k_a] @ b_[k_b, j_b]).astype(accumulator_dtype)
                            else:
                                partsum += a_[i_a, k_a] @ b_[k_b, j_b]
                        except:
                            print(f"Something bad happened: {comm.rank=}, {i=}, {j=}, {k=}, {i_a=}, {k_a=}, {k_b=}, {j_b=}")

                out_[i, j] = partsum

    return out_


def bd_sandwich_distr(
    a: DBSparse,
    b: DBSparse,
    out: DBSparse | None,
    in_num_diag: int = 3,
    out_num_diag: int = 7,
    start_block: int = 0,
    end_block: int = None,
    comm: Intracomm = comm,
    spillover_correction: bool = False,
    accumulator_dtype=None,

):
    """Matrix multiplication of two `a @ b` BD DBSparse matrices.

    Parameters
    ----------
    a : DBSparse
        The first block diagonal matrix.
    b : DBSparse
        The second block diagonal matrix.
    out : DBSparse
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

    num_blocks = len(a.block_sizes)
    end_block = end_block or num_blocks
    accumulator_dtype = accumulator_dtype or a.dtype
    local_keys = set()
    for i in range(start_block, end_block):
        for j in range(max(start_block, i - in_num_diag // 2), min(num_blocks, i + in_num_diag // 2 + 1)):
            local_keys.add((i, j))
    for j in range(start_block, end_block):
        for i in range(max(end_block, j - in_num_diag // 2), min(num_blocks, j + in_num_diag // 2 + 1)):
            local_keys.add((i, j))
    a_ = BlockMatrix(a, local_keys, (start_block, start_block))
    b_ = BlockMatrix(b, local_keys, (start_block, start_block))

    tmp_num_diag = 2 * in_num_diag - 1
    tmp = bd_matmul_distr(a_, b_, None, in_num_diag, in_num_diag, tmp_num_diag, start_block, end_block, comm, False, accumulator_dtype)
    out_ = bd_matmul_distr(tmp, a_, out, tmp_num_diag, in_num_diag, out_num_diag, start_block, end_block, comm, False, accumulator_dtype)

    if spillover_correction:

        # NOTE: This only works for BTD matrices with open ends.

        # Corrections accounting for the fact that the matrices should have
        # open ends.
        if start_block == 0:
            out_[0, 0] += (
                a_[1, 0] @ b_[0, 1] @ a_[0, 0]
                + a_[0, 0] @ b_[1, 0] @ a_[0, 1]
                + a_[1, 0] @ b_[0, 0] @ a_[0, 1]
            )
            out_[0, 1] += a_[1, 0] @ b_[0, 1] @ a_[0, 1]
            out_[1, 0] += a_[1, 0] @ b_[1, 0] @ a_[0, 1]
        
        if end_block == a.num_blocks:
            m1 = a.num_blocks - 1
            m2 = a.num_blocks - 2
            out_[m1, m1] += (
                a_[m2, m1] @ b_[m1, m2] @ a_[m1, m1]
                + a_[m1, m1] @ b_[m2, m1] @ a_[m1, m2]
                + a_[m2, m1] @ b_[m1, m1] @ a_[m1, m2]
            )
            out_[m1, m2] += a_[m2, m1] @ b_[m1, m2] @ a_[m1, m2]
            out_[m2, m1] += a_[m2, m1] @ b_[m2, m1] @ a_[m1, m2]
    
    return out_
    
