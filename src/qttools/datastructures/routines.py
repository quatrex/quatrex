# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from qttools import xp
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import DSDBSparse


def correct_out_range_index(i: int, k: int, num_blocks: int):
    # find the index of block in the matrix being repeated into open-end
    # based on the difference of row and col, ie diagonal
    diag = k - i
    k_1 = min(max(k, 0), num_blocks - 1)
    i_1 = k_1 - diag  # keep the same diag
    i_2 = min(max(i_1, 0), num_blocks - 1)
    k_2 = i_2 + diag  # keep the same diag
    return (i_2, k_2)


def bd_sandwich(
    a: DSDBSparse,
    b: DSDBSparse,
    out: DSDBSparse,
    in_num_diag: int = 3,
    out_num_diag: int = 7,
):
    """Compute the sandwich product `a @ b @ a` BD DSDBSparse matrices.

    Parameters
    ----------
    a : DSDBSparse
        The first block tridiagonal matrix.
    b : DSDBSparse
        The second block tridiagonal matrix.
    out : DSDBSparse
        The output matrix. This matrix must have the same block size as
        `a`, and `b`. It will compute up to `out_num_diag` diagonals.
    in_num_diag: int
        The number of diagonals in input matrices
    out_num_diag: int
        The number of diagonals in output matrices

    """
    if (
        a.distribution_state == "nnz"
        or b.distribution_state == "nnz"
        or out.distribution_state == "nnz"
    ):
        raise ValueError(
            "Matrix multiplication is not supported for matrices in nnz distribution state."
        )
    num_blocks = len(a.block_sizes)

    a_is_hermitian = a.symmetry and a.symmetry_op(1 + 1j) != (1 - 1j)

    a_ = a.stack[...]
    b_ = b.stack[...]
    out_ = out.stack[...]

    for i in range(num_blocks):

        ab_ik = [None] * num_blocks * 2

        for m in range(i - in_num_diag // 2, i + in_num_diag // 2 + 1):

            out_range = (m < 0) or (m >= num_blocks)
            if out_range:
                continue
            else:
                a_i, a_m = i, m

            a_im = a_.blocks[a_i, a_m]

            for k in range(m - in_num_diag // 2, m + in_num_diag // 2 + 1):
                out_range = (k < 0) or (k >= num_blocks) or (m < 0) or (m >= num_blocks)
                if out_range:
                    continue
                else:
                    b_m, b_k = m, k
                if ab_ik[k] is None:
                    ab_ik[k] = a_im @ b_.blocks[b_m, b_k]
                else:
                    ab_ik[k] += a_im @ b_.blocks[b_m, b_k]

        if out.symmetry:
            range_j_min = i
        else:
            range_j_min = max(i - out_num_diag // 2, 0)

        for j in range(range_j_min, min(i + out_num_diag // 2 + 1, num_blocks)):

            partsum = 0

            for k in range(j - in_num_diag // 2, j + in_num_diag // 2 + 1):
                out_range = (k < 0) or (k >= num_blocks)

                if out_range:
                    continue

                if ab_ik[k] is None:
                    continue
                if a_is_hermitian:
                    partsum += ab_ik[k] @ a_.blocks[k, j]
                else:
                    partsum += ab_ik[k] @ a_.blocks[j, k].swapaxes(-1, -2).conj()

            out_.blocks[i, j] = partsum


def btd_sandwich(
    a: DSDBSparse,
    b: DSDBSparse,
    out: DSDBSparse,
    spillover_correction: bool = False,
):
    """Compute the sandwich product `a @ b @ a` BTD DSDBSparse matrices.

    Parameters
    ----------
    a : DSDBSparse
        The first block tridiagonal matrix.
    b : DSDBSparse
        The second block tridiagonal matrix.
    out : DSDBSparse
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


class BlockMatrix:
    """Block-sparse matrix class, including halo blocks for communication.

    Any local block keys are stored in the blocks of the dsdbsparse,
    while non-local block keys are stored in a separate dictionary.

    Parameters
    ----------
    dsdbsparse : DSDBSparse
        The underlying DSDBSparse matrix.
    local_keys : set[tuple[int, int]]
        The set of block keys that are local to the current rank.
    origin : tuple[int, int]
        The global index of the first local block. This is used to
        compute the local block keys from the global block keys.
    mapping : dict[tuple[int, int], xp.ndarray], optional
        A mapping from non-local block keys to their corresponding data
        arrays. This is used to store halo blocks that are communicated
        between ranks. The default is None, which means that there are
        no non-local blocks.

    """

    def __init__(
        self,
        dsdbsparse: DSDBSparse,
        local_keys: set[tuple[int, int]],
        origin: tuple[int, int],
        mapping: dict | None = None,
    ):
        """Initializes the BlockMatrix."""
        self.dsdbsparse = dsdbsparse
        self.local_keys = local_keys
        self.origin = origin
        self.blocks = self.dsdbsparse.blocks

        # Cache for non-local blocks.
        self._cache = dict(mapping or {})

    def __getitem__(self, key):
        """Gets the block corresponding to the given key."""
        if key in self._cache:
            return self._cache[key]
        if key in self.local_keys:
            key = (key[0] - self.origin[0], key[1] - self.origin[1])
            return self.blocks[key]

        rank = comm.block.rank if comm.block is not None else 0
        raise KeyError(f"Something bad happened: {rank=}, {key=}, {self.origin=}")

    def __setitem__(self, key, val):
        """Sets the block corresponding to the given key."""
        if key in self.local_keys:
            key = (key[0] - self.origin[0], key[1] - self.origin[1])
            self.blocks[key] = val
        else:
            self._cache[key] = val

    def toarray(self):
        """Converts the BlockMatrix to a dense array.

        Debugging method to check the correctness of the block matrix.

        """
        size = int(sum(self.dsdbsparse.block_sizes))
        out = xp.zeros((size, size), dtype=self.dsdbsparse.data.dtype)
        for i, (isz, ioff) in enumerate(
            zip(self.dsdbsparse.block_sizes, self.dsdbsparse.block_offsets)
        ):
            for j, (jsz, joff) in enumerate(
                zip(self.dsdbsparse.block_sizes, self.dsdbsparse.block_offsets)
            ):
                try:
                    out[ioff : ioff + isz, joff : joff + jsz] = self[i, j]
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
):
    """Communicate halo blocks between neighboring ranks assuming arrow
    partitioning.

    NOTE: The method works ONLY IF the ranks need to communicate ONLY with their
    immediate neighbors, i.e., rank - 1 and rank + 1.

    Parameters
    ----------
    a : BlockMatrix
        The first block matrix.
    b : BlockMatrix
        The second block matrix.
    a_num_diag : int
        The number of diagonals in the first block matrix.
    b_num_diag : int
        The number of diagonals in the second block matrix.
    start_block : int
        The index of the first block to communicate.
    end_block : int
        The index of the last block to communicate.

    """

    num_blocks = a.dsdbsparse.num_blocks
    a_ssz = a.dsdbsparse.shape[:-2]
    b_ssz = b.dsdbsparse.shape[:-2]
    bsz = a.dsdbsparse.block_sizes
    dtype = a.dsdbsparse.dtype
    a_off = a_num_diag // 2
    b_off = b_num_diag // 2
    c_off = a_off + b_off
    rank = comm.block.rank

    # Send halo blocks to previous rank
    def _send_to_previous():
        if start_block > 0:
            for i in range(start_block, min(num_blocks, start_block + c_off)):
                for j in range(
                    max(start_block, i - a_off),
                    min(a.dsdbsparse.num_blocks, i + a_off + 1),
                ):
                    comm.block.send(a[i, j], rank - 1)
            for j in range(start_block, min(num_blocks, start_block + c_off)):
                for i in range(
                    max(start_block, j - b_off),
                    min(b.dsdbsparse.num_blocks, j + b_off + 1),
                ):
                    comm.block.send(b[i, j], rank - 1)

    # Receive halo blocks from next rank
    def _recv_from_next():
        if end_block < a.dsdbsparse.num_blocks:
            for i in range(end_block, min(num_blocks, end_block + c_off)):
                for j in range(
                    max(end_block, i - a_off),
                    min(a.dsdbsparse.num_blocks, i + a_off + 1),
                ):
                    a[i, j] = xp.empty((a_ssz) + (bsz[i], bsz[j]), dtype=dtype)
                    comm.block.recv(a[i, j], rank + 1)
        if end_block < b.dsdbsparse.num_blocks:
            for j in range(end_block, min(num_blocks, end_block + c_off)):
                for i in range(
                    max(end_block, j - b_off),
                    min(b.dsdbsparse.num_blocks, j + b_off + 1),
                ):
                    b[i, j] = xp.empty((b_ssz) + (bsz[i], bsz[j]), dtype=dtype)
                    comm.block.recv(b[i, j], rank + 1)

    # Send halo blocks to next rank
    def _send_to_next():
        if end_block < a.dsdbsparse.num_blocks:
            for i in range(end_block, min(num_blocks, end_block + a_off)):
                for j in range(max(0, i - a_off), min(end_block, i + a_off + 1)):
                    comm.block.send(a[i, j], rank + 1)
        if end_block < b.dsdbsparse.num_blocks:
            for j in range(end_block, min(num_blocks, end_block + b_off)):
                for i in range(max(0, j - b_off), min(end_block, j + b_off + 1)):
                    comm.block.send(b[i, j], rank + 1)

    # Receive halo blocks from previous rank
    def _recv_from_previous():
        if start_block > 0:
            for i in range(start_block, min(num_blocks, start_block + a_off)):
                for j in range(max(0, i - a_off), min(start_block, i + a_off + 1)):
                    a[i, j] = xp.empty((a_ssz) + (bsz[i], bsz[j]), dtype=dtype)
                    comm.block.recv(a[i, j], rank - 1)
            for j in range(start_block, min(num_blocks, start_block + b_off)):
                for i in range(max(0, j - b_off), min(start_block, i + b_off + 1)):
                    b[i, j] = xp.empty((b_ssz) + (bsz[i], bsz[j]), dtype=dtype)
                    comm.block.recv(b[i, j], rank - 1)

    if rank % 2 == 0:
        _send_to_previous()
        _recv_from_next()
        _send_to_next()
        _recv_from_previous()
    else:
        _recv_from_next()
        _send_to_previous()
        _recv_from_previous()
        _send_to_next()


def bd_matmul_distr(
    a: DSDBSparse | BlockMatrix,
    b: DSDBSparse | BlockMatrix,
    out: DSDBSparse | None,
    a_num_diag: int = 3,
    b_num_diag: int = 3,
    out_num_diag: int = 5,
    start_block: int = 0,
    end_block: int | None = None,
) -> BlockMatrix:
    """Matrix multiplication of two `a @ b` BD DSDBSparse matrices.

    Parameters
    ----------
    a : DSDBSparse
        The first block diagonal matrix.
    b : DSDBSparse
        The second block diagonal matrix.
    out : DSDBSparse
        The output matrix. This matrix must have the same block size as `a` and
        `b`. It will compute up to `out_num_diag` diagonals.
    in_num_diag: int
        The number of diagonals in input matrices
    out_num_diag: int
        The number of diagonals in output matrices
    start_block: int
        The index of the first block to compute.
    end_block: int
        The index of the last block to compute. If None, it will compute up to
        the last block.

    Returns
    -------
    BlockMatrix
        The resulting block matrix of the multiplication. Even if the output is
        not None, the method returns the corresponding BlockMatrix for
        convenience.

    """
    if isinstance(a, DSDBSparse) and a.distribution_state == "nnz":
        raise ValueError(
            "Matrix multiplication is not supported for matrices in nnz distribution state."
        )

    if isinstance(b, DSDBSparse) and b.distribution_state == "nnz":
        raise ValueError(
            "Matrix multiplication is not supported for matrices in nnz distribution state."
        )
    if isinstance(out, DSDBSparse) and out.distribution_state == "nnz":
        raise ValueError(
            "Matrix multiplication is not supported for matrices in nnz distribution state."
        )

    if isinstance(a, BlockMatrix):
        a_ = a
        num_blocks = len(a.dsdbsparse.block_sizes)
        end_block = end_block or num_blocks
    else:
        num_blocks = len(a.block_sizes)
        end_block = end_block or num_blocks
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

    # call blocking backend
    arrow_partition_halo_comm(
        a_,
        b_,
        a_num_diag,
        b_num_diag,
        start_block,
        end_block,
    )

    # Make sure the output matrix is initialized to zero.
    if out is not None:
        out.data[:] = 0
        local_keys = set()
        for i in range(start_block, end_block):
            for j in range(start_block, min(num_blocks, i + out_num_diag // 2 + 1)):
                local_keys.add((i, j))
        for j in range(start_block, end_block):
            for i in range(end_block, min(num_blocks, j + out_num_diag // 2 + 1)):
                local_keys.add((i, j))
        out_ = BlockMatrix(out, local_keys, (start_block, start_block))
    else:
        out_ = BlockMatrix(b_.dsdbsparse, set(), (start_block, start_block))

    for sector in (
        (start_block, end_block, start_block, num_blocks),
        (end_block, num_blocks, start_block, end_block),
    ):

        brow_start, brow_end, bcol_start, bcol_end = sector

        for i in range(brow_start, brow_end):
            for j in range(
                max(i - out_num_diag // 2, bcol_start),
                min(i + out_num_diag // 2 + 1, bcol_end),
            ):
                partsum = None

                for k in range(i - a_num_diag // 2, i + a_num_diag // 2 + 1):
                    if abs(j - k) > b_num_diag // 2:
                        continue
                    out_range = (k < 0) or (k >= num_blocks)
                    if out_range:
                        continue
                    else:
                        if out_range:
                            i_a, k_a = correct_out_range_index(i, k, num_blocks)
                            k_b, j_b = correct_out_range_index(k, j, num_blocks)
                        else:
                            i_a, k_a = i, k
                            k_b, j_b = k, j
                        if partsum is None:
                            partsum = a_[i_a, k_a] @ b_[k_b, j_b]
                        else:
                            partsum += a_[i_a, k_a] @ b_[k_b, j_b]

                out_[i, j] = partsum

    return out_


def bd_sandwich_distr(
    a: DSDBSparse,
    b: DSDBSparse,
    out: DSDBSparse,
    in_num_diag: int = 3,
    out_num_diag: int = 7,
    start_block: int = 0,
    end_block: int = None,
) -> None:
    """Matrix multiplication of three `a @ b @ a` BD DSDBSparse matrices.

    Parameters
    ----------
    a : DSDBSparse
        The first block diagonal matrix.
    b : DSDBSparse
        The second block diagonal matrix.
    out : DSDBSparse
        The output matrix. This matrix must have the same block size as
        `a` and `b`. It will compute up to `out_num_diag` diagonals.
    in_num_diag: int
        The number of diagonals in input matrices
    out_num_diag: int
        The number of diagonals in output matrices

    """
    if (
        a.distribution_state == "nnz"
        or b.distribution_state == "nnz"
        or out.distribution_state == "nnz"
    ):
        raise ValueError(
            "Matrix multiplication is not supported for matrices in nnz distribution state."
        )

    # Dispatch to more optimized implementations in the non-block distributed case

    num_blocks = len(a.block_sizes)
    end_block = end_block or num_blocks
    local_keys = set()
    for i in range(start_block, end_block):
        for j in range(
            max(start_block, i - in_num_diag // 2),
            min(num_blocks, i + in_num_diag // 2 + 1),
        ):
            local_keys.add((i, j))
    for j in range(start_block, end_block):
        for i in range(
            max(end_block, j - in_num_diag // 2),
            min(num_blocks, j + in_num_diag // 2 + 1),
        ):
            local_keys.add((i, j))
    a_ = BlockMatrix(a, local_keys, (start_block, start_block))
    b_ = BlockMatrix(b, local_keys, (start_block, start_block))

    tmp_num_diag = 2 * in_num_diag - 1
    tmp = bd_matmul_distr(
        a_,
        b_,
        None,
        in_num_diag,
        in_num_diag,
        tmp_num_diag,
        start_block,
        end_block,
    )
    bd_matmul_distr(
        tmp,
        a_,
        out,
        tmp_num_diag,
        in_num_diag,
        out_num_diag,
        start_block,
        end_block,
    )
