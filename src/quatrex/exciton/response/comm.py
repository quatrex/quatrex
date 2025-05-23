from qttools import NDArray, xp
from mpi4py.MPI import COMM_WORLD as comm
from mpi4py import MPI
from mpi4py.MPI import Request
from qttools.kernels.datastructure.cupy.dsdbsparse import find_ranks
from qttools.utils.gpu_utils import get_host, synchronize_current_stream



def distributed_transpose_2darray(data:NDArray,local_c:int,local_r:int):
    assert data.ndim == 2
    assert data.shape[0] == local_r
    assert data.shape[1] == local_c*comm.size
    for i in range(local_r):
        comm.Alltoall(MPI.IN_PLACE,data[i,:])
        buffer = xp.transpose(xp.reshape(data[i,:],(comm.size,local_c)))
        data[i,:] = buffer.flatten()
    data = xp.reshape(xp.transpose(data) , (local_c,local_r*comm.size))   
    return data

def fetch_overlaping_data(
    nnz_to_fetch,
    nnz_rank,
    local_data: NDArray,
    nnz_section_offsets: NDArray,
    tag: int = 0,
):
    """
    Gather the overlaping data from other ranks

    Args:
        nnz_to_fetch (list): list of ids (global) to fetch from other ranks
        nnz_rank (list): list of ranks to fetch from
        local_data (NDArray): local data array
        nnz_section_offsets (NDArray): offsets of the sections in the global data array

    Returns:
        NDArray overlaping data
    """
    if comm.size == 1:
        return xp.array([])

    recbuf = [NDArray] * comm.size
    sendbuf = [NDArray] * comm.size
    send_reqs = []
    recv_reqs = []
    synchronize_current_stream()
    for j in reversed(range(comm.size)):
        if j == comm.rank:
            continue
        inds_rank_to_j = nnz_to_fetch[j][nnz_rank[j] == comm.rank]
        if not inds_rank_to_j.any():
            continue

        sendbuf[j] = get_host(
            local_data[..., inds_rank_to_j - nnz_section_offsets[comm.rank]]
        )
        send_reqs.append(comm.Isend(sendbuf[j], dest=j, tag=tag))

    for i in range(comm.size):
        if i == comm.rank:
            continue
        recv_reqs.append(comm.Irecv(recbuf[i], source=i, tag=tag))

    Request.Waitall(recv_reqs)

    recv_data = xp.concatenate(
        [xp.array(buf) for buf in recbuf if buf is not None], axis=-1
    )
    return recv_data


def find_overlaping_data(
    nnz_section_offsets, num_diag: int, rows: NDArray, cols: NDArray
):
    """
    Figure out the overlaping data on other ranks to fetch from based on an interaction range defined by "num_diag"
    """
    num_rank = len(nnz_section_offsets) - 1
    nnz_to_fetch = []
    nnz_rank = []

    for rank in range(num_rank):

        min_row = (
            rows[nnz_section_offsets[rank] : nnz_section_offsets[rank + 1]].min()
            - num_diag
        )
        max_row = (
            rows[nnz_section_offsets[rank] : nnz_section_offsets[rank + 1]].max()
            + num_diag
        )
        min_col = (
            cols[nnz_section_offsets[rank] : nnz_section_offsets[rank + 1]].min()
            - num_diag
        )
        max_col = (
            cols[nnz_section_offsets[rank] : nnz_section_offsets[rank + 1]].max()
            + num_diag
        )

        mask = (
            (rows >= min_row)
            & (rows <= max_row)
            & (cols >= min_col)
            & (cols <= max_col)
        )

        ids = xp.where(mask)[0]
        ids_in_rank = find_ranks(nnz_section_offsets, ids)

        get_nnz = xp.where(ids_in_rank != rank)[0]

        nnz_to_fetch.append(ids[get_nnz])
        nnz_rank.append(ids_in_rank[get_nnz])

    return nnz_to_fetch, nnz_rank
