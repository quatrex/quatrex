# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from pathlib import Path

import scipy.sparse as sps
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from mpi4py.util import pkl5

from qttools import NDArray, sparse, xp

comm = pkl5.Intracomm(comm)


def get_section_sizes(
    num_elements: int,
    num_sections: int = comm.size,
    strategy: str = "balanced",
) -> tuple[list, int]:
    """Computes the number of un-evenly divided elements per section.

    Parameters
    ----------
    num_elements : int
        The total number of elements to divide.
    num_sections : int, optional
        The number of sections to divide the elements into. Defaults to
        the number of MPI ranks.
    strategy : str, optional
        The strategy to use for dividing the elements. Can be one of
        "balanced" (default) or "greedy". In the "balanced" strategy,
        the elements are divided as evenly as possible across the
        sections. In the "greedy" strategy, the elements are divided
        such that the we get many sections with the maximum number of
        elements.

    Returns
    -------
    section_sizes : list
        The sizes of each section.
    effective_num_elements : int
        The effective number of elements after sectioning.

    Examples
    --------
    >>> get_section_sizes(10, 3, "fair")
    ([4, 3, 3], 12)
    >>> get_section_sizes(10, 3, "greedy")
    ([4, 4, 2], 12)

    """
    quotient, remainder = divmod(num_elements, num_sections)
    if strategy == "balanced":
        section_sizes = remainder * [quotient + 1] + (num_sections - remainder) * [
            quotient
        ]
    elif strategy == "greedy":
        section_sizes = [0] * num_sections
        for i in range(num_sections):
            section_sizes[i] = min(
                quotient + min(remainder, 1), num_elements - sum(section_sizes)
            )
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
    effective_num_elements = max(section_sizes) * num_sections
    return section_sizes, effective_num_elements


def distributed_load(path: Path) -> sparse.spmatrix | NDArray:
    """Loads an array from disk and broadcasts it to all ranks.

    Parameters
    ----------
    path : Path
        The path to the file to load.

    Returns
    -------
    sparse.spmatrix | NDArray
        The loaded array.

    Raises
    ------
    FileNotFoundError
        Occurs on every rank where the file does not exist.

    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix not in [".npz", ".npy"]:
        raise ValueError(f"Invalid file extension: {path.suffix}")

    if comm.rank == 0:
        if path.suffix == ".npz":
            # NOTE: cupyx.scipy.sparse.load_npz does not exist.
            arr = sps.load_npz(path)
            arr = sparse.coo_matrix(arr)
        elif path.suffix == ".npy":
            arr = xp.load(path)
    else:
        arr = None

    arr = comm.bcast(arr, root=0)

    return arr


def get_local_slice(global_array: NDArray, comm: MPI.Comm = comm) -> NDArray:
    """Returns the local slice of a distributed array.

    Parameters
    ----------
    global_array : NDArray
        The global array to slice.

    Returns
    -------
    NDArray
        The local slice of the global array.

    """
    section_sizes, __ = get_section_sizes(global_array.shape[-1], comm.size)
    section_offsets = xp.hstack(([0], xp.cumsum(xp.array(section_sizes))))

    return global_array[
        ..., int(section_offsets[comm.rank]) : int(section_offsets[comm.rank + 1])
    ]


@profiler.profile(level="debug")
def gather_array_nnz(array: NDArray, comm: MPI.Comm, sample_indices: NDArray) -> NDArray:
    """Gathers a distributed array split along nnz (ex. Sigma). I.e. nnz distribution.

    Parameters
    ----------
    array : NDArray
        The local array to gather. Must have shape (num_energies, local_nnz).
    comm : MPI.Comm
        The MPI communicator.
    sample_indices : NDArray
        The indices of the samples to save.

    Returns
    -------
    NDArray
        The gathered array of shape (num_energies, total_nnz).

    """
    rank = comm.Get_rank()
    world_rank = comm.Get_size()

    # if rank == 0:
    #     print(f"=== gather along nnz ===", flush=True)
    comm.barrier()

    # get num of nnz per rank, which is not evenly distributed since nnz 43824 is not necessarily divisible by num ranks
    # use lowercase gather since local_nnz is a generic Python object, not an NDArray
    nnz_lengths = comm.gather(array.shape[1], root=0, )
    comm.barrier()

    # data is strided (ex. a memory view, not contiguous), so make it contiguous first
    # [energy, nnz]
    data = xp.ascontiguousarray(array)
    comm.barrier()

    # print(f"Rank {rank}: data shape to gather: {data.shape}", flush=True)
    comm.barrier()
    # if rank == 0:
    #     print(f"NNZ lengths per rank: {nnz_lengths}", flush=True)
    #     print(f"sample indices length: {len(sample_indices)}", flush=True)
    comm.barrier()

    # to avoid index out of bounds errors, need to gather everything
    #   but only save the sampled indices
    # === original code (inefficient) ===
    recv_buffer = None
    if rank == 0:
        # [rank, energy, nnz]
        recv_buffer = xp.empty(
            (world_rank, array.shape[0], array.shape[1]), dtype=xp.complex128
        )

    comm.Gather(data, recv_buffer, root=0)
    comm.barrier()

    # concat the gathered data to remove the rank dimension
    if rank == 0:
        total_nnz = xp.sum(nnz_lengths)

        # concatenate all ranks along the nnz dimension
        concat = xp.zeros([array.shape[0], total_nnz], dtype=xp.complex128)
        current_index = 0
        for rk in range (world_rank):
            local_nnz = nnz_lengths[rk] 
            concat[:, current_index:current_index+local_nnz] = recv_buffer[rk, :, 0:local_nnz]
            current_index += local_nnz

        return concat[:, sample_indices]
    # === end original code  ====

    # === new code with gatherv ===
    # WARNING: doesn't work correctly
    #   it saves without error, but the nnz indices are jumbled up
    #   i.e. different num `mpiexec -n <NUM>` saves different results
    #   likely due to uneven padding nnz distribution, ignoring for now
    # gatherbuf = None
    # counts = None
    # displ = None
    # # calculate total size (for gatherbuf) and displacements
    # if rank == 0:
    #     total_nnz = xp.sum(nnz_lengths)
    #     gatherbuf = xp.empty((array.shape[0], total_nnz), dtype=xp.complex128)

    #     counts = xp.array(nnz_lengths) * array.shape[0]
    #     displ = xp.array( [sum(counts[:rk]) for rk in range(world_rank)] )

    #     print(f"Gather buffer shape: {gatherbuf.shape}", flush=True)
    #     print(f"Counts: {counts}", flush=True)
    #     print(f"Displacements: {displ}", flush=True)

    # comm.Gatherv(data, [gatherbuf, counts, displ, MPI.COMPLEX16], root=0)
    # comm.barrier()

    # if rank == 0:
    #     print(f"Gathered buffer shape: {gatherbuf.shape}", flush=True)
    #     return gatherbuf[:, sample_indices]
    # === end new code ===


@profiler.profile(level="debug")
def gather_array_stack(array: NDArray, comm: MPI.Comm, sample_indices: NDArray | None = None) -> NDArray:
    """Gathers a distributed array split along energy (ex. G, P, W), i.e. stack distribution.

    Parameters
    ----------
    array : NDArray
        The local array to gather. Must have shape (num_energies, local_nnz).
    comm : MPI.Comm
        The MPI communicator.
    sample_indices : NDArray
        The indices of the samples to save.

    Returns
    -------
    NDArray
        The gathered array of shape (num_energies, total_nnz).

    """
    rank = comm.Get_rank()
    world_rank = comm.Get_size()

    # if not provided, gather all nnz indices
    if sample_indices is None:
        sample_indices = xp.arange(array.shape[1])

    # if rank == 0:
    #     print(f"=== gather along energy ===", flush=True)
    # comm.barrier()

    # get num of energies per rank (not necessarily evenly distributed)
    # use lowercase gather since local_energy_length is a generic Python object, not an NDArray
    energy_lengths = comm.gather(array.shape[0], root=0)
    comm.barrier()

    # only gather the sampled nnz indices, since each rank has all the nnz data for its own energies
    data = xp.ascontiguousarray(array[:, sample_indices])
    comm.barrier()

    # print(f"Rank {rank}: data shape to gather: {data.shape}", flush=True)
    # comm.barrier()
    # if rank == 0:
    #     print(f"energy lengths per rank: {energy_lengths}", flush=True)
    #     print(f"sample indices length: {len(sample_indices)}", flush=True)
    # comm.barrier()

    # ==== original code (inefficient) ====
    # recv_buffer = None
    # if rank == 0:
    #     recv_buffer = xp.empty([world_rank, array.shape[0], len(sample_indices)], dtype=xp.complex128)
    
    # comm.Gather(data, recv_buffer, root=0)
    # comm.barrier()

    # if rank == 0:
    #     total_energy_length = xp.sum(energy_lengths)
    #     # concatentate along the energy dimension since each rank holds a chunk of energies

    #     concat = xp.zeros([total_energy_length, len(sample_indices)], dtype=xp.complex128)
    #     current_index = 0
    #     for rk in range (world_rank):
    #         local_energy_length = energy_lengths[rk] 
    #         concat[current_index:current_index+local_energy_length, :] = recv_buffer[rk, 0:local_energy_length, :]
    #         current_index += local_energy_length
    #     return concat
    # === end original code  ====

    # === new code with gatherv ===
    gatherbuf = None
    counts = None
    displ = None
    # calculate total size (for gatherbuf) and displacements
    if rank == 0:
        # mpi treats 2-D numpy arrays as contiguous 1-D arrays, so need to multiply lengths by the second dimension
        total_energy = xp.sum(energy_lengths)
        gatherbuf = xp.empty((total_energy, len(sample_indices)), dtype=xp.complex128)
        
        counts = xp.array(energy_lengths) * len(sample_indices)
        displ = xp.array( [sum(counts[:rk]) for rk in range(world_rank)] )

        # print(f"Gather buffer shape: {gatherbuf.shape}", flush=True)
        # print(f"Counts: {counts}", flush=True)
        # print(f"Displacements: {displ}", flush=True)

    comm.Gatherv(data, [gatherbuf, counts, displ, MPI.COMPLEX16], root=0)
    comm.barrier()
    
    if rank == 0:
        return gatherbuf
    # === end new code ===

@profiler.profile(level="debug")
def reduce_matrix_over_stack(array: NDArray, comm: MPI.Comm) -> NDArray:
    """Reduces a distributed matrix by sum(abs(A)) over all orbitals. 
    Works for stack-distributed arrays.

    Parameters
    ----------
    array : NDArray
        The local array to reduce. Must have shape (num_energies, local_nnz).
    comm : MPI.Comm
        The MPI communicator.

    Returns
    -------
    NDArray
        The reduced array of shape (num_energies, ).

    """
    rank = comm.Get_rank()
    world_rank = comm.Get_size()

    local_sum = xp.sum(xp.abs(array.data), axis=1)

    # get size of data per rank (number of energies in each rank)
    recvbuf = None
    if rank == 0:
        recvbuf = xp.empty_like(world_rank, dtype=array.dtype)

    # use lowercase gather since `.size` is a generic Python object, not an NDArray
    recvbuf = comm.gather(local_sum.size, root=0)
    energy_per_rank = recvbuf

    # can remove later, but cool to look at to see how energies are divided up
    # if rank == 0:
    #     print(f"Energies sizes per rank: {recvbuf}", flush=True)

    comm.barrier()

    # can't call Allreduce directly on local_sum since different ranks may have different sizes
    # instead use a Gatherv to gather all local_sums to rank 0
    gatherbuf = None
    displ = None
    if rank == 0:
        # calculate total size (for gatherbuf) and displacements
        total_energy = xp.sum(energy_per_rank)
        displ = xp.array( [sum(energy_per_rank[:rk]) for rk in range(world_rank)] )
        gatherbuf = xp.empty(total_energy, dtype=local_sum.dtype)

    comm.Gatherv(local_sum, [gatherbuf, energy_per_rank, displ, MPI.DOUBLE], root=0)

    return gatherbuf
