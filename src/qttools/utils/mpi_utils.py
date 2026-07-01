# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from pathlib import Path

import scipy.sparse as sps
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from mpi4py.util import pkl5

from qttools import NDArray, sparse, xp
from qttools.utils.hdf5_utils import load_hdf5_dict

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


def distributed_load(path: Path) -> sparse.spmatrix | NDArray | dict:
    """Loads an array from disk and broadcasts it to all ranks.

    Parameters
    ----------
    path : Path
        The path to the file to load.

    Returns
    -------
    sparse.spmatrix | NDArray | dict
        The loaded array/s.

    Raises
    ------
    FileNotFoundError
        Occurs on every rank where the file does not exist.

    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix not in [".npz", ".npy", ".h5", ".txt"]:
        raise ValueError(f"Invalid file extension: {path.suffix}")

    if comm.rank == 0:
        if path.suffix == ".npz":
            # NOTE: cupyx.scipy.sparse.load_npz does not exist.
            arr = sps.load_npz(path)
            arr = sparse.coo_matrix(arr)
        elif path.suffix == ".npy":
            arr = xp.load(path)
        elif path.suffix == ".h5":
            arr = load_hdf5_dict(path)
            arr = {
                tuple(map(int, r.strip("[]").split(","))): h_r
                for r, h_r in arr.items()
                if r.startswith("[")
            }
        elif path.suffix == ".txt":
            # Assumes the text file contains integers.
            arr = xp.loadtxt(path, dtype=int)
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


def gather_array_stack(array: NDArray, comm: MPI.Comm, sample_indices: NDArray | None = None) -> NDArray | None:
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

    # get num of energies per rank (not necessarily evenly distributed)
    # use lowercase gather since local_energy_length is a generic Python object, not an NDArray
    energy_lengths = comm.gather(array.shape[0], root=0)
    comm.barrier()

    # only gather the sampled nnz indices, since each rank has all the nnz data for its own energies
    data = xp.ascontiguousarray(array[:, sample_indices])
    comm.barrier()

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

    comm.Gatherv(data, [gatherbuf, counts, displ, MPI.COMPLEX16], root=0)
    comm.barrier()
    
    if rank == 0:
        return gatherbuf
    else:
        return None
