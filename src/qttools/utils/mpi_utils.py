# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pickle
from pathlib import Path

import scipy.sparse as sps
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from mpi4py.util import pkl5

from qttools import NDArray, sparse, xp
from qttools.profiling import Profiler

profiler = Profiler()
comm = pkl5.Intracomm(comm)


@profiler.profile(level="debug")
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


@profiler.profile(level="debug")
def distributed_load(path: Path) -> sparse.spmatrix | NDArray:
    """Loads an array from disk and distributes it to all ranks.

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
    if path.suffix not in [".npz", ".npy", ".pkl"]:
        raise ValueError(f"Invalid file extension: {path.suffix}")

    if comm.rank == 0:
        if path.suffix == ".npz":
            arr = sps.load_npz(path)
            arr = sparse.coo_matrix(arr)
        elif path.suffix == ".npy":
            arr = xp.load(path)
        elif path.suffix == ".pkl":
            with open(path, "rb") as f:
                arr = pickle.load(f)
            for k in arr.keys():
                arr[k] = sparse.coo_matrix(arr[k])

    else:
        arr = None

    arr = comm.bcast(arr, root=0)

    return arr


@profiler.profile(level="debug")
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
