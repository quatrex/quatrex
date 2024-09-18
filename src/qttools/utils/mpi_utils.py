# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.


from pathlib import Path

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from scipy import sparse


def get_section_sizes(num_elements: int, num_sections: int = comm.size) -> tuple:
    """Computes the number of un-evenly divided elements per section.

    Parameters
    ----------
    num_elements : int
        The total number of elements to divide.
    num_sections : int
        The number of sections to divide the elements into. Defaults to
        the number of MPI ranks.

    Returns
    -------
    section_sizes, effective_num_elements : tuple
        A tuple containing the sizes of each section and the effective
        number of elements after division.

    """
    quotient, remainder = divmod(num_elements, num_sections)
    section_sizes = remainder * [quotient + 1] + (num_sections - remainder) * [quotient]
    effective_num_elements = max(section_sizes) * num_sections
    return section_sizes, effective_num_elements


def distributed_load(path: Path) -> sparse.sparray | np.ndarray:
    """Loads an array from disk and distributes it to all ranks.

    Parameters
    ----------
    path : Path
        The path to the file to load.

    Returns
    -------
    sparse.sparray | np.ndarray
        The loaded array.

    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix not in [".npz", ".npy"]:
        raise ValueError(f"Invalid file extension: {path.suffix}")

    if comm.rank == 0:
        if path.suffix == ".npz":
            arr = sparse.load_npz(path)
        elif path.suffix == ".npy":
            arr = np.load(path)

    else:
        arr = None

    arr = comm.bcast(arr, root=0)

    return arr


def get_local_slice(global_array: np.ndarray) -> None:
    """Returns the local slice of a distributed array.

    Parameters
    ----------
    global_array : np.ndarray
        The global array to slice.

    Returns
    -------
    np.ndarray
        The local slice of the global array.

    """
    section_sizes, __ = get_section_sizes(global_array.shape[-1])
    section_offsets = np.cumsum([0] + section_sizes)

    return global_array[
        ..., section_offsets[comm.rank] : section_offsets[comm.rank + 1]
    ]


def check_gpu_aware_mpi() -> bool:
    """Checks if the MPI implementation is GPU-aware.

    This is done by inspecting the MPI info object for the presence of
    the "gpu" memory allocation kind. See [1]_ for more info.

    Returns
    -------
    bool
        True if the MPI implementation is GPU-aware on all ranks, False
        otherwise.

    References
    ----------
    .. [1] https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node279.htm

    """
    info = comm.Get_info()
    gpu_aware = "gpu" in info.get("mpi_memory_alloc_kinds")
    return comm.allreduce(gpu_aware, op=MPI.LAND)
