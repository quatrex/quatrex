# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.


from pathlib import Path

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from scipy import sparse


def get_num_elements_per_section(num_elements, num_sections=comm.size):
    """Computes the number of un-evenly divided elements per section."""
    quotient, remainder = divmod(num_elements, num_sections)
    section_size = quotient + remainder
    total_size = section_size * num_sections
    num_elements_per_section = [section_size] * (num_sections - 1) + [
        num_elements - section_size * (num_sections - 1)
    ]
    return num_elements_per_section, total_size


def distributed_load(path: Path) -> sparse.sparray | np.ndarray:
    """Loads the given sparse matrix from disk and distributes it to all ranks."""

    if comm.rank == 0:
        if path.suffix == ".npz":
            arr = sparse.load_npz(path)
        elif path.suffix == ".npy":
            arr = np.load(path)
        else:
            raise ValueError(f"Unknown file extension: {path.suffix}")

        if comm.size > 1:
            comm.bcast(arr, root=0)

    else:
        arr = comm.bcast(None, root=0)

    return arr


def get_local_slice(global_array: np.ndarray) -> None:
    """Computes the local slice of energies energies and return the corresponding
    sliced energy arraiy."""
    num_elements_per_section, __ = get_num_elements_per_section(global_array.shape[-1])
    section_offsets = np.cumsum([0] + num_elements_per_section)

    return global_array[
        ..., section_offsets[comm.rank] : section_offsets[comm.rank + 1]
    ]
