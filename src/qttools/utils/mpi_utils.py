# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.


from pathlib import Path

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from scipy import sparse


def get_section_sizes(num_elements: int, num_sections: int = comm.size):
    """Computes the number of un-evenly divided elements per section."""
    quotient, remainder = divmod(num_elements, num_sections)
    section_sizes = remainder * [quotient + 1] + (num_sections - remainder) * [quotient]
    effective_num_elements = max(section_sizes) * num_sections
    return section_sizes, effective_num_elements


def distributed_load(path: Path) -> sparse.sparray | np.ndarray:
    """Loads the given sparse matrix from disk and distributes it to all ranks."""

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
    """Computes the local slice of energies energies and return the corresponding
    sliced energy arraiy."""
    section_sizes, __ = get_section_sizes(global_array.shape[-1])
    section_offsets = np.cumsum([0] + section_sizes)

    return global_array[
        ..., section_offsets[comm.rank] : section_offsets[comm.rank + 1]
    ]
