import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from scipy import sparse


def distributed_load(path: str) -> sparse.coo_array:
    """Loads the given sparse matrix from disk and distributes it to all ranks."""

    if comm.rank == 0:
        sparse_array = sparse.load_npz(path)
        if comm.size > 1:
            comm.bcast(sparse_array, root=0)
    else:
        sparse_array = comm.bcast(None, root=0)

    return sparse_array


def slice_local_array(global_array: np.ndarray) -> None:
    """Computes the local slice of energies energies and return the corresponding
    sliced energy arraiy."""

    local_slice = np.array_split(global_array, comm.size)[comm.rank]

    return global_array[local_slice]
