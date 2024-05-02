from mpi4py.MPI import comm

import numpy as np
from scipy import sparse


comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def distributed_load(self, path: str) -> sparse.coo_array:
    """Loads the given sparse matrix from disk and distributes it to all ranks."""

    if comm_rank == 0:
        hamiltonian = sparse.load_npz(path)
        if comm_size > 1:
            comm.bcast(hamiltonian, root=0)
    else:
        hamiltonian = comm.bcast(None, root=0)

    return hamiltonian


def slice_local_array(self, global_array: np.ndarray) -> None:
    """Computes the local slice of energies energies and return the corresponding
    sliced energy arraiy."""

    local_slice = np.array_split(global_array, comm_size)[comm_rank]

    return global_array[local_slice]
