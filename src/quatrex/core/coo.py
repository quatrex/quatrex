try:
    from cupyx import zeros_pinned as zeros

    _GPU = True
except ImportError:
    from numpy import zeros

    _GPU = False


import numpy as np


class COOBatch:

    def __init__(self, n_energies_per_rank: int, nnz: int) -> None:
        self.nnz = nnz
        self.n_energies_per_rank = n_energies_per_rank

        self.rows = zeros(nnz, dtype=np.int32)
        self.cols = zeros(nnz, dtype=np.int32)
        self.data = zeros(self.n_energies_per_rank, nnz, dtype=np.complex128)
