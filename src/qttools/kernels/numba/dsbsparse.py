# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.njit(parallel=True, cache=True)
def find_ranks(nnz_section_offsets: NDArray, inds: NDArray) -> NDArray:
    """Find the ranks of the indices in the offsets.

    Parameters
    ----------
    nnz_section_offsets : NDArray
        The offsets of the non-zero sections.
    inds : NDArray
        The indices to find the ranks for.


    Returns
    -------
    ranks : NDArray
        The ranks of the indices in the offsets.

    """
    ranks = np.zeros(inds.shape[0], dtype=np.int16)
    for i in nb.prange(inds.shape[0]):
        for j in range(nnz_section_offsets.shape[0]):
            cond = int(nnz_section_offsets[j] <= inds[i])
            ranks[i] = ranks[i] * (1 - cond) + j * cond

    return ranks
