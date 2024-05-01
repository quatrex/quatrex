import cupy as cp
from cupy.scipy.sparse import spmatrix


def densify_block(a: spmatrix) -> cp.ndarray:
    """Densify a sparse block given on the GPU.

    Parameters
    ----------
    a : spmatrix
        Sparse matrix to densify. Stored on the GPU.

    Returns
    -------
    cp.ndarray
        Densified matrix. Stored on the GPU.
    """

    return a.todense()
