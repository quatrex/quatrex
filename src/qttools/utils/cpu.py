import numpy as np
from scipy.sparse import spmatrix


def densify_block(a: spmatrix) -> np.ndarray:
    """Densify a sparse block.

    Parameters
    ----------
    a : spmatrix
        Sparse matrix to densify.

    Returns
    -------
    cp.ndarray
        Densified matrix.
    """

    return a.todense()
