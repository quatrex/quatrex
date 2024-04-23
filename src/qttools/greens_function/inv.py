from qttools.datastructures.coogroup import COOGroup
import scipy.sparse as sps
import numpy as np


def _densify(a: COOGroup) -> np.ndarray:
    res = np.array(
        [
            sps.coo_matrix((a.data[i], (a.rows, a.cols)), shape=a.shape[1:]).toarray()
            for i in range(a.length)
        ]
    )
    return res


def _sparsify(a: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> COOGroup:
    res = COOGroup(
        length=a.shape[0],
        data=a[:, rows, cols],
        rows=rows,
        cols=cols,
    )
    return res


def inv_retarded(a: COOGroup) -> np.ndarray:
    """Computes the retarded Green's function.

    Parameters
    ----------
    a : COOGroup
        System matrix.

    Returns
    -------
    COOGroup
        Retarded Green's function.

    """

    a_dense = _densify(a)

    return np.linalg.inv(a_dense)


def inv_lesser_greater(
    a: COOGroup,
    sigma_lesser: COOGroup,
    sigma_greater: COOGroup,
) -> np.ndarray:
    x = inv_retarded(a)
    sigma_lesser_dense = _densify(sigma_lesser)
    sigma_greater_dense = _densify(sigma_greater)

    x_lesser = x @ sigma_lesser_dense @ x.conj().transpose((0, 2, 1))
    x_greater = x @ sigma_greater_dense @ x.conj().transpose((0, 2, 1))

    return _sparsify(x_lesser, a.rows, a.cols), _sparsify(x_greater, a.rows, a.cols)
