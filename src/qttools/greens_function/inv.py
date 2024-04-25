from qttools.datastructures.dbcsr import DBCSR
import scipy.sparse as sps
import numpy as np


def inv_retarded(a: DBCSR) -> np.ndarray:
    """Computes the retarded Green's function.

    Parameters
    ----------
    a : DBCSR
        System matrix.

    Returns
    -------
    np.ndarray
        Retarded Green's function.

    """

    return np.linalg.inv(a.to_dense())


def inv_lesser_greater(
    a: DBCSR,
    sigma_lesser: DBCSR,
    sigma_greater: DBCSR,
    return_retarded: bool = False,
) -> np.ndarray:
    x = inv_retarded(a)

    x_l = x @ sigma_lesser.to_dense() @ x.conj().transpose((0, 2, 1))
    x_g = x @ sigma_greater.to_dense() @ x.conj().transpose((0, 2, 1))

    x_lesser = DBCSR.zeros_like(a)
    x_greater = DBCSR.zeros_like(a)
    for i in range(sigma_lesser.num_blocks):
        _i = slice(*sigma_lesser.block_offsets[i : i + 2])
        x_lesser.set_block(i, i, x_l[..., _i, _i])
        x_greater.set_block(i, i, x_g[..., _i, _i])

    for i in range(sigma_lesser.num_blocks - 1):
        _i = slice(*sigma_lesser.block_offsets[i : i + 2])
        _j = slice(*sigma_lesser.block_offsets[i + 1 : i + 2 + 1])
        x_lesser.set_block(i, i + 1, x_l[..., _i, _j])
        x_lesser.set_block(i + 1, i, x_l[..., _j, _i])
        x_greater.set_block(i, i + 1, x_g[..., _i, _j])
        x_greater.set_block(i + 1, i, x_g[..., _j, _i])

    if not return_retarded:
        return x_lesser, x_greater

    x_retarded = DBCSR.zeros_like(a)
    for i in range(sigma_lesser.num_blocks):
        _i = slice(*sigma_lesser.block_offsets[i : i + 2])
        x_retarded.set_block(i, i, x[..., _i, _i])

    for i in range(sigma_lesser.num_blocks - 1):
        _i = slice(*sigma_lesser.block_offsets[i : i + 2])
        _j = slice(*sigma_lesser.block_offsets[i + 1 : i + 2 + 1])
        x_retarded.set_block(i, i + 1, x[..., _i, _j])
        x_retarded.set_block(i + 1, i, x[..., _j, _i])

    return x_lesser, x_greater, x_retarded
