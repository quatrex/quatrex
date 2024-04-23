# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors. All rights reserved.
import logging
import time

import numpy as np
import numpy.linalg as npla
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def rgf_retarded(a: bsp.BSparse, save_off_diagonal: bool) -> bsp.BSparse:
    """Computes the causal Green's function.

    Parameters
    ----------
    a : bsp.BSparse
        System matrix.
    save_off_diag : bool
        If True, the off-diagonal blocks of the Green's function are
        saved, by default False

    Returns
    -------
    bsp.BSparse
        Causal Green's function.

    """

    x = bsp.zeros(a.bshape, a.dtype)

    x[0, 0] = npla.inv(a[0, 0])

    # Forwards sweep.
    t = time.perf_counter()
    for i in range(a.bshape[0] - 1):
        j = i + 1
        x[j, j] = npla.inv(a[j, j] - a[j, i] @ x[i, i] @ a[i, j])

    logger.debug(f"Forwards sweep completed ({time.perf_counter() - t:.2f} s).")

    # Backwards sweep.
    t = time.perf_counter()
    for i in range(a.bshape[0] - 2, -1, -1):
        j = i + 1

        x_ii = x[i, i]
        x_jj = x[j, j]
        a_ij = a[i, j]

        x_ji = -x_jj @ a[j, i] @ x_ii
        if save_off_diagonal:
            x[j, i] = x_ji
            x[i, j] = -x_ii @ a_ij @ x_jj

        x[i, i] = x_ii - x_ii @ a_ij @ x_ji

    logger.debug(f"Backwards sweep completed ({time.perf_counter() - t:.2f} s).")

    return x


def rgf_lesser_greater(
    a_diag_blocks: ArrayLike,
    a_lower_blocks: ArrayLike,
    a_upper_blocks: ArrayLike,
    sigma_l_diag_blocks: ArrayLike,
    sigma_l_lower_blocks: ArrayLike,
    sigma_l_upper_blocks: ArrayLike,
    sigma_g_diag_blocks: ArrayLike,
    sigma_g_lower_blocks: ArrayLike,
    sigma_g_upper_blocks: ArrayLike,
    save_off_diagonal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the lesser Green's function.

    Parameters
    ----------
    a_diag_blocks : ArrayLike
        Diagonal blocks of the block-tridiagonal system matrix.
    a_lower_blocks : ArrayLike
        Lower diagonal blocks of the block-tridiagonal system matrix.
    a_upper_blocks : ArrayLike
        Upper diagonal blocks of the block-tridiagonal system matrix.
    sigma_l_diag_blocks : ArrayLike
        Diagonal blocks of the block-tridiagonal self-energy matrix.
    sigma_l_lower_blocks : ArrayLike
        Lower diagonal blocks of the block-tridiagonal self-energy
        matrix.
    sigma_l_upper_blocks : ArrayLike
        Upper diagonal blocks of the block-tridiagonal self-energy
        matrix.
    sigma_g_diag_blocks : ArrayLike
        Diagonal blocks of the block-tridiagonal self-energy matrix.
    sigma_g_lower_blocks : ArrayLike
        Lower diagonal blocks of the block-tridiagonal self-energy
        matrix.
    sigma_g_upper_blocks : ArrayLike
        Upper diagonal blocks of the block-tridiagonal self-energy
        matrix.
    save_off_diagonal : bool
        If True, the off-diagonal blocks of the Green's function are
        saved.

    Returns
    -------
    ArrayLike
        Diagonal blocks of the Lesser Green's function.
    ArrayLike
        Lower diagonal blocks of the Lesser Green's function.
    ArrayLike
        Upper diagonal blocks of the Lesser Green's function.
    ArrayLike
        Diagonal blocks of the Greater Green's function.
    ArrayLike
        Lower diagonal blocks of the Greater Green's function.
    ArrayLike
        Upper diagonal blocks of the Greater Green's function.
    """

    return ()
