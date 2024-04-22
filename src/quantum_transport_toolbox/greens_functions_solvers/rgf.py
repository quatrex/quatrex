# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors. All rights reserved.
import logging
import time

import numpy as np
import numpy.linalg as npla
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def _rgf_solve(
    a_diag_blocks: ArrayLike,
    a_lower_blocks: ArrayLike,
    a_upper_blocks: ArrayLike,
    b_diag_blocks: ArrayLike,
    b_lower_blocks: ArrayLike,
    b_upper_blocks: ArrayLike,
    symmetric: bool = False,
    save_off_diagonal: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solves a block-tridiagonal linear system using RGF.

    Parameters
    ----------
    a_diag_blocks : ArrayLike
        Diagonal blocks of the block-tridiagonal system matrix.
    a_lower_blocks : ArrayLike
        Lower diagonal blocks of the block-tridiagonal system matrix.
    a_upper_blocks : ArrayLike
        Upper diagonal blocks of the block-tridiagonal system matrix.
    b_diag_blocks : ArrayLike
        Diagonal blocks of the block-tridiagonal right-hand side of the
        linear system.
    b_lower_blocks : ArrayLike
        Lower diagonal blocks of the block-tridiagonal right-hand side
        of the linear system.
    b_upper_blocks : ArrayLike
        Upper diagonal blocks of the block-tridiagonal right-hand side
        of the linear system.
    symmetric : bool, optional
        If True, the system matrix is assumed to be symmetric.
    save_off_diagonal : bool
        If True, the off-diagonal blocks of the Green's function are
        saved.

    Returns
    -------
    ArrayLike
        Diagonal blocks solution of the linear system.
    ArrayLike
        Lower diagonal blocks solution of the linear system.
    ArrayLike
        Upper diagonal blocks solution of the linear system.
    """

    return ()


def _rgf_lesser(
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
