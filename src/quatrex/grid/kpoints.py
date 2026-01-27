# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np

from qttools import NDArray


def monkhorst_pack(size: tuple[int]) -> NDArray:
    """Constructs a Monkhorst-Pack grid of k-points.

    Parameters
    ----------
    size : tuple[int]
        Grid dimensions as (nx, ny, nz) specifying the number of
        k-points along each reciprocal lattice direction.

    Returns
    -------
    kpts : NDArray
        Array of k-points with shape (nx*ny*nz, 3). Each row contains
        the (kx, ky, kz) coordinates of a k-point in reduced units.

    """
    kpts = np.indices(size).transpose((1, 2, 3, 0)).reshape((-1, 3))
    return (kpts + 0.5) / size - 0.5
