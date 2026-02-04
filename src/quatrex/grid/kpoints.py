# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np

from qttools import NDArray


def monkhorst_pack(size: tuple[int], shift: tuple[float]) -> NDArray:
    """Constructs a Monkhorst-Pack grid of k-points.

    This is implemented to produce arbitrary dimensions of k-point grids.

    Parameters
    ----------
    size : tuple[int]
        Grid dimensions specifying the number of
        k-points along each reciprocal lattice direction.
    shift : tuple[float]
        Shift of the grid in each direction. Each component should be in the range (-1, 1).

    Returns
    -------
    kpts : NDArray
        Array of k-points with shape (nx*ny*nz, 3). Each row contains
        the (kx, ky, kz) coordinates of a k-point in reduced units.

    """
    kpts = np.moveaxis(np.indices(size), 0, -1).reshape((-1, len(size)))
    return (kpts + 0.5) / size - 0.5 + np.array(shift)
