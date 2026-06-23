# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.


from qttools.kernels.inplace.cupy.inplace import (
    scatter_add_scaled,
    scatter_add_scaled_obc,
)

__all__ = ["scatter_add_scaled", "scatter_add_scaled_obc"]
