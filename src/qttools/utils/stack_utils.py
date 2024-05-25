# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np


def scale_stack(stacked: np.ndarray, factor: np.ndarray, axis: int = 0) -> np.ndarray:
    """Scales the given stack by the given factor."""
    if not stacked.shape[axis] == factor.shape[0]:
        raise ValueError("The shape of the stack and the factor do not match.")

    for i in range(stacked.shape[axis]):
        stacked.swapaxes(axis, 0)[i] *= factor[i]

    return stacked
