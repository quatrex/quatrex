# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from qttools import xp


def scale_stack(stacked: xp.ndarray, factor: xp.ndarray, axis: int = 0) -> xp.ndarray:
    """Scales the given stack by the given factor."""
    if not stacked.shape[axis] == factor.shape[0]:
        raise ValueError("The shape of the stack and the factor do not match.")

    for i in range(stacked.shape[axis]):
        stacked.swapaxes(axis, 0)[i] *= factor[i]

    return stacked
