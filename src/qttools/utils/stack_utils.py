# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import NDArray


def scale_stack(stacked: NDArray, factor: NDArray, axis: int = 0) -> NDArray:
    """Scales the given stack by the given factor.

    Parameters
    ----------
    stacked : NDArray
        The stack to scale.
    factor : NDArray
        The factor to scale the stack by.
    axis : int, optional
        The axis along which to scale the stack.

    Returns
    -------
    NDArray
        The scaled stack of arrays

    """
    if not stacked.shape[axis] == factor.shape[0]:
        raise ValueError("The shape of the stack and the factor do not match.")

    for i in range(stacked.shape[axis]):
        stacked.swapaxes(axis, 0)[i] *= factor[i]

    return stacked
