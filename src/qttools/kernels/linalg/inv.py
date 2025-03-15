# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, xp
from qttools.profiling import Profiler

profiler = Profiler()


def _invert(a: NDArray) -> NDArray:
    return xp.linalg.inv(a)


def _solve(a: NDArray) -> NDArray:
    return xp.linalg.solve(a, xp.broadcast_to(xp.eye(a.shape[-1]), a.shape))


_inv = _invert
if xp.__name__ == "cupy":
    name = xp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
    if name.startswith("NVIDIA"):
        from cupy.cublas import set_batched_gesv_limit

        set_batched_gesv_limit(1024)
        _inv = _solve


@profiler.profile(level="debug")
def inv(
    a: NDArray,
) -> NDArray:
    """Computes the (batched) inverse of a matrix.

    Parameters
    ----------
    a : NDArray
        The (batched) matrix.

    Returns
    -------
    NDArray
        The inverse (batched) of the matrix.

    """
    return _inv(a)
