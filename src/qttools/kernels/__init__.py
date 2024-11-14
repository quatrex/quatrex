from qttools import xp

if xp.__name__ == "numpy":
    from qttools.kernels.numba import dsbcoo as dsbcoo_kernels
    from qttools.kernels.numba import dsbcsr as dsbcsr_kernels
    from qttools.kernels.numba import dsbsparse as dsbsparse_kernels
    from qttools.kernels.numba import obc as obc_kernels
elif xp.__name__ == "cupy":
    from qttools.kernels.cuda import dsbcoo as dsbcoo_kernels
    from qttools.kernels.cuda import dsbcsr as dsbcsr_kernels
    from qttools.kernels.cuda import dsbsparse as dsbsparse_kernels
    from qttools.kernels.cuda import obc as obc_kernels

else:
    raise ValueError(f"Unrecognized ARRAY_MODULE '{xp.__name__}'")


__all__ = ["dsbsparse_kernels", "dsbcoo_kernels", "dsbcsr_kernels", "obc_kernels"]
