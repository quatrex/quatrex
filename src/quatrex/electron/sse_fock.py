# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time

import numpy as np
from qttools import NDArray, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import synchronize_device

from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy

profiler = Profiler()


@profiler.profile(level="api")
def fft_circular_convolve(a: xp.ndarray, b: xp.ndarray, axes: tuple[int]) -> xp.ndarray:
    """Computes the circular convolution of two arrays using the FFT."""
    # Extract the shapes of the arrays along the axes as tuples.
    nka = tuple(a.shape[i] for i in axes)
    nkb = tuple(b.shape[i] for i in axes)
    a_fft = xp.fft.fftn(a, nka, axes=axes)
    b_fft = xp.fft.fftn(b, nkb, axes=axes)
    return xp.fft.ifftn(a_fft * b_fft, axes=axes)


class SigmaFock(ScatteringSelfEnergy):
    """Computes the bare Fock self-energy.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The Quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.
    electron_energies : NDArray
        The energies for the electron system.

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        coulomb_matrix: DSDBSparse,
        electron_energies: NDArray,
    ):
        """Initializes the bare Fock self-energy."""
        self.energies = electron_energies
        number_of_kpoints = quatrex_config.electron.number_of_kpoints
        self.prefactor = (
            1j
            / (2 * xp.pi * np.prod(number_of_kpoints))
            * (self.energies[1] - self.energies[0])
        )
        (
            coulomb_matrix.dtranspose()
            if coulomb_matrix.distribution_state != "nnz"
            else None
        )
        self.coulomb_matrix_data = (
            coulomb_matrix.data[0] / quatrex_config.coulomb_screening.epsilon_r
        )

    @profiler.profile(level="api")
    def compute(self, g_lesser: DSDBSparse, out: tuple[DSDBSparse, ...]) -> None:
        """Computes the Fock self-energy.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The lesser Green's function.
        out : tuple[DSDBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_retarded.

        """
        # TODO: Check again if we really need to transpose the matrices
        # here.
        t_all2all_start = time.perf_counter()
        (sigma_retarded,) = out
        for m in (g_lesser, sigma_retarded):
            # These should both already be in nnz-distribution.
            m.dtranspose() if m.distribution_state != "nnz" else None
        synchronize_device()
        t_all2all_end = time.perf_counter()
        comm.barrier()
        t_all2all_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    SigmaFock: stack->nnz transpose: {t_all2all_end - t_all2all_start:.3f} s",
                flush=True,
            )
            print(
                f"    SigmaFock: stack->nnz transpose all: {t_all2all_end_all - t_all2all_start:.3f} s",
                flush=True,
            )

        # Compute the electron density by summing over energies.
        t_sse_start = time.perf_counter()
        gl_density = self.prefactor * g_lesser.data.sum(axis=0)
        sigma_retarded.data += fft_circular_convolve(
            gl_density,
            self.coulomb_matrix_data,
            axes=tuple(range(gl_density.ndim - 1)),
        )
        synchronize_device()
        t_sse_end = time.perf_counter()
        comm.barrier()
        t_sse_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    SigmaFock: SSE computation: {t_sse_end - t_sse_start:.3f} s",
                flush=True,
            )
            print(
                f"    SigmaFock: SSE computation all: {t_sse_end_all - t_sse_start:.3f} s",
                flush=True,
            )

        # NOTE: The electron Green's functions and self-energies must
        # not be transposed back to stack distribution, as they are
        # needed in nnz distribution for the other interactions.
