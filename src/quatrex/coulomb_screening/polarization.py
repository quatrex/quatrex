# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from qttools.datastructures import DSBSparse
from qttools.utils.gpu_utils import xp

from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy


def fft_correlate(a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
    """Computes the correlation of two arrays using the FFT."""
    n = a.shape[0] + b.shape[0] - 1
    a_fft = xp.fft.fftn(a, (n,), axes=(0,))
    b_fft = xp.fft.fftn(b[::-1], (n,), axes=(0,))
    return xp.fft.ifftn(a_fft * b_fft, axes=(0,))


class PCoulombScreening(ScatteringSelfEnergy):
    def __init__(
        self,
        config: QuatrexConfig,
        coulomb_screening_energies: xp.ndarray,
    ) -> None:
        self.energies = coulomb_screening_energies
        self.prefactor = -1j / np.pi * (self.energies[1] - self.energies[0])

    def compute(
        self, g_lesser: DSBSparse, g_greater: DSBSparse, out: tuple[DSBSparse, ...]
    ) -> None:
        """Computes the polarization."""
        p_lesser, p_greater, p_retarded = out
        # Transpose the matrices to nnz distribution.
        for m in (g_lesser, g_greater, p_lesser, p_greater, p_retarded):
            m.dtranspose() if m.distribution_state != "nnz" else None

        p_g_full = self.prefactor * fft_correlate(g_greater.data, -g_lesser.data.conj())
        p_l_full = -p_g_full[::-1].conj()
        p_lesser._data[
            p_lesser._stack_padding_mask, ..., : p_lesser.nnz_section_sizes[comm.rank]
        ] = p_l_full[: g_lesser.shape[0]]
        p_greater._data[
            p_greater._stack_padding_mask, ..., : p_greater.nnz_section_sizes[comm.rank]
        ] = p_g_full[: g_greater.shape[0]]
        p_retarded._data = (p_greater._data - p_lesser._data) / 2

        # Transpose the matrices to stack distribution.
        for m in (g_lesser, g_greater, p_lesser, p_greater, p_retarded):
            m.dtranspose() if m.distribution_state != "stack" else None
