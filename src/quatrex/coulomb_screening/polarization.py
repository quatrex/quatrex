# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, xp
from qttools.datastructures import DSBSparse
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import synchronize_device
from qttools.utils.mpi_utils import get_section_sizes

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy

profiler = Profiler()


@profiler.profile(level="api")
def fft_correlate(a: NDArray, b: NDArray) -> NDArray:
    """Computes the correlation of two arrays using FFT.

    Parameters
    ----------
    a : NDArray
        First array.
    b : NDArray
        Second array.

    Returns
    -------
    NDArray
        The cross-correlation of the two arrays.

    """
    n = a.shape[0] + b.shape[0] - 1
    a_fft = xp.fft.fft(a.T, n, axis=1)
    b_fft = xp.fft.fft(b.T[::-1], n, axis=1)
    return xp.fft.ifft(a_fft * b_fft, axis=1).T


class PCoulombScreening(ScatteringSelfEnergy):
    """Computes the dynamic polarization from the electronic system.

    Parameters
    ----------
    quatrex_config : Path
        Quatrex configuration file.
    coulomb_screening_energies : NDArray
        The energies for the Coulomb screening

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        coulomb_screening_energies: NDArray,
    ) -> None:
        """Initializes the polarization."""
        self.energies = coulomb_screening_energies
        self.ne = len(self.energies)
        self.prefactor = -1j / xp.pi * xp.abs(self.energies[1] - self.energies[0])
        self.batch_size = compute_config.convolve.batch_size

    @profiler.profile(level="api")
    def compute(
        self, g_lesser: DSBSparse, g_greater: DSBSparse, out: tuple[DSBSparse, ...]
    ) -> None:
        """Computes the polarization.

        Parameters
        ----------
        g_lesser : DSBSparse
            The lesser Green's function.
        g_greater : DSBSparse
            The greater Green's function.
        out : tuple[DSBSparse, ...]
            The output matrices for the polarization. The order is
            p_lesser, p_greater, p_retarded.

        """
        p_lesser, p_greater, p_retarded = out

        # Barrier to synchronize ranks.
        t_all2all_start = time.perf_counter()
        with profiler.profile_range("stack->nnz transpose", level="debug"):
            # Transpose the matrices to nnz distribution.
            for m in (g_lesser, g_greater):
                # These should ideally already be in nnz-distribution.
                m.dtranspose() if m.distribution_state != "nnz" else None
            for m in (p_lesser, p_greater):
                # These only need the correct shape, so discard the data.
                m.dtranspose(discard=True) if m.distribution_state != "nnz" else None

        synchronize_device()
        t_all2all_end = time.perf_counter()
        comm.Barrier()
        t_all2all_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    PCoulombScreening: stack->nnz transpose time: {t_all2all_end-t_all2all_start:.3f}")
            print(
                f"    PCoulombScreening: stack->nnz transpose time all: {t_all2all_end_all-t_all2all_start:.3f}"
            )

        t_polarization_start = time.perf_counter()
        if p_greater.data.shape[-1] != 0:

            with profiler.profile_range("Polarization computation", level="debug"):

                if xp.__name__ == "cupy":
                    free_memory, _ = xp.cuda.Device().mem_info
                    num_buffers = 5 # closer to 4 but overapproximating
                    avail_buffer_size = free_memory // num_buffers
                    ne = g_lesser.data.shape[0]
                    no = g_greater.data.shape[-1]
                    batch_size = avail_buffer_size // (2 * ne * 16)  # 16 bytes for complex128
                    batch_size = min(batch_size, no)
                    batches = int(np.ceil(no / batch_size))
                    batch_size = int(np.ceil(no / batches))  # Balance last batch
                    self.batch_size = batch_size
                    if comm.rank == 0:
                        print(f"Free GiB: {free_memory/(1024**3):.3f}, Batches: {batches}, Batch size: {batch_size}", flush=True)
                else:
                    if self.batch_size is None:
                        # NOTE: This is a temporary solution. The batch size should be
                        # calculated in the configuration.
                        self.batch_size = p_greater.data.shape[-1]

                batch_counts, _ = get_section_sizes(
                    p_greater.data.shape[-1],
                    int(np.ceil(p_greater.data.shape[-1] / self.batch_size)),
                )

                batch_displacements = np.cumsum(
                    np.concatenate(([0], np.array(batch_counts)))
                )

                for start, end in zip(batch_displacements, batch_displacements[1:]):
                    batch = slice(start, end)

                    p_g_full = self.prefactor * fft_correlate(
                        g_greater.data[:, batch], -g_lesser.data[:, batch].conj()
                    )
                    p_l_full = -p_g_full[::-1].conj()
                    # TODO: the datastructures does not allow for easy slicing of the
                    # data. This is a workaround.
                    # Fill the matrices with the data. Take second part of the
                    # energy convolution.
                    p_lesser._data[p_lesser._stack_padding_mask, ..., batch] = p_l_full[
                        self.ne - 1 :
                    ]
                    p_greater._data[p_greater._stack_padding_mask, ..., batch] = p_g_full[
                        self.ne - 1 :
                    ]

        # Barrier before communication
        synchronize_device()
        t_polarization_end = time.perf_counter()
        comm.Barrier()
        t_polarization_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    PCoulombScreening: Polarization computation time: {t_polarization_end-t_polarization_start:.3f}")
            print(f"    PCoulombScreening: Polarization computation time all: {t_polarization_end_all-t_polarization_start:.3f}")


        t_all2all2_start = time.perf_counter()
        # Transpose the matrices to stack distribution.
        with profiler.profile_range("nnz->stack transpose", level="debug"):
            t0 = time.perf_counter()
            for m in (p_lesser, p_greater):
                m.dtranspose() if m.distribution_state != "stack" else None
            # NOTE: The Green's functions must not be transposed back to
            # stack distribution, as they are needed in nnz distribution for
            # the other interactions.
        synchronize_device()
        t_all2all2_end = time.perf_counter()
        comm.Barrier()
        t_all2all2_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    PCoulombScreening: nnz->stack transpose time: {t_all2all2_end-t_all2all2_start:.3f}")
            print(f"    PCoulombScreening: nnz->stack transpose time all: {t_all2all2_end_all-t_all2all2_start:.3f}")

        # Enforce anti-Hermitian symmetry and calculate Pr.
        t_symmetrization_start = time.perf_counter()
        p_lesser.symmetrize(xp.subtract)
        p_greater.symmetrize(xp.subtract)

        # Discard the real part.
        p_lesser._data.real = 0
        p_greater._data.real = 0

        p_retarded.data = (p_greater.data - p_lesser.data) / 2

        synchronize_device()
        t_symmetrization_end = time.perf_counter()
        comm.Barrier()
        t_symmetrization_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    PCoulombScreening: Symmetrization time: {t_symmetrization_end-t_symmetrization_start:.3f}")
            print(f"    PCoulombScreening: Symmetrization time all: {t_symmetrization_end_all-t_symmetrization_start:.3f}")
