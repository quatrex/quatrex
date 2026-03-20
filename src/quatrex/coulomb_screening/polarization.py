# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, xp
from qttools.datastructures import DSDBSparse
from qttools.fft import fft_convolve, fft_correlate_kpoints
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import free_mempool
from qttools.utils.mpi_utils import get_section_sizes
from quatrex.core.config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy

profiler = Profiler()

if xp.__name__ == "cupy":
    cache = xp.fft.config.get_plan_cache()


def hilbert_transform(a: NDArray, energies: NDArray) -> NDArray:
    """Computes the Hilbert transform of the array a, assuming the symmetries of the
    polarization, i.e \([P^{\lessgtr}_{ij}(\omega)]^{\dagger} = -P^{\gtrless}_{ij}(-\omega)\).
    This becomes \(a(-\omega)=a^{*}(\omega)\), where a is \(a=P^>-P^<\).

    Assumes that the first axis corresponds to the energy axis.

    Parameters
    ----------
    a : NDArray
        The array to transform.
    energies : NDArray
        The energy values corresponding to the first axis of a.

    Returns
    -------
    NDArray
         The Hilbert transform of a.

    """
    # eta for removing the singularity. See Cauchy principal value.
    de = energies[1] - energies[0]
    eta = de / 2
    energy_differences = (
        xp.expand_dims(energies - energies[0], tuple(range(1, a.ndim))) + eta
    )
    ne = energies.size

    hilbert_kernel = 1 / energy_differences
    b = fft_convolve(a, hilbert_kernel)[:ne]
    # Negative frequencies of a
    b += fft_convolve(a[::-1].conj(), hilbert_kernel)[-ne:]
    # Negative frequencies of the kernel
    hilbert_kernel = -hilbert_kernel[::-1]
    b += fft_convolve(a, hilbert_kernel)[-ne:]

    return b


class PCoulombScreening(ScatteringSelfEnergy):
    """Computes the dynamic polarization from the electronic system.

    Parameters
    ----------
    config : QuatrexConfig
        Quatrex configuration object.
    coulomb_screening_energies : NDArray
        The energies for the Coulomb screening

    """

    def __init__(
        self, config: QuatrexConfig, coulomb_screening_energies: NDArray
    ) -> None:
        """Initializes the polarization."""
        self.energies = coulomb_screening_energies
        self.kpoint_volume = np.prod(config.device.kpoint_grid)
        self.ne = len(self.energies)
        self.prefactor = (
            -1j
            / (xp.pi)
            * xp.abs(self.energies[1] - self.energies[0])
            / self.kpoint_volume
        )
        self.batch_size = config.compute.convolve.batch_size

        self.use_approximation = config.coulomb_screening.use_polarization_approximation
        self.compute_hilbert_retarded = (
            config.coulomb_screening.compute_hilbert_retarded_polarization
        )

    @profiler.profile(label="PCoulombScreening", level="default", comm=comm)
    def compute(
        self, g_lesser: DSDBSparse, g_greater: DSDBSparse, out: tuple[DSDBSparse, ...]
    ) -> None:
        """Computes the polarization.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The lesser Green's function.
        g_greater : DSDBSparse
            The greater Green's function.
        out : tuple[DSDBSparse, ...]
            The output matrices for the polarization. The order is
            p_lesser, p_greater, p_retarded.

        """
        p_lesser, p_greater, p_retarded = out

        # Barrier to synchronize ranks.
        with profiler.profile_range(
            label="PCoulombScreening: stack->nnz transpose", level="default", comm=comm
        ):

            # Transpose the matrices to nnz distribution.
            for m in (g_lesser, g_greater):
                # These should ideally already be in nnz-distribution.
                m.dtranspose() if m.distribution_state != "nnz" else None
            for m in (p_lesser, p_greater):
                # These only need the correct shape, so discard the data.
                m.dtranspose(discard=True) if m.distribution_state != "nnz" else None
            if self.compute_hilbert_retarded:
                (
                    p_retarded.dtranspose(discard=True)
                    if (p_retarded.distribution_state != "nnz")
                    else None
                )

        with profiler.profile_range(
            label="PCoulombScreening: Polarization computation",
            level="default",
            comm=comm,
        ):

            if p_greater.data.shape[-1] != 0:
                if xp.__name__ == "cupy":
                    free_mempool()
                    free_memory, _ = xp.cuda.Device().mem_info
                    num_buffers = 6  # closer to 4 but overapproximating
                    avail_buffer_size = free_memory // num_buffers
                    ne = g_lesser.data.shape[0]
                    no = g_greater.data.shape[-1]
                    batch_size = avail_buffer_size // (
                        2 * ne * 16
                    )  # 16 bytes for complex128
                    batch_size = max(min(batch_size, no), 1)
                    batches = int(np.ceil(no / batch_size))
                    batch_size = int(np.ceil(no / batches))  # Balance last batch
                    if self.batch_size is not None and batch_size < self.batch_size:
                        cache.clear()
                    self.batch_size = batch_size
                    if comm.rank == 0:
                        print(
                            f"Free GiB: {free_memory/(1024**3):.3f}, Batches: {batches}, Batch size: {batch_size}",
                            flush=True,
                        )
                        print(cache.show_info(), flush=True)
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

                    p_g_full = self.prefactor * fft_correlate_kpoints(
                        g_greater.data[..., batch], -g_lesser.data[..., batch].conj()
                    )
                    p_l_full = -p_g_full[::-1].conj()
                    # TODO: the datastructures does not allow for easy slicing of the
                    # data. This is a workaround.
                    # Fill the matrices with the data. Take second part of the
                    # energy convolution.
                    p_lesser.data[..., batch] = p_l_full[self.ne - 1 :]
                    p_greater.data[..., batch] = p_g_full[self.ne - 1 :]
                    # Note that only the hermitian part is computed here.

                    if self.compute_hilbert_retarded:
                        p_retarded.data[..., batch] = (
                            -(self.prefactor / 2)
                            * (
                                hilbert_transform(
                                    (
                                        p_greater.data[..., batch]
                                        - p_lesser.data[..., batch]
                                    ),
                                    self.energies,
                                )
                            )
                            * self.kpoint_volume
                        )

        with profiler.profile_range(
            label="PCoulombScreening: nnz->stack transpose", level="default", comm=comm
        ):

            # Transpose the matrices to stack distribution.
            for m in (p_lesser, p_greater):
                m.dtranspose() if m.distribution_state != "stack" else None
            if self.compute_hilbert_retarded:
                (
                    p_retarded.dtranspose()
                    if (p_retarded.distribution_state != "stack")
                    else None
                )
            # NOTE: The Green's functions must not be transposed back to
            # stack distribution, as they are needed in nnz distribution for
            # the other interactions.

        # Enforce anti-Hermitian symmetry and calculate Pr.
        with profiler.profile_range(
            label="PCoulombScreening: Symmetrization", level="default", comm=comm
        ):
            if not p_lesser.symmetry:
                p_lesser.symmetrize(xp.subtract)
                p_greater.symmetrize(xp.subtract)
                p_retarded.symmetrize(xp.add)

            if not self.compute_hilbert_retarded:
                p_retarded.data[:] = 0

            # Discard the real part of lesser/greater and imag part of retarded
            if self.use_approximation:
                p_lesser.data.real = 0
                p_greater.data.real = 0
                p_retarded.data.imag = 0

            p_retarded.data += (p_greater.data - p_lesser.data) / 2
