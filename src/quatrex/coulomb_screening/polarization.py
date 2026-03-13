# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import numpy as np
from scipy import interpolate
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
        self.config = config        # liyongda (11 Mar 2026): just pass in the whole config, makes my life easier
        self.energies = coulomb_screening_energies
        self.kpoint_volume = np.prod(config.device.kpoint_grid)     # product of array elements, Default kpoints is (1,1,1), so kpoint_volume=1
        self.ne = len(self.energies)
        self.prefactor = (
            -1j
            / (xp.pi)
            * xp.abs(self.energies[1] - self.energies[0])
            / self.kpoint_volume
        )
        self.batch_size = config.compute.convolve.batch_size

        self.discard_real_parts = config.coulomb_screening.discard_real_parts
        self.compute_retarded = config.coulomb_screening.compute_retarded_polarization

    @profiler.profile(label="PCoulombScreening", level="default", comm=comm)
    def compute(
        self, g_lesser: DSDBSparse, g_greater: DSDBSparse, adaptive_points, out: tuple[DSDBSparse, ...]
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
            if self.compute_retarded:
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

                    if adaptive_points is not None:
                        # Interpolate g to the adaptive grid. This is needed for the FFT-based convolution.
                        k = self.config.scba.adaptive_interpolation_order
                        ne = g_lesser.data.shape[0]     # number of energy points in the original grid
                        # oversampling ratio, how much to blow up the adaptive grid in interpolation
                        r = self.config.scba.adaptive_interpolation_oversampling_ratio
                        n_fine = int(r * ne)        # r could be 0.5
                        n_conv_fine = 2*n_fine -1
                        n_conv = 2*ne - 1

                        energy_min = self.config.electron.energy_window_min
                        energy_max = self.config.electron.energy_window_max

                        # liyongda (12 Mar 2026): don't use self.energies, since it's the energies shifted to start from 0eV for Coulomb screening                        
                        fine_energies = np.linspace(energy_min, energy_max, n_fine)        # interpolation grid
                        fine_energies_conv = np.linspace(energy_min-energy_max, energy_max-energy_min, n_conv_fine) # convolution grid for conv(interpolated)
                        energies_conv = np.linspace(energy_min-energy_max, energy_max-energy_min, n_conv) # convolution grid for conv(original)

                        g_greater_fine = interpolate.make_interp_spline(adaptive_points, g_greater.data[..., batch], axis=0, k=k)(fine_energies)
                        g_lesser_fine = interpolate.make_interp_spline(adaptive_points, g_lesser.data[..., batch], axis=0, k=k)(fine_energies)

                        greater_fft = xp.fft.fft(g_greater_fine, n_conv_fine, axis=0)
                        lesser_fft = xp.fft.fft(-g_lesser_fine.conj(), n_conv_fine, axis=0)

                        p_g_full = xp.fft.ifft(greater_fft * lesser_fft, axis=0)

                        # go back to original grid length (if different length)
                        if (len(p_g_full) != n_conv):
                            p_g_full = interpolate.make_interp_spline(fine_energies_conv, p_g_full, axis=0, k=k)(energies_conv)
                        assert(len(p_g_full) == n_conv)
                    else:
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

                    # default quatrex config compute_retarded_polarization = false
                    # liyongda (04 Mar 2026) todo: convert Hilbert Transform in retarded polarizatoin to use adaptive grid
                    if self.compute_retarded:
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
            if self.compute_retarded:
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

            if not self.compute_retarded:
                p_retarded.data[:] = 0

            # Discard the real part.
            if self.discard_real_parts:
                p_lesser.data.real = 0
                p_greater.data.real = 0
                p_retarded.data.imag = 0

            p_retarded.data += (p_greater.data - p_lesser.data) / 2
