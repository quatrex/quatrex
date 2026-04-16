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
        self, iteration: int, g_lesser: DSDBSparse, g_greater: DSDBSparse, source_adaptive_points: NDArray, target_adaptive_points: NDArray, out: tuple[DSDBSparse, ...]
    ) -> None:
        """Computes the polarization.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The lesser Green's function.
        g_greater : DSDBSparse
            The greater Green's function.
        source_adaptive_points : NDArray
            The adaptive points for the source.
        target_adaptive_points : NDArray
            The adaptive points for the target.
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

                    if source_adaptive_points is not None:
                        # Interpolate g to the adaptive grid. This is needed for the FFT-based convolution.
                        k = self.config.scba.adaptive_interpolation_order
                        ne = g_lesser.data.shape[0]     # number of energy points in the original grid
                        # oversampling ratio, how much to blow up the adaptive grid in interpolation
                        r = self.config.scba.adaptive_interpolation_oversampling_ratio
                        n_fine = int(r * ne)        # r could be 0.5
                        n_conv_fine = 2*n_fine -1
                        n_conv = 2*ne - 1

                        # self.prefactor is only computed 1x on init, assuming uniform energy grid
                        adaptive_prefactor = self.prefactor / r  # because of interpolation to a finer grid, we need to divide by the oversampling ratio to keep the prefactor consistent
                        
                        # self.energies is the shifted energy grid to start from 0eV, directly use the quatrex config to create the min/max
                        energy_min = self.config.electron.energy_window_min
                        energy_max = self.config.electron.energy_window_max

                        # liyongda (12 Mar 2026): don't use self.energies, since it's the energies shifted to start from 0eV for Coulomb screening                        
                        fine_energies = np.linspace(energy_min, energy_max, n_fine)        # interpolation grid
                        fine_energies_conv = np.linspace(energy_min-energy_max, energy_max-energy_min, n_conv_fine) # convolution grid for conv(interpolated)
                        energies_conv = np.linspace(energy_min-energy_max, energy_max-energy_min, n_conv) # convolution grid for conv(original)

                        g_greater_fine = interpolate.make_interp_spline(source_adaptive_points, g_greater.data[..., batch], axis=0, k=k)(fine_energies)
                        g_lesser_fine = interpolate.make_interp_spline(source_adaptive_points, g_lesser.data[..., batch], axis=0, k=k)(fine_energies)

                        # liyongda (25 Mar 2026): debugging with r=1 and sanity check _compute_adaptive_grid
                        #   should always return true since fine grid and source grid are the same
                        # if comm.Get_size() == 1:  # only do sanity check with 1 rank, otherwise it's batched and nnz range is different
                            # assert(xp.allclose(fine_energies, source_adaptive_points)), "Fine energies and source adaptive points should be the same for sanity check"
                            # atol=1e-5
                            # for nnz in range(g_greater_fine.shape[1]):
                            #     assert xp.allclose(
                            #         g_greater.data[:,nnz],
                            #         g_greater_fine[:,nnz],
                            #         atol=atol
                            #     ), f"Greater part interpolation failed for nnz={nnz}"
                            #     assert xp.allclose(
                            #         g_lesser.data[:,nnz],
                            #         g_lesser_fine[:,nnz],
                            #         atol=atol
                            #     ), f"Lesser part interpolation failed for nnz={nnz}"

                        # add a flip to lesser
                        greater_fft = xp.fft.fft(g_greater_fine, n_conv_fine, axis=0)
                        lesser_fft = xp.fft.fft(-g_lesser_fine[::-1].conj(), n_conv_fine, axis=0)

                        p_g_full = xp.fft.ifft(greater_fft * lesser_fft, axis=0) * adaptive_prefactor
                        # p_l_full = -p_g_full[::-1].conj()

                        # liyongda (19 Mar 2026): debugging saves, only run with 1 rank
                        # xp.save(self.config.output_dir /  f"p_g_full_adaptive_iter{iteration}.npy", p_g_full)
                        # xp.save(self.config.output_dir /  f"energies_conv_adaptive_iter{iteration}.npy", fine_energies_conv)
                        
                        # # before interpolation
                        # xp.save(self.config.output_dir /  f"source_adaptive_points_iter{iteration}.npy", source_adaptive_points)
                        # xp.save(self.config.output_dir /  f"g_greater_{iteration}.npy", g_greater.data[..., batch])
                        # xp.save(self.config.output_dir /  f"g_lesser_{iteration}.npy", g_lesser.data[..., batch])
                        
                        # # after interpolation
                        # xp.save(self.config.output_dir /  f"g_greater_fine_adaptive_iter{iteration}.npy", g_greater_fine)
                        # xp.save(self.config.output_dir /  f"g_lesser_fine_adaptive_iter{iteration}.npy", g_lesser_fine)
                        # xp.save(self.config.output_dir /  f"fine_energies_adaptive_iter{iteration}.npy", fine_energies)


                        assert(p_g_full.shape[0] == n_conv_fine)

                        # go back to original grid length (if different length)
                        # liyongda(17 Mar 2026): going to uniform grid for P, crutch solution
                        # if (len(p_g_full) != n_conv):
                        #     p_g_full = interpolate.make_interp_spline(fine_energies_conv, p_g_full, axis=0, k=k)(energies_conv)
                        # assert(len(p_g_full) == n_conv)

                        # liyongda (17 Mar 2026): map full interpolated convolution to adaptive grid
                        # find index for zero in convolution grid
                        zero_index = np.argmin(np.abs(fine_energies_conv))
                        
                        # only go ne points (not necessarily the full half)
                        # liyongda (17 Mar 2026): lesser_unifrom directly uses p_g_full without intermediate p_l_full
                        #   tested to work in `/usr/scratch/mont-fort8/yongli/document/sandbox/fft_libraries/adaptive_integration/interpolation.ipynb`
                        
                        # liyongda (25 Mar 2026): the problem is the below slicing
                        # greater_uniform = p_g_full[zero_index:zero_index+n_fine]
                        # lesser_uniform = -p_g_full[::-1][zero_index:zero_index+n_fine].conj()  # don't create extra p_l_full, just use p_g_full with the symmetry relation       
                        
                        p_l_full = -p_g_full[::-1].conj()
                        greater_uniform = p_g_full[n_fine-1:]
                        lesser_uniform = p_l_full[n_fine-1:]

                        # fine_energies_lesser = -fine_energies_conv[zero_index:zero_index+n_fine][::-1]
                        # fine_energies_greater = fine_energies_conv[zero_index:zero_index+n_fine]
                        fine_energies_p = fine_energies - fine_energies[0]  # shift to start at 0 eV

                        # interpolate the target grid
                        # target target_adaptive_points (0 to 20 eV) 
                        # fine_energies_greater (0 to 20 eV)
                        # fine_energies_lesser (-20 to 0 eV)
                        # fine_energies_greater += target_adaptive_points[0]      # map to same range as target_adaptive_points
                        # fine_energies_lesser += target_adaptive_points[-1]
                        # p_greater.data[..., batch] = interpolate.make_interp_spline(fine_energies_greater, greater_uniform, axis=0, k=k)(target_adaptive_points)
                        # p_lesser.data[..., batch] = interpolate.make_interp_spline(fine_energies_lesser, lesser_uniform, axis=0, k=k)(target_adaptive_points)
                        p_greater.data[..., batch] = interpolate.make_interp_spline(fine_energies_p, greater_uniform, axis=0, k=k)(target_adaptive_points)
                        p_lesser.data[..., batch] = interpolate.make_interp_spline(fine_energies_p, lesser_uniform, axis=0, k=k)(target_adaptive_points)

                        # liyongda (25 Mar 2026): debugging with r=1 and sanity check _compute_adaptive_grid
                        #   should always return true since fine grid and target grid are the same
                        # if comm.Get_size() == 1:  # only do sanity check with 1 rank, otherwise it's batched and nnz range is different
                        #     atol=1e-5
                        #     for nnz in range(p_greater.shape[1]):
                        #         assert xp.allclose(
                        #             p_greater.data[:,nnz],
                        #             greater_uniform[:,nnz],
                        #             atol=atol
                        #         ), f"Greater part interpolation failed for nnz={nnz}"
                        #         assert xp.allclose(
                        #             p_lesser.data[:,nnz],
                        #             lesser_uniform[:,nnz],
                        #             atol=atol
                        #         ), f"Lesser part interpolation failed for nnz={nnz}"

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

                        # liyongda (19 Mar 2026): debugging saves, only run with 1 rank
                        # energy_min = self.config.electron.energy_window_min
                        # energy_max = self.config.electron.energy_window_max
                        # ne = g_lesser.data.shape[0]
                        # energies_conv = xp.linspace(energy_min-energy_max, energy_max-energy_min, 2*ne-1)

                        # xp.save(self.config.output_dir /  f"p_g_full_iter{iteration}.npy", p_g_full)
                        # xp.save(self.config.output_dir /  f"energies_conv_iter{iteration}.npy", energies_conv)
                        # xp.save(self.config.output_dir /  f"g_greater_{iteration}.npy", g_greater.data[..., batch])
                        # xp.save(self.config.output_dir /  f"g_lesser_{iteration}.npy", g_lesser.data[..., batch])

                    # default quatrex config compute_retarded_polarization = false
                    # liyongda (04 Mar 2026) todo: convert Hilbert Transform in retarded polarizatoin to use adaptive grid
                    # liyongda (16 Mar 2026): Anders said Hilbert Transform computation is unstable. Ignoring for now. Maybe make it work in the future.
                    #   recommended I just raise an error
                    if self.compute_retarded:
                        if target_adaptive_points is not None:
                            raise NotImplementedError("Retarded polarization computation with Hilbert Transform with adaptive grid is not implemented yet. Please use a uniform grid")
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
