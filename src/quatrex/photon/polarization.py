# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from scipy.constants import (
    angstrom,
    electron_mass,
    elementary_charge,
    epsilon_0,
    hbar,
    speed_of_light,
)

from qttools import NDArray, sparse, xp
from qttools.comm import comm as qttools_comm
from qttools.datastructures import DSDBSparse
from qttools.datastructures.routines import bd_matmul_distr, bd_sandwich_distr
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import free_mempool, synchronize_device
from qttools.utils.mpi_utils import distributed_load, get_section_sizes

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.electron import ElectronSolver

profiler = Profiler()

if xp.__name__ == "cupy":
    cache = xp.fft.config.get_plan_cache()


@profiler.profile(level="api")
def fft_convolve(a: NDArray, b: NDArray) -> NDArray:
    """Computes the convolution of two arrays using FFT.

    Parameters
    ----------
    a : NDArray
        First array.
    b : NDArray
        Second array.

    Returns
    -------
    NDArray
        The convolution of the two arrays.

    """
    n = a.shape[0] + b.shape[0] - 1
    a_fft = xp.fft.fft(a, n, axis=0)
    b_fft = xp.fft.fft(b, n, axis=0)
    return xp.fft.ifft(a_fft * b_fft, axis=0)


@profiler.profile(level="api")
def fft_correlate(a: NDArray, b: NDArray) -> NDArray:
    """Computes the correlation of two arrays using FFT.

    Computes: c[E] = ∫ a(E') b(E'-E) dE' = sum_{E_diff} a(E+E_diff) * b(E_diff)
    
    This is implemented as:
    1. Reverse b in the first dimension (energy)
    2. Compute FFT convolution with zero-padding to size 2*ne-1
    3. Take the last ne elements of the result

    Parameters
    ----------
    a : NDArray
        First array (energy dimension is axis 0).
    b : NDArray
        Second array (energy dimension is axis 0).

    Returns
    -------
    NDArray
        The cross-correlation of the two arrays.

    """
    n = a.shape[0] + b.shape[0] - 1
    ne = a.shape[0]
    a_fft = xp.fft.fft(a, n, axis=0)
    b_fft = xp.fft.fft(b[::-1], n, axis=0)
    corr_full = xp.fft.ifft(a_fft * b_fft, axis=0)
    return corr_full[ne-1:]  # Take last ne elements


class PiPhoton(ScatteringSelfEnergy):
    """Computes the photon polarization from electron-photon scattering.

    This implements the photon self-energy (polarization) using a 4-term formula:
    π^<(E) = ∑_{jk}[
        ∫ dE' M_{ij}·G_{jk}^<(E')·M_{kl}·G_{li}^>(E'-E) +
        ∫ dE' M_{ji}·G_{ik}^<(E')·M_{kl}·G_{lj}^>(E'-E) +
        ∫ dE' M_{ij}·G_{jl}^<(E')·M_{lk}·G_{ki}^>(E'-E) +
        ∫ dE' M_{ji}·G_{il}^<(E')·M_{lk}·G_{kj}^>(E'-E)
    ]

    This is computed as element-wise energy correlations:
    π^< = (M@G^<@M) ⊗ G^> + (G^<@M) ⊗ (G^>@M) + (M@G^<) ⊗ (M@G^>) + G^< ⊗ (M@G^>@M)

    where M is the electron-photon interaction matrix, G are the electron
    Green's functions, and ⊗ denotes element-wise energy correlation.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The Quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.
    photon_energies : NDArray
        The energies for the photon system.
    electron_energies : NDArray
        The energies for the electron system.

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        photon_energies: NDArray,
        electron_energies: NDArray,
    ) -> None:
        """Initializes the photon polarization."""
        self.compute_config = compute_config
        self.photon_energies = photon_energies
        self.electron_energies = electron_energies
        self.num_photon_energies = photon_energies.size
        self.num_electron_energies = electron_energies.size
        self.prefactor = 1j / (2 * xp.pi) * (
            self.electron_energies[1] - self.electron_energies[0]
        )

        # Load Hamiltonian and orbital positions
        hamiltonian_sparray, block_sizes = ElectronSolver.load_hamiltonian(
            quatrex_config
        )
        orbital_positions = distributed_load(quatrex_config.input_dir / "grid.npy")

        # Get polarization direction and light intensity
        polarization = xp.array(quatrex_config.photon.polarization, dtype=float)
        intensity = quatrex_config.photon.light_intensity
        self.photon_energy = quatrex_config.photon.photon_energy

        # Normalize polarization vector
        polarization = polarization / xp.linalg.norm(polarization)

        # Compute the electron-photon interaction matrix M
        m_sparray = self._compute_interaction_matrix(
            polarization, hamiltonian_sparray, orbital_positions, intensity
        )

        # Convert to distributed sparse format
        self.interaction_matrix = compute_config.dsdbsparse_type.from_sparray(
            m_sparray,
            block_sizes=block_sizes,
            global_stack_shape=(qttools_comm.stack.size,),
            symmetry=quatrex_config.scba.symmetric,
            symmetry_op=lambda a: -a.conj(),
        )

        self.big_block_sizes = None
        self.batch_size = compute_config.convolve.batch_size

    def _compute_interaction_matrix(
        self,
        polarization: NDArray,
        hamiltonian_sparray: sparse.coo_matrix,
        orbital_positions: NDArray,
        intensity: float,
    ) -> sparse.coo_matrix:
        """Computes the electron-photon interaction matrix.

        Parameters
        ----------
        polarization : NDArray
            The normalized polarization vector of the light.
        hamiltonian_sparray : sparse.coo_matrix
            The Hamiltonian matrix in sparse format.
        orbital_positions : NDArray
            The positions of the orbitals.
        intensity : float
            The light intensity.

        Returns
        -------
        sparse.coo_matrix
            The interaction matrix M_ij.

        """
        prefactor = (
            angstrom
            * 1j
            * (hbar * elementary_charge / electron_mass)
            / (2.0e0 * epsilon_0 * speed_of_light)
        )
        prefactor *= intensity / elementary_charge / self.photon_energy**2

        interaction_matrix = sparse.coo_matrix(hamiltonian_sparray)
        for s, (i, j) in enumerate(
            zip(hamiltonian_sparray.row, hamiltonian_sparray.col)
        ):
            interaction_matrix.data[s] = (
                xp.dot((orbital_positions[i] - orbital_positions[j]), polarization)
                * hamiltonian_sparray.data[s]
            )
        interaction_matrix.data *= prefactor
        return interaction_matrix

    @profiler.profile(level="api")
    def compute(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        out: tuple[DSDBSparse, ...],
    ) -> None:
        """Computes the photon polarization (self-energy).

        Implements the formula with 4 permutation terms:
        π_{il}^<(E) = ∑_{jk}[
            ∫ dE' M_{ij}·G_{jk}^<(E')·M_{kl}·G_{li}^>(E'-E) +
            ∫ dE' M_{ji}·G_{ik}^<(E')·M_{kl}·G_{lj}^>(E'-E) +
            ∫ dE' M_{ij}·G_{jl}^<(E')·M_{lk}·G_{ki}^>(E'-E) +
            ∫ dE' M_{ji}·G_{il}^<(E')·M_{lk}·G_{kj}^>(E'-E)
        ]

        Note: This is a placeholder implementation. The exact computation strategy
        for the 4-term formula needs clarification on how to efficiently compute
        the element-wise correlations with different G^> index patterns.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The lesser electron Green's function.
        g_greater : DSDBSparse
            The greater electron Green's function.
        out : tuple[DSDBSparse, ...]
            The output matrices for the polarization. The order is
            pi_lesser, pi_greater, pi_retarded.

        """
        pi_lesser, pi_greater, pi_retarded = out

        t_block_reorder_start = time.perf_counter()

        # Save the block sizes for later
        if self.big_block_sizes is None:
            self.big_block_sizes = self.interaction_matrix.block_sizes

        # Enforce that the block sizes are the same
        self.interaction_matrix.block_sizes = g_lesser.block_sizes

        synchronize_device()
        t_block_reorder_end = time.perf_counter()
        comm.Barrier()
        t_block_reorder_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    PiPhoton: block reorder: {t_block_reorder_end - t_block_reorder_start:.3f} s",
                flush=True,
            )
            print(
                f"    PiPhoton: block reorder all: {t_block_reorder_end_all - t_block_reorder_start:.3f} s",
                flush=True,
            )

        # Step 1: Ensure matrices are in stack distribution for block operations
        t_all2all_start = time.perf_counter()
        with profiler.profile_range("ensure stack distribution", level="debug"):
            for m in (
                self.interaction_matrix,
                g_lesser,
                g_greater,
            ):
                m.dtranspose() if m.distribution_state != "stack" else None

        synchronize_device()
        t_all2all_end = time.perf_counter()
        comm.Barrier()
        t_all2all_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    PiPhoton: ensure stack distribution: {t_all2all_end - t_all2all_start:.3f} s",
                flush=True,
            )
            print(
                f"    PiPhoton: ensure stack distribution all: {t_all2all_end_all - t_all2all_start:.3f} s",
                flush=True,
            )

        # Step 2: Compute intermediate products for the 4-term formula
        # The 4 einsum permutation terms can be expressed as GEMM + element-wise products:
        # Term 1: (M@G<@M)[i,l](E') ⊙ G>.T[i,l](E'-E) = (M@G<@M)[i,l](E') with G>[l,i](E'-E)
        # Term 2: (G<@M)[i,l](E') ⊙ (G>@M).T[i,l](E'-E) = (G<@M)[i,l](E') with (G>@M)[l,i](E'-E)
        # Term 3: (M@G<)[i,l](E') ⊙ (M@G>).T[i,l](E'-E) = (M@G<)[i,l](E') with (M@G>)[l,i](E'-E)
        # Term 4: (M@G>)[i,l](E') ⊙ (M@G<).T[i,l](E'-E) = (M@G>)[i,l](E') with (M@G<)[l,i](E'-E)
        # where the right-hand side shows which matrix elements to correlate
        #
        # Intermediate products needed:
        # m_gl_m = M @ G^< @ M      (for Term 1)
        # gl_m = G^< @ M            (for Term 2)
        # m_gl = M @ G^<            (for Term 3 & 4)
        # m_gg = M @ G^>            (for Term 3 & 4)
        # gg_m = G^> @ M            (for Term 2)
        t_sandwich_start = time.perf_counter()

        # Determine block range for this rank
        local_blocks, _ = get_section_sizes(
            len(self.interaction_matrix.block_sizes), qttools_comm.block.size
        )
        start_block = sum(local_blocks[: qttools_comm.block.rank])
        end_block = start_block + local_blocks[qttools_comm.block.rank]

        # Allocate matrices for intermediate products
        m_gl = self.compute_config.dsdbsparse_type.zeros_like(g_lesser)    # M @ G^<
        m_gl_m = self.compute_config.dsdbsparse_type.zeros_like(g_lesser)  # M @ G^< @ M
        gl_m = self.compute_config.dsdbsparse_type.zeros_like(g_lesser)    # G^< @ M
        m_gg = self.compute_config.dsdbsparse_type.zeros_like(g_greater)   # M @ G^>
        m_gg_m = self.compute_config.dsdbsparse_type.zeros_like(g_greater) # M @ G^> @ M
        gg_m = self.compute_config.dsdbsparse_type.zeros_like(g_greater)   # G^> @ M

        with profiler.profile_range("Intermediate products computation", level="debug"):
            # Compute M @ G^<
            bd_matmul_distr(
                self.interaction_matrix,
                g_lesser,
                out=m_gl,
                start_block=start_block,
                end_block=end_block,
                spillover_correction=False,
            )
            
            # Compute M @ G^< @ M (for Term 1)
            bd_matmul_distr(
                m_gl,
                self.interaction_matrix,
                out=m_gl_m,
                start_block=start_block,
                end_block=end_block,
                spillover_correction=False,
            )
            
            # Compute G^< @ M (for Term 2)
            bd_matmul_distr(
                g_lesser,
                self.interaction_matrix,
                out=gl_m,
                start_block=start_block,
                end_block=end_block,
                spillover_correction=False,
            )
            
            # Compute M @ G^> (for Terms 3)
            bd_matmul_distr(
                self.interaction_matrix,
                g_greater,
                out=m_gg,
                start_block=start_block,
                end_block=end_block,
                spillover_correction=False,
            )
            
            # Compute M @ G^> @ M (for Term 4)
            bd_matmul_distr(
                m_gg,
                self.interaction_matrix,
                out=m_gg_m,
                start_block=start_block,
                end_block=end_block,
                spillover_correction=False,
            )
            
            # Compute G^> @ M (for Term 2)
            bd_matmul_distr(
                g_greater,
                self.interaction_matrix,
                out=gg_m,
                start_block=start_block,
                end_block=end_block,
                spillover_correction=False,
            )

        synchronize_device()
        t_sandwich_end = time.perf_counter()
        comm.Barrier()
        t_sandwich_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    PiPhoton: Intermediate products computation: {t_sandwich_end - t_sandwich_start:.3f} s",
                flush=True,
            )
            print(
                f"    PiPhoton: Intermediate products computation all: {t_sandwich_end_all - t_sandwich_start:.3f} s",
                flush=True,
            )

        # Step 2.5: Create spatial transposes while in stack distribution
        # We need transposes for: G<, G>, M@G<, M@G>, G>@M, M@G>@M
        t_transpose_start = time.perf_counter()
        with profiler.profile_range("Spatial transpose matrices", level="debug"):
            g_lesser_T = self.compute_config.dsdbsparse_type.zeros_like(g_lesser)
            g_greater_T = self.compute_config.dsdbsparse_type.zeros_like(g_greater)
            m_gl_T = self.compute_config.dsdbsparse_type.zeros_like(m_gl)
            m_gg_T = self.compute_config.dsdbsparse_type.zeros_like(m_gg)
            gg_m_T = self.compute_config.dsdbsparse_type.zeros_like(gg_m)
            m_gg_m_T = self.compute_config.dsdbsparse_type.zeros_like(m_gg_m)
            
            # Transpose G^< and G^>
            g_lesser.transpose(out=g_lesser_T)
            g_greater.transpose(out=g_greater_T)
            
            # Transpose M@G^< and M@G^>
            m_gl.transpose(out=m_gl_T)
            m_gg.transpose(out=m_gg_T)
            
            # Transpose G^>@M
            gg_m.transpose(out=gg_m_T)
            
            # Transpose M@G>@M
            m_gg_m.transpose(out=m_gg_m_T)

        synchronize_device()
        t_transpose_end = time.perf_counter()
        comm.Barrier()
        t_transpose_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    PiPhoton: Spatial transpose: {t_transpose_end - t_transpose_start:.3f} s",
                flush=True,
            )
            print(
                f"    PiPhoton: Spatial transpose all: {t_transpose_end_all - t_transpose_start:.3f} s",
                flush=True,
            )

        # Step 3: Transpose to nnz distribution for energy correlation
        t_all2all2_start = time.perf_counter()
        with profiler.profile_range("stack->nnz transpose", level="debug"):
            for m in (
                m_gl,
                m_gl_m,
                gl_m,
                m_gg,
                m_gg_m,
                gg_m,
                g_lesser,
                g_greater,
                g_lesser_T,
                g_greater_T,
                m_gl_T,
                m_gg_T,
                gg_m_T,
                m_gg_m_T,
                pi_lesser,
                pi_greater,
                pi_retarded,
            ):
                m.dtranspose() if m.distribution_state != "nnz" else None

        synchronize_device()
        t_all2all2_end = time.perf_counter()
        comm.Barrier()
        t_all2all2_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    PiPhoton: stack->nnz transpose: {t_all2all2_end - t_all2all2_start:.3f} s",
                flush=True,
            )
            print(
                f"    PiPhoton: stack->nnz transpose all: {t_all2all2_end_all - t_all2all2_start:.3f} s",
                flush=True,
            )

        # Step 4: Perform element-wise energy correlation in nnz distribution
        # The 4 terms with corrected transposes:
        # Term 1: (M@G<@M)[i,l](E') ⊙ G>[l,i](E'-E)
        # Term 2: (M@G>@M)[i,l](E') ⊙ G<[l,i](E'-E)
        # Term 3: (M@G<)[i,l](E') ⊙ (M@G>)[l,i](E'-E)
        # Term 4: (M@G>)[i,l](E') ⊙ (M@G<)[l,i](E'-E)
        
        t_polarization_start = time.perf_counter()
        
        # Because of padding there could be no ij elements
        if g_greater.data.shape[-1] != 0:

            with profiler.profile_range("Polarization computation", level="debug"):

                if xp.__name__ == "cupy":
                    free_mempool()
                    free_memory, _ = xp.cuda.Device().mem_info
                    num_buffers = 12  # conservative estimate
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
                        self.batch_size = g_greater.data.shape[-1]

                batch_counts, _ = get_section_sizes(
                    g_greater.data.shape[-1],
                    int(np.ceil(g_greater.data.shape[-1] / self.batch_size)),
                )

                batch_displacements = np.cumsum(
                    np.concatenate(([0], np.array(batch_counts)))
                )

                # Prepare FFT arrays for correlation
                n = g_lesser.data.shape[0] + g_greater.data.shape[0] - 1
                ne = g_lesser.data.shape[0]

                # Compute all 4 terms using GEMM + spatial transpose:
                # Correlation formula: c[E] = ∫ f(E') g(E'-E) dE' = sum_{E_diff} f(E+E_diff) * g(E_diff)
                # FFT implementation: ifft(fft(f) * fft(g[::-1]))[ne-1:]
                # where g is reversed in energy and we take the last ne elements
                #
                # Term 1: (M@G<@M)[i,l](E') ⊙ G>.T[i,l](E'-E) = (M@G<@M)[i,l](E') ⊙ G>[l,i](E'-E)
                # Term 2: (G<@M)[i,l](E') ⊙ (G>@M).T[i,l](E'-E) = (G<@M)[i,l](E') ⊙ (G>@M)[l,i](E'-E)
                # Term 3: (M@G<)[i,l](E') ⊙ (M@G>).T[i,l](E'-E) = (M@G<)[i,l](E') ⊙ (M@G>)[l,i](E'-E)
                # Term 4: G<[i,l](E') ⊙ (M@G>@M).T[i,l](E'-E) = G<[i,l](E') ⊙ (M@G>@M)[l,i](E'-E)
                for start, end in zip(batch_displacements, batch_displacements[1:]):
                    batch = slice(start, end)

                    # Term 1: correlate (M@G<@M)[i,l] with G>[l,i]
                    # Note: g_greater_T already reversed in energy ([::-1]) for correlation
                    term1_fft = xp.fft.fft(m_gl_m.data[:, batch].T, n, axis=1)
                    g_T_fft = xp.fft.fft(g_greater_T.data[::-1, batch].T, n, axis=1)
                    corr1 = xp.multiply(term1_fft, g_T_fft)
                    result1_full = xp.fft.ifft(corr1, axis=1)
                    result1 = self.prefactor * result1_full[:, ne-1:]  # Take last ne elements
                    
                    # Term 2: correlate (G<@M)[i,l] with (G>@M)[l,i]
                    term2_fft = xp.fft.fft(gl_m.data[:, batch].T, n, axis=1)
                    gg_m_T_fft = xp.fft.fft(gg_m_T.data[::-1, batch].T, n, axis=1)
                    corr2 = xp.multiply(term2_fft, gg_m_T_fft)
                    result2_full = xp.fft.ifft(corr2, axis=1)
                    result2 = self.prefactor * result2_full[:, ne-1:]  # Take last ne elements
                    
                    # Term 3: correlate (M@G<)[i,l] with (M@G>)[l,i]
                    term3_fft = xp.fft.fft(m_gl.data[:, batch].T, n, axis=1)
                    m_gg_T_fft = xp.fft.fft(m_gg_T.data[::-1, batch].T, n, axis=1)
                    corr3 = xp.multiply(term3_fft, m_gg_T_fft)
                    result3_full = xp.fft.ifft(corr3, axis=1)
                    result3 = self.prefactor * result3_full[:, ne-1:]  # Take last ne elements
                    
                    # Term 4: correlate G<[i,l] with (M@G>@M)[l,i]
                    term4_fft = xp.fft.fft(g_lesser.data[:, batch].T, n, axis=1)
                    m_gg_m_T_fft = xp.fft.fft(m_gg_m_T.data[::-1, batch].T, n, axis=1)
                    corr4 = xp.multiply(term4_fft, m_gg_m_T_fft)
                    result4_full = xp.fft.ifft(corr4, axis=1)
                    result4 = self.prefactor * result4_full[:, ne-1:]  # Take last ne elements
                    
                    # Sum all 4 terms (no need to reverse - extraction from [ne-1:] gives correct order)
                    pi_lesser.data[..., batch] += (result1 + result2 + result3 + result4).T

        synchronize_device()
        t_polarization_end = time.perf_counter()
        comm.Barrier()
        t_polarization_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    PiPhoton: Polarization computation: {t_polarization_end - t_polarization_start:.3f} s",
                flush=True,
            )
            print(
                f"    PiPhoton: Polarization computation all: {t_polarization_end_all - t_polarization_start:.3f} s",
                flush=True,
            )

        # Step 5: Transpose polarization matrices back to stack distribution
        t_all2all3_start = time.perf_counter()
        # Transpose the matrices to stack distribution
        with profiler.profile_range("nnz->stack transpose", level="debug"):
            for m in (pi_lesser, pi_greater, pi_retarded):
                m.dtranspose() if m.distribution_state != "stack" else None
            # Clean up intermediate matrices by discarding their data
            for m in (m_gl, m_gl_m, m_gg, m_gg_m, g_lesser_T, g_greater_T, m_gl_T, m_gg_T, m_gg_m_T):
                m.dtranspose(discard=True) if m.distribution_state != "stack" else None
            # Don't transpose electron Green's functions back - they may be needed
            # by other interactions in nnz distribution

        t_all2all3_end = time.perf_counter()
        comm.Barrier()
        t_all2all3_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    PiPhoton: nnz->stack transpose: {t_all2all3_end - t_all2all3_start:.3f} s",
                flush=True,
            )
            print(
                f"    PiPhoton: nnz->stack transpose all: {t_all2all3_end_all - t_all2all3_start:.3f} s",
                flush=True,
            )

        # Step 6: Enforce symmetries and derive π^> from π^<
        t_symmetrization_start = time.perf_counter()
        
        # Enforce spatial anti-Hermitian symmetry on π^<: π_ij = -π_ji*
        if not pi_lesser.symmetry:
            pi_lesser.symmetrize(xp.subtract)

        # Discard the real part (polarization should be purely imaginary)
        pi_lesser.data.real = 0

        # Derive π^> from energy symmetry: π^>(E) = -π^<(-E)†
        # This automatically enforces the bosonic energy symmetry relation
        pi_greater.data = -pi_lesser.data[::-1].conj()

        # Compute retarded polarization from lesser and greater
        pi_retarded.data = (pi_greater.data - pi_lesser.data) / 2

        synchronize_device()
        t_symmetrization_end = time.perf_counter()
        comm.Barrier()
        t_symmetrization_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    PiPhoton: Symmetrization time: {t_symmetrization_end - t_symmetrization_start:.3f} s",
                flush=True,
            )
            print(
                f"    PiPhoton: Symmetrization time all: {t_symmetrization_end_all - t_symmetrization_start:.3f} s",
                flush=True,
            )

        t_block_reorder2_start = time.perf_counter()
        # Recover original block sizes
        self.interaction_matrix.block_sizes = self.big_block_sizes
        synchronize_device()
        t_block_reorder2_end = time.perf_counter()
        comm.Barrier()
        t_block_reorder2_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    PiPhoton: block reorder back: {t_block_reorder2_end - t_block_reorder2_start:.3f} s",
                flush=True,
            )
            print(
                f"    PiPhoton: block reorder back all: {t_block_reorder2_end_all - t_block_reorder2_start:.3f} s",
                flush=True,
            )
