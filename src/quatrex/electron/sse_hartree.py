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


class SigmaHartree(ScatteringSelfEnergy):
    """Computes the bare Hartree self-energy.

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
        """Initializes the bare Hartree self-energy."""
        self.energies = electron_energies
        self.kpoint_volume = np.prod(quatrex_config.electron.number_of_kpoints)
        self.prefactor = 1j / (2 * xp.pi) * (self.energies[1] - self.energies[0])
        (
            coulomb_matrix.dtranspose()
            if coulomb_matrix.distribution_state != "stack"
            else None
        )
        self.coulomb_matrix = coulomb_matrix
        # Roll the k-point axes such that they agree with the k-point ordering of the Hamiltonian/Green's functions.
        #number_of_kpoints = xp.array(
        #    [
        #        1 if k <= 1 else k
        #        for k in self.quatrex_config.electron.number_of_kpoints
        #    ]
        #)
        #roll_index = number_of_kpoints // 2
        ## TODO: Dubbelcheck this
        #self.coulomb_matrix_data = xp.roll(
        #    self.coulomb_matrix_data, shift=-roll_index, axis=tuple(range(1, 1 + len(number_of_kpoints))))
        #self.coulomb_matrix_data = xp.ascontiguousarray(self.coulomb_matrix_data)


    @profiler.profile(level="api")
    def compute(self, g_lesser: DSDBSparse, out: tuple[DSDBSparse, ...]) -> None:
        """Computes the Hartree self-energy.

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
            m.dtranspose() if m.distribution_state != "stack" else None
        synchronize_device()
        t_all2all_end = time.perf_counter()
        comm.barrier()
        t_all2all_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    SigmaHartree: stack->nnz transpose: {t_all2all_end - t_all2all_start:.3f} s",
                flush=True,
            )
            print(
                f"    SigmaHartree: stack->nnz transpose all: {t_all2all_end_all - t_all2all_start:.3f} s",
                flush=True,
            )

        # Compute the electron density by summing over energies.
        t_sse_start = time.perf_counter()
        if g_lesser.data.shape[-1] != 0:
            # NOTE: This sum is different than the Fock self-energy
            gl_density = self.prefactor * g_lesser.diagonal().sum(axis=0)
            recvbuff = xp.empty_like(gl_density)
            # Sum the density over all MPI ranks.
            # TODO: Inplace all_reduce?
            comm.stack.all_reduce(recvbuff, gl_density, op="sum")
            gl_density = recvbuff
            # Perform the Multiplication with the Coulomb matrix \sum_j V_ij rho_j in the orbital basis.
            hartree_potential = xp.zeros(g_lesser.shape[:-1], dtype=xp.complex128)
            num_blocks = len(self.coulomb_matrix.block_sizes)
            for i in range(num_blocks):
                row_start = sum(self.coulomb_matrix.block_sizes[:i])
                row_end = row_start + self.coulomb_matrix.block_sizes[i]
                for j in range(max(i-1, 0), min(i+2, num_blocks)):
                    col_start = sum(self.coulomb_matrix.block_sizes[:j])
                    col_end = col_start + self.coulomb_matrix.block_sizes[j]
                    hartree_potential[
                        ..., row_start:row_end
                    ] += (
                        self.coulomb_matrix.blocks[i, j]
                        @ gl_density[..., col_start:col_end]
                    )
            sigma_retarded.fill_diagonal(
               hartree_potential 
            )

        synchronize_device()
        t_sse_end = time.perf_counter()
        comm.barrier()
        t_sse_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"    SigmaHartree: SSE computation: {t_sse_end - t_sse_start:.3f} s",
                flush=True,
            )
            print(
                f"    SigmaHartree: SSE computation all: {t_sse_end_all - t_sse_start:.3f} s",
                flush=True,
            )

        # NOTE: The electron Green's functions and self-energies must
        # not be transposed back to stack distribution, as they are
        # needed in nnz distribution for the other interactions.
