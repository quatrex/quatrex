# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time

import numpy as np
from qttools import NDArray, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import synchronize_device

from qttools.utils.stack_utils import scale_stack
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
        self.prefactor = -1j / (2 * xp.pi) * (self.energies[1] - self.energies[0])
        self.coulomb_matrix = coulomb_matrix.zeros_like(coulomb_matrix)
        (
            self.coulomb_matrix.dtranspose()
            if self.coulomb_matrix.distribution_state != "stack"
            else None
        )
        # Roll the k-point axes such that they agree with the k-point ordering of the Hamiltonian/Green's functions.
        number_of_kpoints = xp.array(
            [
                1 if k <= 1 else k
                for k in quatrex_config.electron.number_of_kpoints
            ]
        )
        roll_index = number_of_kpoints // 2
        # TODO: Dubbelcheck this
        self.coulomb_matrix.data[:] = xp.roll(
            coulomb_matrix.data, shift=-roll_index, axis=tuple(range(1, 1 + len(number_of_kpoints)))
        )


    @profiler.profile(level="api")
    def compute(self, spectral_function: DSDBSparse, occupancy: NDArray, intrinsic_occupancy: NDArray, out: tuple[DSDBSparse, ...]) -> None:
        """Computes the Hartree self-energy. Note that it is the free charge that is used to compute the Hartree potential,
        i.e., the intrinsic charge is subtracted from the electron density. This is because the intrinsic charge to some degree
        is already included in the Hamiltonian.

        Parameters
        ----------
        spectral_function : DSDBSparse
            The spectral function.
        occupancy : NDArray
            The occupancy of the states, for calculating the electron density.
        intrinsic_occupancy : float
            The intrinsic occupancy of the states, for calculating the intrinsic electron density.
        out : tuple[DSDBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_retarded.

        """
        free_charge_occupancy = occupancy - intrinsic_occupancy
        # Add new axis for k-points and orbitals.
        free_charge_occupancy = free_charge_occupancy.reshape(
            (-1,) + (1,) * (spectral_function.data.ndim - 1)
        )
        # TODO: Check again if we really need to transpose the matrices
        # here.
        t_all2all_start = time.perf_counter()
        (sigma_retarded,) = out
        for m in (spectral_function, sigma_retarded):
            # Should already be in stack format.
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
        if spectral_function.data.shape[-1] != 0:
            # NOTE: This sum is different than the Fock self-energy
            gl_density = self.prefactor * scale_stack(spectral_function.diagonal(), -free_charge_occupancy).sum(axis=0)
            recvbuff = xp.empty_like(gl_density)
            # Sum the density over all MPI ranks.
            # TODO: In-place all_reduce?
            comm.stack.all_reduce(recvbuff, gl_density, op="sum")
            gl_density = recvbuff
            # Should it have an energy dimension?
            hartree_potential = xp.zeros(spectral_function.shape[:-1], dtype=xp.complex128)
            # Perform the Multiplication with the Coulomb matrix \sum_j V_ij rho_j in the orbital basis.
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
                        self.coulomb_matrix.blocks[i, j][0]
                        @ gl_density[..., col_start:col_end, np.newaxis]
                    )[..., 0]
            sigma_retarded.fill_diagonal(
               hartree_potential + sigma_retarded.diagonal() 
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
