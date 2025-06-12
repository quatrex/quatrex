# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time

import numpy as np
from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_host, synchronize_device
from qttools.utils.input_utils import create_hamiltonian, cutoff_hr
from qttools.utils.mpi_utils import distributed_load, get_section_sizes

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy
from quatrex.core.utils import assemble_kpoint_dsb

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
        compute_config: ComputeConfig,
        electron_energies: NDArray,
        sparsity_pattern: sparse.coo_matrix,
    ):
        """Initializes the bare Fock self-energy."""
        self.energies = electron_energies
        number_of_kpoints = quatrex_config.electron.number_of_kpoints
        self.prefactor = (
            1j
            / (2 * xp.pi * np.prod(number_of_kpoints))
            * (self.energies[1] - self.energies[0])
        )
        if quatrex_config.device.construct_from_unit_cell:
            coulomb_matrix_unit_cells = distributed_load(
                quatrex_config.input_dir / "coulomb_matrix_unit_cells.npy"
            ).astype(xp.complex128)

            section_sizes, __ = get_section_sizes(
                quatrex_config.device.number_of_supercells, comm.block.size
            )
            section_offsets = np.hstack(([0], np.cumsum(section_sizes)))
            start_block = section_offsets[comm.block.rank]
            end_block = section_offsets[comm.block.rank + 1]

            # Apply the cutoff to the Coulomb matrix.
            if quatrex_config.device.R_cutoff is not None:
                coulomb_matrix_unit_cells = cutoff_hr(
                    coulomb_matrix_unit_cells,
                    R_cutoff=quatrex_config.device.R_cutoff,
                )
            coulomb_matrix_dict = {}
            for periodic_shift in xp.ndindex(
                quatrex_config.device.cells_in_periodic_directions
            ):
                for i in range(1, -2, -2):
                    if i == -1 and not any(periodic_shift):
                        break
                    periodic_shift = tuple([i * ps for ps in periodic_shift])
                    coulomb_matrix_sparray, block_sizes = create_hamiltonian(
                        coulomb_matrix_unit_cells,
                        quatrex_config.device.number_of_supercells,
                        quatrex_config.device.transport_direction,
                        quatrex_config.device.unit_cell_per_supercell,
                        block_start=start_block,
                        block_end=end_block,
                        periodic_shift=periodic_shift,
                        return_sparse=True,
                    )
                    coulomb_matrix_dict[periodic_shift] = coulomb_matrix_sparray.astype(
                        xp.complex128
                    )
            coulomb_matrix_sparray = sum(coulomb_matrix_dict.values())
            coulomb_matrix_sparray.sum_duplicates()

            block_sizes = get_host(block_sizes)
            block_sizes = np.asarray(
                [block_sizes[0]] * quatrex_config.device.number_of_supercells
            )

        else:
            try:
                coulomb_matrix_sparray = distributed_load(
                    quatrex_config.input_dir / "coulomb_matrix.npz"
                ).astype(xp.complex128)
                coulomb_matrix_dict = None
            except FileNotFoundError:
                coulomb_matrix_dict = distributed_load(
                    quatrex_config.input_dir / "coulomb_matrix.pkl"
                )
                coulomb_matrix_sparray = sum(coulomb_matrix_dict.values())

            # Load block sizes for the coulomb matrix.
            block_sizes = get_host(
                distributed_load(quatrex_config.input_dir / "block_sizes.npy")
            )

        # Create the DSDBSparse object.
        # TODO: This is pretty wasteful memory-wise.
        # Workaround: Use comm size as global stack shape.
        coulomb_matrix = compute_config.dsdbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=(comm.stack.size,)
            + tuple([k for k in number_of_kpoints if k > 1]),
            symmetry=quatrex_config.scba.symmetric,
            symmetry_op=xp.conj,
        )
        coulomb_matrix.data = 0.0
        if coulomb_matrix_dict is not None:
            number_of_kpoints = xp.array(
                [1 if k <= 1 else k for k in number_of_kpoints]
            )
            assemble_kpoint_dsb(
                coulomb_matrix,
                coulomb_matrix_dict,
                number_of_kpoints,
                -(number_of_kpoints // 2),
            )
        else:
            coulomb_matrix += coulomb_matrix_sparray
        del coulomb_matrix_sparray
        del coulomb_matrix_dict

        # Make sure that the Coulomb matrix is Hermitian.
        coulomb_matrix.symmetrize()
        coulomb_matrix.dtranspose()
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
