# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import os
import time

import numba as nb
import numpy as np
import scipy.sparse as sps
from serinv.algs import ddbtasinv

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.profiling import Profiler, decorate_methods
from qttools.utils.gpu_utils import get_device, get_host
from qttools.utils.input_utils import create_hamiltonian, cutoff_hr
from qttools.utils.mpi_utils import distributed_load, get_local_slice, get_section_sizes
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.subsystem import SubsystemSolver
from quatrex.exciton.response.pair_interactions import (
    compute_cross_product_indices,
    compute_pair_sparsity_pattern,
)

profiler = Profiler()


@nb.njit(fastmath=True)
def _determine_block_sizes_tridiagonalblocked(row, col):
    block_size = 1
    for i, j in zip(row, col):
        if (
            j > (i // block_size) * block_size + block_size * 2
            or j < (i // block_size) * block_size - block_size * 2
        ):
            block_size = int(np.ceil(abs(j - (i // block_size) * block_size) / 2))
    return block_size


@nb.njit(fastmath=True)
def _determine_block_sizes_TBA(row, col, tip_size):
    block_size = 1
    for i, j in zip(row - tip_size, col - tip_size):
        if i >= 0 and j >= 0:
            if (
                j > (i // block_size) * block_size + block_size * 2
                or j < (i // block_size) * block_size - block_size * 2
            ):
                block_size = int(np.ceil(abs(j - (i // block_size) * block_size) / 2))
    return block_size


@decorate_methods(profiler.profile(level="api"), exclude=["solve"])
class ExcitonSolver(SubsystemSolver):
    """Solves the exciton dynamics.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The quatrex simulation configuration.
    compute_config : ComputeConfig
        The compute configuration.
    energies : np.ndarray
        The energies at which to solve.

    """

    system = "exciton"

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: NDArray,
        electron_energies: NDArray,
        sparsity_pattern: sparse.coo_matrix = None,
        dtype=xp.complex128,
    ) -> None:
        super().__init__(quatrex_config, compute_config, energies)

        self.local_energies = get_local_slice(energies, comm.stack)
        self.dtype = dtype
        self.ordering = "arrowhead"

        if comm.rank == 0:
            print(
                "=========================================================", flush=True
            )
            print(" Initializing BSE solver...", flush=True)
            print(" System info ", flush=True)
        self.quatrex_config = quatrex_config
        self.compute_config = compute_config
        self.electron_energies = electron_energies
        self.exciton_energies = energies
        self.small_block_sizes = np.array([sparsity_pattern.shape[0]])
        # The sparsity pattern of the single-particle GF.
        num_sites = sparsity_pattern.shape[0]
        self.num_sites = num_sites
        self.sparsity = sparsity_pattern.tocoo()
        self.cutoff = max(abs(self.sparsity.col - self.sparsity.row))
        if comm.rank == 0:
            print(f"  1-particle matrix NNZ ={self.sparsity.nnz}", flush=True)
        self._load_coulomb_matrix()

    def _calc_pair_sparsity(self):
        """Computes the sparsity pattern of pair interactions and the block-size."""
        if comm.rank == 0:
            print("  1-particle DoF (N) =", self.num_sites, flush=True)
            print("  2-particle DoF (N^2) =", self.num_sites**2, flush=True)
            print(
                " Computing 2-particle matrix sparsity pattern...",
                flush=True,
            )
        if self.ordering == "arrowhead":
            # permute the sparsity pattern to allow arrowhead ordering of the pair-interaction matrix
            coo = self.sparsity.copy()
            row = coo.row
            col = coo.col
            nnz = row.shape[0]
            keys = xp.zeros((2, nnz), dtype=int)
            keys[0] = get_device(row)
            keys[1] = get_device(row != col)
            perm = get_host(xp.lexsort(keys))
            row = coo.row[perm]
            col = coo.col[perm]
            coo.row = row
            coo.col = col
            self.permutation = perm
            self.permuted_sparsity = coo
            # construct a lookup table of reordered indices matrix
            lut = xp.zeros((self.num_sites, self.num_sites), dtype=xp.uint32)
            lut[self.permuted_sparsity.row, self.permuted_sparsity.col] = range(
                len(self.permuted_sparsity.row)
            )

            coo = compute_pair_sparsity_pattern(
                get_host(self.permuted_sparsity.row),
                get_host(self.permuted_sparsity.col),
                get_host(lut),
            )
            coo = sps.coo_matrix(coo)
            self.pair_sparsity_bta = sparse.coo_matrix(coo)
            self.inverse_table_bta = lut

        #
        # to store the indexing of sparsity.data in a coo-matrix as item (i,j) for fast elementwise access
        lut = xp.zeros((self.num_sites, self.num_sites), dtype=xp.uint32)
        lut[self.sparsity.row, self.sparsity.col] = range(len(self.sparsity.row))

        G_row = get_host(self.sparsity.row)
        G_col = get_host(self.sparsity.col)
        G_index = get_host(lut)
        coo = compute_pair_sparsity_pattern(G_row, G_col, G_index)
        coo = sps.coo_matrix(coo)
        coo = sparse.coo_matrix(coo)

        if comm.rank == 0:
            print(
                " Computing cross product indices...",
                flush=True,
            )
        self.ik, self.Lj = compute_cross_product_indices(
            get_host(coo.row), get_host(coo.col), G_row, G_col, G_index
        )
        if self.ordering == "arrowhead":

            print(" Computing cross product indices BTA...", flush=True)
            self.ik_bta, self.Lj_bta = compute_cross_product_indices(
                get_host(self.pair_sparsity_bta.row),
                get_host(self.pair_sparsity_bta.col),
                get_host(self.permuted_sparsity.row),
                get_host(self.permuted_sparsity.col),
                G_index,
            )

        print(" Determine BTA block sizes...", flush=True)
        self.blocksize = _determine_block_sizes_tridiagonalblocked(
            get_host(coo.row), get_host(coo.col)
        )  # block size of the matrix
        self.pair_sparsity = coo
        self.inverse_table = lut
        self.nnz = len(coo.row)
        self.size = len(self.sparsity.row)  # size of the pair-interaction matrix
        self.num_blocks = int(np.ceil(self.size / self.blocksize))
        self.totalsize = int(self.blocksize) * int(
            self.num_blocks
        )  # total size of the block matrix L with padding

        if comm.rank == 0:
            print("  +--------------- Block ordering ---------------+ ")
            print("  + compressed 2-particle matrix size =", self.totalsize, flush=True)
            print("  + block size=", self.blocksize, flush=True)
            print("  + number of blocks=", self.num_blocks, flush=True)
            print("  + nonzero elements=", self.nnz / 1e6, " Million", flush=True)
            print(
                "  + nonzero ratio = ",
                self.nnz / (self.totalsize) ** 2 * 100,
                " %",
                flush=True,
            )

        if self.ordering == "arrowhead":
            self.tipsize = self.num_sites

            self.arrow_blocksize = _determine_block_sizes_TBA(
                get_host(self.pair_sparsity_bta.row),
                get_host(self.pair_sparsity_bta.col),
                self.tipsize,
            )
            self.arrow_num_blocks = int(
                np.ceil((self.size - self.num_sites) / self.arrow_blocksize)
            )
            self.arrowsize = int(self.arrow_blocksize) * int(self.arrow_num_blocks)

            self.bta_totalsize = self.arrowsize + self.tipsize

            if comm.rank == 0:
                print("  +--------------- BTA ordering ------------------+ ")
                print(
                    "  + compressed 2-particle matrix size =",
                    self.bta_totalsize,
                    flush=True,
                )
                print("  + arrow size=", self.arrowsize, flush=True)
                # print("  + arrow bandwidth=", self.arrow_bandwidth, flush=True)
                print("  + arrow block size=", self.arrow_blocksize, flush=True)
                print("  + arrow number of blocks=", self.arrow_num_blocks, flush=True)
                print("  + tip size=", self.tipsize, flush=True)
                print("  + nonzero elements=", self.nnz / 1e6, " Million", flush=True)
                print(
                    "  + nonzero ratio = ",
                    self.nnz / (self.bta_totalsize) ** 2 * 100,
                    " %",
                    flush=True,
                )
                print(
                    "=========================================================",
                    flush=True,
                )

        return

    def _load_coulomb_matrix(self):
        # Load the Coulomb matrix.
        if self.quatrex_config.device.construct_from_unit_cell:
            coulomb_matrix_unit_cells = distributed_load(
                self.quatrex_config.input_dir / "coulomb_matrix_unit_cells.npy"
            ).astype(xp.complex128)
            # Determine the local slice of the data.
            # NOTE: This is arrow-wise partitioning.
            # TODO: Allow more options, e.g., block row-wise partitioning.
            section_sizes, __ = get_section_sizes(
                self.quatrex_config.device.number_of_supercells, comm.block.size
            )
            section_offsets = np.hstack(([0], np.cumsum(section_sizes)))
            start_block = section_offsets[comm.block.rank]
            end_block = section_offsets[comm.block.rank + 1]

            coulomb_matrix_sparray, __ = create_hamiltonian(
                cutoff_hr(
                    coulomb_matrix_unit_cells,
                    R_cutoff=self.quatrex_config.device.unit_cell_per_supercell,
                ),
                self.quatrex_config.device.number_of_supercells,
                self.quatrex_config.device.transport_direction,
                self.quatrex_config.device.unit_cell_per_supercell,
                block_start=start_block,
                block_end=end_block,
                return_sparse=True,
            )
            coulomb_matrix_sparray = coulomb_matrix_sparray.astype(xp.complex128)
            coulomb_matrix_sparray.sum_duplicates()

        else:
            coulomb_matrix_sparray = distributed_load(
                self.quatrex_config.input_dir / "coulomb_matrix.npz"
            ).astype(xp.complex128)

        self.coulomb_matrix = self.compute_config.dsdbsparse_type.from_sparray(
            coulomb_matrix_sparray,
            block_sizes=self.small_block_sizes,
            global_stack_shape=(comm.stack.size,),
            symmetry=self.quatrex_config.scba.symmetric,
            symmetry_op=xp.conj,
        )

    def _compute_obc(self) -> None:
        """Computes open boundary conditions."""
        pass

    @profiler.profile(level="basic")
    def solve(
        self,
        g_lesser: DSDBSparse,
        g_greater: DSDBSparse,
        g_retarded: DSDBSparse,
        screened_coulomb: DSDBSparse,
        out: tuple[DSDBSparse, ...],
    ):
        """Solves for the 2-particle Green's function.

        Parameters
        ----------
        g_lesser : DSDBSparse
            The 1-particle lesser Green's function.
        g_greater : DSDBSparse
            The 1-particle greater Green's function.
        g_retarded : DSDBSparse
            The 1-particle retarded Green's function.
        out : tuple[DSDBSparse, ...]
            The output matrices. The order is (lesser, greater,
            retarded).
        """
        self._alloc_L0_bta(self.exciton_energies.shape[0])

        self._calc_kernel_bta(screened_coulomb)

        self._calc_L0_less_fft_bta(g_greater, g_lesser, self.electron_energies)

        P2, P2_lesser, P2_greater = self._solve_L_interacting_BTA()

        if out is not None:
            kernel_tip = self.kernel_bta[: self.tipsize, : self.tipsize]
            reorder = self.permuted_sparsity.row[: self.tipsize]
            k = kernel_tip[reorder[:, None], reorder]

            w_lesser, w_greater = out

            if comm.rank == 0:
                print(" Computing W...", flush=True)

            for ie in range(self.exciton_energies.shape[0]):

                tmp = k @ P2_lesser[:, :, ie]
                tmp = tmp.imag * 1j
                w_lesser.data[ie, :] = tmp[w_lesser.rows, w_lesser.cols]

                tmp = k @ P2_greater[:, :, ie]
                tmp = tmp.imag * 1j
                w_greater.data[ie, :] = tmp[w_greater.rows, w_greater.cols]

        if comm.rank == 0:
            print("Writing output of BSE for debug...", flush=True)
        if not os.path.exists(self.quatrex_config.output_dir):
            os.mkdir(self.quatrex_config.output_dir)
        filename = "BSE_P2R"
        xp.save(self.quatrex_config.output_dir / filename, P2)

        filename = "BSE_P2L"
        xp.save(self.quatrex_config.output_dir / filename, P2_lesser)

        filename = "BSE_P2G"
        xp.save(self.quatrex_config.output_dir / filename, P2_greater)

        self._free_L0_bta()

        return

    def _alloc_L0_bta(self, num_energies, dtype=xp.complex128):
        """Allocates the non-interacting two-particle Green's function L0 in BTA ordering."""
        if comm.rank == 0:
            print("  Allocating L0 in BTA ordering...", flush=True)

        ARRAY_SHAPE = (self.bta_totalsize, self.bta_totalsize)
        BLOCK_SIZES = np.array(
            [int(self.tipsize)]
            + [int(self.arrow_blocksize)] * int(self.arrow_num_blocks)
        )
        GLOBAL_STACK_SHAPE = (num_energies,)
        self.num_E = num_energies

        data = xp.zeros(self.nnz, dtype=self.dtype)
        coords = (self.pair_sparsity_bta.row, self.pair_sparsity_bta.col)
        coo = sparse.coo_matrix((data, coords), shape=ARRAY_SHAPE)
        dsdbsparse_type = self.compute_config.dsdbsparse_type

        self.L0_lesser_bta = dsdbsparse_type.from_sparray(
            coo, BLOCK_SIZES, GLOBAL_STACK_SHAPE
        )
        self.L0_greater_bta = dsdbsparse_type.from_sparray(
            coo, BLOCK_SIZES, GLOBAL_STACK_SHAPE
        )
        self.L0_r_bta = dsdbsparse_type.from_sparray(
            coo, BLOCK_SIZES, GLOBAL_STACK_SHAPE
        )

        return

    def _calc_kernel_bta(self, screened_coulomb_matrix):
        if comm.rank == 0:
            print(" Calculating BSE kernel in BTA ordering...", flush=True)

        V = self.coulomb_matrix
        W = screened_coulomb_matrix

        self.kernel_bta = sps.lil_array(
            (self.size, self.size),
            dtype=self.dtype,
        )
        rows = get_host(self.inverse_table_bta[V.rows, V.rows])
        cols = get_host(self.inverse_table_bta[V.cols, V.cols])
        self.kernel_bta[rows, cols] = -get_host(V.data)

        rows = get_host(self.inverse_table_bta[W.rows, W.cols])
        cols = rows
        self.kernel_bta[rows, cols] += get_host(W.data[0])

        self.kernel_bta *= 1j
        self.kernel_bta = sparse.csr_matrix(self.kernel_bta)

    def _free_L0_bta(self):
        del self.L0_greater_bta
        del self.L0_lesser_bta
        del self.L0_r_bta
        return

    def _calc_L0_less_fft_bta(
        self,
        GG: DSDBSparse,
        GL: DSDBSparse,
        G_energies: NDArray,
        step_E: int = 1,
        batch_size: int = 10000,
    ):
        if self.L0_lesser_bta.distribution_state == "stack":
            self.L0_lesser_bta.dtranspose()
            self.L0_greater_bta.dtranspose()
            self.L0_r_bta.dtranspose()

        if GG.distribution_state == "stack":
            GG.dtranspose()

        if GL.distribution_state == "stack":
            GL.dtranspose()

        if comm.rank == 0:
            print("  Calculating L0 BTA...", flush=True)
        start_time = time.time()

        G_nen = GG.shape[0]
        prefactor = (
            -1j / np.pi * (G_energies[1] - G_energies[0])
        )  # only works for equispaced energies

        n = GG.shape[0] + GL.shape[0] - 1

        GG_fft = xp.fft.fftn(GG.data, (n,), axes=(0,))
        GL_fft = xp.fft.fftn(GL.data[::-1], (n,), axes=(0,))

        batch_counts, _ = get_section_sizes(
            self.nnz,
            int(np.ceil(self.nnz / batch_size)),
        )

        batch_displacements = np.cumsum(np.concatenate(([0], np.array(batch_counts))))

        for start, end in zip(batch_displacements, batch_displacements[1:]):
            batch = slice(start, end)

            L_t = xp.multiply(
                GG_fft[:, self.ik_bta[batch]], GL_fft[:, self.Lj_bta[batch]]
            )

            LG = prefactor * xp.fft.ifftn(L_t, axes=(0,))
            self.L0_greater_bta.data[..., batch] = LG[
                G_nen - 1 : G_nen - 1 + self.num_E * step_E : step_E
            ]

            self.L0_lesser_bta.data[..., batch] = -LG[
                -G_nen : -G_nen - self.num_E * step_E : -step_E
            ].conj()

        self.L0_r_bta.data = (
            1j * xp.imag(self.L0_greater_bta.data - self.L0_lesser_bta.data) / 2
        )

        finish_time = time.time()

        print(
            " rank ", comm.rank, "compute time=", finish_time - start_time, flush=True
        )

        # transpose to stack distribution
        self.L0_lesser_bta.dtranspose()
        self.L0_greater_bta.dtranspose()
        self.L0_r_bta.dtranspose()

        return

    def _solve_L_interacting_BTA(
        self,
        return_P3: bool = False,
        return_L: bool = False,
    ):
        if comm.rank == 0:
            print(" Solve BSE ...", flush=True)

        if self.L0_r_bta.distribution_state != "stack":
            self.L0_r_bta.dtranspose()

        start_time = time.time()

        kernel_tip = self.kernel_bta[: self.tipsize, : self.tipsize]
        kernel_diag = xp.zeros(int(self.bta_totalsize - self.tipsize), dtype=self.dtype)
        kernel_diag[: int(self.size - self.tipsize)] = self.kernel_bta[
            self.tipsize :, self.tipsize :
        ].diagonal()

        A_arrow_right_blocks = xp.zeros(
            (int(self.arrow_num_blocks), int(self.arrow_blocksize), int(self.tipsize)),
            dtype=self.L0_r_bta.dtype,
        )
        A_arrow_bottom_blocks = xp.zeros(
            (int(self.arrow_num_blocks), int(self.tipsize), int(self.arrow_blocksize)),
            dtype=self.L0_r_bta.dtype,
        )
        A_diagonal_blocks = xp.zeros(
            (
                int(self.arrow_num_blocks),
                int(self.arrow_blocksize),
                int(self.arrow_blocksize),
            ),
            dtype=self.L0_r_bta.dtype,
        )
        A_upper_diagonal_blocks = xp.zeros(
            (
                int(self.arrow_num_blocks - 1),
                int(self.arrow_blocksize),
                int(self.arrow_blocksize),
            ),
            dtype=self.L0_r_bta.dtype,
        )
        A_lower_diagonal_blocks = xp.zeros(
            (
                int(self.arrow_num_blocks - 1),
                int(self.arrow_blocksize),
                int(self.arrow_blocksize),
            ),
            dtype=self.L0_r_bta.dtype,
        )

        local_nen = self.L0_r_bta.stack_shape[0]
        P2 = xp.zeros(
            (self.tipsize, self.tipsize, local_nen), dtype=self.L0_r_bta.dtype
        )
        P2_lesser = xp.zeros(
            (self.tipsize, self.tipsize, local_nen), dtype=self.L0_lesser_bta.dtype
        )
        P2_greater = xp.zeros(
            (self.tipsize, self.tipsize, local_nen), dtype=self.L0_greater_bta.dtype
        )

        # P3 = xp.zeros(
        #     (self.tipsize, self.blocksize * self.arrow_num_blocks, local_nen),
        #     dtype=self.L0_r_bta.dtype,
        # )

        for ie in range(local_nen):

            print(" rank=", comm.rank, "ie=", ie + 1, "/", local_nen, flush=True)

            # build system matrix: A = I - L0 @ K
            # Note: SerinV takes BTA pointing down, so the block ordering should be reversed and
            #       each block matrix should be transposed and flipped.

            A_arrow_tip_block = xp.transpose(
                xp.flip(
                    -self.L0_r_bta.stack[ie].blocks[0, 0] @ kernel_tip
                    + xp.eye(self.tipsize)
                )
            )

            for k in range(self.arrow_num_blocks):
                A_diagonal_blocks[-k - 1, :, :] = xp.transpose(
                    xp.flip(
                        -self.L0_r_bta.stack[ie].blocks[k + 1, k + 1]
                        @ xp.diag(
                            kernel_diag[
                                self.arrow_blocksize
                                * k : self.arrow_blocksize
                                * (k + 1)
                            ]
                        )
                    )
                    + xp.eye(int(self.arrow_blocksize))
                )

            for k in range(self.arrow_num_blocks - 1):
                A_upper_diagonal_blocks[-k - 1, :, :] = xp.transpose(
                    xp.flip(
                        -self.L0_r_bta.stack[ie].blocks[k + 1, k + 2]
                        @ xp.diag(
                            kernel_diag[
                                self.arrow_blocksize
                                * (k + 1) : self.arrow_blocksize
                                * (k + 2)
                            ]
                        )
                    )
                )
                A_lower_diagonal_blocks[-k - 1, :, :] = xp.transpose(
                    xp.flip(
                        -self.L0_r_bta.stack[ie].blocks[k + 2, k + 1]
                        @ xp.diag(
                            kernel_diag[
                                self.arrow_blocksize
                                * (k) : self.arrow_blocksize
                                * (k + 1)
                            ]
                        )
                    )
                )

            for k in range(self.arrow_num_blocks):
                A_arrow_bottom_blocks[-k - 1, :, :] = xp.transpose(
                    xp.flip(-self.L0_r_bta.stack[ie].blocks[k + 1, 0] @ kernel_tip)
                )
                A_arrow_right_blocks[-k - 1, :, :] = xp.transpose(
                    xp.flip(
                        -self.L0_r_bta.stack[ie].blocks[0, k + 1]
                        @ xp.diag(
                            kernel_diag[
                                self.arrow_blocksize
                                * k : self.arrow_blocksize
                                * (k + 1)
                            ]
                        )
                    )
                )

            # solve system matrix

            (
                X_diagonal_blocks_serinv,
                X_lower_diagonal_blocks_serinv,
                X_upper_diagonal_blocks_serinv,
                X_arrow_bottom_blocks_serinv,
                X_arrow_right_blocks_serinv,
                X_arrow_tip_block_serinv,
            ) = ddbtasinv(
                A_diagonal_blocks,
                A_lower_diagonal_blocks,
                A_upper_diagonal_blocks,
                A_arrow_bottom_blocks,
                A_arrow_right_blocks,
                A_arrow_tip_block,
            )

            # first, we need to transpose and flip back the BTA matrix output from SerinV
            # extract P2 from the tip of BTA matrix L =  A^{-1} @ L0, and P2 := -i L_tip

            tmp = (
                -1j
                * xp.transpose(xp.flip(X_arrow_tip_block_serinv))
                @ self.L0_r_bta.stack[ie].blocks[0, 0]
            )
            for k in range(self.arrow_num_blocks):
                tmp += (
                    -1j
                    * xp.transpose(xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :]))
                    @ self.L0_r_bta.stack[ie].blocks[k + 1, 0]
                )

            reorder = self.permuted_sparsity.row[: self.tipsize]
            P2[reorder[:, None], reorder, ie] = tmp[:, :]

            # P2 lesser

            tmp = (
                -1j
                * xp.transpose(xp.flip(X_arrow_tip_block_serinv))
                @ self.L0_lesser_bta.stack[ie].blocks[0, 0]
                @ xp.transpose(xp.flip(X_arrow_tip_block_serinv.T.conj()))
            )
            for k in range(self.arrow_num_blocks):
                tmp += (
                    -1j
                    * xp.transpose(xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :]))
                    @ self.L0_lesser_bta.stack[ie].blocks[k + 1, k + 1]
                    @ xp.transpose(
                        xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :].T.conj())
                    )
                )
                if k < self.arrow_num_blocks - 1:
                    tmp += (
                        -1j
                        * xp.transpose(
                            xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :])
                        )
                        @ self.L0_lesser_bta.stack[ie].blocks[k + 1, k + 2]
                        @ xp.transpose(
                            xp.flip(X_arrow_right_blocks_serinv[-k - 2, :, :].T.conj())
                        )
                    )
                    tmp += (
                        -1j
                        * xp.transpose(
                            xp.flip(X_arrow_right_blocks_serinv[-k - 2, :, :])
                        )
                        @ self.L0_lesser_bta.stack[ie].blocks[k + 2, k + 1]
                        @ xp.transpose(
                            xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :].T.conj())
                        )
                    )

            reorder = self.permuted_sparsity.row[: self.tipsize]
            P2_lesser[reorder[:, None], reorder, ie] = tmp[:, :]

            # P2 greater

            tmp = (
                -1j
                * xp.transpose(xp.flip(X_arrow_tip_block_serinv))
                @ self.L0_greater_bta.stack[ie].blocks[0, 0]
                @ xp.transpose(xp.flip(X_arrow_tip_block_serinv.T.conj()))
            )
            for k in range(self.arrow_num_blocks):
                tmp += (
                    -1j
                    * xp.transpose(xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :]))
                    @ self.L0_greater_bta.stack[ie].blocks[k + 1, 0]
                    @ xp.transpose(xp.flip(X_arrow_tip_block_serinv.T.conj()))
                )
                tmp += (
                    -1j
                    * xp.transpose(xp.flip(X_arrow_tip_block_serinv))
                    @ self.L0_greater_bta.stack[ie].blocks[0, k + 1]
                    @ xp.transpose(
                        xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :].T.conj())
                    )
                )

                tmp += (
                    -1j
                    * xp.transpose(xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :]))
                    @ self.L0_greater_bta.stack[ie].blocks[k + 1, k + 1]
                    @ xp.transpose(
                        xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :].T.conj())
                    )
                )
                if k < self.arrow_num_blocks - 1:
                    tmp += (
                        -1j
                        * xp.transpose(
                            xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :])
                        )
                        @ self.L0_greater_bta.stack[ie].blocks[k + 1, k + 2]
                        @ xp.transpose(
                            xp.flip(X_arrow_right_blocks_serinv[-k - 2, :, :].T.conj())
                        )
                    )
                    tmp += (
                        -1j
                        * xp.transpose(
                            xp.flip(X_arrow_right_blocks_serinv[-k - 2, :, :])
                        )
                        @ self.L0_greater_bta.stack[ie].blocks[k + 2, k + 1]
                        @ xp.transpose(
                            xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :].T.conj())
                        )
                    )

            reorder = self.permuted_sparsity.row[: self.tipsize]
            P2_greater[reorder[:, None], reorder, ie] = tmp[:, :]

            # extract Gamma from upper-arrow block of L = A^{-1} @ L0, and Gamma_ijk := L_iijk
            # L_01 = A_00 @ L0_01 + A_01 @ L0_11

            # tmp2 = xp.zeros(
            #     (self.tipsize, self.blocksize, self.num_blocks), dtype=xp.complex128
            # )
            # for k in range(self.num_blocks):
            #     tmp2[:, :, k] += (
            #         xp.transpose(xp.flip(X_arrow_tip_block_serinv))
            #         @ self.L0mat.stack[ie].blocks[0, k + 1]
            #     )
            #     tmp2[:, :, k] += (
            #         xp.transpose(xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :]))
            #         @ self.L0mat.stack[ie].blocks[k + 1, k + 1]
            #     )
            #     if k > 0:
            #         tmp2[:, :, k] += (
            #             xp.transpose(
            #                 xp.flip(X_arrow_right_blocks_serinv[-(k - 1) - 1, :, :])
            #             )
            #             @ self.L0mat.stack[ie].blocks[k, k + 1]
            #         )
            #     if k < self.num_blocks - 1:
            #         tmp2[:, :, k] += (
            #             xp.transpose(
            #                 xp.flip(X_arrow_right_blocks_serinv[-(k + 1) - 1, :, :])
            #             )
            #             @ self.L0mat.stack[ie].blocks[k + 2, k + 1]
            #         )

            # i = self.table[0, :self.tipsize]
            # j = self.table[0, self.tipsize:self.size]
            # k = self.table[1, self.tipsize:self.size]
            # P3[:, :, ie] = tmp2.reshape(
            #     (tmp2.shape[0], tmp2.shape[1] * tmp2.shape[2])
            # )

            # for row in range(self.tipsize):
            #     i = self.table[0, row]
            #     for ib in range(self.num_blocks):
            #         for ic in range(self.blocksize):
            #             col = ib * self.blocksize + ic + self.tipsize

            #             if col < self.size:
            #                 j = self.table[0, col]
            #                 k = self.table[1, col]

            #             P3[self.table[0, :], j, k, ie] = tmp2[:, ic, ib]
        finish_time = time.time()
        print(
            " rank ",
            comm.rank,
            "solve time=",
            finish_time - start_time,
            flush=True,
        )
        return P2, P2_lesser, P2_greater
