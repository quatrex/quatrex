# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
import time

import cupy as cp
import numba as nb
import numpy as np
import os
from cupyx.profiler import time_range

# from cupyx.scipy import sparse as cusparse
from qttools.comm import comm
from mpi4py.MPI import Request
from qttools import NDArray, sparse
from qttools.datastructures import DSDBCOO, DSDBSparse
from qttools.utils.gpu_utils import get_device, get_host, synchronize_current_stream, xp
# from qttools.comm import GPU_AWARE_MPI  # , distributed_load

from qttools.utils.input_utils import create_hamiltonian, cutoff_hr
from qttools.utils.mpi_utils import distributed_load, get_section_sizes

# from qttools.utils.sparse_utils import product_sparsity_pattern
# from qttools.utils.stack_utils import scale_stack
# from scipy import sparse
from serinv.algs import ddbtasinv

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig

from quatrex.exciton.response.polarization import correlate, kron_correlate

# from quatrex.core.compute_config import ComputeConfig
# from quatrex.core.quatrex_config import QuatrexConfig
# from quatrex.core.statistics import bose_einstein
# from quatrex.core.subsystem import SubsystemSolver
# from quatrex.coulomb_screening.utils import assemble_boundary_blocks

GPU_AWARE = False


@nb.njit(parallel=True, fastmath=True)
def _compute_pair_sparsity_pattern(
    row: np.ndarray, col: np.ndarray, dense: np.ndarray
) -> np.ndarray:
    """Computes the sparsity pattern for a pair-interaction matrix A(a,b,c,d) flattened
    into a COO matrix by combining first two and last two index.

    Parameters
    ----------
    sparsity : sparse.coo_matrix
        The sparsity pattern of interaction matrix.

    Returns
    -------
    NDArray
        The pair-interaction operator sparsity pattern.

    """
    nnz = row.shape[0]
    dense_pair = np.zeros((nnz, nnz), dtype=np.bool)
    for i, (a, b) in enumerate(zip(row, col)):
        for j, (c, d) in enumerate(zip(row, col)):
            dense_pair[i, j] = (dense[a, c] != 0) and (dense[b, d] != 0)
    return dense_pair


#    pair_cols = []
#    pair_rows = []
#    for i, a in enumerate(row):
#        b = col[i]
#        for j, c in enumerate(row):
#            d = col[j]
#            if (dense[a, c] != 0) and (dense[b, d] != 0):
#                #dense_pair[i,j] = True
#                pair_cols.append(i)
#                pair_rows.append(j)
#    rows, cols = xp.array(pair_rows), xp.array(pair_cols)
#    return sparse.coo_matrix((xp.ones_like(rows, dtype=xp.float32), (rows, cols)))


@time_range()
@nb.njit(parallel=True, fastmath=True)
def _get_mapping_numba(
    row: np.ndarray,
    col: np.ndarray,
    L_rows: np.ndarray,
    L_cols: np.ndarray,
    my_rank: int,
    L_nnz_section_offsets: np.ndarray,
):
    L_idx = np.zeros((row.shape[0], row.shape[1]), nb.int32) - 1
    for i in nb.prange(row.shape[0]):
        for j in nb.prange(row.shape[1]):
            ind = np.where((L_rows == row[i, j]) & (L_cols == col[i, j]))[0]
            if ind.size == 0:
                continue
            data_rank = np.where(L_nnz_section_offsets <= ind[0])[0][-1]
            if my_rank != data_rank:
                continue
            L_idx[i, j] = ind[0] - L_nnz_section_offsets[data_rank]

    return L_idx


_get_mapping_kernel = cp.RawKernel(
    r"""
extern "C" __global__
void get_mapping(
    const int* row,
    const int* col,
    const int* L_rows,
    const int* L_cols,
    const int* L_nnz_section_offsets,
    int* L_idx,
    int row_size,
    int col_size,
    int my_rank,
    int L_rows_size,
    int num_offsets
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < row_size && j < col_size) {
        int ind = -1;
        for (int k = 0; k < L_rows_size; ++k) {
            int cond = (
                (L_rows[k] == row[i * col_size + j])
                && (L_cols[k] == col[i * col_size + j])
            );
            ind = ind * (1 - cond) + k * cond;
        }
        int data_rank = -1;
        for (int k = 0; k < num_offsets; ++k) {
            int cond = L_nnz_section_offsets[k] <= ind;
            data_rank = data_rank * (1 - cond) + k * cond;
        }
        int cond = (my_rank == data_rank);
        L_idx[i * col_size + j] = (ind - L_nnz_section_offsets[data_rank]) * cond + (cond - 1);
    }
}
""",
    "get_mapping",
)


# @time_range()
def _get_mapping_raw(
    row: NDArray,
    col: NDArray,
    L_rows: NDArray,
    L_cols: NDArray,
    my_rank: int,
    L_nnz_section_offsets: NDArray,
):
    row_size, col_size = row.shape
    L_idx = xp.full((row_size, col_size), -1, dtype=cp.int32)
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (row_size + threads_per_block[0] - 1) // threads_per_block[0],
        (col_size + threads_per_block[1] - 1) // threads_per_block[1],
    )
    _get_mapping_kernel(
        blocks_per_grid,
        threads_per_block,
        (
            row.astype(cp.int32),
            col.astype(cp.int32),
            L_rows.astype(cp.int32),
            L_cols.astype(cp.int32),
            L_nnz_section_offsets.astype(cp.int32),
            L_idx,
            row_size,
            col_size,
            my_rank,
            L_rows.shape[0],
            L_nnz_section_offsets.shape[0],
        ),
    )
    return L_idx


def batched_assign(row: NDArray, col: NDArray, DSDBCOO: DSDBCOO, dense: NDArray):
    inds = _get_mapping_raw(
        row,
        col,
        DSDBCOO.rows,
        DSDBCOO.cols,
        comm.rank,
        DSDBCOO.nnz_section_offsets,
    )
    valid = xp.where(inds != -1)

    DSDBCOO._data[xp.ix_(DSDBCOO._stack_padding_mask, inds[valid])] = dense[:, *valid]


def _impose_bta_sparsity(
    a: xp.ndarray,
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    out: xp.ndarray | None,
):
    """Impose a block tridiagonal arrowhead sparsity to a dense array."""
    bta = a.copy()
    for i in range(n_diag_blocks):
        for j in range(n_diag_blocks):
            if abs(i - j) > 1:
                bta[
                    arrowhead_blocksize
                    + diagonal_blocksize * i : arrowhead_blocksize
                    + diagonal_blocksize * (i + 1),
                    arrowhead_blocksize
                    + diagonal_blocksize * j : arrowhead_blocksize
                    + diagonal_blocksize * (j + 1),
                ] = 0
    if out is not None:
        out = bta
    else:
        return bta


@staticmethod
def _determine_rank_map(offset, ndiag: int, rows: xp.ndarray, cols: xp.ndarray):
    """
    Figure out the info to locate the needed nonzero elements (nnz) whiLn an interaction 
    range of `ndiag` of the nnz on the i-th rank.

        - get_nnz_size: number of the nnz to gether
        - get_nnz_idx: indices ...
        - get_nnz_rank: on which ranks ...

        For example, this gives all the nnz indices needed by i-th rank, which locates on the j-th rank

        mask_i_needs_from_j = np.where(get_nnz_rank[i] == j)[0]

        nnz_i_needs_from_j = get_nnz_idx[i][mask_i_needs_from_j]
    """
    num_rank = len(offset) - 1
    get_nnz_idx = []
    get_nnz_rank = []
    get_nnz_size = []
    for rank in range(num_rank):

        min_row = rows[offset[rank] : offset[rank + 1]].min() - ndiag
        max_row = rows[offset[rank] : offset[rank + 1]].max() + ndiag
        min_col = cols[offset[rank] : offset[rank + 1]].min() - ndiag
        max_col = cols[offset[rank] : offset[rank + 1]].max() + ndiag

        mask = (
            (rows >= min_row)
            & (rows <= max_row)
            & (cols >= min_col)
            & (cols <= max_col)
        )

        idx = xp.where(mask)[0]
        idx_in_rank = xp.array([xp.where(offset <= ind)[0][-1] for ind in idx])

        get_nnz = xp.where(idx_in_rank != rank)[0]
        get_nnz_idx.append(idx[get_nnz])
        get_nnz_rank.append(idx_in_rank[get_nnz])
        get_nnz_size.append(get_nnz.size)

    return get_nnz_size, get_nnz_idx, get_nnz_rank


class BSESolver:
    def solve(
            self,
            G: tuple[DSDBSparse, DSDBSparse],
            W: DSDBSparse,
            Sigma: tuple[DSDBSparse, DSDBSparse, DSDBSparse],
            ):
        self._alloc_L0(self.exciton_energies.shape[0])
        self._alloc_L0_bta()
        
        self.screened_coulomb_matrix = W
        
        if comm.rank == 0:
            print(f"Begin calc kernel of BSE",flush=True)
        self._calc_kernel_bta()

        if comm.rank == 0:
            print(f"Begin calc L0 distributed",flush=True)
        # bse._calc_L0_distributed(self.data.g_greater,self.data.g_lesser,self.electron_energies,inz_batchsize=100)      
        self._calc_L0_v1(G.g_greater,G.g_lesser,self.electron_energies)
        
        if comm.rank == 0:
            print(f"Begin permuting L0 to BTA",flush=True)
        self._permute_L0_toBTA()
        
        if comm.rank == 0:
            print(f"Begin solving L interacting",flush=True)
        P2 = self._solve_L_interacting_BTA()

        self._calc_sigma_BSE(Sigma = Sigma)

        print(f"Writing output of BSE for debug...", flush=True)
        if not os.path.exists(self.quatrex_config.output_dir):
            os.mkdir(self.quatrex_config.output_dir)
        filename = 'BSE_P2'
        xp.save(self.quatrex_config.output_dir / filename, P2)


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

    @time_range()
    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        sparsity_pattern: sparse,
        electron_energies: NDArray,
        exciton_energies: NDArray,
        ordering: str = "normal",
    ) -> None:
        self.quatrex_config = quatrex_config
        self.compute_config = compute_config        
        self.electron_energies = electron_energies
        self.exciton_energies = exciton_energies
        self.small_block_sizes = xp.array([sparsity_pattern.shape[0],])
        # The sparsity pattern of the single-particle GF.
        num_sites = sparsity_pattern.shape[0]
        self.num_sites = num_sites
        self.sparsity = sparsity_pattern.tocoo()
        self.cutoff = max(abs(self.sparsity.col - self.sparsity.row)) + 1
        if comm.rank == 0:
            print(f"  Single-particle matrix NNZ ={self.sparsity.nnz}", flush=True)
            print(f"  Single-particle matrix bandwidth ={self.cutoff}", flush=True)
        self.ordering = ordering
        # The sparsity pattern of the two-particle GF.
        self._pair_sparsity()
        self.coulomb_matrix = self._load_coulomb_matrix()

    # preprocessing the sparsity pattern and decide the block_size and
    # num_blocks in the BTA matrix
    @time_range()
    def _pair_sparsity(self):
        """Computes the sparsity pattern of pair interactions and the block-size."""
        if comm.rank == 0:
            print("  Single-particle real-space size (N) =", self.num_sites, flush=True)
            print(
                "  Two-particle real-space size (N^2) =", self.num_sites**2, flush=True
            )
        if self.ordering == "arrowhead":
            # permute the sparsity pattern to allow arrowhead ordering of the pair-interaction matrix
            coo = self.sparsity.copy()
            row = coo.row
            col = coo.col
            nnz = row.shape[0]
            keys = xp.zeros((2, nnz), dtype=int)
            keys[0] = row
            keys[1] = row != col
            perm = xp.lexsort(keys)
            row = coo.row[perm]
            col = coo.col[perm]
            coo.row = row
            coo.col = col
            self.permuted_sparsity = coo
            # construct a lookup table of reordered indices matrix
            lut = xp.zeros((self.num_sites, self.num_sites), dtype=np.int32)            
            lut[self.permuted_sparsity.row, self.permuted_sparsity.col] = range(
                len(self.permuted_sparsity.row)
            )
            if comm.rank == 0:
                print(
                    "  Begin compute arrowhead-ordering pair sparsity pattern...",
                    flush=True,
                )
            coo = _compute_pair_sparsity_pattern(
                get_host(self.permuted_sparsity.row),
                get_host(self.permuted_sparsity.col),
                get_host(lut),
            )
            coo = sparse.coo_matrix(get_device(coo))
            self.pair_sparsity_bta = coo
            self.inverse_table_bta = lut
        #
        # to store the indexing of sparsity.data in a coo-matrix as item (i,j) for fast elementwise access
        lut = xp.zeros((self.num_sites, self.num_sites), dtype=xp.int32)
        lut[self.sparsity.row, self.sparsity.col] = range(len(self.sparsity.row))
        if comm.rank == 0:
            print(
                "  Begin compute normal-ordering pair sparsity pattern...", flush=True
            )
        coo = _compute_pair_sparsity_pattern(
            get_host(self.sparsity.row), get_host(self.sparsity.col), get_host(lut)
        )
        coo = sparse.coo_matrix(get_device(coo))
        self.pair_sparsity = coo
        self.inverse_table = lut
        self.nnz = len(coo.row)
        self.size = len(self.sparsity.row)  # size of the pair-interaction matrix

        bandwidth = max(self.pair_sparsity.col - self.pair_sparsity.row) + 1
        self.pair_bandwidth = bandwidth
        self.blocksize = bandwidth
        self.num_blocks = int(np.ceil(self.size / self.blocksize))
        self.totalsize = int(self.blocksize) * int(self.num_blocks)

        if comm.rank == 0:
            print("  --- Normal ordering --- ")
            print("  compressed Two-particle matrix size =", self.totalsize, flush=True)
            print("  bandwidth=", bandwidth, flush=True)
            print("  block size=", self.blocksize, flush=True)
            print("  number of blocks=", self.num_blocks, flush=True)
            print("  nonzero elements=", self.nnz / 1e6, " Million", flush=True)
            print(
                "  nonzero ratio = ",
                self.nnz / (self.totalsize) ** 2 * 100,
                " %",
                flush=True,
            )

        if self.ordering == "arrowhead":
            self.arrow_mask = (self.pair_sparsity_bta.row > self.num_sites) & (
                self.pair_sparsity_bta.col > self.num_sites
            )
            self.arrow_bandwidth = (
                np.max(
                    self.pair_sparsity_bta.col[self.arrow_mask]
                    - self.pair_sparsity_bta.row[self.arrow_mask]
                )
                + 1
            )
            self.arrow_blocksize = self.arrow_bandwidth
            self.arrow_num_blocks = int(
                np.ceil((self.size - self.num_sites) / self.arrow_blocksize)
            )
            self.arrowsize = int(self.arrow_blocksize) * int(self.arrow_num_blocks)
            self.tipsize = self.num_sites
            self.bta_totalsize = self.arrowsize + self.tipsize

            if comm.rank == 0:
                print("  --- BTA ordering --- ")
                print(
                    "  compressed Two-particle matrix size =",
                    self.bta_totalsize,
                    flush=True,
                )
                print("  total arrow size=", self.arrowsize, flush=True)
                print("  arrow bandwidth=", self.arrow_bandwidth, flush=True)
                print("  arrow block size=", self.arrow_blocksize, flush=True)
                print("  arrow number of blocks=", self.arrow_num_blocks, flush=True)
                print("  tip block size=", self.tipsize, flush=True)
                print("  nonzero elements=", self.nnz / 1e6, " Million", flush=True)
                print(
                    "  nonzero ratio = ",
                    self.nnz / (self.bta_totalsize) ** 2 * 100,
                    " %",
                    flush=True,
                )

        return

    @time_range()
    def _alloc_L0(self, num_energy: int, dtype=np.complex128):
        ARRAY_SHAPE = (self.totalsize, self.totalsize)
        BLOCK_SIZES = np.array([int(self.blocksize)] * int(self.num_blocks))
        GLOBAL_STACK_SHAPE = (num_energy,)
        self.num_E = num_energy

        data = xp.zeros(len(self.pair_sparsity.row), dtype=dtype)
        coords = (self.pair_sparsity.row, self.pair_sparsity.col)
        coo = sparse.coo_matrix((data, coords), shape=ARRAY_SHAPE)

        self.L0_lesser = DSDBCOO.from_sparray(coo, BLOCK_SIZES, GLOBAL_STACK_SHAPE)
        self.L0_greater = DSDBCOO.from_sparray(coo, BLOCK_SIZES, GLOBAL_STACK_SHAPE)
        self.L0_r = DSDBCOO.from_sparray(coo, BLOCK_SIZES, GLOBAL_STACK_SHAPE)

        return

    @time_range()
    def _calc_L0_distributed(
        self,
        GG: DSDBCOO,
        GL: DSDBCOO,
        G_energies: NDArray,
        step_E: int = 1,
        inz_batchsize: int = 4,
    ):
        start_time = time.time()
        self.g_lesser = GL
        self.g_greater = GG
        if self.L0_lesser.distribution_state == "stack":
            self.L0_lesser.dtranspose()
            self.L0_greater.dtranspose()
            self.L0_r.dtranspose()

        if GG.distribution_state == "stack":
            GG.dtranspose()
            GL.dtranspose()

        finish_time = time.time()
        if comm.rank == 0:
            print(" dtranspose time=", finish_time - start_time, flush=True)
        start_time = finish_time

        G_nen = GG.data.shape[0]

        if xp.isnan(GG.data).any():
            raise ValueError(f"rank {comm.rank}: GG contains NaNs")
        if xp.isnan(GL.data).any():
            raise ValueError(f"rank {comm.rank}: GL contains NaNs")

        self.G_energies = G_energies
        self.prefactor = -1j / np.pi * (self.G_energies[1] - self.G_energies[0])

        # determin the rank map of GF data in nnz distribution

        get_nnz_size, get_nnz_idx, get_nnz_rank = _determine_rank_map(
            GG.nnz_section_offsets,
            self.cutoff,
            GG.rows,
            GG.cols,
        )

        gg_recbuf = [None] * comm.size
        gl_recbuf = [None] * comm.size
        gg_sendbuf = [None] * comm.size
        gl_sendbuf = [None] * comm.size

        # start sending GF data around

        synchronize_current_stream()
        send_reqs = []
        for j in reversed(range(comm.size)):
            if j == comm.rank:
                continue
            inds_rank_to_j = get_nnz_idx[j][get_nnz_rank[j] == comm.rank]
            if not inds_rank_to_j.any():
                continue

            if not GPU_AWARE:
                gg_sendbuf[j] = GG.data[
                    ..., inds_rank_to_j - GG.nnz_section_offsets[comm.rank]
                ]
                gg_sendbuf[j] = get_host(gg_sendbuf[j])
                if np.isnan(gg_sendbuf[j]).any():
                    raise ValueError(f"rank {comm.rank}: gg send buffer contains NaNs")

                send_reqs.append(comm.Isend(gg_sendbuf[j], dest=j, tag=1))

                gl_sendbuf[j] = GL.data[
                    ..., inds_rank_to_j - GL.nnz_section_offsets[comm.rank]
                ]
                gl_sendbuf[j] = get_host(gl_sendbuf[j])
                if np.isnan(gl_sendbuf[j]).any():
                    raise ValueError(f"rank {comm.rank}: gl send buffer contains NaNs")

                send_reqs.append(comm.Isend(gl_sendbuf[j], dest=j, tag=0))
            else:
                gg_sendbuf[j] = GG.data[
                    ..., inds_rank_to_j - GG.nnz_section_offsets[comm.rank]
                ]
                if np.isnan(gg_sendbuf[j]).any():
                    raise ValueError(f"rank {comm.rank}: gg send buffer contains NaNs")

                send_reqs.append(comm.Isend(gg_sendbuf[j], dest=j, tag=1))
                gl_sendbuf[j] = GL.data[
                    ..., inds_rank_to_j - GL.nnz_section_offsets[comm.rank]
                ]
                if np.isnan(gl_sendbuf[j]).any():
                    raise ValueError(f"rank {comm.rank}: gl send buffer contains NaNs")

                send_reqs.append(comm.Isend(gl_sendbuf[j], dest=j, tag=0))

        recv_reqs = []

        for i in range(comm.size):
            if i == comm.rank:
                continue
            mask_buffer = get_nnz_rank[comm.rank] == i
            if not mask_buffer.any():
                continue

            if not GPU_AWARE:
                gg_recbuf[i] = np.zeros((G_nen, int(mask_buffer.sum())), dtype=GG.dtype)
                gl_recbuf[i] = np.zeros((G_nen, int(mask_buffer.sum())), dtype=GL.dtype)
            else:
                gg_recbuf[i] = xp.zeros((G_nen, int(mask_buffer.sum())), dtype=GG.dtype)
                gl_recbuf[i] = xp.zeros((G_nen, int(mask_buffer.sum())), dtype=GL.dtype)

            print(f" Posting receive {i}-->{comm.rank}", flush=True)

            recv_reqs.append(comm.Irecv(gl_recbuf[i], source=i, tag=0))
            recv_reqs.append(comm.Irecv(gg_recbuf[i], source=i, tag=1))

        # compute terms that only require local GF data

        start_inz_g = int(GG.nnz_section_offsets[comm.rank])
        end_inz_g = int(GG.nnz_section_offsets[comm.rank + 1])

        inz = xp.arange(start_inz_g, end_inz_g)
        jnz = inz

        for iinz in range(0, len(inz), inz_batchsize):

            print(f"      batch={iinz}", flush=True)

            row = self.inverse_table[
                GG.rows[inz[iinz : iinz + inz_batchsize, None]], GG.cols[jnz[:]]
            ]
            col = self.inverse_table[
                GG.cols[inz[iinz : iinz + inz_batchsize, None]], GG.rows[jnz[:]]
            ]

            L_x_full = self.prefactor * kron_correlate(
                GG.data[:, iinz : iinz + inz_batchsize], GL.data
            )

            batched_assign(
                row,
                col,
                self.L0_greater,
                L_x_full[G_nen - 1 : G_nen - 1 + self.num_E * step_E : step_E],
            )

            L_x_full = self.prefactor * kron_correlate(
                GL.data[:, iinz : iinz + inz_batchsize], GG.data
            )

            batched_assign(
                row,
                col,
                self.L0_lesser,
                L_x_full[G_nen - 1 : G_nen - 1 + self.num_E * step_E : step_E],
            )

        # compute terms that require GF data from MPI recv buffer

        # wait for all the send-recv requests to finish

        Request.Waitall(recv_reqs)

        print(f" rank {comm.rank} receive done", flush=True)

        # after recv check the integrity of recv data

        if comm.size > 1:
            gg_recbuf = xp.concatenate(
                [xp.array(gg) for gg in gg_recbuf if gg is not None], axis=-1
            )
            gl_recbuf = xp.concatenate(
                [xp.array(gl) for gl in gl_recbuf if gl is not None], axis=-1
            )

            if xp.isnan(gl_recbuf).any():
                raise ValueError(f"rank {comm.rank}: gl buffer contains NaNs")
            if xp.isnan(gg_recbuf).any():
                raise ValueError(f"rank {comm.rank}: gg buffer contains NaNs")

        for req in recv_reqs:
            req.free()

        for req in send_reqs:
            req.free()

        if comm.size > 1:

            jnz = get_nnz_idx[comm.rank]

            for iinz in range(0, len(inz), inz_batchsize):

                print(f"      batch={iinz}", flush=True)

                # local and buf

                row = self.inverse_table[
                    GG.rows[inz[iinz : iinz + inz_batchsize, None]], GG.cols[jnz[:]]
                ]
                col = self.inverse_table[
                    GG.cols[inz[iinz : iinz + inz_batchsize, None]], GG.rows[jnz[:]]
                ]

                L_x_full = self.prefactor * kron_correlate(
                    GG.data[:, iinz : iinz + inz_batchsize], gl_recbuf
                )

                batched_assign(
                    row,
                    col,
                    self.L0_greater,
                    L_x_full[G_nen - 1 : G_nen - 1 + self.num_E * step_E : step_E],
                )

                L_x_full = self.prefactor * kron_correlate(
                    GL.data[:, iinz : iinz + inz_batchsize], gg_recbuf
                )

                batched_assign(
                    row,
                    col,
                    self.L0_lesser,
                    L_x_full[G_nen - 1 : G_nen - 1 + self.num_E * step_E : step_E],
                )

                # buf and local

                row = self.inverse_table[
                    GG.rows[jnz[:, None]], GG.cols[inz[iinz : iinz + inz_batchsize]]
                ]
                col = self.inverse_table[
                    GG.cols[jnz[:, None]], GG.rows[inz[iinz : iinz + inz_batchsize]]
                ]

                L_x_full = self.prefactor * kron_correlate(
                    gg_recbuf, GL.data[:, iinz : iinz + inz_batchsize]
                )

                batched_assign(
                    row,
                    col,
                    self.L0_greater,
                    L_x_full[G_nen - 1 : G_nen - 1 + self.num_E * step_E : step_E],
                )

                L_x_full = self.prefactor * kron_correlate(
                    gl_recbuf, GG.data[:, iinz : iinz + inz_batchsize]
                )

                batched_assign(
                    row,
                    col,
                    self.L0_lesser,
                    L_x_full[G_nen - 1 : G_nen - 1 + self.num_E * step_E : step_E],
                )

            for iinz in range(0, len(jnz), inz_batchsize):

                print(f"      batch={iinz}", flush=True)

                # buf and buf

                row = self.inverse_table[
                    GG.rows[jnz[iinz : iinz + inz_batchsize, None]], GG.cols[jnz[:]]
                ]

                col = self.inverse_table[
                    GG.cols[jnz[iinz : iinz + inz_batchsize, None]], GG.rows[jnz[:]]
                ]

                L_x_full = self.prefactor * kron_correlate(
                    gg_recbuf[:, iinz : iinz + inz_batchsize], gl_recbuf
                )

                batched_assign(
                    row,
                    col,
                    self.L0_greater,
                    L_x_full[G_nen - 1 : G_nen - 1 + self.num_E * step_E : step_E],
                )

                L_x_full = self.prefactor * kron_correlate(
                    gl_recbuf[:, iinz : iinz + inz_batchsize], gg_recbuf
                )

                batched_assign(
                    row,
                    col,
                    self.L0_lesser,
                    L_x_full[G_nen - 1 : G_nen - 1 + self.num_E * step_E : step_E],
                )

        self.L0_r.data = (self.L0_greater.data - self.L0_lesser.data) / 2

        finish_time = time.time()
        print(
            " rank ", comm.rank, "compute time=", finish_time - start_time, flush=True
        )
        start_time = finish_time

        # transpose to stack distribution
        self.L0_lesser.dtranspose()
        self.L0_greater.dtranspose()
        self.L0_r.dtranspose()

        finish_time = time.time()
        print(
            f" rank {comm.rank} waiting + dtranspose time=",
            finish_time - start_time,
            flush=True,
        )

        return

    @time_range()
    def _calc_L0_v1(
        self, GG: DSDBCOO, GL: DSDBCOO, G_energies: NDArray, step_E: int = 1
    ):

        if self.L0_lesser.distribution_state == "stack":
            self.L0_lesser.dtranspose()
            self.L0_greater.dtranspose()
            self.L0_r.dtranspose()

        nnz_section_offsets = np.hstack(
            ([0], np.cumsum(self.L0_lesser.nnz_section_sizes))
        )
        start_inz = int(nnz_section_offsets[comm.rank])
        end_inz = int(nnz_section_offsets[comm.rank + 1])
        G_nen = GG.shape[0]
        self.G_energies = G_energies
        self.prefactor = -1j / np.pi * (self.G_energies[1] - self.G_energies[0])
        self.g_lesser = GL
        self.g_greater = GG

        L_g = xp.zeros((G_nen * 2 - 1, end_inz - start_inz), dtype=GG.dtype)
        L_l = xp.zeros_like(L_g)

        for inz in range(start_inz, end_inz):
            row = self.L0_lesser.rows[inz]
            col = self.L0_lesser.cols[inz]

            i = self.sparsity.row[row]
            j = self.sparsity.col[row]
            k = self.sparsity.row[col]
            L = self.sparsity.col[col]

            L_g[:, inz - start_inz] = self.prefactor * correlate(GG[i, k], GL[L, j])
            L_l[:, inz - start_inz] = self.prefactor * correlate(GL[i, k], GG[L, j])

        self.L0_greater._data[
            xp.ix_(self.L0_greater._stack_padding_mask, range(start_inz, end_inz))
        ] = L_g[G_nen - 1 : G_nen - 1 + self.num_E * step_E : step_E, :]

        self.L0_lesser._data[
            xp.ix_(self.L0_lesser._stack_padding_mask, range(start_inz, end_inz))
        ] = L_l[G_nen - 1 : G_nen - 1 + self.num_E * step_E : step_E, :]

        # transpose to stack distribution
        self.L0_lesser.dtranspose()
        self.L0_greater.dtranspose()
        return

    def _calc_L0_less_fft(
        self, GG, GL, start_inz, end_inz, inz_batchsize: int = 4, step_E: int = 1
    ):

        num_inz = end_inz - start_inz
        G_nen = GG.shape[0]

        n = GG.shape[0] + GL.shape[0] - 1

        GL_fft = xp.fft.fftn(GL[::-1], (n,), axes=(0,))
        for iinz in range(0, num_inz, inz_batchsize):
            print(f"      batch={iinz}", flush=True)
            GG_fft = xp.fft.fftn(
                GG[:, start_inz + iinz : start_inz + iinz + inz_batchsize],
                (n,),
                axes=(0,),
            )

            for inz in range(start_inz, end_inz):
                row = self.L0_lesser.rows[inz]
                col = self.L0_lesser.cols[inz]

                i = self.sparsity.row[row]
                j = self.sparsity.col[row]
                k = self.sparsity.row[col]
                L = self.sparsity.col[col]

                L_g = self.prefactor * xp.fft.ifftn(
                    GG_fft[i, k] * GL_fft[L, j], (n,), axes=(0,)
                )
                self.L0_greater._data[self.L0_greater._stack_padding_mask, inz] = (
                    L_g[G_nen - 1 : G_nen - 1 + self.num_E * step_E : step_E]
                )
        GL_fft = None

        GG_fft = xp.fft.fftn(GG[::-1], (n,), axes=(0,))
        for iinz in range(0, num_inz, inz_batchsize):
            print(f"      batch={iinz}", flush=True)
            GL_fft = xp.fft.fftn(
                GL[:, start_inz + iinz : start_inz + iinz + inz_batchsize],
                (n,),
                axes=(0,),
            )

            for inz in range(start_inz, end_inz):
                row = self.L0_lesser.rows[inz]
                col = self.L0_lesser.cols[inz]

                i = self.sparsity.row[row]
                j = self.sparsity.col[row]
                k = self.sparsity.row[col]
                L = self.sparsity.col[col]

                L_l = self.prefactor * xp.fft.ifftn(
                    GL_fft[i, k] * GG_fft[L, j], (n,), axes=(0,)
                )
                self.L0_lesser._data[self.L0_lesser._stack_padding_mask, inz] = (
                    L_l[G_nen - 1 : G_nen - 1 + self.num_E * step_E : step_E]
                )
        GG_fft = None


    def _calc_L0_less_fft_v2(
        self, GG_data:NDArray, GL_data:NDArray, start_inz, end_inz, inz_batchsize: int = 4, step_E: int = 1
    ):

        num_inz = end_inz - start_inz
        G_nen = GG_data.shape[0]

        n = GG_data.shape[0] + GL_data.shape[0] - 1

        GL_fft = xp.fft.fftn(GL_data[::-1], (n,), axes=(0,))
        sparsity_csr = self.sparsity.tocsr()

        for iinz in range(0, num_inz, inz_batchsize):
            print(f"      batch={iinz}", flush=True)
            GG_fft = xp.fft.fftn(
                GG_data[:, start_inz + iinz : start_inz + iinz + inz_batchsize],
                (n,),
                axes=(0,),
            )

            for jjnz in range(start_inz, end_inz):

                i = self.sparsity.row[iinz + start_inz]
                j = self.sparsity.col[iinz + start_inz]
                k = self.sparsity.row[jjnz]
                L = self.sparsity.col[jjnz]

                ind = np.where((self.L0_greater.rows == iinz) & (self.L0_greater.cols == jjnz))[0]
                if ind.size == 0:
                    continue
                data_rank = np.where(self.L0_greater.section_offsets <= ind[0])[0][-1]
                if comm.rank != data_rank:
                    continue
                idx = ind[0] - self.L0_greater.section_offsets[data_rank]

                if sparsity_csr[i,k] != -1 and sparsity_csr[L,j] != -1:

                    L_g = self.prefactor * xp.fft.ifftn(
                        GG_fft[:, self.inverse_table[i, k]] * GL_fft[:, self.inverse_table[L, j]], (n,), axes=(0,)
                    )
                    self.L0_greater._data[xp.ix_(self.L0_greater._stack_padding_mask, idx)] = (
                        L_g[G_nen - 1 : G_nen - 1 + self.num_E * step_E : step_E]
                    )
        GL_fft = None

        

    def _calc_kernel(self):
        
        self.kernel = sparse.lil_array(
            (self.totalsize, self.totalsize),
            dtype=self.L0_greater.dtype,
        )

        for i in range(self.num_sites):
            for j in range(self.num_sites):
                row = int(self.inverse_table[i, i])
                col = int(self.inverse_table[j, j])
                if V[i, j] is not None:
                    self.kernel[row, col] = -V[i, j]

        for i in range(self.num_sites):
            for j in range(self.num_sites):
                row = int(self.inverse_table[i, j])
                col = int(self.inverse_table[i, j])
                if W[i, j] is not None:
                    # print(W[i,j])
                    self.kernel[row, col] += W[i, j][1]

        self.kernel *= 1j
        self.kernel = self.kernel.tocoo()

    @time_range()
    def _calc_kernel_bta(self):
        
        kernel_tip = xp.zeros(
            (self.tipsize, self.tipsize), dtype=self.L0_lesser.dtype
        )
        kernel_diag = xp.zeros(
            (self.bta_totalsize - self.tipsize), dtype=self.L0_lesser.dtype
        )

        for i in range(self.num_sites):
            for j in range(self.num_sites):
                row = int(self.inverse_table_bta[i, i])
                col = int(self.inverse_table_bta[j, j])
                kernel_tip[row, col] = -V[i, j]
                if row == col:
                    kernel_tip[row, col] += W[i, j][1]

        for row in range(self.tipsize, self.size):
            i = self.permuted_sparsity.row[row]
            j = self.permuted_sparsity.col[row]
            kernel_diag[row - self.tipsize] += W[i, j][1]

        kernel_diag *= 1j
        kernel_tip *= 1j

        self.kernel_bta = sparse.lil_array(
            (self.bta_totalsize, self.bta_totalsize),
            dtype=self.L0_greater.dtype,
        )
        for i in range(self.num_sites):
            for j in range(self.num_sites):
                self.kernel_bta[i, j] = kernel_tip[i, j]
        for i in range(self.num_sites, self.bta_totalsize):
            self.kernel_bta[i, i] = kernel_diag[i - self.num_sites]
        self.kernel_bta = self.kernel_bta.tocoo()

    @time_range()
    def _alloc_L0_bta(self, dtype=xp.complex128):
        ARRAY_SHAPE = (self.bta_totalsize, self.bta_totalsize)
        BLOCK_SIZES = np.array(
            [int(self.tipsize)]
            + [int(self.arrow_blocksize)] * int(self.arrow_num_blocks)
        )
        GLOBAL_STACK_SHAPE = (self.num_E,)

        data = xp.zeros(len(self.pair_sparsity_bta.row), dtype=self.L0_greater.dtype)
        coords = (self.pair_sparsity_bta.row, self.pair_sparsity_bta.col)
        coo = sparse.coo_matrix((data, coords), shape=ARRAY_SHAPE)

        self.L0_lesser_bta = DSDBCOO.from_sparray(coo, BLOCK_SIZES, GLOBAL_STACK_SHAPE)
        self.L0_greater_bta = DSDBCOO.from_sparray(
            coo, BLOCK_SIZES, GLOBAL_STACK_SHAPE
        )
        self.L0_r_bta = DSDBCOO.from_sparray(coo, BLOCK_SIZES, GLOBAL_STACK_SHAPE)

        return

    @time_range()
    def _permute_L0_toBTA(self):
        if self.L0_lesser.distribution_state != "stack":
            self.L0_lesser.dtranspose()
            self.L0_greater.dtranspose()
            self.L0_r.dtranspose()
        if self.L0_lesser_bta.distribution_state != "stack":
            self.L0_lesser_bta.dtranspose()
            self.L0_greater_bta.dtranspose()
            self.L0_r_bta.dtranspose()

        # reorder L0 matrix to BTA shape
        start_time = time.time()

        # compute the row and col in the BTA ordering

        perm_rows = self.inverse_table_bta[
            self.sparsity.row[self.L0_lesser.rows],
            self.sparsity.col[self.L0_lesser.rows],
        ]
        perm_cols = self.inverse_table_bta[
            self.sparsity.row[self.L0_lesser.cols],
            self.sparsity.col[self.L0_lesser.cols],
        ]

        self.L0_lesser_bta[perm_rows, perm_cols] = self.L0_lesser[
            self.L0_lesser.rows, self.L0_lesser.cols
        ]
        self.L0_greater_bta[perm_rows, perm_cols] = self.L0_greater[
            self.L0_greater.rows, self.L0_greater.cols
        ]
        self.L0_r_bta[perm_rows, perm_cols] = self.L0_r[
            self.L0_r.rows, self.L0_r.cols
        ]

        finish_time = time.time()
        print(
            " rank ",
            comm.rank,
            "reorder to BTA matrix time=",
            finish_time - start_time,
            flush=True,
        )
        start_time = finish_time
        return

    @time_range()
    def _densesolve_L_interacting_bta(self, return_P3=False, return_L=False):
        """mostly only for debugging purpose as reference solution"""
        if self.L0_r_bta.distribution_state != "stack":
            self.L0_r_bta.dtranspose()

        (kernel_tip, kernel_diag) = self.kernel_bta

        local_nen = self.L0_r_bta.stack_shape[0]
        K = xp.zeros((self.size, self.size), dtype=self.L0_lesser_bta.dtype)

        if return_L:
            L = xp.zeros(
                (local_nen, self.size, self.size),
                dtype=self.L0_lesser_bta.dtype,
            )
        if return_P3:
            P3 = xp.zeros(
                (local_nen, self.tipsize, self.tipsize, self.tipsize),
                dtype=self.L0_lesser_bta.dtype,
            )
        else:
            P2 = xp.zeros(
                (local_nen, self.tipsize, self.tipsize),
                dtype=self.L0_lesser_bta.dtype,
            )

        K[: self.tipsize, : self.tipsize] = kernel_tip
        K[self.tipsize :, self.tipsize :] = xp.diag(
            kernel_diag[: self.size - self.tipsize]
        )

        with time_range("dense solve", color_id=comm.rank):
            for ie in range(local_nen):
                print("rank=", comm.rank, "ie=", ie + 1, "/", local_nen, flush=True)

                data = self.L0_r_bta.data[ie]

                coords = (self.L0_r_bta.rows, self.L0_r_bta.cols)

                L0 = sparse.coo_matrix(
                    (data, coords), shape=(self.size, self.size)
                ).todense()

                A = -L0 @ K + xp.diag(
                    xp.ones(self.size, dtype=self.L0_lesser_bta.dtype)
                )

                # OBC

                # call dense solver

                invA = xp.linalg.inv(A)

                # impose sparsity pattern of BTA, for a proper comparison with selected solver

                _impose_bta_sparsity(
                    invA, self.blocksize, self.tipsize, self.num_blocks, out=invA
                )

                # multiply RHS
                A = invA @ L0

                if return_L:
                    L[ie, :, :] = A

                if return_P3:
                    # 3-tensor polarization including the vertex $P3(123) = G(14) G(52) Gamma(453)$
                    for row in range(self.tipsize):
                        i = self.permuted_sparsity.row[row]
                        for col in range(self.size):
                            j = self.permuted_sparsity.row[col]
                            k = self.permuted_sparsity.col[col]
                            P3[ie, i, j, k] = A[row, col]
                else:
                    # polarization including the vertex $P2(13) = P3(113)$
                    reorder = self.permuted_sparsity.row[: self.tipsize]
                    P2[ie, reorder[:, None], reorder] = (
                        -1j * A[: self.tipsize, : self.tipsize]
                    )

        if return_L:
            return L

        if return_P3:
            return P3
        else:
            return P2

    @time_range()
    def _densesolve_L_interacting(self, return_P3=False, return_L=False):
        """mostly only for debugging purpose as reference solution"""
        if self.L0_lesser.distribution_state != "stack":
            self.L0_lesser.dtranspose()

        self.L0_r = self.L0_lesser

        K = self.kernel.todense()  # .to_dense()[0,:,:]

        local_nen = self.L0_r.stack_shape[0]

        if return_L:
            L = xp.zeros(
                (local_nen, self.size, self.size),
                dtype=self.L0_lesser_bta.dtype,
            )

        if return_P3:
            P3 = xp.zeros(
                (local_nen, self.tipsize, self.tipsize, self.tipsize),
                dtype=self.L0_lesser.dtype,
            )
        else:
            P2 = xp.zeros(
                (local_nen, self.tipsize, self.tipsize), dtype=self.L0_lesser.dtype
            )

        with time_range("dense solve", color_id=comm.rank):
            for ie in range(local_nen):
                print(" rank=", comm.rank, "ie=", ie + 1, "/", local_nen, flush=True)

                data = self.L0_r.data[ie]
                coords = (self.L0_r.rows, self.L0_r.cols)

                L0 = sparse.coo_matrix(
                    (data, coords), shape=(self.size, self.size)
                ).todense()

                A = -L0 @ K + xp.diag(xp.ones(self.size, dtype=self.L0_lesser.dtype))

                # OBC

                # call dense solver

                invA = xp.linalg.inv(A)

                # multiply RHS
                A = invA @ L0

                if return_L:
                    # compute the row and col in the BTA ordering

                    perm_rows = self.inverse_table_bta[
                        self.sparsity.row[self.L0_lesser.rows],
                        self.sparsity.col[self.L0_lesser.rows],
                    ]
                    perm_cols = self.inverse_table_bta[
                        self.sparsity.row[self.L0_lesser.cols],
                        self.sparsity.col[self.L0_lesser.cols],
                    ]

                    L[ie, perm_rows, perm_cols] = A[
                        self.L0_lesser.rows, self.L0_lesser.cols
                    ]

                if return_P3:
                    # 3-tensor polarization including the vertex $P3(123) = G(14) G(52) Gamma(453)$
                    for i in range(self.num_sites):
                        row = self.inverse_table[i, i]
                        for col in range(self.size):
                            j = self.sparsity.row[col]
                            k = self.sparsity.col[col]
                            P3[ie, i, j, k] = A[row, col]
                else:
                    # polarization including the vertex $P2(13) = P3(113)$
                    for i in range(self.num_sites):
                        row = self.inverse_table[i, i]
                        for j in range(self.num_sites):
                            col = self.inverse_table[j, j]
                            P2[ie, i, j] = -1j * A[row, col]
        if return_L:
            return L

        if return_P3:
            return P3
        else:
            return P2

    @time_range()
    def _calc_L0_retarded(self): ...

    @time_range()
    def _calc_L0_retarded_bta(self): ...

    @time_range()
    def _solve_L_interacting_BTA(
        self,
        return_P3: bool = False,
        return_L: bool = False,
    ):
        if self.L0_r_bta.distribution_state != "stack":
            self.L0_r_bta.dtranspose()

        start_time = time.time()
        (kernel_tip, kernel_diag) = self.kernel_bta

        K = xp.zeros(
            (self.bta_totalsize, self.bta_totalsize), dtype=self.L0_r_bta.dtype
        )
        K[: self.tipsize, : self.tipsize] = kernel_tip
        K[self.tipsize :, self.tipsize :] = np.diag(kernel_diag)

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

        # P3 = xp.zeros(
        #     (self.tipsize, self.blocksize * self.arrow_num_blocks, local_nen),
        #     dtype=self.L0_r_bta.dtype,
        # )

        for ie in range(local_nen):
            with time_range("construct serinv inputs", color_id=comm.rank):
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
                        xp.flip(
                            -self.L0_r_bta.stack[ie].blocks[k + 1, 0] @ kernel_tip
                        )
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
            with time_range("serinv", color_id=comm.rank):
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

            with time_range("extract P", color_id=comm.rank):
                tmp = (
                    -1j
                    * xp.transpose(xp.flip(X_arrow_tip_block_serinv))
                    @ self.L0_r_bta.stack[ie].blocks[0, 0]
                )
                for k in range(self.arrow_num_blocks):
                    tmp += (
                        -1j
                        * xp.transpose(
                            xp.flip(X_arrow_right_blocks_serinv[-k - 1, :, :])
                        )
                        @ self.L0_r_bta.stack[ie].blocks[k + 1, 0]
                    )
                # for row in range(self.tipsize):
                #     for col in range(self.tipsize):
                #         i = self.table[0, row]
                #         j = self.table[0, col]

                reorder = self.permuted_sparsity.row[: self.tipsize]
                P2[reorder[:, None], reorder, ie] = tmp[:, :]

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
        return P2
