import time 
from qttools import NDArray,_DType, sparse, xp
from qttools.datastructures import DSDBSparse
#for computation of sparsity_pattern
from qttools.utils.sparse_utils import product_sparsity_pattern_dsdbsparse
from qttools.utils.mpi_utils import get_section_sizes, get_local_slice, distributed_load
from qttools.comm import comm # En theory pas necessaire, peut travailler avec empty, mais par secu prefere le garder puis l enlever une fois le code marche (maybe karma here watching)
from qttools.utils.gpu_utils import get_host, synchronize_device

#Get Open Boundary Conditions
from qttools.greens_function_solver.solver import OBCBlocks
# Mat multi
from qttools.datastructures.routines import bd_matmul_distr, bd_sandwich_distr

#Import for the Class
from quatrex.core.subsystem import SubsystemSolver
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.utils import compute_num_connected_blocks, get_periodic_superblocks, homogenize
#Import Photon related staff:
from quatrex.photon.load import IOConfig, load_distances
from quatrex.photon.utils import D0_delta_product, D0_tensor


#maps where the structure of nonzeros will land and builds a COO matrix to reprensent the pattern
def _compute_sparsity_pattern(*matrices: DSDBSparse, dtype: _DType = None) -> sparse.coo_matrix:
    """Computes the sparsity pattern of the product of several DSDBSparse matrices."""
    num_blocks = matrices[0].num_blocks
    local_blocks, _ = get_section_sizes(num_blocks, comm.block.size)
    start_block = sum(local_blocks[: comm.block.rank])
    end_block = start_block + local_blocks[comm.block.rank]
    rows, cols = product_sparsity_pattern_dsdbsparse(*matrices, start_block=start_block, end_block=end_block, spillover=True)
    shape = matrices[0].shape[-2:]
    dtype = dtype or matrices[0].dtype
    return sparse.coo_matrix((xp.ones_like(rows), (rows, cols)), shape=shape, dtype=dtype)


class PhotonSolver(SubsystemSolver):
    """Solves the dynamics of the transveral part of the field.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The quatrex simulation configuration.
    compute_config : ComputeConfig
        The compute configuration.
    energies : NDArray
        The energies at which to solve.

    Outputs
    -------
    out : tuple[DSDBSparse, ...]
    """

    system = "photon"

    def __init__(
        self,
        quatrex_config:QuatrexConfig,
        compute_config:ComputeConfig,
        photon_energies:NDArray,
        sparsity_pattern: sparse.coo_matrix,
        ) -> None: 
        
        """iInitializes the solver."""
        # super().__init__(quatrex_config, compute_config,photon_energies) #calls the parent class’s constructor. Should the energy be added there? 
 
        self.local_energies = get_local_slice(photon_energies,comm.stack)#Each process gets a subset — its “local” portion of the total energy grid — to work on in parallel.
        
        #LOAD the Input DATAs: For us the positions & set the size of the matrixes 
        io_cfg = IOConfig(
            input_dir=quatrex_config.input_dir,
            device = quatrex_config.device,
            example_input_dir = Path("/home/sem25h7/project2/quatrex/examples/carbon-nanotube/inputs/"),
        )
        distance_unit_cells = load_distances(io_cfg)
    
        #LOAD2: Initial photon green function D0 (is already sparse: no need to initialise a pattern?)
        self.D0_matrix = D0_tensor(distance_unit_cells, self.local_energies).astype(
                xp.complex128, copy=False,
            )

        #LOAD3: D0 @ delta_perp product
        self.D0_delta = D0_delta_product(distance_unit_cells, self.local_energies).astype(
                xp.complex128, copy=False,
            )
    
        #TODO: be sure that no self._set_block_sizes(self.block_sizes) needed : the matmulti do not change blocksize of matrix
        #Get pattern that is going to hunt us the whole code: photon_matrix akka D0 pattern
        #TODO: Naming Photon_matrix not sure the best for coherence
        self.photon_matrix = compute_config.dsdbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.float32),
            block_sizes = self.small_block_sizes,
            global_stack_shape=(comm.stack.size,),
            symmetry=quatrex_config.scba.symmetric,
            symmetry_op=xp.conj,
        )

        #Block repartition realisation and Check

        #Get Number of conneted Blocks - conditions on the blocks leads to this condition check (PB: Schwachstelle fuer mich) 
        #look that number of block have the cut as conditionned + have a number oof block check (should be int)
        self.num_connected_blocks = (
            quatrex_config.photon.num_connected_blocks
        )        
        if self.num_connected_blocks == "auto":
            self.num_connected_blocks = compute_num_connected_blocks(sparsity_pattern,self.small_block_sizes)
        if len(self.small_block_sizes) % self.num_connected_blocks !=0:
            raise ValueError(f"Number of blocks must be divisible by {self.num_connected_blocks}")
        self.block_sizes = (
            self.small_block_sizes[
                : len(self.small_block_sizes) // self.num_connected_blocks
            ]
            * self.num_connected_blocks
        )
        self.small_block_sizes = (self.small_block_sizes[:len(self.small_block_sizes)//self.num_connected_blocks]*self.num_connected_blocks)

        # Check that the provided block sizes match the photon_matrix.
        if self.small_block_sizes.sum() != self.photon_matrix.shape[-2]:
            raise ValueError(
                "Block sizes do not match photon matrix. "
                f"{self.small_block_sizes.sum()} != {self.photon_matrix.shape[-2]}"
            )

        #Memory Allocation
        # Allocate memory for the system_matrix M: M Pattern is the pattern of D_0 times Polarizationn (I - D0 @ P).
        M_sparsity_pattern = _compute_sparsity_pattern(
            self.photon_matrix, 
            self.photon_matrix, 
            dtype=xp.float32,
        )

        self.system_matrix = compute_config.dsdbsparse_type.from_sparray(M_sparsity_pattern.astype(xp.complex128),block_sizes=self.block_sizes,global_stack_shape=self.photon_energies.shape)
        self.system_matrix.free_data()
        # Explicitely try to free the memory for the sparsity pattern.
        del M_sparsity_pattern

        # Allocate memory for the Term B_lesser and B_greater: B = (D0@deltaT) @ PI @ (D0@deltaT)^T
        B_sparsity_pattern = _compute_sparsity_pattern(
            self.photon_matrix,
            self.photon_matrix, #TODO: check how many times this pattern is wished
            self.photon_matrix,
            dtype=xp.float32,
        )

        self.B_lesser = compute_config.dsdbsparse_type.from_sparray(
            B_sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape,
            symmetry=quatrex_config.scba.symmetric,
            symmetry_op=lambda a: -a.conj(),
        )        
        self.B_greater = compute_config.dsdbsparse_type.zeros_like(self.B_lesser)
        # Explicitely try to free the memory for the sparsity pattern.
        del B_sparsity_pattern
        self.B_greater.free_data()
        self.B_lesser.free_data()

        # Boundary conditions.
        # Optional: Numerical hilfe for convergence
        self.dos_peak_limit = quatrex_config.photon.dos_peak_limit
        self.obc_blocks = OBCBlocks(num_blocks=self.system_matrix.num_local_blocks)
        self.block_sections = quatrex_config.photon.obc.block_sections
        self.solve_call_count = 0
        self.filtering_iteration_limit = (quatrex_config.photon.filtering_iteration_limit)

    def _set_block_sizes(self, block_sizes: NDArray) -> None:
        """Sets the block sizes of all matrices.

        Parameters
        ----------
        block_sizes : NDArray
            The new block sizes.

        """
        self.system_matrix.block_sizes = block_sizes
        self.B_lesser.block_sizes = block_sizes
        self.B_greater.block_sizes = block_sizes

    def _compute_obc(self) -> None:
        """Computes open boundary conditions."""
        #TODO: LEFT OBC: Init rank !=0
        
        # LEFT CONTACT OBC: Surface Greens function: Lyaponoc
        
        m_10, m_00, m_01 = get_periodic_superblocks(
            a_ii=self.system_matrix.blocks[0, 0],
            a_ji=self.system_matrix.blocks[1, 0],
            a_ij=self.system_matrix.blocks[0, 1],
            block_sections=self.block_sections,
        )
        x_00 = self.obc(a_ii=m_00, a_ij=m_01, a_ji=m_10, contact="left")
        m_10_x_00 = m_10 @ x_00
        self.obc_blocks.retarded[0] = m_10_x_00 @ m_01

        t_lyapunov_start = time.perf_counter()
        # Compute and apply the left lesser/greater boundary self-energy.
        a_00_lesser = m_10_x_00 @ self.B_lesser.blocks[0, 1]
        a_00_greater = m_10_x_00 @ self.B_greater.blocks[0, 1]

        q_00_lesser = (x_00@ (self.B_lesser.blocks[0, 0] - (a_00_lesser - a_00_lesser.conj().swapaxes(-1, -2))) @ x_00.conj().swapaxes(-1, -2))
        q_00_greater = (x_00@ (self.B_greater.blocks[0, 0]- (a_00_greater - a_00_greater.conj().swapaxes(-1, -2)))@ x_00.conj().swapaxes(-1, -2))

        b_00 = x_00 @ m_10
        q_00 = xp.stack((q_00_lesser, q_00_greater))

        d_00_lesser, d_00_greater = self.lyapunov(b_00, q_00, "left")

        self.obc_blocks.lesser[0] = m_10 @ d_00_lesser @ m_10.conj().swapaxes(-1, -2) - (a_00_lesser - a_00_lesser.conj().swapaxes(-1, -2))
        self.obc_blocks.greater[0] = m_10 @ d_00_greater @ m_10.conj().swapaxes(-1, -2) - (a_00_greater - a_00_greater.conj().swapaxes(-1, -2))

        #Lyapunov 
        synchronize_device()
        t_lyapunov_end = time.perf_counter()
        comm.stack.barrier()
        t_lyapunov_end_all = time.perf_counter()
        if comm.stack.rank == 0:
            print(f"        Lyapunov: {t_lyapunov_end-t_lyapunov_start:.3f}",flush=True)
            print(f"        Lyapunov all: {t_lyapunov_end_all-t_lyapunov_start:.3f}",flush=True)
    
        # RIGHT CONTACT OBC: Surface Greens function: Lyaponoc
        n = self.system_matrix.num_local_blocks - 1
        m = n - 1

        #METHOD: function get_periodic_superblocks is implemented in one way so when going from right to left need to do this trick of flipping and bop
        m_mn, m_nn, m_nm = get_periodic_superblocks(
            # Twist it, flip it, ...
            a_ii=xp.flip(self.system_matrix.blocks[n, n], axis=(-2, -1)),
            a_ji=xp.flip(self.system_matrix.blocks[m, n], axis=(-2, -1)),
            a_ij=xp.flip(self.system_matrix.blocks[n, m], axis=(-2, -1)),
            block_sections=self.block_sections,
        )
        # ... bop it.
        m_nn = xp.flip(m_nn, axis=(-2, -1))
        m_nm = xp.flip(m_nm, axis=(-2, -1))
        m_mn = xp.flip(m_mn, axis=(-2, -1))
        x_nn = self.obc(
            # Twist it, flip it, ...
            a_ii=xp.flip(m_nn, axis=(-2, -1)),
            a_ij=xp.flip(m_nm, axis=(-2, -1)),
            a_ji=xp.flip(m_mn, axis=(-2, -1)),
            contact="right",
        )
        # ... bop it.
        x_nn = xp.flip(x_nn, axis=(-2, -1))
        m_mn_x_nn = m_mn @ x_nn

        self.obc_blocks.retarded[-1] = m_mn_x_nn @ m_nm

        # Compute and apply the right lesser/greater boundary self-energy.
        a_nn_lesser = m_mn_x_nn @ self.B_lesser.blocks[n, m]
        a_nn_greater = m_mn_x_nn @ self.B_greater.blocks[n, m]

        q_nn_lesser = (x_nn@ (self.B_lesser.blocks[n, n]- (a_nn_lesser - a_nn_lesser.conj().swapaxes(-1, -2)))@ x_nn.conj().swapaxes(-1, -2))
        q_nn_greater = (x_nn@ (self.B_greater.blocks[n, n]- (a_nn_greater - a_nn_greater.conj().swapaxes(-1, -2)))@ x_nn.conj().swapaxes(-1, -2))

        b_nn = x_nn @ m_mn
        q_nn = xp.stack((q_nn_lesser, q_nn_greater))

        d_nn_lesser, d_nn_greater = self.lyapunov(b_nn, q_nn, "right")

        self.obc_blocks.lesser[-1] = m_mn @ d_nn_lesser @ m_mn.conj().swapaxes(
            -1, -2
        ) - (a_nn_lesser - a_nn_lesser.conj().swapaxes(-1, -2))

        self.obc_blocks.greater[-1] = m_mn @ d_nn_greater @ m_mn.conj().swapaxes(
            -1, -2
        ) - (a_nn_greater - a_nn_greater.conj().swapaxes(-1, -2))

    
    #ASSEMBLE system_matrix: M = I - D_0 @ Pi 
    def _assemble_system_matrix(self, p_retarded: DSDBSparse) -> None:
        """Assembles the system matrix."""

        self.system_matrix.data = 0.0
        local_blocks, _ = get_section_sizes(
            len(self.system_matrix.block_sizes), comm.block.size
        )
        start_block = sum(local_blocks[: comm.block.rank])
        end_block = start_block + local_blocks[comm.block.rank]

        bd_matmul_distr(
            self.D0_matrix,
            p_retarded, 
            out=self.system_matrix,
            start_block=start_block,
            end_block=end_block,
            spillover_correction=True,
        )
        xp.negative(self.system_matrix.data, out=self.system_matrix.data)
        self.system_matrix += sparse.eye(self.system_matrix.shape[-1]) #I- D0Pi?

    def solve(
        self,
        p_lesser:DSDBSparse,
        p_greater:DSDBSparse,
        p_retarded: DSDBSparse, 
        out: tuple[DSDBSparse, ...],
    ) -> None:
        """Solves for the transversal part of the field.

            Parameters
            ----------
            p_lesser : DSDBSparse
                The lesser transversal polarization.
            p_greater : DSDBSparse
                The greater transversal polarization.
            p_retarded : DSDBSparse
                The retarded transversal polarization.
            out : tuple[DSDBSparse, ...]
                The output matrices. The order is (lesser, greater,
                retarded).

            """

        t_set_blocksize_start = time.perf_counter()
    
        self.system_matrix.allocate_data()
        self.B_lesser.allocate_data()
        self.B_greater.allocate_data()
        self.D0_matrix.allocate_data()
        self.D0_delta.allocate_data()
        #Update: Change the block element sizes to match the Photon matrix o RGF Algo possible.
        self._set_block_sizes(self.small_block_sizes)
        
        synchronize_device()
        t_set_blocksize_end = time.perf_counter()
        comm.barrier()
        t_set_blocksize_end_all = time.perf_counter()
        if comm.rank == 0:
            print(f"    Set block sizes: {t_set_blocksize_end-t_set_blocksize_start:.3f}",flush=True)
            print(f"    Set block sizes all: {t_set_blocksize_end_all-t_set_blocksize_start:.3f}",flush=True)
        
            # STEP1: Assemble the system matrix (Includes matrix multiplication): M = I-Do(w)Polarization(w).
            self._assemble_system_matrix(p_retarded, self.D0_matrix) #UPDATES the values matrix system_matrix
         
            #STEP2: Get the overall intermediate term B lesser greater 
            local_blocks, _ = get_section_sizes(len(self.photon_matrix.block_sizes), comm.block.size)
            start_block = sum(local_blocks[: comm.block.rank])
            end_block = start_block + local_blocks[comm.block.rank]

            #Compute B lesser and greater :  Ddelta*Pola*Ddelta
            bd_sandwich_distr( 
                self.D0_delta,            
                p_lesser,
                out=self.B_lesser,
                start_block=start_block,
                end_block=end_block, #let them empty
                spillover_correction=True,
            )

            bd_sandwich_distr(
                self.D0_delta,            
                p_greater,
                out=self.B_greater,
                start_block=start_block,
                end_block=end_block,
                spillover_correction=True,
            )       
            # Go back to normal block sizes.
            t_set_blocksize_start = time.perf_counter()
            self._set_block_sizes(self.block_sizes)
            
            t_set_blocksize_end = time.perf_counter()
            comm.barrier()
            t_set_blocksize_end_all = time.perf_counter()
            if comm.rank == 0:
                print(f"    Set block sizes: {t_set_blocksize_end-t_set_blocksize_start:.3f}",flush=True)
                print(f"    Set block sizes all: {t_set_blocksize_end_all-t_set_blocksize_start:.3f}",flush=True)

            # Apply the OBC algorithm.
            t_obc_start = time.perf_counter()
            self._compute_obc()
            synchronize_device()
            t_obc_end = time.perf_counter()
            comm.barrier()
            t_obc_end_all = time.perf_counter()
            if comm.rank == 0:
                print(f"    OBC: {t_obc_end-t_obc_start:.3f}", flush=True)
                print(f"    OBC all: {t_obc_end_all-t_obc_start:.3f}", flush=True)

            # Solve the system via RGF Algo
            if comm.block.size > 1:
                t_solve_start = time.perf_counter()
                self.solver_dist.selected_solve(
                    a=self.system_matrix,  # M
                    sigma_lesser=self.B_lesser,  # B Lesser
                    sigma_greater=self.B_greater, # B Greater
                    obc_blocks=self.obc_blocks,
                    out=out,
                    return_retarded=False,
                )
                synchronize_device()
                t_solve_end = time.perf_counter()
                comm.barrier()
                t_solve_end_all = time.perf_counter()
                if comm.rank == 0:
                    print(f"    Solve: {t_solve_end-t_solve_start:.3f}", flush=True)
                    print(f"    Solve all: {t_solve_end_all-t_solve_start:.3f}", flush=True)

            else:
                t_solve_start = time.perf_counter()
                self.solver.selected_solve(
                    a=self.system_matrix,
                    sigma_lesser=self.B_lesser,
                    sigma_greater=self.B_greater,
                    obc_blocks=self.obc_blocks,
                    out=out,
                    return_retarded=False,
                )
                synchronize_device()
                t_solve_end = time.perf_counter()
                comm.barrier()
                t_solve_end_all = time.perf_counter()
                if comm.rank == 0:
                    print(f"    Solve: {t_solve_end-t_solve_start:.3f}", flush=True)
                    print(f"    Solve all: {t_solve_end_all-t_solve_start:.3f}", flush=True)

            #Now we have D we can say bye to the zwischen terms 
            self.system_matrix.free_data()
            self.B_lesser.free_data()
            self.B_greater.free_data()

            #Get the result
            d_lesser, d_greater, *__ = out
            if comm.stack.rank == 0:
                d_greater.data[0, :] = 0.0
                d_lesser.data[0, :] = 0.0

            self.solve_call_count += 1



if __name__ == "__main__":
    import time
    import numpy as np
    import scipy.sparse as sp
    from pathlib import Path
    from quatrex.photon.utils import make_grids

    # --- I/O & grids ------------------------------------------------------
    input_dir = Path("/home/sem25h7/project2/quatrex/examples/carbon-nanotube/inputs/")
    hamiltonian = sp.load_npz(input_dir/"hamiltonian.npz").tocsr()
    N= hamiltonian.shape[0]
    num_orbitals = 768
 
    energies, photon_energies = make_grids(E_min=-0.1, E_max=0.1, n_points=21,photon_energy_min=0.1, photon_energy_max=0.3)
    num_photon_energies = photon_energies.size
    num_electron_energies = energies.size
    nonzero_indices = xp.nonzero(hamiltonian)

    
    class DummyCfg:
            class Dev:
                construct_from_unit_cell = False
            device = Dev()
            input_dir = Path(".")   # Path object
    qc = DummyCfg()

    # Not so easy to make a dummy for that i need help
    # class DummyComputeCfg:
    #     dsdbsparse_type = DSDBSparse

    cc = object()  # not used here

    d_solver = PhotonSolver(
        quatrex_config=qc,  # Placeholder; not used in this test harness.
        compute_config=cc,  # Placeholder; not used in this test harness.
        photon_energies=photon_energies,
        sparsity_pattern=sp.coo_matrix((N, N)),  # Placeholder; not used in this test harness.
    )

    ##### Provisorisch : creation of the p_lesser and p_greater 
    rows, cols = nonzero_indices
    p_lesser = xp.zeros((num_electron_energies, num_orbitals, num_orbitals,3,3), dtype=complex)
    p_greater = xp.zeros_like(p_lesser)
    p_retarded = xp.zeros_like(p_lesser)
    random_matrix_d = (xp.random.rand(num_electron_energies, len(rows), 3, 3) +
                    1j*xp.random.rand(num_electron_energies, len(rows), 3, 3))
    for idx, (i, j) in enumerate(zip(rows, cols)):
        p_lesser[:, i, j, :, :] = random_matrix_d[:, idx, :, :]
        p_greater[:, i, j, :, :] = random_matrix_d[:, idx, :, :].conj()  # Just for testing
        p_retarded[:, i, j, :, :] = 0.5 * (p_lesser[:, i, j, :, :] - p_greater[:, i, j, :, :])  # Just for testing


    #outputs   
    D_less = xp.zeros((num_photon_energies, num_orbitals, num_orbitals, 3, 3), dtype=complex)
    D_grea = xp.zeros_like(D_less)
    D_ret  = xp.zeros_like(D_less)

    print("Starting D computation...")
    t0 = time.perf_counter()
    d_solver.solve(p_lesser, p_greater, p_retarded)
    t1 = time.perf_counter()
    print(f"D computation finished in {t1 - t0:.3f} s")
    print("shapes:", D_less.shape, D_grea.shape, D_ret.shape)