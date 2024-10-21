# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
from qttools.datastructures import DSBSparse
from qttools.utils.gpu_utils import xp
from qttools.utils.mpi_utils import distributed_load
from qttools.utils.stack_utils import scale_stack
from scipy import sparse

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.statistics import fermi_dirac
from quatrex.core.subsystem import SubsystemSolver


class ElectronSolver(SubsystemSolver):
    """Solves for the lesser electron Green's function.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The quatrex simulation configuration.
    compute_config : ComputeConfig
        The compute configuration.
    energies : np.ndarray
        The energies at which to solve.

    """

    system = "electron"

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: xp.ndarray,
    ) -> None:
        """Initializes the electron solver."""
        super().__init__(quatrex_config, compute_config, energies)

        # Load the device Hamiltonian.
        self.hamiltonian_sparray = distributed_load(
            quatrex_config.input_dir / "hamiltonian.npz"
        ).astype(xp.complex128)

        self.block_sizes = distributed_load(
            quatrex_config.input_dir / "block_sizes.npy"
        )
        self.block_offsets = xp.hstack(([0], xp.cumsum(self.block_sizes)))
        # Check that the provided block sizes match the Hamiltonian.
        if self.block_sizes.sum() != self.hamiltonian_sparray.shape[0]:
            raise ValueError(
                "Block sizes do not match Hamiltonian. "
                f"{self.block_sizes.sum()} != {self.hamiltonian_sparray.shape[0]}"
            )
        # Load the overlap matrix.
        try:
            self.overlap_sparray = distributed_load(
                quatrex_config.input_dir / "overlap.npz"
            ).astype(xp.complex128)
        except FileNotFoundError:
            # No overlap provided. Assume orthonormal basis.
            self.overlap_sparray = sparse.eye(
                self.hamiltonian_sparray.shape[0],
                format="coo",
                dtype=self.hamiltonian_sparray.dtype,
            )

        self.overlap_sparray = self.overlap_sparray.tolil()
        # Check that the overlap matrix and Hamiltonian matrix match.
        if self.overlap_sparray.shape != self.hamiltonian_sparray.shape:
            raise ValueError(
                "Overlap matrix and Hamiltonian matrix have different shapes."
            )

        # Construct the bare system matrix.
        self.bare_system_matrix = compute_config.dbsparse_type.from_sparray(
            self.hamiltonian_sparray,
            block_sizes=self.block_sizes,
            global_stack_shape=(self.energies.size,),
            densify_blocks=[(i, i) for i in range(len(self.block_sizes))],
        )
        self.bare_system_matrix.data[:] = 0.0

        self.bare_system_matrix += self.overlap_sparray
        scale_stack(self.bare_system_matrix.data[:], self.local_energies)
        self.bare_system_matrix -= self.hamiltonian_sparray

        # Load the potential.
        try:
            self.potential = distributed_load(
                quatrex_config.input_dir / "potential.npy"
            )
            if self.potential.size != self.hamiltonian_sparray.shape[0]:
                raise ValueError(
                    "Potential matrix and Hamiltonian have different shapes."
                )
        except FileNotFoundError:
            # No potential provided. Assume zero potential.
            self.potential = xp.zeros(
                self.hamiltonian_sparray.shape[0], dtype=self.hamiltonian_sparray.dtype
            )

        self.bare_system_matrix -= sparse.diags(self.potential)

        self.system_matrix = compute_config.dbsparse_type.zeros_like(
            self.bare_system_matrix
        )

        # Boundary conditions.
        self.eta = quatrex_config.electron.eta
        self.left_occupancies = fermi_dirac(
            self.local_energies - quatrex_config.electron.left_fermi_level,
            quatrex_config.electron.temperature,
        )
        self.right_occupancies = fermi_dirac(
            self.local_energies - quatrex_config.electron.right_fermi_level,
            quatrex_config.electron.temperature,
        )

        # Allocate memory for the OBC blocks.
        self.obc_blocks_retarded_left = xp.zeros_like(self.system_matrix.blocks[0, 0])
        self.obc_blocks_retarded_right = xp.zeros_like(
            self.system_matrix.blocks[-1, -1]
        )
        self.obc_blocks_lesser_left = xp.zeros_like(self.system_matrix.blocks[0, 0])
        self.obc_blocks_lesser_right = xp.zeros_like(self.system_matrix.blocks[-1, -1])
        self.obc_blocks_greater_left = xp.zeros_like(self.system_matrix.blocks[0, 0])
        self.obc_blocks_greater_right = xp.zeros_like(self.system_matrix.blocks[-1, -1])

    def update_potential(self, new_potential: xp.ndarray) -> None:
        """Updates the potential matrix."""
        potential_diff_matrix = sparse.diags(new_potential - self.potential)
        self.bare_system_matrix -= potential_diff_matrix
        self.potential = new_potential

    def _get_block(self, lil: sparse.lil_array, index: tuple) -> xp.ndarray:
        """Gets a block from a LIL matrix."""
        row, col = index
        row = row + len(self.block_sizes) if row < 0 else row
        col = col + len(self.block_sizes) if col < 0 else col
        block = lil[
            self.block_offsets[row] : self.block_offsets[row + 1],
            self.block_offsets[col] : self.block_offsets[col + 1],
        ].toarray()
        return block

    def _apply_obc(self, sse_lesser, sse_greater) -> None:
        """Applies the OBC algorithm."""
        # Extract the overlap matrix blocks.
        s_00 = self._get_block(self.overlap_sparray, (0, 0))
        s_01 = self._get_block(self.overlap_sparray, (0, 1))
        s_10 = self._get_block(self.overlap_sparray, (1, 0))
        s_nn = self._get_block(self.overlap_sparray, (-1, -1))
        s_nm = self._get_block(self.overlap_sparray, (-1, -2))
        s_mn = self._get_block(self.overlap_sparray, (-2, -1))

        # Compute surface Green's functions.
        g_00 = self.obc(
            self.system_matrix.blocks[0, 0] + 1j * self.eta * s_00,
            self.system_matrix.blocks[0, 1] + 1j * self.eta * s_01,
            self.system_matrix.blocks[1, 0] + 1j * self.eta * s_10,
            "left",
        )
        g_nn = self.obc(
            self.system_matrix.blocks[-1, -1] + 1j * self.eta * s_nn,
            self.system_matrix.blocks[-1, -2] + 1j * self.eta * s_nm,
            self.system_matrix.blocks[-2, -1] + 1j * self.eta * s_mn,
            "right",
        )

        # Apply the retarded boundary self-energy.
        self.system_matrix.blocks[0, 0] -= (
            self.system_matrix.blocks[1, 0] @ g_00 @ self.system_matrix.blocks[0, 1]
        )
        self.system_matrix.blocks[-1, -1] -= (
            self.system_matrix.blocks[-2, -1] @ g_nn @ self.system_matrix.blocks[-1, -2]
        )

        # Compute and apply the lesser boundary self-energy.
        a_00 = g_00.conj().transpose(0, 2, 1) - g_00
        a_nn = g_nn.conj().transpose(0, 2, 1) - g_nn
        scale_stack(a_00, self.left_occupancies)
        scale_stack(a_nn, self.right_occupancies)

        sse_lesser.blocks[0, 0] += (
            self.system_matrix.blocks[1, 0] @ a_00 @ self.system_matrix.blocks[0, 1]
        )
        sse_lesser.blocks[-1, -1] += (
            self.system_matrix.blocks[-2, -1] @ a_nn @ self.system_matrix.blocks[-1, -2]
        )

        # Compute and apply the greater boundary self-energy.
        a_00 = g_00.conj().transpose(0, 2, 1) - g_00
        a_nn = g_nn.conj().transpose(0, 2, 1) - g_nn
        scale_stack(a_00, 1 - self.left_occupancies)
        scale_stack(a_nn, 1 - self.right_occupancies)

        sse_greater.blocks[0, 0] -= (
            self.system_matrix.blocks[1, 0] @ a_00 @ self.system_matrix.blocks[0, 1]
        )
        sse_greater.blocks[-1, -1] -= (
            self.system_matrix.blocks[-2, -1] @ a_nn @ self.system_matrix.blocks[-1, -2]
        )

    def _assemble_system_matrix(self, sse_retarded: DSBSparse) -> None:
        """Assembles the system matrix."""
        self.system_matrix.data[:] = self.bare_system_matrix.data
        self.system_matrix -= sse_retarded

    def _stash_contact_blocks(
        self,
        sse_lesser: DSBSparse,
        sse_greater: DSBSparse,
        sse_retarded: DSBSparse,
    ):
        """Stashes the contact OBC blocks."""
        self.obc_blocks_retarded_left[:] = sse_retarded.blocks[0, 0]
        self.obc_blocks_retarded_right[:] = sse_retarded.blocks[-1, -1]
        self.obc_blocks_lesser_left[:] = sse_lesser.blocks[0, 0]
        self.obc_blocks_lesser_right[:] = sse_lesser.blocks[-1, -1]
        self.obc_blocks_greater_left[:] = sse_greater.blocks[0, 0]
        self.obc_blocks_greater_right[:] = sse_greater.blocks[-1, -1]

    def _recover_contact_blocks(
        self,
        sse_lesser: DSBSparse,
        sse_greater: DSBSparse,
        sse_retarded: DSBSparse,
    ):
        """Recovers the contact OBC blocks."""
        sse_retarded.blocks[0, 0] = self.obc_blocks_retarded_left[:]
        sse_retarded.blocks[-1, -1] = self.obc_blocks_retarded_right[:]
        sse_lesser.blocks[0, 0] = self.obc_blocks_lesser_left[:]
        sse_lesser.blocks[-1, -1] = self.obc_blocks_lesser_right[:]
        sse_greater.blocks[0, 0] = self.obc_blocks_greater_left[:]
        sse_greater.blocks[-1, -1] = self.obc_blocks_greater_right[:]

    def solve(
        self,
        sse_lesser: DSBSparse,
        sse_greater: DSBSparse,
        sse_retarded: DSBSparse,
        out: tuple[DSBSparse, ...],
    ):
        """Solves for the lesser electron Green's function."""
        self._stash_contact_blocks(sse_lesser, sse_greater, sse_retarded)

        print("Assembling system matrix.", flush=True)
        self._assemble_system_matrix(sse_retarded)

        print("Applying OBC.", flush=True)
        self._apply_obc(sse_lesser, sse_greater)

        print("Computing electron Green's function.", flush=True)
        self.solver.selected_solve(
            a=self.system_matrix,
            sigma_lesser=sse_lesser,
            sigma_greater=sse_greater,
            out=out,
            return_retarded=True,
        )

        print("Recovering contact OBC blocks.", flush=True)
        self._recover_contact_blocks(sse_lesser, sse_greater, sse_retarded)

        return out
