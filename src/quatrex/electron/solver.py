# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
import numpy as np
from qttools.datastructures import DSBSparse
from qttools.utils import mpi_utils, stack_utils
from scipy import sparse

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.statistics import fermi_dirac
from quatrex.core.subsystem import SubsystemSolver


class ElectronSolver(SubsystemSolver):
    """Solves for the lesser electron Green's function."""

    system = "electron"

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: np.ndarray,
        **kwargs,
    ) -> None:
        """Initializes the solver."""
        super().__init__(quatrex_config, compute_config, energies)

        # load Hamiltonian matrix and block_sizes vector, raise error if not found
        self.hamiltonian_sparray = mpi_utils.distributed_load(
            quatrex_config.input_dir / "hamiltonian.npz"
        ).astype(np.complex128)

        self.block_sizes = mpi_utils.distributed_load(
            quatrex_config.input_dir / "block_sizes.npy"
        )

        # Check wether total sizes from blocksizes and hamiltonian shape match
        if self.block_sizes.sum() != self.hamiltonian_sparray.shape[0]:
            raise ValueError(
                f"Sum of block sizes does not match Hamiltonian size. {self.block_sizes.sum()} != {self.hamiltonian_sparray.shape[0]}"
            )

        # load overlap matrix, set to identity if None
        try:
            self.overlap_sparray = mpi_utils.distributed_load(
                quatrex_config.input_dir / "overlap.npz"
            ).astype(np.complex128)
        except FileNotFoundError:
            self.overlap_sparray = sparse.eye(
                self.hamiltonian_sparray.shape[0],
                format="coo",
                dtype=self.hamiltonian_sparray.dtype,
            )

        if self.overlap_sparray.shape != self.hamiltonian_sparray.shape:
            raise ValueError(
                "Overlap matrix and Hamiltonian matrix have different shapes."
            )

        self.bare_system_matrix = self.dbsparse.from_sparray(
            self.hamiltonian_sparray,
            block_sizes=self.block_sizes,
            global_stack_shape=(self.energies.size,),
            densify_blocks=[(0, 0), (-1, -1)],
        )
        self.bare_system_matrix.data[:] = 0.0

        self.bare_system_matrix += self.overlap_sparray
        stack_utils.scale_stack(
            self.bare_system_matrix.data[:], mpi_utils.get_local_slice(self.energies)
        )
        self.bare_system_matrix -= self.hamiltonian_sparray

        # load potential matrix, set to diagonal zero if None
        try:
            self.potential = mpi_utils.distributed_load(
                quatrex_config.input_dir / "potential.npy"
            )
            if self.potential.size != self.hamiltonian_sparray.shape[0]:
                raise ValueError(
                    "Potential matrix and Hamiltonian have different shapes."
                )
        except FileNotFoundError:
            # File does not exist. Set potential to zero.
            self.potential = np.zeros(
                self.hamiltonian_sparray.shape[0], dtype=self.hamiltonian_sparray.dtype
            )

        self.bare_system_matrix -= sparse.diags(self.potential)

        self.system_matrix = self.dbsparse.zeros_like(self.bare_system_matrix)

        # Boundary conditions.
        self.eta = quatrex_config.electron.eta
        self.left_occupancies = fermi_dirac(
            mpi_utils.get_local_slice(self.energies)
            - quatrex_config.electron.left_fermi_level,
            quatrex_config.electron.temperature,
        )
        self.right_occupancies = fermi_dirac(
            mpi_utils.get_local_slice(self.energies)
            - quatrex_config.electron.right_fermi_level,
            quatrex_config.electron.temperature,
        )

        # Allocated memory for resetting SSE OBC blocks.
        self.obc_blocks_retarded_left = np.zeros_like(self.system_matrix[0, 0])
        self.obc_blocks_retarded_right = np.zeros_like(self.system_matrix[-1, -1])
        self.obc_blocks_lesser_left = np.zeros_like(self.system_matrix[0, 0])
        self.obc_blocks_lesser_right = np.zeros_like(self.system_matrix[-1, -1])
        self.obc_blocks_greater_left = np.zeros_like(self.system_matrix[0, 0])
        self.obc_blocks_greater_right = np.zeros_like(self.system_matrix[-1, -1])

    def update_potential(self, new_potential: np.ndarray) -> None:
        """Updates the potential matrix."""
        potential_diff_matrix = sparse.diags(new_potential - self.potential)
        self.bare_system_matrix -= potential_diff_matrix
        self.potential = new_potential

    def _apply_obc(self, sse_lesser, sse_greater) -> None:
        """Applies the OBC algorithm."""
        s_00 = self.overlap_sparray.tolil()[
            : self.block_sizes[0], : self.block_sizes[0]
        ].toarray()

        s_nn = self.overlap_sparray.tolil()[
            -self.block_sizes[-1] :, -self.block_sizes[-1] :
        ].toarray()

        g_00 = self.obc(
            self.system_matrix[0, 0] + 1j * self.eta * s_00,
            self.system_matrix[0, 1],
            self.system_matrix[1, 0],
            "left",
        )
        g_nn = self.obc(
            self.system_matrix[-1, -1] + 1j * self.eta * s_nn,
            self.system_matrix[-1, -2],
            self.system_matrix[-2, -1],
            "right",
        )
        self.system_matrix[0, 0] -= (
            self.system_matrix[1, 0] @ g_00 @ self.system_matrix[0, 1]
        )
        self.system_matrix[-1, -1] -= (
            self.system_matrix[-2, -1] @ g_nn @ self.system_matrix[-1, -2]
        )

        a_00 = g_00.conj().transpose(0, 2, 1) - g_00
        a_nn = g_nn.conj().transpose(0, 2, 1) - g_nn
        stack_utils.scale_stack(a_00, self.left_occupancies)
        stack_utils.scale_stack(a_nn, self.right_occupancies)

        sse_lesser[0, 0] += self.system_matrix[1, 0] @ a_00 @ self.system_matrix[0, 1]
        sse_lesser[-1, -1] += (
            self.system_matrix[-2, -1] @ a_nn @ self.system_matrix[-1, -2]
        )

        a_00 = g_00.conj().transpose(0, 2, 1) - g_00
        a_nn = g_nn.conj().transpose(0, 2, 1) - g_nn
        stack_utils.scale_stack(a_00, 1 - self.left_occupancies)
        stack_utils.scale_stack(a_nn, 1 - self.right_occupancies)

        sse_greater[0, 0] -= self.system_matrix[1, 0] @ a_00 @ self.system_matrix[0, 1]
        sse_greater[-1, -1] -= (
            self.system_matrix[-2, -1] @ a_nn @ self.system_matrix[-1, -2]
        )

    def _assemble_system_matrix(self, sse_retarded: DSBSparse) -> None:
        """Assembles the system matrix."""
        self.system_matrix.data[:] = self.bare_system_matrix.data
        self.system_matrix -= sse_retarded

    def solve(
        self,
        sse_lesser: DSBSparse,
        sse_greater: DSBSparse,
        sse_retarded: DSBSparse,
        out: tuple[DSBSparse, ...],
    ):
        """Solves for the lesser electron Green's function."""
        self.obc_blocks_retarded_left[:] = sse_retarded[0, 0]
        self.obc_blocks_retarded_right[:] = sse_retarded[-1, -1]
        self.obc_blocks_lesser_left[:] = sse_lesser[0, 0]
        self.obc_blocks_lesser_right[:] = sse_lesser[-1, -1]
        self.obc_blocks_greater_left[:] = sse_greater[0, 0]
        self.obc_blocks_greater_right[:] = sse_greater[-1, -1]

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
        sse_retarded[0, 0] = self.obc_blocks_retarded_left[:]
        sse_retarded[-1, -1] = self.obc_blocks_retarded_right[:]
        sse_lesser[0, 0] = self.obc_blocks_lesser_left[:]
        sse_lesser[-1, -1] = self.obc_blocks_lesser_right[:]
        sse_greater[0, 0] = self.obc_blocks_greater_left[:]
        sse_greater[-1, -1] = self.obc_blocks_greater_right[:]

        return out
