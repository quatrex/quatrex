# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

from abc import ABC, abstractmethod

from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBSparse
from qttools.utils.stack_utils import scale_stack

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.subsystem import SubsystemSolver


class Hamiltonian:
    def __init__(self):
        pass


class GreenFunction(SubsystemSolver):
    """class for Green's function solver."""

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: xp.ndarray,
    ) -> None:
        """Initializes the GF solver."""
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
        self.eta = quatrex_config.electron.eta
        self.bare_system_matrix -= (
            self.hamiltonian_sparray - 1j * self.eta * self.overlap_sparray
        )

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
        self.eta_obc = quatrex_config.electron.eta_obc
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

        self.i_left = None
        self.i_right = None

    # def __init__(self,energies:NDArray | None) -> None:
    #     """Initializes the Green's function."""
    #     self.energies = energies
    #     self.invGR0 = None
    #     self.GR0 = None
    #     self.SigmaR = None
    #     self.SigmaG = None
    #     self.SigmaL = None
    #     self.H = None
    #     self.S = None
    #     self.GR = None
    #     self.GL = None
    #     self.GG = None
    #     RGFSolver = RGF()

    def SolveDyson(
        self,
        SigmaR: DSBSparse,
        SigmaL: DSBSparse,
        SigmaG: DSBSparse,
        out: tuple[DSBSparse, ...],
    ) -> None:
        """Solves the Dyson equation of Green's function, using the non-interacting GF, or its inverse, or the Hamiltonian."""

        if self.invG0 is not None:
            # self.G_r = inv(self.invG0 - self.Sigma)
            self.system_matrix += self.invG0
            self.system_matrix -= self.Sigma

            RGFSolver.selected_solve(
                self.system_matrix,
                self.SigmaL,
                self.SigmaG,
                out=(self.G_r, self.G_l, self.G_g),
            )

            return

        elif self.GF0 is not None:
            # self.G_r = inv(1 - self.GF0 @ self.Sigma) @ self.GF0
            self.system_matrix = MATMUL(self.GF0, self.Sigma)
            sigL = triMATMUL(self.GF0, self.SigmaL)
            sigG = triMATMUL(self.GF0, self.SigmaG)
            # sigL = self.GF0 @ self.SigmaL @ dagger(self.GF0)
            # sigG = self.GF0 @ self.SigmaG @ dagger(self.GF0)
            RGFSolver.selected_solver(
                self.system_matrix, sigL, sigG, out=(self.G_r, self.G_l, self.G_g)
            )

            return

        elif self.H is not None:
            # self.G_r = inv(omega + 1j*eta - self.H - self.Sigma)
            set_diag(self.system_matrix, (omega + 1j * eta))
            self.system_matrix -= self.H
            self.system_matrix -= self.Sigma
            RGFSolver.selected_solver(
                self.system_matrix,
                self.SigmaL,
                self.SigmaG,
                out=(self.G_r, self.G_l, self.G_g),
            )

            return

    def LDOS(
        self,
        *args,
        **kwargs,
    ):
        """Computes the local density of states."""
        ...

    def DOS(
        self,
        *args,
        **kwargs,
    ):
        """Computes the density of states."""
        ...

    def current(
        self,
        *args,
        **kwargs,
    ):
        """Computes the current."""
        ...

    def charge(
        self,
        *args,
        **kwargs,
    ):
        """Computes the charge."""
        ...

    def selected_solve(
        self,
        g0: DSBSparse,
        sigma_retarded: DSBSparse,
        sigma_lesser: DSBSparse,
        sigma_greater: DSBSparse,
        out: tuple[DSBSparse, ...] | None = None,
        return_retarded: bool = False,
        return_current: bool = False,
    ) -> None | tuple | NDArray:
        r"""Produces elements of the solution to the congruence equation.

        This method produces selected elements of the solution to the
        relation:

        \[
            X^{\lessgtr} = A^{-1} \Sigma^{\lessgtr} A^{-\dagger}
        \]

        Parameters
        ----------
        a : DSBSparse
            Matrix to invert.
        sigma_lesser : DSBSparse
            Lesser matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        sigma_greater : DSBSparse
            Greater matrix. This matrix is expected to be
            skew-hermitian, i.e. \(\Sigma_{ij} = -\Sigma_{ji}^*\).
        obc_blocks : OBCBlocks, optional
            OBC blocks for lesser, greater and retarded Green's
            functions. By default None.
        out : tuple[DSBSparse, ...] | None, optional
            Preallocated output matrices, by default None
        return_retarded : bool, optional
            Wether the retarded Green's function should be returned
            along with lesser and greater, by default False
        return_current : bool, optional
            Whether to compute and return the current for each layer via
            the Meir-Wingreen formula. By default False.

        Returns
        -------
        None | tuple | NDArray
            If `out` is None, returns None. Otherwise, the solutions are
            returned as DSBParse matrices. If `return_retarded` is True,
            returns a tuple with the retarded Green's function as the
            last element. If `return_current` is True, returns the
            current for each layer.

        """
        # Initialize empty lists for the dense diagonal blocks.
        xr_diag_blocks: list[NDArray | None] = [None] * a.num_blocks
        xl_diag_blocks: list[NDArray | None] = [None] * a.num_blocks
        xg_diag_blocks: list[NDArray | None] = [None] * a.num_blocks

        # Check if the current should be computed.
        if return_current:
            # Allocate a buffer for the current.
            current = xp.zeros((a.shape[0], a.num_blocks - 1), dtype=a.dtype)

        # Get list of batches to perform
        batches_sizes, batches_slices = get_batches(a.shape[0], self.max_batch_size)

        # If out is not none, xr will be the third element of the tuple.
        if out is not None:
            xl, xg, *xr = out
            if len(xr) == 0:
                # Allocate the retarded Green's function.
                xr = a.__class__.zeros_like(a)
            elif len(xr) == 1:
                # Unpack the tuple.
                xr = xr[0]
            else:
                raise ValueError("Invalid number of output matrices.")

        else:
            xr = a.__class__.zeros_like(a)
            xl = a.__class__.zeros_like(a)
            xg = a.__class__.zeros_like(a)

        # Perform the selected solve by batches.
        for i in range(len(batches_sizes)):
            stack_slice = slice(int(batches_slices[i]), int(batches_slices[i + 1]), 1)

            a_ = a.stack[stack_slice]
            sigma_lesser_ = sigma_lesser.stack[stack_slice]
            sigma_greater_ = sigma_greater.stack[stack_slice]

            xr_ = xr.stack[stack_slice]
            xl_ = xl.stack[stack_slice]
            xg_ = xg.stack[stack_slice]

            # Check if there are OBC blocks for the current layer.
            obc_r = obc_blocks.retarded[0]
            a_00 = a_.blocks[0, 0] if obc_r is None else a_.blocks[0, 0] - obc_r
            obc_l = obc_blocks.lesser[0]
            sl_00 = (
                sigma_lesser_.blocks[0, 0]
                if obc_l is None
                else sigma_lesser_.blocks[0, 0] + obc_l
            )
            obc_g = obc_blocks.greater[0]
            sg_00 = (
                sigma_greater_.blocks[0, 0]
                if obc_g is None
                else sigma_greater_.blocks[0, 0] + obc_g
            )

            xr_00 = xp.linalg.inv(a_00)
            xr_00_dagger = xr_00.conj().swapaxes(-2, -1)
            xr_diag_blocks[0] = xr_00
            xl_diag_blocks[0] = xr_00 @ sl_00 @ xr_00_dagger
            xg_diag_blocks[0] = xr_00 @ sg_00 @ xr_00_dagger

            # Forwards sweep.
            for i in range(a.num_blocks - 1):
                j = i + 1

                # Check if there are OBC blocks for the current layer.
                obc_r = obc_blocks.retarded[j]
                a_jj = a_.blocks[j, j] if obc_r is None else a_.blocks[j, j] - obc_r
                obc_l = obc_blocks.lesser[j]
                sl_jj = (
                    sigma_lesser_.blocks[j, j]
                    if obc_l is None
                    else sigma_lesser_.blocks[j, j] + obc_l
                )
                obc_g = obc_blocks.greater[j]
                sg_jj = (
                    sigma_greater_.blocks[j, j]
                    if obc_g is None
                    else sigma_greater_.blocks[j, j] + obc_g
                )

                # Get the blocks that are used multiple times.
                a_ji = a_.blocks[j, i]
                xr_ii = xr_diag_blocks[i]

                # Precompute the transposes that are used multiple times.
                a_ji_dagger = a_ji.conj().swapaxes(-2, -1)

                # Precompute some terms that are used multiple times.
                a_ji_xr_ii = a_ji @ xr_ii
                a_ji_xr_ii_sl_ij = a_ji_xr_ii @ sigma_lesser_.blocks[i, j]
                a_ji_xr_ii_sg_ij = a_ji_xr_ii @ sigma_greater_.blocks[i, j]

                xr_jj = xp.linalg.inv(a_jj - a_ji @ xr_ii @ a_.blocks[i, j])
                xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)
                xr_diag_blocks[j] = xr_jj

                xl_diag_blocks[j] = (
                    xr_jj
                    @ (
                        sl_jj
                        + a_ji @ xl_diag_blocks[i] @ a_ji_dagger
                        + a_ji_xr_ii_sl_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sl_ij
                    )
                    @ xr_jj_dagger
                )

                xg_diag_blocks[j] = (
                    xr_jj
                    @ (
                        sg_jj
                        + a_ji @ xg_diag_blocks[i] @ a_ji_dagger
                        + a_ji_xr_ii_sg_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sg_ij
                    )
                    @ xr_jj_dagger
                )

            # We need to write the last diagonal blocks to the output.
            xr_.blocks[-1, -1] = xr_diag_blocks[-1]
            xl_.blocks[-1, -1] = xl_diag_blocks[-1]
            xg_.blocks[-1, -1] = xg_diag_blocks[-1]

            # Backwards sweep.
            for i in range(a.num_blocks - 2, -1, -1):
                j = i + 1

                # Get the blocks that are used multiple times.
                xr_ii = xr_diag_blocks[i]
                xr_jj = xr_diag_blocks[j]
                a_ij = a_.blocks[i, j]
                a_ji = a_.blocks[j, i]
                xl_ii = xl_diag_blocks[i]
                xl_jj = xl_diag_blocks[j]
                xg_ii = xg_diag_blocks[i]
                xg_jj = xg_diag_blocks[j]
                sigma_lesser_ij = sigma_lesser_.blocks[i, j]
                sigma_greater_ij = sigma_greater_.blocks[i, j]

                # Precompute the transposes that are used multiple times.
                xr_jj_dagger = xr_jj.conj().swapaxes(-2, -1)
                xr_ii_dagger = xr_ii.conj().swapaxes(-2, -1)
                a_ij_dagger = a_ij.conj().swapaxes(-2, -1)
                sigma_greater_ji = -sigma_greater_ij.conj().swapaxes(-2, -1)
                sigma_lesser_ji = -sigma_lesser_ij.conj().swapaxes(-2, -1)

                # Precompute the terms that are used multiple times.
                xr_jj_dagger_aij_dagger = xr_jj_dagger @ a_ij_dagger
                a_ji_dagger_xr_jj_dagger = a_ji.conj().swapaxes(-2, -1) @ xr_jj_dagger
                a_ij_dagger_xr_ii_dagger = a_ij_dagger @ xr_ii_dagger
                a_ij_xr_jj = a_ij @ xr_jj
                xr_ii_a_ij = xr_ii @ a_ij
                xr_jj_a_ji = xr_jj @ a_ji
                xr_ii_a_ij_xr_jj_a_ji = xr_ii_a_ij @ xr_jj_a_ji
                xr_ii_a_ij_xl_jj = xr_ii_a_ij @ xl_jj
                xr_ii_a_ij_xg_jj = xr_ii_a_ij @ xg_jj

                temp_1_l = (
                    xr_ii
                    @ (
                        sigma_lesser_ij @ xr_jj_dagger_aij_dagger
                        + a_ij_xr_jj @ sigma_lesser_ji
                    )
                    @ xr_ii_dagger
                )

                temp_1_g = (
                    xr_ii
                    @ (
                        sigma_greater_ij @ xr_jj_dagger_aij_dagger
                        + a_ij_xr_jj @ sigma_greater_ji
                    )
                    @ xr_ii_dagger
                )

                temp_2_l = xr_ii_a_ij_xr_jj_a_ji @ xl_ii

                temp_2_g = xr_ii_a_ij_xr_jj_a_ji @ xg_ii

                xl_.blocks[i, j] = (
                    -xr_ii_a_ij_xl_jj
                    - xl_ii @ a_ji_dagger_xr_jj_dagger
                    + xr_ii @ sigma_lesser_ij @ xr_jj_dagger
                )

                xg_.blocks[i, j] = (
                    -xr_ii_a_ij_xg_jj
                    - xg_ii @ a_ji_dagger_xr_jj_dagger
                    + xr_ii @ sigma_greater_ij @ xr_jj_dagger
                )

                xl_.blocks[j, i] = (
                    -xl_jj @ a_ij_dagger_xr_ii_dagger
                    - xr_jj_a_ji @ xl_ii
                    + xr_jj @ sigma_lesser_ji @ xr_ii_dagger
                )

                xg_.blocks[j, i] = (
                    -xg_jj @ a_ij_dagger_xr_ii_dagger
                    - xr_jj_a_ji @ xg_ii
                    + xr_jj @ sigma_greater_ji @ xr_ii_dagger
                )

                if return_current:
                    a_ji_xr_ii_sl_ij = a_ji @ xr_ii @ sigma_lesser_ij
                    a_ji_xr_ii_sg_ij = a_ji @ xr_ii @ sigma_greater_ij
                    sigma_lesser_tilde = (
                        a_ji @ xl_ii @ a_ji_dagger
                        + a_ji_xr_ii_sl_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sl_ij
                    )
                    sigma_greater_tilde = (
                        a_ji @ xg_ii @ a_ji_dagger
                        + a_ji_xr_ii_sg_ij.conj().swapaxes(-2, -1)
                        - a_ji_xr_ii_sg_ij
                    )
                    current[stack_slice, i] = xp.trace(
                        sigma_greater_tilde @ xl_diag_blocks[j]
                        - xg_diag_blocks[j] @ sigma_lesser_tilde,
                        axis1=-2,
                        axis2=-1,
                    )

                # NOTE: Cursed Python multiple assignment syntax.
                xl_.blocks[i, i] = xl_diag_blocks[i] = (
                    xl_ii
                    + xr_ii_a_ij_xl_jj @ a_ij_dagger_xr_ii_dagger
                    - temp_1_l
                    + (temp_2_l - temp_2_l.conj().swapaxes(-2, -1))
                )
                xg_.blocks[i, i] = xg_diag_blocks[i] = (
                    xg_ii
                    + xr_ii_a_ij_xg_jj @ a_ij_dagger_xr_ii_dagger
                    - temp_1_g
                    + (temp_2_g - temp_2_g.conj().swapaxes(-2, -1))
                )

                xr_.blocks[i, i] = xr_diag_blocks[i] = (
                    xr_ii + xr_ii_a_ij_xr_jj_a_ji @ xr_ii
                )

        if out is None:
            if return_retarded:
                if return_current:
                    return xl, xg, xr, current
                return xl, xg, xr
            return xl, xg

        if return_current:
            return current
