# Copyright 2023-2024 ETH Zurich and QuaTrEx authors. All rights reserved.
import numpy as np

from scipy import sparse

from quatrex.core.config import QuatrexConfig
from quatrex.core.subsystem import SubsystemSolver
from quatrex.core.statistics import fermi_dirac

from quatrex.core.coo import COOBatch


class ElectronSolver(SubsystemSolver):
    """Solves for the lesser electron Green's function."""

    system = "electron"

    def __init__(
        self,
        config: QuatrexConfig,
        sigma_lesser: sparse.coo_array | None = None,
        sigma_greater: sparse.coo_array | None = None,
        **kwargs,
    ) -> None:
        """Initializes the solver."""
        super().__init__(config)

        self.hamiltonian = sparse.load_npz(config.input_dir / "hamiltonian.npz")
        potential = np.load(config.input_dir / "potential.npy")

        self.potential = sparse.diags(potential, format="coo")

        self.sigma_lesser = sigma_lesser
        self.sigma_greater = sigma_greater

    def apply_obc(
        self,
        system_matrices: COOBatch,
        return_sigma_obc: bool = False,
        occupancies_l: tuple[float, float] = None,
        occupancies_g: tuple[float, float] = None,
    ) -> tuple[COOBatch, COOBatch | None, COOBatch | None]:
        """Applies the OBC to the system matrix.

        Parameters
        ----------
        m : bsp.BSparse
            The system matrix.
        return_sigma_lesser : bool
            If True, the lesser boundary self-energy is returned.
        occupancies_l : tuple[float, float]
            The lesser left and right occupancies at the energy point.
        occupancies_g : tuple[float, float]
            The greater left and right occupancies at the energy point.

        Returns
        -------
        m : bsp.BSparse
            The system matrix with OBC applied.
        sigma_lesser : bsp.BSparse
            The lesser boundary self-energy.
        sigma_greater : bsp.BSparse
            The greater boundary self-energy.

        """
        ir = self.interaction_range

        m_01 = m[:ir, ir : 2 * ir].toarray()
        m_10 = m[ir : 2 * ir, :ir].toarray()
        m_nm = m[-ir:, -2 * ir : -ir].toarray()
        m_mn = m[-2 * ir : -ir, -ir:].toarray()

        m_00 = m[:ir, :ir].toarray()
        m_nn = m[-ir:, -ir:].toarray()

        g_00 = self.obc(
            m_00 + 1j * self.eta * np.eye(*m_00.shape), m_01, m_10, side="left"
        )
        g_nn = self.obc(
            m_nn + 1j * self.eta * np.eye(*m_nn.shape), m_nm, m_mn, side="right"
        )

        sigma_obc_00 = m_10 @ g_00 @ m_01
        sigma_obc_nn = m_mn @ g_nn @ m_nm

        # Apply self-energy. This is an issue with bsparse.
        for row, col in np.ndindex(ir, ir):
            m[row, col] -= sigma_obc_00[
                row * m[0, 0].shape[0] : (row + 1) * m[0, 0].shape[0],
                col * m[0, 0].shape[1] : (col + 1) * m[0, 0].shape[1],
            ]
            m[-ir + row, -ir + col] -= sigma_obc_nn[
                row * m[0, 0].shape[0] : (row + 1) * m[0, 0].shape[0],
                col * m[0, 0].shape[1] : (col + 1) * m[0, 0].shape[1],
            ]

        if not return_sigma_obc:
            return m

        if occupancies_l is None or occupancies_g is None:
            raise ValueError("Occupancies must be set to compute sigma_lesser.")

        sigma_l_obc_00 = m_10 @ (occupancies_l[0] * (g_00.conj().T - g_00)) @ m_01
        sigma_l_obc_nn = m_mn @ (occupancies_l[-1] * (g_nn.conj().T - g_nn)) @ m_nm

        sigma_g_obc_00 = -m_10 @ (occupancies_g[0] * (g_00.conj().T - g_00)) @ m_01
        sigma_g_obc_nn = -m_mn @ (occupancies_g[-1] * (g_nn.conj().T - g_nn)) @ m_nm

        sigma_l_obc = m.copy() * 0.0
        sigma_g_obc = m.copy() * 0.0
        # Construct self-energy. This is an issue with bsparse.
        for row, col in np.ndindex(ir, ir):
            sigma_l_00_rc = sigma_l_obc_00[
                row * m[0, 0].shape[0] : (row + 1) * m[0, 0].shape[0],
                col * m[0, 0].shape[1] : (col + 1) * m[0, 0].shape[1],
            ]
            sigma_l_nn_rc = sigma_l_obc_nn[
                row * m[0, 0].shape[0] : (row + 1) * m[0, 0].shape[0],
                col * m[0, 0].shape[1] : (col + 1) * m[0, 0].shape[1],
            ]
            sigma_g_00_rc = sigma_g_obc_00[
                row * m[0, 0].shape[0] : (row + 1) * m[0, 0].shape[0],
                col * m[0, 0].shape[1] : (col + 1) * m[0, 0].shape[1],
            ]
            sigma_g_nn_rc = sigma_g_obc_nn[
                row * m[0, 0].shape[0] : (row + 1) * m[0, 0].shape[0],
                col * m[0, 0].shape[1] : (col + 1) * m[0, 0].shape[1],
            ]

            sigma_l_obc[row, col] = sigma_l_00_rc
            sigma_l_obc[-ir + row, -ir + col] = sigma_l_nn_rc

            sigma_g_obc[row, col] = sigma_g_00_rc
            sigma_g_obc[-ir + row, -ir + col] = sigma_g_nn_rc

        return m, sigma_l_obc, sigma_g_obc

    def assemble_system_matrix(self, energy: float) -> bsp.BSparse:
        """Assembles the system matrix for a given energy.

        Parameters
        ----------
        energy : float
            The energy.

        Returns
        -------
        system_matrix : bsp.BSparse
            The system matrix.

        """
        energy_diag = bsp.BCOO.from_array(
            energy * np.eye(self.hamiltonian.shape[0], dtype=np.complex128),
            sizes=(self.hamiltonian.row_sizes, self.hamiltonian.col_sizes),
        )
        if self.sigma_lesser is None or self.sigma_greater is None:
            return energy_diag - self.hamiltonian - self.potential

        index = np.where(np.isclose(self.energies, energy))[0][0]

        sigma_lesser = self.sigma_lesser[index]
        sigma_greater = self.sigma_greater[index]

        sigma_causal = 0.5 * (sigma_greater - sigma_lesser)

        return energy_diag - self.hamiltonian - self.potential - sigma_causal
