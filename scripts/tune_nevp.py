# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from qttools.nevp import Beyn, Full
from qttools.obc import Spectral
from scipy import sparse

from quatrex.core.quatrex_config import QuatrexConfig, parse_config


class BeynTuner:
    """A class to tune the subspace NEVP solver for a given problem.

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex config file.
    num_energy_samples : int, optional
        The number of energy samples to use for tuning. Defaults to 100.
    min_num_quad_points : int, optional
        The minimum number of quadrature points to use. Defaults to 5.
    max_num_quad_points : int, optional
        The maximum number of quadrature points to use. Defaults to 40.

    """

    def __init__(
        self,
        config: QuatrexConfig,
        num_energy_samples: int = 50,
        min_num_quad_points: int = 5,
        max_num_quad_points: int = 40,
    ) -> None:
        """Initializes the tuner."""

        energies = np.load(config.input_dir / "electron_energies.npy")

        if num_energy_samples > len(energies):
            self.energies = energies
        else:
            # Pick an evenly-spaced subset from the energies for tuning.
            energy_indices = np.linspace(
                0, len(energies) - 1, num_energy_samples, dtype=int
            )
            self.energies = energies[energy_indices]

        hamiltonian = (
            sparse.load_npz(config.input_dir / "hamiltonian.npz")
            .astype(np.complex128)
            .tolil()
        )
        block_sizes = np.load(config.input_dir / "block_sizes.npy")
        try:
            overlap = (
                sparse.load_npz(config.input_dir / "overlap.npz")
                .astype(np.complex128)
                .tolil()
            )
        except FileNotFoundError:
            # No overlap provided. Assume orthonormal basis.
            overlap = sparse.eye(
                hamiltonian.shape[0],
                format="lil",
                dtype=hamiltonian.dtype,
            )

        # Extract the system matrix for the blocks we need.
        s = slice(None, sum(block_sizes[:2]))
        overlap = overlap[s, s].toarray()
        hamiltonian = hamiltonian[s, s].toarray()

        eta = config.electron.eta + config.electron.eta_obc

        system_matrix = (
            self.energies[:, np.newaxis, np.newaxis] + 1j * eta
        ) * overlap - hamiltonian

        i_ = slice(None, block_sizes[0])
        j_ = slice(block_sizes[0], sum(block_sizes[:2]))
        self.a_ji = system_matrix[:, j_, i_]
        self.a_ii = system_matrix[:, i_, i_]
        self.a_ij = system_matrix[:, i_, j_]

        # Extract the OBC parameters that will not change.
        self.obc_kwargs = {
            "block_sections": config.electron.obc.block_sections,
            "min_decay": config.electron.obc.min_decay,
            "max_decay": config.electron.obc.max_decay,
            "num_ref_iterations": config.electron.obc.num_ref_iterations,
            "x_ii_formula": config.electron.obc.x_ii_formula,
            "two_sided": config.electron.obc.two_sided,
            "treat_pairwise": config.electron.obc.treat_pairwise,
            "pairing_threshold": config.electron.obc.pairing_threshold,
            "min_propagation": config.electron.obc.min_propagation,
        }

        # Extract the NEVP parameters that will not change.
        self.nevp_kwargs = {
            "r_o": config.electron.obc.r_o,
            "r_i": config.electron.obc.r_i,
        }
        self.config = config
        self.min_num_quad_points = min_num_quad_points
        self.max_num_quad_points = max_num_quad_points

        obc = Spectral(Full(), **self.obc_kwargs)
        self.a_xx = obc._extract_subblocks(self.a_ji, self.a_ii, self.a_ij)

    def _compute_subspace_dimension(self) -> int:
        """Determines the number of eigenvalues to target in the subspace.

        The naive approach is to compute all the eigenvalues and then
        count the number of eigenvalues that lie within the contour.

        Note that this is not the most efficient approach, but it is
        the most straightforward.

        """
        full_nevp_solver = Full()
        ws, __ = full_nevp_solver(self.a_xx)

        # Count the number of eigenvalues that lie within the contour.
        mask = (np.abs(ws) < self.nevp_kwargs["r_o"]) & (
            np.abs(ws) > self.nevp_kwargs["r_i"]
        )
        m_0 = np.sum(mask, axis=1).max()

        print(f"Computed subspace dimension: {m_0}")

        return m_0

    def _tune_subspace_solver(self, m_0: int, processes: int) -> None:
        """Tunes the subspace NEVP solver for the given problem.

        This varies the number of quadrature points and finds the
        optimal value for the given subspace dimension.

        Parameters
        ----------
        m_0 : int
            The number of eigenvalues to target in the subspace.
        processes : int
            The number of processes to use for the evaluation of
            different numbers of quadrature points.

        """
        global _evaluate

        def _evaluate(num_quad_points: int) -> float:
            """Evaluates the subspace NEVP solver."""
            nevp = Beyn(
                m_0=m_0,
                num_quad_points=num_quad_points,
                **self.nevp_kwargs,
            )
            ws, vs = nevp(self.a_xx)

            residuals = np.zeros_like(ws)
            for i in range(ws.shape[1]):
                w = ws[:, i][:, None, None]
                v = vs[:, :, i] / np.linalg.norm(vs[:, :, i], axis=1)[:, None]
                with np.errstate(divide="ignore", invalid="ignore"):
                    residual = np.einsum(
                        "eij,ej->ei", self.a_ji / w + self.a_ii + self.a_ij * w, v
                    )
                    residuals[:, i] = np.linalg.norm(residual, axis=1)

            residuals = np.nan_to_num(residuals, posinf=0, neginf=0)

            # Filter outlier eigenmodes (robust Z-score method).
            medians = np.median(residuals, axis=-1)[..., None]
            median_abs_deviations = np.median(np.abs(residuals - medians), axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                z_scores = (
                    0.6745 * (residuals - medians) / median_abs_deviations[..., None]
                )
            spurious_mask = np.abs(z_scores) > 30  # Very generous threshold.

            res = np.abs(np.mean(residuals[~spurious_mask]))
            print(f"{num_quad_points:15} | {res:.5e}")
            return res

        num_quad_points = np.arange(self.min_num_quad_points, self.max_num_quad_points)
        print("Starting brute-force search for optimal number of quadrature points.")
        print("num_quad_points | average residual")
        with mp.Pool(processes=processes) as pool:
            results = pool.map(_evaluate, num_quad_points)

        opt_ind = np.argmin(results)
        opt_num_quad_points = num_quad_points[opt_ind]
        print(f"Determined optimal number of quadrature points: {opt_num_quad_points}")
        print(f"Minimum average residual: {results[opt_ind]}")

        return opt_num_quad_points

    def _validate_tuning(self, m_0: int, num_quad_points: int) -> None:
        """Validates the tuning of the subspace NEVP solver."""
        nevp = Beyn(
            m_0=m_0,
            num_quad_points=num_quad_points,
            **self.nevp_kwargs,
        )
        obc = Spectral(nevp, **self.obc_kwargs)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_ii = obc(a_ii=self.a_ii, a_ij=self.a_ij, a_ji=self.a_ji, contact="left")

        abs_rec_error = np.linalg.norm(
            x_ii - np.linalg.inv(self.a_ii - self.a_ji @ x_ii @ self.a_ij), axis=(1, 2)
        )
        rel_rec_error = abs_rec_error / np.linalg.norm(x_ii, axis=(1, 2))

        print(f"Average absolute recursion error: {abs_rec_error.mean()}")
        print(f"Average relative recursion error: {rel_rec_error.mean()}")

        nevp = Full()
        obc = Spectral(nevp, **self.obc_kwargs)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_ii = obc(a_ii=self.a_ii, a_ij=self.a_ij, a_ji=self.a_ji, contact="left")

        abs_rec_error = np.linalg.norm(
            x_ii - np.linalg.inv(self.a_ii - self.a_ji @ x_ii @ self.a_ij), axis=(1, 2)
        )
        rel_rec_error = abs_rec_error / np.linalg.norm(x_ii, axis=(1, 2))

        print(
            f"Full reference; average absolute recursion error: {abs_rec_error.mean()}"
        )
        print(
            f"Full reference; average relative recursion error: {rel_rec_error.mean()}"
        )

    def tune(self, processes: int = 1) -> None:
        """Tunes the subspace NEVP solver for the given problem."""
        m_0 = int(self._compute_subspace_dimension() * 1.5)
        print(f"Tuning for subspace dimension: {m_0}")
        num_quad_points = self._tune_subspace_solver(m_0, processes=processes)
        self._validate_tuning(m_0, num_quad_points)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    config = parse_config(path)
    tuner = BeynTuner(config)
    tuner.tune()
