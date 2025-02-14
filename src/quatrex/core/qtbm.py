# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import os
import time
from dataclasses import dataclass, field

from cupyx.profiler import time_range
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp, obc
from qttools.utils.mpi_utils import distributed_load
from cupyx.scipy.sparse.linalg import spsolve

from qttools.utils.mpi_utils import get_local_slice

from quatrex.core.statistics import fermi_dirac

from quatrex.core.compute_config import ComputeConfig

from qttools.nevp import NEVP, Beyn, Full

from quatrex.core.quatrex_config import (
    OBCConfig,
    QuatrexConfig,
)


@dataclass
class Observables:
    """Observable quantities for the SCBA."""

    # --- Electrons ----------------------------------------------------
    electron_ldos: NDArray = None
    electron_density: NDArray = None
    hole_density: NDArray = None
    electron_current: dict = field(default_factory=dict)
    
    electron_transmission: NDArray = None
    electron_DOS: NDArray = None

    valence_band_edges: NDArray = None
    conduction_band_edges: NDArray = None

    excess_charge_density: NDArray = None


class QTBM:
    """Quantum Transmitting Boundary Method (QTBM) solver.

    Parameters
    ----------
    quatrex_config : Path
        Quatrex configuration file.
    compute_config : Path, optional
        Compute configuration file, by default None. If None, the
        default compute parameters are used.

    """

    @time_range()
    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig | None = None,
    ) -> None:
        """Initializes a QTBM instance."""
        self.quatrex_config = quatrex_config

        if compute_config is None:
            compute_config = ComputeConfig()

        self.compute_config = compute_config

        self.observables = Observables()

        self.electron_energies = distributed_load(
            self.quatrex_config.input_dir / "electron_energies.npy"
        )

        self.local_energies = get_local_slice(self.electron_energies)
        self.obc = self._configure_obc(getattr(quatrex_config, "electron").obc)

        self.observables.electron_transmission = xp.zeros_like(self.local_energies)
        self.observables.electron_DOS = xp.zeros_like(self.local_energies)

        # Load the device Hamiltonian.
        self.hamiltonian_sparray = distributed_load(
            quatrex_config.input_dir / "hamiltonian.npz"
        ).astype(xp.complex128)

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
        
        self.hamiltonian_sparray = self.hamiltonian_sparray.tocsr()
        self.overlap_sparray = self.overlap_sparray.tocsr()

        # Check that the overlap matrix and Hamiltonian matrix match.
        if self.overlap_sparray.shape != self.hamiltonian_sparray.shape:
            raise ValueError(
                "Overlap matrix and Hamiltonian matrix have different shapes."
            )
        
        #Load potential (TODO)

        # Contacts.
        self.flatband = quatrex_config.electron.flatband
        self.eta_obc = quatrex_config.electron.eta_obc

        # Extract contact Hamiltonians. (TODO)

        self.n_block = 234 ##FOR TESTING ONLY
        ## CREATE MASKS FOR EVERY CONTACT

        # Band edges and Fermi levels.
        # TODO: This only works for small potential variations accross
        # the device.
        # TODO: During this initialization we should compute the contact
        # band structures and extract the correct fermi levels & band
        # edges from there.
        #self.band_edge_tracking = quatrex_config.electron.band_edge_tracking
        #self.delta_fermi_level_conduction_band = (
        #    quatrex_config.electron.conduction_band_edge
        #    - quatrex_config.electron.fermi_level
        #)
        #self.left_mid_gap_energy = quatrex_config.electron.left_fermi_level
        #self.right_mid_gap_energy = quatrex_config.electron.right_fermi_level

        self.temperature = quatrex_config.electron.temperature

        self.left_fermi_level = quatrex_config.electron.left_fermi_level
        self.right_fermi_level = quatrex_config.electron.right_fermi_level

        self.left_occupancies = fermi_dirac(
            self.local_energies - self.left_fermi_level, self.temperature
        )
        self.right_occupancies = fermi_dirac(
            self.local_energies - self.right_fermi_level, self.temperature
        )

    def _configure_nevp(self, obc_config: OBCConfig) -> NEVP:
        """Configures the NEVP solver from the config.

        Parameters
        ----------
        obc_config : OBCConfig
            The OBC configuration.

        Returns
        -------
        NEVP
            The configured NEVP solver.

        """
        if obc_config.nevp_solver == "beyn":
            return Beyn(
                r_o=obc_config.r_o,
                r_i=obc_config.r_i,
                m_0=obc_config.m_0,
                num_quad_points=obc_config.num_quad_points,
            )
        if obc_config.nevp_solver == "full":
            return Full()

        raise NotImplementedError(
            f"NEVP solver '{obc_config.nevp_solver}' not implemented."
        )

    def _configure_obc(self, obc_config: OBCConfig) -> obc.OBCSolver:
        """Configures the OBC algorithm from the config.

        Parameters
        ----------
        obc_config : OBCConfig
            The OBC configuration.

        Returns
        -------
        obc.OBCSolver
            The configured OBC solver.

        """
        if obc_config.algorithm == "sancho-rubio":
            raise NotImplementedError(
                f"Sancho-rubio OBC algorithm does not work with QTBM, please use spectral OBC solver."
            )

        elif obc_config.algorithm == "spectral":
            nevp = self._configure_nevp(obc_config)
            obc_solver = obc.Spectral(
                nevp=nevp,
                block_sections=obc_config.block_sections,
                min_decay=obc_config.min_decay,
                max_decay=obc_config.max_decay,
                num_ref_iterations=obc_config.num_ref_iterations,
                x_ii_formula=obc_config.x_ii_formula,
                two_sided=obc_config.two_sided,
                treat_pairwise=obc_config.treat_pairwise,
                pairing_threshold=obc_config.pairing_threshold,
                min_propagation=obc_config.min_propagation,
            )

        else:
            raise NotImplementedError(
                f"OBC algorithm '{obc_config.algorithm}' not implemented."
            )

        if obc_config.memoizer.enable:
            obc_solver = obc.OBCMemoizer(
                obc_solver,
                obc_config.memoizer.num_ref_iterations,
                obc_config.memoizer.convergence_tol,
            )

        return obc_solver

    def run(self) -> None:
        """Runs the QTBM"""
        print("Entering QTBM calculation", flush=True) if comm.rank == 0 else None
        times = []
        for i,E in enumerate(self.local_energies):

            print(f"Iteration {i}", flush=True) if comm.rank == 0 else None

            # append for iteration time
            times.append(time.perf_counter())

            times.append(time.perf_counter())

            self.system_matrix = self.hamiltonian_sparray - E * self.overlap_sparray

            t_solve = time.perf_counter() - times.pop()
            (
                print(f"Time for constructing bare sys. matrix: {t_solve:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            times.append(time.perf_counter())

            # Compute the boundary self-energy and the injection vector

            S_L, inj_L = self.obc(
            self.system_matrix[:self.n_block,:self.n_block].toarray(),
            self.system_matrix[:self.n_block,self.n_block:self.n_block*2].toarray(),
            self.system_matrix[self.n_block:self.n_block*2,:self.n_block].toarray(),
            "left",
            return_inj = True,
            )

            S_R, inj_R = self.obc(
            self.system_matrix[-self.n_block:,-self.n_block:].toarray(),
            self.system_matrix[-self.n_block:,-2*self.n_block:-self.n_block].toarray(),
            self.system_matrix[-2*self.n_block:-self.n_block:,-self.n_block:].toarray(),
            "right",
            return_inj = True,
            )

            t_solve = time.perf_counter() - times.pop()
            (
                print(f"Time for OBC: {t_solve:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            times.append(time.perf_counter())
            # Set up sytem matrix and rhs for electron solver.

            self.system_matrix[:self.n_block, :self.n_block] -= S_L
            self.system_matrix[-self.n_block:, -self.n_block:] -= S_R

            inj_V = xp.zeros((self.system_matrix.shape[0],inj_L.shape[1]+inj_R.shape[1]), dtype=xp.complex128)
            inj_V[:self.n_block,:inj_L.shape[1]] = inj_L
            inj_V[-self.n_block:,-inj_R.shape[1]:] = inj_R 

            t_solve = time.perf_counter() - times.pop()
            (
                print(f"Time to set up system of eq.: {t_solve:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            times.append(time.perf_counter())
            # Solve for the wavefunction

            self.phi = spsolve(self.system_matrix, inj_V)

            t_solve = time.perf_counter() - times.pop()
            (
                print(f"Time for electron solver: {t_solve:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )

            # Compute transmission and DOS
            T01 = self.system_matrix[:self.n_block,self.n_block:2*self.n_block]

            phi_L = self.phi[:self.n_block,:inj_L.shape[1]]
            phi_L_D = self.phi[:,:inj_L.shape[1]]
            phi_L_2 = self.phi[self.n_block:2*self.n_block,:inj_L.shape[1]]

            if(phi_L.size != 0):
                self.observables.electron_transmission[i] = xp.trace(2*xp.imag(phi_L.T.conj() @ T01 @phi_L_2))
                self.observables.electron_DOS[i]=xp.real(xp.sum(xp.multiply(phi_L_D.conj(), self.overlap_sparray @ phi_L_D))/(2*xp.pi))

            t_iteration = time.perf_counter() - times.pop()
            (
                print(f"Time for iteration: {t_iteration:.2f} s", flush=True)
                if comm.rank == 0
                else None
            )
        
