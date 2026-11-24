# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the core class for Schroedinger calculations."""

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.profiling import Profiler
from quatrex.core.config import QuatrexConfig
from quatrex.core.transport import TransportSolver
from quatrex.device import SchroedingerDevice

profiler = Profiler()


class Schroedinger(TransportSolver):

    def __init__(self, config: QuatrexConfig, device: SchroedingerDevice) -> None:
        """Initializes the Schroedinger solver."""

        self.device = device
        self.config = config

        self.num_orbitals = device.hamiltonians[0, 0, 0].shape[0]

        self.solver = sparse.linalg.spsolve

    def _assemble_system_operator(
        self,
        kpoint: xp.complex128,
    ) -> sparse.linalg.LinearOperator:

        hamiltonian = None
        for r, h_r in self.device.hamiltonians.items():
            phase_factor = np.exp(2j * np.pi * np.dot(kpoint, r))
            if hamiltonian is None:
                hamiltonian = h_r * phase_factor
            else:
                hamiltonian += h_r * phase_factor

        overlap = None
        for r, s_r in self.device.overlap_matrices.items():
            phase_factor = np.exp(2j * np.pi * np.dot(kpoint, r))
            if overlap is None:
                overlap = s_r * phase_factor
            else:
                overlap += s_r * phase_factor

        overlap = overlap + sparse.triu(overlap, k=1).conj().T
        hamiltonian = hamiltonian + sparse.triu(hamiltonian, k=1).conj().T

        hamiltonian += 0.5 * (
            overlap.multiply(self.device.potential)
            + overlap.multiply(self.device.potential[None, :])
        )

        # TODO: Set the correct sigma value based on the band gap or fermi level.
        sigma = -4  # xp.min(self.device.potential) -4

        def matvec(x: NDArray) -> NDArray:
            return self.solver(
                A=hamiltonian - sigma * overlap,
                b=overlap @ x,
            )

        operator = sparse.linalg.LinearOperator(
            shape=(self.num_orbitals, self.num_orbitals),
            matvec=matvec,
            dtype=xp.complex128,
        )

        return operator, sigma

    def set_potential(self, potential: NDArray):
        """Sets the potential for the QTBM calculation.

        This method can be used to update the potential for
        self-consistent calculations.

        Parameters
        ----------
        potential : NDArray
            The new potential values to be set in the system matrix.

        """
        if potential.shape[0] == self.device.atom_coordinates.shape[0]:
            # Upscale the potential to the number of orbitals
            orbitals_per_atom = [
                self.config.device.num_orbitals_per_atom.get(species, 1)
                for species in self.device.atomic_species
            ]
            potential = xp.repeat(potential, orbitals_per_atom, axis=0)

        self.device.potential = potential

    def get_charge_density(self) -> NDArray:
        """Gets the charge density from the QTBM calculation.

        This method integrates the local density of states to obtain the
        charge density. This is typically used in self-consistent
        calculations where the charge density is needed to update the
        potential.

        Returns
        -------
        charge_density : NDArray
            The computed charge density for the device.

        """
        pass

    def run(self) -> None:

        comm.barrier()

        for kpoint_ind, kpoint in enumerate(self.device.kpoints):

            operator, sigma = self._assemble_system_operator(kpoint)

            w, v = sparse.linalg.eigs(
                operator, k=100
            )  # , k=self.config.schroedinger.num_modes)

            # normalize the modes
            v /= xp.linalg.norm(v, axis=0)

            # get the correct eigenvalues
            w = 1 / w + sigma
