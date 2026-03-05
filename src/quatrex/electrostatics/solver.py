# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
import warnings
from abc import ABC, abstractmethod

import ase
import numpy as np
import skfem

from qttools import NDArray, sparse
from qttools.comm import comm
from quatrex.core.constants import epsilon_0
from quatrex.electrostatics import assembly
from quatrex.electrostatics.density_response import DensityModel


class PoissonSolver(ABC):
    """Abstract base class for Poisson solvers."""

    def __init__(
        self,
        mesh: skfem.MeshTet,
        basis: skfem.Basis,
        structure: ase.Atoms,
        potential_constraints: dict[str, tuple[float, NDArray]],
    ):
        """Initializes the Poisson solver."""
        self.potential_constraints = potential_constraints

        self.mfc_transform = assembly.assemble_mfc_transform(
            mesh, basis, structure.cell, structure.pbc
        )
        self.stiffness_matrix = self._assemble_stiffness_matrix(
            basis, potential_constraints, self.mfc_transform
        )

    @staticmethod
    def _assemble_stiffness_matrix(
        basis: skfem.Basis,
        potential_constraints: dict[str, tuple[float, NDArray]],
        mfc_transform: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """Assembles the stiffness matrix and applies the MFC transformation."""

        K = assembly.assemble_stiffness_matrix(
            basis,
            # TODO: Placeholder permittivity.
            epsilon_r=np.ones(basis.N),
        )

        # Apply the MFC transformation to the stiffness matrix.
        K_hat = mfc_transform @ K @ mfc_transform.T

        # Replace zeroed out diagonal elements with 1.
        diag = K_hat.diagonal()
        diag[diag == 0] = 1.0
        K_hat.setdiag(diag)

        # Enforce the Dirichlet constraints.
        skfem.enforce(
            A=K_hat,
            D={
                key: basis.get_dofs(nodes=inds)
                for key, (__, inds) in potential_constraints.items()
            },
            overwrite=True,
        )

        return K_hat

    def _enforce_potential_constraints(self, charge_density: NDArray) -> NDArray:
        """Enforces the Dirichlet constraints on the right-hand side vector."""

        for value, indices in self.potential_constraints.values():
            charge_density[indices] = value

        return charge_density

    @abstractmethod
    def solve(self, charge_density: NDArray, potential: NDArray = None) -> NDArray:
        """Solves the Poisson equation for a given charge density."""
        ...


class DirectPoissonSolver(PoissonSolver):
    """Direct solver for the Poisson equation."""

    def __init__(
        self,
        mesh: skfem.MeshTet,
        basis: skfem.Basis,
        structure: ase.Atoms,
        potential_constraints: dict[str, tuple[float, NDArray]],
        fixed_density: NDArray,
    ):
        """Initializes the direct Poisson solver."""
        super().__init__(mesh, basis, structure, potential_constraints)
        self.fixed_density = fixed_density
        self.preconditioner = sparse.diags(1 / self.stiffness_matrix.diagonal())

    def solve(self, charge_density: NDArray, potential: NDArray) -> NDArray:
        """Solves the Poisson equation for a given charge density."""
        charge_density = self._enforce_potential_constraints(
            -(charge_density + self.fixed_density) / epsilon_0
        )
        potential, __ = sparse.linalg.bicgstab(
            self.stiffness_matrix,
            charge_density,
            M=self.preconditioner,
        )
        return self.mfc_transform.T @ potential


class NonlinearPoissonSolver(PoissonSolver):
    """Solver for the nonlinear Poisson equation."""

    def __init__(
        self,
        mesh: skfem.MeshTet,
        basis: skfem.Basis,
        structure: ase.Atoms,
        potential_constraints: dict[str, tuple[float, NDArray]],
        fixed_density: NDArray,
        atom_inds: NDArray,
        density_model: DensityModel,
        max_iterations: int = 20,
        convergence_tol: float = 1e-3,
    ):
        """Initializes the nonlinear Poisson solver."""
        super().__init__(mesh, basis, structure, potential_constraints)
        self.basis = basis
        self.fixed_density = fixed_density
        self.atom_inds = atom_inds
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.density_model_factory = density_model

    def solve(self, charge_density: NDArray, potential: NDArray) -> NDArray:
        """Solves the nonlinear Poisson equation for a given charge density and initial potential guess."""

        density_model: DensityModel = self.density_model_factory(
            charge_density[self.atom_inds], potential[self.atom_inds]
        )

        new_potential = self._enforce_potential_constraints(np.copy(potential))

        density = np.zeros_like(charge_density)
        density_derivative = np.zeros_like(charge_density)

        for iteration in range(self.max_iterations):  # Max iterations
            density[self.atom_inds] = density_model.density(
                new_potential[self.atom_inds]
            )
            density_derivative[self.atom_inds] = density_model.density_derivative(
                new_potential[self.atom_inds]
            )

            residual = (
                self.stiffness_matrix @ new_potential
                + (density + self.fixed_density) / epsilon_0
            )
            jacobian = (
                self.stiffness_matrix + sparse.diags(density_derivative) / epsilon_0
            )

            # Enforce the potential constraints on the Jacobian and residual.
            skfem.enforce(
                A=jacobian,
                D={
                    key: self.basis.get_dofs(nodes=inds)
                    for key, (__, inds) in self.potential_constraints.items()
                },
                overwrite=True,
            )

            for value, indices in self.potential_constraints.values():
                residual[indices] = 0.0

            preconditioner = sparse.diags(1 / jacobian.diagonal())
            # Solve the linearized problem for the update.
            delta_potential, __ = sparse.linalg.bicgstab(
                jacobian,
                residual,
                M=preconditioner,
            )

            new_potential -= delta_potential

            if np.max(np.abs(delta_potential)) < self.convergence_tol:
                break

        else:  # Did not break, i.e. max_iterations reached.
            if comm.rank == 0:
                warnings.warn(
                    "Nonlinear solver did not converge within the maximum number of iterations.",
                    RuntimeWarning,
                )

        return self.mfc_transform.T @ new_potential
