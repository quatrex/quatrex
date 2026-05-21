# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
import functools
import warnings
from abc import ABC, abstractmethod

import numpy as np
import skfem

from qttools import NDArray, sparse
from qttools.comm import comm
from quatrex.core.config import ElectrostaticsConfig, QuatrexConfig
from quatrex.core.constants import epsilon_0
from quatrex.electrostatics import assembly
from quatrex.electrostatics.density_response import (
    DensityModel,
    OMENDensityModel,
    SingleBandDensityModel,
)
from quatrex.electrostatics.meshing import DeviceMesh


class PoissonSolver(ABC):
    """Abstract base class for Poisson solvers."""

    def __init__(
        self,
        config: QuatrexConfig,
        device_mesh: DeviceMesh,
        potential_constraints: dict[str, tuple[float, NDArray]],
    ):
        """Initializes the Poisson solver."""

        mesh, self.basis = assembly.initialize_tetrahedral_mesh(device_mesh.mesh)

        self.mfc_transform = assembly.assemble_mfc_transform(
            mesh,
            self.basis,
            device_mesh.structure.cell,
            device_mesh.structure.pbc,
        )

        self.potential_constraints = potential_constraints

        epsilon_r = np.ones_like(mesh.p[0]) * config.electrostatics.default_epsilon_r
        for region in config.device.geometry.regions:

            region_epsilon_r = getattr(region.properties, "epsilon_r", None)
            if region_epsilon_r is None:
                continue

            inds = device_mesh.region_node_inds[region.name]
            epsilon_r[inds] = region_epsilon_r

        self.stiffness_matrix = self._assemble_stiffness_matrix(
            basis=self.basis,
            epsilon_r=epsilon_r,
            potential_constraints=self.potential_constraints,
            mfc_transform=self.mfc_transform,
        )

    @staticmethod
    def _assemble_stiffness_matrix(
        basis: skfem.Basis,
        epsilon_r: NDArray,
        potential_constraints: dict[str, tuple[float, NDArray]],
        mfc_transform: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """Assembles the stiffness matrix and applies the MFC transformation."""

        K = assembly.assemble_stiffness_matrix(basis, epsilon_r=epsilon_r)

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
        config: QuatrexConfig,
        device_mesh: DeviceMesh,
        potential_constraints: dict[str, tuple[float, NDArray]],
        fixed_density: NDArray,
    ):
        """Initializes the direct Poisson solver."""
        super().__init__(
            config=config,
            device_mesh=device_mesh,
            potential_constraints=potential_constraints,
        )
        self.fixed_density = fixed_density
        self.preconditioner = sparse.diags(1 / self.stiffness_matrix.diagonal())

    def solve(
        self,
        charge_density: NDArray,
        potential: NDArray,
    ) -> NDArray:
        """Solves the Poisson equation for a given charge density."""
        charge_density = self._enforce_potential_constraints(
            -(self.fixed_density - charge_density) / epsilon_0
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
        config: QuatrexConfig,
        device_mesh: DeviceMesh,
        potential_constraints: dict[str, tuple[float, NDArray]],
        fixed_density: NDArray,
    ):
        """Initializes the nonlinear Poisson solver."""
        super().__init__(
            config=config,
            device_mesh=device_mesh,
            potential_constraints=potential_constraints,
        )

        self.atom_inds = device_mesh.region_node_inds["atoms"]

        self.fixed_density = fixed_density

        self.max_iterations = config.electrostatics.max_iterations
        self.convergence_tol = config.electrostatics.convergence_tol

        self.initialize_density_model = self._configure_density_model(
            config.electrostatics
        )

    @staticmethod
    def _configure_density_model(electrostatics_config: ElectrostaticsConfig):
        """Configures the density model based on the provided configuration."""
        if electrostatics_config.density_model == "omen":
            return OMENDensityModel
        if electrostatics_config.density_model == "single-band":
            return functools.partial(
                SingleBandDensityModel,
                dim=electrostatics_config.density_model_dim,
            )
        raise ValueError(
            f"Unknown density model: {electrostatics_config.density_model}"
        )

    def solve(
        self,
        charge_density: NDArray,
        potential: NDArray,
        initial_density_derivative: NDArray | None = None,
    ) -> NDArray:
        """Solves the nonlinear Poisson equation.

        Parameters
        ----------
        charge_density : NDArray
            The charge density.
        potential : NDArray
            The initial potential guess.
        initial_density_derivative : NDArray, optional
            An optional initial guess for the density derivative, used
            for the first iteration. This is especially useful for the
            QTBM solver, where a first density derivative can be
            computed analytically using Fermi-Dirac statistics. If not
            provided, the density derivative will be computed from the
            density model in the first iteration.

        Returns
        -------
        potential : NDArray
            The converged potential after solving the nonlinear Poisson
            equation.

        """

        density_model: DensityModel = self.initialize_density_model(
            charge_density[self.atom_inds], potential[self.atom_inds]
        )

        new_potential = self._enforce_potential_constraints(np.copy(potential))

        density = np.zeros_like(charge_density)
        density_derivative = np.zeros_like(charge_density)

        for iteration in range(self.max_iterations):
            if comm.rank == 0:
                print(f"Nonlinear Poisson iteration {iteration}", flush=True)
            use_initial_values = (
                initial_density_derivative is not None and iteration == 0
            )
            if use_initial_values:
                # If there is an initial density derivative provided,
                # use it for the first iteration. Otherwise, compute it
                # from the density model.
                computed_density = charge_density[self.atom_inds]
                computed_density_derivative = initial_density_derivative[self.atom_inds]
            else:
                if comm.rank == 0:
                    print(
                        "    Computing density and density derivative from the density model.",
                        flush=True,
                    )
                computed_density = density_model.density(new_potential[self.atom_inds])
                computed_density_derivative = density_model.density_derivative(
                    new_potential[self.atom_inds]
                )

            density[self.atom_inds] = computed_density
            density_derivative[self.atom_inds] = computed_density_derivative

            if comm.rank == 0:
                print("    Assembling residual and Jacobian.", flush=True)
            residual = (
                self.stiffness_matrix @ new_potential
                + (self.fixed_density - density) / epsilon_0
            )
            jacobian = (
                self.stiffness_matrix - sparse.diags(density_derivative) / epsilon_0
            )

            # Enforce potential constraints on Jacobian and residual.
            if comm.rank == 0:
                print("    Enforcing potential constraints.", flush=True)
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

            if comm.rank == 0:
                print("    Solving linearized problem.", flush=True)
            # Solve the linearized problem for the update.
            delta_potential, __ = sparse.linalg.bicgstab(
                jacobian,
                residual,
                M=preconditioner,
            )

            new_potential -= delta_potential

            if comm.rank == 0:
                print(
                    "    Residual norm: {:.6e}".format(np.max(np.abs(delta_potential))),
                    flush=True,
                )
            if np.max(np.abs(delta_potential)) < self.convergence_tol:
                break

        else:  # Did not break, i.e. max_iterations reached.
            if comm.rank == 0:
                warnings.warn(
                    "Nonlinear solver did not converge within the maximum number of iterations.",
                    RuntimeWarning,
                )

        return self.mfc_transform.T @ new_potential
