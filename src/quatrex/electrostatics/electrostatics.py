# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Main class for handling electrostatics."""

import numpy as np

from qttools import NDArray
from qttools.comm import comm
from qttools.utils.mpi_utils import distributed_load
from quatrex.core.config import QuatrexConfig
from quatrex.electrostatics.meshing import DeviceMesh
from quatrex.electrostatics.solver import (
    DirectPoissonSolver,
    NonlinearPoissonSolver,
    PoissonSolver,
)


class ElectrostaticSolver:
    """Main class for handling the electrostatics.

    This class is responsible for setting up the electrostatic problem.
    It initializes the device mesh, configures the potential constraints
    and fixed charge density based on the device geometry and doping
    profiles, and sets up the Poisson solver according to the specified
    solving scheme. The `solve` method takes the charge density in the
    orbital basis and the potential from the previous iteration,
    transforms them to real space, and solves the Poisson equation to
    compute the new potential in real space, which is then transformed
    back to the orbital basis and returned.

    Parameters
    ----------
    config : QuatrexConfig
        The configuration object.

    """

    call_count = 0

    def __init__(self, config: QuatrexConfig):
        """Initializes the electrostatic solver."""

        self.config = config

        self.device_mesh = DeviceMesh.from_config(config)

        self.num_dofs = self.device_mesh.mesh.points.shape[0]
        self.atom_inds = self.device_mesh.region_node_inds["atoms"]

        # Configure gates and dopants that define the problem setup.
        potential_constraints, fixed_density = self._configure_constraints(
            config=config,
            device_mesh=self.device_mesh,
        )

        self.poisson_solver = self._configure_poisson_solver(
            config=config,
            device_mesh=self.device_mesh,
            potential_constraints=potential_constraints,
            fixed_density=fixed_density,
        )

        # TODO: To get the actual real-space charge density from the
        # orbital occupations, we need the basis functions.
        # self.orbital_basis = ...

        self.initial_guess_strategy = config.electrostatics.initial_guess

    @staticmethod
    def _configure_constraints(
        config: QuatrexConfig,
        device_mesh: DeviceMesh,
    ) -> tuple[dict[str, tuple[float, NDArray]], NDArray]:
        """Configures the constraints for the electrostatic problem."""

        potential_constraints = {}
        for region in config.device.geometry.regions:
            if not hasattr(region.properties, "voltage"):
                continue

            potential_constraints[region.name] = (
                region.properties.voltage,
                device_mesh.region_node_inds[region.name],
            )

        # in nm^3
        volume_per_atom = device_mesh.structure.get_volume() / len(
            device_mesh.structure
        )
        fixed_density = np.zeros(device_mesh.mesh.points.shape[0])
        for region in config.device.geometry.regions:
            if not hasattr(region.properties, "donor_concentration") or not hasattr(
                region.properties, "acceptor_concentration"
            ):
                continue
            if (
                region.properties.donor_concentration is None
                or region.properties.acceptor_concentration is None
            ):
                continue

            inds = device_mesh.region_node_inds[region.name]
            inds = np.intersect1d(
                inds, device_mesh.region_node_inds["atoms"], assume_unique=True
            )
            fixed_density[inds] = (
                (
                    # These are given in m^-3.
                    region.properties.donor_concentration
                    - region.properties.acceptor_concentration
                )
                * 1e-27  # Convert from m^-3 to nm^-3.
                * volume_per_atom
            )

        return potential_constraints, fixed_density

    @staticmethod
    def _configure_poisson_solver(
        config: QuatrexConfig,
        device_mesh: DeviceMesh,
        potential_constraints: dict[str, tuple[float, NDArray]],
        fixed_density: NDArray,
    ) -> PoissonSolver:
        """Configures the Poisson solver for the electrostatic solver.

        Parameters
        ----------
        config : QuatrexConfig
            The configuration object containing the Poisson solver settings.

        Returns
        -------
        PoissonSolver
            The configured Poisson solver.

        """
        if config.electrostatics.solving_scheme == "direct":
            return DirectPoissonSolver(
                config=config,
                device_mesh=device_mesh,
                potential_constraints=potential_constraints,
                fixed_density=fixed_density,
            )
        if config.electrostatics.solving_scheme == "root-finding":
            return NonlinearPoissonSolver(
                config=config,
                device_mesh=device_mesh,
                potential_constraints=potential_constraints,
                fixed_density=fixed_density,
            )

        raise ValueError(
            f"Unknown solving scheme: {config.electrostatics.solving_scheme}"
        )

    def generate_initial_guess(self) -> NDArray:
        """Provides an initial guess for the potential."""
        if self.initial_guess_strategy == "zero":
            if comm.rank == 0:
                print("using zero initial guess", flush=True)
            return np.zeros(self.atom_inds.size)

        if self.initial_guess_strategy == "constraints":
            # TODO: This is currently a very bad initial guess. The
            # contact region voltages should be taken into account, and
            # the fixed charges should be neglected.
            if comm.rank == 0:
                print("using initial guess from constraints", flush=True)
            from scipy import sparse

            rhs = self.poisson_solver._enforce_potential_constraints(
                np.zeros(self.num_dofs)
            )
            potential, info = sparse.linalg.bicgstab(
                self.poisson_solver.stiffness_matrix,
                rhs,
                # Small perturbation to avoid zero initial guess.
                x0=np.zeros(self.num_dofs) + 1e-6,
                M=sparse.diags(1 / self.poisson_solver.stiffness_matrix.diagonal()),
            )

            return (self.poisson_solver.mfc_transform.T @ potential)[self.atom_inds]

        if self.initial_guess_strategy == "file":
            if comm.rank == 0:
                print("using initial guess from file", flush=True)
            potential = distributed_load(self.config.input_dir / "potential.npy")
            return potential

        raise ValueError(f"Unknown initial guess type: {self.initial_guess_strategy}")

    def solve(self, charge_density: NDArray, potential: NDArray) -> NDArray:
        """Solves the Poisson equation for a given charge density.

        Parameters
        ----------
        charge_density : NDArray
            The charge density in the orbital basis.
        potential : NDArray
            The potential in the orbital basis from the previous
            iteration.

        Returns
        -------
        potential : NDArray
            The computed potential in the orbital basis.

        """
        self.call_count += 1

        real_space_charge_density = np.zeros(self.num_dofs)
        real_space_charge_density[self.atom_inds] = charge_density

        if comm.rank == 0:
            np.save(
                self.config.output_dir
                / f"real_space_charge_density_{self.call_count}.npy",
                real_space_charge_density,
            )

        real_space_potential = np.zeros(self.num_dofs)
        real_space_potential[self.atom_inds] = potential

        potential = self.poisson_solver.solve(
            real_space_charge_density, real_space_potential
        )

        if comm.rank == 0:
            np.save(
                self.config.output_dir / f"real_space_potential_{self.call_count}.npy",
                potential,
            )

        return potential[self.atom_inds]
