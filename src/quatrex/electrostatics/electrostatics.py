# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Main class for handling electrostatics."""

import numpy as np
import skfem

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.utils.gpu_utils import get_device, get_host
from qttools.utils.mpi_utils import distributed_load
from quatrex.core.config import QuatrexConfig
from quatrex.core.qtbm import QTBM
from quatrex.core.scba import SCBA
from quatrex.electrostatics.geometry_config import VolumeProperties
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

    def __init__(self, config: QuatrexConfig, transport_solver: QTBM | SCBA):
        """Initializes the electrostatic solver."""

        self.config = config

        self.device_mesh = DeviceMesh.from_config(config)

        self.num_dofs = self.device_mesh.mesh.points.shape[0]
        self.atom_inds = xp.asarray(self.device_mesh.region_node_inds["atoms"])

        # Configure gates and dopants that define the problem setup.
        potential_constraints, fixed_density = self._configure_constraints(
            config=config,
            device_mesh=self.device_mesh,
            transport_solver=transport_solver,
        )

        self.initial_guess_strategy = config.electrostatics.initial_guess

        if self.initial_guess_strategy == "constraints":
            # NOTE: When solving an initial guess from constraints, we
            # impose Dirichlet potential constraints in the contacts as
            # well, not only in the gates.
            self.contact_potential_constraints = (
                self._configure_contact_potential_constraints(
                    config=config,
                    atom_inds=self.atom_inds,
                    transport_solver=transport_solver,
                )
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

    @staticmethod
    def _configure_constraints(
        config: QuatrexConfig,
        device_mesh: DeviceMesh,
        transport_solver: QTBM | SCBA,
    ) -> tuple[dict[str, tuple[float, NDArray]], NDArray]:
        """Configures the constraints for the electrostatic problem."""

        # Determine the reference contact. Needed to determine the
        # reference conduction band edge - Fermi level difference for
        # the potential constraints.
        if isinstance(transport_solver, QTBM):
            for contact in transport_solver.device.contacts:
                if contact.voltage == 0.0:
                    delta_fermi_level_conduction_band = (
                        contact.delta_fermi_level_conduction_band
                    )
                    break

        elif isinstance(transport_solver, SCBA):
            if config.electron.left_contact.voltage == 0.0:
                delta_fermi_level_conduction_band = (
                    transport_solver.electron_solver.left_delta_fermi_level_conduction_band
                )
            if config.electron.right_contact.voltage == 0.0:
                delta_fermi_level_conduction_band = (
                    transport_solver.electron_solver.right_delta_fermi_level_conduction_band
                )

        potential_constraints = {}
        for region in config.device.geometry.regions:
            if not hasattr(region.properties, "voltage"):
                continue

            constraint_value = -region.properties.voltage
            if (
                config.electrostatics.electron_affinity is not None
                and region.properties.work_function is not None
            ):
                # TODO: Fix this to include the fermi level to
                # conduction band edge difference as well.
                constraint_value += (
                    region.properties.work_function
                    - config.electrostatics.electron_affinity
                    - delta_fermi_level_conduction_band
                )
                if comm.rank == 0:
                    print(
                        f"Adding work function difference to potential constraint for region {region.name}: {constraint_value} V",
                        flush=True,
                    )

            potential_constraints[region.name] = (
                constraint_value,
                device_mesh.region_node_inds[region.name],
            )

        volume_per_atom = device_mesh.structure.get_volume() / len(
            device_mesh.structure
        )
        fixed_density = np.zeros(device_mesh.mesh.points.shape[0])
        for region in config.device.geometry.regions:
            if not isinstance(region.properties, VolumeProperties):
                continue

            if (
                region.properties.donor_concentration == 0.0
                and region.properties.acceptor_concentration == 0.0
            ):
                continue

            inds = device_mesh.region_node_inds[region.name]
            inds = np.intersect1d(
                inds, device_mesh.region_node_inds["atoms"], assume_unique=True
            )

            # NOTE: Doping concentrations are given in cm^-3, so we
            # convert to Å^-3 below.
            doping_concentration = 1e-24 * (
                region.properties.donor_concentration
                - region.properties.acceptor_concentration
            )
            fixed_density[inds] = doping_concentration * volume_per_atom

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

    @staticmethod
    def _configure_contact_potential_constraints(
        config: QuatrexConfig,
        atom_inds: NDArray,
        transport_solver: QTBM | SCBA,
    ) -> dict[str, tuple[float, NDArray]]:
        """Gets the potential constraints for the contacts based on the
        transport solver.

        This is used when generating an initial guess for the potential
        by solving the Poisson equation with the constraints.

        Parameters
        ----------
        config : QuatrexConfig
            The configuration object containing the contact settings.
        atom_inds : NDArray
            The indices of the mesh nodes corresponding to the atomic
            positions, which is needed to determine where to apply the
            contact potential constraints.
        transport_solver : QTBM | SCBA
            The transport solver, which may be needed to determine the
            contact potential constraints.

        Returns
        -------
        contact_potential_constraints : dict[str, tuple[float, NDArray]]
            A dictionary mapping contact names to tuples of (potential
            value, indices of the nodes where the constraint is applied).

        """

        if isinstance(transport_solver, QTBM):
            # In the wavefunction formalism we go through all contacts
            # and gather their potentials.
            contact_potential_constraints = {}
            for contact in transport_solver.device.contacts:
                contact_inds = atom_inds[contact.origin_atom_indices]
                contact_potential_constraints[contact.name] = (
                    -contact.voltage,
                    contact_inds,
                )

            return contact_potential_constraints

        # In NEGF the contacts are the first and last blocks of
        # orbitals.
        if isinstance(config.device.block_size, int):
            left_block_size = right_block_size = config.device.block_size
        else:
            left_block_size = config.device.block_size[0]
            right_block_size = config.device.block_size[-1]

        left_contact_inds = atom_inds[:left_block_size]
        right_contact_inds = atom_inds[-right_block_size:]

        contact_potential_constraints = {
            config.electron.left_contact.name: (
                -config.electron.left_contact.voltage,
                left_contact_inds,
            ),
            config.electron.right_contact.name: (
                -config.electron.right_contact.voltage,
                right_contact_inds,
            ),
        }
        return contact_potential_constraints

    def generate_initial_guess(self) -> NDArray:
        """Provides an initial guess for the potential.

        The initial guess can be generated based on different strategies
        specified in the configuration, such as using a zero potential,
        loading from a file, or solving an initial guess from the
        constraints.

        Parameters
        ----------
        transport_solver : TransportSolver
            The transport solver, which may be needed to solve an
            initial guess from the constraints.

        Returns
        -------
        potential : NDArray
            The initial guess for the potential in the orbital basis.

        """
        if self.initial_guess_strategy == "zero":
            if comm.rank == 0:
                print("using zero initial guess", flush=True)
            return xp.zeros(self.atom_inds.size)

        if self.initial_guess_strategy == "file":
            if comm.rank == 0:
                print("using initial guess from file", flush=True)
            potential = distributed_load(self.config.input_dir / "potential.npy")
            return potential

        if self.initial_guess_strategy == "constraints":
            if comm.rank == 0:
                print("using initial guess from constraints", flush=True)

            extended_potential_constraints = (
                self.poisson_solver.potential_constraints
                | self.contact_potential_constraints
            )

            rhs = np.zeros(self.num_dofs)
            for value, indices in extended_potential_constraints.values():
                rhs[indices] = value

            stiffness_matrix = skfem.enforce(
                A=self.poisson_solver.stiffness_matrix,
                D={
                    key: self.poisson_solver.basis.get_dofs(nodes=inds)
                    for key, (__, inds) in extended_potential_constraints.items()
                },
            )

            diagonal = stiffness_matrix.diagonal()
            preconditioner = sparse.linalg.LinearOperator(
                stiffness_matrix.shape, matvec=lambda x: x / diagonal
            )
            potential, info = sparse.linalg.bicgstab(
                stiffness_matrix,
                rhs,
                M=preconditioner,
                # NOTE: Zero starting guess can lead to parameter
                # breakdown, this seems more robust.
                x0=preconditioner @ rhs,
            )

            return (self.poisson_solver.mfc_transform.T @ potential)[self.atom_inds]

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

        real_space_charge_density = xp.zeros(self.num_dofs)
        real_space_charge_density[self.atom_inds] = charge_density

        if comm.rank == 0:
            xp.save(
                self.config.output_dir
                / f"real_space_charge_density_{self.call_count}.npy",
                real_space_charge_density,
            )

        real_space_potential = xp.zeros(self.num_dofs)
        real_space_potential[self.atom_inds] = potential

        potential = self.poisson_solver.solve(
            get_host(real_space_charge_density), get_host(real_space_potential)
        )
        potential = get_device(potential)

        if comm.rank == 0:
            xp.save(
                self.config.output_dir / f"real_space_potential_{self.call_count}.npy",
                potential,
            )

        return potential[self.atom_inds]
