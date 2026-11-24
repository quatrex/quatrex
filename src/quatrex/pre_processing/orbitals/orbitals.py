# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""This module contains classes and functions for handling real-space
orbitals."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import ase.io
import numpy as np
import pyvista as pv
from rich.pretty import pprint
from scipy import sparse

from qttools import NDArray
from quatrex.core.config import QuatrexConfig
from quatrex.pre_processing.orbitals.remeshing import remesh_for_orbitals


@dataclass(frozen=True)
class RemeshingOptions:
    """Options concerning the remeshing for orbital projection."""

    initial_spacing: float = 2.0
    """The initial grid spacing for the orbital projections.

    This value determines the resolution of the initial mesh before any
    remeshing is applied.

    """

    padding: float = 5.0
    """Padding around the structure for the initial grid.

    This ensures that the orbital projections are well captured,
    especially near the boundaries of the simulation domain.

    """
    max_node_distance_from_atoms: float = 7.5
    """Maximum distance from atoms to consider for orbital projections.

    Points that are further away than this distance from any atom will
    be dropped.

    """

    duplicate_tolerance: float = 1e-4
    """Tolerance for deduplicating points in the mesh."""


DEFAULT_REMESHING_OPTIONS = RemeshingOptions()


class OrbitalEvaluator(ABC):
    """Abstract base class for real-space orbitals.

    This class defines the interface for evaluating orbitals, their
    gradients, and Hessians at given positions in real space. Subclasses
    must implement these methods to provide specific functionality for
    different types of orbitals, such as those from CP2K or Wannier90.

    """

    @abstractmethod
    def evaluate_orbitals(self, positions: NDArray) -> NDArray:
        """Evaluate the values of the orbitals at the given positions.

        Parameters
        ----------
        positions : NDArray
            The positions at which to evaluate the orbital values.

        Returns
        -------
        orbitals : NDArray
            The evaluated orbitals at the given positions. The shape of
            the output will be (M, N), where M is the total number of
            orbitals and N is the number of points.

        """
        ...

    @abstractmethod
    def evaluate_orbital_gradients(self, positions: NDArray) -> NDArray:
        """Evaluate gradients of the orbitals at the given positions.

        Parameters
        ----------
        positions : NDArray
            The positions at which to evaluate the orbital gradients.

        Returns
        -------
        gradients : NDArray
            The evaluated orbital gradients at the given positions. The
            shape of the output will be (M, N, 3), where M is the total
            number of orbitals, N is the number of points, and 3
            corresponds to the x, y, z components of the gradient.

        """
        ...

    @abstractmethod
    def evaluate_orbital_hessians(self, positions: NDArray) -> NDArray:
        """Evaluate Hessians of the orbitals at the given positions.

        Parameters
        ----------
        positions : NDArray
            The positions at which to evaluate the orbital Hessians.

        Returns
        -------
        hessians : NDArray
            The evaluated orbital Hessians at the given positions. The
            shape of the output will be (M, N, 3, 3), where M is the
            total number of orbitals, N is the number of points 3x3
            corresponds to the Hessian matrix.

        """
        ...


def _deduplicate_points(points: NDArray, atol: float) -> NDArray:
    """Deduplicate points in a 3D array based on a tolerance.

    Parameters
    ----------
    points : NDArray
        A 2D array of shape (N, 3) representing N points in 3D space.
    atol : float
        The tolerance for deduplication. Points that are closer than
        this distance will be considered duplicates.

    Returns
    -------
    unique_points : NDArray
        A 2D array of shape (M, 3) containing the unique points, where M
        <= N.

    """
    keys = np.floor((points - points.min(axis=0)) / atol).astype(np.int64)
    dims = keys.max(axis=0) + 1
    # NOTE: Requires that prod(dims) does not exceed 2**63, which is
    # almost always true in practice.
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    __, idx = np.unique(flat, return_index=True)
    return points[np.sort(idx)]


class DeviceOrbitalProjector(ABC):
    """Class for handling orbital projections.

    This class is responsible for managing the remeshing and evaluation
    of orbital projections based on the provided configuration.

    Parameters
    ----------
    config : QuatrexConfig
        The main quatrex configuration.

    """

    def __init__(self, config: QuatrexConfig):
        """Initialize the orbital projections from a config."""
        self.config = config

        if config.material_model != "atomistic":
            raise ValueError(
                "Orbital projections are only applicable to `atomistic` "
                "material descriptions."
            )

        try:
            self.structure = ase.io.read(config.input_dir / "structure.xyz")
            self.structure.center()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Structure file 'structure.xyz' not found in {config.input_dir}. "
                "Please ensure the structure file is present for atomistic simulations."
            )

        self.orbital_evaluator = self._init_orbital_evaluator(config, self.structure)

    @staticmethod
    @abstractmethod
    def _init_orbital_evaluator(
        config: QuatrexConfig, structure: ase.Atoms
    ) -> OrbitalEvaluator:
        """Initialize the real-space orbitals based on the configuration.

        Parameters
        ----------
        config : QuatrexConfig
            The main quatrex configuration.
        structure : ase.Atoms
            The atomic structure for which the orbitals are to be
            initialized.

        Returns
        -------
        orbitals : RealSpaceOrbitals
            An instance of a subclass of RealSpaceOrbitals, initialized
            based on the specified orbital projection method in the
            configuration.

        """
        ...

    @abstractmethod
    def get_initial_mesh(
        self,
        options: RemeshingOptions | None = None,
    ) -> pv.UnstructuredGrid:
        """Get an initial mesh guess for the orbital projections.

        Parameters
        ----------
        options : RemeshingOptions, optional
            Options concerning the remeshing for orbital projection.
            Uses default values if not provided.

        Returns
        -------
        initial_mesh : pv.UnstructuredGrid
            The initial mesh to be used for remeshing.

        """
        ...

    def _filter_nodes(
        self, mesh_nodes: NDArray, options: RemeshingOptions | None = None
    ) -> NDArray:
        """Filter the nodes of the orbital mesh.

        Three main filtering criteria are applied:

        1. Only nodes that are within a certain distance from the atoms
        are kept. This ensures that the orbital projections are relevant
        to the atomic structure.
        2. Nodes that are too close to the atoms are removed to avoid
        issues with gmsh.
        3. Nodes that are not contained within the structure cell or sit
        on the cell boundary are removed to ensure that the fragment
        call in gmsh does not fail.

        Parameters
        ----------
        mesh_nodes : NDArray
            The nodes of the orbital mesh to be filtered.
        options : RemeshingOptions, optional
            Options concerning the remeshing for orbital projection.
            Uses default values if not provided.

        Returns
        -------
        filtered_nodes : NDArray
            The nodes of the orbital mesh that pass the filtering
            criteria.

        """
        options = options or DEFAULT_REMESHING_OPTIONS

        device_structure = self.structure.copy()
        if self.config.device.construct_from_unit_cell:
            transport_ind = "abc".index(self.config.device.transport_direction)
            num_repetitions = (
                self.config.device.num_transport_cells
                * self.config.device.neighbor_cell_cutoff[transport_ind]
            )
            repeats = [num_repetitions if i == transport_ind else 1 for i in range(3)]
            device_structure = device_structure.repeat(repeats)
            device_structure.pbc[transport_ind] = False

        nodes_to_keep = []
        for atom_position in device_structure.positions:
            distances = np.linalg.norm(mesh_nodes - atom_position, axis=1)
            # Extract the node positions that are within a certain distance
            # from the atoms.
            nodes_within_cutoff = np.where(
                distances < options.max_node_distance_from_atoms
            )[0]

            # Remove points too close to the atoms.
            nodes_on_atoms = np.where(distances < options.duplicate_tolerance)[0]

            nodes_to_keep.append(np.setdiff1d(nodes_within_cutoff, nodes_on_atoms))

        nodes_to_keep = np.unique(np.concatenate(nodes_to_keep))

        fractional_coords = (
            np.linalg.inv(device_structure.get_cell()) @ mesh_nodes.T
        ).T
        inside_cell = np.where(
            np.all(
                (fractional_coords > options.duplicate_tolerance)
                & (fractional_coords < 1 - options.duplicate_tolerance),
                axis=1,
            )
        )[0]

        nodes_to_keep = np.intersect1d(nodes_to_keep, inside_cell)

        filtered_nodes = _deduplicate_points(
            mesh_nodes[nodes_to_keep], options.duplicate_tolerance
        )

        return filtered_nodes

    def remesh(self) -> NDArray:
        """Generate a new mesh for the orbitals.

        Returns
        -------
        orbital_nodes : NDArray
            The nodes of the remeshed orbital projections that should
            get included in the device mesh.

        """
        orbital_mesh, remesh_info = remesh_for_orbitals(
            self.get_initial_mesh(),
            self.orbital_evaluator.evaluate_orbital_hessians,
        )

        pprint(remesh_info)

        orbital_nodes = orbital_mesh.points

        if self.config.device.construct_from_unit_cell:
            transport_ind = "abc".index(self.config.device.transport_direction)
            num_repetitions = (
                self.config.device.num_transport_cells
                * self.config.device.neighbor_cell_cutoff[transport_ind]
            )

            orbital_nodes = np.concatenate(
                [
                    orbital_nodes + i * self.structure.get_cell()[transport_ind]
                    for i in range(num_repetitions)
                ],
                axis=0,
            )

        orbital_nodes = self._filter_nodes(orbital_nodes)

        return orbital_nodes

    @abstractmethod
    def assemble_projections(
        self, mesh_nodes: NDArray
    ) -> tuple[sparse.csr_matrix, list[sparse.csr_matrix]]:
        """Evaluate the orbital projections on the given device mesh.

        Parameters
        ----------
        mesh_nodes : NDArray
            A 2D array of shape (N, 3) containing the coordinates of the
            mesh nodes in real space, where N is the number of nodes.

        Returns
        -------
        projections : sparse.csr_matrix
            A sparse matrix containing the evaluated basis functions on
            the mesh. Each row corresponds to a point in the device
            mesh, and each column corresponds to a basis function.
        gradient_matrices : list of sparse.csr_matrix
            A list of three sparse matrices, each corresponding to the
            gradient of the basis functions in the x, y, and z
            directions, respectively. Each matrix has the same shape as
            `projections`, with rows corresponding to mesh nodes and
            columns corresponding to basis functions.

        """
        ...
