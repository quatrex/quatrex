# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the base contact class."""

from abc import ABC, abstractmethod

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_host
from quatrex.bandstructure.contact import (
    contact_band_edges,
    contact_doping_density,
    contact_fermi_level,
)
from quatrex.contact.discovery import real_space_discovery, simplified_discovery
from quatrex.core.config import ContactConfig

profiler = Profiler()


class BaseContact(ABC):
    """Class representing a contact for QTBM calculations.

    Parameters
    ----------
    device : BaseDevice
        The device object to which this contact is attached. Contains
        the Hamiltonian, overlap matrices, and atomic structure
        information.
    contact_config : ContactConfig
        The configuration object containing the contact settings such as
        lattice vectors, origin, transport direction, and Fermi level
        information.
    sparsity_pattern : sparse.spmatrix
        The sparsity pattern of the device Hamiltonian, used to identify
        the contact orbitals and their connectivity.

    Attributes
    ----------
    device : BaseDevice
        The device object to which this contact is attached.
    name : str
        The contact identifier.
    transport_direction : int
        Transport direction index (0, 1, or 2).
    unit_cell_orbital_indices : dict
        Dict of orbital indices for each contact cell indexed by (i, j,
        k) tuples.
    origin_key : tuple[int, int, int]
        The key corresponding to the origin cell in the
        unit_cell_orbital_indices.
    transverse_repetition_grid: NDArray
        Number of periodic repetitions in the two transverse directions.
    transport_repetitions : int
        Number of repetitions needed in transport direction for
        convergence.
    orbital_indices : NDArray
        Flattened array of orbital indices for the contact, sorted first
        in transport direction, then in transverse directions.
    orbital_indices_per_layer : list[NDArray]
        List of orbital indices for each layer in the transport
        direction, sorted first in transverse directions, then in
        transport direction.
    transverse_to_transport_indices : NDArray
        Indices to reorder the coupling matrix from transverse-first to
        transport-first ordering.
    fermi_level : float
        Fermi level of the contact in eV.
    mid_gap_energy : float
        Mid-gap energy of the contact in eV.
    conduction_band_edge : float
        Energy of the conduction band edge in eV.
    voltage : float
        Voltage applied to the contact in V.
    temperature : float
        Temperature of the contact in K.

    """

    def __init__(
        self,
        device,
        contact_config: ContactConfig,
        sparsity_pattern: sparse.spmatrix,
    ):
        """Initializes the contact object."""

        if comm.rank == 0:
            print(f"Initializing contact '{contact_config.name}'...", flush=True)

        self.device = device
        self.contact_config = contact_config
        if contact_config.transport_direction not in ["a", "b", "c"]:
            raise ValueError("Direction must be one of 'a', 'b', or 'c'.")

        self.name = contact_config.name
        self.transport_direction = "abc".index(contact_config.transport_direction)

        if contact_config._contact_finder_method == "real_space":
            self.unit_cell_orbital_indices, repetition_grid, self.origin_key = (
                real_space_discovery(
                    hamiltonian=sparsity_pattern,
                    atomic_species=device.atomic_species,
                    atom_coordinates=device.atom_coordinates,
                    orbital_offsets=device.orbital_offsets,
                    contact_config=contact_config,
                )
            )
        elif contact_config._contact_finder_method == "from_unit":
            self.unit_cell_orbital_indices, repetition_grid, self.origin_key = (
                simplified_discovery(
                    num_orbitals=len(device.orbital_coordinates),
                    device_config=device.config.device,
                    contact_config=contact_config,
                )
            )
        else:
            raise NotImplementedError(
                f"Contact finder method '{contact_config._contact_finder_method}' not implemented."
            )

        self.transport_repetitions = repetition_grid[self.transport_direction]
        self.transverse_repetition_grid = (
            repetition_grid[: self.transport_direction]
            + repetition_grid[self.transport_direction + 1 :]
        )

        if comm.rank == 0:
            print(
                f"    Number of repetitions in transport direction: {self.transport_repetitions}",
                flush=True,
            )

        ny, nz = self.transverse_repetition_grid

        # Orbitals for contact (where to apply the OBC)
        # Sorted first in transport direction, then in transverse directions
        self.orbital_indices = np.concatenate(
            [
                self.unit_cell_orbital_indices[i, j, k]
                for j, k, i in np.ndindex(ny, nz, self.transport_repetitions)
            ]
        )
        # When getting the coupling matrix (01) for spill over,
        # it is more efficient to have it sorted first in transverse, then in transport
        # The orbital list is then different.
        # We keep it separated over slice over transport direction.
        self.orbital_indices_per_layer = [
            np.concatenate(
                [self.unit_cell_orbital_indices[i, j, k] for j, k in np.ndindex(ny, nz)]
            )
            for i in range(self.transport_repetitions + 1)
        ]

        # We then need to sort the 10 matrix to have the same ordering as the contact OBCs
        origin_num_orbitals = len(self.unit_cell_orbital_indices[self.origin_key])
        self.transverse_to_transport_indices = np.concatenate(
            [
                np.arange(origin_num_orbitals)
                + i * origin_num_orbitals
                + k * origin_num_orbitals * ny * nz
                for i in range(ny * nz)
                for k in range(self.transport_repetitions)
            ],
            dtype=int,
        )[None, :]

        self.fermi_level = contact_config.fermi_level
        self.mid_gap_energy = contact_config.mid_gap_energy
        self.conduction_band_edge = contact_config.conduction_band_edge
        self.voltage = contact_config.voltage
        self.temperature = contact_config.temperature

        if contact_config._contact_finder_method == "real_space":
            lattice_vectors = contact_config.lattice_vectors
        elif contact_config._contact_finder_method == "from_unit":
            lattice_vectors = device.lattice_vectors
        else:
            raise NotImplementedError(
                f"Contact finder method '{contact_config._contact_finder_method}' not implemented."
            )

        self.cell_volume = np.abs(np.linalg.det(lattice_vectors)) * np.prod(
            repetition_grid
        )

        if comm.rank == 0:
            print(f"    Fermi level: {self.fermi_level} eV", flush=True)
            print(f"    Mid-gap energy: {self.mid_gap_energy} eV", flush=True)
            print(
                f"    Conduction band edge: {self.conduction_band_edge} eV",
                flush=True,
            )
            print(f"    Voltage: {self.voltage} V", flush=True)
            print(f"    Temperature: {self.temperature} K", flush=True)

    @abstractmethod
    def compute_contact_bandstructure(
        self,
        kpoints_transport: NDArray,
    ) -> NDArray:
        """Computes the band structure for the contact along the
        transport direction.

        Parameters
        ----------
        kpoints_transport : NDArray
            The k-points along the transport direction.

        Returns
        -------
        e_k : NDArray
            The eigenvalues for the contact band structure.

        """
        pass

    def compute_contact_band_properties(
        self,
    ) -> tuple[float, float, float]:
        """Computes the Fermi level for the contact from the Hamiltonian and
        overlap matrices.

        Returns
        -------
        fermi_level : float
            The computed Fermi level in eV.
        mid_gap_energy : float
            The recomputed mid-gap energy based on the band structure.
        conduction_band_edge : float
            The energy of the conduction band edge in eV.

        """
        contact_config = self.contact_config
        device_config = self.device.device_config

        kpoints_transport = xp.linspace(
            -xp.pi,
            xp.pi,
            contact_config.num_kpoints_transport,
            endpoint=False,
        )

        e_k = self.compute_contact_bandstructure(
            kpoints_transport=kpoints_transport,
        )

        # Average over transverse k-points.
        e_k = xp.mean(e_k, axis=1)

        # Recompute the actual mid-gap energy from the band structure.
        valence_band_edge, conduction_band_edge = contact_band_edges(
            e_k, self.mid_gap_energy
        )
        mid_gap_energy = 0.5 * (conduction_band_edge + valence_band_edge)

        # The Fermi level is provided, no need to compute.
        if contact_config.fermi_level is not None:
            if comm.rank == 0:
                print(
                    f"Computing contact properties for contact '{contact_config.name}'...",
                    flush=True,
                )
                print(f"    Fermi level: {contact_config.fermi_level} eV", flush=True)
                print(
                    f"    Conduction band minimum: {conduction_band_edge} eV",
                    flush=True,
                )
                print(f"    Valence band maximum: {valence_band_edge} eV", flush=True)
                print(f"    Recomputed mid-gap energy: {mid_gap_energy} eV", flush=True)

            return contact_config.fermi_level, mid_gap_energy, conduction_band_edge

        doping_density = contact_doping_density(
            coordinates=get_host(
                self.device.orbital_coordinates[
                    self.unit_cell_orbital_indices[self.origin_key]
                ]
            ),
            geometry_regions=device_config.geometry.regions,
        )

        fermi_level = contact_fermi_level(
            e_k=e_k,
            kpoints=kpoints_transport,
            mid_gap_energy=self.mid_gap_energy,
            cell_volume=self.cell_volume,
            doping_density=doping_density,
            temperature=self.temperature,
        )

        if comm.rank == 0:
            print(
                f"Computing contact properties for contact '{contact_config.name}'...",
                flush=True,
            )
            print(f"    Doping density: {doping_density} Å^-3", flush=True)
            print(f"    Fermi level: {fermi_level} eV", flush=True)
            print(f"    Conduction band minimum: {conduction_band_edge} eV", flush=True)
            print(f"    Valence band maximum: {valence_band_edge} eV", flush=True)
            print(f"    Recomputed mid-gap energy: {mid_gap_energy} eV", flush=True)

        return fermi_level, mid_gap_energy, conduction_band_edge
