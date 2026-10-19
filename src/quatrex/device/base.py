# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the base device class for electronic transport calculations."""

import warnings
from pathlib import Path

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, xp
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.utils.gpu_utils import get_host
from qttools.utils.mpi_utils import distributed_load
from quatrex.contact import BaseContact
from quatrex.core.config import DeviceConfig, QuatrexConfig
from quatrex.device.inputs import create_coordinate_grid, distributed_read_xyz
from quatrex.grid import monkhorst_pack


def map_potential_to_orbitals(
    potential: NDArray,
    atomic_species: NDArray,
    device_config: DeviceConfig,
) -> NDArray:
    """Maps a potential to the (spin-)orbital basis of the device.

    The potential can be given per atom, per spatial orbital (only
    distinguished from per (spin-)orbital if spin is treated
    explicitly), or per (spin-)orbital.

    Parameters
    ----------
    potential : NDArray
        The potential.
    atomic_species : NDArray
        Array of atom symbols for each atom.
    device_config : DeviceConfig
        The device configuration.

    Returns
    -------
    NDArray
        The potential for each (spin-)orbital.

    """
    potential = xp.asarray(potential)
    orbitals_per_atom = [device_config.orbitals_per_atom(s) for s in atomic_species]
    num_orbitals = sum(orbitals_per_atom)
    num_spin_channels = device_config.num_spin_channels

    # NOTE: If atom species is only 'X', then we still repeat.
    if potential.shape[0] == len(atomic_species):
        return xp.repeat(potential, orbitals_per_atom, axis=0)
    if potential.shape[0] == num_orbitals:
        return potential
    if num_spin_channels > 1 and potential.shape[0] * num_spin_channels == num_orbitals:
        return xp.repeat(potential, num_spin_channels, axis=0)

    raise ValueError(
        f"Potential shape {potential.shape} does not match the number of "
        f"atoms ({len(atomic_species)}) or orbitals "
        f"({num_orbitals // num_spin_channels} spatial, {num_orbitals} total)."
    )


class BaseDevice:
    """A quantum device for electronic transport calculations.

    Parameters
    ----------
    config : QuatrexConfig
        Configuration object containing input paths, device parameters,
        and computational settings.

    Attributes
    ----------
    config : QuatrexConfig
        Reference to the configuration object.
    orbital_coordinates : NDArray
        Array of orbital coordinates.
    atom_coordinates : NDArray
        Array of atomic coordinates.
    atomic_species : NDArray
        Array of atom symbols for each atom. NOTE: This array is always
        on the host since CuPy does not support string arrays.
    orbital_offsets : NDArray
        Array of cumulative orbital counts, used to map from atoms to
        orbitals. orbital_offsets[i] gives the starting orbital index
        for atom i.
    lattice_vectors : NDArray
        Array of lattice vectors for the device.
    potential : NDArray, optional
        Array of electrostatic potential for each orbital. Can be either
        None if no potential is provided or a 1D array where the 1D
        index corresponds to the orbital index or the atom index
        depending on the shape of the provided potential.
    kpoints : NDArray
        Array of k-points for the device, generated using a Monkhorst-Pack
        grid based on the configuration.
    contacts : list[BaseContact]
        List of Contact objects representing the semi-infinite leads
        connected to this device.
    hamiltonians : dict | DSDBSparse | None
        Hamiltonian matrices in either real space (QTBM) or k-space
        (SCBA).
    overlap_matrices : dict | DSDBSparse | None
        Overlap matrices in either real space (QTBM) or k-space (SCBA).

    """

    def __init__(self, config: QuatrexConfig) -> None:
        """Initializes a BaseDevice object from configuration."""

        self.config = config
        self.device_config = config.device

        (
            self.orbital_coordinates,
            self.atom_coordinates,
            self.atomic_species,
            self.lattice_vectors,
        ) = self._load_structure(config)

        # TODO Device/Contact currently assumes that these quantities are on the host
        self.atom_coordinates = get_host(self.atom_coordinates)

        self._init_orbitals()
        self.num_spatial_orbitals = self._count_input_spatial_orbitals()

        self.potential = self._load_potential(
            self.config.input_dir,
            self.atomic_species,
            self.device_config,
        )

        self.kpoints = monkhorst_pack(
            self.device_config.kpoint_grid, self.device_config.kpoint_shift
        )
        self.num_kpoints = len(self.kpoints)

        # Child classes will initialize the Hamiltonian and contacts in
        # their own init methods
        self.contacts: list[BaseContact] = []
        self.hamiltonians: dict | DSDBSparse | None = None
        # TODO: Should be `None` for QTBM if the basis is orthogonal.
        # No identity matrix should be allocated.
        self.overlap_matrices: dict | DSDBSparse | None = None

    @staticmethod
    def _load_potential(
        input_dir: Path,
        atomic_species: NDArray,
        device_config: DeviceConfig,
    ) -> NDArray:
        """Loads electrostatic potential data from input files.

        Attempts to load the electrostatic potential from potential.npy
        in the input directory. The potential can be provided either at
        the atomic level or at the orbital level (see
        `map_potential_to_orbitals`).

        Parameters
        ----------
        input_dir : Path
            Directory containing the `potential.npy` file.
        atomic_species : NDArray
            Array of atom symbols for each atom. NOTE: This array is
            always on the host since CuPy does not support string
            arrays.
        device_config : DeviceConfig
            The device configuration.

        Returns
        -------
        NDArray
            The electrostatic potential array. If no potential file is
            found, returns an array of zeros with length equal to the
            total number of (spin-)orbitals.

        """
        try:
            potential = distributed_load(input_dir / "potential.npy")
        except FileNotFoundError:
            num_orbitals = sum(map(device_config.orbitals_per_atom, atomic_species))
            return xp.zeros(num_orbitals)

        return map_potential_to_orbitals(potential, atomic_species, device_config)

    @staticmethod
    def _load_structure(
        config: QuatrexConfig,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Loads the orbital coordinates, atom coordinates, atomic
        species, and lattice vectors for the device.

        Note
        ----
        The lattice vectors if constructed from the unit cell are the
        unit cell vectors and not full device.

        Parameters
        ----------
        config : QuatrexConfig
            The Quatrex configuration.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray, NDArray]
            The orbital coordinates, atom coordinates, atomic species,
            and lattice vectors.

        """

        structure_file = config.input_dir / "structure.xyz"
        if not structure_file.exists():
            raise FileNotFoundError(f"Structure file {structure_file} not found.")
        lattice_vectors, atom_coordinates, atomic_species = distributed_read_xyz(
            structure_file
        )

        orbitals_per_atom = [config.device.orbitals_per_atom(s) for s in atomic_species]
        atom_coordinates = xp.asarray(atom_coordinates)
        orbital_coordinates = xp.repeat(atom_coordinates, orbitals_per_atom, axis=0)

        if config.device.construct_from_unit_cell:

            transport_ind = "abc".index(config.device.transport_direction)

            orbital_coordinates = create_coordinate_grid(
                orbital_coordinates,
                config.device.num_transport_cells
                * config.device.neighbor_cell_cutoff[transport_ind],
                transport_ind,
                xp.asarray(lattice_vectors),
            )

            atom_coordinates = create_coordinate_grid(
                atom_coordinates,
                config.device.num_transport_cells
                * config.device.neighbor_cell_cutoff[transport_ind],
                transport_ind,
                xp.asarray(lattice_vectors),
            )

            atomic_species = np.concatenate(
                [atomic_species]
                * config.device.neighbor_cell_cutoff[transport_ind]
                * config.device.num_transport_cells
            )

        return orbital_coordinates, atom_coordinates, atomic_species, lattice_vectors

    def _init_orbitals(self) -> None:
        """Initializes the orbital indexing system for the device.

        Sets up the mapping between atoms and orbitals by determining
        how many orbitals each atom has and creating cumulative indexing
        arrays.

        The number of orbitals per atom is determined from the
        configuration, with a default of 1 orbital per atom if not
        specified.

        """
        orbitals_per_atom = np.fromiter(
            map(self.device_config.orbitals_per_atom, self.atomic_species),
            dtype=np.int32,
        )
        # Create a vector with the starting orbital for each atom
        self.orbital_offsets = np.hstack(([0], np.cumsum(orbitals_per_atom)))

    def _count_input_spatial_orbitals(self) -> int:
        """Counts the spatial orbitals that the input matrices refer to.

        This is the number of orbitals (without spin) in the structure
        file, i.e., in the unit cell if the device is constructed from
        it.

        Returns
        -------
        int
            The number of spatial orbitals in the input structure.

        """
        num_spatial_orbitals = (
            self.orbital_offsets[-1] // self.device_config.num_spin_channels
        )
        if self.device_config.construct_from_unit_cell:
            transport_ind = "abc".index(self.device_config.transport_direction)
            num_spatial_orbitals //= (
                self.device_config.num_transport_cells
                * self.device_config.neighbor_cell_cutoff[transport_ind]
            )
        return int(num_spatial_orbitals)

    def _validate_contacts(self):
        """Validates that all required contact parameters are set.

        Raises warnings if any required parameters are missing. If the
        Fermi level or other contact parameters are not set, they will
        be computed automatically. It is recommended to run the
        pre-processing step to compute these parameters beforehand.

        Note
        ----
        This method assumes that the potential is not backed in the
        Hamiltonian.

        """

        # NOTE: The Fermi level (chemical potential) is always needed to
        # compute currents. If this is not set in the contact
        # configuration, it can be computed from the contact's
        # electronic structure and a mid-gap energy guess for the
        # respective contact. The mid-gap energy is also needed for a
        # reference contact when computing excess charge densities. The
        # conduction band edge is needed in a reference contact when
        # doing self-consistent Schrödinger-Poisson, to set the boundary
        # conditions for the Poisson equation.
        for contact_config, contact in zip(self.device_config.contacts, self.contacts):
            if (
                self.config.electron.band_edge_tracking
                or contact_config.fermi_level is None
                or (
                    self.config.scsp is not None
                    and (
                        contact_config.mid_gap_energy is None
                        or contact_config.conduction_band_edge is None
                    )
                )
            ):
                if comm.rank == 0:
                    print(
                        f"Computing Fermi level for contact {contact_config.name}",
                        flush=True,
                    )
                    warnings.warn(
                        "Recomputing the Fermi level and other contact parameters.\n"
                        "Please call `quatrex pre-process` beforehand to avoid this\n"
                        "or manually set the parameters in the contact configuration file.",
                    )
                # NOTE: `compute_contact_band_properties` will access the
                # set properties of the contact. and thus it is called at
                # the end.
                (
                    contact.fermi_level,
                    contact.mid_gap_energy,
                    contact.conduction_band_edge,
                ) = contact.compute_contact_band_properties()
