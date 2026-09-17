# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the base device class for electronic transport calculations."""

import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, xp
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.utils.gpu_utils import get_host
from qttools.utils.mpi_utils import distributed_load
from quatrex.contact import BaseContact
from quatrex.core.config import QuatrexConfig
from quatrex.device.inputs import create_coordinate_grid, distributed_read_xyz
from quatrex.grid import monkhorst_pack


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

        self.potential = self._load_potential(
            self.config.input_dir,
            self.atom_coordinates,
            self.atomic_species,
            self.device_config.num_orbitals_per_atom,
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
        atom_coordinates: NDArray,
        atomic_species: NDArray,
        num_orbitals_per_atom: dict[str, int],
    ) -> NDArray:
        """Loads electrostatic potential data from input files.

        Attempts to load the electrostatic potential from potential.npy
        in the input directory. The potential can be provided either at
        the atomic level or at the orbital level.

        Parameters
        ----------
        input_dir : Path
            Directory containing the `potential.npy` file.
        atom_coordinates : NDArray
            Array of atomic coordinates.
        atomic_species : NDArray
            Array of atom symbols for each atom. NOTE: This array is
            always on the host since CuPy does not support string
            arrays.
        num_orbitals_per_atom : dict[str, int]
            Dictionary mapping atomic species to the number of orbitals
            per atom.

        Returns
        -------
        NDArray
            The electrostatic potential array. If no potential file is
            found, returns an array of zeros with length equal to the
            total number of orbitals.


        """
        orbitals_per_atom = [
            num_orbitals_per_atom.get(species, 1) for species in atomic_species
        ]
        num_orbitals = np.sum(np.array(orbitals_per_atom))

        try:
            potential = distributed_load(input_dir / "potential.npy")

            # NOTE: If atom species is only 'X', then we still repeat.
            if potential.shape[0] == atom_coordinates.shape[0]:
                # Upscale the potential to the number of orbitals
                potential = xp.repeat(potential, orbitals_per_atom, axis=0)
            elif potential.shape[0] != num_orbitals:
                raise ValueError(
                    "Potential shape does not match number of atoms or orbitals."
                )

        except FileNotFoundError:
            potential = xp.zeros(num_orbitals)

        return potential

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

        orbitals_per_atom = [
            config.device.num_orbitals_per_atom.get(s, 1) for s in atomic_species
        ]
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
            map(
                defaultdict(lambda: 1, self.device_config.num_orbitals_per_atom).get,
                self.atomic_species,
            ),
            dtype=np.int32,
        )
        # Create a vector with the starting orbital for each atom
        self.orbital_offsets = np.hstack(([0], np.cumsum(orbitals_per_atom)))

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
