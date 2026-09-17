# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the QTBM device class for electronic transport calculations."""

import warnings

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qttools import sparse, xp
from quatrex.contact import QTBMContact
from quatrex.core.config import QuatrexConfig
from quatrex.device.base import BaseDevice
from quatrex.device.inputs import load_matrices


class QTBMDevice(BaseDevice):
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
        Array of atom symbols for each atom. NOTE: This array is always on the
        host since CuPy does not support string arrays.
    orbital_offsets : NDArray
        Array of cumulative orbital counts, used to map from atoms to
        orbitals. orbital_offsets[i] gives the starting orbital index
        for atom i.
    potential : NDArray, optional
        Array of electrostatic potential for each orbital.
        Can be either None if no potential is provided or a 1D array
        where the 1D index corresponds to the orbital index or the
        atom index depending on the shape of the provided potential.
    contacts : list[QTBMContact]
        List of Contact objects representing the semi-infinite leads
        connected to this device.
    hamiltonians : dict
        Dictionary of Hamiltonian matrices indexed by (i, j, k) lattice
        vectors. Keys are tuples representing the lattice vector
        indices, values are sparse CSR matrices.
    overlap_matrices : dict
        Dictionary of overlap matrices with the same indexing as
        hamiltonians. For orthogonal basis sets, defaults to identity
        matrices.
    gamma_only : bool
        True if only the Gamma point (0,0,0) Hamiltonian is available,
        indicating that k-point calculations are not possible.

    """

    def __init__(self, config: QuatrexConfig) -> None:
        super().__init__(config)
        self._init_hamiltonian()
        self._add_contacts()

    def _add_contacts(self):
        """Initializes and attaches contacts to the device.

        Creates Contact objects for each contact defined in the device
        configuration. Each contact represents a semi-infinite lead
        connected to the finite device region, providing boundary
        conditions for transport calculations.

        """

        for contact_config in self.device_config.contacts:
            self.contacts.append(
                QTBMContact(
                    device=self,
                    contact_config=contact_config,
                    sparsity_pattern=self.hamiltonians[(0, 0, 0)],
                )
            )

        if comm.rank == 0:
            print(
                f"Device initialized with {len(self.contacts)} contacts.",
                flush=True,
            )

    def _init_hamiltonian(self) -> None:
        """Initializes Hamiltonian and overlap matrices from files.

        Loads sparse matrices from .h5 files in the input directory.
        Files should be named "hamiltonian.h5" and
        "overlap.h5" where the keys are strings of [i,j,k]
        representing lattice vector indices.

        For missing overlap matrices, identity matrices are assumed
        (orthogonal basis). The (0,0,0) Hamiltonian matrix is mandatory
        and its absence raises an error.

        """

        self.gamma_only = False

        if not (self.config.input_dir / "hamiltonian.h5").exists():
            raise ValueError("Hamiltonian matrix not found.")

        self.hamiltonians = load_matrices(
            self.config, "hamiltonian", force_complex=False
        )

        for r, h_r in self.hamiltonians.items():
            if not h_r.shape[0] == h_r.shape[1]:
                raise ValueError(
                    f"Hamiltonian matrix at index {r} is not square. "
                    f"Shape: {h_r.shape}"
                )

            # assert all hamiltonians are sparse matrices
            if not isinstance(h_r, sparse.spmatrix):
                raise TypeError(
                    f"Hamiltonian matrix at index {r} is not a sparse matrix.\n"
                    f"Matrix type: {type(h_r)}"
                )

            self.hamiltonians[r] = sparse.csr_matrix(self.hamiltonians[r])

            if self.hamiltonians[r].dtype in [np.complex64, np.complex128]:
                self.matrices_complex = True

            if not self.hamiltonians[r].has_canonical_format:
                self.hamiltonians[r].sum_duplicates()
                self.hamiltonians[r].sort_indices()

        size = self.hamiltonians[(0, 0, 0)].shape[0]

        if (self.config.input_dir / "overlap.h5").exists():
            self.overlap_matrices = load_matrices(self.config, "overlap")

            for r in self.overlap_matrices:
                if (
                    self.overlap_matrices[r].shape[0]
                    != self.overlap_matrices[r].shape[1]
                ):
                    raise ValueError(
                        f"Overlap matrix at index {r} is not square. "
                        f"Shape: {self.overlap_matrices[r].shape}"
                    )

                if self.overlap_matrices[r].shape != (size, size):
                    raise ValueError(
                        f"Overlap matrix at index {r} has incompatible "
                        f"shape with Hamiltonian. Expected {(size, size)}, "
                        f"got {self.overlap_matrices[r].shape}."
                    )

                # assert all overlap_matrices are sparse matrices
                if not isinstance(self.overlap_matrices[r], sparse.spmatrix):
                    raise TypeError(
                        f"Overlap matrix at index {r} is not a sparse matrix."
                    )

                self.overlap_matrices[r] = sparse.csr_matrix(self.overlap_matrices[r])

                if self.overlap_matrices[r].dtype in [np.complex64, np.complex128]:
                    self.matrices_complex = True

                if not self.overlap_matrices[r].has_canonical_format:
                    self.overlap_matrices[r].sum_duplicates()
                    self.overlap_matrices[r].sort_indices()

        else:
            if comm.rank == 0:
                warnings.warn(
                    "No overlap matrices found. Assuming identity matrix.",
                )
            self.overlap_matrices = {
                (0, 0, 0): sparse.eye(size, dtype=xp.float64, format="csr")
            }

        if comm.rank == 0:
            print(f"Loaded {len(self.hamiltonians)} Hamiltonian matrices", flush=True)
            print(f"Loaded {len(self.overlap_matrices)} overlap matrices", flush=True)

        if len(self.hamiltonians) == 1:
            self.gamma_only = True
