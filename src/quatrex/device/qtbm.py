# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the QTBM device class for electronic transport calculations."""

import warnings

import numpy as np
import scipy

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures.dcsx import DCSX
from qttools.utils.mpi_utils import get_section_sizes
from quatrex.contact import QTBMContact
from quatrex.core.config import QuatrexConfig
from quatrex.device.base import BaseDevice
from quatrex.device.inputs import load_matrices
from quatrex.device.nested_dissection import nested_dissection_orbitals


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

    """

    def __init__(self, config: QuatrexConfig) -> None:
        super().__init__(config)

        self.are_matrices_complex: bool = False
        self._init_hamiltonian()

        if self.config.electron.solver.order == "nested_dissection":
            self._reorder_matrices()

        self.row_offsets = self._get_row_offsets()
        self._add_contacts()
        self._convert_to_dcsx()

    def _reorder_matrices(self):
        """Reorders the Hamiltonian and overlap matrices using nested dissection."""
        if self.config.electron.solver.order != "nested_dissection":
            raise ValueError(
                "Nested dissection reordering is only applied when "
                "`config.electron.solver.order` is set to 'nested_dissection'."
            )
        target_leaf_size = self.config.electron.solver.target_leaf_size
        total_atoms = len(self.atom_coordinates)
        ideal_depth = max(1, int(np.log2(total_atoms / target_leaf_size)))

        n_orb_per_atom = [
            self.device_config.num_orbitals_per_atom[s] for s in self.atomic_species
        ]
        self.nd_result = nested_dissection_orbitals(
            self.atom_coordinates,
            n_orb_per_atom,
            max_depth=ideal_depth,
            matrices=[*self.hamiltonians.values(), *self.overlap_matrices.values()],
        )

        hamiltonians = {}
        overlap_matrices = {}

        for r, h_r in self.hamiltonians.items():
            r_flipped = (-r[0], -r[1], -r[2])

            h_flip = sparse.csr_array(self.hamiltonians[r_flipped])
            if (
                hasattr(h_flip, "has_canonical_format")
                and not h_flip.has_canonical_format
            ):
                h_flip.sum_duplicates()
                h_flip.sort_indices()

            hamiltonians[r] = (
                self.hamiltonians[r] + sparse.triu(h_flip, k=1).conj().transpose()
            )

        for r, s_r in self.overlap_matrices.items():
            r_flipped = (-r[0], -r[1], -r[2])

            s_flip = sparse.csr_array(self.overlap_matrices[r_flipped])
            if (
                hasattr(s_flip, "has_canonical_format")
                and not s_flip.has_canonical_format
            ):
                s_flip.sum_duplicates()
                s_flip.sort_indices()

            overlap_matrices[r] = (
                self.overlap_matrices[r] + sparse.triu(s_flip, k=1).conj().transpose()
            )

        order = self.nd_result.orbital_perm
        if hasattr(order, "get"):
            order = order.get()
        order = np.asarray(order)

        # 2. Reorder orbitals and trim lower triangle
        for r, h_r in hamiltonians.items():
            tmp = h_r.get() if hasattr(h_r, "get") else h_r

            # Apply permutation on CPU
            tmp = tmp[order, :][:, order]
            tmp = scipy.sparse.triu(tmp).tocsr()

            # Clean up structure to guarantee clean indices for future operations
            tmp.sum_duplicates()
            tmp.sort_indices()

            self.hamiltonians[r] = tmp

        for r, s_r in overlap_matrices.items():
            tmp = s_r.get() if hasattr(s_r, "get") else s_r

            tmp = tmp[order, :][:, order]
            tmp = scipy.sparse.triu(tmp).tocsr()

            tmp.sum_duplicates()
            tmp.sort_indices()

            self.overlap_matrices[r] = tmp

        self.orbital_coordinates = self.orbital_coordinates[
            self.nd_result.orbital_perm, :
        ]
        self.atom_coordinates = self.atom_coordinates[self.nd_result.atom_perm, :]
        self.atomic_species = self.atomic_species[self.nd_result.atom_perm]
        self.potential = (
            self.potential[self.nd_result.orbital_perm]
            if self.potential is not None
            else None
        )

        self.orbital_offsets = self._init_orbitals()

    def _get_row_offsets(self) -> NDArray:
        """Computes the row offsets for distributed execution.

        Returns
        -------
        NDArray
            Array of row offsets for each process in the communicator.

        """
        section_sizes, __ = get_section_sizes(
            self.orbital_coordinates.shape[0], comm.block.size
        )
        return np.cumsum([0] + section_sizes)

    def _convert_to_dcsx(self):
        """Converts Hamiltonian and overlap matrices to DCSX format."""
        # NOTE: In the first step, naively split the matrix by uniform by the number of rows.
        # NOTE: This can not be really unified with SCBA because of the `block` requirement in SCBA.
        # NOTE: Currently, the contacts need the full graph and thus the
        # conversion to `DCSX` is happening not in `_init_hamiltonian`.
        # NOTE: We know at this point `h_r` is upper diagonal.
        for r, h_r in self.hamiltonians.items():
            tmp = h_r[
                self.row_offsets[comm.block.rank] : self.row_offsets[
                    comm.block.rank + 1
                ],
                :,
            ]
            self.hamiltonians[r] = DCSX.from_sparray(
                sparray=tmp,
                symmetry="upper-triangular",
            )

        for r, s_r in self.overlap_matrices.items():
            tmp = s_r[
                self.row_offsets[comm.block.rank] : self.row_offsets[
                    comm.block.rank + 1
                ],
                :,
            ]
            self.overlap_matrices[r] = DCSX.from_sparray(
                sparray=tmp,
                symmetry="upper-triangular",
            )

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
                    row_offsets=self.row_offsets,
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

        if not (self.config.input_dir / "hamiltonian.h5").exists():
            raise ValueError("Hamiltonian matrix not found.")

        # NOTE: `load_matrices` enforces that all of them have the same
        # shape.
        self.hamiltonians = load_matrices(self.config, "hamiltonian")

        for r, h_r in self.hamiltonians.items():
            # assert all hamiltonians are sparse matrices
            if not isinstance(h_r, sparse.spmatrix):
                raise TypeError(
                    f"Hamiltonian matrix at index {r} is not a sparse matrix.\n"
                    f"Matrix type: {type(h_r)}"
                )

            self.hamiltonians[r] = sparse.csr_matrix(self.hamiltonians[r])

            if not self.hamiltonians[r].has_canonical_format:
                self.hamiltonians[r].sum_duplicates()
                self.hamiltonians[r].sort_indices()

        if (self.config.input_dir / "overlap.h5").exists():
            self.overlap_matrices = load_matrices(self.config, "overlap")

            for r, s_r in self.overlap_matrices.items():
                if s_r.shape != self.hamiltonians[(0, 0, 0)].shape:
                    raise ValueError(
                        f"Overlap matrix at index {r} has incompatible "
                        "shape with Hamiltonian. Expected "
                        f"{self.hamiltonians[(0, 0, 0)].shape}, "
                        f"got {s_r.shape}."
                    )

                # assert all overlap_matrices are sparse matrices
                if not isinstance(s_r, sparse.spmatrix):
                    raise TypeError(
                        f"Overlap matrix at index {r} is not a sparse matrix."
                    )

                self.overlap_matrices[r] = sparse.csr_matrix(self.overlap_matrices[r])

                if not self.overlap_matrices[r].has_canonical_format:
                    self.overlap_matrices[r].sum_duplicates()
                    self.overlap_matrices[r].sort_indices()

        else:
            if comm.rank == 0:
                warnings.warn(
                    "No overlap matrices found. Assuming identity matrix.",
                )
            self.overlap_matrices = {
                (0, 0, 0): sparse.eye(
                    self.hamiltonians[(0, 0, 0)].shape[0],
                    dtype=xp.float64,
                    format="csr",
                )
            }

        # NOTE: `load_matrices` enforces that all of them have the same
        # type.
        self.are_matrices_complex = (
            self.hamiltonians[(0, 0, 0)].dtype == np.complex128
        ) or (self.overlap_matrices[(0, 0, 0)].dtype == np.complex128)

        if (len(self.hamiltonians) == 1) and self.device_config.kpoint_grid != (
            1,
            1,
            1,
        ):
            raise ValueError(
                "The device only has a Gamma point Hamiltonian, "
                "but more than one k-point is configured."
            )

        if comm.rank == 0:
            print(
                f"Loaded {len(self.hamiltonians)} Hamiltonian matrices with size {self.hamiltonians[(0, 0, 0)].shape}",
                flush=True,
            )
            print(
                f"Loaded {len(self.overlap_matrices)} overlap matrices with size {self.overlap_matrices[(0, 0, 0)].shape}",
                flush=True,
            )
