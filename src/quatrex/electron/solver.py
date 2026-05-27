# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
from typing import Literal

import numpy as np

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.datastructures.dsdbsparse import _DStackView
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.profiling import Profiler
from qttools.toeplitz.toeplitz import get_periodic_superblocks, homogenize
from qttools.utils.mpi_utils import get_local_slice, get_section_sizes
from qttools.utils.solvers_utils import get_batches
from qttools.utils.stack_utils import scale_stack
from quatrex.bandstructure.band_edges import find_renormalized_eigenvalues
from quatrex.bandstructure.contact import (
    contact_band_edges,
    contact_band_structure,
    contact_doping_density,
    contact_fermi_level,
)
from quatrex.core.config import QuatrexConfig
from quatrex.core.statistics import fermi_dirac
from quatrex.core.subsystem import SubsystemSolver
from quatrex.device import Device
from quatrex.device.contact import get_inverse_order, order_block
from quatrex.device.inputs import assemble_matrix

profiler = Profiler()


class ElectronSolver(SubsystemSolver):
    """Solves the electron dynamics.

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex simulation configuration.
    energies : np.ndarray
        The energies at which to solve.

    """

    system = "electron"

    def __init__(
        self,
        config: QuatrexConfig,
        energies: NDArray,
        sparsity_pattern: sparse.coo_matrix,
    ) -> None:
        """Initializes the electron solver."""
        super().__init__(config, energies)

        self.local_energies = get_local_slice(energies, comm.stack)

        # Load the device Hamiltonian.
        self.hamiltonian, hamiltonian_sparsity_pattern = assemble_matrix(
            config=config,
            matrix_name="hamiltonian",
            sparsity_pattern=None,
            shift_kpoints=False,
        )

        # Make sure that the the system matrix sparsity is a superset of
        # self-energy and Hamiltonian sparsity.
        sparsity_pattern += hamiltonian_sparsity_pattern

        del hamiltonian_sparsity_pattern
        self.block_sizes = self.hamiltonian.block_sizes

        try:
            # Attempt to load the device overlap matrix.
            self.overlap, overlap_sparsity_pattern = assemble_matrix(
                config=config,
                matrix_name="overlap",
                sparsity_pattern=None,
                shift_kpoints=False,
            )

            # Make sure that the the system matrix sparsity is a superset of
            # self-energy and overlap sparsity.
            sparsity_pattern += overlap_sparsity_pattern
            # Check that the overlap matrix and Hamiltonian matrix match.
            if self.overlap.shape != self.hamiltonian.shape:
                raise ValueError(
                    "Overlap matrix and Hamiltonian matrix have different shapes."
                )

            if comm.rank == 0:
                print("Non-orthogonal basis detected.", flush=True)

        except FileNotFoundError:
            self.overlap = None
            if comm.rank == 0:
                print("No overlap matrix found. Assuming orthogonal basis.", flush=True)

        # Allocate memory for the system matrix.
        self.system_matrix = config.compute.dsdbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=self.block_sizes,
            global_stack_shape=self.energies.shape
            + tuple([int(k) for k in config.device.kpoint_grid if k > 1]),
        )
        self.system_matrix.free_data()  # Free any previously allocated data
        del sparsity_pattern

        self.block_offsets = np.hstack(([0], np.cumsum(self.block_sizes)))
        # Check that the provided block sizes match the Hamiltonian.
        if self.block_sizes.sum() != self.hamiltonian.shape[-2]:
            raise ValueError(
                "Block sizes do not match Hamiltonian. "
                f"{self.block_sizes.sum()} != {self.hamiltonian.shape[-2]}"
            )

        # Load the potential.
        # TODO: The structure should not be reloaded here.
        # This will be fixed when the device is unified.
        __, atom_coordinates, atomic_species = Device.load_structure(config)
        self.potential = Device.load_potential(
            config.input_dir,
            atom_coordinates,
            atomic_species,
            config.device.num_orbitals_per_atom,
        )

        if self.potential.size != self.hamiltonian.shape[-2]:
            raise ValueError("Potential matrix and Hamiltonian have different shapes.")
        self.eta = config.electron.eta
        self.eta_obc = config.electron.eta_obc

        # Contacts.
        self.flatband = config.electron.flatband
        if self.flatband and comm.rank == 0:
            print("Flatband conditions detected", flush=True)

        self.compute_meir_wingreen_current = config.electron.solver.compute_current

        self.dos_peak_limit = config.electron.dos_peak_limit

        # Band edges and Fermi levels.
        self.band_edge_tracking = config.electron.band_edge_tracking

        orbitals_per_atom = [
            config.device.num_orbitals_per_atom.get(species, 1)
            for species in atomic_species
        ]
        orbital_coordinates = np.repeat(atom_coordinates, orbitals_per_atom, axis=0)
        left_band_edge_info = xp.empty(3, dtype=float)
        if comm.block.rank == 0:
            # Quantities related to the left contact.
            left_band_edge_info = self._configure_contact_band_edges(
                config=config,
                hamiltonian=self.hamiltonian,
                overlap=self.overlap,
                coordinates=orbital_coordinates[: self.block_sizes[0]],
                side="left",
            )

        # Communicate the band edge info to all ranks.
        comm.block.bcast(sendrecvbuf=left_band_edge_info, root=0)
        (
            self.left_fermi_level,
            self.left_mid_gap_energy,
            self.left_delta_fermi_level_conduction_band,
        ) = left_band_edge_info

        self.left_voltage = config.electron.left_contact.voltage
        self.left_mid_gap_energy -= self.left_voltage

        self.left_temperature = config.electron.left_contact.temperature

        mu_left = self.left_fermi_level - self.left_voltage
        self.left_occupancies = fermi_dirac(
            self.local_energies - mu_left, self.left_temperature
        )

        right_band_edge_info = xp.empty(3, dtype=float)
        if comm.block.rank == comm.block.size - 1:
            # Quantities related to the right contact.
            right_band_edge_info = self._configure_contact_band_edges(
                config=config,
                hamiltonian=self.hamiltonian,
                overlap=self.overlap,
                coordinates=orbital_coordinates[-self.block_sizes[-1] :],
                side="right",
            )

        # Communicate the band edge info to all ranks.
        comm.block.bcast(sendrecvbuf=right_band_edge_info, root=comm.block.size - 1)
        (
            self.right_fermi_level,
            self.right_mid_gap_energy,
            self.right_delta_fermi_level_conduction_band,
        ) = right_band_edge_info

        self.right_voltage = config.electron.right_contact.voltage
        self.right_mid_gap_energy -= self.right_voltage
        self.right_temperature = config.electron.right_contact.temperature
        # Compute contact chemical potentials and occupancies.
        mu_right = self.right_fermi_level - self.right_voltage
        self.right_occupancies = fermi_dirac(
            self.local_energies - mu_right, self.right_temperature
        )

        if comm.rank == 0:
            print(
                f"Left contact: \n"
                f"  Fermi level: {self.left_fermi_level} eV\n"
                f"  Mid-gap energy: {self.left_mid_gap_energy} eV\n"
                f"  Conduction band edge - Fermi level: {self.left_delta_fermi_level_conduction_band} eV\n"
                f"Right contact: \n"
                f"  Fermi level: {self.right_fermi_level} eV\n"
                f"  Mid-gap energy: {self.right_mid_gap_energy} eV\n"
                f"  Conduction band edge - Fermi level: {self.right_delta_fermi_level_conduction_band} eV\n",
                flush=True,
            )

        # Prepare Buffers for OBC.
        self.obc_blocks = OBCBlocks(num_blocks=self.system_matrix.num_local_blocks)
        self.block_sections = config.electron.obc.block_sections

        self.call_count = 0
        self.filtering_iteration_limit = config.electron.filtering_iteration_limit

        self.max_batch_size = config.electron.max_batch_size

    @staticmethod
    def _configure_contact_band_edges(
        config: QuatrexConfig,
        hamiltonian: DSDBSparse,
        overlap: DSDBSparse | None,
        coordinates: NDArray,
        side: Literal["left", "right"],
    ) -> NDArray:
        """Configures the contact band edges and Fermi level.

        Parameters
        ----------
        config : QuatrexConfig
            The quatrex simulation configuration.
        hamiltonian : DSDBSparse
            The Hamiltonian matrix of the contact.
        overlap : DSDBSparse | None
            The overlap matrix of the contact. If None, the overlap is
            assumed to be the identity.
        coordinates : NDArray
            The orbital coordinates of the contact. This is needed to
            determine the doping density of the contact.
        side : Literal["left", "right"]
            The contact side for which to configure the band edges.

        Returns
        -------
        band_edge_info : NDArray
            An array containing the Fermi level, mid-gap energy and the
            difference between the conduction band edge and the Fermi
            level of the contact. The order is (fermi_level,
            mid_gap_energy, delta_fermi_level_conduction_band). The
            `delta_fermi_level_conduction_band` returns NaN if Fermi
            level is provided and the band edge tracking is disabled.

        """
        if comm.block.size != 1:
            if comm.block.rank == 0 and side != "left":
                raise ValueError(
                    "Left contact band edge configuration must only be performed on the first block rank."
                )
            if comm.block.rank == comm.block.size - 1 and side != "right":
                raise ValueError(
                    "Right contact band edge configuration must only be performed on the last block rank."
                )

        contact_config = getattr(config.electron, f"{side}_contact")

        if (
            not config.electron.band_edge_tracking
            and contact_config.fermi_level is not None
        ):
            # If band edge tracking is disabled and the Fermi level is
            # provided, we can directly return the provided Fermi level.
            # The mid-gap energy might still be set to compute excess
            # carrier density for Poisson solver. The difference between
            # the conduction band edge and the Fermi level is definitely
            # not needed in this case, so we return NaN for that.
            mid_gap_energy = (
                xp.nan
                if contact_config.mid_gap_energy is None
                else contact_config.mid_gap_energy
            )
            return xp.array([contact_config.fermi_level, mid_gap_energy, xp.nan])

        # TODO: Block sectioning could be easily integrated here. Also
        # exploit that these are Hermitian.
        n = hamiltonian.num_local_blocks - 1
        m = n - 1
        diagonal_inds = (0, 0) if side == "left" else (n, n)
        upper_inds = (0, 1) if side == "left" else (n, m)

        h_xx = (
            hamiltonian.blocks[*upper_inds[::-1]],
            hamiltonian.blocks[*diagonal_inds],
            hamiltonian.blocks[*upper_inds],
        )

        if overlap is not None:
            s_xx = (
                overlap.blocks[*upper_inds[::-1]],
                overlap.blocks[*diagonal_inds],
                overlap.blocks[*upper_inds],
            )
        else:
            s_xx = None

        kpoints_transport = np.linspace(
            -np.pi, np.pi, contact_config.num_kpoints_transport
        )
        e_k = contact_band_structure(kpoints_transport, h_xx, s_xx)

        # Average over all dimensions, except for the transport k-point
        # dimension and the last dimension corresponding to the
        # eigenvalues.
        e_k = np.mean(e_k, axis=tuple(range(1, e_k.ndim - 1)))
        e_k = np.sort(e_k, axis=-1)

        valence_band_edge, conduction_band_edge = contact_band_edges(
            e_k, contact_config.mid_gap_energy
        )
        mid_gap_energy = 0.5 * (conduction_band_edge + valence_band_edge)

        if comm.rank == 0:
            print(
                f"{side.capitalize()} contact band edges:\n"
                f"  Conduction band edge: {conduction_band_edge} eV\n"
                f"  Valence band edge: {valence_band_edge} eV\n",
                flush=True,
            )

        if contact_config.fermi_level is not None:
            # The Fermi level is provided, no need to compute.
            delta_fermi_level_conduction_band = (
                conduction_band_edge - contact_config.fermi_level
            )

            return xp.array(
                [
                    contact_config.fermi_level,
                    mid_gap_energy,
                    delta_fermi_level_conduction_band,
                ]
            )

        doping_density = contact_doping_density(
            coordinates=coordinates,
            geometry_regions=config.device.geometry.regions,
        )

        fermi_level = contact_fermi_level(
            e_k=e_k,
            kpoints=kpoints_transport,
            mid_gap_energy=mid_gap_energy,
            cell_volume=np.abs(np.linalg.det(contact_config.lattice_vectors)),
            doping_density=doping_density,
            temperature=contact_config.temperature,
        )

        return xp.array(
            [fermi_level, mid_gap_energy, conduction_band_edge - fermi_level]
        )

    def _update_fermi_levels(
        self, left_band_edges: NDArray, right_band_edges: NDArray
    ) -> None:
        """Updates the Fermi levels.

        Parameters
        ----------
        out : tuple[DSDBSparse, ...]
            The Green's function tuple. In the order (lesser, greater,
            retarded).

        """
        self.left_mid_gap_energy = xp.mean(left_band_edges)
        self.right_mid_gap_energy = xp.mean(right_band_edges)

        __, left_conduction_band_edge = left_band_edges
        __, right_conduction_band_edge = right_band_edges

        self.left_fermi_level = (
            left_conduction_band_edge - self.left_delta_fermi_level_conduction_band
        )
        self.right_fermi_level = (
            right_conduction_band_edge - self.right_delta_fermi_level_conduction_band
        )

        (
            print(
                f"Updating conduction band edges: "
                f"{left_conduction_band_edge:.6f}, {right_conduction_band_edge:.6f}\n",
                f"Updating Fermi levels: {self.left_fermi_level:.6f}, {self.right_fermi_level:.6f}",
                flush=True,
            )
            if comm.rank == 0
            else None
        )

        mu_left = self.left_fermi_level - self.left_voltage
        self.left_occupancies = fermi_dirac(
            self.local_energies - mu_left, self.left_temperature
        )
        mu_right = self.right_fermi_level - self.right_voltage
        self.right_occupancies = fermi_dirac(
            self.local_energies - mu_right, self.right_temperature
        )

    def _compute_contact_obc(
        self,
        contact: str,
        diagonal_inds: tuple,
        upper_inds: tuple,
        occupancies: NDArray,
        order: str | NDArray | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Computes the OBC for a specific contact.

        Parameters
        ----------
        contact : str
            The contact for which to compute the OBC.
            Used for profiling and caching purposes.
        diagonal_inds : tuple
            The indices of the diagonal blocks corresponding to the contact.
        upper_inds : tuple
            The indices of the upper off-diagonal blocks corresponding to the contact.
        occupancies : NDArray
            The occupancies of the contact at the local energies.
        order : str | NDArray | None, optional
            The permutation of the blocks to achieve the same order as the canonical left contact.
            If None, the left contact order is assumed.
            Instead of an explicit permutation, the string "reverse" can be passed
            to reverse the order of the blocks, which is equivalent to the right contact order.

        Returns
        -------
        obc_retarded : NDArray
            The retarded OBC for the contact.
        obc_lesser : NDArray
            The lesser OBC for the contact.
        obc_greater : NDArray
            The greater OBC for the contact.

        """

        inverse_order = get_inverse_order(order)

        m_10, m_00, m_01 = get_periodic_superblocks(
            a_ji=order_block(self.system_matrix.blocks[*upper_inds[::-1]], order),
            a_ii=order_block(self.system_matrix.blocks[*diagonal_inds], order),
            a_ij=order_block(self.system_matrix.blocks[*upper_inds], order),
            block_sections=self.block_sections,
        )

        if self.overlap is None:
            s_10 = xp.zeros_like(m_10, dtype=m_10.dtype)
            s_00 = 1j * self.eta_obc * xp.eye(m_00.shape[-1], dtype=m_00.dtype)
            s_01 = xp.zeros_like(m_01, dtype=m_01.dtype)
        else:
            # Extract the overlap matrix blocks.
            s_10 = 1j * self.eta_obc * self.overlap.blocks[*upper_inds[::-1]]
            s_00 = 1j * self.eta_obc * self.overlap.blocks[*diagonal_inds]
            s_01 = 1j * self.eta_obc * self.overlap.blocks[*upper_inds]

        # TODO: use residuals to filter "bad" energies
        g_00, *__ = self.obc(
            (m_00 + s_00, m_01 + s_01, m_10 + s_10),
            contact="G: " + contact,
        )
        # Apply the retarded boundary self-energy.
        sigma_00 = m_10 @ g_00 @ m_01
        gamma_00 = 1j * (sigma_00 - sigma_00.conj().swapaxes(-2, -1))

        # Compute and apply the lesser boundary self-energy.
        obc_lesser = 1j * scale_stack(gamma_00.copy(), occupancies)
        # Compute and apply the greater boundary self-energy.
        obc_greater = 1j * scale_stack(gamma_00.copy(), occupancies - 1)

        return (
            order_block(sigma_00, inverse_order),
            order_block(obc_lesser, inverse_order),
            order_block(obc_greater, inverse_order),
        )

    @profiler.profile(label="ElectronSolver: OBC", level="default", comm=comm)
    def _compute_obc(self, batch_slice: slice) -> None:
        """Computes open boundary conditions.

        Parameters
        ----------
        batch_slice : slice
            The slice of the energy stack corresponding to the current batch.

        """
        if comm.block.rank == 0:
            obc_retarded, obc_lesser, obc_greater = self._compute_contact_obc(
                contact="left-" + str(batch_slice),
                diagonal_inds=(0, 0),
                upper_inds=(0, 1),
                occupancies=self.left_occupancies[batch_slice],
            )
            self.obc_blocks.retarded[0] = obc_retarded
            self.obc_blocks.lesser[0] = obc_lesser
            self.obc_blocks.greater[0] = obc_greater

        if comm.block.rank == comm.block.size - 1:
            n = self.system_matrix.num_local_blocks - 1
            m = n - 1
            obc_retarded, obc_lesser, obc_greater = self._compute_contact_obc(
                contact="right-" + str(batch_slice),
                diagonal_inds=(n, n),
                upper_inds=(n, m),
                occupancies=self.right_occupancies[batch_slice],
                order="reverse",
            )
            self.obc_blocks.retarded[-1] = obc_retarded
            self.obc_blocks.lesser[-1] = obc_lesser
            self.obc_blocks.greater[-1] = obc_greater

    def _add_overlap(
        self,
    ) -> None:
        """Adds the overlap matrix to the system matrix.

        This modifies the system matrix in-place, i.e. the result is stored in
        `self.system_matrix`.

        Parameters
        ----------
        self.system_matrix : DSDBSparse
            The matrix to add to.
        b : DSDBSparse
            The matrix to add.

        """
        if not isinstance(self.overlap, DSDBSparse):
            raise ValueError("Overlap matrix must be a DSDBSparse.")

        system_matrix_ = self.system_matrix.stack[...]
        overlap_ = self.overlap.stack[...]
        for i in range(self.system_matrix.num_local_blocks):
            j = i + 1
            system_matrix_.blocks[i, i] += overlap_.blocks[i, i]

            if (
                j >= self.system_matrix.num_local_blocks
                and comm.block.rank == comm.block.size - 1
            ):
                # The last rank does not have these blocks.
                continue

            system_matrix_.blocks[i, j] += overlap_.blocks[i, j]
            system_matrix_.blocks[j, i] += overlap_.blocks[j, i]

    def _apply_potential(
        self,
    ) -> None:
        """Applies the potential to the system matrix.

        This modifies the system matrix in-place, i.e. the result is stored in
        `self.system_matrix`.

        """
        if not isinstance(self.overlap, DSDBSparse):
            raise ValueError("Overlap matrix must be a DSDBSparse.")

        system_matrix_ = self.system_matrix.stack[...]
        overlap_ = self.overlap.stack[...]
        offset = 0
        for i in range(self.system_matrix.num_local_blocks):
            j = i + 1
            s_ii = overlap_.blocks[i, i]
            potential_i = self.potential[offset : offset + s_ii.shape[-1]]

            system_matrix_.blocks[i, i] -= (
                s_ii * potential_i[..., np.newaxis] + s_ii * potential_i
            ) / 2

            offset += s_ii.shape[-1]

            if (
                j >= self.system_matrix.num_local_blocks
                and comm.block.rank == comm.block.size - 1
            ):
                # The last rank does not have these blocks.
                continue

            s_ij = overlap_.blocks[i, j]
            s_ji = overlap_.blocks[j, i]
            potential_j = self.potential[offset : offset + s_ij.shape[-1]]

            system_matrix_.blocks[i, j] -= (
                s_ij * potential_i[..., np.newaxis] + s_ij * potential_j
            ) / 2
            system_matrix_.blocks[j, i] -= (
                s_ji * potential_j[..., np.newaxis] + s_ji * potential_i
            ) / 2

    def _subtract_hamiltonian_and_self_energy(
        self,
        sse_lesser: DSDBSparse | _DStackView,
        sse_greater: DSDBSparse | _DStackView,
        sse_retarded_hermitian: DSDBSparse | _DStackView,
    ) -> None:
        r"""Subtracts the Hamiltonian and the self-energy from the system matrix
        on the block-tridiagonal.

        $$\mathbf{M} \mathrel{{-}{=}} \mathbf{H} + \mathbf{\Sigma}^R +
        \frac{1}{2} \left(\mathbf{\Sigma}^{>} - \mathbf{\Sigma}^{<} \right)$$

        This modifies the system matrix in-place, i.e. the result is stored in
        `self.system_matrix`.

        Parameters
        ----------
        sse_lesser : DSDBSparse | _DStackView
            The lesser self-energy to subtract.
        sse_greater : DSDBSparse | _DStackView
            The greater self-energy to subtract.
        sse_retarded_hermitian : DSDBSparse | _DStackView
            The retarded self-energy to subtract.

        """
        system_matrix_ = self.system_matrix.stack[...]
        hamiltonian_ = self.hamiltonian.stack[...]
        sse_retarded_hermitian_ = sse_retarded_hermitian.stack[...]
        sse_lesser_ = sse_lesser.stack[...]
        sse_greater_ = sse_greater.stack[...]
        for i in range(self.system_matrix.num_local_blocks):
            j = i + 1
            system_matrix_.blocks[i, i] -= (
                sse_retarded_hermitian_.blocks[i, i]
                + 0.5 * (sse_greater_.blocks[i, i] - sse_lesser_.blocks[i, i])
                + hamiltonian_.blocks[i, i]
            )

            if (
                j >= self.system_matrix.num_local_blocks
                and comm.block.rank == comm.block.size - 1
            ):
                # The last rank does not have these blocks.
                continue

            system_matrix_.blocks[i, j] -= (
                sse_retarded_hermitian_.blocks[i, j]
                + 0.5 * (sse_greater_.blocks[i, j] - sse_lesser_.blocks[i, j])
                + hamiltonian_.blocks[i, j]
            )
            system_matrix_.blocks[j, i] -= (
                sse_retarded_hermitian_.blocks[j, i]
                + 0.5 * (sse_greater_.blocks[j, i] - sse_lesser_.blocks[j, i])
                + hamiltonian_.blocks[j, i]
            )

    @profiler.profile(label="ElectronSolver: Assemble", level="default", comm=comm)
    def _assemble_system_matrix(
        self,
        sse_lesser: DSDBSparse | _DStackView,
        sse_greater: DSDBSparse | _DStackView,
        sse_retarded_hermitian: DSDBSparse | _DStackView,
        batch_slice: slice,
    ) -> None:
        """Assembles the system matrix.

        Parameters
        ----------
        sse_lesser : DSDBSparse | _DStackView
            The lesser scattering self-energy.
        sse_greater : DSDBSparse | _DStackView
            The greater scattering self-energy.
        sse_retarded_hermitian : DSDBSparse | _DStackView
            The hermitian part of the retarded scattering self-energy.
        batch_slice : slice
            The slice of the energy stack corresponding to the current batch.

        """
        self.system_matrix.data = 0.0
        if self.overlap is None:
            self.system_matrix.fill_diagonal(1.0)
        else:
            self._add_overlap()

        scale_stack(
            self.system_matrix.data,
            self.local_energies[batch_slice] + 1j * self.eta,
        )

        if self.overlap is None:
            self.system_matrix -= sparse.diags(self.potential, format="csr")
        else:
            self._apply_potential()

        self._subtract_hamiltonian_and_self_energy(
            sse_lesser,
            sse_greater,
            sse_retarded_hermitian,
        )

    def _filter_peaks(self, out: tuple[DSDBSparse, ...]) -> None:
        """Filters out peaks in the Green's functions.

        Parameters
        ----------
        out : tuple[DSDBSparse, ...]
            The Green's function tuple. In the order (lesser, greater,
            retarded).

        """
        g_lesser, g_greater, g_retarded = out
        # local_dos = [
        #     (-xp.diagonal(block, axis1=-2, axis2=-1).imag).mean(-1)
        #     for block in g_retarded.block_diagonal()
        # ]

        g_retarded_diag = g_retarded.diagonal()
        block_sizes = g_retarded.block_sizes
        block_offsets = g_retarded.block_offsets
        local_dos = []
        for i, (bsz, boff) in enumerate(zip(block_sizes, block_offsets)):
            g_retarded_density = -g_retarded_diag[..., boff : boff + bsz].imag.mean(-1)
            local_dos.append(g_retarded_density)

        local_dos = xp.array(local_dos)
        dos = comm.stack.all_gather_v(
            local_dos, axis=1, mask=g_lesser._stack_padding_mask
        )

        dos_gradient = xp.abs(xp.gradient(dos, self.energies, axis=1))
        mask = (xp.max(dos_gradient, axis=0) > self.dos_peak_limit) | (
            xp.max(dos, axis=0) > 10
        )

        section_sizes, __ = get_section_sizes(self.energies.size, comm.stack.size)
        section_offsets = np.hstack(([0], np.cumsum(section_sizes)))
        local_mask = mask[
            section_offsets[comm.stack.rank] : section_offsets[comm.stack.rank + 1]
        ]

        g_lesser.data[local_mask] = 0.0
        g_greater.data[local_mask] = 0.0
        g_retarded.data[local_mask] = 0.0

    @profiler.profile(label="ElectronSolver", level="default", comm=comm)
    def solve(
        self,
        sse_lesser: DSDBSparse,
        sse_greater: DSDBSparse,
        sse_retarded_hermitian: DSDBSparse,
        out: tuple[DSDBSparse, ...],
    ):
        """Solves for the electron Green's function.

        Parameters
        ----------
        sse_lesser : DSDBSparse
            The lesser self-energy.
        sse_greater : DSDBSparse
            The greater self-energy.
        sse_retarded_hermitian : DSDBSparse
            The hermitian part of the retarded self-energy.
        out : tuple[DSDBSparse, ...]
            The output matrices. The order is (lesser, greater,
            retarded).

        """

        if self.flatband:
            with profiler.profile_range(
                label="ElectronSolver: Homogenize", level="default", comm=comm
            ):
                homogenize(sse_greater)
                homogenize(sse_lesser)
                homogenize(sse_retarded_hermitian)

        if self.band_edge_tracking:
            with profiler.profile_range(
                label="ElectronSolver: Band edges", level="default", comm=comm
            ):
                left_band_edges, right_band_edges = find_renormalized_eigenvalues(
                    hamiltonian=self.hamiltonian,
                    overlap=self.overlap,
                    potential=self.potential,
                    sigma_retarded_hermitian=sse_retarded_hermitian,
                    energies=self.energies,
                    conduction_band_guesses=(
                        self.left_fermi_level
                        + self.left_delta_fermi_level_conduction_band,
                        self.right_fermi_level
                        + self.right_delta_fermi_level_conduction_band,
                    ),
                    mid_gap_energies=(
                        self.left_mid_gap_energy,
                        self.right_mid_gap_energy,
                    ),
                    band_edge_config=self.config.compute.band_edge,
                )
                self._update_fermi_levels(left_band_edges, right_band_edges)

        if self.max_batch_size is None:
            max_batch_size = sse_lesser.shape[0]
        else:
            max_batch_size = self.max_batch_size

        batch_sizes, batch_offsets = get_batches(sse_lesser.shape[0], max_batch_size)

        self.meir_wingreen_current = []

        for i in range(len(batch_sizes)):

            batch_slice = slice(int(batch_offsets[i]), int(batch_offsets[i + 1]))
            sse_lesser_batch = sse_lesser.stack[batch_slice]
            sse_greater_batch = sse_greater.stack[batch_slice]
            sse_retarded_hermitian_batch = sse_retarded_hermitian.stack[batch_slice]

            # Free data when the batch size changes
            if i > 0 and batch_sizes[i] != batch_sizes[i - 1]:
                self.system_matrix.free_data()
            self.system_matrix.allocate_data(stack_size=batch_sizes[i])

            self._assemble_system_matrix(
                sse_lesser_batch,
                sse_greater_batch,
                sse_retarded_hermitian_batch,
                batch_slice,
            )

            self._compute_obc(batch_slice)

            with profiler.profile_range(
                label="ElectronSolver: Solve", level="default", comm=comm
            ):
                out_l, out_g, out_r = out
                out_slice = (
                    out_l.stack[batch_slice],
                    out_g.stack[batch_slice],
                    out_r.stack[batch_slice],
                )
                if comm.block.size > 1:
                    self.meir_wingreen_current.append(
                        self.solver_dist.selected_solve(
                            a=self.system_matrix,
                            sigma_lesser=sse_lesser_batch,
                            sigma_greater=sse_greater_batch,
                            obc_blocks=self.obc_blocks,
                            out=out_slice,
                            return_retarded=True,
                            return_current=self.compute_meir_wingreen_current,
                        )
                    )

                else:
                    self.meir_wingreen_current.append(
                        self.solver.selected_solve(
                            a=self.system_matrix,
                            sigma_lesser=sse_lesser_batch,
                            sigma_greater=sse_greater_batch,
                            obc_blocks=self.obc_blocks,
                            out=out_slice,
                            return_retarded=True,
                            return_current=self.compute_meir_wingreen_current,
                        )
                    )

        with profiler.profile_range(
            label="ElectronSolver: Filter", level="default", comm=comm
        ):
            self.system_matrix.free_data()
            if self.call_count < self.filtering_iteration_limit:
                self._filter_peaks(out)

        if self.compute_meir_wingreen_current:
            self.meir_wingreen_current = xp.concatenate(
                self.meir_wingreen_current, axis=0
            )

        self.call_count += 1
