# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the electron solver."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.datastructures.dsdbsparse import _BlockIndexer, _DStackView, _StackView
from qttools.greens_function_solver.solver import BackSubstitutionContext, OBCBlocks
from qttools.profiling import Profiler
from qttools.toeplitz.toeplitz import homogenize, periodize_layer
from qttools.utils.mpi_utils import get_local_slice, get_section_sizes
from qttools.utils.solvers_utils import get_batches
from qttools.utils.stack_utils import scale_stack
from quatrex.bandstructure.band_edges import find_renormalized_eigenvalues
from quatrex.contact.scba import (
    SCBAContact,
    get_inverse_order,
    order_block,
    order_vector,
)
from quatrex.core.config import QuatrexConfig
from quatrex.core.statistics import fermi_dirac
from quatrex.core.subsystem import SubsystemSolver
from quatrex.device import SCBADevice

profiler = Profiler()


def meir_wingreen_current(
    out: NDArray,
) -> Callable[[BackSubstitutionContext], None]:
    """Closure for computing the Meir-Wingreen current.

    This function returns a callback that computes the Meir-Wingreen
    current during the back substitution step of the selected solve. The
    current is computed using the lesser and greater Green's functions
    and the lesser and greater self-energies.

    Parameters
    ----------
    out : NDArray
        Preallocated output array for the current. The shape of the
        array should be (num_batches, num_layers + 1), since this
        includes current between each layer and from/to the leads.

    """

    def callback(ctx: BackSubstitutionContext):
        """Computes the Meir-Wingreen current for the current layer."""
        if (
            comm.block.size == 1
            and 0 <= ctx.i <= len(ctx.obc_blocks.retarded) - 1
            and 0 <= ctx.j <= len(ctx.obc_blocks.retarded) - 1
        ):
            a_ji_dagger = ctx.a_ji.conj().swapaxes(-2, -1)
            a_ji_xr_ii = ctx.a_ji @ ctx.xr_hat_ii
            a_ji_xr_ii_sx_ij = a_ji_xr_ii @ ctx.sigma_lesser_ij
            sigma_lesser_tilde = (
                ctx.a_ji @ ctx.xl_hat_ii @ a_ji_dagger
                + a_ji_xr_ii_sx_ij.conj().swapaxes(-2, -1)
                - a_ji_xr_ii_sx_ij
            )
            a_ji_xr_ii_sx_ij = a_ji_xr_ii @ ctx.sigma_greater_ij
            sigma_greater_tilde = (
                ctx.a_ji @ ctx.xg_hat_ii @ a_ji_dagger
                + a_ji_xr_ii_sx_ij.conj().swapaxes(-2, -1)
                - a_ji_xr_ii_sx_ij
            )
            out[ctx.stack_slice, ..., ctx.j] = xp.trace(
                sigma_greater_tilde @ ctx.xl_jj - ctx.xg_jj @ sigma_lesser_tilde,
                axis1=-2,
                axis2=-1,
            ).real

        # The contact (lead) currents from the boundary self-energies.
        # NOTE: In distributed mode, only the boundary currents are
        # computed. The remaining currents are set to xp.nan outside of
        # this callback.
        if comm.block.rank == 0 and ctx.j == 0:
            out[ctx.stack_slice, ..., 0] = xp.trace(
                ctx.obc_blocks.greater[0][ctx.stack_slice] @ ctx.xl_jj
                - ctx.xg_jj @ ctx.obc_blocks.lesser[0][ctx.stack_slice],
                axis1=-2,
                axis2=-1,
            ).real

        if (
            comm.block.rank == comm.block.size - 1
            and ctx.j == len(ctx.obc_blocks.retarded) - 1
        ):
            # NOTE: Negative sign is needed to get the current flowing
            # in the correct direction (positive from left to right).
            out[ctx.stack_slice, ..., -1] = -xp.trace(
                ctx.obc_blocks.greater[-1][ctx.stack_slice] @ ctx.xl_jj
                - ctx.xg_jj @ ctx.obc_blocks.lesser[-1][ctx.stack_slice],
                axis1=-2,
                axis2=-1,
            ).real

    return callback


def device_current(
    out: NDArray,
    a_hat: DSDBSparse,
) -> Callable[[BackSubstitutionContext], None]:
    """Closure for computing the device current.

    This function returns a callback that computes the device current
    during the back substitution step of the selected solve. The current
    is computed using the lesser Green's function and the *bare* system
    matrix.

    Parameters
    ----------
    out : NDArray
        Preallocated output array for the current. The shape of the
        array should be (num_batches, num_layers - 1), since this only
        includes current between each layer.
    a_hat : DSDBSparse
        Bare system matrix. This is the system matrix without any
        self-energy contributions.

    """

    def callback(ctx: BackSubstitutionContext):
        """Computes the device current for the current layer."""

        if not (
            0 <= ctx.i <= len(ctx.obc_blocks.retarded) - 1
            and 0 <= ctx.j <= len(ctx.obc_blocks.retarded)
        ):
            # NOTE: The j index indeed can go up to
            # len(ctx.obc_blocks.retarded) in distributed mode.
            return

        a_hat_ = a_hat.stack[ctx.stack_slice]

        # Coherent bond current across the interface between block i and
        # j, using the *dense* off-diagonal G^< block (xl_ij) and the
        # full effective coupling from the system matrix (E*S - H).
        a_hat_ij = a_hat_.blocks[ctx.i, ctx.j]
        a_hat_ji = a_hat_.blocks[ctx.j, ctx.i]

        upward = ctx.i > ctx.j
        layer_ind = ctx.j if upward else ctx.i
        global_layer_ind = a_hat.block_section_offsets[comm.block.rank] + layer_ind
        prefactor = -1 if upward else 1

        xl_ji = -ctx.xl_ij.conj().swapaxes(-2, -1)
        out[ctx.stack_slice, ..., global_layer_ind] = xp.trace(
            prefactor * (ctx.xl_ij @ a_hat_ji - a_hat_ij @ xl_ji),
            axis1=-2,
            axis2=-1,
        ).real

    return callback


def _diag_of_product(a: NDArray, b: NDArray) -> NDArray:
    """Computes the diagonal of the matrix product a @ b."""
    return xp.sum(a * b.swapaxes(-2, -1), axis=-1)


@dataclass
class OverlapSpillOver:
    """Lead quantities for the spill-over of the density into a contact.

    Attributes
    ----------
    retarded : NDArray
        The product m_{0,-1} g S_{-1,0}, where 0 is the device block at
        the contact, -1 the first lead block, m the system matrix and g
        the retarded surface Green's function of the lead.
    advanced : NDArray
        The product m_{-1,0}^h g^h S_{-1,0}.
    occupancies : NDArray
        The occupancies of the contact, shaped to broadcast with the
        blocks above.

    """

    retarded: NDArray
    advanced: NDArray
    occupancies: NDArray


def overlap_density(
    out_lesser: NDArray,
    out_greater: NDArray,
    overlap: DSDBSparse,
    spill_overs: list[list[OverlapSpillOver]],
) -> Callable[[BackSubstitutionContext], None]:
    r"""Closure for computing the density in a non-orthogonal basis.

    This function returns a callback that computes the diagonal of
    \(G^{\lessgtr} S\) during the back substitution step of the
    selected solve.

    The stored Green's functions only contain the elements within the
    interaction cutoff, while the overlap can couple orbitals that are
    further apart. The density is therefore computed from the dense
    blocks, before they are truncated. At the contacts, the orbitals
    also overlap with those in the leads.

    Parameters
    ----------
    out_lesser : NDArray
        Preallocated output array for the (imaginary part of the) lesser
        density. The shape of the array should be (num_batches,
        num_orbitals).
    out_greater : NDArray
        Preallocated output array for the (imaginary part of the)
        greater density.
    overlap : DSDBSparse
        The overlap matrix.
    spill_overs : list[list[OverlapSpillOver]]
        The contact spill-overs for each block.

    """
    offsets = overlap.block_offsets

    def callback(ctx: BackSubstitutionContext):
        """Adds the contributions of the current layer to the density."""
        i, j = ctx.i, ctx.j
        slice_j = slice(int(offsets[j]), int(offsets[j + 1]))

        s_jj = overlap.blocks[j, j]
        if i >= 0:
            slice_i = slice(int(offsets[i]), int(offsets[i + 1]))
            s_ij = overlap.blocks[i, j]
            s_ji = overlap.blocks[j, i]

        # NOTE: Lesser and greater lead Green's functions only differ in
        # their occupancies: f for lesser, f - 1 for greater.
        for out, x_ij, x_jj, shift in (
            (out_lesser, ctx.xl_ij, ctx.xl_jj, 0.0),
            (out_greater, ctx.xg_ij, ctx.xg_jj, 1.0),
        ):
            # Same symmetrization as for the stored diagonal blocks.
            x_jj = 0.5 * (x_jj - x_jj.conj().swapaxes(-2, -1))
            out[ctx.stack_slice, ..., slice_j] += _diag_of_product(x_jj, s_jj).imag

            if i >= 0:
                x_ji = -x_ij.conj().swapaxes(-2, -1)
                out[ctx.stack_slice, ..., slice_i] += _diag_of_product(x_ij, s_ji).imag
                out[ctx.stack_slice, ..., slice_j] += _diag_of_product(x_ji, s_ij).imag

            # With the lead in equilibrium,
            #   G^<_{0,-1} = -G^<_{00} m_{-1,0}^h g^h
            #                + f G^r_{00} (m_{0,-1} g - m_{-1,0}^h g^h).
            for spill_over in spill_overs[j]:
                retarded = spill_over.retarded[ctx.stack_slice]
                advanced = spill_over.advanced[ctx.stack_slice]
                occupancies = spill_over.occupancies[ctx.stack_slice] - shift
                out[ctx.stack_slice, ..., slice_j] += (
                    occupancies * _diag_of_product(ctx.xr_jj, retarded - advanced)
                    - _diag_of_product(x_jj, advanced)
                ).imag

    return callback


def get_lead_potential(
    potential: NDArray,
    contact: SCBAContact,
    block_offsets: NDArray,
    global_offset: int,
    mode: str,
) -> NDArray:
    """Gets the potential that enters the OBCs of a contact.

    Parameters
    ----------
    potential : NDArray
        The potential for each local orbital of the device.
    contact : SCBAContact
        The contact for which to get the potential.
    block_offsets : NDArray
        The local block offsets of the device.
    global_offset : int
        The global index of the first local orbital.
    mode : str
        How the potential is continued into the lead. Either "device",
        "unit-cell" or "average" (see `ElectronConfig.lead_potential`).

    Returns
    -------
    NDArray
        The potential, where the device blocks that enter the OBCs (the
        diagonal and the first off-diagonal block) are set to the
        potential of the lead. For "device", this is the unchanged
        input, otherwise it is a copy.

    """
    if mode == "device":
        return potential

    i, j = contact.diagonal_inds[0], contact.upper_inds[1]
    start = int(block_offsets[min(i, j)])
    stop = int(block_offsets[max(i, j) + 1])
    block_size = int(block_offsets[i + 1] - block_offsets[i])

    lead_potential = potential.copy()

    if mode == "average":
        # The outermost layer is the first one in the contact's order.
        layer_size = block_size // contact.transport_repetitions
        layer = order_vector(potential[start:stop], contact.order)[:layer_size]
        lead_potential[start:stop] = layer.mean()
        return lead_potential

    if mode == "unit-cell":
        # NOTE: Equivalent orbitals have the same position in the index
        # arrays of the different unit cells.
        cells = contact.unit_cell_orbital_indices
        cell_potential = potential[cells[0, 0, 0] - global_offset]

        num_layers = contact.transport_repetitions
        ny, nz = contact.transverse_repetition_grid
        # Direction from the outermost block into the device.
        inward_shift = block_size if j > i else -block_size
        for layer, k, l in np.ndindex(num_layers, ny, nz):
            inds = cells[layer, k, l] - global_offset
            lead_potential[inds] = cell_potential
            lead_potential[inds + inward_shift] = cell_potential
        return lead_potential

    raise ValueError(f"Unknown lead potential: {mode}.")


class SystemMatrix(_StackView):
    """Class representing the system matrix for the electron solver.

    Parameters
    ----------
    stack_shape : tuple
        The shape of the stack.
    stack_index : tuple
        The index of the stack.
    energies : NDArray
        The energies at which to solve.
    hamiltonian : DSDBSparse | _DStackView
        The Hamiltonian matrix.
    overlap : DSDBSparse | _DStackView | None, optional
        The overlap matrix.
    sse_lesser : DSDBSparse | _DStackView | None, optional
        The lesser self-energy matrix.
    sse_greater : DSDBSparse | _DStackView | None, optional
        The greater self-energy matrix.
    sse_retarded_hermitian : DSDBSparse | _DStackView | None, optional
        The retarded self-energy matrix.
    potential : NDArray | None, optional
        The potential energy matrix.

    """

    _DELEGATED = (
        "distribution_state",
        "num_blocks",
        "block_sizes",
        "num_local_blocks",
        "block_section_offsets",
    )

    def __init__(
        self,
        stack_shape: tuple,
        stack_index: tuple,
        energies: NDArray,
        hamiltonian: DSDBSparse | _DStackView,
        overlap: DSDBSparse | _DStackView | None = None,
        sse_lesser: DSDBSparse | _DStackView | None = None,
        sse_greater: DSDBSparse | _DStackView | None = None,
        sse_retarded_hermitian: DSDBSparse | _DStackView | None = None,
        potential: NDArray | None = None,
    ) -> None:
        """Initializes the system matrix."""
        super().__init__(stack_shape, stack_index)
        self._energies = energies
        self._hamiltonian = hamiltonian
        self._overlap = overlap
        self._sse_lesser = sse_lesser
        self._sse_greater = sse_greater
        self._sse_retarded_hermitian = sse_retarded_hermitian
        self._potential = potential

    def _reindexed(self, stack_index: tuple) -> "SystemMatrix":
        """Returns a new system matrix with the given stack index."""
        return SystemMatrix(
            self._stack_shape,
            stack_index,
            self._energies[stack_index[0]],
            self._hamiltonian.stack[*((0,) + stack_index[1:])],
            (
                self._overlap.stack[*((0,) + stack_index[1:])]
                if self._overlap is not None
                else None
            ),
            (
                self._sse_lesser.stack[stack_index]
                if self._sse_lesser is not None
                else None
            ),
            (
                self._sse_greater.stack[stack_index]
                if self._sse_greater is not None
                else None
            ),
            (
                self._sse_retarded_hermitian.stack[stack_index]
                if self._sse_retarded_hermitian is not None
                else None
            ),
            self._potential,
        )

    def with_potential(self, potential: NDArray) -> "SystemMatrix":
        """Returns the same system matrix with a different potential.

        Parameters
        ----------
        potential : NDArray
            The potential of the new system matrix.

        Returns
        -------
        SystemMatrix
            The new system matrix. The original one is not changed.

        """
        return SystemMatrix(
            self._stack_shape,
            self._stack_index,
            self._energies,
            self._hamiltonian,
            self._overlap,
            self._sse_lesser,
            self._sse_greater,
            self._sse_retarded_hermitian,
            potential,
        )

    def __getattr__(self, name: str):
        """Delegates attribute access to the Hamiltonian if the
        attribute is in the _DELEGATED list.

        Raises AttributeError if the attribute is not found.

        """
        if name in self._DELEGATED:
            return getattr(self._hamiltonian, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    @property
    def dtype(self) -> xp.dtype:
        """Returns the data type of the system matrix."""
        # NOTE: The Hamiltonian could be potentially real
        # but the system matrix is always complex
        # TODO: Handling cases without self-energy
        return xp.complex128

    def _make_block_indexer(self) -> "SystemMatrixBlockIndexer":
        """Constructs the block indexer for this view."""
        return SystemMatrixBlockIndexer(
            stack_shape=self._stack_shape,
            energies=self._energies,
            hamiltonian=self._hamiltonian,
            overlap=self._overlap,
            sse_lesser=self._sse_lesser,
            sse_greater=self._sse_greater,
            sse_retarded_hermitian=self._sse_retarded_hermitian,
            potential=self._potential,
            stack_index=self._stack_index,
        )


class SystemMatrixBlockIndexer(_BlockIndexer):
    """Block indexer for the System Matrix.

    Parameters
    ----------
    stack_shape : tuple
        The shape of the stack.
    energies : NDArray
        The energies at which to solve.
    hamiltonian : DSDBSparse | _DStackView
        The Hamiltonian matrix.
    overlap : DSDBSparse | _DStackView | None, optional
        The overlap matrix.
    sse_lesser : DSDBSparse | _DStackView | None, optional
        The lesser self-energy matrix.
    sse_greater : DSDBSparse | _DStackView | None, optional
        The greater self-energy matrix.
    sse_retarded_hermitian : DSDBSparse | _DStackView | None, optional
        The retarded self-energy matrix.
    potential : NDArray | None, optional
        The potential energy matrix.
    stack_index : tuple, optional
        The index of the stack.

    """

    def __init__(
        self,
        stack_shape: tuple,
        energies: NDArray,
        hamiltonian: DSDBSparse | _DStackView,
        overlap: DSDBSparse | _DStackView | None = None,
        sse_lesser: DSDBSparse | _DStackView | None = None,
        sse_greater: DSDBSparse | _DStackView | None = None,
        sse_retarded_hermitian: DSDBSparse | _DStackView | None = None,
        potential: NDArray | None = None,
        stack_index: tuple = (Ellipsis,),
    ) -> None:
        """Initializes the System Matrix indexer."""
        super().__init__(stack_index)
        self._stack_shape = stack_shape
        self._energies = energies
        self._hamiltonian = hamiltonian
        self._overlap = overlap
        self._sse_lesser = sse_lesser
        self._sse_greater = sse_greater
        self._sse_retarded_hermitian = sse_retarded_hermitian
        self._potential = potential

    def _normalize_index(self, index: tuple) -> tuple:
        """Normalizes the block index.

        Parameters
        ----------
        index : tuple
            The block index to normalize.

        """
        if self._hamiltonian.distribution_state != "stack":
            raise ValueError(
                "Block indexing is only supported in 'stack' distribution state."
            )
        if self._overlap is not None and self._overlap.distribution_state != "stack":
            raise ValueError(
                "Block indexing is only supported in 'stack' distribution state."
            )
        if (
            self._sse_lesser is not None
            and self._sse_lesser.distribution_state != "stack"
        ):
            raise ValueError(
                "Block indexing is only supported in 'stack' distribution state."
            )
        if (
            self._sse_greater is not None
            and self._sse_greater.distribution_state != "stack"
        ):
            raise ValueError(
                "Block indexing is only supported in 'stack' distribution state."
            )
        if (
            self._sse_retarded_hermitian is not None
            and self._sse_retarded_hermitian.distribution_state != "stack"
        ):
            raise ValueError(
                "Block indexing is only supported in 'stack' distribution state."
            )
        if len(index) != 2:
            raise IndexError("Exactly two block indices are required.")

        row, col = index
        if isinstance(row, slice) or isinstance(col, slice):
            raise NotImplementedError("Slicing is not supported.")

        if row < 0 or col < 0:
            raise IndexError("Negative block indices are not supported.")

        if row >= len(self._hamiltonian.local_block_sizes) or col >= len(
            self._hamiltonian.local_block_sizes
        ):
            raise IndexError("Block index out of bounds.")

        return row, col

    def _apply_overlap(
        self,
        row: int,
        col: int,
        out: NDArray,
    ) -> NDArray:
        """Applies the overlap to the system matrix block.

        Parameters
        ----------
        row : int
            The row index of the block.
        col : int
            The column index of the block.
        out : NDArray
            The block to which the overlap is applied.

        Returns
        -------
        NDArray
            The block with the overlap applied.

        """
        num_dims = len(self._hamiltonian.local_stack_shape)
        if self._overlap is not None:
            overlap = self._overlap.blocks[row, col]
            out = out + self._energies.reshape(
                -1, *((1,) * (num_dims + 1))
            ) * overlap.reshape(1, *self._stack_shape[1:], *out.shape[-2:])
        elif row == col:
            out = out + self._energies.reshape(-1, *((1,) * (num_dims + 1))) * xp.eye(
                out.shape[-1], dtype=out.dtype
            ).reshape(*((1,) * num_dims + out.shape[-2:]))

        return out

    def _apply_potential(
        self,
        row: int,
        col: int,
        out: NDArray,
    ) -> NDArray:
        """Applies the potential to the system matrix block.

        Parameters
        ----------
        row : int
            The row index of the block.
        col : int
            The column index of the block.
        out : NDArray
            The block to which the potential is applied.

        Returns
        -------
        NDArray
            The block with the potential applied.

        """
        if self._potential is not None:
            if self._overlap is not None:
                s_ij = self._overlap.blocks[row, col]
                potential_i = self._potential[
                    self._hamiltonian.local_block_offsets[
                        row
                    ] : self._hamiltonian.local_block_offsets[row + 1]
                ]
                if row == col:
                    out -= (
                        s_ij * potential_i[..., np.newaxis] + s_ij * potential_i
                    ) / 2
                else:
                    potential_j = self._potential[
                        self._hamiltonian.local_block_offsets[
                            col
                        ] : self._hamiltonian.local_block_offsets[col + 1]
                    ]
                    out -= (
                        s_ij * potential_i[..., np.newaxis] + s_ij * potential_j
                    ) / 2
            else:
                if row == col:
                    out -= (
                        xp.eye(out.shape[-1], dtype=out.dtype)
                        * self._potential[
                            self._hamiltonian.local_block_offsets[
                                row
                            ] : self._hamiltonian.local_block_offsets[row + 1]
                        ]
                    )

        return out

    def _apply_self_energy(
        self,
        row: int,
        col: int,
        out: NDArray,
    ) -> NDArray:
        r"""Substracts the self-energy from the system matrix block.

        $$\mathbf{A}_{ij} \mathrel{{-}{=}} \mathbf{\Sigma}^R_{ij} +
        \frac{1}{2} \left(\mathbf{\Sigma}^{>}_{ij} -
        \mathbf{\Sigma}^{<}_{ij} \right)$$

        Note
        ----
        Only substracts when the self-energy is provided. If the
        self-energy is not provided, the block is returned unchanged.

        Parameters
        ----------
        row : int
            The row index of the block.
        col : int
            The column index of the block.
        out : NDArray
            The block to which the self-energy is subtracted.

        Returns
        -------
        NDArray
            The block with the self-energy subtracted.

        """
        if self._sse_retarded_hermitian is not None:
            out = out - self._sse_retarded_hermitian.blocks[row, col]

        if self._sse_lesser is not None:
            out = out + 0.5 * self._sse_lesser.blocks[row, col]

        if self._sse_greater is not None:
            out = out - 0.5 * self._sse_greater.blocks[row, col]

        return out

    def __getitem__(self, index: tuple) -> NDArray:
        """Gets the requested block from the system matrix.

        Parameters
        ----------
        index : tuple
            The block index to retrieve.

        Returns
        -------
        NDArray
            The requested block.

        """
        row, col = self._normalize_index(index)

        out = -self._hamiltonian.blocks[row, col]
        # add extra dimension for the energy
        out = out.reshape(1, *self._stack_shape[1:], *out.shape[-2:])
        out = self._apply_overlap(row, col, out)
        out = self._apply_potential(row, col, out)
        out = self._apply_self_energy(row, col, out)

        return out

    def __setitem__(self, index: tuple, block: NDArray) -> None:
        """Sets the requested block in the data structure."""
        raise NotImplementedError(
            "Setting blocks is not supported in SystemMatrixBlockIndexer."
        )


class ElectronSolver(SubsystemSolver):
    """Solves the electron dynamics.

    Parameters
    ----------
    config : QuatrexConfig
        The quatrex simulation configuration.
    device : SCBADevice
        The device for which to solve the subsystem.
    energies : np.ndarray
        The energies at which to solve.

    """

    system = "electron"

    def __init__(
        self,
        config: QuatrexConfig,
        device: SCBADevice,
        energies: NDArray,
    ) -> None:
        """Initializes the electron solver."""
        super().__init__(config, device, energies)

        self.local_energies = get_local_slice(energies, comm.stack)

        # Will be initialized in the `_assemble_system_matrix` method.
        self.system_matrix = None
        self.bare_system_matrix = None

        self.eta = config.electron.eta
        self.eta_obc = config.electron.eta_obc

        # Contacts.
        self.flatband = config.electron.flatband
        if self.flatband and comm.rank == 0:
            print("Flatband conditions detected", flush=True)

        self.lead_potential = config.electron.lead_potential

        self.compute_meir_wingreen_current = config.outputs.meir_wingreen_currents
        self.compute_device_current = config.outputs.device_currents

        self.dos_peak_limit = config.electron.dos_peak_limit

        # Band edges and Fermi levels.
        self.band_edge_tracking = config.electron.band_edge_tracking

        self.delta_fermi_level_conduction_band = {}
        self.occupancies = {}
        for contact in self.device.contacts:
            if self.band_edge_tracking:
                # TODO: The conduction band edge that gets computed from
                # the contact band structure has a slightly different
                # meaning from what is used for the band edge tracking.
                # See issue #352 for more details.
                self.delta_fermi_level_conduction_band[contact] = (
                    contact.conduction_band_edge - contact.fermi_level
                )
            mu = contact.fermi_level - contact.voltage
            self.occupancies[contact] = fermi_dirac(
                self.local_energies - mu, contact.temperature
            )

        # Prepare Buffers for OBC.
        self.obc_blocks = OBCBlocks(num_blocks=device.hamiltonians.num_local_blocks)

        self.meir_wingreen_current = None
        self.device_current = None

        # Densities in non-orthogonal bases, computed from the dense
        # blocks of the Green's functions. Only supported by the
        # (non-distributed) RGF solver.
        self.compute_overlap_density = (
            device.overlap_matrices is not None
            and config.electron.solver.algorithm == "rgf"
            and comm.block.size == 1
            and (
                config.outputs.electron_density
                or config.outputs.hole_density
                or config.scsp is not None
            )
        )
        self.lesser_density = None
        self.greater_density = None
        self.overlap_spill_overs = [
            [] for __ in range(device.hamiltonians.num_local_blocks)
        ]

        self.call_count = 0
        self.filtering_iteration_limit = config.electron.filtering_iteration_limit

        self.max_batch_size = config.electron.max_batch_size

    def _update_fermi_level(
        self,
        contact: SCBAContact,
        band_edges: NDArray,
    ) -> None:
        """Updates the Fermi levels.

        Note
        ----
        This method overwrites properties of the contact.

        Parameters
        ----------
        contact : SCBAContact
            The contact for which to update the Fermi level.
        band_edges : NDArray
            The band edges.

        """
        contact.mid_gap_energy = xp.mean(band_edges)
        __, conduction_band_edge = band_edges
        contact.fermi_level = (
            conduction_band_edge - self.delta_fermi_level_conduction_band[contact]
        )
        mu = contact.fermi_level - contact.voltage
        self.occupancies[contact] = fermi_dirac(
            self.local_energies - mu, contact.temperature
        )
        if comm.rank == 0:
            print(
                f"{contact.name} conduction band edge: {conduction_band_edge:.6f}\n",
                f"{contact.name} Fermi level: {contact.fermi_level:.6f}",
                flush=True,
            )

    def _compute_contact_obc(
        self,
        contact: SCBAContact,
        contact_str: str,
        occupancies: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Computes the OBC for a specific contact.

        Parameters
        ----------
        contact : SCBAContact
            The contact for which to compute the OBC.
        contact_str : str
            The contact for which to compute the OBC.
            Used for profiling and caching purposes.
        occupancies : NDArray
            The occupancies of the contact at the local energies.

        Returns
        -------
        obc_retarded : NDArray
            The retarded OBC for the contact.
        obc_lesser : NDArray
            The lesser OBC for the contact.
        obc_greater : NDArray
            The greater OBC for the contact.

        """
        obc_solver = contact.obc_solvers["electron"]
        order = contact.order
        block_sections = contact.transport_repetitions
        diagonal_inds = contact.diagonal_inds
        upper_inds = contact.upper_inds
        inverse_order = get_inverse_order(order)

        system_matrix = self.system_matrix
        if self.lead_potential != "device":
            hamiltonians = self.device.hamiltonians
            system_matrix = system_matrix.with_potential(
                get_lead_potential(
                    system_matrix._potential,
                    contact,
                    block_offsets=hamiltonians.local_block_offsets,
                    global_offset=hamiltonians.global_block_offset,
                    mode=self.lead_potential,
                )
            )

        m_10, m_00, m_01 = periodize_layer(
            (
                order_block(system_matrix.blocks[*upper_inds[::-1]], order),
                order_block(system_matrix.blocks[*diagonal_inds], order),
                order_block(system_matrix.blocks[*upper_inds], order),
            ),
            block_sections=block_sections,
        )

        if self.device.overlap_matrices is None:
            s_10 = xp.zeros_like(m_10, dtype=m_10.dtype)
            s_00 = 1j * self.eta_obc * xp.eye(m_00.shape[-1], dtype=m_00.dtype)
            s_01 = xp.zeros_like(m_01, dtype=m_01.dtype)
        else:
            # Extract the overlap matrix blocks.
            s_10 = (
                1j
                * self.eta_obc
                * self.device.overlap_matrices.blocks[*upper_inds[::-1]]
            )
            s_00 = (
                1j * self.eta_obc * self.device.overlap_matrices.blocks[*diagonal_inds]
            )
            s_01 = 1j * self.eta_obc * self.device.overlap_matrices.blocks[*upper_inds]

        # TODO: use residuals to filter "bad" energies
        g_00, *__ = obc_solver(
            (m_10 + s_10, m_00 + s_00, m_01 + s_01),
            contact="G: " + contact_str,
        )
        # Apply the retarded boundary self-energy.
        sigma_00 = m_10 @ g_00 @ m_01
        gamma_00 = 1j * (sigma_00 - sigma_00.conj().swapaxes(-2, -1))

        if self.compute_overlap_density:
            s_01 = self._get_contact_overlap(contact)
            m_01_dagger_g_dagger = (g_00 @ m_01).conj().swapaxes(-2, -1)
            self.overlap_spill_overs[contact.diagonal_inds[0]].append(
                OverlapSpillOver(
                    retarded=order_block(m_10 @ (g_00 @ s_01), inverse_order),
                    advanced=order_block(m_01_dagger_g_dagger @ s_01, inverse_order),
                    occupancies=occupancies.reshape(-1, *((1,) * (m_01.ndim - 2))),
                )
            )

        # Compute and apply the lesser boundary self-energy.
        obc_lesser = 1j * scale_stack(gamma_00.copy(), occupancies)
        # Compute and apply the greater boundary self-energy.
        obc_greater = 1j * scale_stack(gamma_00.copy(), occupancies - 1)

        return (
            order_block(sigma_00, inverse_order),
            order_block(obc_lesser, inverse_order),
            order_block(obc_greater, inverse_order),
        )

    def _get_contact_overlap(self, contact: SCBAContact) -> NDArray:
        """Gets the overlap between the lead and the device.

        Parameters
        ----------
        contact : SCBAContact
            The contact for which to get the overlap.

        Returns
        -------
        s_01 : NDArray
            The (ordered and periodized) overlap block that couples the
            first lead block (rows) to the device (columns).

        """
        overlap = self.device.overlap_matrices
        __, __, s_01 = periodize_layer(
            tuple(
                order_block(overlap.blocks[*inds], contact.order)
                for inds in (
                    contact.upper_inds[::-1],
                    contact.diagonal_inds,
                    contact.upper_inds,
                )
            ),
            block_sections=contact.transport_repetitions,
        )
        return s_01

    @profiler.profile(label="ElectronSolver: OBC", level="default", comm=comm)
    def _compute_obc(self, batch_slice: slice) -> None:
        """Computes open boundary conditions.

        Parameters
        ----------
        batch_slice : slice
            The slice of the energy stack corresponding to the current batch.

        """
        for spill_overs in self.overlap_spill_overs:
            spill_overs.clear()

        for contact in self.device.contacts:
            if comm.block.rank == contact.owning_rank:
                obc_retarded, obc_lesser, obc_greater = self._compute_contact_obc(
                    contact=contact,
                    contact_str=f"{contact.name}-" + str(batch_slice),
                    occupancies=self.occupancies[contact][batch_slice],
                )
                idx = contact.diagonal_inds[0]
                self.obc_blocks.retarded[idx] = obc_retarded
                self.obc_blocks.lesser[idx] = obc_lesser
                self.obc_blocks.greater[idx] = obc_greater

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
        self.system_matrix = SystemMatrix(
            stack_shape=sse_lesser.local_stack_shape,
            stack_index=(...,),
            energies=self.local_energies[batch_slice] + 1j * self.eta,
            hamiltonian=self.device.hamiltonians,
            overlap=self.device.overlap_matrices,
            potential=self.device.potential[
                self.device.hamiltonians.global_block_offset :
            ],
            sse_lesser=sse_lesser,
            sse_greater=sse_greater,
            sse_retarded_hermitian=sse_retarded_hermitian,
        )
        self.bare_system_matrix = SystemMatrix(
            stack_shape=sse_lesser.local_stack_shape,
            stack_index=(...,),
            energies=self.local_energies[batch_slice] + 1j * self.eta,
            hamiltonian=self.device.hamiltonians,
            overlap=self.device.overlap_matrices,
            potential=self.device.potential[
                self.device.hamiltonians.global_block_offset :
            ],
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

        g_retarded_diag = g_retarded.diagonal()
        g_retarded_diag = comm.block.all_gather_v(g_retarded_diag, axis=-1)

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

        if self.compute_overlap_density:
            self.lesser_density[local_mask] = 0.0
            self.greater_density[local_mask] = 0.0

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
                for contact in self.device.contacts:
                    band_edges = xp.empty(2, dtype=float)
                    if comm.block.rank == contact.owning_rank:
                        band_edges = find_renormalized_eigenvalues(
                            hamiltonian=self.device.hamiltonians,
                            overlap=self.device.overlap_matrices,
                            potential=self.device.potential,
                            sigma_retarded_hermitian=sse_retarded_hermitian,
                            energies=self.energies,
                            conduction_band_guess=contact.fermi_level
                            + self.delta_fermi_level_conduction_band[contact],
                            mid_gap_energy=contact.mid_gap_energy,
                            diagonal_inds=contact.diagonal_inds,
                            upper_inds=contact.upper_inds,
                            order=contact.order,
                            block_sections=contact.transport_repetitions,
                            band_edge_config=self.config.compute.band_edge,
                        )
                    comm.block.bcast(band_edges, root=contact.owning_rank)
                    self._update_fermi_level(contact, band_edges)

        if self.max_batch_size is None:
            max_batch_size = sse_lesser.shape[0]
        else:
            max_batch_size = self.max_batch_size

        batch_sizes, batch_offsets = get_batches(sse_lesser.shape[0], max_batch_size)

        if self.compute_meir_wingreen_current:
            self.meir_wingreen_current = xp.zeros(
                (*sse_lesser.local_stack_shape, sse_lesser.num_blocks + 1),
                dtype=xp.float64,
            )
        if self.compute_device_current:
            self.device_current = xp.zeros(
                (*sse_lesser.local_stack_shape, sse_lesser.num_blocks - 1),
                dtype=xp.float64,
            )

        if self.compute_overlap_density:
            density_shape = (*sse_lesser.local_stack_shape, sse_lesser.shape[-1])
            self.lesser_density = xp.zeros(density_shape, dtype=xp.float64)
            self.greater_density = xp.zeros(density_shape, dtype=xp.float64)

        domain_distributed = comm.block.size > 1
        solver = self.solver_dist if domain_distributed else self.solver

        for i in range(len(batch_sizes)):
            batch_slice = slice(int(batch_offsets[i]), int(batch_offsets[i + 1]))
            sse_lesser_batch = sse_lesser.stack[batch_slice]
            sse_greater_batch = sse_greater.stack[batch_slice]
            sse_retarded_hermitian_batch = sse_retarded_hermitian.stack[batch_slice]

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

                callback_list = []
                if self.compute_meir_wingreen_current:
                    callback_list.append(
                        meir_wingreen_current(self.meir_wingreen_current[batch_slice])
                    )
                if self.compute_device_current:
                    callback_list.append(
                        device_current(
                            self.device_current[batch_slice], self.bare_system_matrix
                        )
                    )
                if self.compute_overlap_density:
                    callback_list.append(
                        overlap_density(
                            self.lesser_density[batch_slice],
                            self.greater_density[batch_slice],
                            self.device.overlap_matrices,
                            self.overlap_spill_overs,
                        )
                    )

                solver.selected_solve(
                    a=self.system_matrix,
                    sigma_lesser=sse_lesser_batch,
                    sigma_greater=sse_greater_batch,
                    obc_blocks=self.obc_blocks,
                    out=out_slice,
                    return_retarded=True,
                    callbacks=callback_list,
                )

        # In the domain-distributed case, we need to allreduce the
        # current across the block communicator to get the total current
        # for each layer. NOTE: We use allreduce instead of allgather
        # since every rank allocates the full current.
        if self.compute_meir_wingreen_current and domain_distributed:
            # TODO: Only boundary currents are currently supported in
            # distributed mode. Invalidate the remaining layers by
            # setting them to xp.nan.
            self.meir_wingreen_current[..., 1:-1] = xp.nan

            total_meir_wingreen_current = xp.zeros_like(self.meir_wingreen_current)
            comm.block.all_reduce(
                self.meir_wingreen_current, total_meir_wingreen_current, op="sum"
            )
            self.meir_wingreen_current = total_meir_wingreen_current

        if self.compute_device_current and domain_distributed:
            total_device_current = xp.zeros_like(self.device_current)
            comm.block.all_reduce(self.device_current, total_device_current, op="sum")
            self.device_current = total_device_current

        with profiler.profile_range(
            label="ElectronSolver: Filter", level="default", comm=comm
        ):
            if self.call_count < self.filtering_iteration_limit:
                self._filter_peaks(out)

        self.call_count += 1
