from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from quatrex.core.config import QuatrexConfig
from quatrex.core.statistics import bose_einstein
from quatrex.device.inputs import (
    _create_matrix_from_unit_cells,
    trim_tight_binding_matrix,
)

from .equilibrium_screening import EquilibriumScreening
from .rpa_compute import BrillouinZoneMesh, build_uniform_brillouin_zone_mesh


@dataclass(frozen=True)
class EquilibriumRPABridgeResult:
    """Cached real-space screened interactions for the NEGF Coulomb SSE path."""

    w_retarded_matrices: list[sparse.coo_matrix]
    w_lesser_matrices: list[sparse.coo_matrix]
    w_greater_matrices: list[sparse.coo_matrix]


class EquilibriumRPAScreeningBridge:
    """Bridge equilibrium Bloch-RPA screening onto the NEGF screened-interaction buffers.

    The initial implementation is intentionally conservative:

    - it only supports device inputs constructed from a unit cell,
    - it requires no transverse k-point sampling,
    - it assumes the RPA periodic axis is the transport axis.

    These guards keep the new path isolated from the established NEGF workflow.
    """

    def __init__(
        self,
        config: QuatrexConfig,
        screening_energies: np.ndarray,
        template: DSDBSparse,
    ) -> None:
        self.config = config
        self.screening_energies = np.asarray(screening_energies, dtype=np.float64)
        self.template = template
        self._solver = EquilibriumScreening(
            matrix_polarization=getattr(
                config.coulomb_screening,
                "matrix_valued_polarization",
                False,
            ),
            frequency_axis="real",
        )
        self._cached_result: EquilibriumRPABridgeResult | None = None

    def populate(
        self,
        w_retarded: DSDBSparse | None,
        w_lesser: DSDBSparse,
        w_greater: DSDBSparse,
    ) -> None:
        """Populate the distributed screened-interaction tensors used by SCBA."""

        result = self._cached_result or self._build_cached_result()
        self._cached_result = result

        if w_retarded is not None and w_retarded._data is None:
            w_retarded.allocate_data()
        if w_lesser._data is None:
            w_lesser.allocate_data()
        if w_greater._data is None:
            w_greater.allocate_data()

        if w_retarded is not None:
            w_retarded.data[:] = 0.0
        w_lesser.data[:] = 0.0
        w_greater.data[:] = 0.0

        local_energy_count = int(self.template.stack_section_sizes[self._stack_rank])
        energy_offset = int(
            np.sum(self.template.stack_section_sizes[: self._stack_rank])
        )

        for local_index in range(local_energy_count):
            global_index = energy_offset + local_index
            if w_retarded is not None:
                w_retarded.stack[(local_index,)] = result.w_retarded_matrices[
                    global_index
                ]
            w_lesser.stack[(local_index,)] = result.w_lesser_matrices[global_index]
            w_greater.stack[(local_index,)] = result.w_greater_matrices[global_index]

    @property
    def _stack_rank(self) -> int:
        return int(comm.stack.rank)

    def _build_cached_result(self) -> EquilibriumRPABridgeResult:
        result = None
        if global_comm.rank == 0:
            print("Computing equilibrium RPA screening cache...", flush=True)
            result = self._build_cached_result_on_root()
            print("Finished equilibrium RPA screening cache.", flush=True)

        result = global_comm.bcast(result, root=0)
        return result

    def _build_cached_result_on_root(self) -> EquilibriumRPABridgeResult:
        self._validate_supported_configuration()

        mesh = self._build_mesh()
        chemical_potential = self._resolve_chemical_potential()
        print(
            "RPA cache: loading Hamiltonian and Coulomb inputs "
            f"(nk={mesh.k_points.size}, nq={mesh.q_points.size}, "
            f"nw={mesh.frequencies.size})...",
            flush=True,
        )
        inputs = self._solver.load_inputs_from_config(
            self.config,
            hamiltonian_matrix_name=self.config.coulomb_screening.hamiltonian_matrix_name,
            coulomb_matrix_name=self.config.coulomb_screening.coulomb_matrix_name,
        )
        scaled_inputs = type(inputs)(
            hamiltonian_blocks=inputs.hamiltonian_blocks,
            coulomb_blocks={
                translation: block / self.config.coulomb_screening.epsilon_r
                for translation, block in inputs.coulomb_blocks.items()
            },
        )

        print(
            "RPA cache: computing screened interactions on q/frequency grid...",
            flush=True,
        )
        grid_result = self._solver.solve_grid_from_inputs(
            inputs=scaled_inputs,
            mesh=mesh,
            chemical_potential=chemical_potential,
            temperature=self.config.coulomb_screening.temperature,
            periodic_axis=self.config.coulomb_screening.periodic_axis,
            lattice_constant=self.config.coulomb_screening.lattice_constant,
            broadening=self.config.coulomb_screening.broadening,
        )

        print("RPA cache: building lesser/greater spectral functions...", flush=True)
        w_spectral_function = grid_result.screened_interactions - np.swapaxes(
            grid_result.screened_interactions.conj(), -1, -2
        )
        p_retarded_qw = np.asarray(
            grid_result.polarization_result.polarization,
            dtype=np.complex128,
        )
        if p_retarded_qw.ndim != 4:
            raise NotImplementedError(
                "RPA environment screening requires matrix_valued_polarization=True."
            )
        bose = np.asarray(
            bose_einstein(
                xp.asarray(self.screening_energies),
                self.config.coulomb_screening.temperature,
            )
        )
        bose = bose.astype(np.complex128, copy=False)[
            np.newaxis, :, np.newaxis, np.newaxis
        ]
        w_lesser_qw = bose * w_spectral_function
        w_greater_qw = (1.0 + bose) * w_spectral_function

        print("RPA cache: transforming retarded W to transport matrices...", flush=True)
        w_retarded_matrices = self._build_transport_matrices(
            mesh,
            grid_result.screened_interactions,
        )
        print("RPA cache: transforming lesser W to transport matrices...", flush=True)
        w_lesser_matrices = self._build_transport_matrices(mesh, w_lesser_qw)
        print("RPA cache: transforming greater W to transport matrices...", flush=True)
        w_greater_matrices = self._build_transport_matrices(mesh, w_greater_qw)
        return EquilibriumRPABridgeResult(
            w_retarded_matrices=w_retarded_matrices,
            w_lesser_matrices=w_lesser_matrices,
            w_greater_matrices=w_greater_matrices,
        )

    def _build_mesh(self) -> BrillouinZoneMesh:
        base_mesh = build_uniform_brillouin_zone_mesh(
            num_k_points=self.config.coulomb_screening.num_k_points,
            num_q_points=self.config.coulomb_screening.num_q_points,
            num_frequencies=1,
            max_frequency=0.0,
            lattice_constant=self.config.coulomb_screening.lattice_constant,
            include_zero_q=self.config.coulomb_screening.include_zero_q,
        )
        return BrillouinZoneMesh(
            k_points=base_mesh.k_points,
            q_points=base_mesh.q_points,
            frequencies=self.screening_energies,
        )

    def _build_transport_matrices(
        self,
        mesh: BrillouinZoneMesh,
        screened_interactions_qw: np.ndarray,
    ) -> list[sparse.coo_matrix]:
        transport_axis = "xyz".index(self.config.device.transport_direction)
        periodic_axis = self.config.coulomb_screening.periodic_axis
        if periodic_axis is None:
            periodic_axis = transport_axis

        q_points = np.asarray(mesh.q_points, dtype=np.float64)
        nq = q_points.size
        max_translation = nq // 2
        translations = np.arange(-max_translation, max_translation + 1, dtype=int)
        axis_sizes = [1, 1, 1]
        axis_sizes[periodic_axis] = translations.size

        matrices: list[sparse.coo_matrix] = []
        for frequency_index in range(screened_interactions_qw.shape[1]):
            q_slice = screened_interactions_qw[:, frequency_index]
            unit_cells = np.zeros(
                tuple(axis_sizes) + q_slice.shape[-2:],
                dtype=np.complex128,
            )
            for translation in translations:
                block = (
                    np.sum(
                        np.exp(
                            -1j
                            * q_points
                            * translation
                            * self.config.coulomb_screening.lattice_constant
                        )[:, np.newaxis, np.newaxis]
                        * q_slice,
                        axis=0,
                        dtype=np.complex128,
                    )
                    / nq
                )
                block_index = [0, 0, 0]
                block_index[periodic_axis] = translation + max_translation
                unit_cells[tuple(block_index)] = block

            # Match the device-side transport-cell construction by trimming the
            # Fourier-reconstructed translation range back to the configured
            # neighbor-cell cutoff before expanding into a transport matrix.
            unit_cells = trim_tight_binding_matrix(
                tight_binding_matrix=unit_cells,
                neighbor_cell_cutoff=self.config.device.neighbor_cell_cutoff,
            )
            matrix_sparray, __, __ = _create_matrix_from_unit_cells(
                self.config, unit_cells
            )
            matrices.append(matrix_sparray.astype(xp.complex128))

        return matrices

    def _resolve_chemical_potential(self) -> float:
        if self.config.coulomb_screening.chemical_potential is not None:
            return float(self.config.coulomb_screening.chemical_potential)
        if self.config.electron.fermi_level is not None:
            return float(self.config.electron.fermi_level)
        return float(self.config.electron.left_fermi_level)

    def _validate_supported_configuration(self) -> None:
        if not self.config.device.construct_from_unit_cell:
            raise NotImplementedError(
                "RPA screening under NEGF currently supports only construct_from_unit_cell=True."
            )

        transport_axis = "xyz".index(self.config.device.transport_direction)
        periodic_axis = self.config.coulomb_screening.periodic_axis
        if periodic_axis is None:
            periodic_axis = transport_axis
        if periodic_axis != transport_axis:
            raise NotImplementedError(
                "RPA screening under NEGF currently requires the RPA periodic axis to match the transport direction."
            )

        transverse_k_grid = list(self.config.device.kpoint_grid)
        transverse_k_grid.pop(transport_axis)
        if any(k > 1 for k in transverse_k_grid):
            raise NotImplementedError(
                "RPA screening under NEGF currently supports only devices without transverse k-point sampling."
            )

        if not self.config.coulomb_screening.include_zero_q:
            raise NotImplementedError(
                "RPA screening under NEGF currently requires include_zero_q=True."
            )
