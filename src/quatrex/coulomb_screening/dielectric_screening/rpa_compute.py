r"""Brillouin-zone RPA polarization from a translation-resolved Hamiltonian.

This module implements the Hamiltonian-to-polarization part of the dielectric
screening workflow:

1. Load a translation-resolved unit-cell Hamiltonian.
2. Assemble and diagonalize the Bloch Hamiltonian on a Brillouin-zone mesh.
3. Build occupations and band-overlap form factors on that mesh.
4. Evaluate the RPA polarization entering later screening calculations.

The implementation follows

.. math::

	\Pi(q, i\omega) = \sum_k \sum_{\gamma,\gamma'}
	\frac{(f_{k+q}^{\gamma'} - f_k^\gamma) F_{k, k+q}^{\gamma \gamma'}}
	{E_{k+q}^{\gamma'} - E_k^\gamma - i\omega}.

The arrays are expected to be provided in a form that already encodes the
momentum transfer ``q`` through the ``k + q`` quantities. Bare Coulomb
interactions and screened interactions are handled elsewhere in the package.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.io
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from quatrex.core.config import QuatrexConfig


FrequencyAxis = Literal["imaginary", "real"]


@dataclass(frozen=True)
class BlochBandStructure:
    """Band structure obtained by diagonalizing a Bloch Hamiltonian on a 1D grid."""

    k_points: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]


@dataclass(frozen=True)
class PolarizationResult:
    """Container for the band structure and the resulting RPA polarization."""

    band_structure: BlochBandStructure
    polarization: NDArray[np.complex128]


@dataclass(frozen=True)
class BrillouinZoneMesh:
    """Discrete Brillouin-zone mesh and response grid for Coulomb screening."""

    k_points: NDArray[np.float64]
    q_points: NDArray[np.float64]
    frequencies: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "k_points", _as_float64(self.k_points, name="k_points")
        )
        object.__setattr__(
            self, "q_points", _as_float64(self.q_points, name="q_points")
        )
        object.__setattr__(
            self,
            "frequencies",
            _as_float64(self.frequencies, name="frequencies"),
        )
        if self.k_points.ndim != 1:
            raise ValueError("k_points must be a one-dimensional array.")
        if self.q_points.ndim != 1:
            raise ValueError("q_points must be a one-dimensional array.")
        if self.frequencies.ndim != 1:
            raise ValueError("frequencies must be a one-dimensional array.")


@dataclass(frozen=True)
class ScreeningChannels:
    """Degeneracy factors for channels not explicitly resolved in the Hamiltonian basis."""

    spin_degeneracy: float = 1.0
    valley_degeneracy: float = 1.0

    def __post_init__(self) -> None:
        if self.spin_degeneracy <= 0.0:
            raise ValueError("spin_degeneracy must be positive.")
        if self.valley_degeneracy <= 0.0:
            raise ValueError("valley_degeneracy must be positive.")

    @property
    def multiplicity(self) -> float:
        return float(self.spin_degeneracy * self.valley_degeneracy)


_TRANSLATION_KEY_PATTERN = re.compile(
    r"^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]$"
)


def _as_complex128(array: ArrayLike, *, name: str) -> NDArray[np.complex128]:
    values = np.asarray(array)
    if values.ndim == 0:
        raise ValueError(f"{name} must be at least one-dimensional.")
    return values.astype(np.complex128, copy=False)


def _as_float64(array: ArrayLike, *, name: str) -> NDArray[np.float64]:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError(f"{name} must be at least one-dimensional.")
    return values


def _parse_translation_key(key: str) -> tuple[int, int, int]:
    match = _TRANSLATION_KEY_PATTERN.match(key)
    if match is None:
        raise ValueError(f"Unsupported translation-block key: {key!r}")
    return tuple(int(group) for group in match.groups())


def load_translation_blocks(
    mat_file: str | Path,
) -> dict[tuple[int, int, int], NDArray[np.complex128]]:
    """Load translation-resolved matrices from a MATLAB .mat file.

    The expected key format is ``[Rx, Ry, Rz]``, matching the Wannier-like unit-cell
    files in the carbon-nanotube examples.
    """

    data = scipy.io.loadmat(Path(mat_file))
    blocks: dict[tuple[int, int, int], NDArray[np.complex128]] = {}
    for key, value in data.items():
        if key.startswith("__"):
            continue
        translation = _parse_translation_key(key)
        array = np.asarray(value, dtype=np.complex128)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(
                f"Block {key!r} in {mat_file} must be a square matrix, got shape {array.shape}."
            )
        blocks[translation] = array

    if not blocks:
        raise ValueError(f"No translation blocks found in {mat_file}.")
    return blocks


def resolve_unit_cell_matrix_path(
    config: QuatrexConfig,
    matrix_name: str,
) -> Path:
    """Resolve a unit-cell matrix path from the Quatrex input directory."""

    return Path(config.input_dir) / f"{matrix_name}.mat"


def load_translation_blocks_from_config(
    config: QuatrexConfig,
    *,
    matrix_name: str = "hamiltonian",
) -> dict[tuple[int, int, int], NDArray[np.complex128]]:
    """Load translation blocks for a unit-cell matrix from ``config.input_dir``."""

    return load_translation_blocks(resolve_unit_cell_matrix_path(config, matrix_name))


def infer_periodic_axis(
    blocks: dict[tuple[int, int, int], NDArray[np.complex128]]
) -> int:
    """Infer the periodic axis from the nonzero translation vectors in the blocks."""

    varying_axes = [
        axis
        for axis in range(3)
        if len({translation[axis] for translation in blocks}) > 1
    ]
    if len(varying_axes) != 1:
        raise ValueError(
            "Could not infer a unique periodic axis from the translation blocks; "
            "specify periodic_axis explicitly."
        )
    return varying_axes[0]


def build_bloch_hamiltonian(
    translation_blocks: dict[tuple[int, int, int], NDArray[np.complex128]],
    k_points: ArrayLike,
    *,
    periodic_axis: int | None = None,
    lattice_constant: float = 1.0,
) -> NDArray[np.complex128]:
    """Assemble the Bloch Hamiltonian ``H(k) = sum_R H_R exp(i k R a)``.

    The ``k_points`` should be given in reciprocal units consistent with the chosen
    lattice constant. For the nanotube unit-cell example, the natural periodic axis is
    ``z`` and only the third translation component varies.
    """

    k_points_array = _as_float64(k_points, name="k_points")
    if k_points_array.ndim != 1:
        raise ValueError("k_points must be a one-dimensional array.")
    if periodic_axis is None:
        periodic_axis = infer_periodic_axis(translation_blocks)
    if periodic_axis not in (0, 1, 2):
        raise ValueError("periodic_axis must be 0, 1, or 2.")

    first_block = next(iter(translation_blocks.values()))
    norb = first_block.shape[0]
    bloch_hamiltonian = np.zeros((k_points_array.size, norb, norb), dtype=np.complex128)
    for translation, block in translation_blocks.items():
        phase = np.exp(
            1j * k_points_array * translation[periodic_axis] * lattice_constant
        )
        bloch_hamiltonian += phase[:, np.newaxis, np.newaxis] * block[np.newaxis, :, :]
    return bloch_hamiltonian


def diagonalize_bloch_hamiltonian(
    bloch_hamiltonian: ArrayLike,
    k_points: ArrayLike,
) -> BlochBandStructure:
    """Diagonalize ``H(k)`` on a 1D ``k`` grid."""

    bloch_hamiltonian_array = _as_complex128(
        bloch_hamiltonian, name="bloch_hamiltonian"
    )
    k_points_array = _as_float64(k_points, name="k_points")
    if bloch_hamiltonian_array.ndim != 3:
        raise ValueError("bloch_hamiltonian must have shape (nk, norb, norb).")
    if bloch_hamiltonian_array.shape[0] != k_points_array.size:
        raise ValueError("k_points must match the first axis of bloch_hamiltonian.")

    eigenvalues, eigenvectors = np.linalg.eigh(bloch_hamiltonian_array)
    return BlochBandStructure(
        k_points=k_points_array,
        eigenvalues=eigenvalues.astype(np.float64, copy=False),
        eigenvectors=eigenvectors.astype(np.complex128, copy=False),
    )


def fermi_dirac_distribution(
    energies: ArrayLike,
    *,
    chemical_potential: float,
    temperature: float,
    boltzmann_constant: float = 8.617333262145e-5,
) -> NDArray[np.float64]:
    """Return Fermi-Dirac occupations for energies in eV and temperature in K."""

    energies_array = _as_float64(energies, name="energies")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")
    if temperature == 0.0:
        occupations = np.zeros_like(energies_array)
        occupations[energies_array < chemical_potential] = 1.0
        occupations[energies_array == chemical_potential] = 0.5
        return occupations

    beta = 1.0 / (boltzmann_constant * temperature)
    argument = np.clip((energies_array - chemical_potential) * beta, -700.0, 700.0)
    return 1.0 / (np.exp(argument) + 1.0)


def map_q_points_to_k_shifts(
    q_points: ArrayLike,
    k_points: ArrayLike,
    *,
    tolerance: float = 1e-8,
) -> NDArray[np.int64]:
    """Map momentum transfers onto discrete index shifts of a uniform periodic k grid."""

    q_points_array = _as_float64(q_points, name="q_points")
    k_points_array = _as_float64(k_points, name="k_points")
    if q_points_array.ndim != 1:
        raise ValueError("q_points must be a one-dimensional array.")
    if k_points_array.ndim != 1 or k_points_array.size < 2:
        raise ValueError(
            "k_points must be a one-dimensional array with at least two points."
        )

    dk = np.diff(k_points_array)
    if not np.allclose(dk, dk[0], atol=tolerance, rtol=0.0):
        raise ValueError("k_points must form a uniform grid.")

    shift_float = q_points_array / dk[0]
    shifts = np.rint(shift_float).astype(np.int64)
    if not np.allclose(shift_float, shifts, atol=tolerance, rtol=0.0):
        raise ValueError(
            "Each q point must correspond to an integer multiple of the k-grid spacing."
        )
    return shifts


def build_uniform_brillouin_zone_mesh(
    *,
    num_k_points: int,
    num_frequencies: int,
    max_frequency: float,
    num_q_points: int | None = None,
    lattice_constant: float = 1.0,
    include_zero_q: bool = True,
) -> BrillouinZoneMesh:
    """Build a uniform 1D Brillouin-zone mesh compatible with the RPA solver.

    The Bloch phase convention in :func:`build_bloch_hamiltonian` is periodic under
    ``k -> k + 2 pi / lattice_constant``. This helper samples a single 1D Brillouin
    zone on a uniform grid and constructs a commensurate set of momentum transfers
    ``q`` as integer multiples of the ``k`` spacing, which is required by
    :func:`map_q_points_to_k_shifts`.
    """

    if num_k_points < 2:
        raise ValueError("num_k_points must be at least 2.")
    if num_q_points is not None and num_q_points < 1:
        raise ValueError("num_q_points must be positive when provided.")
    if num_frequencies < 1:
        raise ValueError("num_frequencies must be at least 1.")
    if max_frequency < 0.0:
        raise ValueError("max_frequency must be non-negative.")
    if lattice_constant <= 0.0:
        raise ValueError("lattice_constant must be positive.")

    bz_width = 2.0 * np.pi / lattice_constant
    dk = bz_width / num_k_points
    k_points = np.linspace(
        -np.pi / lattice_constant,
        np.pi / lattice_constant - dk,
        num_k_points,
        dtype=np.float64,
    )

    full_q_shifts = np.arange(
        -num_k_points // 2, (num_k_points + 1) // 2, dtype=np.int64
    )
    full_q_points = full_q_shifts.astype(np.float64) * dk
    if not include_zero_q:
        full_q_points = full_q_points[full_q_points != 0.0]

    if num_q_points is None or num_q_points >= full_q_points.size:
        q_points = full_q_points
    else:
        center = full_q_points.size // 2
        half_window = num_q_points // 2
        start = max(center - half_window, 0)
        stop = start + num_q_points
        q_points = full_q_points[start:stop]
        if q_points.size != num_q_points:
            raise ValueError(
                "num_q_points is too large for the requested zero-q handling."
            )

    frequencies = np.linspace(0.0, max_frequency, num_frequencies, dtype=np.float64)
    return BrillouinZoneMesh(
        k_points=k_points,
        q_points=q_points,
        frequencies=frequencies,
    )


def compute_band_overlap_form_factors(
    eigenvectors: ArrayLike,
    q_shifts: ArrayLike,
) -> NDArray[np.float64]:
    """Compute ``|<u_{k,b}|u_{k+q,b'}>|^2`` on a discrete 1D k grid."""

    eigenvectors_array = _as_complex128(eigenvectors, name="eigenvectors")
    q_shifts_array = np.asarray(q_shifts, dtype=np.int64)
    if eigenvectors_array.ndim != 3:
        raise ValueError("eigenvectors must have shape (nk, norb, nbands).")

    nk = eigenvectors_array.shape[0]
    shifted_indices = (
        np.arange(nk, dtype=np.int64)[np.newaxis, :] + q_shifts_array[:, np.newaxis]
    ) % nk
    shifted_vectors = eigenvectors_array[shifted_indices]
    overlaps = np.einsum(
        "kib,qkic->qkbc", eigenvectors_array.conj(), shifted_vectors, optimize=True
    )
    return np.abs(overlaps) ** 2


def compute_rpa_polarization_from_bands(
    band_structure: BlochBandStructure,
    q_points: ArrayLike,
    frequencies: ArrayLike,
    *,
    chemical_potential: float,
    temperature: float,
    state_multiplicity: float = 1.0,
    broadening: float = 0.0,
    frequency_axis: FrequencyAxis = "imaginary",
) -> NDArray[np.complex128]:
    """Compute RPA polarization from a 1D Bloch band structure on a uniform grid."""

    q_shifts = map_q_points_to_k_shifts(q_points, band_structure.k_points)
    occupations = fermi_dirac_distribution(
        band_structure.eigenvalues,
        chemical_potential=chemical_potential,
        temperature=temperature,
    )
    nk = band_structure.k_points.size
    shifted_indices = (
        np.arange(nk, dtype=np.int64)[np.newaxis, :] + q_shifts[:, np.newaxis]
    ) % nk
    shifted_energies = band_structure.eigenvalues[shifted_indices]
    shifted_occupations = occupations[shifted_indices]
    form_factors = compute_band_overlap_form_factors(
        band_structure.eigenvectors, q_shifts
    )
    polarization = compute_rpa_polarization(
        band_structure.eigenvalues,
        shifted_energies,
        occupations,
        shifted_occupations,
        form_factors,
        frequencies,
        broadening=broadening,
        frequency_axis=frequency_axis,
    )
    return state_multiplicity * polarization


def compute_rpa_polarization_matrix_from_bands(
    band_structure: BlochBandStructure,
    q_points: ArrayLike,
    frequencies: ArrayLike,
    *,
    chemical_potential: float,
    temperature: float,
    state_multiplicity: float = 1.0,
    broadening: float = 0.0,
    frequency_axis: FrequencyAxis = "imaginary",
) -> NDArray[np.complex128]:
    """Compute orbital-resolved RPA polarization matrices from Bloch states.

    The returned array has shape ``(nq, n_omega, norb, norb)``. Summing over the
    last two axes recovers the scalar band-overlap RPA polarization computed by
    :func:`compute_rpa_polarization_from_bands`.
    """

    q_shifts = map_q_points_to_k_shifts(q_points, band_structure.k_points)
    occupations = fermi_dirac_distribution(
        band_structure.eigenvalues,
        chemical_potential=chemical_potential,
        temperature=temperature,
    )
    nk = band_structure.k_points.size
    shifted_indices = (
        np.arange(nk, dtype=np.int64)[np.newaxis, :] + q_shifts[:, np.newaxis]
    ) % nk
    shifted_energies = band_structure.eigenvalues[shifted_indices]
    shifted_occupations = occupations[shifted_indices]
    polarization = compute_rpa_polarization_matrix(
        band_structure.eigenvalues,
        shifted_energies,
        occupations,
        shifted_occupations,
        band_structure.eigenvectors,
        q_shifts,
        frequencies,
        broadening=broadening,
        frequency_axis=frequency_axis,
    )
    return state_multiplicity * polarization


def _rpa_denominator(
    energy_differences: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    *,
    broadening: float,
    frequency_axis: FrequencyAxis,
) -> NDArray[np.complex128]:
    """Build the RPA response denominator for imaginary or real frequency."""

    frequency_shape = (1,) * energy_differences.ndim + (frequencies.size,)
    frequency_grid = frequencies.reshape(frequency_shape)
    if frequency_axis == "imaginary":
        return energy_differences[..., np.newaxis] - 1j * (frequency_grid + broadening)
    if frequency_axis == "real":
        return energy_differences[..., np.newaxis] - frequency_grid - 1j * broadening
    raise ValueError("frequency_axis must be either 'imaginary' or 'real'.")


def compute_rpa_polarization(
    energies_k: ArrayLike,
    energies_kq: ArrayLike,
    occupations_k: ArrayLike,
    occupations_kq: ArrayLike,
    form_factors: ArrayLike,
    frequencies: ArrayLike,
    *,
    k_weights: ArrayLike | None = None,
    normalize_k_sum: bool = True,
    broadening: float = 0.0,
    frequency_axis: FrequencyAxis = "imaginary",
) -> NDArray[np.complex128]:
    r"""Compute the RPA polarization for a set of momenta and frequencies.

    Parameters
    ----------
    energies_k : array-like, shape (nk, nbands)
            Band energies :math:`E_k^\gamma`.
    energies_kq : array-like, shape (nq, nk, nbands)
            Shifted band energies :math:`E_{k+q}^{\gamma'}` for each momentum transfer.
    occupations_k : array-like, shape (nk, nbands)
            Occupations :math:`f_k^\gamma`.
    occupations_kq : array-like, shape (nq, nk, nbands)
            Shifted occupations :math:`f_{k+q}^{\gamma'}`.
    form_factors : array-like, shape (nq, nk, nbands, nbands)
            Form factors :math:`F_{k, k+q}^{\gamma\gamma'}`.
    frequencies : array-like, shape (n_omega,)
            Frequency grid. For imaginary-axis RPA this is the real-valued Matsubara
            frequency grid entering as :math:`i\omega`. For real-axis RPA this is
            the real energy-transfer grid :math:`\omega`.
    k_weights : array-like, optional, shape (nk,)
            Weights for the Brillouin-zone sum over ``k``.
    normalize_k_sum : bool, default True
            Divide the final ``k`` sum by ``nk`` when no explicit ``k_weights`` are
            provided.
    broadening : float, default 0.0
            Positive infinitesimal. On the imaginary axis this preserves the legacy
            ``-i (omega + broadening)`` denominator; on the real axis it enters as
            the retarded ``-i eta`` term.
    frequency_axis : {"imaginary", "real"}, default "imaginary"
            Select the response denominator. ``"real"`` computes the retarded
            real-frequency response used by the NEGF bridge.

    Returns
    -------
    numpy.ndarray, shape (nq, n_omega)
            The complex polarization :math:`\Pi(q, i\omega)`.
    """

    energies_k_array = _as_float64(energies_k, name="energies_k")
    energies_kq_array = _as_float64(energies_kq, name="energies_kq")
    occupations_k_array = _as_float64(occupations_k, name="occupations_k")
    occupations_kq_array = _as_float64(occupations_kq, name="occupations_kq")
    form_factors_array = _as_complex128(form_factors, name="form_factors")
    frequencies_array = _as_float64(frequencies, name="frequencies")

    if energies_k_array.ndim != 2:
        raise ValueError("energies_k must have shape (nk, nbands).")
    if occupations_k_array.shape != energies_k_array.shape:
        raise ValueError("occupations_k must match energies_k shape.")

    nk, nbands = energies_k_array.shape

    expected_shifted_shape = (energies_kq_array.shape[0], nk, nbands)
    if energies_kq_array.ndim != 3 or energies_kq_array.shape[1:] != (nk, nbands):
        raise ValueError(
            "energies_kq must have shape (nq, nk, nbands) matching energies_k."
        )
    if occupations_kq_array.shape != expected_shifted_shape:
        raise ValueError("occupations_kq must match energies_kq shape.")

    nq = energies_kq_array.shape[0]

    if form_factors_array.shape != (nq, nk, nbands, nbands):
        raise ValueError("form_factors must have shape (nq, nk, nbands, nbands).")
    if frequencies_array.ndim != 1:
        raise ValueError("frequencies must be a one-dimensional array.")

    energy_differences = (
        energies_kq_array[:, :, np.newaxis, :]
        - energies_k_array[np.newaxis, :, :, np.newaxis]
    )
    occupation_differences = (
        occupations_kq_array[:, :, np.newaxis, :]
        - occupations_k_array[np.newaxis, :, :, np.newaxis]
    )
    numerator = occupation_differences * form_factors_array

    denominator = _rpa_denominator(
        energy_differences,
        frequencies_array,
        broadening=broadening,
        frequency_axis=frequency_axis,
    )
    integrand = numerator[..., np.newaxis] / denominator

    if k_weights is not None:
        k_weights_array = _as_float64(k_weights, name="k_weights")
        if k_weights_array.shape != (nk,):
            raise ValueError("k_weights must have shape (nk,).")
        integrand = (
            integrand
            * k_weights_array[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        )
        polarization = integrand.sum(axis=(1, 2, 3), dtype=np.complex128)
    else:
        polarization = integrand.sum(axis=(1, 2, 3), dtype=np.complex128)
        if normalize_k_sum and nk > 0:
            polarization = polarization / nk

    return polarization


def compute_rpa_polarization_matrix(
    energies_k: ArrayLike,
    energies_kq: ArrayLike,
    occupations_k: ArrayLike,
    occupations_kq: ArrayLike,
    eigenvectors: ArrayLike,
    q_shifts: ArrayLike,
    frequencies: ArrayLike,
    *,
    k_weights: ArrayLike | None = None,
    normalize_k_sum: bool = True,
    broadening: float = 0.0,
    frequency_axis: FrequencyAxis = "imaginary",
) -> NDArray[np.complex128]:
    r"""Compute the orbital-resolved RPA polarization matrix.

    For each transition ``(k, gamma) -> (k+q, gamma')`` this evaluates the
    orbital-density vertex

    .. math::

            M_i = u^*_{k,\gamma,i} u_{k+q,\gamma',i}

    and accumulates ``M_i M_j^*`` with the same RPA denominator used by the
    scalar polarization. The result has shape ``(nq, n_omega, norb, norb)``.
    """

    energies_k_array = _as_float64(energies_k, name="energies_k")
    energies_kq_array = _as_float64(energies_kq, name="energies_kq")
    occupations_k_array = _as_float64(occupations_k, name="occupations_k")
    occupations_kq_array = _as_float64(occupations_kq, name="occupations_kq")
    eigenvectors_array = _as_complex128(eigenvectors, name="eigenvectors")
    q_shifts_array = np.asarray(q_shifts, dtype=np.int64)
    frequencies_array = _as_float64(frequencies, name="frequencies")

    if energies_k_array.ndim != 2:
        raise ValueError("energies_k must have shape (nk, nbands).")
    if occupations_k_array.shape != energies_k_array.shape:
        raise ValueError("occupations_k must match energies_k shape.")

    nk, nbands = energies_k_array.shape
    if eigenvectors_array.ndim != 3 or eigenvectors_array.shape[0] != nk:
        raise ValueError("eigenvectors must have shape (nk, norb, nbands).")
    norb = eigenvectors_array.shape[1]
    if eigenvectors_array.shape[2] != nbands:
        raise ValueError("eigenvectors band dimension must match energies_k.")

    expected_shifted_shape = (energies_kq_array.shape[0], nk, nbands)
    if energies_kq_array.ndim != 3 or energies_kq_array.shape[1:] != (nk, nbands):
        raise ValueError(
            "energies_kq must have shape (nq, nk, nbands) matching energies_k."
        )
    if occupations_kq_array.shape != expected_shifted_shape:
        raise ValueError("occupations_kq must match energies_kq shape.")
    if q_shifts_array.shape != (energies_kq_array.shape[0],):
        raise ValueError("q_shifts must have shape (nq,).")
    if frequencies_array.ndim != 1:
        raise ValueError("frequencies must be a one-dimensional array.")

    k_weight_array = None
    if k_weights is not None:
        k_weight_array = _as_float64(k_weights, name="k_weights")
        if k_weight_array.shape != (nk,):
            raise ValueError("k_weights must have shape (nk,).")

    nq = energies_kq_array.shape[0]
    n_omega = frequencies_array.size
    polarization = np.empty((nq, n_omega, norb, norb), dtype=np.complex128)
    k_indices = np.arange(nk, dtype=np.int64)

    for q_index, q_shift in enumerate(q_shifts_array):
        shifted_vectors = eigenvectors_array[(k_indices + q_shift) % nk]
        transition_vertices = (
            eigenvectors_array.conj()[:, :, :, np.newaxis]
            * shifted_vectors[:, :, np.newaxis, :]
        )
        transition_vertices = np.moveaxis(transition_vertices, 1, -1)

        energy_differences = (
            energies_kq_array[q_index, :, np.newaxis, :]
            - energies_k_array[:, :, np.newaxis]
        )
        occupation_differences = (
            occupations_kq_array[q_index, :, np.newaxis, :]
            - occupations_k_array[:, :, np.newaxis]
        )
        denominator = _rpa_denominator(
            energy_differences,
            frequencies_array,
            broadening=broadening,
            frequency_axis=frequency_axis,
        )
        weights = occupation_differences[..., np.newaxis] / denominator
        if k_weight_array is not None:
            weights = weights * k_weight_array[:, np.newaxis, np.newaxis, np.newaxis]

        polarization[q_index] = np.einsum(
            "kbcw,kbci,kbcj->wij",
            weights,
            transition_vertices,
            transition_vertices.conj(),
            optimize=True,
        )
        if k_weight_array is None and normalize_k_sum and nk > 0:
            polarization[q_index] /= nk

    return polarization


class RPAPolarization:
    """RPA solver that maps a Hamiltonian file onto a momentum-resolved polarization."""

    def __init__(
        self,
        *,
        channels: ScreeningChannels | None = None,
        frequency_axis: FrequencyAxis = "imaginary",
    ) -> None:
        self.channels = channels or ScreeningChannels()
        if frequency_axis not in ("imaginary", "real"):
            raise ValueError("frequency_axis must be either 'imaginary' or 'real'.")
        self.frequency_axis = frequency_axis

    def load_bloch_band_structure(
        self,
        hamiltonian_file: str | Path,
        mesh: BrillouinZoneMesh,
        *,
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
    ) -> BlochBandStructure:
        """Load translation blocks from ``hamiltonian_file`` and diagonalize ``H(k)``."""

        translation_blocks = load_translation_blocks(hamiltonian_file)
        return self.load_bloch_band_structure_from_blocks(
            translation_blocks,
            mesh,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
        )

    def load_bloch_band_structure_from_blocks(
        self,
        translation_blocks: dict[tuple[int, int, int], NDArray[np.complex128]],
        mesh: BrillouinZoneMesh,
        *,
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
    ) -> BlochBandStructure:
        """Diagonalize ``H(k)`` from preloaded translation blocks."""

        bloch_hamiltonian = build_bloch_hamiltonian(
            translation_blocks,
            mesh.k_points,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
        )
        return diagonalize_bloch_hamiltonian(bloch_hamiltonian, mesh.k_points)

    def compute_polarization(
        self,
        band_structure: BlochBandStructure,
        mesh: BrillouinZoneMesh,
        *,
        chemical_potential: float,
        temperature: float,
        broadening: float = 0.0,
    ) -> NDArray[np.complex128]:
        """Compute the Coulomb-screening polarization from Bloch states."""

        return compute_rpa_polarization_from_bands(
            band_structure,
            mesh.q_points,
            mesh.frequencies,
            chemical_potential=chemical_potential,
            temperature=temperature,
            state_multiplicity=self.channels.multiplicity,
            broadening=broadening,
            frequency_axis=self.frequency_axis,
        )

    def compute_polarization_matrix(
        self,
        band_structure: BlochBandStructure,
        mesh: BrillouinZoneMesh,
        *,
        chemical_potential: float,
        temperature: float,
        broadening: float = 0.0,
    ) -> NDArray[np.complex128]:
        """Compute the orbital-resolved RPA polarization from Bloch states."""

        return compute_rpa_polarization_matrix_from_bands(
            band_structure,
            mesh.q_points,
            mesh.frequencies,
            chemical_potential=chemical_potential,
            temperature=temperature,
            state_multiplicity=self.channels.multiplicity,
            broadening=broadening,
            frequency_axis=self.frequency_axis,
        )

    def solve(
        self,
        *,
        hamiltonian_file: str | Path,
        mesh: BrillouinZoneMesh,
        chemical_potential: float,
        temperature: float,
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
        broadening: float = 0.0,
    ) -> PolarizationResult:
        """Compute the RPA polarization from a unit-cell Hamiltonian file."""

        band_structure = self.load_bloch_band_structure(
            hamiltonian_file,
            mesh,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
        )
        polarization = self.compute_polarization(
            band_structure,
            mesh,
            chemical_potential=chemical_potential,
            temperature=temperature,
            broadening=broadening,
        )
        return PolarizationResult(
            band_structure=band_structure,
            polarization=polarization,
        )

    def solve_from_translation_blocks(
        self,
        *,
        translation_blocks: dict[tuple[int, int, int], NDArray[np.complex128]],
        mesh: BrillouinZoneMesh,
        chemical_potential: float,
        temperature: float,
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
        broadening: float = 0.0,
    ) -> PolarizationResult:
        """Compute the RPA polarization from preloaded translation blocks."""

        band_structure = self.load_bloch_band_structure_from_blocks(
            translation_blocks,
            mesh,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
        )
        polarization = self.compute_polarization(
            band_structure,
            mesh,
            chemical_potential=chemical_potential,
            temperature=temperature,
            broadening=broadening,
        )
        return PolarizationResult(
            band_structure=band_structure,
            polarization=polarization,
        )

    def solve_from_config(
        self,
        config: QuatrexConfig,
        *,
        mesh: BrillouinZoneMesh,
        chemical_potential: float,
        temperature: float,
        matrix_name: str = "hamiltonian",
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
        broadening: float = 0.0,
    ) -> PolarizationResult:
        """Compute the RPA polarization from blocks loaded via ``config.input_dir``."""

        translation_blocks = load_translation_blocks_from_config(
            config,
            matrix_name=matrix_name,
        )
        return self.solve_from_translation_blocks(
            translation_blocks=translation_blocks,
            mesh=mesh,
            chemical_potential=chemical_potential,
            temperature=temperature,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
            broadening=broadening,
        )

    def solve_matrix_from_translation_blocks(
        self,
        *,
        translation_blocks: dict[tuple[int, int, int], NDArray[np.complex128]],
        mesh: BrillouinZoneMesh,
        chemical_potential: float,
        temperature: float,
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
        broadening: float = 0.0,
    ) -> PolarizationResult:
        """Compute matrix-valued RPA polarization from preloaded translation blocks."""

        band_structure = self.load_bloch_band_structure_from_blocks(
            translation_blocks,
            mesh,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
        )
        polarization = self.compute_polarization_matrix(
            band_structure,
            mesh,
            chemical_potential=chemical_potential,
            temperature=temperature,
            broadening=broadening,
        )
        return PolarizationResult(
            band_structure=band_structure,
            polarization=polarization,
        )

    def solve_matrix_from_config(
        self,
        config: QuatrexConfig,
        *,
        mesh: BrillouinZoneMesh,
        chemical_potential: float,
        temperature: float,
        matrix_name: str = "hamiltonian",
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
        broadening: float = 0.0,
    ) -> PolarizationResult:
        """Compute matrix-valued RPA polarization from blocks loaded via config."""

        translation_blocks = load_translation_blocks_from_config(
            config,
            matrix_name=matrix_name,
        )
        return self.solve_matrix_from_translation_blocks(
            translation_blocks=translation_blocks,
            mesh=mesh,
            chemical_potential=chemical_potential,
            temperature=temperature,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
            broadening=broadening,
        )

    def solve_from_unit_cell_file(
        self,
        *,
        hamiltonian_file: str | Path,
        k_points: ArrayLike,
        q_points: ArrayLike,
        frequencies: ArrayLike,
        chemical_potential: float,
        temperature: float,
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
        broadening: float = 0.0,
    ) -> PolarizationResult:
        """Compatibility wrapper around :meth:`solve` using explicit arrays."""

        mesh = BrillouinZoneMesh(
            k_points=k_points,
            q_points=q_points,
            frequencies=frequencies,
        )
        return self.solve(
            hamiltonian_file=hamiltonian_file,
            mesh=mesh,
            chemical_potential=chemical_potential,
            temperature=temperature,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
            broadening=broadening,
        )


class RPACompute(RPAPolarization):
    """Backward-compatible alias for the Hamiltonian-to-polarization workflow."""

    def solve_from_unit_cell_file(
        self,
        *,
        hamiltonian_file: str | Path,
        k_points: ArrayLike,
        q_points: ArrayLike,
        frequencies: ArrayLike,
        chemical_potential: float,
        temperature: float,
        periodic_axis: int | None = None,
        lattice_constant: float = 1.0,
        broadening: float = 0.0,
    ) -> PolarizationResult:
        """Compute the RPA polarization directly from a unit-cell Hamiltonian block file.

        This is the high-level entry point intended for the carbon-nanotube unit-cell
        example, whose Hamiltonian file contains translation blocks keyed by lattice
        vectors such as ``[0, 0, 1]``.
        """

        return super().solve_from_unit_cell_file(
            hamiltonian_file=hamiltonian_file,
            k_points=k_points,
            q_points=q_points,
            frequencies=frequencies,
            chemical_potential=chemical_potential,
            temperature=temperature,
            periodic_axis=periodic_axis,
            lattice_constant=lattice_constant,
            broadening=broadening,
        )


__all__ = [
    "BlochBandStructure",
    "BrillouinZoneMesh",
    "build_uniform_brillouin_zone_mesh",
    "load_translation_blocks",
    "load_translation_blocks_from_config",
    "compute_rpa_polarization_matrix",
    "compute_rpa_polarization_matrix_from_bands",
    "resolve_unit_cell_matrix_path",
    "PolarizationResult",
    "RPACompute",
    "RPAPolarization",
    "ScreeningChannels",
]
