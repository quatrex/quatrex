# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
import numpy as np
import pytest

from quatrex.bandstructure.contact import contact_fermi_level
from quatrex.core.statistics import fermi_dirac

KPOINTS = np.linspace(-np.pi, np.pi, 400, endpoint=False)
KPOINTS_TRANSVERSE = np.linspace(-np.pi, np.pi, 7, endpoint=False) + np.pi / 7
CELL_VOLUME = 50.0
TEMPERATURE = 300.0


def _dirac_bands() -> np.ndarray:
    """Two-band Dirac-cone model with shape (nk, nk_transverse, 2)."""
    kx, ky = np.meshgrid(KPOINTS, KPOINTS_TRANSVERSE, indexing="ij")
    energy = np.sqrt(kx**2 + ky**2)
    return np.stack([-energy, energy], axis=-1)


def _excess_density(e_k: np.ndarray, fermi_level: float) -> float:
    """Net excess density, averaging occupations over transverse k."""
    n_k = fermi_dirac(e_k[..., 1] - fermi_level, TEMPERATURE).mean(axis=1)
    p_k = fermi_dirac(fermi_level - e_k[..., 0], TEMPERATURE).mean(axis=1)
    return 2 * np.trapezoid(n_k - p_k, KPOINTS) / (2 * np.pi * CELL_VOLUME)


@pytest.mark.parametrize("doping_density", [-2e-4, 0.0, 1e-4, 5e-4])
def test_transverse_kpoints(doping_density: float):
    """Tests that the Fermi level reproduces the doping density when
    occupations (not eigenvalues) are averaged over transverse k."""
    e_k = _dirac_bands()
    fermi_level = contact_fermi_level(
        e_k=e_k,
        kpoints=KPOINTS,
        mid_gap_energy=0.0,
        cell_volume=CELL_VOLUME,
        doping_density=doping_density,
        temperature=TEMPERATURE,
    )
    assert np.isclose(
        _excess_density(e_k, fermi_level), doping_density, rtol=1e-3, atol=1e-8
    )


def test_single_transverse_kpoint():
    """Tests that 2D (legacy) input matches 3D input with one transverse k."""
    e_k = _dirac_bands()[:, :1, :]
    kwargs = {
        "kpoints": KPOINTS,
        "mid_gap_energy": 0.0,
        "cell_volume": CELL_VOLUME,
        "doping_density": 1e-4,
        "temperature": TEMPERATURE,
    }
    assert np.isclose(
        contact_fermi_level(e_k=e_k, **kwargs),
        contact_fermi_level(e_k=e_k[:, 0, :], **kwargs),
    )
