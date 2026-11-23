# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Tests the charge density computed from the NEGF observables."""

from types import SimpleNamespace

import numpy as np
import pytest

from qttools import xp
from qttools.utils.gpu_utils import get_host
from quatrex.core.scba import SCBA

NUM_ORBITALS = 6
ENERGIES = np.linspace(-1.0, 1.0, 41)


def _make_scba(electron_density: np.ndarray, hole_density: np.ndarray) -> SCBA:
    """Creates an SCBA with only what the charge density needs."""
    scba = SCBA.__new__(SCBA)
    scba.electron_energies = xp.asarray(ENERGIES)
    scba.observables = SimpleNamespace(
        electron_density=xp.asarray(electron_density),
        hole_density=xp.asarray(hole_density),
    )
    scba.device = SimpleNamespace(
        potential=xp.linspace(-0.1, 0.1, NUM_ORBITALS),
        contacts=[SimpleNamespace(voltage=0.0, mid_gap_energy=0.0, temperature=300.0)],
    )
    scba.data = SimpleNamespace(
        orbital_offsets=np.array([0, NUM_ORBITALS // 2, NUM_ORBITALS])
    )
    return scba


@pytest.mark.parametrize("kpoint_shape", [(3,), (2, 4)])
def test_charge_density_kpoint_average(kpoint_shape: tuple[int, ...]):
    """Tests that the charge is the average over transverse k-points."""
    rng = np.random.default_rng(0)
    shape = (ENERGIES.size, *kpoint_shape, NUM_ORBITALS)
    electron_density = rng.random(shape)
    hole_density = rng.random(shape)

    charge_density = _make_scba(electron_density, hole_density).get_charge_density()

    reference = np.mean(
        [
            get_host(
                _make_scba(
                    electron_density[:, *index], hole_density[:, *index]
                ).get_charge_density()
            )
            for index in np.ndindex(kpoint_shape)
        ],
        axis=0,
    )
    assert charge_density.shape == (2,)
    assert np.allclose(get_host(charge_density), reference)


def test_spectral_density_of_states_kpoint_average():
    """Tests that identical k-points give the result without k-points."""
    rng = np.random.default_rng(0)
    electron_density = rng.random((ENERGIES.size, NUM_ORBITALS))
    hole_density = rng.random((ENERGIES.size, NUM_ORBITALS))

    reference = _make_scba(
        electron_density, hole_density
    ).get_spectral_density_of_states()
    result = _make_scba(
        np.repeat(electron_density[:, None], 3, axis=1),
        np.repeat(hole_density[:, None], 3, axis=1),
    ).get_spectral_density_of_states()

    assert np.allclose(result.spectral_dos, reference.spectral_dos)
    assert np.allclose(result.chemical_potentials, reference.chemical_potentials)
