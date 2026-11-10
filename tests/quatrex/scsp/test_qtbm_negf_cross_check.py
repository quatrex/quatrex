# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Cross-checks ballistic SCSP with the wave-function (QTBM) and the
NEGF formalism.

Without scattering, both formalisms describe the same physics, so every
SCSP iteration must produce the same charge density and potential. With
the "ldos" density model, the two formalisms use different (but in
equilibrium identical) density response models: the contact-resolved
LDOS for QTBM and the spectral function with fitted quasi-Fermi levels
for NEGF. The equilibrium cases therefore also cross-check these.

"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from quatrex.cli.main import run as cli_run

EXAMPLES_DIR = Path(__file__).parents[3].resolve() / "examples"

# NOTE: The SCBA requires at least one interaction. A pseudo-phonon with
# vanishing coupling makes all scattering self-energies zero, i.e., the
# transport is ballistic.
BALLISTIC_SCBA = """[scba]
max_iterations = 1
phonon = true

[phonon]
model = "pseudo-scattering"
phonon_energy = 0.05
deformation_potential = 0.0
interaction_cutoff = 1.0

[outputs]
electron_density = true
hole_density = true

[scsp]"""

RUN_SLOW = os.environ.get("QUATREX_RUN_SLOW_TESTS", "0") == "1"


@dataclass
class CrossCheckCase:
    """A QTBM/NEGF cross-check derived from an example config.

    Attributes
    ----------
    example : str
        The example directory relative to `examples/`.
    num_iterations : int
        The number of SCSP iterations to compare.
    common : list[tuple[str, str]]
        Replacements applied to the config for both formalisms.
    negf : list[tuple[str, str]]
        Additional replacements for the NEGF config.

    """

    example: str
    num_iterations: int
    common: list[tuple[str, str]] = field(default_factory=list)
    negf: list[tuple[str, str]] = field(default_factory=list)


CNT_NEGF = [
    ('formalism = "wf"', 'formalism = "negf"'),
    ("[device]\n", "[device]\nblock_size = 32\n"),
    ('solver.direct_solver = "superlu"', 'solver.algorithm = "rgf"'),
    ("[scsp]", BALLISTIC_SCBA),
]

CASES = [
    pytest.param(
        CrossCheckCase(
            example="w90/carbon-nanotube/qtbm-poisson",
            num_iterations=3,
            negf=CNT_NEGF,
        ),
        id="cnt-bias",
    ),
    pytest.param(
        CrossCheckCase(
            example="w90/carbon-nanotube/qtbm-poisson",
            num_iterations=3,
            common=[
                ("voltage = -0.2", "voltage = 0.0"),
                ("density_model_dim = 1", 'density_model = "ldos"'),
            ],
            negf=CNT_NEGF,
        ),
        id="cnt-equilibrium-ldos",
    ),
    pytest.param(
        CrossCheckCase(
            example="cp2k/graphene/qtbm-poisson",
            num_iterations=2,
            common=[
                # Reduced settings to keep the test affordable.
                ("kpoint_grid = [1, 3, 1]", "kpoint_grid = [1, 1, 1]"),
                ("max_iterations = 10", "max_iterations = 2"),
                ("energy_window_num = 241", "energy_window_num = 61"),
                ('density_model = "graphene"', 'density_model = "ldos"'),
                ("voltage = -0.05", "voltage = 0.0"),
            ],
            negf=[
                ('formalism = "wf"', 'formalism = "negf"'),
                ("[device]\n", "[device]\nblock_size = 416\n"),
                ("[electron]\n", '[electron]\nsolver.algorithm = "rgf"\n'),
                # NOTE: The SCBA contacts require the right contact to
                # be the exact mirror of the device's last block. In
                # each slice, the device orbitals of the lower half (in
                # y) come first, so the right contact's transverse
                # vector is reversed (same lead, different origin).
                (
                    (
                        "origin = [48, -0.001, -0.001]\n"
                        "    lattice_vectors = [[-2.461, 0, 0], [0, 4.263, 0], "
                        "[0, 0, 6.709]]"
                    ),
                    (
                        "origin = [48, 8.52484, -0.001]\n"
                        "    lattice_vectors = [[-2.461, 0, 0], [0, -4.263, 0], "
                        "[0, 0, 6.709]]"
                    ),
                ),
                ("[scsp]", BALLISTIC_SCBA),
            ],
        ),
        id="graphene-equilibrium-ldos",
        marks=[
            pytest.mark.skipif(
                not RUN_SLOW, reason="Slow (~20 min), set QUATREX_RUN_SLOW_TESTS=1."
            ),
            # NOTE: For this non-orthogonal basis with transverse
            # periodic images, the NEGF spectral function integrates to
            # ~6% more states than the QTBM LDOS (which matches the band
            # structure), uniformly in the device. The overlap used in
            # the NEGF density seems to be inconsistent with the one in
            # its system matrix. Remove the xfail once fixed.
            pytest.mark.xfail(
                strict=True,
                reason="NEGF overlap-weighted density overcounts states "
                "for non-orthogonal bases with transverse periodicity.",
            ),
        ],
    ),
]


def _apply(config_text: str, replacements: list[tuple[str, str]]) -> str:
    """Applies unique replacements, failing loudly if the example changed."""
    for old, new in replacements:
        assert config_text.count(old) == 1, f"Expected one occurrence of {old!r}."
        config_text = config_text.replace(old, new)
    return config_text


def _run(config_path: Path) -> Path:
    """Runs quatrex for the config and returns the output directory."""
    cli_run(config_path, abort_on_exception=False)
    return config_path.parent / "outputs"


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("case", CASES)
def test_ballistic_scsp_qtbm_negf(
    case: CrossCheckCase, tmp_path: Path, adjust_config_paths: Callable
):
    """Tests that ballistic QTBM and NEGF SCSP iterations agree."""
    example_config = EXAMPLES_DIR / case.example / "quatrex_config.toml"

    outputs = {}
    for formalism, replacements in (
        ("wf", case.common),
        ("negf", case.common + case.negf),
    ):
        config_path = tmp_path / formalism / "quatrex_config.toml"
        config_path.parent.mkdir()
        adjust_config_paths(example_config, config_path)
        config_path.write_text(_apply(config_path.read_text(), replacements))
        outputs[formalism] = _run(config_path)

    for iteration in range(1, case.num_iterations + 1):
        for name in ("real_space_charge_density", "real_space_potential"):
            wf = np.load(outputs["wf"] / f"{name}_{iteration}.npy")
            negf = np.load(outputs["negf"] / f"{name}_{iteration}.npy")

            # Guard against a trivially passing comparison.
            assert np.abs(wf).max() > 1e-4, f"{name}_{iteration} is ~zero."
            assert np.allclose(negf, wf, rtol=0, atol=1e-5 * np.abs(wf).max()), (
                f"{name}_{iteration}: max deviation "
                f"{np.abs(negf - wf).max():.3e}, max value {np.abs(wf).max():.3e}"
            )
