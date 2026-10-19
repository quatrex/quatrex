# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Checks that explicit spin reproduces the spinless reference outputs."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from quatrex.cli.main import run as cli_run

EXAMPLES_DIR = Path(__file__).parents[3].resolve() / "examples"

# Relation between the spinful outputs and the spinless references:
# (how to reduce the spin axis, factor relative to the reference).
SPIN_RULES = {
    # Transmission and LDOS are traces without a spin degeneracy factor.
    "transmission": ("sum", 2.0),
    "dos_": ("sum", 2.0),
    # Currents and densities include the spin degeneracy factor.
    "current_": ("sum", 1.0),
    "excess_": ("sum", 1.0),
    "electron_energies": (None, 1.0),
}


def _reduce_spin(test: np.ndarray, reference: np.ndarray, mode: str | None):
    """Sums over the (interleaved) spin axis of a spinful output."""
    if mode is None or test.shape == reference.shape:
        return test
    (axis,) = [
        i for i, (a, b) in enumerate(zip(test.shape, reference.shape)) if a == 2 * b
    ]
    shape = test.shape[:axis] + (reference.shape[axis], 2) + test.shape[axis + 1 :]
    return test.reshape(shape).sum(axis=axis + 1)


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("example", ["cp2k/carbon-chain/qtbm"])
def test_spin_expanded(example: str, tmp_path: Path, adjust_config_paths: Callable):
    """Runs a spinless example with expanded spin inputs."""
    example_path = EXAMPLES_DIR / example
    tmp_config_path = tmp_path / "quatrex_config.toml"
    adjust_config_paths(example_path / "quatrex_config.toml", tmp_config_path)

    config_text = tmp_config_path.read_text()
    assert "[device]\n" in config_text
    config_text = config_text.replace(
        "[device]\n", "[device]\nspin = true\nexpand_spinless_inputs = true\n", 1
    )
    tmp_config_path.write_text(config_text)

    cli_run(tmp_config_path, abort_on_exception=False)

    output_dir = tmp_path / "outputs"
    reference_dir = example_path / "reference-outputs"
    outputs = sorted(output_dir.glob("*.npy"))
    assert outputs

    mismatches = []
    for output_file in outputs:
        mode, factor = next(
            rule for key, rule in SPIN_RULES.items() if output_file.name.startswith(key)
        )
        reference = np.load(reference_dir / output_file.name)
        test = _reduce_spin(np.load(output_file), reference, mode)
        if test.shape != reference.shape or not np.allclose(
            test, factor * reference, rtol=1e-4, atol=1e-4
        ):
            mismatches.append(output_file.name)

    assert not mismatches, f"Spinful outputs do not match: {mismatches}"
