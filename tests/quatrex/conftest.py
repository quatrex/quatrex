# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
import tomllib
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parents[2].resolve() / "examples"

assert EXAMPLES_DIR.exists()

CONFIGS = list(EXAMPLES_DIR.glob("**/quatrex_config.toml"))
EXAMPLES = [
    pytest.param(
        (config.parent, "dist" in config.parent.stem),
        id="-".join(config.parent.parts[-3:]),
    )
    for config in CONFIGS
]


@pytest.fixture(params=EXAMPLES, scope="function")
def example(request: pytest.FixtureRequest) -> tuple[Path, bool]:
    return request.param


@pytest.fixture
def adjust_config_paths():
    """
    A factory fixture that returns a function to adjust config paths.
    """

    def _adjust_config_paths(quatrex_config_path: Path, tmp_config_path: Path):
        """Adjusts the input directory path in the temporary config to point
        to the example's input directory.

        Parameters
        ----------
        quatrex_config_path : Path
            The path to the original config file in the example directory.
        tmp_config_path : Path
            The path to the temporary config file that will be used for
            testing.

        """
        # Read the original config to find the input directory.
        with open(quatrex_config_path, "rb") as f:
            config = tomllib.load(f)

        config_text = quatrex_config_path.read_text()

        input_dir = config.get("input_dir")
        if input_dir is None:
            # If the input directory is not specified, we assume it is
            # "inputs" relative to the config file.
            abs_input_dir = str((quatrex_config_path.parent / "inputs").resolve())
            config_text = f'input_dir = "{abs_input_dir}"\n' + config_text

        elif not Path(input_dir).is_absolute():
            abs_input_dir = str((quatrex_config_path.parent / input_dir).resolve())
            config_text = config_text.replace(input_dir, abs_input_dir)

        # Copy the config and replace the input directory with the absolute path.
        tmp_config_path.write_text(config_text)

    return _adjust_config_paths
