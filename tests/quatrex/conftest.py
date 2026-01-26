# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parents[2].resolve() / "examples"

assert EXAMPLES_DIR.exists()

CONFIGS = list(EXAMPLES_DIR.glob("**/quatrex_config.toml"))
EXAMPLES = [(config.parent, "dist" in config.parent.stem) for config in CONFIGS]


@pytest.fixture(params=EXAMPLES, autouse=True, scope="function")
def example(request: pytest.FixtureRequest) -> tuple[Path, bool]:
    return request.param
