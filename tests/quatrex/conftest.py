# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
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


@pytest.fixture(params=EXAMPLES, autouse=True, scope="function")
def example(request: pytest.FixtureRequest) -> tuple[Path, bool]:
    return request.param
