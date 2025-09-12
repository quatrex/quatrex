# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.

import pytest

from quatrex.examples import ALLOWED_EXAMPLES

# only small examples for testing
TEST_EXAMPLES = [
    "carbon-nanotube",
]

assert len(TEST_EXAMPLES) == len(set(TEST_EXAMPLES))
assert set(TEST_EXAMPLES).issubset(ALLOWED_EXAMPLES.keys())


EXAMPLES = [pytest.param(example, id=example) for example in TEST_EXAMPLES]


@pytest.fixture(params=EXAMPLES)
def example_name(request: pytest.FixtureRequest) -> str:
    return request.param
