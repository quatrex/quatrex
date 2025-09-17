# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.

import pytest

from quatrex.examples import ALLOWED_EXAMPLES

# only small examples for testing
TEST_EXAMPLES = [
    "carbon-nanotube:",
]

DIST_TEST_EXAMPLES = [
    "carbon-nanotube:dist",
]

assert len(TEST_EXAMPLES) == len(set(TEST_EXAMPLES))
assert set(TEST_EXAMPLES).issubset(ALLOWED_EXAMPLES.keys())
assert len(DIST_TEST_EXAMPLES) == len(set(DIST_TEST_EXAMPLES))
assert set(DIST_TEST_EXAMPLES).issubset(ALLOWED_EXAMPLES.keys())

NON_DISTIRBUTED_EXAMPLES = [
    pytest.param(example, id=example) for example in TEST_EXAMPLES
]
DOMAIN_DISTTRIBUTED_EXAMPLES = [
    pytest.param(example, id=example) for example in DIST_TEST_EXAMPLES
]
EXAMPLES = NON_DISTIRBUTED_EXAMPLES + DOMAIN_DISTTRIBUTED_EXAMPLES


@pytest.fixture(params=EXAMPLES)
def example(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=NON_DISTIRBUTED_EXAMPLES)
def non_distributed_example(request: pytest.FixtureRequest) -> str:
    return request.param
