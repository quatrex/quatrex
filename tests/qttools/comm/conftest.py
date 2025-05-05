# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools.comm.comm import _backends

BACKEND_TYPE = [pytest.param(backend, id=backend) for backend in _backends]

BLOCK_COMM_SIZES = [
    pytest.param(1, id="1"),
    pytest.param(2, id="2"),
    pytest.param(4, id="4"),
]


@pytest.fixture(params=BACKEND_TYPE)
def backend_type(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=BLOCK_COMM_SIZES)
def block_comm_size(request: pytest.FixtureRequest) -> int:
    return request.param
