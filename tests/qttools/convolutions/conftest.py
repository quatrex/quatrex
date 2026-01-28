# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

ARRAY_SHAPES = [
    pytest.param((20,), id="1D-array"),
    pytest.param((20, 5), id="2D-array"),
    pytest.param((20, 5, 3), id="3D-array"),
]


@pytest.fixture(params=ARRAY_SHAPES)
def array_shape(request) -> tuple[int, ...]:
    return request.param
