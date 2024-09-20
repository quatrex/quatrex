# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import pytest


GLOBAL_STACK_SHAPES = [
    pytest.param((10,), id="1D-stack"),
    pytest.param((7, 2), id="2D-stack"),
    pytest.param((9, 2, 4), id="3D-stack"),
]


@pytest.fixture(params=GLOBAL_STACK_SHAPES, autouse=True)
def global_stack_shape(request):
    return request.param
