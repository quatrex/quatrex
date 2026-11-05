# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

SIZES = [
    pytest.param(10, id="size-10"),
    pytest.param(20, id="size-20"),
    pytest.param(21, id="size-21"),
]


LOCAL_STACK_SHAPES = [
    pytest.param(tuple(), id="no-stack"),
    pytest.param((10,), id="1D-stack"),
    pytest.param((7, 2), id="2D-stack"),
    pytest.param((9, 2, 4), id="3D-stack"),
]


SYMMETRY = [
    pytest.param(None, id="non-symmetric"),
    pytest.param("skew-hermitian", id="skew-hermitian"),
    pytest.param("hermitian", id="hermitian"),
    pytest.param("symmetric", id="symmetric"),
    pytest.param("skew-symmetric", id="skew-symmetric"),
    pytest.param("upper-triangular", id="upper-triangular"),
]


@pytest.fixture(params=SIZES)
def size(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=LOCAL_STACK_SHAPES)
def local_stack_shape(request: pytest.FixtureRequest) -> tuple:
    return request.param


@pytest.fixture(params=SYMMETRY)
def symmetry(request: pytest.FixtureRequest) -> str | None:
    return request.param
