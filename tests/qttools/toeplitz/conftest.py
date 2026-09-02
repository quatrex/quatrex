# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes fixtures for testing the qttools.toeplitz module."""

import numpy as np
import pytest

TRANSPORT_INDS = [0, 1, 2]

GRID_SHAPES = [(1, 5, 5), (3, 5, 3), (1, 3, 1)]

BLOCK_SIZES = [1, 2, 3]

PHASES = [
    (1.0, 1.0),
    (np.exp(1j * np.pi / 4), np.exp(1j * np.pi / 4)),
    (np.exp(-1j * np.pi / 2), np.exp(-1j * np.pi / 2)),
]


@pytest.fixture(params=TRANSPORT_INDS)
def transport_ind(request):
    return request.param


@pytest.fixture(params=GRID_SHAPES)
def grid_shape(request):
    return request.param


@pytest.fixture(params=BLOCK_SIZES)
def block_size(request):
    return request.param


@pytest.fixture(params=PHASES)
def phases(request):
    return request.param
