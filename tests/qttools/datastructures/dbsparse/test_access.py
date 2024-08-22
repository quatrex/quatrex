# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import pytest
from scipy import sparse

from qttools.datastructures import DSBCSR
from qttools.datastructures.dsbsparse import DSBSparse


def test_indexing(): ...


@pytest.mark.parametrize(
    "block_sizes",
    [
        pytest.param(np.array([10] * 10), id="constant-block-size"),
        pytest.param(np.array([5] * 5 + [10] * 5 + [5] * 5), id="mixed-block-size"),
    ],
)
@pytest.mark.parametrize(
    "global_stack_shape",
    [
        pytest.param((1,), id="no-stack"),
        pytest.param((10,), id="1D-stack"),
    ],
)
@pytest.mark.parametrize("dbsparse_type", [DSBCSR])
def test_diagonal(
    coo: sparse.coo_array,
    dbsparse_type: DSBSparse,
    block_sizes: np.ndarray,
    global_stack_shape: int | tuple[int],
):
    """Tests that we can get the correct diagonal elements."""
    dense = np.repeat(
        coo.toarray()[np.newaxis, :, :],
        global_stack_shape[0],
        axis=0,
    )
    reference = np.diagonal(dense, axis1=-2, axis2=-1)

    dbsparse = dbsparse_type.from_sparray(
        coo,
        block_sizes=block_sizes,
        global_stack_shape=global_stack_shape,
    )

    assert np.allclose(reference, dbsparse.diagonal())


def test_spy(): ...
