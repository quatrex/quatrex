# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import pytest
from scipy import sparse

from qttools.datastructures import DSBCSR
from qttools.datastructures.dsbsparse import DSBSparse


@pytest.mark.parametrize("dbsparse_type", [DSBCSR])
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
        pytest.param((10,), id="1D-stack"),
    ],
)
@pytest.mark.parametrize(
    "densify_blocks",
    [
        pytest.param(None, id="no-densify"),
        pytest.param([(0, 0), (-1, -1)], id="densify-boundary"),
        pytest.param([(2, 4)], id="densify-random"),
    ],
)
@pytest.mark.parametrize(
    "stack_slice",
    [
        pytest.param(None, id="entire-stack"),
        pytest.param(slice(3, 5, 1), id="stack-slice"),
    ],
)
def test_to_dense(
    coo: sparse.coo_array,
    dbsparse_type: DSBSparse,
    block_sizes: np.ndarray,
    global_stack_shape: int | tuple[int],
    densify_blocks: list[tuple[int]] | None,
    stack_slice: tuple[int] | None,
):
    dense_ref = coo.toarray()

    dbsparse = dbsparse_type.from_sparray(
        coo, block_sizes, global_stack_shape, densify_blocks
    )

    dense_stack = dbsparse.to_dense(stack_slice=stack_slice)

    if stack_slice is None:
        testing_stack_size = len(global_stack_shape)
    else:
        testing_stack_size = (stack_slice.stop - stack_slice.start) // stack_slice.step

    assert np.allclose(
        np.repeat(dense_ref[np.newaxis, :, :], testing_stack_size, axis=0), dense_stack
    )
