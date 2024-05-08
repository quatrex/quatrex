import numpy as np
import pytest
from scipy import sparse

from qttools.datastructures import DBCSR
from qttools.datastructures.dbsparse import DBSparse


@pytest.mark.parametrize(
    "block_sizes",
    [
        pytest.param(np.array([10] * 10), id="constant-block-size"),
        pytest.param(np.array([5] * 5 + [10] * 5 + [5] * 5), id="mixed-block-size"),
    ],
)
@pytest.mark.parametrize(
    "stack_shape",
    [
        pytest.param((10,), id="1D-stack"),
        pytest.param((10, 2), id="2D-stack"),
    ],
)
@pytest.mark.parametrize(
    "densify_blocks",
    [
        pytest.param(None, id="no-densify"),
        pytest.param([(0, 0), (-1, -1)], id="densify"),
    ],
)
@pytest.mark.parametrize("dbsparse_type", [DBCSR])
def test_from_sparray(
    coo: sparse.coo_array,
    dbsparse_type: DBSparse,
    block_sizes: np.ndarray,
    stack_shape: int | tuple[int],
    densify_blocks: list[tuple[int]] | None,
):
    """Tests that the from_sparray method works."""
    dbsparse = dbsparse_type.from_sparray(coo, block_sizes, stack_shape, densify_blocks)
    assert np.allclose(dbsparse.to_dense()[(0,) * len(stack_shape)], coo.toarray())
