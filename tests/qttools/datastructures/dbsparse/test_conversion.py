# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import pytest
from scipy import sparse

from qttools.datastructures import DSBCOO, DSBCSR
from qttools.datastructures.dsbsparse import DSBSparse


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
        pytest.param((10, 2), id="2D-stack"),
        pytest.param((10, 2, 4), id="3D-stack"),
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
@pytest.mark.parametrize("dbsparse_type", [DSBCSR, DSBCOO])
class TestConversion:
    """Tests for the conversion methods of DSBSparse."""

    def test_to_dense(
        self,
        coo: sparse.coo_array,
        dbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: int | tuple[int],
        densify_blocks: list[tuple[int]] | None,
    ):
        """Tests that we can convert a DSBSparse matrix to dense."""
        reference = np.broadcast_to(coo.toarray(), global_stack_shape + coo.shape)

        dbsparse = dbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
            densify_blocks=densify_blocks,
        )

        assert np.allclose(reference, dbsparse.to_dense())

    def test_ltranspose(
        self,
        coo: sparse.coo_array,
        dbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: int | tuple[int],
        densify_blocks: list[tuple[int]] | None,
    ):
        """Tests that we can transpose a DSBSparse matrix."""
        dense = np.broadcast_to(coo.toarray(), global_stack_shape + coo.shape)
        reference = np.swapaxes(dense, -2, -1)

        dbsparse = dbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
            densify_blocks=densify_blocks,
        )

        dbsparse.ltranspose()  # In-place transpose.

        assert np.allclose(reference, dbsparse.to_dense())
