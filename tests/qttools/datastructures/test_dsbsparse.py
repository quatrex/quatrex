# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from contextlib import nullcontext

import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD as comm
from scipy import sparse

from qttools.datastructures.dsbsparse import DSBSparse, _block_view
from qttools.utils.gpu_utils import get_device
from qttools.utils.mpi_utils import get_section_sizes


@pytest.mark.usefixtures("densify_blocks")
class TestCreation:
    """Tests the creation methods of DSBSparse."""

    def test_from_sparray(
        self,
        coo: sparse.coo_array,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: int | tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the creation of DSBSparse matrices from sparse arrays."""
        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        assert np.array_equiv(coo.toarray(), dsbsparse.to_dense())

    def test_zeros_like(
        self,
        coo: sparse.coo_array,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: int | tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the creation of a zero DSBSparse matrix with the same shape as another."""
        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        zeros = dsbsparse_type.zeros_like(dsbsparse)
        assert (zeros.to_dense() == 0).all()


def _unsign_index(row: int, col: int, num_blocks) -> tuple:
    """Adjusts the sign to allow negative indices and checks bounds."""
    row = num_blocks + row if row < 0 else row
    col = num_blocks + col if col < 0 else col
    in_bounds = 0 <= row < num_blocks and 0 <= col < num_blocks
    return row, col, in_bounds


def _get_dense_index(dense: np.ndarray, block: tuple, block_sizes: np.ndarray) -> tuple:
    """Returns the dense index for a block."""
    block_offsets = np.hstack(([0], np.cumsum(block_sizes)))
    num_blocks = len(block_sizes)

    # Normalize negative indices.
    row, col, in_bounds = _unsign_index(*block, num_blocks)

    index = [slice(None)] * dense.ndim
    index[-2:] = (
        slice(block_offsets[row], block_offsets[row + 1]),
        slice(block_offsets[col], block_offsets[col + 1]),
    )

    return index, in_bounds


class TestConversion:
    """Tests for the conversion methods of DSBSparse."""

    def test_to_dense(
        self,
        coo: sparse.coo_array,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
    ):
        """Tests that we can convert a DSBSparse matrix to dense."""
        reference = np.broadcast_to(coo.toarray(), global_stack_shape + coo.shape)

        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )

        assert np.allclose(reference, dsbsparse.to_dense())

    def test_ltranspose(
        self,
        coo: sparse.coo_array,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
    ):
        """Tests that we can transpose a DSBSparse matrix."""
        dense = np.broadcast_to(coo.toarray(), global_stack_shape + coo.shape)
        reference = np.swapaxes(dense, -2, -1)

        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )

        # Transpose forth.
        dsbsparse.ltranspose()  # In-place transpose.

        assert np.allclose(reference, dsbsparse.to_dense())

        # Transpose back.
        dsbsparse.ltranspose()

        assert np.allclose(dense, dsbsparse.to_dense())


class TestAccess:
    """Tests for the access methods of DSBSparse."""

    @pytest.mark.usefixtures("accessed_block")
    def test_get_block(
        self,
        coo: sparse.coo_array,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        accessed_block: tuple,
    ):
        """Tests that we can get the correct block."""
        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsbsparse.to_dense()

        index, in_bounds = _get_dense_index(dense, accessed_block, block_sizes)
        reference_block = dense[*index]

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            assert np.allclose(reference_block, dsbsparse.blocks[accessed_block])

    @pytest.mark.usefixtures("accessed_block", "densify_blocks")
    def test_set_block(
        self,
        coo: sparse.coo_array,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
        accessed_block: tuple,
    ):
        """Tests that we can set a block and not modify sparsity structure."""
        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
            densify_blocks=densify_blocks,
        )
        dense = dsbsparse.to_dense()

        index, in_bounds = _get_dense_index(dense, accessed_block, block_sizes)

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            dsbsparse.blocks[accessed_block] = get_device(np.ones_like(dense[*index]))

        if densify_blocks is not None and accessed_block in densify_blocks:
            # Sparsity structure should be modified.
            assert (dsbsparse.to_dense()[*index] == 1).all()
        else:
            # Sparsity structure should not be modified.
            dense[*index][dense[*index].nonzero()] = 1
            assert np.allclose(dense, dsbsparse.to_dense())

    def test_spy(
        self,
        coo: sparse.coo_array,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
    ):
        """Tests that we can get the correct sparsity pattern."""
        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        inds = np.lexsort((coo.col, coo.row))
        ref_col, ref_row = coo.col[inds], coo.row[inds]

        rows, cols = dsbsparse.spy()
        inds = np.lexsort((cols, rows))
        col, row = cols[inds], rows[inds]

        assert np.allclose(ref_col, col)
        assert np.allclose(ref_row, row)

    def test_diagonal(
        self,
        coo: sparse.coo_array,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
    ):
        """Tests that we can get the correct diagonal elements."""
        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsbsparse.to_dense()

        reference = np.diagonal(dense, axis1=-2, axis2=-1)
        assert np.allclose(reference, dsbsparse.diagonal())


@pytest.mark.usefixtures("densify_blocks")
class TestArithmetic:
    """Tests for the arithmetic operations of DSBSparse."""

    def test_iadd(
        self,
        coo: sparse.coo_array,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the in-place addition of a DSBSparse matrix."""
        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        dense = dsbsparse.to_dense()

        dsbsparse += dsbsparse

        assert np.allclose(dense + dense, dsbsparse.to_dense())

    def test_isub(
        self,
        coo: sparse.coo_array,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the in-place subtraction of a DSBSparse matrix."""
        dsbsparse_1 = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        dense_1 = dsbsparse_1.to_dense()

        dsbsparse_2 = dsbsparse_type.from_sparray(
            2 * coo, block_sizes, global_stack_shape, densify_blocks
        )
        dense_2 = dsbsparse_2.to_dense()

        dsbsparse_1 -= dsbsparse_2

        assert np.allclose(dense_1 - dense_2, dsbsparse_1.to_dense())

    def test_imul(
        self,
        coo: sparse.coo_array,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the in-place multiplication of a DSBSparse matrix."""
        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        dense = dsbsparse.to_dense()

        dsbsparse *= dsbsparse

        assert np.allclose(dense * dense, dsbsparse.to_dense())

    def test_neg(
        self,
        coo: sparse.coo_array,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the negation of a DSBSparse matrix."""
        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        dense = dsbsparse.to_dense()

        assert np.allclose(-dense, (-dsbsparse).to_dense())


# Shape of the
ARRAY_SHAPE = (12, 10, 30)


@pytest.fixture(autouse=True)
def array() -> np.ndarray:
    """Returns a random dense array."""
    return np.random.rand(*ARRAY_SHAPE)


@pytest.mark.parametrize(
    "axis",
    [
        pytest.param(0, id="axis-0"),
        pytest.param(-1, id="axis-(-1)"),
    ],
)
@pytest.mark.parametrize(
    "num_blocks",
    [
        pytest.param(2, id="2-blocks"),
        pytest.param(3, id="3-blocks"),
        pytest.param(5, id="5-blocks"),
    ],
)
def test_block_view(array: np.ndarray, axis: int, num_blocks: int):
    """Tests the block view helper function."""
    with (
        pytest.raises(ValueError)
        if ARRAY_SHAPE[axis] % num_blocks != 0
        else nullcontext()
    ):
        array = get_device(array)
        view = _block_view(array, axis, num_blocks)
        assert view.shape[0] == num_blocks

        for i in range(num_blocks):
            index = [slice(None)] * array.ndim
            size = array.shape[axis] // num_blocks
            index[axis] = slice(i * size, (i + 1) * size)
            assert (array[*index] == view[i]).all()


@pytest.mark.mpi(min_size=2)
class TestDistribution:
    """Tests for the distribution methods of DSBSparse."""

    @pytest.mark.usefixtures("densify_blocks")
    def test_from_sparray(
        self,
        coo: sparse.coo_array,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests distributed creation of DSBSparse matrices from sparrays."""
        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        assert np.array_equiv(coo.toarray(), dsbsparse.to_dense())

        stack_section_sizes, __ = get_section_sizes(global_stack_shape[0], comm.size)
        section_size = stack_section_sizes[comm.rank]
        local_stack_shape = (section_size,) + global_stack_shape[1:]
        assert dsbsparse.to_dense().shape == (*local_stack_shape,) + coo.shape

    def test_dtranspose(
        self,
        coo: sparse.coo_array,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
    ):
        """Tests the distributed transpose method."""
        dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        assert dsbsparse.distribution_state == "stack"

        original_data = dsbsparse._data.copy()

        # Transpose forth.
        dsbsparse.dtranspose()
        assert dsbsparse.distribution_state == "nnz"

        # Transpose back.
        dsbsparse.dtranspose()
        assert dsbsparse.distribution_state == "stack"

        comm.barrier()

        assert np.allclose(original_data, dsbsparse._data)
