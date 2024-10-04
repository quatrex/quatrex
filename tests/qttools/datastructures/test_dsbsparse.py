# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from contextlib import nullcontext

import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD as comm
from scipy import sparse

from qttools.datastructures.dsbsparse import DSBSparse, _block_view
from qttools.utils.gpu_utils import get_device, get_host
from qttools.utils.mpi_utils import get_section_sizes


def _create_coo(sizes) -> sparse.coo_array:
    """Returns a random complex sparse array."""
    size = int(np.sum(sizes))
    rng = np.random.default_rng()
    density = rng.uniform(low=0.1, high=0.3)
    return sparse.random(size, size, density=density, format="coo", dtype=np.complex128)


@pytest.mark.usefixtures("densify_blocks")
class TestCreation:
    """Tests the creation methods of DSBSparse."""

    def test_from_sparray(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: int | tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the creation of DSBSparse matrices from sparse arrays."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        assert np.array_equiv(coo.toarray(), get_host(dsbsparse.to_dense()))

    def test_zeros_like(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: int | tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the creation of a zero DSBSparse matrix with the same shape as another."""
        coo = _create_coo(block_sizes)
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


def _get_block_inds(block: tuple, block_sizes: np.ndarray) -> tuple:
    """Returns the equivalent dense indices for a block."""
    block_offsets = np.hstack(([0], np.cumsum(block_sizes)))
    num_blocks = len(block_sizes)

    # Normalize negative indices.
    row, col, in_bounds = _unsign_index(*block, num_blocks)
    index = (
        slice(block_offsets[row], block_offsets[row + 1]),
        slice(block_offsets[col], block_offsets[col + 1]),
    )

    return index, in_bounds


class TestConversion:
    """Tests for the conversion methods of DSBSparse."""

    def test_to_dense(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
    ):
        """Tests that we can convert a DSBSparse matrix to dense."""
        coo = _create_coo(block_sizes)

        reference = np.broadcast_to(coo.toarray(), global_stack_shape + coo.shape)
        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )

        assert np.allclose(reference, dsbsparse.to_dense())

    def test_ltranspose(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
    ):
        """Tests that we can transpose a DSBSparse matrix."""
        coo = _create_coo(block_sizes)

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

    @pytest.mark.usefixtures("accessed_element")
    def test_getitem(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests that we can get individual matrix elements."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsbsparse.to_dense()

        reference = dense[..., *accessed_element]
        assert np.allclose(reference, dsbsparse[accessed_element])

    @pytest.mark.usefixtures("accessed_element")
    def test_setitem(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests that we can set individual matrix elements."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsbsparse.to_dense()

        dsbsparse[accessed_element] = get_device(42)

        dense[..., *accessed_element][dense[..., *accessed_element].nonzero()] = 42
        assert np.allclose(dense, dsbsparse.to_dense())

    @pytest.mark.usefixtures("accessed_block")
    def test_get_block(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        accessed_block: tuple,
    ):
        """Tests that we can get the correct block."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
        reference_block = dense[..., *inds]

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            assert np.allclose(reference_block, dsbsparse.blocks[accessed_block])

    @pytest.mark.usefixtures("accessed_block", "densify_blocks")
    def test_set_block(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
        accessed_block: tuple,
    ):
        """Tests that we can set a block and not modify sparsity structure."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
            densify_blocks=densify_blocks,
        )
        dense = dsbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            dsbsparse.blocks[accessed_block] = get_device(
                np.ones_like(dense[..., *inds])
            )

        if densify_blocks is not None and accessed_block in densify_blocks:
            # Sparsity structure should be modified.
            assert (dsbsparse.to_dense()[..., *inds] == 1).all()
        else:
            # Sparsity structure should not be modified.
            dense[..., *inds][dense[..., *inds].nonzero()] = 1
            assert np.allclose(dense, dsbsparse.to_dense())

    @pytest.mark.usefixtures("accessed_block", "stack_index")
    def test_get_block_substack(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        accessed_block: tuple,
        stack_index: tuple,
    ):
        """Tests that we can get the correct block from a substack."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
        inds = (
            stack_index
            + (slice(None),) * (len(global_stack_shape) - len(stack_index))
            + inds
        )
        reference_block = dense[inds]

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            assert np.allclose(
                reference_block, dsbsparse.stack[stack_index].blocks[accessed_block]
            )

    @pytest.mark.usefixtures("accessed_block", "densify_blocks", "stack_index")
    def test_set_block_substack(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
        accessed_block: tuple,
        stack_index: tuple,
    ):
        """Tests that we can set a block in a substack and not modify sparsity structure."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
            densify_blocks=densify_blocks,
        )
        dense = dsbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
        inds = (
            stack_index
            + (slice(None),) * (len(global_stack_shape) - len(stack_index))
            + inds
        )

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            dsbsparse.stack[stack_index].blocks[accessed_block] = get_device(
                np.ones_like(dense[inds])
            )

        if densify_blocks is not None and accessed_block in densify_blocks:
            # Sparsity structure should be modified.
            assert (dsbsparse.to_dense()[inds] == 1).all()
        else:
            # Sparsity structure should not be modified.
            dense[inds][dense[inds].nonzero()] = 1
            assert np.allclose(dense, dsbsparse.to_dense())

    def test_spy(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
    ):
        """Tests that we can get the correct sparsity pattern."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        inds = np.lexsort((coo.col, coo.row))
        ref_col, ref_row = coo.col[inds], coo.row[inds]

        rows, cols = dsbsparse.spy()
        inds = np.lexsort((get_host(cols), get_host(rows)))
        col, row = cols[inds], rows[inds]

        assert np.allclose(ref_col, col)
        assert np.allclose(ref_row, row)

    def test_diagonal(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
    ):
        """Tests that we can get the correct diagonal elements."""
        coo = _create_coo(block_sizes)
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
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the in-place addition of a DSBSparse matrix."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        dense = dsbsparse.to_dense()

        dsbsparse += dsbsparse

        assert np.allclose(dense + dense, dsbsparse.to_dense())

    def test_isub(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the in-place subtraction of a DSBSparse matrix."""
        coo = _create_coo(block_sizes)

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
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the in-place multiplication of a DSBSparse matrix."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        dense = dsbsparse.to_dense()

        dsbsparse *= dsbsparse

        assert np.allclose(dense * dense, dsbsparse.to_dense())

    def test_neg(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the negation of a DSBSparse matrix."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        dense = dsbsparse.to_dense()

        assert np.allclose(-dense, (-dsbsparse).to_dense())


# Shape of the dense array.
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
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests distributed creation of DSBSparse matrices from sparrays."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo = comm.bcast(coo, root=0)

        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        assert np.array_equiv(coo.toarray(), get_host(dsbsparse.to_dense()))

        stack_section_sizes, __ = get_section_sizes(global_stack_shape[0], comm.size)
        section_size = stack_section_sizes[comm.rank]
        local_stack_shape = (section_size,) + global_stack_shape[1:]
        assert dsbsparse.to_dense().shape == (*local_stack_shape,) + coo.shape

    def test_dtranspose(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
    ):
        """Tests the distributed transpose method."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo = comm.bcast(coo, root=0)

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

    @pytest.mark.usefixtures("accessed_element")
    def test_getitem_stack(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests distributed access of individual matrix elements."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo = comm.bcast(coo, root=0)

        dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsbsparse.to_dense()

        reference = dense[..., *accessed_element]
        assert np.allclose(reference, dsbsparse[accessed_element])

    @pytest.mark.usefixtures("accessed_element")
    def test_setitem_stack(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests distributed setting of individual matrix elements."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo = comm.bcast(coo, root=0)

        dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsbsparse.to_dense()

        dsbsparse[accessed_element] = get_device(42)

        dense[..., *accessed_element][dense[..., *accessed_element].nonzero()] = 42
        assert np.allclose(dense, dsbsparse.to_dense())

    @pytest.mark.usefixtures("accessed_element")
    def test_getitem_nnz(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests distributed access of individual matrix elements."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo = comm.bcast(coo, root=0)

        dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsbsparse.to_dense()
        rows, cols = dsbsparse.spy()
        row, col, __ = _unsign_index(*accessed_element, dense.shape[-1])
        ind = np.where((rows == row) & (cols == col))[0]

        reference = dense[..., *accessed_element].flatten()[0]

        dsbsparse.dtranspose()

        if len(ind) == 0:
            with pytest.raises(IndexError):
                dsbsparse[accessed_element]
            return

        rank = np.where(dsbsparse.nnz_section_offsets <= ind[0])[0][-1]
        if rank == comm.rank:
            assert np.allclose(reference, dsbsparse[accessed_element])
        else:
            with pytest.raises(IndexError):
                dsbsparse[accessed_element]

    @pytest.mark.usefixtures("accessed_element")
    def test_setitem_nnz(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests distributed setting of individual matrix elements."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo = comm.bcast(coo, root=0)

        dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsbsparse.to_dense()
        rows, cols = dsbsparse.spy()
        row, col, __ = _unsign_index(*accessed_element, dense.shape[-1])
        ind = np.where((rows == row) & (cols == col))[0]

        if len(ind) == 0:
            return

        dense[..., *accessed_element][dense[..., *accessed_element].nonzero()] = 42

        dsbsparse.dtranspose()

        rank = np.where(dsbsparse.nnz_section_offsets <= ind[0])[0][-1]
        if rank == comm.rank:
            dsbsparse[accessed_element] = 42
            dsbsparse.dtranspose()
        else:
            with pytest.raises(IndexError):
                dsbsparse[accessed_element] = 42
            dsbsparse.dtranspose()
            return

        assert np.allclose(dense, dsbsparse.to_dense())
