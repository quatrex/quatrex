# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from contextlib import nullcontext

import functools
import pytest
from mpi4py.MPI import COMM_WORLD as comm
from types import ModuleType

from qttools import DTypeLike, NDArray, sparse, xp
from qttools.datastructures.dsbsparse import DSBSparse, _block_view
from qttools.utils.mpi_utils import get_section_sizes


def _create_coo(sizes: NDArray, dtype: DTypeLike = xp.complex128, integer: bool = False) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    rng = xp.random.default_rng()
    density = rng.uniform(low=0.5, high=0.8)

    def _rvs(size=None, rng=rng):
        if integer:
            return rng.integers(-5, 5, size=size)
        return rng.uniform(size=size)
    
    is_complex = xp.iscomplexobj(dtype(0))

    if is_complex:
        coo = sparse.random(size, size, density=density, format="coo", data_rvs=_rvs).astype(dtype)
        coo.data += 1j * _rvs(size=coo.nnz)
    else:
        coo = sparse.random(size, size, density=density, format="coo", data_rvs=_rvs).astype(dtype)
    assert coo.dtype == dtype

    return coo


class TestCreation:
    """Tests the creation methods of DSBSparse."""

    def test_from_sparray(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: int | tuple,
        banded_block_size: int,
        datatype: DTypeLike,
    ):
        """Tests the creation of DSBSparse matrices from sparse arrays."""
        coo = _create_coo(block_sizes, dtype=datatype)
        ref = coo.toarray()

        dsbsparse = dsbanded_type.from_sparray(
            coo, block_sizes, global_stack_shape, banded_block_size=banded_block_size
        )
        if isinstance(dsbsparse, tuple):
            val = dsbsparse[0].to_dense() + 1j * dsbsparse[1].to_dense()
        else:
            val = dsbsparse.to_dense()
        
        assert xp.array_equiv(ref, val)

    def test_zeros_like(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: int | tuple,
    ):
        """Tests the creation of a zero DSBSparse matrix with the same shape as another."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbanded_type.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        if isinstance(dsbsparse, tuple):
            zeros = dsbanded_type.zeros_like(dsbsparse[0])
        else:
            zeros = dsbanded_type.zeros_like(dsbsparse)
        assert (zeros.to_dense() == 0).all()
    
    def test_eye(
        self,
        dsbanded_type: DSBSparse,
        half_bandwidth: int,
        block_sizes: NDArray,
        global_stack_shape: int | tuple,
    ):
        """Tests the creation of a DSBanded identiy matrix."""
        size = int(sum(block_sizes))
        reference = xp.eye(size, dtype=xp.complex128)
        dsbsparse = dsbanded_type.eye(size, half_bandwidth, block_sizes, global_stack_shape)
        assert xp.allclose(reference, dsbsparse.to_dense())


def _create_new_block_sizes(
    block_sizes: NDArray, block_change_factor: float
) -> NDArray:
    """Creates new block sizes based on the block change factor."""
    rest = 0
    updated_block_sizes = []
    for bs in block_sizes:
        if sum(updated_block_sizes) < sum(block_sizes):
            # Calculate the new block size.
            el = max(int(bs * block_change_factor), 1)
            # Calculate the number of repetitions and the rest. The rest is added to the next block.
            reps, rest = max(divmod(bs + rest, el), (1, 0))
            # Add the new block size to the list.
            updated_block_sizes = updated_block_sizes + [el] * int(reps)
        else:
            # Break if the sum of the updated block sizes is equal or greater than the sum of the original block sizes.
            break
    if sum(updated_block_sizes) != sum(block_sizes):
        # Add the rest to the last block.
        updated_block_sizes[-1] += sum(block_sizes) - sum(updated_block_sizes)
    return xp.asarray(updated_block_sizes)


def _unsign_index(row: int, col: int, num_blocks) -> tuple:
    """Adjusts the sign to allow negative indices and checks bounds."""
    row = num_blocks + row if row < 0 else row
    col = num_blocks + col if col < 0 else col
    in_bounds = 0 <= row < num_blocks and 0 <= col < num_blocks
    return row, col, in_bounds


def _get_block_inds(block: tuple, block_sizes: NDArray) -> tuple:
    """Returns the equivalent dense indices for a block."""
    block_offsets = xp.hstack(([0], xp.cumsum(block_sizes)))
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
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        datatype: DTypeLike,
    ):
        """Tests that we can convert a DSBSparse matrix to dense."""
        coo = _create_coo(block_sizes, dtype=datatype)
        reference = xp.broadcast_to(coo.toarray(), global_stack_shape + coo.shape)

        dsbsparse = dsbanded_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        if isinstance(dsbsparse, tuple):
            val = dsbsparse[0].to_dense() + 1j * dsbsparse[1].to_dense()
        else:
            val = dsbsparse.to_dense()

        assert xp.allclose(reference, val)

    def test_ltranspose(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        datatype: DTypeLike,
    ):
        """Tests that we can transpose a DSBSparse matrix."""
        coo = _create_coo(block_sizes, dtype=datatype)

        dense = xp.broadcast_to(coo.toarray(), global_stack_shape + coo.shape)
        reference = xp.swapaxes(dense, -2, -1)

        dsbsparse = dsbanded_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        if isinstance(dsbsparse, tuple):
            val = (dsbsparse[0].ltranspose(copy=True).to_dense() +
                   1j * dsbsparse[1].ltranspose(copy=True).to_dense())
        else:
            val = dsbsparse.ltranspose(copy=True).to_dense()

        # Test copy transpose
        assert xp.allclose(reference, val)

        # Transpose forth.
        if isinstance(dsbsparse, tuple):
            dsbsparse[0].ltranspose()  # In-place transpose.
            dsbsparse[1].ltranspose()  # In-place transpose.
            val = (dsbsparse[0].to_dense() + 1j * dsbsparse[1].to_dense())
        else:
            dsbsparse.ltranspose()  # In-place transpose.
            val = dsbsparse.to_dense()

        assert xp.allclose(reference, val)

        # Transpose back.
        if isinstance(dsbsparse, tuple):
            dsbsparse[0].ltranspose()
            dsbsparse[1].ltranspose()
            val = (dsbsparse[0].to_dense() + 1j * dsbsparse[1].to_dense())
        else:
            dsbsparse.ltranspose()
            val = dsbsparse.to_dense()

        assert xp.allclose(dense, val)


class TestAccess:
    """Tests for the access methods of DSBSparse."""

    @pytest.mark.usefixtures("accessed_element")
    def test_getitem(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_element: tuple,
        datatype: DTypeLike,
    ):
        """Tests that we can get individual matrix elements."""
        coo = _create_coo(block_sizes, dtype=datatype)
        dsbsparse = dsbanded_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        if isinstance(dsbsparse, tuple):
            dense = dsbsparse[0].to_dense() + 1j * dsbsparse[1].to_dense()
            val = dsbsparse[0][accessed_element] + 1j * dsbsparse[1][accessed_element]
        else:
            dense = dsbsparse.to_dense()
            val = dsbsparse[accessed_element]
        reference = dense[..., *accessed_element]

        assert xp.allclose(reference, val)

    @pytest.mark.usefixtures("num_inds")
    def test_getitem_with_array(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        num_inds: int,
        datatype: DTypeLike,
    ):
        """Tests that we can get multiple matrix elements at once."""
        coo = _create_coo(block_sizes, dtype=datatype)
        dsbsparse = dsbanded_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )

        # Generate a number of unique indices.
        rows = xp.random.choice(coo.shape[0], size=num_inds, replace=False)
        cols = xp.random.choice(coo.shape[1], size=num_inds, replace=False)

        reference_data = coo.tocsr()[rows, cols]
        if sparse.issparse(reference_data):
            reference_data = reference_data.toarray()
        reference = xp.broadcast_to(
            reference_data, global_stack_shape + reference_data.shape[1:]
        )
        if isinstance(dsbsparse, tuple):
            val = dsbsparse[0][rows, cols] + 1j * dsbsparse[1][rows, cols]
        else:
            val = dsbsparse[rows, cols]
        assert xp.allclose(reference, val)

    @pytest.mark.usefixtures("accessed_element")
    def test_setitem(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_element: tuple,
        datatype: DTypeLike,
    ):
        """Tests that we can set individual matrix elements."""
        coo = _create_coo(block_sizes, dtype=datatype)
        dsbsparse = dsbanded_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        if isinstance(dsbsparse, tuple):
            dense = dsbsparse[0].to_dense() + 1j * dsbsparse[1].to_dense()
            dense[..., *accessed_element] = 42 + 1j * 42
            dsbsparse[0][accessed_element] = 42
            dsbsparse[1][accessed_element] = 42
            val = dsbsparse[0].to_dense() + 1j * dsbsparse[1].to_dense()
        else:
            dense = dsbsparse.to_dense()
            dense[..., *accessed_element] = 42
            dsbsparse[accessed_element] = 42
            val = dsbsparse.to_dense()

        # NOTE: Banded datastructures are not sparse and they will write outside the original sparsity pattern.
        # dense[..., *accessed_element][dense[..., *accessed_element].nonzero()] = 42
        assert xp.allclose(dense, val)

    @pytest.mark.usefixtures("accessed_block")
    def test_get_block(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_block: tuple,
    ):
        """Tests that we can get the correct block."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbanded_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        if isinstance(dsbsparse, tuple):
            dense = dsbsparse[0].to_dense() + 1j * dsbsparse[1].to_dense()
        else:
            dense = dsbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
        reference_block = dense[..., *inds]

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            if isinstance(dsbsparse, tuple):
                val = dsbsparse[0].blocks[accessed_block] + 1j * dsbsparse[1].blocks[accessed_block]
            else:
                val = dsbsparse.blocks[accessed_block]
            assert xp.allclose(reference_block, val)

    # @pytest.mark.usefixtures("accessed_block")
    # def test_get_sparse_block(
    #     self,
    #     dsbanded_type: DSBSparse,
    #     block_sizes: NDArray,
    #     global_stack_shape: tuple,
    #     accessed_block: tuple,
    # ):
    #     """Tests that we can get the correct block."""
    #     coo = _create_coo(block_sizes)
    #     dsbsparse = dsbanded_type.from_sparray(
    #         coo,
    #         block_sizes=block_sizes,
    #         global_stack_shape=global_stack_shape,
    #     )
    #     dense = dsbsparse.to_dense()

    #     inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
    #     reference_block = dense[..., *inds]

    #     # We want to get sparse blocks.
    #     dsbsparse.return_dense = False

    #     with pytest.raises(IndexError) if not in_bounds else nullcontext():
    #         if "CSR" in dsbanded_type.__name__:
    #             rowptr, cols, data = dsbsparse.blocks[accessed_block]
    #             for ind in xp.ndindex(reference_block.shape[:-2]):
    #                 block = sparse.csr_matrix(
    #                     (data[ind], cols, rowptr),
    #                     shape=reference_block.shape[-2:],
    #                 )
    #                 assert xp.allclose(reference_block[ind], block.toarray())

    #         elif "COO" in dsbanded_type.__name__:
    #             rows, cols, data = dsbsparse.blocks[accessed_block]
    #             for ind in xp.ndindex(reference_block.shape[:-2]):
    #                 block = sparse.coo_matrix(
    #                     (data[ind], (rows, cols)), shape=reference_block.shape[-2:]
    #                 )
    #                 assert xp.allclose(reference_block[ind], block.toarray())

    #         else:
    #             raise ValueError("Unknown DSBSparse type.")

    @pytest.mark.usefixtures("accessed_block")
    def test_set_block(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_block: tuple,
    ):
        """Tests that we can set a block."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbanded_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        if isinstance(dsbsparse, tuple):
            dense = dsbsparse[0].to_dense() + 1j * dsbsparse[1].to_dense()
        else:
            dense = dsbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            if isinstance(dsbsparse, tuple):
                dsbsparse[0].blocks[accessed_block] = xp.ones_like(dense[..., *inds].real)
                dsbsparse[1].blocks[accessed_block] = xp.ones_like(dense[..., *inds].imag)
            else:
                dsbsparse.blocks[accessed_block] = xp.ones_like(dense[..., *inds])

        # Sparsity structure should be modified.
        if isinstance(dsbsparse, tuple):
            assert (dsbsparse[0].to_dense()[..., *inds] == 1).all()
            assert (dsbsparse[1].to_dense()[..., *inds] == 1).all()
        else:
            assert (dsbsparse.to_dense()[..., *inds] == 1).all()

    @pytest.mark.usefixtures("accessed_block", "stack_index")
    def test_get_block_substack(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_block: tuple,
        stack_index: tuple,
    ):
        """Tests that we can get the correct block from a substack."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbanded_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        if isinstance(dsbsparse, tuple):
            dense = dsbsparse[0].to_dense() + 1j * dsbsparse[1].to_dense()
        else:
            dense = dsbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
        inds = (
            stack_index
            + (slice(None),) * (len(global_stack_shape) - len(stack_index))
            + inds
        )
        reference_block = dense[inds]
        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            if isinstance(dsbsparse, tuple):
                val = (dsbsparse[0].stack[stack_index].blocks[accessed_block] +
                       1j * dsbsparse[1].stack[stack_index].blocks[accessed_block])
            else:
                val = dsbsparse.stack[stack_index].blocks[accessed_block]
            assert xp.allclose(reference_block, val)

    # @pytest.mark.usefixtures("accessed_block", "stack_index")
    # def test_get_sparse_block_substack(
    #     self,
    #     dsbanded_type: DSBSparse,
    #     block_sizes: NDArray,
    #     global_stack_shape: tuple,
    #     accessed_block: tuple,
    #     stack_index: tuple,
    # ):
    #     """Tests that we can get the correct block from a substack."""
    #     coo = _create_coo(block_sizes)
    #     dsbsparse = dsbanded_type.from_sparray(
    #         coo,
    #         block_sizes=block_sizes,
    #         global_stack_shape=global_stack_shape,
    #     )
    #     dense = dsbsparse.to_dense()

    #     inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
    #     inds = (
    #         stack_index
    #         + (slice(None),) * (len(global_stack_shape) - len(stack_index))
    #         + inds
    #     )
    #     reference_block = dense[inds]

    #     # We want to get sparse blocks.
    #     dsbsparse.return_dense = False

    #     with pytest.raises(IndexError) if not in_bounds else nullcontext():
    #         if "CSR" in dsbanded_type.__name__:
    #             rowptr, cols, data = dsbsparse.stack[stack_index].blocks[accessed_block]
    #             for ind in xp.ndindex(reference_block.shape[:-2]):
    #                 block = sparse.csr_matrix(
    #                     (data[ind], cols, rowptr),
    #                     shape=reference_block.shape[-2:],
    #                 )
    #                 assert xp.allclose(reference_block[ind], block.toarray())

    #         elif "COO" in dsbanded_type.__name__:
    #             rows, cols, data = dsbsparse.stack[stack_index].blocks[accessed_block]
    #             for ind in xp.ndindex(reference_block.shape[:-2]):
    #                 block = sparse.coo_matrix(
    #                     (data[ind], (rows, cols)), shape=reference_block.shape[-2:]
    #                 )
    #                 assert xp.allclose(reference_block[ind], block.toarray())

    #         else:
    #             raise ValueError("Unknown DSBSparse type.")

    @pytest.mark.usefixtures("accessed_block", "stack_index")
    def test_set_block_substack(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_block: tuple,
        stack_index: tuple,
    ):
        """Tests that we can set a block in a substack and not modify sparsity structure."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbanded_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        if isinstance(dsbsparse, tuple):
            dense = dsbsparse[0].to_dense() + 1j * dsbsparse[1].to_dense()
        else:
            dense = dsbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
        inds = (
            stack_index
            + (slice(None),) * (len(global_stack_shape) - len(stack_index))
            + inds
        )

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            if isinstance(dsbsparse, tuple):
                dsbsparse[0].stack[stack_index].blocks[accessed_block] = xp.ones_like(dense[inds].real)
                dsbsparse[1].stack[stack_index].blocks[accessed_block] = xp.ones_like(dense[inds].imag)
            else:
                dsbsparse.stack[stack_index].blocks[accessed_block] = xp.ones_like(dense[inds])

        # Sparsity structure should be modified.
        if isinstance(dsbsparse, tuple):
            assert (dsbsparse[0].to_dense()[inds] == 1).all()
            assert (dsbsparse[1].to_dense()[inds] == 1).all()
        else:
            assert (dsbsparse.to_dense()[inds] == 1).all()

    @pytest.mark.usefixtures("block_change_factor")
    def test_block_sizes_setter(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        block_change_factor: float,
    ):
        """Tests that we can update the block sizes correctly."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbanded_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        # Create new block sizes.
        updated_block_sizes = _create_new_block_sizes(block_sizes, block_change_factor)
        # Create a new DSBSparse matrix with the updated block sizes.
        dsbsparse_updated_block_sizes = dsbanded_type.from_sparray(
            coo,
            block_sizes=updated_block_sizes,
            global_stack_shape=global_stack_shape,
        )

        # Update the block sizes.
        if isinstance(dsbsparse, tuple):
            dsbsparse[0].block_sizes = updated_block_sizes
            dsbsparse[1].block_sizes = updated_block_sizes
        else:
            dsbsparse.block_sizes = updated_block_sizes

        # Assert that the two DSBSparse matrices are equivalent.
        if isinstance(dsbsparse, tuple):
            assert (dsbsparse[0].data == dsbsparse_updated_block_sizes[0].data).all()
            assert (dsbsparse[1].data == dsbsparse_updated_block_sizes[1].data).all()
        else:
            assert (dsbsparse.data == dsbsparse_updated_block_sizes.data).all()

    # def test_spy(
    #     self,
    #     dsbanded_type: DSBSparse,
    #     block_sizes: NDArray,
    #     global_stack_shape: tuple,
    # ):
    #     """Tests that we can get the correct sparsity pattern."""
    #     coo = _create_coo(block_sizes)
    #     dsbsparse = dsbanded_type.from_sparray(
    #         coo,
    #         block_sizes=block_sizes,
    #         global_stack_shape=global_stack_shape,
    #     )
    #     inds = xp.lexsort(xp.vstack((coo.col, coo.row)))
    #     ref_col, ref_row = coo.col[inds], coo.row[inds]

    #     rows, cols = dsbsparse.spy()
    #     inds = xp.lexsort(xp.vstack((cols, rows)))
    #     col, row = cols[inds], rows[inds]

    #     assert xp.allclose(ref_col, col)
    #     assert xp.allclose(ref_row, row)

    def test_diagonal(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests that we can get the correct diagonal elements."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbanded_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        if isinstance(dsbsparse, tuple):
            dense = dsbsparse[0].to_dense() + 1j * dsbsparse[1].to_dense()
            val = dsbsparse[0].diagonal() + 1j * dsbsparse[1].diagonal()
        else:
            dense = dsbsparse.to_dense()
            val = dsbsparse.diagonal()

        reference = xp.diagonal(dense, axis1=-2, axis2=-1)
        assert xp.allclose(reference, val)


@pytest.mark.skip
class TestArithmetic:
    """Tests for the arithmetic operations of DSBSparse."""

    def test_iadd(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place addition of a DSBSparse matrix."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbanded_type.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        dense = dsbsparse.to_dense()

        dsbsparse += dsbsparse

        assert xp.allclose(dense + dense, dsbsparse.to_dense())

    # def test_iadd_coo(
    #     self,
    #     dsbanded_type: DSBSparse,
    #     block_sizes: NDArray,
    #     global_stack_shape: tuple,
    #     densify_blocks: list[tuple] | None,
    # ):
    #     """Tests the in-place addition of a DSBSparse matrix with a COO matrix."""
    #     coo = _create_coo(block_sizes)
    #     dsbsparse = dsbanded_type.from_sparray(
    #         coo, block_sizes, global_stack_shape, densify_blocks
    #     )

    #     dsbsparse += coo.copy()

    #     assert xp.allclose(dsbsparse.to_dense(), 2 * coo.toarray())

    def test_isub(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place subtraction of a DSBSparse matrix."""
        coo = _create_coo(block_sizes)

        dsbsparse_1 = dsbanded_type.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        dense_1 = dsbsparse_1.to_dense()

        dsbsparse_2 = dsbanded_type.from_sparray(
            2 * coo, block_sizes, global_stack_shape
        )
        dense_2 = dsbsparse_2.to_dense()

        dsbsparse_1 -= dsbsparse_2

        assert xp.allclose(dense_1 - dense_2, dsbsparse_1.to_dense())

    # def test_isub_coo(
    #     self,
    #     dsbanded_type: DSBSparse,
    #     block_sizes: NDArray,
    #     global_stack_shape: tuple,
    #     densify_blocks: list[tuple] | None,
    # ):
    #     """Tests the in-place subtraction of a DSBSparse matrix with a COO matrix."""
    #     coo = _create_coo(block_sizes)

    #     dsbsparse = dsbanded_type.from_sparray(
    #         coo, block_sizes, global_stack_shape, densify_blocks
    #     )
    #     dense = dsbsparse.to_dense()

    #     dsbsparse -= 2 * coo

    #     assert xp.allclose(dense - 2 * coo.toarray(), dsbsparse.to_dense())

    def test_imul(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place multiplication of a DSBSparse matrix."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbanded_type.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        dense = dsbsparse.to_dense()

        dsbsparse *= dsbsparse

        assert xp.allclose(dense * dense, dsbsparse.to_dense())

    def test_neg(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the negation of a DSBSparse matrix."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbanded_type.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        dense = dsbsparse.to_dense()

        assert xp.allclose(-dense, (-dsbsparse).to_dense())


@pytest.mark.skipif(xp.__name__ != "cupy", reason="DSBanded matmul tests require a GPU.")
class TestMatmul:
    """Tests for matrix multiplications with DSBanded matrices."""

    def test_matmul(
        self,
        dsbanded_matmul_type: tuple[DSBSparse, DSBSparse],
        block_sizes: NDArray,
        global_stack_shape: tuple,
        banded_block_size: int,
        datatype: DTypeLike,
        dtype: tuple[ModuleType, DTypeLike],
    ):
        """Tests the matrix multiplication of a DSBSparse matrix."""

        def _set_torch(dsbsparse: DSBSparse, mod: ModuleType, dt: DTypeLike):

            if isinstance(dsbsparse, tuple):
                matrices = [dsbsparse[0], dsbsparse[1]]
            else:
                matrices = [dsbsparse]

            for dsbsparse in matrices:
                batch_size = functools.reduce(lambda x, y: x * y, dsbsparse.data.shape[:len(dsbsparse.global_stack_shape)])
                banded_data = dsbsparse.data.reshape((batch_size, *dsbsparse.banded_shape))
                if mod.__name__ == "cupy":
                    import torch
                    banded_data = banded_data.astype(dt)
                    dsbsparse.torch = torch.asarray(banded_data, device='cuda')
                else:  # mod.__name__ == "torch"
                    dsbsparse.torch = mod.asarray(banded_data, dtype=dt, device='cuda')


        dsbanded_type_a, dsbanded_type_b = dsbanded_matmul_type
        mod, dt = dtype

        coo = _create_coo(block_sizes, dtype=datatype, integer=True)
        dense = coo.toarray()

        dsbsparse_a = dsbanded_type_a.from_sparray(
            coo, block_sizes, global_stack_shape, banded_block_size=banded_block_size
        )

        dsbsparse_b = dsbanded_type_b.from_sparray(
            coo, block_sizes, global_stack_shape, banded_block_size=banded_block_size
        )

        # if mod.__name__ == "cupy":
        #     dense = dense.astype(dt)
        #     reference = dense @ dense
        # elif mod.__name__ == "torch":
        #     dense = mod.asarray(dense, dtype=dt)
        #     reference = dense @ dense
        #     reference = xp.asarray(reference)
        # else:
        #     raise NotImplementedError

        reference = dense @ dense
        _set_torch(dsbsparse_a, mod, dt)
        _set_torch(dsbsparse_b, mod, dt)

        if isinstance(dsbsparse_a, tuple):
            if isinstance(dsbsparse_b, tuple):
                real = dsbsparse_a[0] @ dsbsparse_b[0] - dsbsparse_a[1] @ dsbsparse_b[1]
                imag = dsbsparse_a[0] @ dsbsparse_b[1] + dsbsparse_a[1] @ dsbsparse_b[0]
                value = (real, imag)
            else:
                real = dsbsparse_a[0] @ dsbsparse_b
                imag = dsbsparse_a[1] @ dsbsparse_b
                value = (real, imag)
        else:
            if isinstance(dsbsparse_b, tuple):
                real = dsbsparse_a @ dsbsparse_b[0]
                imag = dsbsparse_a @ dsbsparse_b[1]
                value = (real, imag)
            else:
                value = dsbsparse_a @ dsbsparse_b
        if isinstance(value, tuple):
            value = value[0].to_dense() + 1j * value[1].to_dense()
        else:
            value = value.to_dense()

        relerror = xp.linalg.norm(reference - value) / xp.linalg.norm(reference)
        print(relerror)
        assert xp.allclose(dense @ dense, value)


# Shape of the dense array.
ARRAY_SHAPE = (12, 10, 30)


@pytest.fixture(autouse=True)
def array() -> NDArray:
    """Returns a random dense array."""
    return xp.random.rand(*ARRAY_SHAPE)


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
@pytest.mark.skip
def test_block_view(array: NDArray, axis: int, num_blocks: int):
    """Tests the block view helper function."""
    with (
        pytest.raises(ValueError)
        if ARRAY_SHAPE[axis] % num_blocks != 0
        else nullcontext()
    ):
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

    def test_from_sparray(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests distributed creation of DSBSparse matrices from sparrays."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo: sparse.coo_matrix = comm.bcast(coo, root=0)

        dsbsparse = dsbanded_type.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        assert xp.array_equiv(coo.toarray(), dsbsparse.to_dense())

        stack_section_sizes, __ = get_section_sizes(global_stack_shape[0], comm.size)
        section_size = stack_section_sizes[comm.rank]
        local_stack_shape = (section_size,) + global_stack_shape[1:]
        assert dsbsparse.to_dense().shape == (*local_stack_shape,) + coo.shape

    def test_dtranspose(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the distributed transpose method."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo: sparse.coo_matrix = comm.bcast(coo, root=0)

        dsbsparse = dsbanded_type.from_sparray(coo, block_sizes, global_stack_shape)
        assert dsbsparse.distribution_state == "stack"

        original_data = dsbsparse._data.copy()

        # Transpose forth.
        dsbsparse.dtranspose()
        assert dsbsparse.distribution_state == "nnz"

        # Transpose back.
        dsbsparse.dtranspose()
        assert dsbsparse.distribution_state == "stack"

        comm.barrier()

        assert xp.allclose(original_data, dsbsparse._data)

    @pytest.mark.usefixtures("accessed_element")
    def test_getitem_stack(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests distributed access of individual matrix elements."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo: sparse.coo_matrix = comm.bcast(coo, root=0)

        dsbsparse = dsbanded_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsbsparse.to_dense()

        reference = dense[..., *accessed_element]
        print(dsbsparse[accessed_element].shape, flush=True) if comm.rank == 0 else None
        print(reference.shape, flush=True) if comm.rank == 0 else None

        assert xp.allclose(reference, dsbsparse[accessed_element])

    @pytest.mark.usefixtures("accessed_element")
    def test_setitem_stack(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests distributed setting of individual matrix elements."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo: sparse.coo_matrix = comm.bcast(coo, root=0)

        dsbsparse = dsbanded_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsbsparse.to_dense()

        dsbsparse[accessed_element] = 42

        # NOTE: Banded datastructures are not sparse and they will write outside the original sparsity pattern.
        # dense[..., *accessed_element][dense[..., *accessed_element].nonzero()] = 42
        dense[..., *accessed_element] = 42
        assert xp.allclose(dense, dsbsparse.to_dense())

    @pytest.mark.usefixtures("accessed_element")
    def test_getitem_nnz(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests distributed access of individual matrix elements."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo: sparse.coo_matrix = comm.bcast(coo, root=0)

        dsbsparse = dsbanded_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsbsparse.to_dense()
        row, col, __ = _unsign_index(*accessed_element, dense.shape[-1])
        ind = [dsbsparse.flatten_index((row, col))]

        reference = dense[..., *accessed_element].flatten()[0]

        dsbsparse.dtranspose()

        if len(ind) == 0:
            with pytest.raises(IndexError):
                dsbsparse[accessed_element]
            return

        rank = xp.where(dsbsparse.nnz_section_offsets <= ind[0])[0][-1]
        if rank == comm.rank:
            assert xp.allclose(reference, dsbsparse[accessed_element])
        else:
            assert dsbsparse[accessed_element].shape[-1] == 0

    @pytest.mark.usefixtures("accessed_element")
    def test_setitem_nnz(
        self,
        dsbanded_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests distributed setting of individual matrix elements."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo: sparse.coo_matrix = comm.bcast(coo, root=0)

        dsbsparse = dsbanded_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsbsparse.to_dense()
        row, col, __ = _unsign_index(*accessed_element, dense.shape[-1])
        ind = [dsbsparse.flatten_index((row, col))]

        if len(ind) == 0:
            return

        # NOTE: Banded datastructures are not sparse and they will write outside the original sparsity pattern.
        # dense[..., *accessed_element][dense[..., *accessed_element].nonzero()] = 42
        dense[..., *accessed_element] = 42

        dsbsparse.dtranspose()

        dsbsparse[accessed_element] = 42

        dsbsparse.dtranspose()

        assert xp.allclose(dense, dsbsparse.to_dense())


if __name__ == "__main__":
    pytest.main(["-s", __file__])
