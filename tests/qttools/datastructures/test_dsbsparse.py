# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from contextlib import nullcontext
import pytest
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, sparse, xp
from qttools.datastructures.dsbcoo import DSBCOO
from qttools.datastructures.dsbsparse import DSBSparse, _block_view
from qttools.datastructures.routines import banded_matmul
from qttools.kernels import dsbcoo_kernels
from qttools.utils.mpi_utils import get_section_sizes
from qttools.utils.sparse_utils import product_sparsity_pattern

torch_cuda_avail = False
try:
    import torch
    if torch.cuda.is_available():
        torch_cuda_avail = True
except (ImportError, ModuleNotFoundError):
    pass


def _create_coo(sizes: NDArray, dtype=xp.complex128) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    rng = xp.random.default_rng()
    density = rng.uniform(low=0.1, high=0.3)
    coo = sparse.random(size, size, density=density, format="coo").astype(dtype)
    if dtype == xp.complex128:
        coo.data += 1j * rng.uniform(size=coo.nnz)
    return coo


def create_dense_band_matrix(N, r, device="cuda", dtype=torch.float64, seed: int = 0, fixed_val=None):

    # Create indices for all positions
    i, j = torch.meshgrid(
        torch.arange(N, device=device),
        torch.arange(N, device=device),
        indexing="ij",
    )

    # Create band mask
    mask = torch.abs(i - j) <= r

    # Create random matrix and apply mask
    if fixed_val is not None:
        A = torch.full((N, N), fixed_val, device=device, dtype=dtype) * mask
    else:
        A = torch.randn(N, N, device=device, dtype=dtype) * mask

    if dtype == torch.complex128:
        A += 1j * A
    A_np = A.detach().cpu().numpy()
    A_cupy = xp.array(A_np)

    return sparse.coo_matrix(A_cupy)


@pytest.mark.usefixtures("densify_blocks")
class TestCreation:
    """Tests the creation methods of DSBSparse."""

    def test_from_sparray(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: int | tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the creation of DSBSparse matrices from sparse arrays."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        assert xp.array_equiv(coo.toarray(), dsbsparse.to_dense())

    def test_zeros_like(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
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
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests that we can convert a DSBSparse matrix to dense."""
        coo = _create_coo(block_sizes)

        reference = xp.broadcast_to(coo.toarray(), global_stack_shape + coo.shape)
        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )

        assert xp.allclose(reference, dsbsparse.to_dense())

    def test_ltranspose(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests that we can transpose a DSBSparse matrix."""
        coo = _create_coo(block_sizes)

        dense = xp.broadcast_to(coo.toarray(), global_stack_shape + coo.shape)
        reference = xp.swapaxes(dense, -2, -1)

        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )

        # Test copy transpose
        assert xp.allclose(reference, dsbsparse.ltranspose(copy=True).to_dense())

        # Transpose forth.
        dsbsparse.ltranspose()  # In-place transpose.

        assert xp.allclose(reference, dsbsparse.to_dense())

        # Transpose back.
        dsbsparse.ltranspose()

        assert xp.allclose(dense, dsbsparse.to_dense())


class TestAccess:
    """Tests for the access methods of DSBSparse."""

    @pytest.mark.usefixtures("accessed_element")
    def test_getitem(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
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

        reference = dense[(..., *accessed_element)]
        assert xp.allclose(reference, dsbsparse[accessed_element])

    @pytest.mark.usefixtures("num_inds")
    def test_getitem_with_array(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        num_inds: int,
    ):
        """Tests that we can get multiple matrix elements at once."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
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
        assert xp.allclose(reference, dsbsparse[rows, cols])

    @pytest.mark.usefixtures("accessed_element")
    def test_setitem(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
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

        dsbsparse[accessed_element] = 42

        dense[(..., *accessed_element)][dense[(..., *accessed_element)].nonzero()] = 42
        assert xp.allclose(dense, dsbsparse.to_dense())

    @pytest.mark.usefixtures("accessed_block")
    def test_get_block(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
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
        reference_block = dense[(..., *inds)]

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            assert xp.allclose(reference_block, dsbsparse.blocks[accessed_block])

    @pytest.mark.usefixtures("accessed_block")
    def test_get_sparse_block(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
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
        reference_block = dense[(..., *inds)]

        # We want to get sparse blocks.
        dsbsparse.return_dense = False

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            if "CSR" in dsbsparse_type.__name__:
                rowptr, cols, data = dsbsparse.blocks[accessed_block]
                for ind in xp.ndindex(reference_block.shape[:-2]):
                    block = sparse.csr_matrix(
                        (data[ind], cols, rowptr),
                        shape=reference_block.shape[-2:],
                    )
                    assert xp.allclose(reference_block[ind], block.toarray())

            elif "COO" in dsbsparse_type.__name__:
                rows, cols, data = dsbsparse.blocks[accessed_block]
                for ind in xp.ndindex(reference_block.shape[:-2]):
                    block = sparse.coo_matrix(
                        (data[ind], (rows, cols)), shape=reference_block.shape[-2:]
                    )
                    assert xp.allclose(reference_block[ind], block.toarray())

            else:
                raise ValueError("Unknown DSBSparse type.")

    @pytest.mark.usefixtures("accessed_block", "densify_blocks")
    def test_set_block(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
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
            dsbsparse.blocks[accessed_block] = xp.ones_like(dense[(..., *inds)])

        if densify_blocks is not None and accessed_block in densify_blocks:
            # Sparsity structure should be modified.
            assert (dsbsparse.to_dense()[(..., *inds)] == 1).all()
        else:
            # Sparsity structure should not be modified.
            dense[(..., *inds)][dense[(..., *inds)].nonzero()] = 1
            assert xp.allclose(dense, dsbsparse.to_dense())

    @pytest.mark.usefixtures("accessed_block", "stack_index")
    def test_get_block_substack(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
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
            assert xp.allclose(
                reference_block, dsbsparse.stack[stack_index].blocks[accessed_block]
            )

    @pytest.mark.usefixtures("accessed_block", "stack_index")
    def test_get_sparse_block_substack(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
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

        # We want to get sparse blocks.
        dsbsparse.return_dense = False

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            if "CSR" in dsbsparse_type.__name__:
                rowptr, cols, data = dsbsparse.stack[stack_index].blocks[accessed_block]
                for ind in xp.ndindex(reference_block.shape[:-2]):
                    block = sparse.csr_matrix(
                        (data[ind], cols, rowptr),
                        shape=reference_block.shape[-2:],
                    )
                    assert xp.allclose(reference_block[ind], block.toarray())

            elif "COO" in dsbsparse_type.__name__:
                rows, cols, data = dsbsparse.stack[stack_index].blocks[accessed_block]
                for ind in xp.ndindex(reference_block.shape[:-2]):
                    block = sparse.coo_matrix(
                        (data[ind], (rows, cols)), shape=reference_block.shape[-2:]
                    )
                    assert xp.allclose(reference_block[ind], block.toarray())

            else:
                raise ValueError("Unknown DSBSparse type.")

    @pytest.mark.usefixtures("accessed_block", "densify_blocks", "stack_index")
    def test_set_block_substack(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
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
            dsbsparse.stack[stack_index].blocks[accessed_block] = xp.ones_like(
                dense[inds]
            )

        if densify_blocks is not None and accessed_block in densify_blocks:
            # Sparsity structure should be modified.
            assert (dsbsparse.to_dense()[inds] == 1).all()
        else:
            # Sparsity structure should not be modified.
            dense[inds][dense[inds].nonzero()] = 1
            assert xp.allclose(dense, dsbsparse.to_dense())

    @pytest.mark.usefixtures("block_change_factor")
    def test_block_sizes_setter(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        block_change_factor: float,
    ):
        """Tests that we can update the block sizes correctly."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        # Create new block sizes.
        updated_block_sizes = _create_new_block_sizes(block_sizes, block_change_factor)
        # Create a new DSBSparse matrix with the updated block sizes.
        dsbsparse_updated_block_sizes = dsbsparse_type.from_sparray(
            coo,
            block_sizes=updated_block_sizes,
            global_stack_shape=global_stack_shape,
        )

        # Update the block sizes.
        dsbsparse.block_sizes = updated_block_sizes

        # Assert that the two DSBSparse matrices are equivalent.
        assert (dsbsparse.data == dsbsparse_updated_block_sizes.data).all()

    def test_spy(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests that we can get the correct sparsity pattern."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        inds = xp.lexsort(xp.vstack((coo.col, coo.row)))
        ref_col, ref_row = coo.col[inds], coo.row[inds]

        rows, cols = dsbsparse.spy()
        inds = xp.lexsort(xp.vstack((cols, rows)))
        col, row = cols[inds], rows[inds]

        assert xp.allclose(ref_col, col)
        assert xp.allclose(ref_row, row)

    def test_diagonal(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
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

        reference = xp.diagonal(dense, axis1=-2, axis2=-1)
        assert xp.allclose(reference, dsbsparse.diagonal())


@pytest.mark.usefixtures("densify_blocks")
class TestArithmetic:
    """Tests for the arithmetic operations of DSBSparse."""

    def test_iadd(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
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

        assert xp.allclose(dense + dense, dsbsparse.to_dense())

    def test_iadd_coo(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the in-place addition of a DSBSparse matrix with a COO matrix."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )

        dsbsparse += coo.copy()

        assert xp.allclose(dsbsparse.to_dense(), 2 * coo.toarray())

    def test_isub(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
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

        assert xp.allclose(dense_1 - dense_2, dsbsparse_1.to_dense())

    def test_isub_coo(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the in-place subtraction of a DSBSparse matrix with a COO matrix."""
        coo = _create_coo(block_sizes)

        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        dense = dsbsparse.to_dense()

        dsbsparse -= 2 * coo

        assert xp.allclose(dense - 2 * coo.toarray(), dsbsparse.to_dense())

    def test_imul(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
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

        assert xp.allclose(dense * dense, dsbsparse.to_dense())

    def test_neg(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the negation of a DSBSparse matrix."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        dense = dsbsparse.to_dense()

        assert xp.allclose(-dense, (-dsbsparse).to_dense())

    def test_matmul(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests the matrix multiplication of a DSBSparse matrix."""
        coo = _create_coo(block_sizes)
        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        dense = dsbsparse.to_dense()

        assert xp.allclose(dense @ dense, (dsbsparse @ dsbsparse).to_dense())

    def test_matmul_banded(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
        band_r: int = 2,
        debug_prints: bool = False,
    ):
        """
        Tests the banded matrix multiplication of a DSBCOO matrix.
        It currently works ONLY for:
        - DBSCOO format (no support for DSBCSR. A minor thing, can be easily
            implemented)
        - square matrices
        - needs pytorch and CUDA
        """

        if dsbsparse_type != DSBCOO:
            return
        if not torch_cuda_avail:
            return
        band_N = int(sum(block_sizes))
        coo = create_dense_band_matrix(band_N, band_r, dtype=torch.float32)

        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        dense = dsbsparse.to_dense()
        ref_result = torch.tensor(dense @ dense)

        product_rows, product_cols = product_sparsity_pattern(coo, coo)
        block_sort_index = dsbcoo_kernels.compute_block_sort_index(
            product_rows, product_cols, dsbsparse.block_sizes
        )
        product = DSBCOO(
            data=xp.zeros(
                dsbsparse.stack_shape + (product_rows.size,), dtype=dsbsparse.dtype
            ),
            rows=product_rows[block_sort_index],
            cols=product_cols[block_sort_index],
            block_sizes=dsbsparse.block_sizes,
            global_stack_shape=dsbsparse.global_stack_shape,
        )

        kwargs = {
            "source_dtype": torch.float16,
            "dest_dtype": torch.float16,
            "allow_tf32": True,
            "BLK_M": 16,
            "BLK_N": 16,
            "BLK_K": 16,
        }
        banded_matmul(dsbsparse, dsbsparse, product, **kwargs)
        sparse_result = product.to_dense()

        if debug_prints:
            # print test parameters
            print(
                f"dsbsparse_type: {dsbsparse_type}, block_sizes: {block_sizes}, \
                global_stack_shape: {global_stack_shape}, \
                densify_blocks: {densify_blocks}, band_r: {band_r}"
            )

            if xp.allclose(ref_result, sparse_result, rtol=1e-2, atol=1e-2):
                print("Verification passed!")
            else:
                print("Verification failed!")
                print("ref_result:\n", ref_result[0].detach().cpu().numpy()[:16, :16])
                print(
                    "sparse_result:\n",
                    sparse_result[0].detach().cpu().numpy()[:16, :16],
                )
                exit()
        else:
            # smaller required precision, since band matmul is in mixed precision
            assert xp.allclose(ref_result, sparse_result, rtol=1e-2, atol=1e-2)


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
            assert (array[(*index,)] == view[i]).all()


@pytest.mark.mpi(min_size=2)
class TestDistribution:
    """Tests for the distribution methods of DSBSparse."""

    @pytest.mark.usefixtures("densify_blocks")
    def test_from_sparray(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None,
    ):
        """Tests distributed creation of DSBSparse matrices from sparrays."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo: sparse.coo_matrix = comm.bcast(coo, root=0)

        dsbsparse = dsbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape, densify_blocks
        )
        assert xp.array_equiv(coo.toarray(), dsbsparse.to_dense())

        stack_section_sizes, __ = get_section_sizes(global_stack_shape[0], comm.size)
        section_size = stack_section_sizes[comm.rank]
        local_stack_shape = (section_size,) + global_stack_shape[1:]
        assert dsbsparse.to_dense().shape == (*local_stack_shape,) + coo.shape

    def test_dtranspose(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the distributed transpose method."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo: sparse.coo_matrix = comm.bcast(coo, root=0)

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

        assert xp.allclose(original_data, dsbsparse._data)

    @pytest.mark.usefixtures("accessed_element")
    def test_getitem_stack(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests distributed access of individual matrix elements."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo: sparse.coo_matrix = comm.bcast(coo, root=0)

        dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsbsparse.to_dense()

        reference = dense[(..., *accessed_element)]
        print(dsbsparse[accessed_element].shape, flush=True) if comm.rank == 0 else None
        print(reference.shape, flush=True) if comm.rank == 0 else None

        assert xp.allclose(reference, dsbsparse[accessed_element])

    @pytest.mark.usefixtures("accessed_element")
    def test_setitem_stack(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests distributed setting of individual matrix elements."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo: sparse.coo_matrix = comm.bcast(coo, root=0)

        dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsbsparse.to_dense()

        dsbsparse[accessed_element] = 42

        dense[(..., *accessed_element)][dense[(..., *accessed_element)].nonzero()] = 42
        assert xp.allclose(dense, dsbsparse.to_dense())

    @pytest.mark.usefixtures("accessed_element")
    def test_getitem_nnz(
        self,
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests distributed access of individual matrix elements."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo: sparse.coo_matrix = comm.bcast(coo, root=0)

        dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsbsparse.to_dense()
        rows, cols = dsbsparse.spy()
        row, col, __ = _unsign_index(*accessed_element, dense.shape[-1])
        ind = xp.where((rows == row) & (cols == col))[0]

        reference = dense[(..., *accessed_element)].flatten()[0]

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
        dsbsparse_type: DSBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests distributed setting of individual matrix elements."""
        coo = _create_coo(block_sizes) if comm.rank == 0 else None
        coo: sparse.coo_matrix = comm.bcast(coo, root=0)

        dsbsparse = dsbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsbsparse.to_dense()
        rows, cols = dsbsparse.spy()
        row, col, __ = _unsign_index(*accessed_element, dense.shape[-1])
        ind = xp.where((rows == row) & (cols == col))[0]

        if len(ind) == 0:
            return

        dense[(..., *accessed_element)][dense[(..., *accessed_element)].nonzero()] = 42

        dsbsparse.dtranspose()

        dsbsparse[accessed_element] = 42

        dsbsparse.dtranspose()

        assert xp.allclose(dense, dsbsparse.to_dense())


def run_standalone_matmul_test(debug_prints: bool = False):
    # create an instance of the test class
    from qttools import NDArray, xp
    from qttools.datastructures import DSBCOO, DSBCSR, DSBSparse

    DSBSPARSE_TYPES = [DSBCOO]

    BLOCK_SIZES = [
        xp.array([2] * 16),
        xp.array([2] * 3 + [4] * 2 + [2] * 3),
        xp.array([2] * 64),
    ]

    DENSIFY_BLOCKS = [
        None,
        [(0, 0), (-1, -1)],
        [(2, 4)],
    ]

    ACCESSED_BLOCKS = [
        (0, 0),
        (-1, -1),
        (2, 4),
        (-9, 3),
    ]

    ACCESSED_ELEMENTS = [
        (0, 0),
        (-1, -1),
        (2, -7),
    ]

    GLOBAL_STACK_SHAPES = [
        (1,),
        (10,),
        (7, 2),
        (9, 2, 4),
    ]

    NUM_INDS = [
        5,
        10,
        20,
    ]

    STACK_INDICES = [
        (5,),
        (slice(1, 4),),
        (Ellipsis,),
    ]

    BLOCK_CHANGE_FACTORS = [
        1.0,
        0.5,
        2.0,
    ]

    for dbsparse_type in DSBSPARSE_TYPES:
        for block_size in BLOCK_SIZES:
            for global_stack_shape in GLOBAL_STACK_SHAPES:
                for densify_blocks in DENSIFY_BLOCKS:
                    TestArithmetic().test_matmul_banded(
                        dbsparse_type,
                        block_size,
                        global_stack_shape,
                        densify_blocks,
                        debug_prints=debug_prints
                    )


if __name__ == "__main__":
    import numpy as np

    # setup print options for numpy and torch
    np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=2)
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=500)
    # Run the standalone tests
    run_standalone_matmul_test(debug_prints=True)
