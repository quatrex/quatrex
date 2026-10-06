# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import sparse, xp
from qttools.comm import comm
from qttools.datastructures.csx import symmetry_ops
from qttools.datastructures.dcsx import DCSX
from qttools.utils.mpi_utils import get_section_sizes


@pytest.fixture(autouse=True, scope="module", params=[6, 3, 1])
def configure_comm(request):
    """Setup any state specific to the execution of the given module."""
    block_comm_size = request.param

    if global_comm.size < block_comm_size:
        pytest.skip(
            f"Skipping test for block comm size {block_comm_size} with global comm size {global_comm.size}."
        )

    # Configure the comm singleton with the parameterized block_comm_size
    comm.configure(
        block_comm_size=block_comm_size,
        override=True,
    )


def _create_coo(
    size: int,
    symmetry: str | None = None,
) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""

    rng = xp.random.default_rng(seed=42)
    density = rng.uniform(low=0.1, high=0.3)
    coo = sparse.random(size, size, density=density, format="coo").astype(xp.complex128)
    coo.setdiag(rng.uniform(size=size) + 1j * rng.uniform(size=size))
    coo.data += 1j * rng.uniform(size=coo.nnz)

    # make the sparsity pattern symmetric
    coo_ = coo.copy()
    coo_.data[:] = rng.uniform(size=coo_.nnz) + 1j * rng.uniform(size=coo_.nnz)
    coo = coo + coo_.T
    coo = coo.tocoo()

    if symmetry is not None:
        coo_t = coo.copy()
        coo_t.data[:] = symmetry_ops[symmetry](coo_t.data)
        coo = coo + coo_t.T
        # Keep only the upper triangular part
        coo = sparse.triu(coo, format="coo")
        return coo

    return coo


def _create_coo_dcsx(
    size: int,
    local_stack_shape: tuple,
    symmetry: str | None = None,
    from_indices: bool = False,
) -> tuple[sparse.coo_matrix, sparse.coo_matrix, DCSX]:
    """Returns a random complex sparse array
    and a DCSX matrix with the same sparsity pattern.
    """
    coo = _create_coo(size, symmetry=symmetry) if global_comm.rank == 0 else None
    coo = global_comm.bcast(coo, root=0)

    section_sizes, __ = get_section_sizes(size, comm.block.size)
    row_offsets = np.cumsum([0] + section_sizes)

    local_sparray = coo.tocsr()[
        row_offsets[comm.block.rank] : row_offsets[comm.block.rank + 1], :
    ]

    if from_indices:
        local_sparray = local_sparray.tocoo()
        dcsx = DCSX.from_sparray(
            row_ind=local_sparray.row,
            col_ind=local_sparray.col,
            shape=local_sparray.shape,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
            dtype=local_sparray.dtype,
        )
    else:
        dcsx = DCSX.from_sparray(
            sparray=local_sparray,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )
    dcsx.data = local_sparray.data

    return local_sparray, coo, dcsx


class TestCreation:
    """Tests the creation methods of DCSX."""

    @pytest.mark.parametrize("from_indices", [True, False])
    def test_from_sparray(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
        from_indices: bool,
    ):
        """Tests the creation of DCSX matrices from sparse arrays."""

        __, coo, dcsx = _create_coo_dcsx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
            from_indices=from_indices,
        )
        dense_dcsx = dcsx._to_dense()
        if symmetry is not None:
            reference = coo.toarray() + xp.triu(
                symmetry_ops[symmetry](coo.toarray()), k=1
            ).swapaxes(-1, -2)
        else:
            reference = coo.toarray()

        assert xp.array_equiv(reference, dense_dcsx)


@pytest.mark.mpi(min_size=2)
class TestCreationDist(TestCreation):
    """Tests all tests of TestCreation in distributed setting."""

    pass


class TestConversion:
    """Tests for the conversion methods of DCSX."""

    def test_to_dense(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can convert a DCSX matrix to dense."""
        __, coo, dcsx = _create_coo_dcsx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )
        if symmetry is not None:
            reference = coo.toarray() + xp.triu(
                symmetry_ops[symmetry](coo.toarray()), k=1
            ).swapaxes(-1, -2)
        else:
            reference = coo.toarray()

        reference = xp.broadcast_to(reference, local_stack_shape + (size, size))

        assert xp.allclose(reference, dcsx._to_dense())

    def test_graph_analysis(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can perform graph analysis on a DCSX matrix."""

        if symmetry is None:
            pytest.skip("Graph analysis is only relevant for symmetric matrices.")

        __, __, dcsx = _create_coo_dcsx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )
        # Just check that it runs without errors or deadlocks.
        dcsx._graph_analysis()

    def test_expand_sparsity(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can expand the sparsity of a DCSX matrix."""

        if symmetry is None:
            pytest.skip("Graph analysis is only relevant for symmetric matrices.")

        local_coo, coo, dcsx = _create_coo_dcsx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )
        # Just check that it runs without errors or deadlocks.
        row_ind, col_ind, __ = dcsx.expand_sparsity()

        test = sparse.coo_matrix(
            (xp.ones_like(row_ind), (row_ind, col_ind)), shape=local_coo.shape
        ).toarray()

        coo.data[:] = 1.0
        reference = coo + coo.T
        reference.data[:] = 1.0
        reference = reference.toarray()

        reference = reference[
            dcsx.row_offsets[comm.block.rank] : dcsx.row_offsets[comm.block.rank + 1], :
        ]

        assert xp.allclose(test, reference)

    def test_expand_symmetry(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can expand the symmetry of a DCSX matrix."""

        if symmetry is None or symmetry == "upper-triangular":
            pytest.skip("Graph analysis is only relevant for symmetric matrices.")

        __, coo, dcsx = _create_coo_dcsx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )
        # Just check that it runs without errors or deadlocks.
        full_dcsx = dcsx.expand_symmetry()

        test = full_dcsx._to_dense()
        reference = coo.toarray() + xp.triu(
            symmetry_ops[symmetry](coo.toarray()), k=1
        ).swapaxes(-1, -2)
        reference = xp.broadcast_to(reference, local_stack_shape + (size, size))

        assert xp.allclose(test, reference)


@pytest.mark.mpi(min_size=2)
class TestConversionDist(TestConversion):
    """Tests all tests of TestConversion in distributed setting."""

    pass


class TestInplace:
    """Tests for the inplace methods of DCSX."""

    def test_add_(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can add a DCSX matrix to another DCSX matrix."""
        __, coo, a = _create_coo_dcsx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )

        # randomly mask the `coo` matrix to create a new sparse matrix with a subset of the sparsity pattern of `coo`
        rng = xp.random.default_rng(seed=42)
        # choose a random subset of the non-zero entries of `coo` to keep
        mask = rng.random(coo.nnz) > 0.5
        coo = sparse.coo_matrix(
            (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape
        )

        local_sparray = coo.tocsr()[
            a.row_offsets[comm.block.rank] : a.row_offsets[comm.block.rank + 1], :
        ]

        b = DCSX.from_sparray(
            sparray=local_sparray,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )

        a_dense = a._to_dense()
        b_dense = b._to_dense()

        a.add_(b)

        assert xp.allclose(a._to_dense(), a_dense + b_dense)

    def test_multiply_colwise(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can multiply a DCSX matrix by a column-wise vector."""
        local_coo, __, a = _create_coo_dcsx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )

        rng = xp.random.default_rng(seed=42)
        colwise = rng.uniform(size=a.cols) + 1j * rng.uniform(size=a.cols)

        a.multiply_(colwise)
        coo = local_coo.multiply(colwise).toarray()
        reference = xp.broadcast_to(coo, local_stack_shape + (a.rows, a.cols))

        assert xp.allclose(a.toarray(), reference)

    def test_multiply_rowwise(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can multiply a DCSX matrix by a row-wise vector."""
        local_coo, __, a = _create_coo_dcsx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )

        rng = xp.random.default_rng(seed=42)
        rowwise = rng.uniform(size=a.rows) + 1j * rng.uniform(size=a.rows)
        rowwise = rowwise[:, None]

        a.multiply_(rowwise)
        coo = local_coo.multiply(rowwise).toarray()
        reference = xp.broadcast_to(coo, local_stack_shape + (a.rows, a.cols))

        assert xp.allclose(a.toarray(), reference)


@pytest.mark.mpi(min_size=2)
class TestInplaceDist(TestInplace):
    """Tests all tests of TestInplace in distributed setting."""

    pass


class TestAccess:
    """Tests for the access methods of DCSX."""

    def test_get_tile(
        self,
        size: int,
        local_stack_shape: tuple,
    ):
        """Tests that we can get a tile from a DCSX matrix."""
        local_coo, __, a = _create_coo_dcsx(
            size=size,
            local_stack_shape=local_stack_shape,
        )

        rng = xp.random.default_rng(seed=42)

        rows = xp.arange(a.rows)
        cols = xp.arange(a.cols)

        mask = rng.random(a.rows) > 0.5
        rows = rows[mask]

        mask = rng.random(a.cols) > 0.5
        cols = cols[mask]

        test_tile = a.get_tile(rows, cols).toarray()

        ref_tile = local_coo.tocsr()[rows, :][:, cols].toarray()
        ref_tile = xp.broadcast_to(ref_tile, local_stack_shape + ref_tile.shape)

        assert xp.allclose(test_tile, ref_tile)

    def test_get_tile_empty(
        self,
        size: int,
        local_stack_shape: tuple,
    ):
        """Tests that we can get an empty tile from a DCSX matrix."""

        __, __, a = _create_coo_dcsx(
            size=size,
            local_stack_shape=local_stack_shape,
        )

        rows = xp.array([], dtype=xp.int64)

        test_tile = a.get_tile(rows).toarray()

        assert test_tile.shape[-2] == 0
        assert test_tile.shape[-1] == a.cols


@pytest.mark.mpi(min_size=2)
class TestAccessDist(TestAccess):
    """Tests all tests of TestAccess in distributed setting."""

    pass


class TestOperations:
    """Tests for the operation on DCSX matrices."""

    @pytest.mark.parametrize("rhs_size", [(10,), tuple()])
    def test_local_matmul(
        self,
        size: int,
        local_stack_shape: tuple,
        rhs_size: tuple,
    ):
        """Tests that we can get a tile from a DCSX matrix."""
        __, coo, a = _create_coo_dcsx(
            size=size,
            local_stack_shape=local_stack_shape,
        )

        rng = xp.random.default_rng(seed=42)
        rhs = rng.uniform(size=(size,) + rhs_size) + 1j * rng.uniform(
            size=(size,) + rhs_size
        )

        out = a @ rhs

        dense = coo.toarray()

        dense = dense[
            a.row_offsets[comm.block.rank] : a.row_offsets[comm.block.rank + 1], :
        ]
        ref = dense @ rhs
        ref = xp.broadcast_to(ref, local_stack_shape + ref.shape)

        assert xp.allclose(out, ref)


@pytest.mark.mpi(min_size=2)
class TestOperationsDist(TestOperations):
    """Tests all tests of TestOperations in distributed setting."""

    pass
