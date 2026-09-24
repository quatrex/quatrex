# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import sparse, xp
from qttools.comm import comm
from qttools.datastructures.dcsx import DCSX
from qttools.datastructures.dsdbsparse import symmetry_ops
from qttools.utils.mpi_utils import get_section_sizes


@pytest.fixture(autouse=True, scope="module", params=[3, 1])
def configure_comm(request):
    """Setup any state specific to the execution of the given module."""
    block_comm_size = request.param

    _default_config = {
        "send_recv": "device_mpi",
    }

    if global_comm.size < block_comm_size:
        pytest.skip(
            f"Skipping test for block comm size {block_comm_size} with global comm size {global_comm.size}."
        )

    # Configure the comm singleton with the parameterized block_comm_size
    comm.configure(
        block_comm_size=block_comm_size,
        block_comm_config=_default_config,
        stack_comm_config=_default_config,
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

    dcsx = DCSX.from_sparray(
        sparray=local_sparray,
        local_stack_shape=local_stack_shape,
        symmetry=symmetry,
    )
    return local_sparray, coo, dcsx


class TestCreation:
    """Tests the creation methods of DCSX."""

    def test_from_sparray(
        self,
        size: int,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests the creation of DCSX matrices from sparse arrays."""

        __, coo, dcsx = _create_coo_dcsx(
            size=size,
            local_stack_shape=global_stack_shape,
            symmetry=symmetry,
        )
        dense_dcsx = dcsx._to_dense()
        if symmetry is not None:
            reference = coo.toarray() + xp.triu(
                symmetry_ops[symmetry](coo.toarray()), k=1
            ).swapaxes(-1, -2)
        else:
            reference = coo.toarray()

        assert xp.array_equiv(reference, dense_dcsx)
        if symmetry is not None:
            assert xp.array_equiv(
                dense_dcsx, symmetry_ops[symmetry](dense_dcsx).swapaxes(-1, -2)
            )


@pytest.mark.mpi(min_size=2)
class TestCreationDist(TestCreation):
    """Tests all tests of TestCreation in distributed setting."""

    pass


class TestConversion:
    """Tests for the conversion methods of DCSX."""

    def test_to_dense(
        self,
        size: int,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can convert a DCSX matrix to dense."""
        __, coo, dcsx = _create_coo_dcsx(
            size=size,
            local_stack_shape=global_stack_shape,
            symmetry=symmetry,
        )
        if symmetry is not None:
            reference = coo.toarray() + xp.triu(
                symmetry_ops[symmetry](coo.toarray()), k=1
            ).swapaxes(-1, -2)
        else:
            reference = coo.toarray()

        reference = xp.broadcast_to(reference, global_stack_shape + (size, size))

        assert xp.allclose(reference, dcsx._to_dense())

    def test_graph_analysis(
        self,
        size: int,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can perform graph analysis on a DCSX matrix."""

        if symmetry is None:
            pytest.skip("Graph analysis is only relevant for symmetric matrices.")

        __, __, dcsx = _create_coo_dcsx(
            size=size,
            local_stack_shape=global_stack_shape,
            symmetry=symmetry,
        )
        # Just check that it runs without errors or deadlocks.
        dcsx._graph_analysis()

    def test_expand_symmetry(
        self,
        size: int,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can expand the symmetry of a DCSX matrix."""

        if symmetry is None:
            pytest.skip("Graph analysis is only relevant for symmetric matrices.")

        __, coo, dcsx = _create_coo_dcsx(
            size=size,
            local_stack_shape=global_stack_shape,
            symmetry=symmetry,
        )
        # Just check that it runs without errors or deadlocks.
        full_dcsx = dcsx.expand_symmetry()

        test = full_dcsx._to_dense()
        reference = coo.toarray() + xp.triu(
            symmetry_ops[symmetry](coo.toarray()), k=1
        ).swapaxes(-1, -2)
        reference = xp.broadcast_to(reference, global_stack_shape + (size, size))

        assert xp.allclose(test, reference)


@pytest.mark.mpi(min_size=2)
class TestConversionDist(TestConversion):
    """Tests all tests of TestConversion in distributed setting."""

    pass


class TestInplace:
    """Tests for the inplace methods of DCSX."""

    def test_add_symmetric(
        self,
        size: int,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can add a DCSX matrix to another DCSX matrix."""
        __, coo, a = _create_coo_dcsx(
            size=size,
            local_stack_shape=global_stack_shape,
            symmetry=symmetry,
        )

        # randomly mask the `coo` matrix to create a new sparse matrix with a subset of the sparsity pattern of `coo`
        rng = xp.random.default_rng(seed=42)
        # choose a random subset of the non-zero entries of `coo` to keep
        mask = rng.choice([False, True], size=coo.nnz)
        coo = sparse.coo_matrix(
            (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape
        )

        local_sparray = coo.tocsr()[
            a.row_offsets[comm.block.rank] : a.row_offsets[comm.block.rank + 1], :
        ]

        b = DCSX.from_sparray(
            sparray=local_sparray,
            local_stack_shape=global_stack_shape,
            symmetry=symmetry,
        )

        a_dense = a._to_dense()
        b_dense = b._to_dense()

        a.add_(b)

        assert xp.allclose(a._to_dense(), a_dense + b_dense)

    def test_add_nonsymmetric(
        self,
        size: int,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can add a symmetric DCSX matrix to a non-symmetric DCSX matrix."""
        __, coo, a = _create_coo_dcsx(
            size=size,
            local_stack_shape=global_stack_shape,
        )

        # randomly mask the `coo` matrix to create a new sparse matrix
        # with a subset of the sparsity pattern of `coo`
        rng = xp.random.default_rng(seed=42)
        # choose a random subset of the non-zero entries of `coo` to keep
        mask = rng.choice([False, True], size=coo.nnz)
        coo = sparse.coo_matrix(
            (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape
        )
        coo = sparse.triu(coo)

        local_sparray = coo.tocsr()[
            a.row_offsets[comm.block.rank] : a.row_offsets[comm.block.rank + 1], :
        ]

        b = DCSX.from_sparray(
            sparray=local_sparray,
            local_stack_shape=global_stack_shape,
            symmetry=symmetry,
        )

        a_dense = a._to_dense()
        b_dense = b._to_dense()

        a.add_(b)

        assert xp.allclose(a._to_dense(), a_dense + b_dense)

    def test_multiply_colwise(
        self,
        size: int,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can multiply a DCSX matrix by a column-wise vector."""
        local_coo, __, a = _create_coo_dcsx(
            size=size,
            local_stack_shape=global_stack_shape,
            symmetry=symmetry,
        )

        rng = xp.random.default_rng(seed=42)
        colwise = rng.uniform(size=a.cols) + 1j * rng.uniform(size=a.cols)

        a.multiply_(colwise)
        coo = local_coo.multiply(colwise).toarray()
        reference = xp.broadcast_to(coo, global_stack_shape + (a.rows, a.cols))

        assert xp.allclose(a.toarray(), reference)

    def test_multiply_rowwise(
        self,
        size: int,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can multiply a DCSX matrix by a row-wise vector."""
        local_coo, __, a = _create_coo_dcsx(
            size=size,
            local_stack_shape=global_stack_shape,
            symmetry=symmetry,
        )

        rng = xp.random.default_rng(seed=42)
        rowwise = rng.uniform(size=a.rows) + 1j * rng.uniform(size=a.rows)
        rowwise = rowwise[:, None]

        a.multiply_(rowwise)
        coo = local_coo.multiply(rowwise).toarray()
        reference = xp.broadcast_to(coo, global_stack_shape + (a.rows, a.cols))

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
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can get a tile from a DCSX matrix."""
        local_coo, __, a = _create_coo_dcsx(
            size=size,
            local_stack_shape=global_stack_shape,
            symmetry=symmetry,
        )

        rng = xp.random.default_rng(seed=42)

        rows = xp.arange(a.rows)
        cols = xp.arange(a.cols)

        mask = rng.choice([False, True], size=a.rows)
        rows = rows[mask]

        mask = rng.choice([False, True], size=a.cols)
        cols = cols[mask]

        test_tile = a.get_tile(rows, cols).toarray()

        ref_tile = local_coo.tocsr()[rows, :][:, cols].toarray()
        ref_tile = xp.broadcast_to(ref_tile, global_stack_shape + ref_tile.shape)

        assert xp.allclose(test_tile, ref_tile)

    def test_get_tile_unsymmetrize(
        self,
        size: int,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can get a tile from a DCSX matrix and unsymmetrize it."""
        if symmetry is None:
            pytest.skip("Unsymmetrization is only relevant for symmetric matrices.")

        __, coo, a = _create_coo_dcsx(
            size=size,
            local_stack_shape=global_stack_shape,
            symmetry=symmetry,
        )

        rng = xp.random.default_rng(seed=42)

        rows = xp.arange(a.rows)
        cols = xp.arange(a.cols)

        mask = rng.choice([False, True], size=a.rows)
        rows = rows[mask]

        mask = rng.choice([False, True], size=a.cols)
        cols = cols[mask]

        test_tile = a.get_tile(rows, cols, unsymmetrize=True).toarray()

        dense = coo.toarray()
        dense += xp.triu(symmetry_ops[symmetry](dense), k=1).swapaxes(-1, -2)

        dense = dense[
            a.row_offsets[comm.block.rank] : a.row_offsets[comm.block.rank + 1], :
        ]

        ref_tile = dense[rows, :][:, cols]
        ref_tile = xp.broadcast_to(ref_tile, global_stack_shape + ref_tile.shape)

        assert xp.allclose(test_tile, ref_tile)

    @pytest.mark.parametrize("unsymmetrize", [True, False])
    def test_get_tile_empty(
        self,
        size: int,
        global_stack_shape: tuple,
        symmetry: str | None,
        unsymmetrize: bool,
    ):
        """Tests that we can get an empty tile from a DCSX matrix."""

        if unsymmetrize and symmetry is None:
            pytest.skip("Unsymmetrization is only relevant for symmetric matrices.")

        __, __, a = _create_coo_dcsx(
            size=size,
            local_stack_shape=global_stack_shape,
            symmetry=symmetry,
        )

        rows = xp.array([], dtype=xp.int64)

        test_tile = a.get_tile(rows, unsymmetrize=unsymmetrize).toarray()

        assert test_tile.shape[-2] == 0
        assert test_tile.shape[-1] == a.cols


@pytest.mark.mpi(min_size=2)
class TestAccessDist(TestAccess):
    """Tests all tests of TestAccess in distributed setting."""

    pass
