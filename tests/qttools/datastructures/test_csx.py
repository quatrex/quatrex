# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import sparse, xp
from qttools.datastructures.csx import CSX
from qttools.datastructures.dsdbsparse import symmetry_ops


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


def _create_coo_csx(
    size: int,
    local_stack_shape: tuple,
    symmetry: str | None = None,
) -> tuple[sparse.coo_matrix, CSX]:
    """Returns a random complex sparse array
    and a CSX matrix with the same sparsity pattern.
    """
    coo = _create_coo(size, symmetry=symmetry)

    csx = CSX.from_sparray(
        sparray=coo,
        local_stack_shape=local_stack_shape,
        symmetry=symmetry,
    )
    return coo, csx


class TestCreation:
    """Tests the creation methods of CSX."""

    def test_from_sparray(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests the creation of CSX matrices from sparse arrays."""

        coo, csx = _create_coo_csx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )
        dense_dcsx = csx._to_dense()
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


class TestConversion:
    """Tests for the conversion methods of CSX."""

    def test_to_dense(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can convert a CSX matrix to dense."""
        coo, csx = _create_coo_csx(
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

        assert xp.allclose(reference, csx._to_dense())

    def test_expand_symmetry(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can expand the symmetry of a CSX matrix."""

        if symmetry is None:
            pytest.skip("Graph analysis is only relevant for symmetric matrices.")

        coo, csx = _create_coo_csx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )
        # Just check that it runs without errors or deadlocks.
        full_dcsx = csx.expand_symmetry()

        test = full_dcsx._to_dense()
        reference = coo.toarray() + xp.triu(
            symmetry_ops[symmetry](coo.toarray()), k=1
        ).swapaxes(-1, -2)
        reference = xp.broadcast_to(reference, local_stack_shape + (size, size))

        assert xp.allclose(test, reference)


class TestInplace:
    """Tests for the inplace methods of CSX."""

    def test_add_symmetric(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can add a CSX matrix to another CSX matrix."""
        coo, a = _create_coo_csx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )

        # randomly mask the `coo` matrix to create a new sparse matrix with a subset of the sparsity pattern of `coo`
        rng = xp.random.default_rng(seed=42)
        # choose a random subset of the non-zero entries of `coo` to keep
        mask = rng.choice([False, True], size=coo.nnz)
        coo = sparse.coo_matrix(
            (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape
        )

        b = CSX.from_sparray(
            sparray=coo,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )

        a_dense = a._to_dense()
        b_dense = b._to_dense()

        a.add_(b)

        assert xp.allclose(a._to_dense(), a_dense + b_dense)

    def test_add_nonsymmetric(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can add a symmetric CSX matrix to a non-symmetric CSX matrix."""
        coo, a = _create_coo_csx(
            size=size,
            local_stack_shape=local_stack_shape,
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

        b = CSX.from_sparray(
            sparray=coo,
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
        """Tests that we can multiply a CSX matrix by a column-wise vector."""
        coo, a = _create_coo_csx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )

        rng = xp.random.default_rng(seed=42)
        colwise = rng.uniform(size=size) + 1j * rng.uniform(size=size)

        a.multiply_(colwise)
        coo = coo.multiply(colwise).toarray()
        reference = xp.broadcast_to(coo, local_stack_shape + (size, size))

        assert xp.allclose(a.toarray(), reference)

    def test_multiply_rowwise(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can multiply a CSX matrix by a row-wise vector."""
        coo, a = _create_coo_csx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )

        rng = xp.random.default_rng(seed=42)
        rowwise = rng.uniform(size=size) + 1j * rng.uniform(size=size)
        rowwise = rowwise[:, None]

        a.multiply_(rowwise)
        coo = coo.multiply(rowwise).toarray()
        reference = xp.broadcast_to(coo, local_stack_shape + (size, size))

        assert xp.allclose(a.toarray(), reference)


class TestAccess:
    """Tests for the access methods of CSX."""

    def test_get_tile(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can get a tile from a CSX matrix."""
        coo, a = _create_coo_csx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )

        rng = xp.random.default_rng(seed=42)

        rows = xp.arange(size)
        cols = xp.arange(size)

        mask = rng.choice([False, True], size=size)
        rows = rows[mask]

        mask = rng.choice([False, True], size=size)
        cols = cols[mask]

        test_tile = a.get_tile(rows, cols).toarray()

        ref_tile = coo.tocsr()[rows, :][:, cols].toarray()
        ref_tile = xp.broadcast_to(ref_tile, local_stack_shape + ref_tile.shape)

        assert xp.allclose(test_tile, ref_tile)

    def test_get_tile_unsymmetrize(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can get a tile from a CSX matrix and unsymmetrize it."""
        if symmetry is None:
            pytest.skip("Unsymmetrization is only relevant for symmetric matrices.")

        coo, a = _create_coo_csx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )

        rng = xp.random.default_rng(seed=42)

        rows = xp.arange(size)
        cols = xp.arange(size)

        mask = rng.choice([False, True], size=size)
        rows = rows[mask]

        mask = rng.choice([False, True], size=size)
        cols = cols[mask]

        test_tile = a.get_tile(rows, cols, unsymmetrize=True).toarray()

        dense = coo.toarray()
        dense += xp.triu(symmetry_ops[symmetry](dense), k=1).swapaxes(-1, -2)
        ref_tile = dense[rows, :][:, cols]
        ref_tile = xp.broadcast_to(ref_tile, local_stack_shape + ref_tile.shape)

        assert xp.allclose(test_tile, ref_tile)

    @pytest.mark.parametrize("unsymmetrize", [True, False])
    def test_get_tile_empty(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
        unsymmetrize: bool,
    ):
        """Tests that we can get an empty tile from a CSX matrix."""

        if unsymmetrize and symmetry is None:
            pytest.skip("Unsymmetrization is only relevant for symmetric matrices.")

        __, a = _create_coo_csx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )

        rows = xp.array([], dtype=xp.int64)

        test_tile = a.get_tile(rows, unsymmetrize=unsymmetrize).toarray()

        assert test_tile.shape[-2] == 0
        assert test_tile.shape[-1] == a.cols


class TestOperations:
    """Tests for the operation on CSX matrices."""

    @pytest.mark.parametrize("rhs_size", [(10,), tuple()])
    def test_matmul(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
        rhs_size: tuple,
    ):
        """Tests that we can get a tile from a CSX matrix."""
        coo, a = _create_coo_csx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )

        rng = xp.random.default_rng(seed=42)
        rhs = rng.uniform(size=(size,) + rhs_size) + 1j * rng.uniform(
            size=(size,) + rhs_size
        )

        out = a @ rhs
        dense = coo.toarray()
        ref = dense @ rhs
        if symmetry is not None:
            ref += xp.triu(symmetry_ops[symmetry](dense), k=1).swapaxes(-1, -2) @ rhs

        ref = xp.broadcast_to(ref, local_stack_shape + ref.shape)

        assert xp.allclose(out, ref)
