# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from pathlib import Path

import pytest

from qttools import NDArray, _DType, xp
from qttools.utils.input_utils import (
    create_coordinate_grid,
    create_hamiltonian,
    cutoff_hr,
    get_hamiltonian_block,
    read_hr_dat,
)

wannier90_hr_path = Path(__file__).parent / "data" / "wannier90_hr.dat"
R_ref = xp.loadtxt(Path(__file__).parent / "data" / "R_ref.txt", dtype=int)


@pytest.mark.parametrize(
    "return_all, dtype, read_fast",
    [
        (False, xp.complex128, False),
        (True, xp.complex128, False),
        (True, xp.complex128, True),
    ],
)
def test_read_hr_dat(return_all: bool, dtype: _DType, read_fast: bool):
    if return_all:
        hr, R = read_hr_dat(wannier90_hr_path, return_all, dtype, read_fast)
    else:
        hr = read_hr_dat(wannier90_hr_path, return_all, dtype, read_fast)
    for i, r in enumerate(R_ref):
        if return_all:
            assert xp.allclose(R[i], r)
        # NOTE: This assumes specific data in the file, should be generalized
        assert xp.allclose(hr[*r], r[0] + 1j * r[1])


@pytest.mark.parametrize(
    "value_cutoff, R_cutoff, remove_zeros",
    [
        (0.5, None, False),
        (None, 1, False),
        (0.5, 1, False),
        (0.5, None, True),
        (None, 1, True),
        (0.5, 1, True),
    ],
)
def test_cutoff_hr(value_cutoff: float, R_cutoff: int, remove_zeros: bool):
    hr = read_hr_dat(wannier90_hr_path)
    hr_cutoff = cutoff_hr(hr, value_cutoff, R_cutoff, remove_zeros)
    if R_cutoff is None:
        R_cutoff = xp.array([s // 2 if s > 1 else 1 for s in hr.shape[:3]])
    if value_cutoff is None:
        value_cutoff = xp.inf
    if remove_zeros and value_cutoff is None:
        # NOTE: if value_cutoff is not None, we don't know the shape
        if isinstance(R_cutoff, int):
            assert (
                hr_cutoff.shape
                == tuple([2 * R_cutoff + 1 if hrs > 1 else 1 for hrs in hr.shape[:3]])
                + hr.shape[3:]
            )
        else:
            assert (
                hr_cutoff.shape
                == tuple(
                    [
                        2 * r + 1 if hrs > 1 else 1
                        for r, hrs in zip(R_cutoff, hr.shape[:3])
                    ]
                )
                + hr.shape[3:]
            )
    elif value_cutoff is None:
        assert hr_cutoff.shape == hr.shape
    for r in R_ref:
        if (xp.abs(r) <= R_cutoff).all():
            hr_ref = hr[*r][xp.abs(hr[*r]) <= value_cutoff]
            try:
                assert xp.allclose(hr_cutoff[*r], hr_ref)
            # Some R values can have been removed because of the value_cutoff,
            # but then the corresponding hr values should be zero
            except IndexError:
                if remove_zeros:
                    assert (hr_ref == 0).all()
                else:
                    assert False
        else:
            if not remove_zeros:
                assert (hr_cutoff[*r] == 0).all()


@pytest.mark.parametrize(
    "supercell, shift, ref_val",
    [
        (
            (2, 2, 1),
            (0, 0, 0),
            (
                (0 + 0j, 0 + 1j, 1 + 0j, 1 + 1j),
                (0 - 1j, 0 + 0j, 1 - 1j, 1 + 0j),
                (-1 + 0j, -1 + 1j, 0 + 0j, 0 + 1j),
                (-1 - 1j, -1 + 0j, 0 - 1j, 0 + 0j),
            ),
        ),
        (
            (2, 2, 1),
            (1, 1, 0),
            (
                (2 + 2j, 0 + 0j, 0 + 0j, 0 + 0j),
                (2 + 1j, 2 + 2j, 0 + 0j, 0 + 0j),
                (1 + 2j, 0 + 0j, 2 + 2j, 0 + 0j),
                (1 + 1j, 1 + 2j, 2 + 1j, 2 + 2j),
            ),
        ),
        (
            (3, 1, 1),
            (0, 0, 0),
            (
                (0 + 0j, 1 + 0j, 2 + 0j),
                (-1 + 0j, 0 + 0j, 1 + 0j),
                (-2 + 0j, -1 + 0j, 0 + 0j),
            ),
        ),
        (
            (3, 1, 1),
            (1, 0, 0),
            (
                (0 + 0j, 0 + 0j, 0 + 0j),
                (2 + 0j, 0 + 0j, 0 + 0j),
                (1 + 0j, 2 + 0j, 0 + 0j),
            ),
        ),
    ],
)
def test_get_hamiltonian_block(
    supercell: tuple[int, int, int], shift: tuple[int, int, int], ref_val: tuple[tuple]
):
    hr = read_hr_dat(wannier90_hr_path)
    bs = hr.shape[-1]
    block = get_hamiltonian_block(hr, supercell, shift)
    for ind_r in xp.ndindex(supercell):
        br = int(xp.ravel_multi_index(xp.asarray(ind_r), supercell))
        for ind_c in xp.ndindex(supercell):
            bc = int(xp.ravel_multi_index(xp.asarray(ind_c), supercell))
            assert xp.allclose(
                block[br * bs : (br + 1) * bs, bc * bs : (bc + 1) * bs], ref_val[br][bc]
            )


@pytest.mark.parametrize(
    "coords, supercell, lat_vecs",
    [
        (xp.ones((10, 3)), (2, 2, 2), xp.eye(3)),
    ],
)
def test_create_coordinate_grid(
    coords: NDArray, supercell: tuple[int, int, int], lat_vecs: NDArray
):
    grid = create_coordinate_grid(coords, supercell, lat_vecs)
    assert grid.shape == (xp.prod(xp.asarray(supercell)) * 10, 3)
    for ind in xp.ndindex(supercell):
        row_ind = xp.ravel_multi_index(xp.asarray(ind), supercell)
        assert xp.allclose(grid[row_ind * 10 : (row_ind + 1) * 10], 1 + xp.array(ind))


@pytest.mark.parametrize(
    "hr, num_transport_cells, transport_dir, transport_cell, block_start, block_end, return_sparse, cutoff, coords, lat_vecs",
    [
        (
            xp.ones((3, 3, 3, 5, 5)),
            10,
            "x",
            (2, 1, 1),
            None,
            None,
            False,
            2,
            xp.zeros((5, 3)),
            xp.eye(3),
        ),
        (
            xp.ones((3, 3, 3, 5, 5)),
            10,
            "x",
            (2, 1, 1),
            None,
            None,
            True,
            2,
            xp.zeros((5, 3)),
            xp.eye(3),
        ),
        (
            xp.ones((3, 3, 3, 5, 5)),
            10,
            "x",
            (2, 1, 1),
            0,
            2,
            False,
            2,
            xp.zeros((5, 3)),
            xp.eye(3),
        ),
        (
            xp.ones((3, 3, 3, 5, 5)),
            10,
            "x",
            (2, 1, 1),
            0,
            2,
            True,
            2,
            xp.zeros((5, 3)),
            xp.eye(3),
        ),
    ],
    ids=[
        "dense_no-block-inds",
        "sparse_no-block-inds",
        "dense_with-block-inds",
        "sparse_with-block-inds",
    ],
)
def test_create_hamiltonian(
    hr: NDArray,
    num_transport_cells: int,
    transport_dir: int | str,
    transport_cell: list[int],
    block_start: int | None,
    block_end: int | None,
    return_sparse: bool,
    cutoff: float,
    coords: NDArray,
    lat_vecs: NDArray,
):
    hamiltonians = create_hamiltonian(
        hr,
        num_transport_cells,
        transport_dir=transport_dir,
        transport_cell=transport_cell,
        block_start=block_start,
        block_end=block_end,
        return_sparse=return_sparse,
        cutoff=cutoff,
        coords=coords,
        lattice_vectors=lat_vecs,
    )
    block_start = block_start or 0
    block_end = block_end or num_transport_cells
    # Number of unit cells in a transport cell
    transport_cell_size = int(xp.prod(xp.array(transport_cell)))
    if return_sparse:
        assert len(hamiltonians) == 2
        sparse_hamiltonian, block_sizes = hamiltonians
        # Assumes the block sizes are the same for all blocks
        assert sparse_hamiltonian.shape[0] == block_sizes[0] * num_transport_cells
        assert sparse_hamiltonian.shape[1] == block_sizes[0] * num_transport_cells
        assert len(block_sizes) == block_end - block_start
        num_wann_per_supercell = int(block_sizes[0])
        num_wann = num_wann_per_supercell // transport_cell_size
        # Make sure the sparse hamiltonian is csr to be able to slice it
        if sparse_hamiltonian.format != "csr":
            sparse_hamiltonian = sparse_hamiltonian.tocsr()
    else:
        assert len(hamiltonians) == 3

        h_diag, h_upper, h_lower = hamiltonians
        num_wann = int(hr.shape[-1])
        num_wann_per_supercell = num_wann * transport_cell_size
        assert h_diag.shape == (
            (block_end - block_start) * num_wann_per_supercell,
            num_wann_per_supercell,
        )
        assert h_upper.shape == (
            (block_end - block_start - (1 if block_end == num_transport_cells else 0))
            * num_wann_per_supercell,
            num_wann_per_supercell,
        )
        assert h_lower.shape == (
            (block_end - block_start - (1 if block_end == num_transport_cells else 0))
            * num_wann_per_supercell,
            num_wann_per_supercell,
        )

    for i in range(block_start, block_end):
        block_slice = slice(
            i * num_wann_per_supercell, (i + 1) * num_wann_per_supercell
        )
        # Assume cut-off is large enough to include all interaction in diagonal blocks
        if return_sparse:
            assert xp.allclose(
                sparse_hamiltonian[block_slice, block_slice].todense(), 1
            )
        else:
            assert xp.allclose(h_diag[block_slice], 1)
        if i < num_transport_cells - 1:
            if return_sparse:
                off_diagonal_block_slice = slice(
                    (i + 1) * num_wann_per_supercell, (i + 2) * num_wann_per_supercell
                )
                h_upper = sparse_hamiltonian[
                    block_slice, off_diagonal_block_slice
                ].todense()
                h_lower = sparse_hamiltonian[
                    off_diagonal_block_slice, block_slice
                ].todense()
            for r in range(transport_cell_size):
                row_unit_slice = slice(r * num_wann, (r + 1) * num_wann)
                for c in range(transport_cell_size):
                    col_unit_slice = slice(c * num_wann, (c + 1) * num_wann)
                    if r <= c:
                        assert xp.allclose(
                            h_upper[block_slice][row_unit_slice, col_unit_slice], 0
                        )
                        assert xp.allclose(
                            h_lower[block_slice][col_unit_slice, row_unit_slice], 0
                        )
                    else:
                        assert xp.allclose(
                            h_upper[block_slice][row_unit_slice, col_unit_slice], 1
                        )
                        assert xp.allclose(
                            h_lower[block_slice][col_unit_slice, row_unit_slice], 1
                        )
