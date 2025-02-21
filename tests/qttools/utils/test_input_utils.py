# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from pathlib import Path

import pytest

from qttools import NDArray, _DType, xp
from qttools.utils.input_utils import (
    create_coordinate_grid,
    create_hamiltonian,
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
    "supercell, shift, ref_val",
    [
        ((2, 2, 1), (0, 0, 0), ((0+0j, 0+1j, 1+0j, 1+1j),
                                (0-1j, 0+0j, 1-1j, 1+0j),
                                (-1+0j, -1+1j, 0+0j, 0+1j),
                                (-1-1j, -1+0j, 0-1j, 0+0j))),
        ((2, 2, 1), (1, 1, 0), ((2+2j, 0+0j, 0+0j, 0+0j),
                                (2+1j, 2+2j, 0+0j, 0+0j),
                                (1+2j, 0+0j, 2+2j, 0+0j),
                                (1+1j, 1+2j, 2+1j, 2+2j))),
        ((3, 1, 1), (0, 0, 0), ((0+0j, 1+0j, 2+0j),
                                (-1+0j, 0+0j, 1+0j),
                                (-2+0j, -1+0j, 0+0j))),
        ((3, 1, 1), (1, 0, 0), ((0+0j, 0+0j, 0+0j),
                                (2+0j, 0+0j, 0+0j),
                                (1+0j, 2+0j, 0+0j))),
    ],
)
def test_get_hamiltonian_block(supercell: tuple[int, int, int], shift: tuple[int, int, int], ref_val: tuple[tuple]):
    hr = read_hr_dat(wannier90_hr_path)
    bs = hr.shape[-1]
    block = get_hamiltonian_block(hr, supercell, shift)
    for ind_r in xp.ndindex(supercell):
        br = xp.ravel_multi_index(ind_r, supercell)
        for ind_c in xp.ndindex(supercell):
            bc = xp.ravel_multi_index(ind_c, supercell)
            assert xp.allclose(block[br * bs : (br + 1) * bs, bc * bs : (bc + 1) * bs], ref_val[br][bc])


@pytest.mark.parametrize(
    "coords, supercell, lat_vecs",
    [
        (xp.ones((9, 3)), (2, 2, 2), xp.eye(3)),
    ],
)
def test_create_coordinate_grid(coords: NDArray, supercell: tuple[int, int, int], lat_vecs: NDArray):
    grid = create_coordinate_grid(coords, supercell, lat_vecs)
    assert grid.shape == (xp.prod(supercell) * 10, 3)
    for ind in xp.ndindex(supercell):
        row_ind = xp.ravel_multi_index(ind, supercell) 
        assert xp.allclose(grid[row_ind * 10 : (row_ind + 1) * 10], 1 + sum(ind))


@pytest.mark.parametrize(
    "hr, transport_cells, transport_dir, transport_cell, cutoff, coords, lat_vecs",
    [
        (xp.random.rand(10, 10, 10, 10, 10), (2, 2, 2), (1, 0, 0), (1, 0, 0), 1, xp.random.rand(10, 3), xp.random.rand(3, 3)),
    ],
)
def test_create_hamiltonian(
    hr: NDArray,
    transport_cells: int,
    transport_dir: int | str,
    transport_cell: list[int],
    cutoff: float,
    coords: NDArray,
    lat_vecs: NDArray,
):
    hamiltonians = create_hamiltonian(
        hr,
        transport_cells,
        transport_dir,
        transport_cell,
        cutoff,
        coords,
        lat_vecs,
    )
    assert len(hamiltonians) == 3
    for h in hamiltonians:
        assert h.shape == (xp.prod(transport_cells) * 10, xp.prod(transport_cells) * 10)
