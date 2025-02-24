import numpy as xp
import pytest
from qttools.datastructures.dsbcoo import DSBCOO
from scipy import sparse

from quatrex.coulomb_screening.utils import assemble_boundary_blocks


def _create_coo(sizes) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    rng = xp.random.default_rng()
    density = rng.uniform(low=0.1, high=0.3)
    coo = sparse.random(size, size, density=density, format="coo").astype(xp.complex128)
    coo.data += 1j * rng.uniform(size=coo.nnz)
    return coo


@pytest.mark.parametrize("block_sizes", [6 * [6], 4 * [10]])
@pytest.mark.parametrize("nbc", [1, 2, 3])
def test_assemble_boundary_blocks(block_sizes, nbc):
    """Test the function `assemble_boundary_blocks`."""
    block_sizes = xp.array(block_sizes)
    num_blocks = block_sizes.size
    coo = _create_coo(block_sizes)
    global_stack_shape = (3,)
    dsbcoo = DSBCOO.from_sparray(coo, block_sizes, global_stack_shape)
    diag_left = xp.zeros(
        global_stack_shape + (nbc * block_sizes[0], nbc * block_sizes[0]),
        dtype=xp.complex128,
    )
    right = xp.zeros_like(diag_left)
    below = xp.zeros_like(diag_left)
    diag_right = xp.zeros(
        global_stack_shape + (nbc * block_sizes[-1], nbc * block_sizes[-1]),
        dtype=xp.complex128,
    )
    above = xp.zeros_like(diag_right)
    left = xp.zeros_like(diag_right)
    assemble_boundary_blocks(
        diag_left, right, below, diag_right, above, left, dsbcoo, nbc
    )
    for i in range(nbc):
        for j in range(nbc):
            assert xp.allclose(
                dsbcoo.blocks[max(i - j, 0), max(j - i, 0)],
                diag_left[
                    ...,
                    i * block_sizes[0] : (i + 1) * block_sizes[0],
                    j * block_sizes[0] : (j + 1) * block_sizes[0],
                ],
            )
            assert xp.allclose(
                dsbcoo.blocks[min(i - j, 0) - 1, min(j - i, 0) - 1],
                diag_right[
                    ...,
                    i * block_sizes[-1] : (i + 1) * block_sizes[-1],
                    j * block_sizes[-1] : (j + 1) * block_sizes[-1],
                ],
            )
            if j - i + nbc < num_blocks:
                assert xp.allclose(
                    dsbcoo.blocks[0, j - i + nbc],
                    right[
                        ...,
                        i * block_sizes[0] : (i + 1) * block_sizes[0],
                        j * block_sizes[0] : (j + 1) * block_sizes[0],
                    ],
                )
                assert xp.allclose(
                    dsbcoo.blocks[-(j - i + nbc) - 1, -1],
                    above[
                        ...,
                        i * block_sizes[-1] : (i + 1) * block_sizes[-1],
                        j * block_sizes[-1] : (j + 1) * block_sizes[-1],
                    ],
                )
            if i - j + nbc < num_blocks:
                assert xp.allclose(
                    dsbcoo.blocks[i - j + nbc, 0],
                    below[
                        ...,
                        i * block_sizes[0] : (i + 1) * block_sizes[0],
                        j * block_sizes[0] : (j + 1) * block_sizes[0],
                    ],
                )
                assert xp.allclose(
                    dsbcoo.blocks[-1, -(i - j + nbc) - 1],
                    left[
                        ...,
                        i * block_sizes[-1] : (i + 1) * block_sizes[-1],
                        j * block_sizes[-1] : (j + 1) * block_sizes[-1],
                    ],
                )
