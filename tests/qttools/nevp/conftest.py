# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import itertools

import pytest

from qttools import NDArray, xp
from qttools.nevp import NEVP, Beyn

BLOCK_SIZE = [
    27,
]
BATCH_SIZE = [
    1,
    3,
]

REDUCE = [
    # False, # Tuening parameters for beyn tests is hard to solve for both reduce settings.
    True,
]

BLOCK_SETTINGS = list(itertools.product(BLOCK_SIZE, BATCH_SIZE, REDUCE))

# NOTE: The matrices we generate generally have their eigenvalues close
# to the unit circle. We set the outer radius to 1.2 and the inner
# radius to 0.9. The subspace dimension is chosen sufficiently large to
# capture all the eigenvalues. The number of quadrature points is set to
# a very large number to ensure that the non-spurious eigenvalues get
# approximated very accurately.
CONTOUR_BATCH_SIZES = [2]

SUBSPACE_NEVP_SOLVERS = []

for CONTOUR_BATCH_SIZE in CONTOUR_BATCH_SIZES:
    for use_qr in [
        False,
        True,
    ]:
        for use_pinned_memory in [
            False,
            True,
        ]:
            SUBSPACE_NEVP_SOLVERS.append(
                pytest.param(
                    Beyn(
                        r_o=1.2,
                        r_i=0.9,
                        m_0=20,
                        num_quad_points=50,
                        use_qr=use_qr,
                        contour_batch_size=CONTOUR_BATCH_SIZE,
                        use_pinned_memory=use_pinned_memory,
                    ),
                    id=f"Beyn with QR batch size {CONTOUR_BATCH_SIZE} and use_qr {use_qr}",
                )
            )


# TODO: It's a good idea to generalize the tests with input data
# constructed from a hand-picked eigenvalues and eigenvectors. This
# allows us to choose when the subspace solvers should find the chosen
# eigenvalues and eigenvectors.
@pytest.fixture(params=BLOCK_SETTINGS, autouse=True)
def a_xx(request: pytest.FixtureRequest) -> NDArray:
    """Returns some random complex boundary blocks."""
    size, batch_size, reduce = request.param
    a_xx = [
        xp.random.rand(batch_size, size, size)
        + 1j * xp.random.rand(batch_size, size, size)
        for _ in range(3)
    ]

    if reduce:
        # Introduce some zero columns in a_ji and a_ij
        a_xx[0][:, :, : size // 2] = 0
        a_xx[2][:, :, size // 2 :] = 0

    return tuple([xp.squeeze(a) for a in a_xx])


@pytest.fixture(params=SUBSPACE_NEVP_SOLVERS)
def subspace_nevp(request: pytest.FixtureRequest) -> NEVP:
    """Returns a NEVP solver."""
    return request.param
