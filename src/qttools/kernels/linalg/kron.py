# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
from qttools import NDArray, xp


def kron_matmul(m: NDArray, a: NDArray, vect: NDArray) -> NDArray:
    """Performs Kronecker matrix multiplication.

    Computes the product of a Kronecker product of matrices with a vector:
    (m ⊗ a) @ vect.

    Parameters
    ----------
    a : NDArray
        First matrix in the Kronecker product.
    m : NDArray
        Second matrix in the Kronecker product.
    vect : NDArray
        Vector to be multiplied.

    Returns
    -------
    result : NDArray
        Resulting vector from the multiplication.

    """
    vect_3d = vect.reshape(a.shape[0], m.shape[0], -1, order="F")

    # 2. Apply 'a' to the first dimension (axis 0)
    # tensordot(a, phi, axes=1) is like a @ phi along the first axis
    temp = xp.tensordot(a, vect_3d, axes=1)

    # 3. Apply 'm' to the second dimension (axis 1 of temp)
    # We contract axis 1 of m with axis 1 of temp
    res_simple = xp.tensordot(temp, m, axes=(1, 1))

    res_simple = res_simple.transpose(0, 2, 1).reshape(-1, vect.shape[1], order="F")

    return res_simple
