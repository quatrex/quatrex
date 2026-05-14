# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.


from qttools import NDArray, xp
from qttools.datastructures.dsdbsparse import _block_view


def check_circulant(a: NDArray, sections: int) -> bool:
    """Check if a matrix is block circulant with the given number of sections.

    Parameters
    ----------
    a : NDArray
        The matrix to check.
    sections : int
        The number of sections in the block circulant structure.

    Returns
    -------
    bool
        True if a is block circulant with the given number of sections, False otherwise.

    """
    if a.shape[-1] % sections != 0:
        raise ValueError("The last dimension of a must be divisible by sections.")

    if a.shape[-2] != a.shape[-1]:
        raise ValueError(
            "The second to last dimension of a must be equal to the last dimension of a."
        )

    block_size = a.shape[-1] // sections
    # Take the first block-row (top n rows)
    block_layer = a[..., :block_size, :]
    blocks = xp.split(block_layer, sections, axis=-1)

    for i in range(sections):
        shifted_blocks = blocks[-i:] + blocks[:-i]
        if not xp.allclose(
            a[..., i * block_size : (i + 1) * block_size, :],
            xp.concatenate(shifted_blocks, axis=-1),
        ):
            return False

    return True


def _get_dft_matrix(n: int) -> NDArray:
    """Get the discrete Fourier transform (DFT) matrix of size n x n.

    Parameters
    ----------
    n : int
        The size of the DFT matrix.

    Returns
    -------
    NDArray
        The DFT matrix of size n x n.

    """
    k = xp.arange(n)
    j = xp.arange(n)
    w = xp.exp(-2j * xp.pi * k[:, None] * j / n) / xp.sqrt(n)
    return w


def _get_idft_matrix(n: int) -> NDArray:
    """Get the inverse discrete Fourier transform (IDFT) matrix of size n x n.

    Parameters
    ----------
    n : int
        The size of the IDFT matrix.

    Returns
    -------
    NDArray
        The IDFT matrix of size n x n.

    """
    k = xp.arange(n)
    j = xp.arange(n)
    w = xp.exp(2j * xp.pi * k[:, None] * j / n) / xp.sqrt(n)
    return w


def _transform_matrix(
    a: NDArray,
    sections_x: int = 1,
    sections_y: int = 1,
) -> NDArray:
    r"""Transform a matrix to block diagonal form by exploiting the
    circulant structure of the matrix. This transformation is based on the
    discrete Fourier transform (DFT) and can be applied to systems with
    periodicity in transverse direction. It is assumed that the matrix is
    already sorted according to the transverse direction.
    
    \[
        \mathbf{A} = \begin{bmatrix}
            a & b & c \\
            c & a & b \\
            b & c & a
        \end{bmatrix}
    \]
    which is circulant with `sections_x=3` sections, and 
    each block is again block circulant with `sections_y` sections.

    which can be decomposed with the 2D DFT matrix $\mathbf{F}$ as
    \[
        \mathbf{A} = \mathbf{F}^{-1} \mathbf{B} \mathbf{F}
    \]
    where 
    \[
        \mathbf{F} = \mathbf{F}_x \otimes \mathbf{F}_y \otimes \mathbf{I}
    \]

    This uses explicitly the DFT matrix, which is not the most efficient way for
    many sections, but it is assumed that the number of sections is small.
    The DFT matrices are explicitly used to simplify when transforming non-matrix
    quantities for the boundary conditions.

    Parameters
    ----------
    a : NDArray
        The matrix to transform. The last two dimensions are assumed to be
        square and the last dimension is assumed to be divisible by sections_x
        and sections_y.
    sections_x : int, optional 
        The number of sections in the x direction, by default 1.
    sections_y : int, optional
        The number of sections in the y direction, by default 1.

    Returns
    -------
    NDArray
        The transformed matrix in block diagonal form.

    """

    if a.ndim == 2:
        a = a[None, ...]

    if a.shape[-1] % sections_x != 0:
        raise ValueError("The last dimension of a must be divisible by sections_x.")

    block_size_x = a.shape[-1] // sections_x

    if block_size_x % sections_y != 0:
        raise ValueError("The last dimension of a must be divisible by sections_y.")

    block_size_y = block_size_x // sections_y

    # view along x direction
    a = _block_view(a[..., :block_size_x, :], axis=-1, num_blocks=sections_x)
    # view along y direction
    a = _block_view(a[..., :block_size_y, :], axis=-1, num_blocks=sections_y)

    # a now stores the first block-layer of the system, which has the shape
    # (sections_y, sections_x, batch_size, block_size_y, block_size_x)

    dft_x = _get_dft_matrix(sections_x)
    dft_y = _get_dft_matrix(sections_y)

    # Transform along the y-sections
    a = xp.einsum("ij, jklmn -> iklmn", dft_y, a)

    # Transform along the x-sections
    a = xp.einsum("ij, kjlmn -> kilmn", dft_x, a)

    return a


def _detransform_matrix(
    a: NDArray,
    sections_x: int = 1,
    sections_y: int = 1,
) -> NDArray:
    """Inverse transformation of the block diagonal form to the original matrix
    form. This is the inverse of the _transform_matrix function and uses the
    inverse DFT matrices.

    Parameters
    ----------
    a : NDArray
        The matrix to detransform.
    sections_x : int, optional
        The number of sections in the x direction, by default 1.
    sections_y : int, optional
        The number of sections in the y direction, by default 1.

    Returns
    -------
    NDArray
        The original matrix before transformation.

    """

    if a.ndim != 5:
        raise ValueError(
            "The input matrix must have 5 dimensions after transformation."
        )

    if a.shape[0] != sections_y:
        raise ValueError("The first dimension of a must be equal to sections_y.")
    if a.shape[1] != sections_x:
        raise ValueError("The second dimension of a must be equal to sections_x.")

    block_size_y = a.shape[-1]
    block_size_x = block_size_y * a.shape[0]
    batch_size = a.shape[2]

    idft_x = _get_idft_matrix(sections_x)
    idft_y = _get_idft_matrix(sections_y)

    # Transform along the y-sections
    a = xp.einsum("ij, jklmn -> iklmn", idft_y, a)

    # Transform along the x-sections
    a = xp.einsum("ij, kjlmn -> kilmn", idft_x, a)

    # expand first in the y direction
    # the temporary has the shape (sections_x, batch_size, block_size_x, block_size_x)
    tmp = xp.zeros((sections_x, batch_size, block_size_x, block_size_x), dtype=a.dtype)
    a = a.transpose(1, 2, 3, 0, 4).reshape(
        sections_x, batch_size, block_size_y, block_size_x
    )

    for i in range(0, block_size_x, block_size_y):
        tmp[..., i : i + block_size_y, :] = a
        # roll the layer
        a = xp.roll(a, block_size_y, axis=-1)

    # expand first in the x direction
    tmp = tmp.transpose(1, 2, 0, 3).reshape(
        batch_size, block_size_x, block_size_x * sections_x
    )
    out = xp.zeros(
        (batch_size, block_size_x * sections_x, block_size_x * sections_x),
        dtype=a.dtype,
    )

    for i in range(0, block_size_x * sections_x, block_size_x):
        out[..., i : i + block_size_x, :] = tmp
        # roll the layer
        tmp = xp.roll(tmp, block_size_x, axis=-1)

    return out


def transform_system(
    a_xx: list[NDArray],
    phase: complex = 1,
    sections_x: int = 1,
    sections_y: int = 1,
):
    r"""Transform the boundary system to block diagonal form by exploiting the
    circulant structure of the system. This transformation is based on the
    discrete Fourier transform (DFT) and can be applied to systems with
    periodicity in transverse direction. It is assumed that the system is
    already sorted according to the transverse direction.
    
    A phase needs to be provided for non-gamma point caculation since the the
    system is not circulant, but rather $\phi$-circulant i.e. has the form of:

    \[
        \mathbf{A} = \begin{bmatrix}
            a & b & c \\
            \phi c & a & b \\
            \phi b & \phi c & a
        \end{bmatrix}
    \]

    which can be decomposed into a circulant matrix by 
    \[
        \mathbf{A} = \mathbf{D} \mathbf{B} \mathbf{D}^{-1}
    \]

    where 
    \[
        \mathbf{D} = \begin{bmatrix}
            1 & 0 & 0 \\
            0 & a & 0 \\
            0 & 0 & c
        \end{bmatrix}
    \]


    """
    ...
