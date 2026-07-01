# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.


from qttools import NDArray, xp
from qttools.datastructures.dsdbsparse import _block_view


def _make_1D_block_circulant(
    a: NDArray,
    sections: int,
) -> NDArray:
    """Helper function to transform a matrix into a block circulant matrix with
    the given number of sections."""

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

    matrix = xp.zeros_like(a)
    for i in range(sections):
        shifted_blocks = blocks[-i:] + blocks[:-i]
        matrix[..., i * block_size : (i + 1) * block_size, :] = xp.concatenate(
            shifted_blocks, axis=-1
        )

    return matrix


def _make_2D_block_circulant(
    a: NDArray,
    sections_x: int,
    sections_y: int,
) -> NDArray:
    """Helper function to transform a matrix into a block circulant matrix with
    the given number of sections."""

    if a.shape[-1] % sections_x != 0:
        raise ValueError("The last dimension of a must be divisible by sections_x.")
    if a.shape[-1] % sections_y != 0:
        raise ValueError("The last dimension of a must be divisible by sections_y.")
    if a.shape[-1] % (sections_x * sections_y) != 0:
        raise ValueError(
            "The last dimension of a must be divisible by the section product."
        )

    if a.shape[-2] != a.shape[-1]:
        raise ValueError(
            "The second to last dimension of a must be equal to the last dimension of a."
        )

    block_size_x = a.shape[-1] // sections_x

    # make first circulant in the y direction
    for i in range(0, a.shape[-1], block_size_x):
        a[..., :block_size_x, i : i + block_size_x] = _make_1D_block_circulant(
            a[..., :block_size_x, i : i + block_size_x],
            sections=sections_y,
        )

    return _make_1D_block_circulant(a, sections=sections_x)


def _make_1D_block_phi_circulant(
    a: NDArray,
    phase: NDArray,
    sections: int,
) -> NDArray:
    """Helper function to transform a matrix into a block circulant matrix with
    the given number of sections."""

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

    matrix = xp.zeros_like(a)
    for i in range(sections):
        if i == 0:
            phased_blocks = blocks[-i:]
        else:
            phased_blocks = [
                phase[:, *([None] * (len(block.shape) - 1))] * block
                for block in blocks[-i:]
            ]

        shifted_blocks = phased_blocks + blocks[:-i]
        matrix[..., i * block_size : (i + 1) * block_size, :] = xp.concatenate(
            shifted_blocks, axis=-1
        )

    return matrix


def _make_2D_block_phi_circulant(
    a: NDArray,
    phase_x: NDArray,
    phase_y: NDArray,
    sections_x: int,
    sections_y: int,
) -> NDArray:
    """Helper function to transform a matrix into a block circulant matrix with
    the given number of sections."""

    if a.shape[-1] % sections_x != 0:
        raise ValueError("The last dimension of a must be divisible by sections_x.")
    if a.shape[-1] % sections_y != 0:
        raise ValueError("The last dimension of a must be divisible by sections_y.")
    if a.shape[-1] % (sections_x * sections_y) != 0:
        raise ValueError(
            "The last dimension of a must be divisible by the section product."
        )

    if a.shape[-2] != a.shape[-1]:
        raise ValueError(
            "The second to last dimension of a must be equal to the last dimension of a."
        )

    block_size_x = a.shape[-1] // sections_x

    # make first circulant in the y direction
    for i in range(0, a.shape[-1], block_size_x):
        a[..., :block_size_x, i : i + block_size_x] = _make_1D_block_phi_circulant(
            a[..., :block_size_x, i : i + block_size_x],
            phase_y,
            sections=sections_y,
        )

    return _make_1D_block_phi_circulant(a, phase_x, sections=sections_x)


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


def _2D_dft(a: NDArray):
    """Apply the 2D discrete Fourier transform (DFT) to the input array a. The
    DFT is applied along the first two dimensions of a.
    The DFT is done explictly using the DFT matrices, which is not the most efficient way for
    many sections, but it is assumed that the number of sections is small.

    Parameters
    ----------
    a : NDArray
        The input array to transform. The first two dimensions are assumed to be the section dimensions.

    Returns
    -------
    NDArray
        The transformed array after applying the 2D DFT along the first two dimensions.

    """

    sections_y = a.shape[0]
    sections_x = a.shape[1]

    # stores the first block-layer of the system, which has the shape
    # (sections_y, sections_x, batch_size, block_size_y, block_size_x)
    dft_x = _get_dft_matrix(sections_x)
    dft_y = _get_dft_matrix(sections_y)

    # Transform along the y-sections
    a = xp.einsum("ij, jk... -> ik...", dft_y, a)

    # Transform along the x-sections
    a = xp.einsum("ij, kj... -> ki...", dft_x, a)

    return a * xp.sqrt(sections_x * sections_y)


def _2D_idft(a: NDArray) -> NDArray:
    """Apply the 2D inverse discrete Fourier transform (IDFT) to the input array
    a along the first two dimensions, scaled to match the block-diagonal form.
    The IDFT is done explictly using the DFT matrices, which is not the most
    efficient way for many sections, but it is assumed that the number of
    sections is small.

    Parameters
    ----------
    a : NDArray
        The input array to transform. The first two dimensions are assumed to be
        the section dimensions.

    Returns
    -------
    NDArray
        The transformed array after applying the 2D IDFT along the first two
        dimensions.

    """
    sections_y = a.shape[0]
    sections_x = a.shape[1]

    idft_x = _get_idft_matrix(sections_x)
    idft_y = _get_idft_matrix(sections_y)

    # Transform along the y-sections
    a = xp.einsum("ij, jk... -> ik...", idft_y, a)

    # Transform along the x-sections
    a = xp.einsum("ij, kj... -> ki...", idft_x, a)

    return a / xp.sqrt(sections_x * sections_y)


def transform_circulant(
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

    return _2D_dft(a)


def _core_detransform(
    a: NDArray,
    phase_x: NDArray | None = None,
    phase_y: NDArray | None = None,
) -> NDArray:
    """Core helper that handles the 2D matrix DFT, phase modulation,
    and block-interleaving layout reconstruction.

    Parameters
    ----------
    a : NDArray
        The input array in block diagonal form. The first two dimensions are
        assumed to be the section dimensions, the third dimension is the batch
        dimension, the fourth dimension is the block size, and the fifth
        dimension is the number of eigenvalues per block.
    phase_x : NDArray | None, optional
        The phase factors for the x direction, by default None. If provided, it
        should have shape (batch_size,).
    phase_y : NDArray | None, optional
        The phase factors for the y direction, by default None. If provided, it
        should have shape (batch_size,).

    Returns
    -------
    NDArray
        The detransformed array in the original matrix form. The first dimension
        is the batch dimension, the second and third dimensions are the original
        matrix dimensions, and the fourth dimension is the number of
        eigenvalues.

    """
    sections_y, sections_x = a.shape[0], a.shape[1]
    batch_size = a.shape[2]
    block_size_y = a.shape[3]

    dft_x = _get_dft_matrix(sections_x)
    dft_y = _get_dft_matrix(sections_y)

    if phase_x is not None and phase_y is not None:
        beta_x = phase_x ** (1 / sections_x)
        beta_y = phase_y ** (1 / sections_y)
        bx = xp.array([beta_x**i for i in range(sections_x)]).T
        by = xp.array([beta_y**i for i in range(sections_y)]).T

        step_y = xp.einsum(
            "yk, uk, klbij, by, bu -> yulbij", dft_y, dft_y.conj(), a, by, 1 / by
        )

        spatial_blocks = xp.einsum(
            "xl, vl, yulbij, bx, bv -> yxbvuij", dft_x, dft_x.conj(), step_y, bx, 1 / bx
        )
    else:
        step_y = xp.einsum("yk, uk, klbij -> yulbij", dft_y, dft_y.conj(), a)

        spatial_blocks = xp.einsum(
            "xl, vl, yulbij -> yxbvuij", dft_x, dft_x.conj(), step_y
        )

    # Reconstruct the final block-matrix layout
    permuted = xp.transpose(spatial_blocks, (2, 1, 0, 5, 3, 4, 6))

    total_size = sections_x * sections_y * block_size_y
    return permuted.reshape(batch_size, total_size, total_size)


def detransform_circulant(
    a: NDArray,
) -> NDArray:
    """Inverse transformation of the block diagonal form to the original matrix
    form. This is the inverse of the `transform_circulant` function and uses the
    inverse DFT matrices.

    Parameters
    ----------
    a : NDArray
        The matrix to detransform.

    Returns
    -------
    NDArray
        The original matrix before transformation.

    """

    if a.ndim != 5:
        raise ValueError(
            "The input matrix must have 5 dimensions after transformation."
        )

    return _core_detransform(a)


def detransform_circulant_vector(
    v: NDArray,
):
    """Inverse transformation for the eigenvectors of the block diagonal form to
    the original matrix

     Parameters
     ----------
     v : NDArray
         The eigenvectors in the block diagonal form. The first two dimensions
         are assumed to be the section dimensions, the third dimension is the
         batch dimension, the fourth dimension is the block size, and the fifth
         dimension is the number of eigenvalues per block.

     Returns
     -------
     NDArray
         The eigenvectors in the original matrix form. The first dimension is the
         batch dimension, the second and third dimensions are the original matrix
         dimensions, and the fourth dimension is the number of eigenvalues.

    """
    sections_y, sections_x = v.shape[0], v.shape[1]

    idft_x = _get_dft_matrix(sections_x)
    idft_y = _get_dft_matrix(sections_y)

    batch_size = v.shape[2]
    block_size = v.shape[3]
    eigenvalues_per_block = v.shape[4]

    v_expanded = xp.einsum("yk, xj, kjbie -> bxyikje", idft_y, idft_x, v)

    N_total = sections_x * sections_y * block_size
    total_eigenvectors = sections_x * sections_y * eigenvalues_per_block

    v_flat = v_expanded.reshape(batch_size, N_total, total_eigenvectors)

    return v_flat


def transform_phi_circulant(
    a: NDArray,
    phase_x: NDArray,
    phase_y: NDArray,
    sections_x: int = 1,
    sections_y: int = 1,
):
    r"""Transform a matrix to block diagonal form by exploiting the
    $\phi$-circulant structure of the matrix. This transformation is based on the
    discrete Fourier transform (DFT). It is assumed that the matrix is
    already sorted according to the transverse direction.
    
    Being $\phi$-circulant implies that the matrix has the form of:

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
            0 & \phi^{\frac{1}{3}} & 0 \\
            0 & 0 & \phi^{\frac{2}{3}}
        \end{bmatrix}
    \]

    For the case of a 2D $\phi$-circulant structure, the matrix has the same
    form as above, but each block is again $\phi$-circulant.

    Parameters
    ----------
    a : NDArray
        The matrix to transform. The last two dimensions are assumed to be
        square and the last dimension is assumed to be divisible by sections_x
        and sections_y.
    phase_x : NDArray
        The phase shift in the x direction. This is the phase per batch.
    phase_y : NDArray
        The phase shift in the y direction. This is the phase per batch.
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

    if not isinstance(phase_x, xp.ndarray):
        raise TypeError("phase_x must be an array.")

    if not isinstance(phase_y, xp.ndarray):
        raise TypeError("phase_y must be an array.")

    if phase_x.ndim > 1 or phase_y.ndim > 1:
        raise ValueError("phase_x and phase_y must be 1D arrays.")

    block_size_x = a.shape[-1] // sections_x

    if block_size_x % sections_y != 0:
        raise ValueError("The last dimension of a must be divisible by sections_y.")

    block_size_y = block_size_x // sections_y

    # view along x direction
    a = _block_view(a[..., :block_size_x, :], axis=-1, num_blocks=sections_x)
    # view along y direction
    a = _block_view(a[..., :block_size_y, :], axis=-1, num_blocks=sections_y)

    # the problem can be transformed to the circulant case
    # where the is a phase shift on the blocks

    beta_x = phase_x ** (1 / sections_x)
    beta_y = phase_y ** (1 / sections_y)

    betas_x = xp.array([beta_x**i for i in range(sections_x)])
    betas_y = xp.array([beta_y**i for i in range(sections_y)])

    beta = betas_y[:, None, :] * betas_x[None, :, :]
    a = a * beta[..., None, None]

    return _2D_dft(a)


def detransform_phi_circulant(
    a: NDArray,
    phase_x: NDArray,
    phase_y: NDArray,
) -> NDArray:
    """Inverse transformation of the block diagonal form to the original matrix
    form. This is the inverse of the `transform_phi_circulant` function and uses the
    inverse DFT matrices.

    Parameters
    ----------
    a : NDArray
        The matrix to detransform.
    phase_x : NDArray
        The phase shift in the x direction. This is the phase per batch.
    phase_y : NDArray
        The phase shift in the y direction. This is the phase per batch.

    Returns
    -------
    NDArray
        The original matrix before transformation.

    """

    if not isinstance(phase_x, xp.ndarray):
        raise TypeError("phase_x must be an array.")

    if not isinstance(phase_y, xp.ndarray):
        raise TypeError("phase_y must be an array.")

    if phase_x.ndim > 1 or phase_y.ndim > 1:
        raise ValueError("phase_x and phase_y must be 1D arrays.")

    return _core_detransform(a, phase_x=phase_x, phase_y=phase_y)


def detransform_phi_circulant_vector(
    v: NDArray,
    phase_x: NDArray,
    phase_y: NDArray,
):
    """Inverse transformation for the eigenvectors of the block diagonal form to
    the original matrix

     Parameters
     ----------
     v : NDArray
         The eigenvectors in the block diagonal form. The first two dimensions
         are assumed to be the section dimensions, the third dimension is the
         batch dimension, the fourth dimension is the block size, and the fifth
         dimension is the number of eigenvalues per block.
     phase_x : NDArray
         The phase shift in the x direction. This is the phase per batch.
     phase_y : NDArray
         The phase shift in the y direction. This is the phase per batch.

     Returns
     -------
     NDArray
         The eigenvectors in the original matrix form. The first dimension is the
         batch dimension, the second and third dimensions are the original matrix
         dimensions, and the fourth dimension is the number of eigenvalues.

    """
    sections_y, sections_x = v.shape[0], v.shape[1]

    v_flat = detransform_circulant_vector(v)

    block_size_y = v.shape[-2]

    # need to apply the inverse of the phase transformation
    beta_x = phase_x ** (1 / sections_x)
    beta_y = phase_y ** (1 / sections_y)

    betas_x = xp.array([beta_x**i for i in range(sections_x)]).T
    betas_y = xp.array([beta_y**i for i in range(sections_y)]).T

    ones = xp.ones(block_size_y, dtype=betas_x.dtype)

    beta = xp.einsum("bi,bj,k->bijk", betas_x, betas_y, ones).reshape(len(phase_x), -1)

    return v_flat * beta[..., None]
