# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, xp

if xp.__name__ == "cupy":
    import cupyx as cpx

    @cpx.jit.rawkernel()
    def _contour_operator(
        output: NDArray,
        a_xx: NDArray,
        z: NDArray,
        batchsize,
        num_quatrature_points,
        blocksize,
        b,
    ):
        # assumes c order of output and a_xx

        idx = int(cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x)
        if idx < batchsize * num_quatrature_points * blocksize * blocksize:

            # batch index
            ie = idx // (num_quatrature_points * blocksize * blocksize)

            # index within the batch
            ijk = idx % (num_quatrature_points * blocksize * blocksize)

            # quatrature point index
            i = ijk // (blocksize * blocksize)

            # index within the block
            jk = ijk % (blocksize * blocksize)

            # row index
            j = jk // blocksize

            # column index
            k = jk % blocksize

            # access quatraure point
            z_i = z[i]

            for h in range(0, 2 * b + 1):
                # offset in list of blocks
                m_idx = h * blocksize * blocksize * batchsize
                # batch offset
                m_idx += ie * blocksize * blocksize
                # block index
                m_idx += j * blocksize + k
                output[idx] += a_xx[m_idx] * z_i ** (h - b)


def operator_inverse(
    a_xx: tuple[NDArray, ...],
    z: NDArray,
    contour_type: xp.dtype,
    in_type: xp.dtype,
    num_threads: int = 1024,
) -> NDArray:
    """Computes the inverse of a matrix polynomial at sample points.

    Parameters
    ----------
    a_xx : tuple[NDArray, ...]
        The coefficients of the matrix polynomial.
    z : NDArray
        The sample points at which to compute the inverse.
    contour_type : xp.dtype
        The data type for the contour integration.
    in_type : xp.dtype
        The data type for the input matrices.
    num_threads : int, optional
        The number of cuda threads to use for the kernel.
        Only relevant for GPU computations.

    Returns
    -------
    NDArray
        The inverse of the matrix polynomial.

    """

    b = len(a_xx) // 2

    if xp.__name__ == "numpy":
        # NOTE: NUMBA kernel is possible
        operator = sum(z**n * a_xn for a_xn, n in zip(a_xx, range(-b, b + 1)))

    elif xp.__name__ == "cupy":
        batchsize, _, blocksize, _ = a_xx[0].shape
        num_quatrature_points = z.size

        operator = xp.zeros(
            (batchsize, num_quatrature_points, blocksize, blocksize),
            dtype=a_xx[0].dtype,
        )

        num_blocks = (
            batchsize * num_quatrature_points * blocksize * blocksize + num_threads - 1
        ) // num_threads

        # NOTE: this lead to memory copy
        # better to always have a_xx as a high dim array
        a_xx = xp.array(a_xx)
        _contour_operator[num_blocks, num_threads](
            operator.reshape(-1),
            a_xx.reshape(-1),
            z.reshape(-1),
            batchsize,
            num_quatrature_points,
            blocksize,
            b,
        )

    return xp.linalg.inv(operator.astype(contour_type)).astype(in_type)
