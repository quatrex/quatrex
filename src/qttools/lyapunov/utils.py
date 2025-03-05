from qttools import NDArray, xp


def system_reduction(
    a: NDArray,
    q: NDArray,
    solve,
    out: None | NDArray = None,
):
    """Computes the solution of the discrete-time Lyapunov equation.

    The system is reduced by rows of A (AXA^H - X + Q = 0) that are all zero.
    This results in a system which is only of size n x n, where n is the number
    of rows with non-zero elements.

    Parameters
    ----------
    a : NDArray
        The system matrix.
    q : NDArray
        The right-hand side matrix.
    solve : function
        The solver to use for the reduced system.
    out : NDArray, optional
        The array to store the result in. If not provided, a new
        array is returned.

    Returns
    -------
    x : NDArray | None
        The solution of the discrete-time Lyapunov equation.

    """

    batch_shape = a.shape[:-2]

    assert a.shape == q.shape
    if out is not None:
        assert out.shape == a.shape

    a = a.reshape(-1, *a.shape[-2:])
    q = q.reshape(-1, *q.shape[-2:])

    if a.ndim == 2:
        a = a[xp.newaxis, ...]
        q = q[xp.newaxis, ...]

    # NOTE: possible to further reduce
    # but it is assumed contiguous rows are non-zero

    # get first row with non-zero elements
    row_start = xp.argmax(xp.sum(xp.abs(a), axis=-1) > 0, axis=-1)
    # get last row with non-zero elements
    row_end = a.shape[-1] - xp.argmax(xp.sum(xp.abs(a), axis=-1)[:, ::-1] > 0, axis=-1)

    # assumes same sparsity pattern for all matrices
    assert xp.all(row_start == row_start[0])
    assert xp.all(row_end == row_end[0])
    row_start = row_start[0]
    row_end = row_end[0]

    a_hat = a[:, row_start:row_end, row_start:row_end]

    x = q.copy()
    x[:, row_start:row_end, row_start:row_end] = 0
    q_hat = q[:, row_start:row_end, row_start:row_end] + (
        a[:, row_start:row_end, :]
        @ x
        @ a[:, row_start:row_end, :].conj().swapaxes(-2, -1)
    )

    x[:, row_start:row_end, row_start:row_end] = solve(a_hat, q_hat)

    x = x.reshape(*batch_shape, *x.shape[-2:])

    if out is None:
        return x
    out[:] = x
