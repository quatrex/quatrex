from qttools import NDArray, xp


def _system_reduction_rows(
    a: NDArray,
    q: NDArray,
    solve,
    row_start,
    row_end,
):
    """Reduces the system by rows of A that are all zero.

    Parameters
    ----------
    a : NDArray
        The system matrix.
    q : NDArray
        The right-hand side matrix.
    solve : function
        The solver to use for the reduced system.
    row_start : int
        The first row with non-zero elements.
    row_end : int
        The last row with non-zero elements.

    Returns
    -------
    x : NDArray
        The solution of the reduced system.

    """

    a_hat = a[:, row_start:row_end, row_start:row_end]

    x = q.copy()
    x[:, row_start:row_end, row_start:row_end] = 0
    q_hat = q[:, row_start:row_end, row_start:row_end] + (
        a[:, row_start:row_end, :]
        @ x
        @ a[:, row_start:row_end, :].conj().swapaxes(-2, -1)
    )

    x[:, row_start:row_end, row_start:row_end] = solve(a_hat, q_hat)

    return x


def _system_reduction_cols(
    a: NDArray,
    q: NDArray,
    solve,
    col_start,
    col_end,
):
    """Reduces the system by columns of A that are all zero.

    Parameters
    ----------
    a : NDArray
        The system matrix.
    q : NDArray
        The right-hand side matrix.
    solve : function
        The solver to use for the reduced system.
    col_start : int
        The first column with non-zero elements.
    col_end : int
        The last column with non-zero elements.

    Returns
    -------
    x : NDArray
        The solution of the reduced system.

    """

    a_hat = a[:, col_start:col_end, col_start:col_end]

    q_hat = q[:, col_start:col_end, col_start:col_end]

    x_hat = solve(a_hat, q_hat)

    x = q.copy() + a[:, :, col_start:col_end] @ x_hat @ a[
        :, :, col_start:col_end
    ].conj().swapaxes(-2, -1)

    return x


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

    # get first and last row/cols with non-zero elements
    nnz_rows = xp.sum(xp.abs(a), axis=-1) > 0
    nnz_cols = xp.sum(xp.abs(a), axis=-2) > 0

    row_start = xp.argmax(nnz_rows, axis=-1)
    row_end = a.shape[-1] - xp.argmax(nnz_rows[:, ::-1], axis=-1)

    col_start = xp.argmax(nnz_cols, axis=-1)
    col_end = a.shape[-2] - xp.argmax(nnz_cols[:, ::-1], axis=-1)

    any_rows = xp.any(nnz_rows, axis=-1)
    any_cols = xp.any(nnz_cols, axis=-1)

    # account for only zero rows/cols
    # else will not reduce
    row_start = xp.min(row_start[any_rows])
    row_end = xp.max(row_end[any_rows])
    col_start = xp.min(col_start[any_cols])
    col_end = xp.max(col_end[any_cols])

    length_row = row_end - row_start
    length_col = col_end - col_start

    # only reduce in either rows or cols
    # TODO: reduce in both directions
    # but not occuring in the current use case
    # Would be calling reduce cols inside reduce rows
    # or reduce rows inside reduce cols
    # Furthermore, possible to reduce to non contiguous rows/cols

    if length_row < length_col:
        x = _system_reduction_rows(a, q, solve, row_start, row_end)
    else:
        x = _system_reduction_cols(a, q, solve, col_start, col_end)

    x = x.reshape(*batch_shape, *x.shape[-2:])

    if out is None:
        return x
    out[:] = x
