from qttools import NDArray, xp
from qttools.profiling import Profiler

profiler = Profiler()


@profiler.profile(level="debug")
def _system_reduction_rows(
    a: NDArray,
    q: NDArray,
    contact: str,
    solve,
    rows_to_reduce: slice,
):
    """Reduces the system by rows of A that are all zero.

    Parameters
    ----------
    a : NDArray
        The system matrix.
    q : NDArray
        The right-hand side matrix.
    contact : str
        The contact to which the boundary blocks belong.
    solve : function
        The solver to use for the reduced system.
    rows_to_reduce : slice
        The slice of rows to reduce.

    Returns
    -------
    x : NDArray
        The solution of the reduced system.

    """

    a_hat = a[..., rows_to_reduce, rows_to_reduce]
    a = xp.broadcast_to(a, q.shape)

    x = q.copy()
    x[..., rows_to_reduce, rows_to_reduce] = 0
    q_hat = q[..., rows_to_reduce, rows_to_reduce] + (
        a[..., rows_to_reduce, :]
        @ x
        @ a[..., rows_to_reduce, :].conj().swapaxes(-2, -1)
    )

    x[..., rows_to_reduce, rows_to_reduce] = solve(a_hat, q_hat, contact)

    return x


@profiler.profile(level="debug")
def _system_reduction_cols(
    a: NDArray,
    q: NDArray,
    contact: str,
    solve,
    cols_to_reduce: slice,
):
    """Reduces the system by columns of A that are all zero.

    Parameters
    ----------
    a : NDArray
        The system matrix.
    q : NDArray
        The right-hand side matrix.
    contact : str
        The contact to which the boundary blocks belong.
    solve : function
        The solver to use for the reduced system.
    cols_to_reduce : slice
        The slice of columns to reduce

    Returns
    -------
    x : NDArray
        The solution of the reduced system.

    """

    a_hat = a[..., cols_to_reduce, cols_to_reduce]

    q_hat = q[..., cols_to_reduce, cols_to_reduce]

    x_hat = solve(a_hat, q_hat, contact)

    a = xp.broadcast_to(a, q.shape)
    x = q.copy() + a[..., :, cols_to_reduce] @ x_hat @ a[
        ..., :, cols_to_reduce
    ].conj().swapaxes(-2, -1)

    return x


@profiler.profile(level="debug")
def system_reduction(
    a: NDArray,
    q: NDArray,
    contact: str,
    solve,
    out: None | NDArray = None,
):
    """Computes the solution of the discrete-time Lyapunov equation.

    The system is reduced by rows of A (AXA^H - X + Q = 0) that are all zero.
    This results in a system which is only of size n x n, where n is the number
    of rows with non-zero elements.

    The matrices a and q can have different ndims with q.ndim >= a.ndim (will broadcast)

    Parameters
    ----------
    a : NDArray
        The system matrix.
    q : NDArray
        The right-hand side matrix.
    contact : str
        The contact to which the boundary blocks belong.
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

    if out is not None:
        assert out.shape == q.shape

    assert q.shape[-2:] == a.shape[-2:]
    assert q.ndim >= a.ndim

    # get first and last row/cols with non-zero elements
    nnz_rows = xp.sum(xp.abs(a), axis=-1) > 0
    nnz_cols = xp.sum(xp.abs(a), axis=-2) > 0

    row_start = xp.argmax(nnz_rows, axis=-1)
    row_end = a.shape[-1] - xp.argmax(nnz_rows[..., ::-1], axis=-1)

    col_start = xp.argmax(nnz_cols, axis=-1)
    col_end = a.shape[-2] - xp.argmax(nnz_cols[..., ::-1], axis=-1)

    any_rows = xp.any(nnz_rows, axis=-1)
    any_cols = xp.any(nnz_cols, axis=-1)

    # account for only zero rows/cols
    # else will not reduce
    rows_to_reduce = slice(xp.min(row_start[any_rows]), xp.max(row_end[any_rows]))
    cols_to_reduce = slice(xp.min(col_start[any_cols]), xp.max(col_end[any_cols]))
    length_row = rows_to_reduce.stop - rows_to_reduce.start
    length_col = cols_to_reduce.stop - cols_to_reduce.start

    # only reduce in either rows or cols
    # TODO: reduce in both directions
    # but not occuring in the current use case
    # Would be calling reduce cols inside reduce rows
    # or reduce rows inside reduce cols
    # Furthermore, possible to reduce to non contiguous rows/cols

    if length_row < length_col:
        x = _system_reduction_rows(a, q, contact, solve, rows_to_reduce)
    else:
        x = _system_reduction_cols(a, q, contact, solve, cols_to_reduce)

    if out is None:
        return x
    out[:] = x
