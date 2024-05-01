from qttools.datastructures.dbsparse import DBSparse


class Solver:

    def __init__(self, config) -> None:
        pass

    def selected_inv(a: DBSparse, out=None, **kwargs) -> None | DBSparse:
        pass

    def selected_solve(
        a: DBSparse,
        sigma_lesser: DBSparse,
        sigma_greater: DBSparse,
        out: tuple | None = None,
        return_retarded: bool = False,
        **kwargs,
    ) -> None | tuple:
        pass
