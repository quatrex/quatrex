from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import sparray


class DBSparse(ABC):
    """Abstract base class for distributed block sparse matrices."""

    @property
    @abstractmethod
    def system(self) -> str: ...

    @abstractmethod
    def from_sparray(
        a: sparray,
        blocksizes: np.ndarray,
        stackshape: tuple = (1,),
        densify_blocks=None,
        pinned=False,
    ):
        pass

    @abstractmethod
    def to_dense(self) -> np.ndarray:
        pass

    @abstractmethod
    def zeros_like(a: DBSparse) -> DBSparse:
        pass

    @abstractmethod
    def get_block(i: int, j: int, dense: bool = False) -> sparray | np.ndarray:
        pass

    @abstractmethod
    def set_block(i: int, j: int, block: np.ndarray) -> None:
        pass

    @abstractmethod
    def block_diagonal(
        offset: int = 0, dense: bool = False
    ) -> list[sparray] | list[np.ndarray]:
        pass

    @abstractmethod
    def diagonal() -> np.ndarray:
        pass

    @abstractmethod
    def local_transpose(copy=False) -> None | DBSparse:
        pass

    @abstractmethod
    def distributed_transpose() -> None:
        pass

    @abstractmethod
    def __iadd__(self, other: DBSparse) -> None:
        pass

    @abstractmethod
    def __imul__(self, other: DBSparse) -> None:
        pass

    @abstractmethod
    def __neg__(self) -> None:
        pass

    @abstractmethod
    def __matmul__(self, other: DBSparse) -> None:
        pass
