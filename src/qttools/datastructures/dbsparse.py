from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import sparray


class DBSparse(ABC):
    """Abstract base class for distributed block sparse matrices."""

    @abstractmethod
    def from_sparray(
        a: sparray,
        blocksizes: np.ndarray,
        stackshape: tuple = (1,),
        densify_blocks=None,
        pinned=False,
    ): ...

    @abstractmethod
    def to_dense(self) -> np.ndarray: ...

    @abstractmethod
    def zeros_like(a: DBSparse) -> DBSparse: ...

    @abstractmethod
    def block_diagonal(
        offset: int = 0, dense: bool = False
    ) -> list[sparray] | list[np.ndarray]: ...

    @abstractmethod
    def diagonal() -> np.ndarray: ...

    @abstractmethod
    def local_transpose(copy=False) -> None | DBSparse: ...

    @abstractmethod
    def distributed_transpose() -> None: ...

    @abstractmethod
    def __setitem__(
        self, idx: tuple[int, int], value: np.ndarray
    ) -> None:

    @abstractmethod
    def __getitem__(
        self, idx: tuple[int, int]
    ) -> sparray:

    @abstractmethod
    def __iadd__(self, other: DBSparse) -> None: ...

    @abstractmethod
    def __imul__(self, other: DBSparse) -> None: ...

    @abstractmethod
    def __neg__(self) -> None: ...

    @abstractmethod
    def __matmul__(self, other: DBSparse) -> None: ...

    @property
    @abstractmethod
    def num_blocks(self) -> np.uint: ...

    @property
    @abstractmethod
    def block_offsets(self) -> np.uint: ...

    @property
    @abstractmethod
    def stack_shape(self) -> np.uint: ...

    @property
    @abstractmethod
    def shape(self) -> np.uint: ...

    @property
    @abstractmethod
    def nzz(self) -> np.uint: ...

    @property
    @abstractmethod
    def return_dense(self) -> bool: ...
    
    @abstractmethod
    @return_dense.setter
    def return_dense(self, value: bool) -> None: ...

