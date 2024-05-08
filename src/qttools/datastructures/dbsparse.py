# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

from abc import ABC, abstractmethod

import numpy as np
import numpy.lib.stride_tricks as npst
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from scipy.sparse import sparray

from qttools.utils.mpi_utils import get_num_elements_per_section


class DBSparse(ABC):
    """Abstract base class for distributed block sparse matrices.

    The data is distributed in blocks across the ranks. The data is
    stored in a padded format to allow for efficient transposition even
    if the number of ranks does not evenly divide the stack dimension /
    the number of non-zero elements.

    """

    def __init__(
        self,
        data: np.ndarray,
        block_sizes: np.ndarray,
        global_stack_shape: tuple,
        return_dense: bool = False,
    ) -> None:
        """Initializes the DBSparse matrix."""
        if isinstance(global_stack_shape, int):
            global_stack_shape = (global_stack_shape,)

        if data.ndim != 2 or len(global_stack_shape) != 1:
            raise NotImplementedError("Currently only 2D data is supported.")

        stack_section_sizes, total_stack_size = get_num_elements_per_section(
            global_stack_shape[0], comm.size
        )
        nnz_section_sizes, total_nnz_size = get_num_elements_per_section(
            data.shape[1], comm.size
        )
        # Padding
        self.data = np.zeros(
            (max(stack_section_sizes), total_nnz_size), dtype=data.dtype
        )
        self.data[: data.shape[0], : data.shape[1]] = data

        self._stack_section_sizes = stack_section_sizes
        self._nnz_section_sizes = nnz_section_sizes
        self._total_stack_size = total_stack_size
        self._total_nnz_size = total_nnz_size

        self._stack_shape = data.shape[:-1]
        self._shape = self.stack_shape + (np.sum(block_sizes), np.sum(block_sizes))
        self._nnz = self.data.shape[-1]

        self._block_sizes = np.asarray(block_sizes).astype(int)
        self._block_offsets = np.hstack(([0], np.cumsum(self._block_sizes)))
        self._num_blocks = len(block_sizes)
        self._return_dense = return_dense

        self._distribution_state = "stack"

    def _unsign_block_index(self, brow: int, bcol: int) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds."""
        brow = self.num_blocks + brow if brow < 0 else brow
        bcol = self.num_blocks + bcol if bcol < 0 else bcol
        if not (0 <= brow < self.num_blocks and 0 <= bcol < self.num_blocks):
            raise IndexError("Block index out of bounds.")

        return brow, bcol

    @abstractmethod
    def __setitem__(self, idx: tuple[int, int], block: np.ndarray) -> None:
        ...

    @abstractmethod
    def __getitem__(self, idx: tuple[int, int]) -> sparray:
        ...

    @abstractmethod
    def __iadd__(self, other: "DBSparse") -> None:
        ...

    @abstractmethod
    def __imul__(self, other: "DBSparse") -> None:
        ...

    @abstractmethod
    def __neg__(self) -> None:
        ...

    @abstractmethod
    def __matmul__(self, other: "DBSparse") -> None:
        ...

    def block_diagonal(self, offset: int = 0) -> list[sparray] | list[np.ndarray]:
        """Returns the block diagonal of the matrix."""
        return [
            self[b, b]
            for b in range(
                max(0, -offset),
                min(self.num_blocks, self.num_blocks - offset),
            )
        ]

    def diagonal(self) -> np.ndarray:
        """Returns the diagonal of the matrix."""
        return np.hstack([np.diagonal(self[b, b]) for b in range(self.num_blocks)])

    @abstractmethod
    def ltranspose(self, copy=False) -> "None | DBSparse":
        """Performs a local transposition of the datastructure."""
        ...

    def _stack_to_nnz_dtranspose(self) -> None:
        """Transpose the data."""
        original_buffer_shape = self.data.shape
        self.data = np.ascontiguousarray(
            npst.as_strided(
                self.data,
                shape=(comm.size, self.data.shape[0], self.data.shape[1] // comm.size),
                strides=(
                    (self.data.shape[1] // comm.size) * self.data.itemsize,
                    self.data.shape[1] * self.data.itemsize,
                    self.data.itemsize,
                ),
            )
        )
        comm.Alltoall(MPI.IN_PLACE, self.data)

        self.data = self.data.reshape(
            original_buffer_shape[0] * comm.size, original_buffer_shape[1] // comm.size
        )

    def _nnz_to_stack_dtranspose(self) -> None:
        """Transpose the data."""
        original_buffer_shape = self.data.shape
        self.data = self.data.reshape(
            comm.size, self.data.shape[0] // comm.size, self.data.shape[1]
        )
        comm.Alltoall(MPI.IN_PLACE, self.data)
        self.data = self.data.transpose(1, 0, 2)
        self.data = self.data.reshape(
            original_buffer_shape[0] // comm.size, original_buffer_shape[1] * comm.size
        )

    def dtranspose(self) -> None:
        """Performs a distributed transposition of the datastructure."""
        if self.distribution_state == "stack":
            self._stack_to_nnz_dtranspose()
            self._distribution_state = "nnz"
        else:
            self._nnz_to_stack_dtranspose()
            self._distribution_state = "stack"

    @abstractmethod
    def to_dense(self) -> np.ndarray:
        ...

    @classmethod
    @abstractmethod
    def from_sparray(
        cls,
        a: sparray,
        block_sizes: np.ndarray,
        stack_shape: tuple | None = None,
        densify_blocks: list[tuple] | None = None,
        pinned=False,
    ) -> "DBSparse":
        ...

    @classmethod
    @abstractmethod
    def zeros_like(cls, a: "DBSparse") -> "DBSparse":
        ...

    @property
    def distribution_state(self) -> str:
        return self._distribution_state

    @property
    def stack_shape(self) -> np.uint:
        return self._stack_shape

    @property
    def shape(self) -> np.uint:
        return self._shape

    @property
    def nnz(self) -> np.uint:
        return self._nnz

    @property
    def num_blocks(self) -> np.uint:
        return self._num_blocks

    @property
    def block_sizes(self) -> np.uint:
        return self._block_sizes

    @property
    def block_offsets(self) -> np.uint:
        return self._block_offsets

    @property
    def return_dense(self) -> bool:
        return self._return_dense

    @return_dense.setter
    def return_dense(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("Return dense must be a boolean.")
        self._return_dense = value

    @property
    def masked_data(self) -> np.ndarray:
        if self._distribution_state == "stack":
            return self.data[
                : self._stack_section_sizes[comm.rank],
                : sum(self._nnz_section_sizes),
            ]
        else:
            return self.data[
                : sum(self._stack_section_sizes), : self._nnz_section_sizes[comm.rank]
            ]
