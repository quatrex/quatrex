# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from abc import ABC, abstractmethod

import numpy as np
import numpy.lib.stride_tricks as npst
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from scipy.sparse import sparray

from qttools.utils.mpi_utils import get_num_elements_per_section


class DSBSparse(ABC):
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

        self.global_stack_shape = global_stack_shape

        # Padding
        self._padded_data = np.zeros(
            (max(stack_section_sizes), total_nnz_size), dtype=data.dtype
        )
        self._padded_data[: data.shape[0], : data.shape[1]] = data

        self._stack_section_sizes = stack_section_sizes
        self._nnz_section_sizes = nnz_section_sizes
        self._total_stack_size = total_stack_size
        self._total_nnz_size = total_nnz_size

        self._stack_shape = data.shape[:-1]
        self._shape = self.stack_shape + (np.sum(block_sizes), np.sum(block_sizes))
        self._nnz = self._padded_data.shape[-1]

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
    def __iadd__(self, other: "DSBSparse") -> None:
        ...

    @abstractmethod
    def __imul__(self, other: "DSBSparse") -> None:
        ...

    @abstractmethod
    def __neg__(self) -> None:
        ...

    @abstractmethod
    def __matmul__(self, other: "DSBSparse") -> None:
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
    def spy(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the row and column indices of the non-zero elements."""
        ...

    @abstractmethod
    def ltranspose(self, copy=False) -> "None | DSBSparse":
        """Performs a local transposition of the datastructure."""
        ...

    def _stack_to_nnz_dtranspose(self) -> None:
        """Transpose the data."""
        original_buffer_shape = self._padded_data.shape
        self._padded_data = np.ascontiguousarray(
            npst.as_strided(
                self._padded_data,
                shape=(
                    comm.size,
                    self._padded_data.shape[0],
                    self._padded_data.shape[1] // comm.size,
                ),
                strides=(
                    (self._padded_data.shape[1] // comm.size)
                    * self._padded_data.itemsize,
                    self._padded_data.shape[1] * self._padded_data.itemsize,
                    self._padded_data.itemsize,
                ),
            )
        )
        comm.Alltoall(MPI.IN_PLACE, self._padded_data)

        self._padded_data = self._padded_data.reshape(
            original_buffer_shape[0] * comm.size, original_buffer_shape[1] // comm.size
        )

    def _nnz_to_stack_dtranspose(self) -> None:
        """Transpose the data."""
        original_buffer_shape = self._padded_data.shape
        self._padded_data = self._padded_data.reshape(
            comm.size,
            self._padded_data.shape[0] // comm.size,
            self._padded_data.shape[1],
        )
        comm.Alltoall(MPI.IN_PLACE, self._padded_data)
        self._padded_data = self._padded_data.transpose(1, 0, 2)
        self._padded_data = self._padded_data.reshape(
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
        global_stack_shape: tuple,
        densify_blocks: list[tuple] | None = None,
        pinned=False,
    ) -> "DSBSparse":
        ...

    @classmethod
    @abstractmethod
    def zeros_like(cls, a: "DSBSparse") -> "DSBSparse":
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
    def data(self) -> np.ndarray:
        if self._distribution_state == "stack":
            return self._padded_data[
                : self._stack_section_sizes[comm.rank],
                : sum(self._nnz_section_sizes),
            ]
        else:
            return self._padded_data[
                : sum(self._stack_section_sizes), : self._nnz_section_sizes[comm.rank]
            ]
