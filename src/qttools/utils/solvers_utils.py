# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import NDArray, xp
from qttools.utils.mpi_utils import get_section_sizes


def get_batches(num_sections: int, max_batch_size: int) -> tuple[list, NDArray]:
    """Computes the number of batches and their sizes.

    Parameters
    ----------
    num_sections : int
        The total number of sections to divide.
    max_batch_size : int
        The maximum size of each batch.

    Returns
    -------
    batches_sizes : list
        The sizes of each batch.
    batches_slices : NDArray
        The offsets of each batch.

    """
    # Get list of batches to perform
    batches_sizes, _ = get_section_sizes(
        num_elements=num_sections,
        num_sections=num_sections // min(max_batch_size, num_sections),
    )
    batches_slices = xp.hstack(([0], xp.cumsum(xp.array(batches_sizes))))

    return batches_sizes, batches_slices
