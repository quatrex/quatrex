# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

from qttools import xp
from qttools.utils.mpi_utils import get_section_sizes


def get_batches(num_sections: int, max_batch_size: int) -> tuple:
    # Get list of batches to perform
    batches_sizes, _ = get_section_sizes(
        num_elements=num_sections,
        num_sections=num_sections // min(max_batch_size, num_sections),
    )
    batches_slices = xp.cumsum(xp.array([0] + batches_sizes))

    return batches_sizes, batches_slices
