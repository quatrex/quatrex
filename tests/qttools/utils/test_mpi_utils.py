import pytest

from qttools.utils.mpi_utils import get_section_sizes


@pytest.mark.parametrize(
    "num_elements, num_sections, expected",
    [
        (10, 2, ([5, 5], 10)),
        (10, 3, ([4, 3, 3], 12)),
        (10, 4, ([3, 3, 2, 2], 12)),
        (10, 6, ([2, 2, 2, 2, 1, 1], 12)),
        (10, 9, ([2, 1, 1, 1, 1, 1, 1, 1, 1], 18)),
        (7, 2, ([4, 3], 8)),
        (7, 7, ([1, 1, 1, 1, 1, 1, 1], 7)),
    ],
)
def test_get_section_sizes(
    num_elements: int, num_sections: int, expected: tuple[list[int], int]
):
    assert get_section_sizes(num_elements, num_sections) == expected
