import pytest

from qttools.utils.mpi_utils import get_section_sizes


@pytest.mark.parametrize(
    "num_elements, num_sections, strategy, expected",
    [
        (10, 2, "balanced", ([5, 5], 10)),
        (10, 3, "greedy", ([4, 4, 2], 12)),
        (7, 3, "balanced", ([3, 2, 2], 9)),
        (7, 3, "greedy", ([3, 3, 1], 9)),
        (7, 7, "balanced", ([1, 1, 1, 1, 1, 1, 1], 7)),
        (7, 7, "greedy", ([1, 1, 1, 1, 1, 1, 1], 7)),
    ],
)
def test_get_section_sizes(
    num_elements: int,
    num_sections: int,
    strategy: str,
    expected: tuple[list[int], int],
):
    assert (
        get_section_sizes(
            num_elements=num_elements,
            num_sections=num_sections,
            strategy=strategy,
        )
        == expected
    )
