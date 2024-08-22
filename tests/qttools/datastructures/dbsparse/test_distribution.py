import pytest


@pytest.mark.mpi(min_size=2)
def test_dtranspose(): ...
