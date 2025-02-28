import time
from math import isclose

import pytest

from qttools.profiling.profiler import Profiler, _ProfilingEvent, _ProfilingRun
from qttools.profiling.utils import decorate_methods


@pytest.fixture
def profiler() -> Profiler:
    return Profiler()


def test_profiler_singleton(profiler):
    """Tests that the Profiler class is a singleton."""
    profiler2 = Profiler()
    assert profiler is profiler2


def test_profiler_event_initialization():
    """Tests that the _ProfilingEvent class is initialized correctly."""
    event = [time.time(), "<function test_func at 0x1234>", 0.1, [0.2, 0.3]]
    rank = 0
    profiling_event = _ProfilingEvent(event, rank)
    assert profiling_event.prof_type == "function"
    assert profiling_event.qualname == "test_func"
    assert profiling_event.host_time == 0.1
    assert profiling_event.device_times == [0.2, 0.3]
    assert profiling_event.rank == rank


def test_profiling_run_initialization():
    """Tests that the _ProfilingRun class is initialized correctly."""
    eventlogs = [[[time.time(), "<function test_func at 0x1234>", 0.1, [0.2, 0.3]]]]
    profiling_run = _ProfilingRun(eventlogs)
    assert len(profiling_run.profiling_events) == 1
    assert profiling_run.profiling_events[0].qualname == "test_func"


def test_profiling_run_get_stats():
    """Tests that the get_stats method returns expected statistics."""
    eventlogs = [
        [[time.time(), "<function test_func at 0x1234>", 0.1, [0.2, 0.3]]],
        [[time.time(), "<function test_func at 0x5678>", 0.2, [0.2, 0.1]]],
    ]
    profiling_run = _ProfilingRun(eventlogs)
    stats = profiling_run.get_stats()
    assert "test_func" in stats
    assert stats["test_func"]["num_calls"] == 2
    assert stats["test_func"]["num_calls_per_rank"] == 1
    assert isclose(stats["test_func"]["total_host_time"], 0.3)
    assert isclose(stats["test_func"]["total_host_time_per_rank"], 0.15)


def test_profiler_decorator(profiler):
    """Tests that the profiler can be used as a decorator."""

    @profiler.profile(level="basic")
    def test_func():
        return "test"

    result = test_func()
    assert result == "test"

    # Check that "test_func" can be found in the eventlog.
    assert any("test_func" in event[1] for event in profiler.eventlog)


def test_profiler_profile_range(profiler):
    """Tests that the profiler can be used to profile a code block."""
    with profiler.profile_range("test_range", level="basic"):
        pass

    # Check that "test_range" can be found in the eventlog.
    assert any("test_range" in event[1] for event in profiler.eventlog)


def test_profiler_dump(profiler, tmp_path):
    """Tests that the profiler can dump the eventlog to a file."""
    filepath = tmp_path / "test.pkl"
    profiler.dump(filepath, format="pickle")
    assert filepath.exists()

    filepath = tmp_path / "test.json"
    profiler.dump(filepath, format="json")
    assert filepath.exists()


def test_decorate_methods(profiler):
    """Tests that the decorate_methods function works."""

    @decorate_methods(profiler.profile(level="basic"), exclude=["__init__"])
    class TestClass:
        def __init__(self):
            self.value = 0

        def test_method(self):
            self.value += 1

        def other_method(self):
            with profiler.profile_range("test_range", level="basic"):
                pass

    test_instance = TestClass()
    test_instance.test_method()
    assert test_instance.value == 1

    test_instance.other_method()

    # Check that "test_method" can be found in the eventlog.
    assert any("test_method" in event[1] for event in profiler.eventlog)
    assert any("test_range" in event[1] for event in profiler.eventlog)
    assert any("other_method" in event[1] for event in profiler.eventlog)
    assert not any("__init__" in event[1] for event in profiler.eventlog)
