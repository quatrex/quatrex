# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
import time
from math import isclose

import pytest

from qttools.profiling.profiler import Profiler, _ProfilingEvent, _ProfilingRun


@pytest.fixture
def profiler() -> Profiler:
    return Profiler()


def test_profiler_singleton(profiler):
    """Tests that the Profiler class is a singleton."""
    profiler2 = Profiler()
    assert profiler is profiler2


def test_profiler_event_initialization():
    """Tests that the _ProfilingEvent class is initialized correctly."""
    event = [time.time(), 0, "test_func", 0.1, 0.2]
    rank = 0
    profiling_event = _ProfilingEvent(event, rank)
    assert profiling_event.label == "test_func"
    assert profiling_event.call_time == 0.1
    assert profiling_event.after_barrier_time == 0.2
    assert profiling_event.rank == rank


def test_profiling_run_initialization():
    """Tests that the _ProfilingRun class is initialized correctly."""
    eventlogs = [[[time.time(), 0, "test_func", 0.1, 0.2]]]
    profiling_run = _ProfilingRun(eventlogs)
    assert len(profiling_run.profiling_events) == 1
    assert profiling_run.profiling_events[0].label == "test_func"


def test_profiling_run_get_stats():
    """Tests that the get_stats method returns expected statistics."""

    # Emulates to have two ranks with two calls to "test_func".
    eventlogs = [
        [[time.time(), 0, "test_func", 0.1, 0.2]],
        [[time.time(), 0, "test_func", 0.2, 0.3]],
    ]
    profiling_run = _ProfilingRun(eventlogs)
    stats = profiling_run.get_stats()
    assert "test_func" in stats
    assert stats["test_func"]["num_calls"] == 2
    assert stats["test_func"]["num_calls_per_rank"] == 1
    assert isclose(stats["test_func"]["total_call_time"], 0.3)
    assert isclose(stats["test_func"]["total_call_time_per_rank"], 0.15)


def test_profiler_decorator(profiler):
    """Tests that the profiler can be used as a decorator."""

    @profiler.profile(label="test function", level="default")
    def test_func():
        return "test"

    result = test_func()
    assert result == "test"

    # Check that "test_func" can be found in the eventlog.
    print(profiler.eventlog)
    assert any("test function" in event[2] for event in profiler.eventlog)


def test_profiler_profile_range(profiler):
    """Tests that the profiler can be used to profile a code block."""
    with profiler.profile_range(label="test range", level="default"):
        pass

    with profiler.profile_range(label="other_test_range", level="debug"):
        __ = 1 + 1

    # Check that "test_range" can be found in the eventlog.
    for event in profiler.eventlog:
        print(event)
    assert any("test range" in event[2] for event in profiler.eventlog)
    assert not any("other_test_range" in event[2] for event in profiler.eventlog)


def test_profiler_dump(profiler, tmp_path):
    """Tests that the profiler can dump the stats to a file."""
    filepath = tmp_path / "test.pkl"
    profiler.set_parameters(
        save_path=filepath,
        save_format="pickle",
    )
    profiler.dump_stats()
    assert filepath.exists()

    filepath = tmp_path / "test.json"
    profiler.set_parameters(
        save_path=filepath,
        save_format="json",
    )
    profiler.dump_stats()
    assert filepath.exists()
