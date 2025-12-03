# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import json
import os
import pickle
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Literal

from mpi4py.MPI import COMM_WORLD as comm

from qttools import strtobool, xp

NVTX_AVAILABLE = xp.__name__ == "cupy" and xp.cuda.nvtx.available

# Set the profiling level.
QTX_PROFILE_LEVEL = os.getenv("QTX_PROFILE_LEVEL", "default").lower()
if QTX_PROFILE_LEVEL not in ("off", "default", "debug"):
    warnings.warn(
        f"Invalid profiling level {QTX_PROFILE_LEVEL=}. Defaulting to 'default'."
    )
    QTX_PROFILE_LEVEL = "default"

# Define the mapping of profiling levels to numbers.
_level_to_num = {"off": 0, "default": 1, "debug": 2}

QTX_PROFILE_COMM_SYNC = strtobool(os.getenv("QTX_PROFILE_COMM_SYNC"), True)


def _get_cuda_devices(return_names: bool = False):
    """Returns the list of available CUDA devices.

    Parameters
    ----------
    return_names
        If the device names should be written out.

    Returns
    ----------
    list
        List of available devices
    """
    if xp.__name__ != "cupy":
        return []
    num_devices = xp.cuda.runtime.getDeviceCount()
    if return_names:
        return [f"cuda:{i}" for i in range(num_devices)]

    return list(range(num_devices))


class _ProfilingEvent:
    """A profiling event object.

    This is basically just there to parse the names of the profiled
    functions.

    Parameters
    ----------
    event : list
        The profiling event data.
    rank : int
        The MPI rank on which the event
        occurred.

    Attributes
    ----------
    datetime : datetime
        The timestamp of the event.
    depth: int
        The depth of the profiled function.
    label : str
        The label of the profiled function.
    call_time : float
        The time spent on the call.
    after_barrier_time : float
        The time spent including the barrier
    rank : int
        The MPI rank on which the event occurred.

    """

    def __init__(self, event: list, rank: int):
        """Initializes the profiling event object."""
        timestamp, depth, label, call_time, after_barrier_time = event
        # TODO: Here we parse the timestamp as a datetime object. It
        # would be very nice to have a trace plot of the profiling
        # data, but this would require a bit more work.
        self.datetime = datetime.fromtimestamp(timestamp)
        self.depth = depth
        self.label = label
        self.call_time = call_time
        self.after_barrier_time = after_barrier_time
        self.rank = rank


class _ProfilingRun:
    """A profiling run object.

    Parameters
    ----------
    eventlogs : list
        A list of profiling events for each rank.

    Attributes
    ----------
    profiling_events : list[_ProfilingEvent]
        A list of parsed profiling events.

    """

    def __init__(self, eventlogs: list[list]):
        """Initializes the profiling run object."""
        profiling_events: list[_ProfilingEvent] = []
        for rank, events in enumerate(eventlogs):
            for event in events:
                profiling_events.append(_ProfilingEvent(event, rank))

        self.profiling_events = profiling_events

    def get_stats(self) -> dict:
        """Returns the profiling statistics.

        This reports some statistics for each profiled function.

        Returns
        -------
        dict
            A dictionary containing the profiling statistics.

        """
        call_stats = defaultdict(list)
        after_barrier_stats = defaultdict(list)
        ranks = defaultdict(set)
        depths = defaultdict(set)
        for event in self.profiling_events:
            call_stats[event.label].append(event.call_time)
            after_barrier_stats[event.label].append(event.after_barrier_time)
            ranks[event.label].add(event.rank)
            depths[event.label].add(event.depth)

        stats = {}
        for key in call_stats:
            call_times = xp.array(call_stats[key])

            num_calls = len(call_times)
            num_ranks = len(ranks[key])
            print(ranks[key])
            total_call_time = float(xp.sum(call_times))

            stats[key] = {
                "num_calls": num_calls,
                "num_participating_ranks": num_ranks,
                "num_calls_per_rank": num_calls / num_ranks,
                "total_call_time": total_call_time,
                "total_call_time_per_rank": total_call_time / num_ranks,
                "average_call_time": float(xp.mean(call_times)),
                "median_call_time": float(xp.median(call_times)),
                "std_call_time": float(xp.std(call_times)),
                "min_call_time": float(xp.min(call_times)),
                "max_call_time": float(xp.max(call_times)),
            }

            after_barrier_times = xp.array(after_barrier_stats[key])
            total_after_barrier_time = float(xp.sum(after_barrier_times))
            stats[key].update(
                {
                    "total_after_barrier_time": total_after_barrier_time,
                    "total_after_barrier_time_per_rank": total_after_barrier_time
                    / num_ranks,
                    "average_after_barrier_time": float(xp.mean(after_barrier_times)),
                    "median_after_barrier_time": float(xp.median(after_barrier_times)),
                    "std_after_barrier_time": float(xp.std(after_barrier_times)),
                    "min_after_barrier_time": float(xp.min(after_barrier_times)),
                    "max_after_barrier_time": float(xp.max(after_barrier_times)),
                }
            )

        return stats


class Profiler:
    """Singleton Profiler class to collect and report profiling data.

    Attributes
    ----------
    eventlog : list
        A list of profiling data.
    devices : list
        A list of CUDA device IDs.

    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Profiler, cls).__new__(cls)

            cls._instance.eventlog = []
            cls._instance.devices = _get_cuda_devices()
            cls._instance.depth = 0

            if xp.__name__ == "cupy":
                # NOTE: this consumes some resources
                # could be moved to the __init__ of the Profiler class
                cls._instance.start_event = xp.cuda.stream.Event()
                cls._instance.end_event = xp.cuda.stream.Event()
                cls._instance.after_barrier_event = xp.cuda.stream.Event()

        return cls._instance

    def _gather_events(self, root: int = 0) -> list:
        """Gathers profiling events.

        Returns
        -------
        list
            A list of profiling events or an empty list.

        """
        all_events = comm.gather(self.eventlog, root=root)
        if comm.rank == root:
            return all_events
        return [[]]

    def get_stats(self) -> dict:
        """Computes statistics from profiling data accross all ranks.

        Returns
        -------
        dict
            A dictionary containing the profiling data.

        """
        return _ProfilingRun(self._gather_events()).get_stats()

    def dump_stats(self, filepath: str, format: Literal["pickle", "json"] = "pickle"):
        """Dumps the profiling statistics to a file.

        Parameters
        ----------
        filepath : str
            The path to the output file. The correct file extension
            will be appended based on the format.
        format : {"pickle", "json"}, optional
            The format in which to save the profiling data.

        """
        if format not in ("pickle", "json"):
            raise ValueError(f"Invalid format {format}.")

        stats = self.get_stats()
        if comm.rank != 0:
            # Only the root rank dumps the stats.
            return

        filepath = os.fspath(filepath)
        os.path.isdir(os.path.dirname(filepath))
        if format == "pickle":
            if not filepath.endswith(".pkl"):
                filepath += ".pkl"
            with open(filepath, "wb") as pickle_file:
                pickle.dump(stats, pickle_file)
        else:
            if not filepath.endswith(".json"):
                filepath += ".json"
            with open(filepath, "w") as json_file:
                json.dump(stats, json_file, indent=4)

    def profile(self, label: str, level: str, comm=None):
        """Profiles a function and adds profiling data to the event log.

        Notes
        -----
        Two environment variables control the profiling behavior:
        - `PROFILE_LEVEL`: The profiling level for functions. The
            following levels are implemented:
            - `"off"`: The function is not profiled.
            - `"default"`: The function is part of the core profiling.
            - `"debug"`: This function only needs to be profiled for
              debugging purposes.
        - `PROFILE_COMM_SYNC`: If set to `True`, a communicator barrier
            is called after the profiled function to ensure that all
            processes are synchronized before recording the end time.
            Through this, differences in between processes can be
            better captured.

        Parameters
        ----------
        label : str
            A label for the profiled range. This is used to identify
            the profiled range in the profiling data.
        level : str
            The profiling level controls whether the function is
            profiled or not. The following levels are implemented:
            - `"off"`: The function is not profiled.
            - `"default"`: The function is part of the core profiling.
            - `"debug"`: This function only needs to be profiled for
              debugging purposes.
        comm : optional
            An optional communicator to use for synchronization

        Returns
        -------
        callable
            The wrapped function with profiling according to the
            specified level.

        """
        if level not in ("off", "default", "debug"):
            raise ValueError(f"Invalid profiling level {level}.")

        if comm is not None:
            assert hasattr(
                comm, "barrier"
            ), "The communicator must have a barrier attribute."

        def decorator(func):
            if _level_to_num[level] > _level_to_num[QTX_PROFILE_LEVEL]:
                return func

            @wraps(func)
            def wrapper(*args, **kwargs):

                self.depth += 1
                timestamp = time.time()

                # NOTE: We maybe need to barrier before starting the timer

                if xp.__name__ == "cupy":
                    if NVTX_AVAILABLE:
                        xp.cuda.nvtx.RangePush(label)

                    self.start_event.record(xp.cuda.get_current_stream())

                else:
                    start_time = time.perf_counter()

                # Call the function.
                result = func(*args, **kwargs)

                if xp.__name__ == "cupy":
                    if NVTX_AVAILABLE:
                        xp.cuda.nvtx.RangePop()

                    self.end_event.record(xp.cuda.get_current_stream())
                    self.end_event.synchronize()
                    call_time = (
                        xp.cuda.get_elapsed_time(self.start_event, self.end_event)
                        * 1e-3
                    )  # Convert to seconds.
                else:
                    call_time = time.perf_counter() - start_time

                if comm is not None and QTX_PROFILE_COMM_SYNC:
                    comm.barrier()
                    if xp.__name__ == "cupy":
                        self.after_barrier_event.record(xp.cuda.get_current_stream())
                        self.after_barrier_event.synchronize()
                        after_barrier_time = (
                            xp.cuda.get_elapsed_time(
                                self.start_event, self.after_barrier_event
                            )
                            * 1e-3
                        )
                    else:
                        after_barrier_time = time.perf_counter() - start_time
                else:
                    after_barrier_time = call_time

                self.eventlog.append(
                    (timestamp, self.depth, label, call_time, after_barrier_time)
                )
                self.depth -= 1

                return result

            return wrapper

        return decorator

    @contextmanager
    def profile_range(self, label: str, level: str):
        """Profiles a range of code.

        This is a context manager that profiles a range of code.

        Parameters
        ----------
        label : str
            A label for the profiled range. This is used to identify
            the profiled range in the profiling data.
        level : str
            The profiling level controls whether the function is
            profiled or not:
            - `"off"`: The function is not profiled.
            - `"default"`: The function is part of the core profiling.
            - `"debug"`: This function only needs to be profiled for
              debugging purposes.

        Yields
        ------
        None
            The context manager does not return anything.

        """
        if level not in ("off", "default", "debug"):
            raise ValueError(f"Invalid profiling level {level}.")

        if _level_to_num[level] > _level_to_num[QTX_PROFILE_LEVEL]:
            yield
            return

        try:
            self.depth += 1
            timestamp = time.time()

            # NOTE: We maybe need to barrier before starting the timer

            if xp.__name__ == "cupy":
                if NVTX_AVAILABLE:
                    xp.cuda.nvtx.RangePush(label)

                self.start_event.record(xp.cuda.get_current_stream())

            else:
                start_time = time.perf_counter()

            yield

        finally:

            if xp.__name__ == "cupy":
                if NVTX_AVAILABLE:
                    xp.cuda.nvtx.RangePop()

                self.end_event.record(xp.cuda.get_current_stream())
                self.end_event.synchronize()
                call_time = (
                    xp.cuda.get_elapsed_time(self.start_event, self.end_event) * 1e-3
                )  # Convert to seconds.
            else:
                call_time = time.perf_counter() - start_time

            if comm is not None and QTX_PROFILE_COMM_SYNC:
                comm.barrier()
                if xp.__name__ == "cupy":
                    self.after_barrier_event.record(xp.cuda.get_current_stream())
                    self.after_barrier_event.synchronize()
                    after_barrier_time = (
                        xp.cuda.get_elapsed_time(
                            self.start_event, self.after_barrier_event
                        )
                        * 1e-3
                    )
                else:
                    after_barrier_time = time.perf_counter() - start_time
            else:
                after_barrier_time = call_time

            self.eventlog.append(
                (timestamp, self.depth, label, call_time, after_barrier_time)
            )
            self.depth -= 1
