# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import json
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Literal

from mpi4py.MPI import COMM_WORLD as comm

from qttools import xp
from qttools.profiling.utils import get_cuda_devices

NVTX_AVAILABLE = xp.__name__ == "cupy" and xp.cuda.nvtx.available


# Set the whether to profile the GPU.
PROFILE_GPU = os.environ.get("PROFILE_GPU", "false").lower()
if PROFILE_GPU in ("y", "yes", "t", "true", "on", "1"):
    if xp.__name__ != "cupy":
        warnings.warn("CUDA is not available. Defaulting to no GPU profiling.")
        PROFILE_GPU = False
    else:
        warnings.warn(
            "GPU profiling is enabled. This will cause device "
            "synchronization for every profiled event."
        )
        PROFILE_GPU = True
elif PROFILE_GPU in ("n", "no", "f", "false", "off", "0"):
    PROFILE_GPU = False
else:
    warnings.warn(f"Invalid truth value {PROFILE_GPU=}. Defaulting to 'false'.")
    PROFILE_GPU = False


# Set the profiling level.
PROFILE_LEVEL = os.environ.get("PROFILE_LEVEL", "basic").lower()
if PROFILE_LEVEL not in ("off", "basic", "api", "debug", "full"):
    warnings.warn(f"Invalid profiling level {PROFILE_LEVEL=}. Defaulting to 'basic'.")
    PROFILE_LEVEL = "basic"

# Define the mapping of profiling levels to numbers.
_level_to_num = {"off": 0, "basic": 1, "api": 2, "debug": 3, "full": 4}


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
    prof_type : str
        The type of the profiling event.
    qualname : str
        The qualified name of the profiled function.
    prof_id : str
        The ID of the profiling event.
    host_time : float
        The time spent on the host.
    device_times : list
        The time spent on each device.
    rank : int
        The MPI rank on which the event occurred.

    """

    def __init__(self, event: list, rank: int):
        """Initializes the profiling event object."""
        timestamp, name, host_time, device_times = event
        # TODO: Here we parse the timestamp as a datetime object. It
        # would be very nice to have a trace plot of the profiling
        # data, but this would require a bit more work.
        self.datetime = datetime.fromtimestamp(timestamp)

        # Names will look like "<function Class.do_something at 0x...>".
        prof_type, qualname, *__ = name.strip("<>").split()
        self.prof_type = prof_type
        self.qualname = qualname

        self.host_time = host_time
        self.device_times = device_times
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
        host_stats = defaultdict(list)
        device_stats = defaultdict(list)
        ranks = defaultdict(set)
        for event in self.profiling_events:
            host_stats[event.qualname].append(event.host_time)
            device_stats[event.qualname].append(event.device_times)
            ranks[event.qualname].add(event.rank)

        stats = {}
        for key in host_stats:
            host_times = xp.array(host_stats[key])

            num_calls = len(host_times)
            num_ranks = len(ranks[key])
            total_host_time = float(xp.sum(host_times))

            stats[key] = {
                "num_calls": num_calls,
                "num_participating_ranks": num_ranks,
                "num_calls_per_rank": num_calls / num_ranks,
                "total_host_time": total_host_time,
                "total_host_time_per_rank": total_host_time / num_ranks,
                "average_host_time": float(xp.mean(host_times)),
                "median_host_time": float(xp.median(host_times)),
                "std_host_time": float(xp.std(host_times)),
                "min_host_time": float(xp.min(host_times)),
                "max_host_time": float(xp.max(host_times)),
            }
            device_times = xp.array(device_stats[key])
            if not xp.any(device_times):
                continue

            total_device_time = float(xp.sum(device_times))
            stats[key].update(
                {
                    "total_device_time": total_device_time,
                    "total_device_time_per_rank": total_device_time / num_ranks,
                    "average_device_time": float(xp.mean(device_times)),
                    "median_device_time": float(xp.median(device_times)),
                    "std_device_time": float(xp.std(device_times)),
                    "min_device_time": float(xp.min(device_times)),
                    "max_device_time": float(xp.max(device_times)),
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
            cls._instance.devices = get_cuda_devices()

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

    def _setup_events(self) -> tuple[list, list]:
        """Sets up CUDA events for each device.

        Returns
        -------
        tuple[list, list]
            A tuple of lists of start and end events for each device.

        """
        start_events = []
        end_events = []

        for device in self.devices:
            current_device = xp.cuda.runtime.getDevice()
            try:
                xp.cuda.runtime.setDevice(device)
                start_events.append(xp.cuda.stream.Event())
                end_events.append(xp.cuda.stream.Event())
            finally:
                xp.cuda.runtime.setDevice(current_device)

        return start_events, end_events

    def _record_events(self, events: list):
        """Records events for each device.

        Parameters
        ----------
        events : list
            A list of events to record.

        """
        for device, event in zip(self.devices, events):
            current_device = xp.cuda.runtime.getDevice()
            try:
                xp.cuda.runtime.setDevice(device)
                event.record(xp.cuda.stream.Stream(device))
            finally:
                xp.cuda.runtime.setDevice(current_device)

    def _synchronize_events(self, events: list):
        """Synchronizes events for each device.

        Parameters
        ----------
        events : list
            A list of events to synchronize.

        """
        for device, event in zip(self.devices, events):
            current_device = xp.cuda.runtime.getDevice()
            try:
                xp.cuda.runtime.setDevice(device)
                event.synchronize()
            finally:
                xp.cuda.runtime.setDevice(current_device)

    def profile(self, level: str = PROFILE_LEVEL):
        """Profiles a function and adds profiling data to the event log.

        Notes
        -----
        Two environment variables control the profiling behavior:
        - `PROFILE_GPU`: Whether to separately measure the time spent on
          the GPU. If turned on, this will cause device synchronization
          for every profiled event.
        - `PROFILE_LEVEL`: The profiling level for functions. The
            following levels are implemented:
            - `"off"`: The function is not profiled.
            - `"basic"`: The function is part of the core profiling.
            - `"api"`: The function is part of the API and does not
              always need to be timed. It is part of the underlying
              infrastructure.
            - `"debug"`: This function only needs to be profiled for
              debugging purposes.
            - `"full"`: The function does not even need to be profiled for
              debugging purposes unless the user explicitly requests it.


        Parameters
        ----------
        level : str, optional
            The profiling level controls whether the function is
            profiled or not. By default, the level is set to the
            PROFILE_LEVEL environment variable. The function is thus
            always profiled. The following levels are implemented:
            - `"off"`: The function is not profiled.
            - `"basic"`: The function is part of the core profiling.
            - `"api"`: The function is part of the API and does not
              always need to be timed. It is part of the underlying
              infrastructure.
            - `"debug"`: This function only needs to be profiled for
              debugging purposes.
            - `"full"`: The function does not even need to be profiled
              for debugging purposes unless the user explicitly requests
              it to be profiled.

        Returns
        -------
        callable
            The wrapped function with profiling according to the
            specified level.

        """
        if level not in ("off", "basic", "api", "debug", "full"):
            raise ValueError(f"Invalid profiling level {level}.")

        def decorator(func):
            if _level_to_num[level] > _level_to_num[PROFILE_LEVEL]:
                return func

            name = func.__str__()

            @wraps(func)
            def wrapper(*args, **kwargs):

                timestamp = time.time()

                if PROFILE_GPU:
                    start_events, end_events = self._setup_events()

                    # Record and sync start events for each device.
                    self._record_events(start_events)
                    self._synchronize_events(start_events)

                    # Record start events for each device.
                    self._record_events(start_events)

                # Push a range to NVTX if available.
                if NVTX_AVAILABLE:
                    xp.cuda.nvtx.RangePush(name)

                host_time = -time.perf_counter()

                # Call the function.
                result = func(*args, **kwargs)

                host_time += time.perf_counter()

                if NVTX_AVAILABLE:
                    xp.cuda.nvtx.RangePop()

                device_times = []
                if PROFILE_GPU:
                    # Record end events for each device.
                    self._record_events(end_events)

                    # Sync to ensure all devices are done.
                    self._synchronize_events(end_events)

                    # Calculate the time spent on each device.
                    for start_event, end_event in zip(start_events, end_events):
                        device_times.append(
                            xp.cuda.get_elapsed_time(start_event, end_event)
                            * 1e-3  # Convert to seconds.
                        )

                self.eventlog.append((timestamp, name, host_time, device_times))

                return result

            return wrapper

        return decorator

    @contextmanager
    def profile_range(self, label: str = "range", level: str = PROFILE_LEVEL):
        """Profiles a range of code.

        This is a context manager that profiles a range of code.

        Parameters
        ----------
        label : str, optional
            A label for the profiled range. This is used to identify
            the profiled range in the profiling data.
        level : str, optional
            The profiling level controls whether the function is
            profiled or not. By default, the function is always
            profiled, irrespective of the PROFILE_LEVEL environment
            variable. The following levels are implemented:
            - `"off"`: The function is not profiled.
            - `"basic"`: The function is part of the core profiling.
            - `"api"`: The function is part of the API and does not
              always need to be timed. It is part of the underlying
              infrastructure.
            - `"debug"`: This function only needs to be profiled for
              debugging purposes.
            - `"full"`: The function does not even need to be profiled
              for debugging purposes unless the user explicitly requests
              it to be profiled.

        Yields
        ------
        None
            The context manager does not return anything.

        """
        if level not in ("off", "basic", "api", "debug", "full"):
            raise ValueError(f"Invalid profiling level {level}.")

        if _level_to_num[level] > _level_to_num[PROFILE_LEVEL]:
            yield
            return

        # This is quite a bit of a hack to get the qualified name of the
        # function in which the context manager is called.
        qualname = "no_qualname"
        if hasattr(sys, "_getframe"):
            qualname = sys._getframe(2).f_code.co_qualname

        label = "." + label.replace(" ", "_")
        name = "<range " + qualname + label + ">"

        try:
            timestamp = time.time()

            if PROFILE_GPU:
                start_events, end_events = self._setup_events()

                # Record and sync start events for each device.
                self._record_events(start_events)
                self._synchronize_events(start_events)

                # Record start events for each device.
                self._record_events(start_events)

            # Push a range to NVTX if available.
            if NVTX_AVAILABLE:
                xp.cuda.nvtx.RangePush(name)

            host_time = -time.perf_counter()

            yield

        finally:

            host_time += time.perf_counter()

            if NVTX_AVAILABLE:
                xp.cuda.nvtx.RangePop()

            device_times = []
            if PROFILE_GPU:
                # Record end events for each device.
                self._record_events(end_events)

                # Sync to ensure all devices are done.
                self._synchronize_events(end_events)

                # Calculate the time spent on each device.
                for start_event, end_event in zip(start_events, end_events):
                    device_times.append(
                        xp.cuda.get_elapsed_time(start_event, end_event)
                        * 1e-3  # Convert to seconds.
                    )

            self.eventlog.append((timestamp, name, host_time, device_times))
