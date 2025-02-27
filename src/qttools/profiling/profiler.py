# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import json
import os
import time
import warnings
from functools import wraps

from mpi4py.MPI import COMM_WORLD as comm

from qttools import xp
from qttools.utils.gpu_utils import get_cuda_devices

# Set the default profiling detail.
PROFILING_DETAIL = os.environ.get("PROFILING_DETAIL", "basic")
if PROFILING_DETAIL.lower() not in ("off", "basic", "detailed"):
    warnings.warn(
        f"Invalid profiling detail {PROFILING_DETAIL}. Defaulting to 'basic'."
    )
    PROFILING_DETAIL = "basic"

# Set the default profiling group/environment.
PROFILING_GROUP = os.environ.get("PROFILING_GROUP", "basic")
if PROFILING_GROUP.lower() not in ("off", "basic", "api", "debug", "full"):
    warnings.warn(f"Invalid profiling group {PROFILING_GROUP}. Defaulting to 'basic'.")
    PROFILING_GROUP = "basic"

# Define the mapping of profiling groups to numbers.
_group_to_num = {"off": 0, "basic": 1, "api": 2, "debug": 3, "full": 4}


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

    def gather_events(self, root: int = 0) -> list:
        """Gathers profiling events.

        Returns
        -------
        list
            A list of profiling events.

        """
        all_events = comm.gather(self.eventlog, root=root)
        if comm.rank == root:
            return all_events
        return []

    def report(self):
        """Generates a report of the total time spent in each function.

        Returns
        -------
        dict
            A dictionary with function names as keys and total time
            spent as values.

        """
        report_data = {}
        for profile_id, func_name, host_time, device_times in self.data:
            report_dict = {"func": func_name, "host": host_time[-1]}
            for dev_id, dev_runtime in enumerate(device_times):
                if dev_runtime:
                    report_dict[f"device_{dev_id}"] = dev_runtime[-1]
            report_data[profile_id] = report_dict
        return report_data

    def dump_json(self, filepath):
        """Dumps the profiling report as a JSON file.

        Parameters
        ----------
        filepath : str
            The path to the output JSON file.

        """
        report_data = self.report()
        with open(filepath, "w") as json_file:
            json.dump(report_data, json_file, indent=4)

    def _setup_events(
        self,
    ) -> tuple[list[xp.cuda.stream.Event], list[xp.cuda.stream.Event]]:
        """Sets up CUDA events for each device.

        Returns
        -------
        tuple[list[xp.cuda.stream.Event], list[xp.cuda.stream.Event]]
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

    def _record_events(self, events: list[xp.cuda.stream.Event]):
        """Records events for each device.

        Parameters
        ----------
        events : list[xp.cuda.stream.Event]
            A list of events to record.

        """
        for device, event in zip(self.devices, events):
            current_device = xp.cuda.runtime.getDevice()
            try:
                xp.cuda.runtime.setDevice(device)
                event.record(xp.cuda.stream.Stream(device))
            finally:
                xp.cuda.runtime.setDevice(current_device)

    def _synchronize_events(self, events: list[xp.cuda.stream.Event]):
        """Synchronizes events for each device.

        Parameters
        ----------
        events : list[xp.cuda.stream.Event]
            A list of events to synchronize.

        """
        for device, event in zip(self.devices, events):
            current_device = xp.cuda.runtime.getDevice()
            try:
                xp.cuda.runtime.setDevice(device)
                event.synchronize()
            finally:
                xp.cuda.runtime.setDevice(current_device)

    def profile(self, detail: str = PROFILING_DETAIL, group: str = PROFILING_GROUP):
        """Profiles a function and adds profiling data to the event log.

        Notes
        -----

        Parameters
        ----------
        detail : str, optional
            With which detail to profile the function. If set, this
            overrides the PROFILING_DETAIL environment variable. The
            following details are available:
            - `"off"`: No profiling. The function is returned as is.
            - `"basic"`: Only the total time spent in the function.
            - `"detailed"`: The total time spent in the function and the
                time spent on each device.
        group : str, optional
            The profiling group controls whether the function is
            profiled or not. By default, the function is always
            profiled, irrespective of the PROFILING_GROUP environment
            variable. The following groups are implemented:
            - `"off"`: The function is not profiled.
            - `"basic"`: The function is part of the core profiling.
            - `"api"`: The function is part of the API and does not
              always need to be timed. It is part of the underlying
              infrastructure.
            - `"debug"`: This function only needs to be profiled for
              debugging purposes.
            - `"full"`: The function does not even need to be profiled for
              debugging purposes unless the user explicitly requests it.

        Returns
        -------
        callable
            The wrapped function with profiling according to the
            specified level.

        """
        if detail not in ("off", "basic", "detailed"):
            raise ValueError(f"Invalid profiling detail {detail}.")

        if group not in ("off", "basic", "api", "debug", "full"):
            raise ValueError(f"Invalid profiling group {group}.")

        def decorator(func):
            if detail == "off" or _group_to_num[group] < _group_to_num[PROFILING_GROUP]:
                return func

            name = func.__str__()

            @wraps(func)
            def wrapper(*args, **kwargs):

                timestamp = time.time()

                if detail == "detailed":
                    start_events, end_events = self._setup_events()

                    # Record and sync start events for each device.
                    self._record_events(start_events)
                    self._synchronize_events(start_events)

                    # Record start events for each device.
                    self._record_events(start_events)

                host_time = -time.perf_counter()

                # Call the function.
                result = func(*args, **kwargs)

                host_time += time.perf_counter()

                device_times = []
                if detail == "detailed":
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
