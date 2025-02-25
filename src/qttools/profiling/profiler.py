# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import json
import time
from functools import wraps

from qttools import xp
from qttools.utils.gpu_utils import get_cuda_devices


class Profiler:
    """Singleton Profiler class to collect and report profiling data.

    Attributes
    ----------
    data : defaultdict
        A dictionary to store profiling start and end times for
        functions.

    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Profiler, cls).__new__(cls)
            cls._instance.data = []

        return cls._instance

    def start(
        self,
        func_name,
        devices,
        start_events,
        finish_events,
        host_runtimes,
        dev_runtimes,
    ):
        """Records the start time of a function.

        Parameters
        ----------
        func_name : str
            The name of the function being profiled.

        """
        # self.data[func_name].append(("start", time.time()))
        for dev_id, event in zip(devices, start_events):
            current_device = xp.cuda.runtime.getDevice()
            try:
                xp.cuda.runtime.setDevice(dev_id)
                event.record(xp.cuda.stream.Stream(dev_id))
            finally:
                xp.cuda.runtime.setDevice(current_device)

        host_runtimes.append(-time.perf_counter())

    def end(
        self,
        func_name,
        devices,
        start_events,
        finish_events,
        host_runtimes,
        dev_runtimes,
    ):
        """Records the end time of a function.

        Parameters
        ----------
        func_name : str
            The name of the function being profiled.

        """
        # self.data[func_name].append(("end", time.time()))

        host_runtimes[-1] += time.perf_counter()

        for dev_id, event in zip(devices, finish_events):
            current_device = xp.cuda.runtime.getDevice()
            try:
                xp.cuda.runtime.setDevice(dev_id)
                event.record(xp.cuda.stream.Stream(dev_id))
            finally:
                xp.cuda.runtime.setDevice(current_device)

        for dev_id, event in zip(devices, finish_events):
            current_device = xp.cuda.runtime.getDevice()
            try:
                xp.cuda.runtime.setDevice(dev_id)
                event.record(xp.cuda.stream.Stream(dev_id))
            finally:
                xp.cuda.runtime.setDevice(current_device)
            event.synchronize()

        for dev_id, start_event, finish_event in zip(
            devices, start_events, finish_events
        ):
            dev_runtimes[dev_id].append(
                xp.cuda.get_elapsed_time(start_event, finish_event) * 1e-3
            )

    def report(self):
        """Generates a report of the total time spent in each function.

        Returns
        -------
        dict
            A dictionary with function names as keys and total time
            spent as values.

        """
        report_data = {}
        for profile_id, func_name, host_runtimes, dev_runtimes in self.data:
            # print(f"{profile_id}: {func_name} - {host_runtimes} - {dev_runtimes}")
            report_dict = {"func": func_name, "host": host_runtimes[-1]}
            for dev_id, dev_runtime in enumerate(dev_runtimes):
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

    def profile(self, name=None):
        """Decorator to profile a function.

        Parameters
        ----------
        func : callable
            The function to be profiled.
        name : str, optional
            The name to use for the function in the profiling report.

        Returns
        -------
        callable
            The wrapped function with profiling.

        """
        fname = name

        def decorator(func):

            name = func.__str__() if fname is None else fname

            devices = get_cuda_devices()
            start_events = []
            finish_events = []

            for dev_id in devices:
                current_device = xp.cuda.runtime.getDevice()
                try:
                    xp.cuda.runtime.setDevice(dev_id)
                    start_events.append(xp.cuda.stream.Event())
                    finish_events.append(xp.cuda.stream.Event())
                finally:
                    xp.cuda.runtime.setDevice(current_device)

            @wraps(func)
            def wrapper(*args, **kwargs):

                profile_id = time.time()

                host_runtimes = []
                dev_runtimes = [[] for _ in devices]

                for dev_id, event in zip(devices, start_events):
                    current_device = xp.cuda.runtime.getDevice()
                    try:
                        xp.cuda.runtime.setDevice(dev_id)
                        event.record(xp.cuda.stream.Stream(dev_id))
                    finally:
                        xp.cuda.runtime.setDevice(current_device)
                    event.synchronize()

                self.start(
                    name,
                    devices,
                    start_events,
                    finish_events,
                    host_runtimes,
                    dev_runtimes,
                )

                result = func(*args, **kwargs)

                self.end(
                    name,
                    devices,
                    start_events,
                    finish_events,
                    host_runtimes,
                    dev_runtimes,
                )

                self.data.append((profile_id, name, host_runtimes, dev_runtimes))

                return result

            return wrapper

        return decorator
