# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes utility functions for memory operations."""

import os

from qttools import xp
from qttools.comm import comm
from qttools.utils.gpu_utils import synchronize_device


def _get_host_meminfo() -> tuple[float, float]:
    """Gets the total and free system memory in kB.

    Returns
    -------
    mem_free: float
        The free system memory in kB. Can be NaN if the information is
        not available.
    mem_total: float
        The total available system memory in kB. Can be NaN if the
        information is not available.

    """
    mem_total = float("nan")
    mem_free = float("nan")

    # Check whether /proc/meminfo is available and readable
    if not os.path.exists("/proc/meminfo") or not os.access("/proc/meminfo", os.R_OK):
        return mem_free, mem_total

    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                # The line looks like: "MemTotal:       16367432 kB"
                mem_total = float(line.split()[1])
            elif line.startswith("MemFree:"):
                # The line looks like: "MemFree:    8134524 kB"
                mem_free = float(line.split()[1])

    return mem_free, mem_total


def print_memory_usage(message: str | None = None, only_zero_rank: bool = True) -> None:
    """Print CPU/GPU memory usage for rank 0.

    Parameters
    ----------
    message: str | None, optional
        An optional message to print before the memory usage
        information.
    only_zero_rank: bool, optional
        If True, only print the memory usage for rank 0. If False, print
        the memory usage for all ranks.

    """
    if comm.rank != 0 and only_zero_rank:
        return

    if message is None:
        message = ""

    message = f"\n[Memory, rank {comm.rank}]: " + message

    message += "\n" + "-" * 80

    host_mem_free, host_mem_total = _get_host_meminfo()
    host_mem_used = host_mem_total - host_mem_free
    message += f"\nCPU {host_mem_used/1024**2:.2f}/{host_mem_total/1024**2:.2f} GB"

    if xp.__name__ == "cupy":
        synchronize_device()
        gpu_mem_free, gpu_mem_total = xp.cuda.Device().mem_info
        gpu_mem_used = gpu_mem_total - gpu_mem_free

        message += f"\nGPU {gpu_mem_used/1024**3:.2f}/{gpu_mem_total/1024**3:.2f} GB"

        pool = xp.get_default_memory_pool()
        message += f"\ncupy memory pool used: {pool.used_bytes()/1024**3:.2f}/{pool.total_bytes()/1024**3:.2f} GB"
        message += f"\ncupy memory pool total: {pool.total_bytes()/1024**3:.2f}/{gpu_mem_total/1024**3:.2f} GB"
        message += f"\noutside memory pool: {(gpu_mem_used - pool.total_bytes())/1024**3:.2f}/{gpu_mem_total/1024**3:.2f} GB"

    message += "\n" + "-" * 80 + "\n"

    print(message, flush=True)
