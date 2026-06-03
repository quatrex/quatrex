# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from mpi4py.MPI import COMM_WORLD as comm

from qttools import xp
from qttools.utils.gpu_utils import synchronize_device


def get_cpu_memory_gb() -> float:
    """Get current CPU memory usage of the current process in GB.

    Returns
    -------
    float
        Current process CPU memory usage in GB, or 0.0 if it cannot be
        determined.

    """
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # VmRSS is in kB
                    return int(line.split()[1]) / 1024 / 1024
    except FileNotFoundError:
        # If status file is not found (e.g., on non-Linux systems),
        # return 0.0
        return 0.0
    return 0.0


def print_memory_usage() -> None:
    """Print CPU/GPU memory usage for rank 0."""
    if comm.rank != 0:
        return

    prefix = "[Memory]"

    cpu_mem_gb = get_cpu_memory_gb()
    if xp.__name__ == "cupy":
        synchronize_device()
        gpu_mem_free, gpu_mem_total = xp.cuda.Device().mem_info
        gpu_mem_used_gb = (gpu_mem_total - gpu_mem_free) / 1024 / 1024 / 1024
        gpu_mem_total_gb = gpu_mem_total / 1024 / 1024 / 1024
        print(
            f"{prefix}: CPU {cpu_mem_gb:.2f} GB, GPU {gpu_mem_used_gb:.2f}/{gpu_mem_total_gb:.2f} GB",
            flush=True,
        )
    else:
        print(f"{prefix}: CPU {cpu_mem_gb:.2f} GB", flush=True)
