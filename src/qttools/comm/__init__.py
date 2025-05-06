# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.

from qttools.comm.comm import QuatrexCommunicator
from qttools.comm.utils import all_gather_v, pad_buffer

# Instantiate the singleton communicator.
comm = QuatrexCommunicator()

__all__ = ["comm", "pad_buffer", "all_gather_v"]
