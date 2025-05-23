# Copyright (c) 2025 ETH Zurich and the authors of the qttools package.

from qttools.comm.comm import QuatrexCommunicator

# Instantiate the singleton communicator.
comm = QuatrexCommunicator()

__all__ = ["comm"]
