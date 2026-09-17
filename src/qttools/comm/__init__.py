# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes the singleton communicator."""

from qttools.comm.comm import QuatrexCommunicator

# Instantiate the singleton communicator.
comm = QuatrexCommunicator()

__all__ = ["comm"]
