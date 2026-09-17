# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the contact classes."""

from quatrex.contact.base import BaseContact
from quatrex.contact.qtbm import QTBMContact
from quatrex.contact.scba import SCBAContact

__all__ = ["BaseContact", "QTBMContact", "SCBAContact"]
