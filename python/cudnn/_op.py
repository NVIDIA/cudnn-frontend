# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral operation contract."""

from abc import ABC, abstractmethod


class Op(ABC):
    """Logical operation specialized from a complete tensor signature."""

    @abstractmethod
    def check_support(self) -> bool:
        """Validate the operation signature and resolve static configuration."""


__all__ = ["Op"]
