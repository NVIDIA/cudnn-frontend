# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral contract for operation kernels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ._tensor_desc import TensorDesc


class OpKernel(ABC):
    """Host-side contract implemented by framework-agnostic kernels.

    Implementations own logical validation and tensor inference. Framework
    adapters remain responsible for turning the inferred descriptors into
    framework tensors or abstract output declarations and for compiling and
    executing the kernel. Callable implementations receive inputs, outputs,
    workspaces, and finally the CUDA stream. Framework adapters may reorder
    those arguments to match a framework-specific launch ABI.
    """

    @abstractmethod
    def check_support(self) -> bool:
        """Validate the logical signature and resolve static kernel state."""

    @abstractmethod
    def infer_output(self) -> tuple[TensorDesc[Any], ...]:
        """Return operation outputs through the framework-neutral descriptor interface.

        A descriptor's ``init_value`` requests scalar initialization before
        launch; ``None`` leaves the output uninitialized.
        """

    def infer_workspace(self) -> tuple[TensorDesc[Any], ...]:
        """Return workspaces through the framework-neutral descriptor interface.

        Workspace descriptors follow the same ``init_value`` convention as
        outputs but are not part of the framework-visible result.
        """

        return ()


__all__ = ["OpKernel"]
