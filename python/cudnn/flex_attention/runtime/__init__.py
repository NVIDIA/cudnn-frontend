# SPDX-License-Identifier: BSD-3-Clause
"""Runtime support for FlexAttention CuTe DSL kernels."""

from .arch import SUPPORTED_ARCHES, get_device_arch

__all__ = ["SUPPORTED_ARCHES", "get_device_arch"]
