# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compatibility exports for the internal CuTe DSL JAX adapter."""

from .._jax.cutedsl import BufferSpec, call_cutedsl

__all__ = ["BufferSpec", "call_cutedsl"]
