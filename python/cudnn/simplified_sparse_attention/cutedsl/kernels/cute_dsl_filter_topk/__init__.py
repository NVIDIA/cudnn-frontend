# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Filtered top-k kernels and compatibility helpers for the CuTe DSL wheel."""

import sys
import types

import cutlass

# ---- cutlass.memory shim (public wheel keeps these in cutlass.utils) ----
if not hasattr(cutlass, "memory"):
    import cutlass.utils as _cu

    _mem = types.ModuleType("cutlass.memory")
    _mem.SmemAllocator = _cu.SmemAllocator
    _mem.SmemPartition = _cu.SmemPartition
    _mem.get_kernel_smem_size = _cu.get_kernel_smem_size
    _mem.get_smem_capacity_in_bytes = _cu.get_smem_capacity_in_bytes
    sys.modules["cutlass.memory"] = _mem
    cutlass.memory = _mem

# ---- cutlass.block shim (vendored copy of the in-tree module) ----
if not hasattr(cutlass, "block"):
    from . import _cutlass_block_compat as _blk

    sys.modules["cutlass.block"] = _blk
    cutlass.block = _blk

from .filtered_top_k_prefill_varlen import (  # noqa: E402
    FilteredTopKKernelVarlenPrefill,
    cute_dsl_topk_prefill_wrapper,
)

__all__ = ["FilteredTopKKernelVarlenPrefill", "cute_dsl_topk_prefill_wrapper"]
