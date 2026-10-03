# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context-parallel SDPA numerics reference / validation helpers (issue #752).

This package is a *test-side* oracle. Nothing here is imported by the production
``cudnn`` package, and nothing here claims that a cuDNN or TransformerEngine GPU
kernel was executed. The three evidence layers are:

``reference_math``
    R0: an independent FP64 mathematical oracle. Explicit attention plus the
    analytic chunk-merge formula. Independent of the code under test.
``te_adapter``
    R1: a merge-only oracle that reproduces the *step order, dtype and rounding*
    of the pinned NVIDIA TransformerEngine context-parallel helpers (see
    ``TE_PINNED_SHA``), plus a separately labelled self-written emulator lane.
    It consumes frozen partials; it never runs a TE attention kernel.
``trace_schema``
    machine-readable CP trace records plus the scheduling invariants that make a
    frozen-partial replay checkable (coverage without duplication, unique slots,
    order preservation, global token order, no padding as valid KV).

Scope limits (see the #752 execution plan): no CP runtime, no NCCL ring tuning,
no training-framework integration, no backward kernels, no global ``chunk_size``
API design.
"""

__all__ = ["reference_math", "te_adapter", "trace_schema"]
