# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared partial-output ABI helpers for paged and split SDPA tests."""

import torch


def _partial_o_dtype(splits, o_dtype):
    """The dtype the compiled kernel WRITES its O argument in.

    Under a split the epilogue stores its fp32 accumulator registers straight to
    the partial workspace; unsplit it TMA-stores the staged SMEM tile in O's own
    dtype.  These tests drive the compiled kernel directly, so they have to hand
    it the buffer the template's ABI expects rather than the caller-facing one.
    """
    return torch.float32 if splits > 1 else o_dtype


def _partial_kwargs(splits, o_p):
    """The extra ``o_partial_f32`` argument a split-compiled kernel carries.

    Keyword, not positional: the slot is LAST in the kernel signature precisely
    so the THD tensors ahead of it do not shift when it is absent.
    """
    return {"o_partial_f32": o_p} if splits > 1 else {}


def _partial_tag(splits, o_dtype):
    """The matching ``dtype_partial`` for ``sm100/split_combine.compile_ptr``.

    Compiling the combine for the wrong width reinterprets the workspace and
    yields garbage rather than an error, so it has to track _partial_o_dtype.
    """
    if splits > 1:
        return "f32"
    return "f16" if o_dtype == torch.float16 else "bf16"
