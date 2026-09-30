# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared angle preprocessing shared by standalone SM80 forward/backward."""

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RopeTable:
    rows: int
    width: int
    artifact: object
    entry: object


def compile_plan(rows, width, device):
    from cudnn.frost.compiled_cache import positional_entry
    from .fwd.kernels.sm80.rope_table import compile_table

    with torch.cuda.device(device):
        artifact = compile_table(rows, width)
    entry = positional_entry(artifact)
    if entry is None:
        raise NotImplementedError("SM80 RoPE table requires a positional tvm-ffi entry")
    return RopeTable(rows, width, artifact, entry)


def prepare(plan, frequencies, device, stream):
    """Called inside the consumer's stream context; retain no buffers."""
    from cudnn._device import ensure_current_context

    # Preserve CPU tables, dtype conversion and flattening accepted by the
    # standalone wrappers. Graph engines do not admit angle-table RoPE.
    context = nullcontext() if torch.cuda.current_device() == device.index else torch.cuda.device(device)
    with context:
        ensure_current_context(int(stream), device.index)
        angles = frequencies.to(dtype=torch.float32, device=device).reshape(frequencies.shape[0], -1)
        if angles.shape[0] != plan.rows or angles.shape[1] < plan.width:
            detail = (
                f"rope_freqs last dim ({angles.shape[1]}) must be >= d_qk//2 ({plan.width})"
                if angles.shape[1] < plan.width
                else f"rope_freqs rows ({angles.shape[0]}) must equal the compiled rope_max_s ({plan.rows})"
            )
            raise ValueError(f"rope_freqs must match the compiled row count and cover d_qk//2: {detail}")
        table = torch.empty((plan.rows, plan.width, 2), dtype=torch.float32, device=device)
        plan.entry(angles.data_ptr(), table.data_ptr(), tuple(angles.stride()), int(stream))
    return table
