# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen Python frame oracle; never imported by production execution."""

import math
from cudnn.sdpa.fwd.prepared_sm80 import ROLES


def _same_geometry(actual, expected):
    # Graph Stats and vector operands may carry extra singleton dimensions.
    return tuple((n, st) for n, st in zip(*actual) if n != 1) == tuple((n, st) for n, st in zip(*expected) if n != 1)


def bind(spec, facts, stream_int, *, scale=None, overridden=None, raw_storage=False):
    frame = []
    for name, op in zip(ROLES, spec.operands):
        f = facts.get(name)
        if op is None:
            if f is not None:
                raise ValueError(f"sdpa_fwd_sm80: {name} was not compiled into this specialization")
            frame.append(None)
            continue
        if f is None:
            raise ValueError(f"sdpa_fwd_sm80: {name} is required by this specialization")
        if f.device not in ((2, spec.device_index), (-1, -1)):
            raise ValueError(f"sdpa_fwd_sm80: {name} must be on CUDA device {spec.device_index}")
        if f.dtype and f.dtype != op.dtype:
            raise ValueError(f"sdpa_fwd_sm80: {name} must be {op.dtype}; got {f.dtype}")
        if not f.ptr or f.ptr % op.alignment:
            raise ValueError(f"sdpa_fwd_sm80: {name} base address must be {op.alignment}-byte aligned")
        if f.span >= 0 and f.span < op.span:
            raise ValueError(f"sdpa_fwd_sm80: {name} backing storage is too small for the declared strides")
        if not raw_storage and name in ("seq_q", "seq_kv", "sink") and f.shape:
            if not f.contiguous or f.numel != math.prod(op.shape):
                raise ValueError(f"sdpa_fwd_sm80: {name} must be contiguous with {math.prod(op.shape)} elements")
        elif not raw_storage and name == "stats" and f.shape:
            # Preserve the standalone Stats storage contract: compact plans may
            # bind a flat contiguous buffer; strided plans use declared strides.
            if f.numel != math.prod(op.shape) or (op.contiguous and not f.contiguous):
                raise ValueError("sdpa_fwd_sm80: Stats must match the declared element count and storage layout")
        elif not raw_storage and name == "bias" and f.shape:
            # The standalone ABI broadcasts the first contiguous [H,SQ,SKV]
            # bias plane; retain that view semantics without constructing it.
            if len(f.shape) != 4 or f.shape[0] < 1 or not _same_geometry((f.shape[1:], f.strides[1:]), (op.shape[1:], op.strides[1:])):
                raise ValueError("sdpa_fwd_sm80: bias must have a contiguous [H,SQ,SKV] first plane")
        elif (not raw_storage or (overridden is not None and name in overridden)) and f.shape:
            if not _same_geometry((f.shape, f.strides), (op.shape, op.strides)):
                raise ValueError(f"sdpa_fwd_sm80: {name} runtime geometry must match this fixed forward plan")
        frame.append(f.ptr)
    scale = spec.scale if scale is None else float(scale)
    if scale == 0:
        raise ValueError("attn_scale = 0 is not supported on this kernel (#1435)")
    frame.extend((scale * math.log2(math.e), 1.0 / scale, stream_int))
    return frame
