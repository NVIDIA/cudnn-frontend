# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan existing SM80 conversions around the native pointer host."""

from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass, replace
from functools import lru_cache, partial
import math

import torch

from .prepared import BufferFacts, facts_of_tensor
from .prepared_sm80 import ROLES, bind, build_spec


@dataclass(frozen=True)
class StagedLaunch:
    core: object
    operands: tuple
    regions: tuple
    workspace_bytes: int
    copies: tuple
    rope: object


def _layout(api):
    from .api_dsl import ws_align

    compact = copy(api)
    operands, regions = [], []
    offset = 0
    for role in ("q", "k", "v", "o"):
        desc = getattr(api, role + "_desc")
        b, h, s, d = desc.shape
        padded_d = (d + 7) // 8 * 8 if role in ("q", "k") else api.flavor_d_v
        operands.append((role, tuple(desc.shape), desc.dtype))
        direct = padded_d == d and all(n == 1 or st % 8 == 0 for n, st in zip(desc.shape[:-1], desc.stride[:-1]))
        if direct:
            continue
        shape, strides = (b, h, s, padded_d), (s * h * padded_d, padded_d, h * padded_d, 1)
        setattr(compact, role + "_desc", replace(desc, shape=shape, stride=strides, stride_order=(3, 1, 2, 0)))
        count = math.prod(shape)
        facts = BufferFacts(0, str(desc.dtype).split(".")[-1], (2, int(desc.device.index or 0)), count, shape, strides)
        regions.append((role, offset, facts, d))
        offset += ws_align(count * 2)
    return compact, tuple(operands), tuple(regions), offset


def workspace_bytes(api):
    return _layout(api)[3]


@lru_cache(maxsize=128)
def _compile_core(device_index, *args):
    from .kernels.sm80.prepared_host import compile_host

    with torch.cuda.device(device_index):
        return compile_host(*args)


def compile_plan(api):
    from cudnn.frost.compiled_cache import positional_entry
    from cudnn.sdpa.rope_table_sm80 import compile_plan as compile_rope
    from .kernels.sm80.staged_copy import compile_gather
    from .kernels.staged_copy import compile_copy

    compact, operands, regions, required = _layout(api)
    core = build_spec(compact, compiler=partial(_compile_core, int(api.q_desc.device.index or 0)))
    copies = []
    for output in (False, True):
        group = tuple(r for r in regions if (r[0] == "o") == output)
        if not group:
            copies.append(None)
            continue
        shapes = tuple((f.shape[0], f.shape[2], f.shape[1], d) for _, _, f, d in group)
        artifact = (
            compile_copy(shapes, (2,) * len(group)) if output else compile_gather(tuple((*shape, f.shape[3]) for shape, (_, _, f, _) in zip(shapes, group)))
        )
        fn = positional_entry(artifact)
        if fn is None:
            raise NotImplementedError("SM80 staged copy requires a positional tvm-ffi entry")
        copies.append((artifact, fn, group))
    rope = compile_rope(api._rope_max_s, api.flavor_d_qk // 2, api.q_desc.device) if api._rope_max_s else None
    return StagedLaunch(core, operands, regions, required, tuple(copies), rope)


def _copy(entry, frame, stream_int):
    entry[1](*frame, stream_int)


def execute(api, tensors, workspace, stream, scale, *, rope_freqs=None):
    from cudnn._device import ensure_current_context
    from cudnn._torch_stream import _raw_current_stream
    from cuda.bindings import driver
    from .api_dsl import _torch_stream_context
    from .prepared import _covering

    staged = api._sm80_copy_spec
    device = api.q_desc.device
    if (workspace is None and staged.workspace_bytes) or (workspace is not None and (workspace.device != device or not workspace.is_contiguous())):
        raise ValueError("SM80 staged forward requires contiguous workspace on the Q device")
    base = api._scratch_base(workspace, "SM80 staged forward", staged.workspace_bytes) if staged.workspace_bytes else 0
    facts = {role: facts_of_tensor(t) for role, t in zip(ROLES, tensors)}
    for role, shape, dtype in staged.operands:
        f = facts[role]
        if f is None or f.shape != shape or f.dtype != str(dtype).split(".")[-1] or f.device != (2, staged.core.device_index):
            raise ValueError(f"SM80 staged {role} must match compiled shape {shape}, dtype {dtype}, device {device}")
        if not f.ptr or f.ptr % 2:
            raise ValueError(f"SM80 staged {role} requires an aligned live address")
        if role == "o" and not _covering(f.shape, f.strides):
            raise ValueError("SM80 staged output must have non-overlapping strides")
    for role, t in zip(ROLES, tensors):
        f = facts[role]
        if f is not None and base < f.ptr + f.span * t.element_size() and f.ptr < base + staged.workspace_bytes:
            raise ValueError(f"SM80 staged workspace overlaps {role}")
    frames = []
    for output, entry in enumerate(staged.copies):
        if entry is None:
            frames.append(None)
            continue
        srcs, dsts, src_strides, dst_strides = [], [], [], []
        for role, offset, template, _ in entry[2]:
            original, temp = facts[role], template._replace(ptr=base + offset)
            src, dst = (temp, original) if output else (original, temp)
            srcs.append(src.ptr)
            dsts.append(dst.ptr)
            src_strides.append(tuple(src.strides[i] for i in (0, 2, 1, 3)))
            dst_strides.append(tuple(dst.strides[i] for i in (0, 2, 1, 3)))
            facts[role] = temp
        frames.append((tuple(srcs), tuple(dsts), tuple(src_strides), tuple(dst_strides)) if output else (tuple(srcs), tuple(dsts), tuple(src_strides)))
    device_context = nullcontext() if torch.cuda.current_device() == device.index else torch.cuda.device(device)
    with device_context:
        context = nullcontext() if stream is None else _torch_stream_context(stream, device, verify_current=True)
        if stream is None:
            raw = _raw_current_stream(torch, device)
            stream = driver.CUstream(torch.cuda.current_stream(device).cuda_stream if raw is None else raw)
        stream_int = int(stream)
        ensure_current_context(stream_int, device.index)
        with context:
            # Preserve the standalone-only angle-table conversion on its
            # consuming stream; the graph rows do not admit RoPE. Keep the
            # temporary table alive through the core launch.
            rope = None
            if rope_freqs is not None:
                from cudnn.sdpa.rope_table_sm80 import prepare as prepare_rope

                rope = prepare_rope(staged.rope, rope_freqs, device, stream_int)
                facts["rope"] = facts_of_tensor(rope)
            # All core/auxiliary validation precedes the first workspace write.
            frame = bind(staged.core, facts, stream_int, scale=scale)
            if frames[0] is not None:
                _copy(staged.copies[0], frames[0], stream_int)
            staged.core.fn(*frame)
            if frames[1] is not None:
                _copy(staged.copies[1], frames[1], stream_int)
