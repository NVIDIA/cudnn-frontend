# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared launches after the existing SM80 wrapper's staging operations."""

from contextlib import nullcontext
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace

import torch

from cudnn.sdpa.fwd.api_dsl import _torch_stream_context, ws_align
from cudnn.sdpa.fwd.prepared import BufferFacts, facts_of_tensor
from .prepared import ROLES, bind
from .prepared_sm80 import build_spec
from .kernels.sm80.prepared_host import workspace_regions


@dataclass(frozen=True)
class StagedLayout:
    declaration: object
    copies: tuple
    staging_bytes: int
    regions: tuple
    workspace_bytes: int


def layout_for(api):
    """Describe only the copies the previous tensor executor already made."""
    plan = SimpleNamespace(**vars(api))
    plan.head_dim_qk, plan.head_dim_v = api.flavor_d_qk, api.flavor_d_v
    plan._thd_token_strides = dict(api._thd_token_strides)
    copies, offset = [], 0
    for role in ROLES[:5] + ROLES[6:9]:
        desc = getattr(api, role + "_desc")
        is_kv = role in ("k", "v", "dk", "dv")
        heads = api.h_kv if is_kv else api.h_q
        seq = api.s_k_max if is_kv else api.s_q_max
        dim = api.flavor_d_v if role in ("v", "o", "do", "dv") else api.flavor_d_qk
        padded = desc.shape[-1] != dim
        compact_strides = (seq * heads * desc.shape[-1], desc.shape[-1], heads * desc.shape[-1], 1)
        compact = all(n == 1 or st == want for n, st, want in zip(desc.shape, desc.stride, compact_strides))
        if api.thd:
            # Packed output casts/folds already truncate the flavor padding
            # and preserve the capacity tail on device; never copy them back.
            copy = padded and role in ROLES[:5]
            batch, seq = 1, api._t_kv_cap if is_kv else api._t_q_cap
        else:
            copy = padded or not compact or role == "dq" or (role in ("dk", "dv") and api.h_q != api.h_kv)
            batch = api.batch_size
        if copy:
            shape = (batch, heads, seq, dim)
            strides = (seq * heads * dim, dim, heads * dim, 1)
            cooked = replace(desc, shape=shape, stride=strides, stride_order=(3, 1, 2, 0))
            setattr(plan, role + "_desc", cooked)
            if api.thd:
                plan._thd_token_strides[role] = heads * dim
            copies.append((role, offset, shape))
            offset += ws_align(math.prod(shape) * 2)
    # Auxiliary accumulators stay in the chain workspace. Prepared casts
    # replace the old wrapper's copies to optional/strided gradient outputs;
    # no extra output staging or conversion domain is introduced.
    plan.dbias_desc = plan.dsink_desc = None
    if api._has_bias and plan.bias_desc is None:
        shape = (api._bias_batch, api.h_q, api.s_q_max, api.s_k_max)
        strides = tuple(math.prod(shape[i + 1 :]) for i in range(4))
        plan.bias_desc = replace(api.q_desc, shape=shape, stride=strides, stride_order=(3, 2, 1, 0), dtype=torch.float32 if api._bias_is_fp32 else api.dtype)
    regions, core_bytes = workspace_regions(plan)
    return StagedLayout(plan, tuple(copies), offset, regions, offset + core_bytes)


def compile_staged(api, d64_module):
    from cudnn.sdpa.rope_table_sm80 import compile_plan as compile_rope

    plan = api._staged_layout.declaration
    plan._kmod, plan._params = api._kmod, api._params
    spec = build_spec(plan, d64_module, staged=True)
    api._staged_copies = _compile_copies(api)
    api._staged_rope = compile_rope(api._rope_max_s, api.flavor_d_qk // 2, api.q_desc.device) if api._has_rope else None
    return spec


def run_staged(api, tensors, workspace, stream, scale, rope_freqs):
    """Bind current staged pointers to one compiled chain; retain no buffers."""
    layout, spec = api._staged_layout, api._staged_prepared
    original = dict(zip(ROLES, tensors))
    device = original["q"].device
    ws = facts_of_tensor(workspace)
    if ws is None or ws.dtype != "uint8" or not ws.contiguous or ws.span < layout.workspace_bytes or ws.device != (2, spec.device_index):
        raise ValueError(f"sdpa_bwd_sm80 requires {layout.workspace_bytes} bytes of contiguous uint8 caller workspace")
    if ws.ptr % 16:
        raise ValueError("sdpa_bwd_sm80 workspace must be 16-byte aligned")
    # Check overlap before any staging write. Shape checks below use metadata
    # only, including THD lengths; no device value is read on the host.
    original_facts = {role: facts_of_tensor(tensor) for role, tensor in original.items()}
    for role, tensor in (*original.items(), ("rope", rope_freqs)):
        f = facts_of_tensor(tensor) if role == "rope" else original_facts[role]
        if f is not None:
            if role == "rope" and tensor.device.type == "cpu":
                continue  # Preserve the standalone wrapper's CPU angle-table input.
            if f.device != (2, spec.device_index):
                raise ValueError(f"sdpa_bwd_sm80: {role} must be on Q's device")
            width = tensor.element_size()
            if ws.ptr < f.ptr + f.span * width and f.ptr < ws.ptr + layout.workspace_bytes:
                raise ValueError(f"sdpa_bwd_sm80: caller workspace overlaps {role}")
    for role in ROLES[:5] + ROLES[6:9]:
        tensor, desc = original[role], getattr(api, role + "_desc")
        f = original_facts[role]
        shape, strides = tuple(desc.shape), tuple(desc.stride)
        if api.thd:
            tokens = api._t_kv_cap if role in ("k", "v", "dk", "dv") else api._t_q_cap
            shape = (1, shape[1], tokens, shape[3])
            strides = (tokens * api._thd_token_strides[role], shape[3], api._thd_token_strides[role], 1)
        if tensor.dtype != api.dtype or f.shape != shape or any(n > 1 and actual != expected for n, actual, expected in zip(shape, f.strides, strides)):
            raise ValueError(f"sdpa_bwd_sm80: {role} must match the declared shape, dtype and strides")
    stats = original["stats"]
    if api.thd:
        op = spec.operands[5]
        if stats.dtype != torch.float32 or tuple(stats.shape) != op.shape or tuple(stats.stride()) != op.strides:
            raise ValueError("THD stats must match the declared packed layout")
    elif api._has_rope:
        stats = api._checked_lse_view(stats.squeeze(-1) if stats.ndim == 4 else stats)
    elif stats.dtype != torch.float32 or stats.numel() != api.batch_size * api.h_q * api.s_q_max or (api._lse_stride is None and not stats.is_contiguous()):
        raise ValueError("stats must match the declared dtype, element count and storage layout")
    if not api.thd and not api._has_rope:
        # Preserve the standalone Stats storage-binding contract without
        # rebuilding its declared as_strided view. A flat runtime container can
        # expose fewer logical elements than its backing allocation covers.
        f, op = original_facts["stats"], spec.operands[5]
        if f.span < op.span:
            available = (stats.untyped_storage().nbytes() - stats.storage_offset() * 4) // 4
            if available < op.span:
                raise ValueError("sdpa_bwd_sm80: stats backing storage is too small for the declared strides")
            f = f._replace(span=op.span)
            original_facts["stats"] = f
        if ws.ptr < f.ptr + op.span * 4 and f.ptr < ws.ptr + layout.workspace_bytes:
            raise ValueError("sdpa_bwd_sm80: caller workspace overlaps stats")
    for role in ("seq_q", "seq_kv"):
        tensor = original[role]
        if tensor is not None and (tensor.dtype != torch.int32 or not tensor.is_contiguous() or tensor.numel() != api.batch_size):
            raise ValueError(f"{role}_lens must contain B contiguous int32 lengths")
        if api.thd and tensor is None:
            raise ValueError(f"SM80 bwd THD: {role}_lens is required")
    # These outputs copy from accumulators after launch and are deliberately
    # absent from the pointer ABI. Validate them before any staging write.
    for role, region in (("dbias", layout.regions[5]), ("dsink", layout.regions[6])):
        f, desc = original_facts[role], getattr(api, role + "_desc")
        if f is None:
            if desc is not None:
                raise ValueError(f"sdpa_bwd_sm80: {role} is required by this specialization")
            continue
        if region is None:
            raise ValueError(f"sdpa_bwd_sm80: {role} was not compiled into this specialization")
        if f.numel != math.prod(region[1]):
            raise ValueError(f"sdpa_bwd_sm80: {role} must contain {math.prod(region[1])} elements")
        expected_shape = tuple(desc.shape) if desc is not None else tuple(region[1])
        shape = f.shape
        if desc is None:
            # Legacy wrappers can omit singleton axes (notably (H,) dSink).
            shape = tuple(n for n in shape if n != 1)
            expected_shape = tuple(n for n in expected_shape if n != 1)
        if shape != expected_shape:
            raise ValueError(f"sdpa_bwd_sm80: {role} must match the declared logical shape {expected_shape}")
        allowed = (desc.dtype,) if desc is not None else (torch.float32, api.dtype) if role == "dbias" else (torch.float32,)
        if original[role].dtype not in allowed:
            raise ValueError(f"sdpa_bwd_sm80: {role} dtype must match its declaration or accumulation output")
    if stream is None:
        stream = torch.cuda.current_stream(device).cuda_stream
    cooked_facts = dict(original_facts, stats=facts_of_tensor(stats) if api._has_rope else original_facts["stats"], dbias=None, dsink=None, rope=None)
    if not api._has_rope:
        _run_copies(api, original_facts, cooked_facts, ws.ptr, int(stream), scale)
        return
    with _torch_stream_context(stream, device):
        if rope_freqs is not None:
            from cudnn.sdpa.rope_table_sm80 import prepare as prepare_rope

            # Existing standalone RoPE preprocessing; the graph never admits
            # this feature. Preserve its table values and launch ordering.
            rope = prepare_rope(api._staged_rope, rope_freqs, device, stream)
            cooked_facts["rope"] = facts_of_tensor(rope)
        # The angle table retains its existing numerical preprocessing and is
        # allocated on this launch stream. All data copies use the same
        # prepared entries, validation and alias ordering as non-RoPE calls.
        _run_copies(api, original_facts, cooked_facts, ws.ptr, int(stream), scale)


def _entry(artifact):
    from cudnn.frost.compiled_cache import positional_entry

    fn = positional_entry(artifact)
    if fn is None:
        raise NotImplementedError("SM80 backward staging requires a positional tvm-ffi entry")
    return artifact, fn


def _compile_copies(api):
    from cudnn.sdpa.fwd.kernels.sm80.staged_copy import compile_gather
    from cudnn.sdpa.fwd.kernels.staged_copy import compile_copy
    from .kernels.sm80.staged_copy import compile_cast

    copies = []
    for output in (False, True):
        group = tuple(c for c in api._staged_layout.copies if (c[0] in ("dq", "dk", "dv")) == output)
        if not group:
            copies.append(None)
            continue
        shapes = tuple((b, s, h, getattr(api, role + "_desc").shape[-1]) for role, _, (b, h, s, _) in group)
        if api.thd:
            shapes = tuple((b, -2 if c[0] in ("k", "v") else -1, h, d) for (b, s, h, d), c in zip(shapes, group))
        artifact = (
            compile_copy(shapes, (2,) * len(group)) if output else compile_gather(tuple((*shape, c[2][-1]) for shape, c in zip(shapes, group)), packed=api.thd)
        )
        # Disjoint outputs share one launch. Runtime aliases retain the old
        # ordered copy-back semantics through prepared single-role entries.
        serial = tuple(_entry(compile_copy((shape,), (2,))) for shape in shapes) if output and len(group) > 1 else ()
        copies.append((*_entry(artifact), group, serial))
    auxiliary = []
    for role, region in (("dbias", api._staged_layout.regions[5]), ("dsink", api._staged_layout.regions[6])):
        if region is None:
            continue
        desc = getattr(api, role + "_desc")
        dtypes = (desc.dtype,) if desc is not None else (torch.float32, api.dtype) if role == "dbias" else (torch.float32,)
        shape = tuple(n for n in region[1] if n != 1)
        shape = (1,) * (4 - len(shape)) + shape
        entries = tuple((str(dtype).split(".")[-1], *_entry(compile_cast(shape, str(dtype).split(".")[-1]))) for dtype in dtypes)
        auxiliary.append((role, api._staged_layout.staging_bytes + region[0], entries))
    return tuple(copies), tuple(auxiliary)


def _copy(entry, frame, stream):
    entry[1](*frame, stream)


def _run_copies(api, original, cooked, base, stream, scale):
    from cudnn._device import ensure_current_context
    from cudnn.sdpa.fwd.prepared import _covering

    spec, layout = api._staged_prepared, api._staged_layout
    copies, auxiliary = api._staged_copies
    frames = []
    for output, entry in enumerate(copies):
        if entry is None:
            frames.append(None)
            continue
        srcs, dsts, src_strides, dst_strides = [], [], [], []
        for role, offset, shape in entry[2]:
            b, h, s, d = shape
            f = original[role]
            if not f.ptr or f.ptr % 2:
                raise ValueError(f"sdpa_bwd_sm80: {role} requires an aligned live address")
            if output and not _covering(f.shape, f.strides):
                raise ValueError(f"sdpa_bwd_sm80: {role} must have non-overlapping strides")
            temp = BufferFacts(base + offset, f.dtype, f.device, math.prod(shape), shape, (s * h * d, d, h * d, 1))
            cooked[role] = temp
            src, dst = (temp, f) if output else (f, temp)
            srcs.append(src.ptr)
            dsts.append(dst.ptr)
            src_strides.append(tuple(src.strides[i] for i in (0, 2, 1, 3)))
            dst_strides.append(tuple(dst.strides[i] for i in (0, 2, 1, 3)))
        frames.append((tuple(srcs), tuple(dsts), tuple(src_strides), tuple(dst_strides)) if output else (tuple(srcs), tuple(dsts), tuple(src_strides)))
    if api.thd and frames[0] is not None:
        frames[0] += (api._t_q_cap, api._t_kv_cap)
    outputs = [original[role] for role, _, _ in copies[1][2]] if copies[1] is not None else []
    serial_scatter = any(a.ptr < b.ptr + b.span * 2 and b.ptr < a.ptr + a.span * 2 for i, a in enumerate(outputs) for b in outputs[i + 1 :])
    aux_frames = []
    for role, offset, entries in auxiliary:
        f = original[role]
        if f is None:
            continue
        width = 4 if f.dtype == "float32" else 2
        if not f.ptr or f.ptr % width or not _covering(f.shape, f.strides):
            raise ValueError(f"sdpa_bwd_sm80: {role} requires aligned non-overlapping storage")
        strides = tuple(st for n, st in zip(f.shape, f.strides) if n != 1)
        strides = (0,) * (4 - len(strides)) + strides
        entry = next(e for e in entries if e[0] == f.dtype)
        aux_frames.append((entry[2], base + offset, f.ptr, strides))
    # All original operands, auxiliary outputs and core bindings are checked
    # before the first gather writes workspace. No tensor or pointer is cached.
    frame = bind(spec, cooked, base + layout.staging_bytes, stream, scale=scale)
    device_context = nullcontext() if torch.cuda.current_device() == spec.device_index else torch.cuda.device(spec.device_index)
    with device_context:
        ensure_current_context(stream, spec.device_index)
        if frames[0] is not None:
            _copy(copies[0], frames[0], stream)
        spec.fn(*frame)
        if frames[1] is not None:
            if serial_scatter:
                for i, entry in enumerate(copies[1][3]):
                    _copy(entry, tuple((leaves[i],) for leaves in frames[1]), stream)
            else:
                _copy(copies[1], frames[1], stream)
        for fn, src, dst, strides in aux_frames:
            fn(src, dst, strides, stream)
