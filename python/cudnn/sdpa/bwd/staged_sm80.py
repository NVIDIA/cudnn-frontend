# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared launches after the existing SM80 wrapper's staging operations."""

from dataclasses import dataclass, replace
import math
from types import SimpleNamespace

import torch

from cudnn.sdpa.fwd.api_dsl import _torch_stream_context, ws_align
from cudnn.sdpa.fwd.prepared import BufferFacts, facts_of_tensor
from .prepared import ROLES, execute
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
    # Auxiliary accumulators stay in the chain workspace. The old wrapper's
    # optional/strided gradient outputs are copied from those same accumulators
    # after launch; no extra output staging or auxiliary copy kernel is added.
    plan.dbias_desc = plan.dsink_desc = None
    if api._has_bias and plan.bias_desc is None:
        shape = (api._bias_batch, api.h_q, api.s_q_max, api.s_k_max)
        strides = tuple(math.prod(shape[i + 1 :]) for i in range(4))
        plan.bias_desc = replace(api.q_desc, shape=shape, stride=strides, stride_order=(3, 2, 1, 0), dtype=torch.float32 if api._bias_is_fp32 else api.dtype)
    regions, core_bytes = workspace_regions(plan)
    return StagedLayout(plan, tuple(copies), offset, regions, offset + core_bytes)


def compile_staged(api, d64_module):
    plan = api._staged_layout.declaration
    plan._kmod, plan._params = api._kmod, api._params
    return build_spec(plan, d64_module, staged=True)


def _view_from_region(workspace, region, dtype):
    offset, shape, strides = region
    return workspace[offset : offset + math.prod(shape) * dtype.itemsize].view(dtype).view(shape)


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
    else:
        stats = api._checked_lse_view(stats.squeeze(-1) if stats.ndim == 4 else stats)
    for role in ("seq_q", "seq_kv"):
        tensor = original[role]
        if tensor is not None and (tensor.dtype != torch.int32 or not tensor.is_contiguous() or tensor.numel() != api.batch_size):
            raise ValueError(f"{role}_lens must contain B contiguous int32 lengths")
        if api.thd and tensor is None:
            raise ValueError(f"SM80 bwd THD: {role}_lens is required")
    if stream is None:
        stream = torch.cuda.current_stream(device).cuda_stream
    cooked = dict(original, stats=stats, dbias=None, dsink=None, rope=None)
    cooked_facts = dict(original_facts, stats=facts_of_tensor(stats), dbias=None, dsink=None, rope=None)
    with _torch_stream_context(stream, device):
        # The byte workspace may be an aligned slice and may have an odd extra
        # byte. Views use its current storage origin, with only the planned
        # extent reinterpreted; the plan never retains this tensor or pointer.
        typed_workspace = workspace[: layout.workspace_bytes].view(api.dtype)
        storage_offset = typed_workspace.storage_offset()
        dtype_name = spec.operands[0].dtype
        for role, offset, shape in layout.copies:
            b, h, s, d = shape
            strides = (s * h * d, d, h * d, 1)
            view = typed_workspace.as_strided(shape, strides, storage_offset + offset // 2)
            cooked[role] = view
            cooked_facts[role] = BufferFacts(ws.ptr + offset, dtype_name, ws.device, math.prod(shape), shape, strides)
            if role in ROLES[:5]:
                width = original[role].shape[-1]
                view[..., :width].copy_(original[role])
                if width != d:
                    view[..., width:].zero_()
        if rope_freqs is not None:
            # Existing standalone RoPE preprocessing; the graph never admits
            # this feature. Preserve its table values and launch ordering.
            rf = rope_freqs.to(dtype=torch.float32, device=device).reshape(rope_freqs.shape[0], -1)
            d2 = api.flavor_d_qk // 2
            if rf.shape[0] != api._rope_max_s or rf.shape[1] < d2:
                raise ValueError("rope_freqs must match the compiled row count and cover d_qk//2")
            angles = rf[:, :d2]
            cooked["rope"] = torch.stack([angles.cos(), angles.sin()], dim=-1).contiguous()
            cooked_facts["rope"] = facts_of_tensor(cooked["rope"])
        # Original I/O geometry was checked above. Every replacement view has
        # the prepared geometry by construction, so no second tensor metadata
        # walk is needed. The common binder still checks pointer/operand bounds.
        execute(spec, cooked_facts, ws.ptr + layout.staging_bytes, int(stream), scale=scale)
        for role, offset, shape in layout.copies:
            if role in ("dq", "dk", "dv"):
                original[role].copy_(cooked[role][..., : original[role].shape[-1]])
        for role, region in (("dbias", layout.regions[5]), ("dsink", layout.regions[6])):
            if original[role] is not None and region is not None:
                core_workspace = workspace[layout.staging_bytes :]
                src = _view_from_region(core_workspace, region, torch.float32)
                original[role].copy_(src.view(original[role].shape))
