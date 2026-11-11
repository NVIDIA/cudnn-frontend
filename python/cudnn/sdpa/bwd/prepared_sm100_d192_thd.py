# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed d192 declarations and per-call binding, separate from the dense ABI."""

import math
import re
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

from cudnn.frost.compiled_cache import positional_entry
from .prepared import ATTRIBUTES, ROLES, BwdLaunchSpec, Operand
from .d192_thd_contract import (
    IMPLEMENTATION,
    KV_PORTS,
    ORIGIN_PORTS,
    Q_PORTS,
    RAGGED_ROLES,
    validate_bounds,
    validate_launch,
    validate_stats_span,
    workspace_layout,
)

THD_ROLES = ROLES[:11] + RAGGED_ROLES
THD_ATTRIBUTES = ATTRIBUTES[:11] + RAGGED_ROLES


def check_support(api):
    """Require positive declared storage capacities; device sequence lengths may be zero."""
    import torch

    from cudnn import _pybind_module
    from cudnn.frost.buffers import cutedsl_state
    from cudnn.frost.device import compute_capability, resolve_device
    from cudnn.sdpa.graph_analyzer import thd_stats_packing
    from .d128_contract import ACCUMULATION_CCS, validate_accumulation_scale

    def require(condition, reason):
        if not condition:
            raise NotImplementedError(f"{api._NAME}: THD {reason}")

    installed, version = cutedsl_state()
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version[1]) if installed and version and version[0] == "nvidia-cutlass-dsl" else None
    require(match is not None and tuple(map(int, match.groups())) >= (4, 8, 0), f"requires nvidia-cutlass-dsl >= 4.8.0; found {version}")
    require(api.q_desc.device.type == "cuda", "Q must reside on CUDA")
    cc = compute_capability(resolve_device(api.q_desc.device))
    require(tuple(cc) in ACCUMULATION_CCS, f"requires SM100 or SM103; got {cc}")
    require(getattr(getattr(_pybind_module, "_SdpaBwdBinder", None), "supports_zero_workspace", False), "requires prepared backward native binding support")
    require(not api.seq_q_lens_present and not api.seq_kv_lens_present, "lengths use the packed metadata path, not dense padding flags")
    require(not api.causal_bottom_right, "bottom-right causal is not supported")
    require(api.window_size_left is None and api.window_size_right is None, "sliding windows and right-band widening are not supported")
    require(not api.deterministic, "use_deterministic_algorithm is not claimed")
    require(all(d is None for d in (api.sink_desc, api.dsink_desc, api.bias_desc, api.dbias_desc)), "sink / dSink and bias / dBias are not supported")
    require(api.tile_m is None and api.tile_n is None, "the kernel has fixed tile geometry")
    for role in ROLES[:9]:
        desc = getattr(api, role + "_desc")
        require(desc.ndim == 4 and all(int(n) > 0 for n in desc.shape), f"{role} requires positive rank-four envelope dimensions")
        require(desc.device == api.q_desc.device, f"{role} must reside on {api.q_desc.device}")
    b, h, sq, d = map(int, api.q_desc.shape)
    bk, hk, sk, dk = map(int, api.k_desc.shape)
    require(d == dk == 192 and api.v_desc.shape[3] == 128, "D_QK must equal 192 and D_V 128")
    require(b == bk and h == hk, "Q and KV require equal batch and head counts (MHA)")
    require(api.q_desc.dtype in (torch.bfloat16, torch.float16), "only BF16 and FP16 are supported")
    require(api.max_total_seq_len_q is not None and api.max_total_seq_len_kv is not None, "requires max_total_seq_len_q and max_total_seq_len_kv")
    require(api.max_total_seq_len_q > 0 and api.max_total_seq_len_kv > 0, "declared packed capacities must be positive")
    declared_q, declared_k = int(api.max_total_seq_len_q), int(api.max_total_seq_len_kv)
    tq, tk = min(b * sq, declared_q), min(b * sk, declared_k)
    ro_spec = api._thd_ro_spec
    require(not set(ro_spec) - set(ORIGIN_PORTS), "ragged offsets name an unknown port")
    tq_phys = declared_q if set(ro_spec) & Q_PORTS else tq
    tk_phys = declared_k if set(ro_spec) & KV_PORTS else tk
    try:
        validate_bounds(b, h, sq, sk, tq, tk, tq_phys, tk_phys)
    except ValueError as error:
        require(False, str(error))
    for role in ROLES[:9]:
        desc = getattr(api, role + "_desc")
        if role == "stats":
            require(tuple(desc.shape) == (b, h, sq, 1) and desc.dtype == torch.float32, "Stats must be FP32 (B, H, S_q_max, 1)")
        else:
            s = sk if role in KV_PORTS else sq
            width = 128 if role in ("v", "o", "do", "dv") else 192
            require(tuple(desc.shape) == (b, h, s, width) and desc.dtype == api.q_desc.dtype, f"{role} must have its declared MHA envelope and Q's dtype")
            _, sh, ss, sd = map(int, desc.stride)
            require(ss == h * width and (h == 1 or sh == width) and sd == 1, f"{role} requires compact packed BSHD rows")
    require(not (api.thd_stats_token_major and api.thd_stats_head_stride), "Stats head stride is head-major-only")
    stats_stride = tuple(map(int, api.stats_desc.stride))
    packing = thd_stats_packing(stats_stride[1], stats_stride[2], h)
    expected_packing = "token_major" if api.thd_stats_token_major else "head_major"
    # H=1 permits both interpretations; both address the same rows when the head is never stepped.
    require(packing == expected_packing or (h == 1 and stats_stride[2] == 1), "Stats strides disagree with the declared packed layout")
    head_pitch = 0 if api.thd_stats_token_major else int(api.thd_stats_head_stride or stats_stride[1])
    require(api.thd_stats_token_major or head_pitch >= tq_phys, "Stats head stride must cover the physical Q-token capacity")
    require(api.thd_stats_token_major or h == 1 or head_pitch == stats_stride[1], "Stats head stride must match its declaration")
    try:
        validate_stats_span(h, tq_phys if api.thd_stats_token_major else head_pitch)
    except ValueError as error:
        require(False, str(error))
    origin_specs, origin_dtypes = [], []
    for row, port in enumerate(ORIGIN_PORTS):
        if port not in ro_spec:
            origin_specs.append(None)
            origin_dtypes.append(None)
            continue
        item = ro_spec[port]
        require(isinstance(item, (tuple, list)) and len(item) == 2, f"{port} offsets require (dtype, multiplier)")
        dtype, multiplier = item
        require(dtype in (torch.int32, torch.int64), f"{port} ragged offset must be INT32 or INT64")
        require(type(multiplier) is int and 0 < multiplier <= 2**63 - 1, f"{port} ragged offset multiplier must be a positive Int64 integer")
        width = 128 if port in ("v", "o", "do", "dv") else 192
        token_stride = (h if api.thd_stats_token_major else 1) if port == "stats" else h * width
        origin_specs.append((row, multiplier, token_stride))
        origin_dtypes.append(str(dtype).split(".")[-1])
    scale = 192**-0.5 if api.scale_softmax is None else float(api.scale_softmax)
    try:
        validate_accumulation_scale(scale)
    except ValueError as error:
        require(False, str(error))
    api.batch_size, api.h_q, api.h_kv = b, h, h
    api.s_q_max, api.s_k_max = sq, sk
    api.head_dim_qk, api.head_dim_v = 192, 128
    api.dtype, api.scale_softmax = api.q_desc.dtype, scale
    api._thd_t_q, api._thd_t_kv = tq, tk
    api._thd_t_q_phys, api._thd_t_kv_phys = tq_phys, tk_phys
    api._thd_head_pitch = head_pitch
    api._thd_origins, api._thd_origin_dtypes = tuple(origin_specs), tuple(origin_dtypes)
    regions, required = workspace_layout(b, h, tq)
    api._thd_workspace_regions, api._thd_workspace_bytes = regions, required
    api._implementation = IMPLEMENTATION
    api._is_supported = True
    return True


class _GuardedEntry:
    """Shared by graph and standalone; metadata cannot alias a gradient write."""

    def __init__(self, entry, operands, workspace_bytes):
        self.entry = entry
        self.spans = tuple(0 if op is None else op.span * op.itemsize for op in operands)
        self.workspace_bytes = workspace_bytes

    def __call__(self, *frame):
        if len(frame) != 24:
            raise ValueError("d192 THD prepared frame must contain twenty operands and four runtime slots")
        form = frame[22]
        if form not in (0, 1, 2, 3):
            raise ValueError("d192 THD length form must encode the Q and KV prefix bits")
        spans = tuple(span + (4 if i == 9 and form & 1 or i == 10 and form & 2 else 0) for i, span in enumerate(self.spans))
        validate_launch(frame[:20], spans, frame[20], self.workspace_bytes, frame[21])
        self.entry(*frame)


def compile_plan(api):
    import cutlass
    import torch

    from cudnn.frost.device import compute_capability, resolve_device
    from .kernels.sm100.prepared_d192_thd_host import compile_host

    b, h = api.batch_size, api.h_q
    operands = []
    for role in ROLES[:9]:
        desc = getattr(api, role + "_desc")
        if role == "stats":
            if api.thd_stats_token_major:
                shape, strides = (api._thd_t_q_phys, h), (h, 1)
            else:
                shape, strides = (1, h, api._thd_head_pitch), (h * api._thd_head_pitch, api._thd_head_pitch, 1)
            alignment = 4
        else:
            tokens = api._thd_t_kv_phys if role in KV_PORTS else api._thd_t_q_phys
            width = 128 if role in ("v", "o", "do", "dv") else 192
            shape, strides = (1, tokens, h, width), (tokens * h * width, h * width, width, 1)
            alignment = 16
        span = 1 + sum((n - 1) * st for n, st in zip(shape, strides))
        if span * desc.dtype.itemsize > 2**63 - 1:
            raise NotImplementedError(f"{api._NAME}: THD {role} exceeds the signed pointer range")
        operands.append(Operand(str(desc.dtype).split(".")[-1], shape, strides, span, alignment, desc.dtype.itemsize))
    operands.extend(Operand("int32", (b,), (1,), b, 4, 4, (b, b + 1)) for _ in range(2))
    for name in api._thd_origin_dtypes:
        width = 8 if name == "int64" else 4
        operands.append(None if name is None else Operand(name, (b + 1,), (1,), b + 1, width, width))
    device = resolve_device(api.q_desc.device)
    cc = compute_capability(device)
    dtype = cutlass.BFloat16 if api.dtype == torch.bfloat16 else cutlass.Float16
    config = (b, h, bool(api.thd_stats_token_major))
    with torch.cuda.device(device):
        entry = compile_host(config, api._thd_origins, api._thd_origin_dtypes, dtype, api.is_causal, cc[0] * 10 + cc[1])
    fn = positional_entry(entry)
    if fn is None:
        raise NotImplementedError("d192 THD requires a positional tvm-ffi entry")
    prefix = (api._thd_t_q, api._thd_t_kv, api._thd_t_q_phys, api._thd_t_kv_phys, api.s_q_max, api.s_k_max, api._thd_head_pitch)
    fn = _GuardedEntry(partial(fn, *prefix), operands, api._thd_workspace_bytes)
    owner = SimpleNamespace(entry=entry, workspace_bytes=api._thd_workspace_bytes, workspace_regions=api._thd_workspace_regions, implementation=IMPLEMENTATION)
    return BwdLaunchSpec(
        owner,
        fn,
        tuple(operands),
        api._thd_workspace_bytes,
        device,
        api.scale_softmax,
        api._NAME,
        length_form=True,
        roles=THD_ROLES,
        attributes=THD_ATTRIBUTES,
        scale_log2=False,
        native_binding=True,
    )


def execute_standalone(api, tensors, workspace, current_stream, scale, optional):
    import torch

    from cudnn._device import ensure_current_context
    from cudnn.sdpa.fwd.prepared import facts_of_tensor
    from .prepared import execute

    allowed = {"seq_q_lens", "seq_kv_lens", "thd_ragged_offsets"}
    if any(value is not None for name, value in optional.items() if name not in allowed):
        raise ValueError("d192 THD received optional operands absent from this specialization")
    bound = dict(optional.get("thd_ragged_offsets") or {})
    if set(bound) != set(api._thd_ro_spec):
        raise ValueError("d192 THD ragged offsets must bind exactly the declared ports")
    values = tensors + (optional.get("seq_q_lens"), optional.get("seq_kv_lens")) + tuple(bound.get(port) for port in ORIGIN_PORTS)
    facts = dict(zip(THD_ROLES, map(facts_of_tensor, values)))
    spec = api._prepared
    geometry = []
    for i, (role, op) in enumerate(zip(THD_ROLES, spec.operands)):
        actual = facts[role]
        geom = None
        if i < 9 and op is not None:
            geom = (op.shape, op.strides)
            if role == "stats" and actual is not None and actual.contiguous and len(actual.shape) == 1 and actual.numel == math.prod(op.shape):
                geom = None
        elif i >= 11 and op is not None and actual is not None:
            if not actual.contiguous or actual.numel != api.batch_size + 1:
                raise ValueError(f"d192 THD {role} must be contiguous with B+1 elements")
        geometry.append(geom)
    ws = facts_of_tensor(workspace)
    if ws is None or ws.dtype != "uint8" or not ws.contiguous or ws.span < spec.workspace_bytes or ws.device != (2, spec.device_index):
        raise ValueError(f"d192 THD requires {spec.workspace_bytes} bytes of contiguous uint8 workspace on CUDA device {spec.device_index}")
    stream = int(torch.cuda.current_stream(api.q_desc.device).cuda_stream if current_stream is None else current_stream)
    context = nullcontext() if torch.cuda.current_device() == spec.device_index else torch.cuda.device(spec.device_index)
    with context:
        ensure_current_context(stream, spec.device_index)
        execute(spec, facts, ws.ptr, stream, scale=scale, geometry=tuple(geometry))
