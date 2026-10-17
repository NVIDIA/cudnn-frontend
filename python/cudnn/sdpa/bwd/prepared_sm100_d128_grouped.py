# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped d128 declarations: native dK/dV partials followed by an FP32 fold."""

import re
from functools import partial
from types import SimpleNamespace

from cudnn.frost.compiled_cache import positional_entry
from .prepared import ATTRIBUTES, ROLES, BwdLaunchSpec, Operand, _same_geometry
from .d128_contract import validate_accumulation_scale
from .d128_grouped_contract import DENSE, THD, dense_workspace_layout, thd_workspace_layout, validate_dimensions, validate_dense_bounds, validate_thd_bounds
from .d128_thd_contract import KV_PORTS, ORIGIN_PORTS, Q_PORTS, validate_stats_span
from .prepared_sm100_d128 import _GuardedAccumulationEntry
from .prepared_sm100_d128_thd import THD_ATTRIBUTES, THD_ROLES, _GuardedEntry


def check_support(api):
    """Grouped declarations only; MHA retains its original prepared routes."""
    import torch

    from cudnn import _pybind_module
    from cudnn.frost.buffers import cutedsl_state
    from cudnn.frost.device import compute_capability, resolve_device

    def require(condition, reason):
        if not condition:
            raise NotImplementedError(f"{api._NAME}: grouped d128 {reason}")

    installed, version = cutedsl_state()
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version[1]) if installed and version and version[0] == "nvidia-cutlass-dsl" else None
    require(match is not None and tuple(map(int, match.groups())) >= (4, 8, 0), f"requires nvidia-cutlass-dsl >= 4.8.0; found {version}")
    require(api.q_desc.device.type == "cuda", "Q must reside on CUDA")
    cc = compute_capability(resolve_device(api.q_desc.device))
    require(tuple(cc) == (10, 0), f"requires SM100; got {cc}")
    require(getattr(getattr(_pybind_module, "_SdpaBwdBinder", None), "supports_zero_workspace", False), "requires prepared backward native binding support")
    require(not api.seq_q_lens_present and not api.seq_kv_lens_present, "dense padding flags are not supported; THD lengths use packed metadata")
    require(not api.causal_bottom_right, "bottom-right causal is not supported")
    require(api.window_size_left is None and api.window_size_right is None, "sliding windows and right-band widening are not supported")
    require(not api.deterministic, "use_deterministic_algorithm is not claimed")
    require(all(d is None for d in (api.sink_desc, api.dsink_desc, api.bias_desc, api.dbias_desc)), "sink / dSink and bias / dBias are not supported")
    require(api.tile_m is None and api.tile_n is None, "the kernel has fixed tile geometry")
    for role in ROLES[:9]:
        desc = getattr(api, role + "_desc")
        require(desc.ndim == 4 and all(int(n) > 0 for n in desc.shape), f"{role} requires positive rank-four dimensions")
        require(desc.device == api.q_desc.device, f"{role} must reside on {api.q_desc.device}")
    b, h, sq, d = map(int, api.q_desc.shape)
    bk, hk, sk, dk = map(int, api.k_desc.shape)
    require(d == dk == 128, "D_QK and D_V must equal 128")
    require(b == bk, "Q and KV must have equal batch counts")
    try:
        validate_dimensions(b, h, hk, sq, sk)
    except ValueError as error:
        require(False, str(error))
    require(api.q_desc.dtype == torch.bfloat16, "supports BF16 grouped attention only; FP16 requires wider dK/dV partial storage")
    for role in ROLES[:9]:
        desc = getattr(api, role + "_desc")
        s, heads = (sk, hk) if role in KV_PORTS else (sq, h)
        expected = (b, heads, s, 1 if role == "stats" else 128)
        require(tuple(desc.shape) == expected, f"{role} must have shape {expected}")
        require(desc.dtype == (torch.float32 if role == "stats" else api.q_desc.dtype), f"{role} must match its declared storage dtype")
    scale = 128**-0.5 if api.scale_softmax is None else float(api.scale_softmax)
    try:
        validate_accumulation_scale(scale)
    except ValueError as error:
        require(False, str(error))
    api.batch_size, api.h_q, api.h_kv = b, h, hk
    api.s_q_max, api.s_k_max = sq, sk
    api.head_dim_qk = api.head_dim_v = 128
    api.dtype, api.scale_softmax = api.q_desc.dtype, scale
    if api.thd:
        _check_thd(api, require)
    else:
        _check_dense(api, require)
    api._is_supported = True
    return True


def _check_dense(api, require):
    b, h, hk, sq, sk = api.batch_size, api.h_q, api.h_kv, api.s_q_max, api.s_k_max
    require(not api._thd_ro_spec, "ragged offsets require THD")
    try:
        validate_dense_bounds(b, h, hk, sq, sk)
    except ValueError as error:
        require(False, str(error))
    for role in ROLES[:9]:
        desc = getattr(api, role + "_desc")
        if role == "stats":
            require(desc.is_contiguous(), "dense Stats must be contiguous")
        else:
            s, heads = (sk, hk) if role in KV_PORTS else (sq, h)
            expected = (b, heads, s, 128), (s * heads * 128, 128, heads * 128, 1)
            require(_same_geometry((desc.shape, desc.stride), expected), f"{role} requires compact BSHD storage")
    api._grouped_workspace_regions, api._grouped_workspace_bytes = dense_workspace_layout(b, h, sq, sk)
    api._implementation = DENSE


def _check_thd(api, require):
    import torch

    from cudnn.sdpa.graph_analyzer import thd_stats_packing

    b, h, hk, sq, sk = api.batch_size, api.h_q, api.h_kv, api.s_q_max, api.s_k_max
    require(api.max_total_seq_len_q is not None and api.max_total_seq_len_kv is not None, "THD requires max_total_seq_len_q and max_total_seq_len_kv")
    require(api.max_total_seq_len_q > 0 and api.max_total_seq_len_kv > 0, "THD declared packed capacities must be positive")
    declared_q, declared_k = int(api.max_total_seq_len_q), int(api.max_total_seq_len_kv)
    tq, tk = min(b * sq, declared_q), min(b * sk, declared_k)
    ro_spec = api._thd_ro_spec
    require(not set(ro_spec) - set(ORIGIN_PORTS), "ragged offsets name an unknown port")
    tq_phys = declared_q if set(ro_spec) & Q_PORTS else tq
    tk_phys = declared_k if set(ro_spec) & KV_PORTS else tk
    try:
        validate_thd_bounds(b, h, hk, sq, sk, tq, tk, tq_phys, tk_phys)
    except ValueError as error:
        require(False, str(error))
    for role in ROLES[:9]:
        if role == "stats":
            continue
        heads = hk if role in KV_PORTS else h
        _, sh, ss, sd = map(int, getattr(api, role + "_desc").stride)
        require(ss == heads * 128 and (heads == 1 or sh == 128) and sd == 1, f"{role} requires compact packed BSHD rows")
    require(not (api.thd_stats_token_major and api.thd_stats_head_stride), "Stats head stride is head-major-only")
    stats_stride = tuple(map(int, api.stats_desc.stride))
    packing = thd_stats_packing(stats_stride[1], stats_stride[2], h)
    expected = "token_major" if api.thd_stats_token_major else "head_major"
    require(packing == expected or (h == 1 and stats_stride[2] == 1), "Stats strides disagree with the declared packed layout")
    pitch = 0 if api.thd_stats_token_major else int(api.thd_stats_head_stride or stats_stride[1])
    require(api.thd_stats_token_major or pitch >= tq_phys, "Stats head stride must cover the physical Q-token capacity")
    require(api.thd_stats_token_major or h == 1 or pitch == stats_stride[1], "Stats head stride must match its declaration")
    try:
        validate_stats_span(h, tq_phys if api.thd_stats_token_major else pitch)
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
        heads = hk if port in KV_PORTS else h
        token_stride = (h if api.thd_stats_token_major else 1) if port == "stats" else heads * 128
        origin_specs.append((row, multiplier, token_stride))
        origin_dtypes.append(str(dtype).split(".")[-1])
    api._thd_t_q, api._thd_t_kv = tq, tk
    api._thd_t_q_phys, api._thd_t_kv_phys = tq_phys, tk_phys
    api._thd_head_pitch = pitch
    api._thd_origins, api._thd_origin_dtypes = tuple(origin_specs), tuple(origin_dtypes)
    api._grouped_workspace_regions, api._grouped_workspace_bytes = thd_workspace_layout(b, h, tq, tk)
    api._implementation = THD


def compile_plan(api):
    import cutlass
    import torch

    from cudnn.frost.device import resolve_device

    b, h, hk = api.batch_size, api.h_q, api.h_kv
    geometry, operands = [], []
    for role in ROLES[:9]:
        desc = getattr(api, role + "_desc")
        if api.thd:
            if role == "stats":
                shape, strides = (
                    ((api._thd_t_q_phys, h), (h, 1))
                    if api.thd_stats_token_major
                    else ((1, h, api._thd_head_pitch), (h * api._thd_head_pitch, api._thd_head_pitch, 1))
                )
            else:
                tokens, heads = (api._thd_t_kv_phys, hk) if role in KV_PORTS else (api._thd_t_q_phys, h)
                shape, strides = (1, tokens, heads, 128), (tokens * heads * 128, heads * 128, 128, 1)
        elif role == "stats":
            shape, strides = tuple(desc.shape[:3]), tuple(desc.stride[:3])
            compact = (h * api.s_q_max, api.s_q_max, 1)
            strides = tuple(want if n == 1 else st for n, st, want in zip(shape, strides, compact))
        else:
            shape = tuple(desc.shape[i] for i in (0, 2, 1, 3))
            strides = tuple(desc.stride[i] for i in (0, 2, 1, 3))
            _, s, heads, d = shape
            compact = (s * heads * d, heads * d, d, 1)
            strides = tuple(want if n == 1 else st for n, st, want in zip(shape, strides, compact))
        span = 1 + sum((int(n) - 1) * int(st) for n, st in zip(shape, strides))
        if span * desc.dtype.itemsize > 2**63 - 1:
            raise NotImplementedError(f"{api._NAME}: grouped {role} exceeds the signed pointer range")
        geometry.append((shape, strides))
        operands.append(Operand(str(desc.dtype).split(".")[-1], shape, strides, span, 4 if role == "stats" else 16, desc.dtype.itemsize))
    dtype = cutlass.BFloat16 if api.dtype == torch.bfloat16 else cutlass.Float16
    device = resolve_device(api.q_desc.device)
    regions, required = api._grouped_workspace_regions, api._grouped_workspace_bytes
    if required != api.scratch_workspace_bytes():
        raise RuntimeError("grouped d128 compiled workspace disagrees with its advertised size")
    if api.thd:
        from .kernels.sm100.prepared_d128_grouped_thd_host import compile_host

        operands.extend(Operand("int32", (b,), (1,), b, 4, 4, (b, b + 1)) for _ in range(2))
        for name in api._thd_origin_dtypes:
            width = 8 if name == "int64" else 4
            operands.append(None if name is None else Operand(name, (b + 1,), (1,), b + 1, width, width))
        args = ((b, h, hk, bool(api.thd_stats_token_major)), api._thd_origins, api._thd_origin_dtypes, dtype, api.is_causal, 100)
    else:
        from .kernels.sm100.prepared_d128_grouped_host import compile_host

        args = (tuple(geometry), regions, dtype, api.is_causal, 100)
    with torch.cuda.device(device):
        entry = compile_host(*args)
    fn = positional_entry(entry)
    if fn is None:
        raise NotImplementedError("grouped d128 requires a positional tvm-ffi entry")
    if api.thd:
        prefix = (api._thd_t_q, api._thd_t_kv, api._thd_t_q_phys, api._thd_t_kv_phys, api.s_q_max, api.s_k_max, api._thd_head_pitch)
        fn = _GuardedEntry(partial(fn, *prefix), operands, required)
    else:
        fn = _GuardedAccumulationEntry(fn, operands, required)
    owner = SimpleNamespace(
        entry=entry, workspace_bytes=required, workspace_regions=regions, implementation=api._implementation, partial_dtype=str(api.dtype).split(".")[-1]
    )
    return BwdLaunchSpec(
        owner,
        fn,
        tuple(operands),
        required,
        device,
        api.scale_softmax,
        api._NAME,
        length_form=api.thd,
        roles=THD_ROLES if api.thd else ROLES[:9],
        attributes=THD_ATTRIBUTES if api.thd else ATTRIBUTES[:9],
        scale_log2=False,
        native_binding=True,
    )
