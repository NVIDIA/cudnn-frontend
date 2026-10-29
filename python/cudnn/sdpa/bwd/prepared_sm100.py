# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan-time geometry and workspace for the large-head backward pointer host."""

import math
from contextlib import nullcontext
from types import SimpleNamespace

from cudnn.frost.compiled_cache import positional_entry
from cudnn.frost.tile_dsl.thd import THD_BWD_META_WORDS
from .prepared import BwdLaunchSpec, Operand, ROLES


def native_io_strides(desc):
    """Canonical TMA strides; unused B/H/S axes carry no storage contract."""
    from cudnn.sdpa.fwd.config_sm100 import canonical_bhsd_strides
    from cudnn.sdpa.graph_analyzer import dense_layout_ok

    shape, strides = tuple(desc.shape), tuple(desc.stride)
    if len(shape) != 4 or len(strides) != 4 or strides[-1] != 1 or not dense_layout_ok(shape, strides):
        return None
    strides = canonical_bhsd_strides(shape, strides)
    if any(int(st) <= 0 or int(st) % 8 for st in strides[:-1]):
        return None
    return strides


def native_io_layout(desc):
    """The declared layout can be addressed directly by the TMA chain."""
    return native_io_strides(desc) is not None


def compile_plan(api, stage2, mm_lo, mm_hi):
    import cutlass
    import torch

    from cudnn.frost.device import compute_capability, resolve_device
    from .api_dsl import _sm100_device_clusters
    from .kernels.sm100.prepared_host import Params, compile_host
    from .kernels.sm120.prepared_host import compact

    b, h, hk, d = api.batch_size, api.h_q, api.h_kv, api.head_dim_qk
    tq = api._t_q_cap if api.thd else api.s_q_max
    tk = api._t_kv_cap if api.thd else api.s_k_max
    actual_batch = 1 if api.thd else b
    geometry = []
    operands = []
    for role in ROLES[:9]:
        desc = getattr(api, role + "_desc")
        shape, strides = tuple(desc.shape), tuple(desc.stride)
        if role == "stats":
            if api.thd:
                geom = compact((tq, h)) if api._thd_lse_token_major else compact((1, h, api._thd_lse_head_stride or tq))
            else:
                geom = (shape[:3], strides[:3])
            alignment = 4
        else:
            if api.thd:
                tokens = tk if role in ("k", "v", "dk", "dv") else tq
                shape = (1, shape[1], tokens, shape[3])
                strides = (max(tokens, 1) * strides[2], *strides[1:])
            else:
                strides = native_io_strides(desc)
                if strides is None:
                    raise ValueError(f"SM100 backward {role} requires native aligned non-overlapping strides")
            geom = tuple(shape[i] for i in (0, 2, 1, 3)), tuple(strides[i] for i in (0, 2, 1, 3))
            alignment = 16
        geometry.append(geom)
        shape, strides = geom
        span = 0 if not math.prod(shape) else 1 + sum((n - 1) * st for n, st in zip(shape, strides))
        operands.append(Operand(str(desc.dtype).split(".")[-1], shape, strides, span, alignment, desc.dtype.itemsize))
    operands.extend([Operand("int32", (b,), (1,), b, 4, 4, (b, b + 1)) if api.thd else None for _ in range(2)])
    offset = 0

    def region(shape, itemsize):
        nonlocal offset
        shape, strides = compact(shape)
        result = offset, shape, strides
        offset += (math.prod(shape) * itemsize + 127) // 128 * 128
        return result

    rows = api._ws_rows_cap if api.thd else api._sq_pad
    delta = region((actual_batch, h, (tq + 127) // 128 * 128), 4)
    scores = region((actual_batch, api._qh_chunk, rows, api._skv_pad), 2)
    dscores = region((actual_batch, api._qh_chunk, rows, api._skv_pad), 2)
    meta = region((THD_BWD_META_WORDS(b) if api.thd else b,), 4)
    desc2 = region((4 * 16 if api.thd else 1,), 8)
    desc3 = region(((b + 1) * 16,), 8) if api.thd else None
    dkw = region((actual_batch, tk, h, d), 2) if hk != h else None
    dvw = region((actual_batch, tk, h, d), 2) if hk != h else None
    if offset != api.scratch_workspace_bytes():
        raise RuntimeError("SM100 backward prepared workspace differs from its advertised requirement")
    # Stage 2's q span per cluster: CLUSTER_Q_ROWS on the 2x2 twin, TILE_M * CTA_MMA on the role split (both 256).
    gran = getattr(stage2.CFG, "CLUSTER_Q_ROWS", stage2.CFG.TILE_M * stage2.CFG.CTA_MMA)
    units = max(1, min(((tq + gran - 1) // gran + b) * h, _sm100_device_clusters(api.q_desc.device, stage2.CFG.CGA_M))) if api.thd else 0
    # The dQ rendering's B head group rides in `params` (the compile key and the traced host both carry it), copied off
    # the record `compile()` rendered: the group = one dQ launch per chunk, 1 = per member (`prepared_host._dq_launches`).
    params = Params(b, h, hk, d, api.s_q_max, api.s_k_max, rows, api._skv_pad, api._qh_chunk, api.thd, api._zero_ws, units, gran, int(api._dq_b_head_group))
    dtype = cutlass.BFloat16 if api.dtype == torch.bfloat16 else cutlass.Float16
    major, minor = compute_capability(resolve_device(api.q_desc.device))
    sm = major * 10 + minor
    regions = delta, scores, dscores, meta, desc2, desc3, dkw, dvw
    geometry = tuple(geometry)
    key = repr((tuple(mod.FROST_SOURCE_DIGEST for mod in (stage2, mm_lo, mm_hi)), params, geometry, regions, sm))
    # The row's name is the spec's, the artifact symbol's (``frost_sdpa_bwd_sm100_prepared`` on the SM100 row, the cc 10.7 d512
    # row's own on ``sdpa_bwd_sm107_d512``) and the launch spec's; the chain is the same for both.
    name = api._NAME
    entry = compile_host(stage2._host, mm_lo._host, mm_hi._host, params, geometry, regions, dtype, sm, key, symbol=f"frost_{name}_prepared")
    owner = SimpleNamespace(entry=entry, workspace_bytes=offset)
    fn = positional_entry(entry)
    if fn is None:
        raise NotImplementedError("SM100 backward requires a positional tvm-ffi entry")
    return BwdLaunchSpec(owner, fn, tuple(operands), offset, int(api.q_desc.device.index or 0), api.scale_softmax, name, True, native_binding=True)


def execute_standalone(api, tensors, workspace, current_stream, scale):
    import torch
    from cudnn._device import ensure_current_context
    from cudnn.sdpa.fwd.prepared import facts_of_tensor
    from .prepared import execute

    spec = api._prepared
    ws = facts_of_tensor(workspace)
    if ws is None or ws.dtype != "uint8" or not ws.contiguous or ws.span < spec.workspace_bytes or ws.device != (2, spec.device_index):
        raise ValueError(f"{spec.name} requires {spec.workspace_bytes} bytes of contiguous uint8 workspace on CUDA device {spec.device_index}")
    if current_stream is None:
        current_stream = torch.cuda.current_stream(api.q_desc.device).cuda_stream
    facts = dict(zip(ROLES, map(facts_of_tensor, tensors)))
    geometry = []
    for i, op in enumerate(spec.operands):
        geom = None
        if i < 9:
            if i == 5:
                # The legacy tensor ABI reshaped contiguous Stats to its fixed
                # dense or packed layout. Preserve flat-storage bindings too.
                if facts["stats"] is not None and (not facts["stats"].contiguous or facts["stats"].numel != math.prod(op.shape)):
                    raise ValueError(f"SM100 backward Stats must be contiguous with {math.prod(op.shape)} elements")
            else:
                geom = tuple(op.shape[j] for j in (0, 2, 1, 3)), tuple(op.strides[j] for j in (0, 2, 1, 3))
        geometry.append(geom)
    stream = int(current_stream)
    device_context = nullcontext() if torch.cuda.current_device() == spec.device_index else torch.cuda.device(spec.device_index)
    with device_context:
        ensure_current_context(stream, spec.device_index)
        execute(spec, facts, ws.ptr, stream, scale=scale, geometry=geometry)
