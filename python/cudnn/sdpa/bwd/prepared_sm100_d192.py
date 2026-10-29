# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage-free plan construction for dense D_QK = 192 / D_V = 128 backward."""

from types import SimpleNamespace

from cudnn.frost.compiled_cache import positional_entry
from cudnn.sdpa.bwd.prepared import ATTRIBUTES, ROLES, BwdLaunchSpec, Operand
from .d128_contract import ACCUMULATION, workspace_layout
from .prepared_sm100_d128 import _GuardedAccumulationEntry


def compile_plan(api):
    import cutlass
    import torch

    from cudnn.frost.device import compute_capability, resolve_device

    geometry, operands = [], []
    for role in ROLES[:9]:
        desc = getattr(api, role + "_desc")
        if role == "stats":
            b, h, s, _ = desc.shape
            shape, strides = (b, h, s), tuple(desc.stride[:3])
            compact = (h * s, s, 1)
            alignment = 4
        else:
            b, h, s, d = desc.shape
            sb, sh, ss, sd = desc.stride
            shape, strides = (b, s, h, d), (sb, ss, sh, sd)
            compact = (s * h * d, h * d, d, 1)
            alignment = 16
        strides = tuple(want if n == 1 else st for n, st, want in zip(shape, strides, compact))
        geometry.append((shape, strides))
        span = 1 + sum((int(n) - 1) * int(st) for n, st in zip(shape, strides))
        operands.append(Operand(str(desc.dtype).split(".")[-1], shape, strides, span, alignment, desc.dtype.itemsize))
    device = resolve_device(api.q_desc.device)
    cc = compute_capability(device)
    dtype = cutlass.BFloat16 if api.dtype == torch.bfloat16 else cutlass.Float16
    regions, required = workspace_layout(api.batch_size, api.h_q, api.s_q_max, api.head_dim_qk)
    from .kernels.sm100.prepared_d192_accum_host import compile_host

    with torch.cuda.device(device):
        entry = compile_host(tuple(geometry), regions, dtype, api.is_causal, cc[0] * 10 + cc[1])
    fn = positional_entry(entry)
    if fn is None:
        raise NotImplementedError(f"{api._NAME} requires a positional tvm-ffi entry")
    if required != api.scratch_workspace_bytes():
        raise RuntimeError("dense d192 compiled workspace disagrees with its advertised size")
    return BwdLaunchSpec(
        SimpleNamespace(entry=entry, workspace_bytes=required, workspace_regions=regions, implementation=ACCUMULATION),
        _GuardedAccumulationEntry(fn, operands, required),
        tuple(operands),
        required,
        device,
        api.scale_softmax,
        api._NAME,
        roles=ROLES[:9],
        attributes=ATTRIBUTES[:9],
        scale_log2=False,
        native_binding=True,
    )
