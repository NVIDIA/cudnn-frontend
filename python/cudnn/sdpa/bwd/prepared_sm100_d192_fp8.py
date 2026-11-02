# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage-free plan construction for dense D_QK = 192 / D_V = 128 per-tensor FP8 backward."""

from types import SimpleNamespace

from cudnn.frost.compiled_cache import positional_entry
from cudnn.sdpa.bwd.prepared import ROLES, BwdLaunchSpec, Operand
from .kernels.sm100.prepared_d128_accum_fp8_host import FP8_AMAX
from .prepared_sm100_d128_fp8 import ATTRIBUTES_FP8_D128, ROLES_FP8_D128, _GuardedEntry


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
    scalar = Operand("float32", (1,), (1,), 1, 4, 4)
    operands += [scalar] * 12
    requested = tuple(name in api.amax_requested for name in FP8_AMAX)
    operands += [scalar if flag else None for flag in requested]
    device = resolve_device(api.q_desc.device)
    cc = compute_capability(device)
    grad_dtype = {torch.float8_e4m3fn: cutlass.Float8E4M3FN, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}[api.grad_dtype]
    from .kernels.sm100.prepared_d192_fp8_host import compile_host, workspace_layout

    regions, required = workspace_layout(api.batch_size, api.h_q, api.s_q_max)

    with torch.cuda.device(device):
        entry = compile_host(tuple(geometry), regions, grad_dtype, api.is_causal, cc[0] * 10 + cc[1], requested)
    fn = positional_entry(entry)
    if fn is None:
        raise NotImplementedError(f"{api._NAME} requires a positional tvm-ffi entry")
    if required != api.scratch_workspace_bytes():
        raise RuntimeError("dense d192 fp8 compiled workspace disagrees with its advertised size")
    return BwdLaunchSpec(
        SimpleNamespace(entry=entry, workspace_bytes=required, workspace_regions=regions, implementation=api._implementation),
        _GuardedEntry(fn, operands, required),
        tuple(operands),
        required,
        device,
        api.scale_softmax,
        api._NAME,
        roles=ROLES_FP8_D128,
        attributes=ATTRIBUTES_FP8_D128,
        scale_log2=False,
        native_binding=False,
    )
