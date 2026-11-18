# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage-free plan construction for dense d128 per-tensor FP8 backward."""

from functools import partial
from types import SimpleNamespace

from cudnn.frost.compiled_cache import positional_entry
from cudnn.sdpa.bwd.prepared import ATTRIBUTES, ROLES, BwdLaunchSpec, Operand
from .d128_contract import validate_launch, workspace_layout
from .kernels.sm100.prepared_d128_accum_fp8_host import FP8_AMAX, FP8_SCALARS

ROLES_FP8_D128 = ROLES[:9] + FP8_SCALARS + FP8_AMAX
ATTRIBUTES_FP8_D128 = ATTRIBUTES[:9] + FP8_SCALARS + FP8_AMAX
_ALIGN = 256


def grouped_workspace_layout(batch, heads_q, sq, skv, d=128):
    """Aligned-payload-relative regions for the grouped chain: the FP32 dQ accumulator and delta over the rounded S_q, then one bf16 dK and dV partial per Q head."""
    (dq_region, delta_region), _ = workspace_layout(batch, heads_q, sq, d)
    rounded = (sq + 127) // 128 * 128
    base = (4 * batch * rounded * heads_q * d + 4 * batch * heads_q * rounded + _ALIGN - 1) // _ALIGN * _ALIGN
    part_bytes = 2 * batch * skv * heads_q * d
    part_shape = (batch, skv, heads_q, d)
    part_strides = (skv * heads_q * d, heads_q * d, d, 1)
    dk_region = (base, part_shape, part_strides)
    dv_region = (base + (part_bytes + _ALIGN - 1) // _ALIGN * _ALIGN, part_shape, part_strides)
    total = dv_region[0] + part_bytes + _ALIGN - 16
    return (dq_region, delta_region, dk_region, dv_region), total


class _GuardedEntry:

    def __init__(self, entry, operands, workspace_bytes):
        self.entry = entry
        self.spans = tuple(op.span * op.itemsize for op in operands[:9])
        self.workspace_bytes = workspace_bytes

    def __call__(self, *frame):
        validate_launch(frame[:9], self.spans, frame[-3], self.workspace_bytes, frame[-2])
        self.entry(*frame)


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
    operands += [scalar] * len(FP8_SCALARS)
    requested = tuple(name in api.amax_requested for name in FP8_AMAX)
    operands += [scalar if flag else None for flag in requested]
    device = resolve_device(api.q_desc.device)
    cc = compute_capability(device)
    grad_dtype = {torch.float8_e4m3fn: cutlass.Float8E4M3FN, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}[api.grad_dtype]
    ratio = api.h_q // api.h_kv
    window = 0 if api.window_size_left is None else int(api.window_size_left) + 1
    if window == 0 and api.head_dim_qk == 128:
        from .kernels.sm100.prepared_d192_fp8_host import compile_host as compile_pair_host, workspace_layout as pair_workspace_layout

        regions, required = pair_workspace_layout(api.batch_size, api.h_q, api.s_q_max)
        compile = partial(compile_pair_host, tuple(geometry), regions, grad_dtype, api.is_causal, cc[0] * 10 + cc[1], requested, 128, 128, ratio)
    elif ratio > 1:
        from .kernels.sm100.prepared_d128_accum_fp8_grouped_host import compile_host as compile_grouped_host

        regions, required = grouped_workspace_layout(api.batch_size, api.h_q, api.s_q_max, api.s_k_max, api.head_dim_qk)
        compile = partial(compile_grouped_host, tuple(geometry), regions, grad_dtype, api.is_causal, cc[0] * 10 + cc[1], requested, ratio, window)
    else:
        from .kernels.sm100.prepared_d128_accum_fp8_host import compile_host as compile_accumulation_host

        regions, required = workspace_layout(api.batch_size, api.h_q, api.s_q_max, api.head_dim_qk)
        compile = partial(compile_accumulation_host, tuple(geometry), regions, grad_dtype, api.is_causal, cc[0] * 10 + cc[1], requested, window)
    with torch.cuda.device(device):
        entry = compile()
    fn = positional_entry(entry)
    if fn is None:
        raise NotImplementedError(f"{api._NAME} requires a positional tvm-ffi entry")
    if required != api.scratch_workspace_bytes():
        raise RuntimeError("dense d128 fp8 compiled workspace disagrees with its advertised size")
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
