# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage-free plan construction for dense d128 backward."""

from types import SimpleNamespace

from cudnn.frost.compiled_cache import positional_entry
from cudnn.sdpa.bwd.prepared import ATTRIBUTES, ROLES, BwdLaunchSpec, Operand
from .d128_contract import ACCUMULATION, validate_launch, workspace_layout


class _GuardedAccumulationEntry:
    """Validate current pointer relationships before the first kernel launch."""

    def __init__(self, entry, operands, workspace_bytes):
        self.entry = entry
        self.spans = tuple(op.span * op.itemsize for op in operands)
        self.workspace_bytes = workspace_bytes

    def __call__(self, q, k, v, o, do, lse, dq, dk, dv, workspace, scale, stream):
        validate_launch((q, k, v, o, do, lse, dq, dk, dv), self.spans, workspace, self.workspace_bytes, scale)
        self.entry(q, k, v, o, do, lse, dq, dk, dv, workspace, scale, stream)


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
        # Singleton pitches address no second element; normalize them for TMA.
        strides = tuple(want if n == 1 else st for n, st, want in zip(shape, strides, compact))
        geometry.append((shape, strides))
        span = 1 + sum((int(n) - 1) * int(st) for n, st in zip(shape, strides))
        operands.append(Operand(str(desc.dtype).split(".")[-1], shape, strides, span, alignment, desc.dtype.itemsize))
    device = resolve_device(api.q_desc.device)
    cc = compute_capability(device)
    dtype = cutlass.BFloat16 if api.dtype == torch.bfloat16 else cutlass.Float16
    regions, required = (), 0
    if api._implementation == ACCUMULATION:
        from .kernels.sm100.prepared_d128_accum_host import compile_host

        regions, required = workspace_layout(api.batch_size, api.h_q, api.s_q_max)
        compile_args = (tuple(geometry), regions, dtype, api.is_causal, cc[0] * 10 + cc[1])
    else:
        from .kernels.sm100.prepared_d128_host import compile_host

        compile_args = (tuple(geometry), dtype, api.is_causal, cc[0] * 10 + cc[1])
    # compile_cached owns device/context-aware reuse; no argument-only outer cache.
    with torch.cuda.device(device):
        entry = compile_host(*compile_args)
    fn = positional_entry(entry)
    if fn is None:
        raise NotImplementedError("sdpa_bwd_sm100_d128 requires a positional tvm-ffi entry")
    if required != api.scratch_workspace_bytes():
        raise RuntimeError("dense d128 compiled workspace disagrees with its advertised size")
    if api._implementation == ACCUMULATION:
        fn = _GuardedAccumulationEntry(fn, operands, required)
    return BwdLaunchSpec(
        SimpleNamespace(entry=entry, workspace_bytes=required, workspace_regions=regions, implementation=api._implementation),
        fn,
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
