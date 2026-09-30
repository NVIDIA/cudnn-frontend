# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan-time geometry, workspace regions and launch spec for the SM107 d=256 backward pointer hosts.

The two rows (``sdpa_bwd_sm107``, ``sdpa_bwd_sm107_fp8``) join the prepared-launch contract of ``bwd/prepared.py``:
``compile_plan`` turns an adapter's fixed plan facts into a ``BwdLaunchSpec`` whose ``fn`` is the positional tvm-ffi
entry of ONE compiled artifact (``kernels/sm107/prepared_host.py``) that runs the whole chain from device pointers.
Per call, ``bind()`` validates every operand against its ``Operand`` and hands the artifact a flat pointer frame; the
graph plan binds the normalized variant pack (``PreparedBwdLaunch``), the standalone adapter binds torch tensors
(``execute_standalone``).  Nothing here touches torch on the execute path.
"""

import math
from types import SimpleNamespace

from cudnn.frost.compiled_cache import positional_entry
from cudnn.sdpa.fwd.api_dsl import ws_align
from .prepared import ATTRIBUTES, BwdLaunchSpec, Operand, ROLES

# The half row binds the nine tensor operands.  The fp8 row appends the twelve scalar descales / scales of
# ``sdpa_fp8_backward`` and the four requested-only amax outputs; their role names ARE the ``SdpaBinding`` field names.
ROLES_F16 = ROLES[:9]
ATTRIBUTES_F16 = ATTRIBUTES[:9]
FP8_SCALARS = (
    "descale_q",
    "descale_k",
    "descale_v",
    "descale_s",
    "scale_s",
    "descale_o",
    "descale_dO",
    "descale_dP",
    "scale_dQ",
    "scale_dK",
    "scale_dV",
    "scale_dP",
)
FP8_AMAX = ("amax_dQ", "amax_dK", "amax_dV", "amax_dP")
ROLES_FP8 = ROLES[:9] + FP8_SCALARS + FP8_AMAX
ATTRIBUTES_FP8 = ATTRIBUTES[:9] + FP8_SCALARS + FP8_AMAX

# Workspace region slots the hosts index (``prepared_host.R_*``), by ``_scratch_plan`` name.
_REGION_SLOTS_F16 = ("delta", "ds_ws", "seq_kv", "desc_words", "q_pad", "do_pad", "lse_pad", "k_pad", "v_pad", "dv_part", "dk_part", "dk_fold", "dv_fold")
_REGION_SLOTS_FP8 = (
    "delta",
    "ds_ws",
    "seq_kv",
    "desc_words",
    "q_pad",
    "do_pad",
    "lse_pad",
    "k_pad",
    "v_pad",
    "dv_part",
    "dk_part",
    "dq_ws",
    "q_bf16",
    "k_bf16",
    "amax_scratch",
)


def _dtype_name(torch_dtype) -> str:
    return str(torch_dtype).split(".")[-1]


def _tensor_operands(api):
    """``(geometry, operands)`` of the nine tensor roles: the kernels' compact ``[B, S, H, D]`` view of the declared
    logical-BHSD / BSHD-physical descriptors (a permute), and the contiguous ``[B, H, S_q]`` Stats."""
    geometry, operands = [], []
    for role in ROLES[:9]:
        desc = getattr(api, role + "_desc")
        shape, strides = tuple(int(x) for x in desc.shape), tuple(int(x) for x in desc.stride)
        if role == "stats":
            geom, alignment = (shape[:3], strides[:3]), 4
        else:
            geom, alignment = (tuple(shape[i] for i in (0, 2, 1, 3)), tuple(strides[i] for i in (0, 2, 1, 3))), 16
        geometry.append(geom)
        shape, strides = geom
        span = 0 if not math.prod(shape) else 1 + sum((n - 1) * st for n, st in zip(shape, strides))
        operands.append(Operand(_dtype_name(desc.dtype), shape, strides, span, alignment, desc.dtype.itemsize))
    return tuple(geometry), operands


def _regions(api, slots):
    """The workspace carve, in ``_scratch_plan`` order, as ``(offset, shape, strides)`` per host slot (None = not carved);
    the running 128-B-aligned offset must reproduce ``scratch_workspace_bytes()`` exactly."""
    offset, by_name = 0, {}
    for name, shape, dtype in api._scratch_shapes():
        shape = tuple(int(x) for x in shape)
        strides = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
        by_name[name] = (offset, shape, strides)
        offset += ws_align(math.prod(shape) * dtype.itemsize)
    if offset != api.scratch_workspace_bytes():
        raise RuntimeError(f"{api._NAME}: prepared workspace layout ({offset} B) differs from its advertised requirement ({api.scratch_workspace_bytes()} B)")
    unknown = set(by_name) - set(slots)
    if unknown:
        raise RuntimeError(f"{api._NAME}: scratch plan names without a host slot: {sorted(unknown)}")
    return tuple(by_name.get(name) for name in slots), offset


def _config(api):
    return (
        api.batch_size,
        api.h_q,
        api.h_kv,
        api.head_dim_qk,
        api.s_q_max,
        api.s_k_max,
        api._sq_pad,
        api._skv_pad,
        api._b_chunk,
        api._qh_chunk,
        bool(api._zero_ws),
        api.dtype.itemsize,
        api._bpe_ds,  # the dS workspace's bytes per element (fp8 row: 1 = e4m3, 2 = the bf16 twin)
    )


def _sm(api) -> int:
    from cudnn.frost.device import compute_capability, resolve_device

    major, minor = compute_capability(resolve_device(api.q_desc.device))
    return major * 10 + minor


def _dsl_dtype(torch_dtype):
    import cutlass

    return {"bfloat16": cutlass.BFloat16, "float16": cutlass.Float16, "float8_e4m3fn": cutlass.Float8E4M3FN}[_dtype_name(torch_dtype)]


def compile_plan(api, main, mm_dk, mm_dq):
    """The half row's spec.  ``main`` / ``mm_dk`` / ``mm_dq`` are the loaded templates (their ``_host`` functions are baked into the
    artifact; their ``FROST_SOURCE_DIGEST`` keys it together with the plan's geometry and carve)."""
    from .kernels.sm107.prepared_host import compile_host_f16

    geometry, operands = _tensor_operands(api)
    regions, offset = _regions(api, _REGION_SLOTS_F16)
    config = _config(api)
    sm = _sm(api)
    dtype = _dsl_dtype(api.dtype)
    key = repr((tuple(mod.FROST_SOURCE_DIGEST for mod in (main, mm_dk, mm_dq)), config, geometry, regions, _dtype_name(api.dtype), sm))
    entry = compile_host_f16(main._host, mm_dk._host, mm_dq._host, config, geometry, regions, dtype, sm, key)
    return _spec(api, entry, operands, offset, "sdpa_bwd_sm107", ROLES_F16, ATTRIBUTES_F16, scale_log2=False)


def compile_plan_fp8(api, main, mm_dk, mm_dq):
    """The fp8 row's spec: the nine tensors, twelve fp32 scalar operands (1 element, 4-byte aligned) and an amax operand per
    requested output (None-specialized otherwise, so ``bind()`` refuses an unrequested amax buffer and requires a requested one)."""
    from .kernels.sm107.prepared_host import compile_host_fp8

    geometry, operands = _tensor_operands(api)
    operands += [Operand("float32", (1,), (1,), 1, 4, 4) for _ in FP8_SCALARS]
    requested = tuple(name in api.amax_requested for name in FP8_AMAX)
    operands += [Operand("float32", (1,), (1,), 1, 4, 4) if flag else None for flag in requested]
    regions, offset = _regions(api, _REGION_SLOTS_FP8)
    config = _config(api)
    sm = _sm(api)
    grad_dtype = _dsl_dtype(api.grad_dtype)
    key = repr((tuple(mod.FROST_SOURCE_DIGEST for mod in (main, mm_dk, mm_dq)), config, geometry, regions, _dtype_name(api.grad_dtype), requested, sm))
    entry = compile_host_fp8(main._host, mm_dk._host, mm_dq._host, config, geometry, regions, grad_dtype, requested, sm, key)
    return _spec(api, entry, operands, offset, "sdpa_bwd_sm107_fp8", ROLES_FP8, ATTRIBUTES_FP8, scale_log2=True)


def _spec(api, entry, operands, offset, name, roles, attributes, *, scale_log2):
    owner = SimpleNamespace(entry=entry, workspace_bytes=offset)
    fn = positional_entry(entry)
    if fn is None:
        raise NotImplementedError(f"{name} requires a positional tvm-ffi entry")
    return BwdLaunchSpec(
        owner,
        fn,
        tuple(operands),
        offset,
        int(api.q_desc.device.index or 0),
        api.scale_softmax,
        name,
        length_form=False,
        roles=roles,
        attributes=attributes,
        scale_log2=scale_log2,
    )


def execute_standalone(api, tensors, workspace, current_stream, scale):
    """The adapter's ``execute``: torch tensors in role order -> facts -> ``bind`` -> the artifact.  The nine tensor roles are
    held to the plan's exact geometry (Stats: contiguous with the plan's element count); scalars and amax bind by facts alone."""
    import torch
    from cudnn.sdpa.fwd.prepared import facts_of_tensor
    from .prepared import execute

    spec = api._prepared
    ws = facts_of_tensor(workspace)
    if ws is None or ws.dtype != "uint8" or not ws.contiguous or ws.span < spec.workspace_bytes or ws.device != (2, spec.device_index):
        raise ValueError(f"{spec.name} requires {spec.workspace_bytes} bytes of contiguous uint8 workspace on CUDA device {spec.device_index}")
    if current_stream is None:
        current_stream = torch.cuda.current_stream(api.q_desc.device).cuda_stream
    facts = dict(zip(spec.roles, map(facts_of_tensor, tensors)))
    geometry = []
    for i, op in enumerate(spec.operands):
        geom = None
        if i < 9 and op is not None:
            if i == 5:
                stats = facts["stats"]
                if stats is not None and (not stats.contiguous or stats.numel != math.prod(op.shape)):
                    raise ValueError(f"{spec.name}: Stats must be contiguous with {math.prod(op.shape)} elements")
            else:
                # The declared logical-BHSD geometry of the kernels' [B, S, H, D] operand view.
                geom = tuple(op.shape[j] for j in (0, 2, 1, 3)), tuple(op.strides[j] for j in (0, 2, 1, 3))
        geometry.append(geom)
    execute(spec, facts, ws.ptr, int(current_stream), scale=scale, geometry=geometry)
