# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine layer shared by the cuTile linear-attention families: the
check_support core both run, and the execute-time buffer gate."""

from __future__ import annotations

import cudnn
from cudnn.frost import buffers

from ..graph_analyzer import BUFFER_NAME_FROM_CUDNN

CUTILE_ALIGN = {"cu_seqlens": 4, "a_log": 4, "dt_bias": 4, "beta": 4}

CUTILE_MIN_CUDART = 13030

CUTILE_MAX_D_QK = 256


def cutile_la_gate(engine: str, facts, op: str, dg_want) -> None:
    """The cuTile LA engines' shared check_support core: the analyzer record,
    the runtime environment, and the gates common to both kernel modules.

    ``dg_want`` is the family's dG output dtype -- fp32 for GDN, the gate's own
    dtype for KDA -- and is the only gate that differs between them. Each
    engine still probes its own kernel module for the cuda.tile runtime, then
    adds its family's feature declines. Mirrors :func:`frost_la_gate`."""
    if facts is None or facts.op != op:
        raise NotImplementedError(f"{engine} supports exactly one {op}/{op}_BWD node")
    if facts.invalid:
        raise NotImplementedError(f"{engine}: {facts.invalid}")
    if buffers.current_sm() is None:
        raise NotImplementedError(f"{engine} requires a CUDA device")
    try:
        from cuda.bindings import runtime
    except ImportError as exc:
        raise NotImplementedError(f"{engine} requires cuda.bindings: {exc}") from exc
    err, cudart_version = runtime.cudaRuntimeGetVersion()
    if int(err) != 0:
        raise NotImplementedError(f"{engine}: cudaRuntimeGetVersion failed ({err})")
    if cudart_version < CUTILE_MIN_CUDART:
        raise NotImplementedError(f"{engine} requires CUDA 13.3+ (found {cudart_version})")
    if facts.checkpoint_every_n_tokens > 0 or facts.wants_state_checkpoints:
        raise NotImplementedError(f"{engine}: per-chunk state_checkpoints output is not supported")

    if not facts.uniform_io:
        raise NotImplementedError(f"{engine}: q/k/v dtypes must match")
    if facts.io_dtype not in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, None):
        raise NotImplementedError(f"{engine}: q/k/v must be fp16/bf16, got {facts.io_dtype}")
    if not facts.thd_layout:
        raise NotImplementedError(f"{engine}: q/k/v must be THD [total_T, heads, dim]")
    if facts.h_k != facts.h_q:
        raise NotImplementedError(f"{engine}: q and k head counts differ ({facts.h_q} vs {facts.h_k})")
    if facts.h_q and facts.h_v % facts.h_q != 0:
        raise NotImplementedError(f"{engine}: v heads ({facts.h_v}) must be a multiple of q heads ({facts.h_q}; GQA-style v broadcast is FROST-only)")
    if facts.d_qk > CUTILE_MAX_D_QK:
        raise NotImplementedError(f"{engine}: head dim K must be <= {CUTILE_MAX_D_QK}, got {facts.d_qk}")
    if facts.cu_dtype not in (cudnn.data_type.INT32, None):
        raise NotImplementedError(f"{engine}: cu_seqlens must be int32 (the device-side table builder reads it directly)")

    fp32 = cudnn.data_type.FLOAT
    for port, got in (
        ("initial_state", facts.state_dtype),
        ("final_state", facts.final_state_dtype),
        ("d_final_state", facts.d_final_state_dtype),
        ("d_initial_state", facts.d_initial_state_dtype),
        ("a_log", facts.a_log_dtype),
        ("dt_bias", facts.dt_bias_dtype),
    ):
        if got not in (fp32, None):
            raise NotImplementedError(f"{engine}: '{port}' must be fp32 (callers convert), got {got}")

    io = facts.io_dtype
    if not facts.is_bwd:
        outputs = {"O": (facts.o_dtype, io), "final_state": (facts.final_state_dtype, fp32)}
    else:
        if facts.has_initial_state != facts.wants_d_initial_state:
            raise NotImplementedError(f"{engine}: d_initial_state output must be requested iff initial_state is given")
        outputs = {
            "dQ": (facts.dq_dtype, io),
            "dK": (facts.dk_dtype, io),
            "dV": (facts.dv_dtype, io),
            "dG": (facts.dg_dtype, dg_want),
            "dBeta": (facts.dbeta_dtype, facts.beta_dtype),
            "d_initial_state": (facts.d_initial_state_dtype, fp32),
        }
    for port, (got, want) in outputs.items():
        if got is not None and got != want:
            raise NotImplementedError(f"{engine}: output {port!r} must be {want} (written in place), got {got}")


def expect_table(node, align=CUTILE_ALIGN) -> dict:
    """Build-time ``{port: (dims, dtype_name, align_bytes)}`` for
    :func:`check_layouts_compact`: bound buffers must match the node's frozen
    geometry exactly (one graph per shape), and base pointers must satisfy the
    kernel entry's alignment claim."""
    table = {}
    for ports in (node.inputs, node.outputs):
        for name, t in ports.items():
            if t is None:
                continue
            dims = tuple(int(d) for d in t.dim) if t.dim else None
            table[name] = (dims, BUFFER_NAME_FROM_CUDNN.get(t.get_data_type()), align.get(name, 16))
    return table


def check_layouts_compact(plan_name: str, expect, names, views) -> None:
    """Execute-time gate: every bound buffer must be CONTIGUOUS (the kernels
    stage rank-merged views and whole-buffer zero fills) and must match the
    node's build-time dims/dtype and base alignment per ``expect``. ``names``
    and ``views`` are the plan's bound port names and the index-aligned
    ``variant_pack.operands`` result."""
    for name, b in zip(names, views):
        shape = tuple(b.shape)
        stride = getattr(b, "stride", None)
        strides = tuple(stride()) if callable(stride) else getattr(b, "strides", None)
        exp = expect.get(name) if expect else None
        if exp is not None:
            dims, dtype_name, align = exp
            if dims is not None and shape != dims:
                raise ValueError(f"{plan_name}: buffer for {name!r} must match the graph's build-time dims {dims}; got {shape}")
            if dtype_name is not None and b.dtype != dtype_name:
                raise ValueError(f"{plan_name}: buffer for {name!r} must be {dtype_name} (the node's declared dtype); got {b.dtype}")
            if align:
                ptr = b.data_ptr()
                if ptr % align != 0:
                    raise ValueError(f"{plan_name}: buffer for {name!r} base pointer must be {align}-byte aligned; got 0x{ptr:x}")
        if not buffers.is_contiguous(shape, strides):
            raise ValueError(
                f"{plan_name}: buffer for {name!r} must be contiguous (the cuTile backend stages rank-merged views); got shape {shape} strides {strides}"
            )
