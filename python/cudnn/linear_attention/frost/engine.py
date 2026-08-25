# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine layer shared by the FROST linear-attention families: the
check_support core all three run, and the compiled-plan wrapper their
``build_plan`` returns."""

from __future__ import annotations

import cudnn
from cudnn.engines.base import CompiledPlan, bind_ports
from cudnn.frost import buffers
from cudnn.frost.workspace import Workspace


def frost_la_gate(engine: str, facts, op: str) -> None:
    """The FROST LA engines' shared check_support core: the analyzer record,
    the device/DSL environment, and the gates common to all three kernels."""
    if facts is None or facts.op != op:
        raise NotImplementedError(f"{engine} supports exactly one {op}/{op}_BWD node")
    if facts.invalid:
        raise NotImplementedError(f"{engine}: {facts.invalid}")
    sm = buffers.current_sm()
    if sm is None or not (100 <= sm <= 103):
        raise NotImplementedError(f"{engine} requires SM100-SM103 (found {sm})")
    installed, version = buffers.cutedsl_state()
    if not installed:
        raise NotImplementedError(f"{engine} requires the cutedsl extra (nvidia-cutlass-dsl), which is not installed")
    if buffers.cutedsl_too_old(version):
        want = ".".join(str(v) for v in buffers.CUTEDSL_MIN_VERSION)
        raise NotImplementedError(f"{engine} requires nvidia-cutlass-dsl >= {want}; found {version[1]}")
    if not facts.uniform_io:
        raise NotImplementedError(f"{engine}: q/k/v dtypes must match")
    if facts.io_dtype not in (cudnn.data_type.BFLOAT16, cudnn.data_type.HALF, None):
        raise NotImplementedError(f"{engine}: q/k/v must be fp16/bf16, got {facts.io_dtype}")
    if not facts.thd_layout:
        raise NotImplementedError(f"{engine}: q/k/v must be THD [total_T, heads, dim]")
    if facts.d_qk != 128 or facts.d_v != 128:
        raise NotImplementedError(f"{engine}: head dims must be 128 (the recurrent state is 128x128), got K={facts.d_qk} V={facts.d_v}")
    if facts.h_k not in (facts.h_q, facts.h_v):
        raise NotImplementedError(f"{engine}: k heads ({facts.h_k}) must match q's ({facts.h_q}) or v's ({facts.h_v}; canonical GQA shares grouped k/v heads)")
    if facts.h_v != facts.h_q and max(facts.h_q, facts.h_v) % min(facts.h_q, facts.h_v) != 0:
        raise NotImplementedError(f"{engine}: q heads ({facts.h_q}) and v heads ({facts.h_v}) must be equal or one a multiple of the other")
    if facts.g_dtype not in (cudnn.data_type.FLOAT, None):
        raise NotImplementedError(f"{engine}: 'g' must be fp32, got {facts.g_dtype}")
    if facts.cu_dtype not in (cudnn.data_type.INT32, cudnn.data_type.INT64, None):
        raise NotImplementedError(f"{engine}: 'cu_seqlens' must be int32/int64, got {facts.cu_dtype}")


def dense_layout_message(plan_name, ports, offender) -> str:
    """Name the port behind ``all_dense_layout``'s failing slot. Buffers pass
    straight to the stride-plumbed kernels, so the one execute-time rule is a
    stride-1 innermost dim; this walk only runs on the way to raising."""
    for slots in ports.values():
        for direction in (slots.inputs, slots.outputs):
            for port, slot in direction.items():
                if slot == offender:
                    return f"{plan_name}: buffer for {port!r} must have a stride-1 innermost dim (buffers pass straight to the kernel)"
    return f"{plan_name}: the buffer at variant-pack slot {offender} must have a stride-1 innermost dim"


class FrostLaPlan(CompiledPlan):
    """A compiled LA executor, driven from the normalized variant pack: the
    port-to-slot join is a property of the graph, so it happens once and is
    kept; between executes only the buffer addresses move."""

    takes_variant_pack = True

    def __init__(self, compiled):
        self.compiled = compiled
        self.ports = None
        self.indices = None

    def get_workspace_size(self) -> int:
        return self.compiled.workspace_bytes()

    def execute(self, graph, variant_pack, ctx) -> None:
        ports = self.ports
        if ports is None:
            ports = self.ports = bind_ports(graph, variant_pack)
            (slots,) = ports.values()
            names = list(slots.inputs) + list(slots.outputs)
            self.indices = list(slots.inputs.values()) + list(slots.outputs.values())
            self.compiled.bind(names)
        ok, offender = variant_pack.all_dense_layout()
        if not ok:
            raise ValueError(dense_layout_message(self.compiled.plan_name, ports, offender))
        views = variant_pack.operands(self.indices)
        workspace = Workspace.over(variant_pack, self.compiled.workspace_size, type(self.compiled).__name__)
        self.compiled.run(views, workspace, ctx.stream)
