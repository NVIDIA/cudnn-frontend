# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Cake chunk-64 KDA with caller-owned scratch."""

from __future__ import annotations

import cudnn
from cudnn.engines.base import BaseEngine
from cudnn.frost import buffers
from cudnn.frost.device import build_device

from ..graph_analyzer import analyze


def check_contract(facts, node):
    """Accept the exact low-level chain contract before compiling."""

    def decline(reason):
        raise NotImplementedError(f"kda_cake_det: {reason}")

    if facts is None or facts.op != "KDA":
        decline("requires one KDA or KDA_BWD node")
    if facts.invalid:
        decline(facts.invalid)
    if not facts.thd_layout or not facts.uniform_io or facts.io_dtype != cudnn.data_type.BFLOAT16:
        decline("requires BF16 THD q/k/v")
    if facts.d_qk != 128 or facts.d_v != 128 or not (facts.h_q == facts.h_k == facts.h_v) or facts.h_q not in (16, 32, 64):
        decline("requires K=V=128, equal q/k/v heads in {16,32,64}")
    if facts.n_seq > 65536:
        decline("requires at most 65536 documents")
    chunk_bound = (facts.total_t + 63) // 64 + facts.n_seq
    chunk_bound += chunk_bound % 2
    if facts.total_t < 1 or facts.n_seq < 1 or max(facts.total_t * facts.h_v * 128, chunk_bound * facts.h_v * 128 * 128) >= 2**31:
        decline("requires positive tokens/sequences and signed-int32 element offsets")
    if facts.cu_dtype != cudnn.data_type.INT32 or facts.g_dtype != cudnn.data_type.FLOAT or facts.beta_dtype != cudnn.data_type.BFLOAT16:
        decline("requires int32 cu_seqlens and FP32 g and BF16 beta")
    if not facts.gates_at_ho or facts.gate_channels != 128:
        decline("requires 128-channel gates on all heads")
    if facts.use_qk_l2norm or facts.safe_gate or facts.use_beta_sigmoid or facts.allow_neg_eigval or facts.beta_guard or facts.gate_domain != "log":
        decline("requires caller-normalized q/k, activated log gates and activated beta")
    if facts.batch_invariant or facts.overwrite_initial_state or facts.has_state_indices:
        decline("does not serve batch invariance or state mutation/indexing")
    allowed_inputs = {"q", "k", "v", "g", "beta", "cu_seqlens", "dO"}
    allowed_outputs = {"O", "dQ", "dK", "dV", "dG", "dBeta"}
    if set(node.inputs) - allowed_inputs or set(node.outputs) - allowed_outputs or facts.checkpoint_every_n_tokens:
        decline("requires zero initial state, no final state, and checkpoint_every_n_tokens=0")
    wants = {
        "q": cudnn.data_type.BFLOAT16,
        "k": cudnn.data_type.BFLOAT16,
        "v": cudnn.data_type.BFLOAT16,
        "dO": cudnn.data_type.BFLOAT16,
        "O": cudnn.data_type.BFLOAT16,
        "dQ": cudnn.data_type.BFLOAT16,
        "dK": cudnn.data_type.BFLOAT16,
        "dV": cudnn.data_type.BFLOAT16,
        "dG": cudnn.data_type.FLOAT,
        "dBeta": cudnn.data_type.BFLOAT16,
        "g": cudnn.data_type.FLOAT,
        "beta": cudnn.data_type.BFLOAT16,
        "cu_seqlens": cudnn.data_type.INT32,
    }
    for name, tensor in node.outputs.items():
        source = {"O": "v", "dQ": "q", "dK": "k", "dV": "v", "dG": "g", "dBeta": "beta"}[name]
        expected = node.inputs.get(source)
        if tensor.dim and expected is not None and tuple(tensor.dim) != tuple(expected.dim):
            decline(f"{name} shape must match {source}")
    for name, tensor in {**node.inputs, **node.outputs}.items():
        if tensor.get_data_type() != wants[name] and not (name in node.outputs and tensor.get_data_type() is None):
            decline(f"{name} has unsupported dtype {tensor.get_data_type()}")
        if tensor.dim and tensor.stride and not buffers.is_contiguous(tensor.dim, tensor.stride):
            decline(f"{name} must be contiguous")


class KdaCakeDetEngine(BaseEngine):
    name = "kda_cake_det"
    behavior_notes = (cudnn.behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph):
        from cudnn._cake import loader

        facts = graph._facts_for(analyze)
        node = next(iter(graph.nodes), None)
        check_contract(facts, node)
        arch = loader.device_arch()
        if arch != "sm_100a":
            raise NotImplementedError(f"{self.name}: requires SM100, got {arch}")
        loader.check_toolkit(arch, self.name)

    def build_plan(self, graph, plan, ctx=None):
        from ..frost.engine import FrostLaPlan
        from .kda_det_plan import KdaCakeDetPlan

        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            return FrostLaPlan(KdaCakeDetPlan(graph._facts_for(analyze), next(iter(graph.nodes))))
