# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CAKE KDA engine: KDA / KDA_BWD nodes on the frozen CAKE C16 training
kernels (``kernels/``), exact SM100 / SM103, bf16.

The engine serves exactly the contract the CAKE kernels were exported with,
and declines everything else to its siblings:

* bf16 q/k/v/g/beta, K = V = 128, k heads = q heads, v heads a multiple of q heads
* the fused training form: ``use_qk_l2norm``, ``use_beta_sigmoid`` (raw beta
  logits), ``safe_gate`` with ``a_log`` + ``dt_bias`` (fp32), gate lower bound
  -5.0, scale 1/sqrt(128)
* fp32 recurrent states and state gradients
* ``checkpoint_every_n_tokens == 16``: the route's per-chunk checkpoints ride
  the op's ``state_checkpoints`` port from forward to backward
* not under CUDA-graph capture (work items are planned on the host)

The engine reproduces FlashInfer's ``recurrent_kda_training_{forward,backward}``
bit-for-bit on the C16 route. It is opt-in (manifest slot ``opt_in=True``, pin
``plan_name="kda_cake"`` under ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1``)
because the frozen kernels' forward token output differs from FLA ``chunk_kda``
and from ``kda_frost`` by ~0.12 relative RMS in FlashInfer's own input regime,
while the recurrent state and the data gradients agree to ~5e-3; see
``test/python/linear_attention/test_kda_cake.py`` for the measured surface.
"""

from __future__ import annotations

import math

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan

from cudnn.frost import buffers
from cudnn.frost.device import build_device
from ..frost.engine import FrostLaPlan
from ..graph_analyzer import analyze
from . import compiler
from .kda_host import HEAD_DIM, LOWER_BOUND, SCALE, build_kda_cake

_NAME = "KdaCakeEngine"


def _decline(message: str) -> None:
    raise NotImplementedError(f"{_NAME}: {message}")


class KdaCakeEngine(BaseEngine):
    """The CAKE-generated C16 training route for single-node KDA graphs (THD layout)."""

    name = "kda_cake"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph) -> None:
        import cudnn

        facts = graph._facts_for(analyze)
        if facts is None or facts.op != "KDA":
            _decline("supports exactly one KDA/KDA_BWD node")
        if facts.invalid:
            _decline(facts.invalid)
        sm = buffers.current_sm()
        if sm not in (100, 103):
            _decline(f"the frozen CAKE kernels target exact SM100 / SM103 (found {sm})")
        try:
            compiler.cuda_include_dirs()
        except compiler.CakeCompileError as exc:
            _decline(str(exc))

        f32, bf16 = cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16
        if not facts.uniform_io or facts.io_dtype != bf16:
            _decline(f"q/k/v must all be bf16, got {facts.io_dtype}")
        if not facts.thd_layout:
            _decline("q/k/v must be THD [total_T, heads, dim]")
        if facts.d_qk != HEAD_DIM or facts.d_v != HEAD_DIM:
            _decline(f"head dims must be {HEAD_DIM}, got K={facts.d_qk} V={facts.d_v}")
        if facts.h_k != facts.h_q:
            _decline(f"k heads ({facts.h_k}) must equal q heads ({facts.h_q})")
        if facts.h_v < facts.h_q or facts.h_v % facts.h_q:
            _decline(f"v heads ({facts.h_v}) must be a multiple of q heads ({facts.h_q})")
        if not facts.gates_at_ho:
            _decline(f"g/beta must carry HO = {facts.h_o} heads")
        if facts.gate_channels != HEAD_DIM:
            _decline(f"g must carry {HEAD_DIM} channels, got {facts.gate_channels}")
        if facts.g_dtype != bf16:
            _decline(f"'g' must be bf16 raw logits, got {facts.g_dtype}")
        if facts.beta_dtype != bf16:
            _decline(f"'beta' must be bf16 raw logits, got {facts.beta_dtype}")
        if not (facts.safe_gate and facts.has_a_log and facts.has_dt_bias):
            _decline("requires safe_gate=True with a_log and dt_bias")
        if facts.a_log_dtype != f32 or facts.dt_bias_dtype != f32:
            _decline("a_log and dt_bias must be fp32")
        if not facts.use_qk_l2norm:
            _decline("requires use_qk_l2norm_in_kernel=True")
        if not facts.use_beta_sigmoid:
            _decline("requires use_beta_sigmoid_in_kernel=True")
        if facts.allow_neg_eigval:
            _decline("allow_neg_eigval is not supported")
        if facts.beta_guard:
            _decline("beta_guard is not supported")
        if facts.batch_invariant:
            _decline("batch_invariant is not supported (dynamic work-item scheduling)")
        if facts.gate_lower_bound is not None and facts.gate_lower_bound != LOWER_BOUND:
            _decline(f"gate_lower_bound is fixed at {LOWER_BOUND}, got {facts.gate_lower_bound}")
        if facts.scale is not None and not math.isclose(facts.scale, SCALE, rel_tol=1e-9, abs_tol=1e-12):
            _decline(f"scale is fixed at 1/sqrt({HEAD_DIM}), got {facts.scale}")
        if facts.state_dtype not in (f32, None) or facts.final_state_dtype not in (f32, None):
            _decline("recurrent states must be fp32")
        if facts.cu_dtype not in (cudnn.data_type.INT32, cudnn.data_type.INT64, None):
            _decline(f"'cu_seqlens' must be int32/int64, got {facts.cu_dtype}")
        if facts.checkpoint_every_n_tokens != 16:
            _decline(f"requires checkpoint_every_n_tokens=16 (the route's checkpoint cadence), got {facts.checkpoint_every_n_tokens}")
        if facts.is_bwd:
            if facts.do_dtype != bf16:
                _decline("'dO' must be bf16")
            if facts.state_checkpoints_dtype not in (bf16, None):
                _decline("'state_checkpoints' must be bf16")
            if facts.dg_dtype not in (bf16, None) or facts.dbeta_dtype not in (bf16, None):
                _decline("'dG' and 'dBeta' must be bf16 (they match g / beta)")
            if facts.d_final_state_dtype not in (f32, None) or facts.d_initial_state_dtype not in (f32, None):
                _decline("state gradients must be fp32")
            if facts.d_a_log_dtype not in (f32, None) or facts.d_dt_bias_dtype not in (f32, None):
                _decline("'d_a_log' / 'd_dt_bias' must be fp32")
        elif facts.state_checkpoints_out_dtype not in (bf16, None):
            _decline("'state_checkpoints' must be bf16")

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            return FrostLaPlan(build_kda_cake(graph))


__all__ = ["KdaCakeEngine"]
