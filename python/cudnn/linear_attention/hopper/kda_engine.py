# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hopper (sm90) KDA engine.

The FROST KDA kernels are Blackwell-only by construction -- they are written on
``tcgen05`` MMA and Tensor Memory, neither of which exists on Hopper -- so
``frost/engine.py`` gates them to ``100 <= sm <= 103 or sm == 107`` and sm90 is
left with no linear-attention path at all (cuTile is the only other backend and
needs the ``cuda.tile`` runtime).

This engine fills that hole with a Hopper-native CuTe DSL kernel: a
chunk-parallel PREP pass followed by a sequential SCAN over the [128, 128]
state, built on ``warpgroup`` (wgmma) against shared memory rather than TMEM.

The kernel chunks at BT = 16 and arranges every exponent reaching ``exp2`` to be
``<= 0``, so the exponentials can only underflow to zero. That matters: the
production gate (``gate_lower_bound = -5``, mean log-decay ~ -2.5) makes a
64-token chunk span ~118 in the exponent, and fp32 overflows at 88.

``initial_state`` IS supported -- the recurrence is seeded from it, so this
serves chunked-prefill continuation as well as whole-sequence prefill. A graph
that omits it is handed a zero seed.

Scope is otherwise deliberately narrow, and everything outside it is DECLINED
rather than silently mis-served -- see :meth:`KdaHopperEngine.check_support`.
"""

from typing import TYPE_CHECKING

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan, bind_ports
from cudnn.frost import buffers
from cudnn.graph_types import NodeType

from ..graph_analyzer import analyze

if TYPE_CHECKING:
    from cudnn._pygraph import pygraph

HOPPER_SM = 90
HEAD_DIM = 128


class KdaHopperPlan(CompiledPlan):
    """Bind the node's ports and hand them to the vendored sm90 kernel.

    The kernel is destination-passing: ``o`` and ``final_state`` are written in
    place. It always produces a final state, so when the graph did not ask for
    one we pass a scratch buffer and drop it.
    """

    takes_variant_pack = True
    plan_name = "KdaHopperEngine"

    def __init__(self, graph):
        from .kernel import kda_prefill_sm90 as kernel

        (node,) = graph.nodes
        self.kernel = kernel
        q, v, cu = (node.inputs[p] for p in ("q", "v", "cu_seqlens"))
        self.total, self.h, self.k = (int(d) for d in q.dim)
        self.v_dim = int(v.dim[2])
        self.n_seqs = int(cu.dim[0]) - 1
        self.want_state = "final_state" in node.outputs
        self.ports = None
        self._scratch_state = None
        self._zero_state = None

    def get_workspace_size(self) -> int:
        # The kernel owns its own scratch (module-level cache keyed by shape and
        # device), so nothing is carved out of the caller's workspace.
        return 0

    def execute(self, graph, variant_pack, ctx) -> None:
        import torch

        if self.ports is None:
            self.ports = bind_ports(graph, variant_pack)
            (slots,) = self.ports.values()
            self.names = list(slots.inputs) + list(slots.outputs)
            self.indices = list(slots.inputs.values()) + list(slots.outputs.values())
        views = variant_pack.operands(self.indices)
        # Operands arrive as cuDNN OperandBuffer views; the vendored kernel is a
        # torch-level API (it allocates its own scratch and goes through
        # from_dlpack), so borrow them as tensors. OperandBuffer implements the
        # DLPack protocol, so this is a view -- no copy.
        nb = {name: torch.from_dlpack(view) for name, view in zip(self.names, views)}

        final_state = nb.get("final_state")
        if final_state is None:
            if self._scratch_state is None:
                self._scratch_state = torch.empty(
                    self.n_seqs,
                    self.h,
                    self.v_dim,
                    self.k,
                    dtype=torch.float32,
                    device=nb["q"].device,
                )
            final_state = self._scratch_state

        # The kernel always reads a seed. A graph without initial_state means
        # "start from zero", so hand it a zero buffer rather than declining.
        initial_state = nb.get("initial_state")
        if initial_state is None:
            if self._zero_state is None:
                self._zero_state = torch.zeros(
                    self.n_seqs,
                    self.h,
                    self.v_dim,
                    self.k,
                    dtype=torch.float32,
                    device=nb["q"].device,
                )
            initial_state = self._zero_state

        stream = ctx.stream if ctx.stream is not None else 0
        with torch.cuda.stream(torch.cuda.ExternalStream(stream)) if stream else _null():
            self.kernel.run(
                nb["q"],
                nb["k"],
                nb["v"],
                nb["g"],
                nb["beta"],
                nb["cu_seqlens"],
                initial_state,
                nb["O"],
                final_state,
            )


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class KdaHopperEngine(BaseEngine):
    """Hopper (sm90) CuTe DSL backend for single-node KDA forward graphs (THD)."""

    name = "kda_hopper"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph: "pygraph") -> None:
        import cudnn

        facts = graph._facts_for(analyze)
        if facts is None or facts.op != "KDA":
            raise NotImplementedError("KdaHopperEngine supports exactly one KDA node")
        if facts.invalid:
            raise NotImplementedError(f"KdaHopperEngine: {facts.invalid}")

        sm = buffers.current_sm()
        if sm != HOPPER_SM:
            raise NotImplementedError(f"KdaHopperEngine is the Hopper path and requires SM90 (found {sm})")

        try:
            from .kernel import kda_prefill_sm90  # noqa: F401 -- availability probe
        except ImportError as exc:
            raise NotImplementedError(f"KdaHopperEngine requires the cutedsl extra: {exc}") from exc

        # --- scope. Each of these is a real limit of the vendored kernel, and
        # declining is the point: a silently mis-served graph is worse than no
        # Hopper path at all.
        if facts.is_bwd:
            raise NotImplementedError("KdaHopperEngine: forward only; there is no Hopper KDA backward kernel yet")
        if facts.checkpoint_every_n_tokens:
            raise NotImplementedError("KdaHopperEngine: state_checkpoints are not produced by the Hopper kernel")
        if facts.safe_gate or facts.has_a_log or facts.has_dt_bias:
            raise NotImplementedError("KdaHopperEngine: the kernel takes log-space g directly; safe_gate/a_log/dt_bias are unsupported")
        if facts.use_beta_sigmoid:
            raise NotImplementedError("KdaHopperEngine: beta must be post-sigmoid; use_beta_sigmoid_in_kernel is unsupported")
        if facts.use_qk_l2norm:
            raise NotImplementedError("KdaHopperEngine: q/k must be pre-normalized; use_qk_l2norm_in_kernel is unsupported")
        if getattr(facts, "gate_domain", "log") != "log":
            raise NotImplementedError("KdaHopperEngine: gate_domain='linear' is unsupported")

        # The kernel bakes in q * 1/sqrt(D); it takes no scale argument, so any
        # other scale would be silently ignored and produce a wrong answer
        # (caught by test_fwd_scale, which saw an rms ratio of 0.91).
        if facts.scale is not None and abs(facts.scale - HEAD_DIM**-0.5) > 1e-9:
            raise NotImplementedError(f"KdaHopperEngine: only the default scale 1/sqrt({HEAD_DIM}) is supported, got {facts.scale}")
        if facts.cu_dtype not in (cudnn.data_type.INT32, None):
            raise NotImplementedError(f"KdaHopperEngine: cu_seqlens must be int32, got {facts.cu_dtype}")

        if facts.d_qk != HEAD_DIM or facts.d_v != HEAD_DIM:
            raise NotImplementedError(f"KdaHopperEngine: head dims must be {HEAD_DIM}, got K={facts.d_qk} V={facts.d_v}")
        if not (facts.h_q == facts.h_k == facts.h_v):
            raise NotImplementedError(
                f"KdaHopperEngine: grouped heads are unsupported; q/k/v head counts must match " f"(got {facts.h_q}/{facts.h_k}/{facts.h_v})"
            )
        if facts.io_dtype not in (cudnn.data_type.BFLOAT16, None):
            raise NotImplementedError(f"KdaHopperEngine: q/k/v must be bf16, got {facts.io_dtype}")
        if facts.g_dtype not in (cudnn.data_type.FLOAT, None):
            raise NotImplementedError(f"KdaHopperEngine: 'g' must be fp32, got {facts.g_dtype}")
        if facts.beta_dtype not in (cudnn.data_type.FLOAT, None):
            raise NotImplementedError(f"KdaHopperEngine: 'beta' must be fp32, got {facts.beta_dtype}")
        if facts.final_state_dtype not in (cudnn.data_type.FLOAT, None):
            raise NotImplementedError(f"KdaHopperEngine: 'final_state' must be fp32, got {facts.final_state_dtype}")
        # The kernel reads and writes the [128, 128] state as fp32 wgmma
        # accumulators; a bf16 state pool would be reinterpreted, not converted.
        if facts.state_dtype not in (cudnn.data_type.FLOAT, None):
            raise NotImplementedError(f"KdaHopperEngine: 'initial_state' must be fp32, got {facts.state_dtype}")
        if not facts.thd_layout:
            raise NotImplementedError("KdaHopperEngine: q/k/v must be THD [total_T, heads, dim]")

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        return KdaHopperPlan(graph)
