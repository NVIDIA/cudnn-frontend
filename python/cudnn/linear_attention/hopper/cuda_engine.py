# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hopper (sm90) KDA engine backed by fused CUDA C++ kernels, both directions.

Second sm90 KDA path, alongside ``kda_engine.KdaHopperEngine``. Same operation,
different implementation: CUDA C++ compiled with NVRTC and launched through the
driver API, rather than a CuTe DSL PREP+SCAN pair.

Forward is one fused ``__global__``. Backward is four (``k_meta``, ``k_prep``,
``k_scan_t``, ``k_bwd``) over a carved workspace, and is the only sm90 KDA
backward there is -- ``kda_engine`` has a forward kernel only, so before this a
Hopper backward fell to ``kda_cutile``.

Why both engines exist: they have different dependency footprints. This one
needs NVRTC and a CUDA toolkit include tree, the CuTe DSL one needs
``nvidia-cutlass-dsl``. Neither is guaranteed present, so declining
independently is better than one engine with two failure modes.

On performance, read these as KERNEL figures, measured standalone on H100 SXM at
the production gate with an in-run control -- not as what a caller sees through
this engine:

* forward: 53.9 us geomean over ten shapes, against the CuTe DSL kernel's
  368.5 us and FlashKDA's 438.3 us (8.12x FlashKDA, winning all ten). It runs
  at 1.6-1.8x of the minimum-memory-traffic roofline, so it is close to
  bandwidth-bound; FlashKDA sits ~26x above that roofline because its cost is a
  serial scan on ~12 CTAs.
* backward: 316.2 us geomean over six shapes against the cuTile backward's
  1765.4 us (5.58x per-call, 4.37x pipelined).

The gap between those and the engine matters here and is not yet closed. The
cuDNN FE op/graph layer costs a fixed ~115 us per call, measured on this path
as engine-minus-direct-launch across four shapes. The forward kernel is fast
enough (47-63 us) that this DOMINATES per-call latency, so the 8.12x does not
survive to the caller per-call; pipelined, the overhead overlaps with GPU work
and falls to ~0 at the larger shapes. Engine-level timing against FlashKDA on
SXM has NOT been measured yet, so no engine-level speedup is claimed.

Correctness IS established for both directions on H100 at the production gate
(``gate_lower_bound = -5``) with a non-zero ``initial_state``: forward 6/6
workloads on both outputs, backward 6/6 shapes on all six gradients scored by
the campaign definition's three-term checker.
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


class KdaHopperCudaPlan(CompiledPlan):
    """Bind the node's ports and launch the fused kernel.

    Destination-passing: ``o`` and ``final_state`` are written in place. The
    kernel always produces a final state, so a graph that did not ask for one
    gets a scratch buffer whose result is dropped.
    """

    takes_variant_pack = True
    plan_name = "KdaHopperCudaEngine"

    def __init__(self, graph):
        (node,) = graph.nodes
        q, v, cu = (node.inputs[p] for p in ("q", "v", "cu_seqlens"))
        self.total, self.h, self.k = (int(d) for d in q.dim)
        self.v_dim = int(v.dim[2])
        self.n_seqs = int(cu.dim[0]) - 1
        self.ports = None
        self._scratch_state = None
        self._zero_state = None
        self._device = None

    def get_workspace_size(self) -> int:
        # The kernel declares a workspace and never touches it, so nothing is
        # carved out of the caller's.
        return 0

    def execute(self, graph, variant_pack, ctx) -> None:
        import torch

        from . import cuda_host

        if self.ports is None:
            self.ports = bind_ports(graph, variant_pack)
            (slots,) = self.ports.values()
            self.names = list(slots.inputs) + list(slots.outputs)
            self.indices = list(slots.inputs.values()) + list(slots.outputs.values())
        views = variant_pack.operands(self.indices)

        # The kernel takes raw device addresses, so unlike the CuTe DSL path
        # there is no DLPack conversion at all on this route -- the OperandBuffer
        # already knows its pointer.
        addr = {name: int(view.data_ptr()) for name, view in zip(self.names, views)}

        if self._device is None:
            self._device = torch.from_dlpack(views[0]).device

        if "final_state" in addr:
            final_state = addr["final_state"]
        else:
            if self._scratch_state is None:
                self._scratch_state = torch.empty(
                    self.n_seqs,
                    self.h,
                    self.v_dim,
                    self.k,
                    dtype=torch.float32,
                    device=self._device,
                )
            final_state = int(self._scratch_state.data_ptr())

        # A graph without initial_state means "seed from zero".
        if "initial_state" in addr:
            initial_state = addr["initial_state"]
        else:
            if self._zero_state is None:
                self._zero_state = torch.zeros(
                    self.n_seqs,
                    self.h,
                    self.v_dim,
                    self.k,
                    dtype=torch.float32,
                    device=self._device,
                )
            initial_state = int(self._zero_state.data_ptr())

        stream = ctx.stream if ctx.stream else torch.cuda.current_stream().cuda_stream
        cuda_host.launch(
            device=self._device.index if self._device.index is not None else 0,
            stream=int(stream),
            q=addr["q"],
            k=addr["k"],
            v=addr["v"],
            g=addr["g"],
            beta=addr["beta"],
            cu_seqlens=addr["cu_seqlens"],
            initial_state=initial_state,
            o=addr["O"],
            final_state=final_state,
            total_tokens=self.total,
            n_seqs=self.n_seqs,
            n_heads=self.h,
        )


class KdaHopperCudaBwdPlan(CompiledPlan):
    """Bind the node's ports and run the four backward kernels.

    Destination-passing: dQ/dK/dV/dG/dBeta (and d_initial_state when asked for)
    are written in place. Unlike the forward, this kernel needs real scratch --
    chunk tables, the UT factors, and three [NCS,H,128,128] state arrays -- so
    the plan declares a workspace and carves it in cuda_bwd_host._layout rather
    than allocating anything of its own.
    """

    takes_variant_pack = True
    plan_name = "KdaHopperCudaEngine"

    def __init__(self, graph):
        (node,) = graph.nodes
        q, cu = (node.inputs[p] for p in ("q", "cu_seqlens"))
        self.total, self.h, self.k = (int(d) for d in q.dim)
        self.n_seqs = int(cu.dim[0]) - 1
        self.ports = None
        self._zeros = {}
        self._scratch_dis = None
        self._device = None

    def get_workspace_size(self) -> int:
        from . import cuda_bwd_host

        return cuda_bwd_host.workspace_bytes(self.total, self.h, self.n_seqs)

    def _zero(self, key, shape, torch):
        """A zero buffer standing in for an absent optional input.

        Cached per plan: the kernel reads these every call, and reallocating (or
        re-zeroing) one per execute would show up directly in the launch path.
        """
        buf = self._zeros.get(key)
        if buf is None:
            buf = torch.zeros(*shape, dtype=torch.float32, device=self._device)
            self._zeros[key] = buf
        return int(buf.data_ptr())

    def execute(self, graph, variant_pack, ctx) -> None:
        import torch

        from . import cuda_bwd_host

        if self.ports is None:
            self.ports = bind_ports(graph, variant_pack)
            (slots,) = self.ports.values()
            self.names = list(slots.inputs) + list(slots.outputs)
            self.indices = list(slots.inputs.values()) + list(slots.outputs.values())
        views = variant_pack.operands(self.indices)

        # Raw device addresses -- no DLPack conversion on this route.
        addr = {name: int(view.data_ptr()) for name, view in zip(self.names, views)}

        if self._device is None:
            self._device = torch.from_dlpack(views[0]).device

        state_shape = (self.n_seqs, self.h, self.k, self.k)
        # initial_state and d_final_state are optional; absent means "zero".
        initial_state = addr.get("initial_state") or self._zero("s0", state_shape, torch)
        d_final_state = addr.get("d_final_state") or self._zero("dfs", state_shape, torch)

        # The kernel always writes d_initial_state; a graph that did not ask for
        # it gets a scratch buffer whose result is dropped.
        if "d_initial_state" in addr:
            d_initial_state = addr["d_initial_state"]
        else:
            if self._scratch_dis is None:
                self._scratch_dis = torch.empty(*state_shape, dtype=torch.float32, device=self._device)
            d_initial_state = int(self._scratch_dis.data_ptr())

        stream = ctx.stream if ctx.stream else torch.cuda.current_stream().cuda_stream
        cuda_bwd_host.launch(
            device=self._device.index if self._device.index is not None else 0,
            stream=int(stream),
            workspace=int(variant_pack.workspace),
            dO=addr["dO"],
            q=addr["q"],
            k=addr["k"],
            v=addr["v"],
            g=addr["g"],
            beta=addr["beta"],
            cu_seqlens=addr["cu_seqlens"],
            initial_state=initial_state,
            d_final_state=d_final_state,
            dq=addr["dQ"],
            dk=addr["dK"],
            dv=addr["dV"],
            dg=addr["dG"],
            dbeta=addr["dBeta"],
            d_initial_state=d_initial_state,
            total_tokens=self.total,
            n_heads=self.h,
            n_seqs=self.n_seqs,
        )


class KdaHopperCudaEngine(BaseEngine):
    """Hopper (sm90) fused-CUDA backend for single-node KDA graphs, both directions (THD)."""

    name = "kda_hopper_cuda"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph: "pygraph") -> None:
        import cudnn

        facts = graph._facts_for(analyze)
        if facts is None or facts.op != "KDA":
            raise NotImplementedError("KdaHopperCudaEngine supports exactly one KDA node")
        if facts.invalid:
            raise NotImplementedError(f"KdaHopperCudaEngine: {facts.invalid}")

        sm = buffers.current_sm()
        if sm != HOPPER_SM:
            raise NotImplementedError(f"KdaHopperCudaEngine is the Hopper path and requires SM90 (found {sm})")

        # NVRTC and a CUDA include tree are the hard dependency here, in place of
        # the CuTe DSL engine's nvidia-cutlass-dsl. Probe it rather than letting
        # the first execute() fail: an engine that cannot compile should decline.
        try:
            from ..cake import compiler  # noqa: F401

            if facts.is_bwd:
                from . import cuda_bwd_host  # noqa: F401
            else:
                from . import cuda_host  # noqa: F401

            compiler.cuda_include_dirs()
        except Exception as exc:  # noqa: BLE001 -- any import/toolkit failure is a decline
            raise NotImplementedError(f"KdaHopperCudaEngine needs NVRTC and a CUDA include tree: {exc}") from exc

        # --- scope. The forward envelope matches the CuTe DSL engine's, so a
        # graph either declines is declined by both; the backward is served
        # only here, since kda_engine.py has no backward kernel.
        # --- direction-specific dtypes. The gradient ports are fixed by the
        # kernel: dQ/dK/dV are written as bf16, dG/dBeta/d_initial_state as fp32.
        if facts.is_bwd:
            for label, dtype, want in (
                ("dO", facts.do_dtype, cudnn.data_type.BFLOAT16),
                ("dQ", facts.dq_dtype, cudnn.data_type.BFLOAT16),
                ("dK", facts.dk_dtype, cudnn.data_type.BFLOAT16),
                ("dV", facts.dv_dtype, cudnn.data_type.BFLOAT16),
                ("dG", facts.dg_dtype, cudnn.data_type.FLOAT),
                ("dBeta", facts.dbeta_dtype, cudnn.data_type.FLOAT),
                ("d_initial_state", facts.d_initial_state_dtype, cudnn.data_type.FLOAT),
                ("d_final_state", facts.d_final_state_dtype, cudnn.data_type.FLOAT),
            ):
                if dtype not in (want, None):
                    raise NotImplementedError(f"KdaHopperCudaEngine: '{label}' must be {want}, got {dtype}")
        if facts.checkpoint_every_n_tokens:
            raise NotImplementedError("KdaHopperCudaEngine: state_checkpoints are not produced by the Hopper kernel")
        if facts.safe_gate or facts.has_a_log or facts.has_dt_bias:
            raise NotImplementedError("KdaHopperCudaEngine: the kernel takes log-space g directly; " "safe_gate/a_log/dt_bias are unsupported")
        if facts.use_beta_sigmoid:
            raise NotImplementedError("KdaHopperCudaEngine: beta must be post-sigmoid; use_beta_sigmoid_in_kernel is unsupported")
        if facts.use_qk_l2norm:
            raise NotImplementedError("KdaHopperCudaEngine: q/k must be pre-normalized; use_qk_l2norm_in_kernel is unsupported")
        if getattr(facts, "gate_domain", "log") != "log":
            raise NotImplementedError("KdaHopperCudaEngine: gate_domain='linear' is unsupported")

        # The kernel bakes in q * 1/sqrt(D) and takes no scale argument, so any
        # other scale would be silently ignored and produce a wrong answer.
        if facts.scale is not None and abs(facts.scale - HEAD_DIM**-0.5) > 1e-9:
            raise NotImplementedError(f"KdaHopperCudaEngine: only the default scale 1/sqrt({HEAD_DIM}) is supported, got {facts.scale}")
        if facts.cu_dtype not in (cudnn.data_type.INT32, None):
            raise NotImplementedError(f"KdaHopperCudaEngine: cu_seqlens must be int32, got {facts.cu_dtype}")
        if facts.d_qk != HEAD_DIM or facts.d_v != HEAD_DIM:
            raise NotImplementedError(f"KdaHopperCudaEngine: head dims must be {HEAD_DIM}, got K={facts.d_qk} V={facts.d_v}")
        if not (facts.h_q == facts.h_k == facts.h_v):
            raise NotImplementedError(
                "KdaHopperCudaEngine: grouped heads are unsupported; q/k/v head counts must match " f"(got {facts.h_q}/{facts.h_k}/{facts.h_v})"
            )
        if facts.io_dtype not in (cudnn.data_type.BFLOAT16, None):
            raise NotImplementedError(f"KdaHopperCudaEngine: q/k/v must be bf16, got {facts.io_dtype}")
        if facts.g_dtype not in (cudnn.data_type.FLOAT, None):
            raise NotImplementedError(f"KdaHopperCudaEngine: 'g' must be fp32, got {facts.g_dtype}")
        if facts.beta_dtype not in (cudnn.data_type.FLOAT, None):
            raise NotImplementedError(f"KdaHopperCudaEngine: 'beta' must be fp32, got {facts.beta_dtype}")
        if facts.final_state_dtype not in (cudnn.data_type.FLOAT, None):
            raise NotImplementedError(f"KdaHopperCudaEngine: 'final_state' must be fp32, got {facts.final_state_dtype}")
        if facts.state_dtype not in (cudnn.data_type.FLOAT, None):
            raise NotImplementedError(f"KdaHopperCudaEngine: 'initial_state' must be fp32, got {facts.state_dtype}")
        if not facts.thd_layout:
            raise NotImplementedError("KdaHopperCudaEngine: q/k/v must be THD [total_T, heads, dim]")

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        (node,) = graph.nodes
        if node.node_type == NodeType.KDA_BWD:
            return KdaHopperCudaBwdPlan(graph)
        return KdaHopperCudaPlan(graph)
