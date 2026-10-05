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

The gap between those kernel figures and what a caller sees is host dispatch,
and it is measured rather than estimated. On H100 SXM at 2048/12/1 one call
costs 148 us of HOST time to issue against a ~60 us kernel, so the path is
CPU-bound -- the GPU finishes before Python can issue the next call.
Decomposed by stubbing stages out:

    op + graph + variant-pack layer     84.1 us   (57%)
    variant_pack.operands()             12.5 us
    the address dict (9x data_ptr)       5.5 us
    the driver launch                   11.7 us   (bare, outside the op)

The dominant term is the cuDNN FE op/graph layer, which every engine pays and
which is not specific to this one -- kda_cutile's host cost on the same shape
is 417 us against this engine's 148 us. The engine's own share is small, and
the one piece of it that could be cached (the ctypes parameter block, ~4 us) is
rebuilt per call precisely so two threads executing one graph cannot hand each
other the other's pointers, which is not worth trading for microseconds.

The effective fix is CUDA graph capture, and this path captures. On H100 SXM at
2048/12/1: host cost 151.4 us -> 3.0 us and wall 176.1 us -> 27.2 us (6.5x),
with replay reproducing eager numerics. Capture requires the steady state to
allocate and synchronise nothing, so
``test_kda_sm90_cuda.test_cuda_graph_capture_replays`` guards it.

Correctness IS established for both directions on H100 at the production gate
(``gate_lower_bound = -5``) with a non-zero ``initial_state``: forward 6/6
workloads on both outputs, backward 6/6 shapes on all six gradients scored by
the campaign definition's three-term checker.
"""

import math
from typing import TYPE_CHECKING

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan, bind_ports
from cudnn.frost import buffers
from cudnn.frost.workspace import Workspace, WorkspaceLayout
from cudnn.graph_types import NodeType

from ..graph_analyzer import analyze
from . import marshal
from .layout import declared_layout_reason

if TYPE_CHECKING:
    from cudnn._pygraph import pygraph

HOPPER_SM = 90
HEAD_DIM = 128


class KdaHopperCudaPlan(CompiledPlan):
    """Bind the node's ports and launch the fused kernel.

    Destination-passing: ``o`` and ``final_state`` are written in place. The
    kernel always produces a final state, so a graph that did not ask for one
    gets a workspace carve whose result is dropped.

    The plan owns no device memory (Rule 8). Its workspace, offsets fixed at
    build: an fp32 state for an absent ``final_state``, an fp32 zero seed for an
    absent ``initial_state``, then the marshal staging for every port whose
    DECLARED dtype or strides the kernel cannot read natively (see marshal.py).
    A graph declaring fp32/int32 packed operands with both state ports needs 0.
    """

    takes_variant_pack = True
    plan_name = "KdaHopperCudaEngine"

    # What the forward kernel reads each port as, by dtype NAME. A port absent
    # here keeps the caller's dtype; a listed port is converted only when it does
    # not already match, so the common case costs one comparison and no copy.
    _WANT_FWD = {
        "cu_seqlens": "int32",
        "g": "float32",
        "beta": "float32",
        "a_log": "float32",
        "dt_bias": "float32",
        "initial_state": "float32",
    }
    _WANT_FWD_OUT = {"final_state": "float32"}

    def __init__(self, graph):
        (node,) = graph.nodes
        q, v, cu = (node.inputs[p] for p in ("q", "v", "cu_seqlens"))
        self.total, self.h_qk, self.k = (int(d) for d in q.dim)
        self.v_dim = int(v.dim[2])
        # cuDNN carries the gate, beta, state and output at HO = max(H_q, H_v).
        # The kernel's head grid is HO; q and k address themselves at h_qk.
        self.h = max(self.h_qk, int(v.dim[1]))
        self.n_seqs = int(cu.dim[0]) - 1
        self.ports = None

        self.state_bytes = self.n_seqs * self.h * self.v_dim * self.k * 4
        layout = WorkspaceLayout()
        self.off_fs = None if "final_state" in node.outputs else layout.add(self.state_bytes)
        self.off_s0 = None if "initial_state" in node.inputs else layout.add(self.state_bytes)
        self.staging = {
            port: (layout.add(math.prod(shape) * buffers.DTYPE_ITEMSIZE[dtype]), dtype, shape)
            for port, dtype, shape in marshal.staging_ports(node, self._WANT_FWD, self._WANT_FWD_OUT)
        }
        self.workspace_size = layout.size

        # In-kernel input fusions, fixed per node: the kernel applies them to the
        # staged tile, so nothing here re-materialises q, k or g.
        from . import cuda_host

        params = node.params
        self.flags = 0
        if params.get("use_qk_l2norm", False):
            self.flags |= cuda_host.FLAG_L2NORM
        if params.get("safe_gate", False):
            self.flags |= cuda_host.FLAG_SAFE_GATE
        if params.get("use_beta_sigmoid", False):
            self.flags |= cuda_host.FLAG_BETA_SIGMOID
        lb = params.get("gate_lower_bound", None)
        self.gate_lower_bound = -5.0 if lb is None else float(lb)
        # The kernel takes the query scale as an argument, so any scale is served.
        scale = params.get("scale", None)
        self.q_scale = cuda_host.DEFAULT_Q_SCALE if scale is None else float(scale)

    def get_workspace_size(self) -> int:
        return self.workspace_size

    def execute(self, graph, variant_pack, ctx) -> None:
        import torch

        from . import cuda_host

        if self.ports is None:
            self.ports = bind_ports(graph, variant_pack)
            (slots,) = self.ports.values()
            self.names = list(slots.inputs) + list(slots.outputs)
            self.indices = list(slots.inputs.values()) + list(slots.outputs.values())
        views = variant_pack.operands(self.indices)

        # Resolved first: every copy and fill below is issued on it.
        stream = ctx.stream if ctx.stream else torch.cuda.current_stream().cuda_stream
        base = int(variant_pack.workspace)
        staging = {}
        if self.workspace_size:
            workspace = Workspace.over(variant_pack, self.workspace_size, self.plan_name)
            staging = {port: workspace.view(off, dtype, shape) for port, (off, dtype, shape) in self.staging.items()}

        # The OperandBuffer already knows its pointer, so there is no DLPack
        # conversion on this route. The kernel never sees a stride, though, so a
        # padded operand would be read as if it were packed: a DECLARED padded
        # input is repacked through its staging carve, an undeclared one raises,
        # and a padded output raises since it is written in place.
        # Layout AND dtype resolved in ONE pass per port: the kernel indexes
        # packed operands from a raw address and wants an int32 chunk table with
        # an fp32 gate/beta/state. Resolving them separately is what let a
        # repacked address be overwritten by the original padded one.
        addr = marshal.resolve_inputs(self.names, views, "KdaHopperCudaEngine", stream, self._WANT_FWD, staging)
        view_of = dict(zip(self.names, views))

        staged_fs = caller_fs = None
        if "final_state" in addr:
            final_state, staged_fs, caller_fs = marshal.stage_output(view_of["final_state"], "float32", stream, staging.get("final_state"), port="final_state")
        else:
            final_state = base + self.off_fs

        # A graph without initial_state means "seed from zero". The carve is
        # re-zeroed ON THE EXECUTION STREAM every call (R4): under capture the
        # memset becomes a graph node, so every replay sees zeros.
        if "initial_state" in addr:
            initial_state = addr["initial_state"]
        else:
            initial_state = base + self.off_s0
            buffers.memset_zero_async(initial_state, self.state_bytes, stream)

        cuda_host.launch(
            device=variant_pack.device,
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
            a_log=addr.get("a_log", 0),
            dt_bias=addr.get("dt_bias", 0),
            gate_lower_bound=self.gate_lower_bound,
            flags=self.flags,
            q_scale=self.q_scale,
            n_qk_heads=self.h_qk,
        )
        # A bf16 final_state was written through an fp32 staging buffer.
        marshal.write_back(staged_fs, caller_fs, stream)


class KdaHopperCudaBwdPlan(CompiledPlan):
    """Bind the node's ports and run the four backward kernels.

    Destination-passing: dQ/dK/dV/dG/dBeta (and d_initial_state when asked for)
    are written in place. Unlike the forward, this kernel needs real scratch --
    chunk tables, the UT factors, and three [NCS,H,128,128] state arrays -- so
    the plan declares a workspace and carves it in cuda_bwd_host._layout rather
    than allocating anything of its own. The same arena carries the plan's own
    regions (Rule 8): fp32 zero seeds for an absent ``initial_state`` /
    ``d_final_state``, scratch for an unrequested ``d_initial_state``, and the
    marshal staging decided from the declared dtypes.
    """

    takes_variant_pack = True
    plan_name = "KdaHopperCudaEngine"

    _WANT_BWD = {
        "cu_seqlens": "int32",
        "g": "float32",
        "beta": "float32",
        "initial_state": "float32",
        "d_final_state": "float32",
    }
    _WANT_BWD_OUT = {"d_initial_state": "float32"}

    def __init__(self, graph):
        (node,) = graph.nodes
        q, cu = (node.inputs[p] for p in ("q", "cu_seqlens"))
        self.total, self.h, self.k = (int(d) for d in q.dim)
        self.n_seqs = int(cu.dim[0]) - 1
        self.ports = None

        self.state_bytes = self.n_seqs * self.h * self.k * self.k * 4
        ports = marshal.staging_ports(node, self._WANT_BWD, self._WANT_BWD_OUT)
        self.staging = {port: (dtype, shape) for port, dtype, shape in ports}
        self.flags = dict(
            zero_s0="initial_state" not in node.inputs,
            zero_dfs="d_final_state" not in node.inputs,
            scratch_dis="d_initial_state" not in node.outputs,
            staging=tuple((port, math.prod(shape), buffers.DTYPE_ITEMSIZE[dtype]) for port, dtype, shape in ports),
        )

    def get_workspace_size(self) -> int:
        from . import cuda_bwd_host

        return cuda_bwd_host.workspace_bytes(self.total, self.h, self.n_seqs, **self.flags)

    def execute(self, graph, variant_pack, ctx) -> None:
        import torch

        from . import cuda_bwd_host

        if self.ports is None:
            self.ports = bind_ports(graph, variant_pack)
            (slots,) = self.ports.values()
            self.names = list(slots.inputs) + list(slots.outputs)
            self.indices = list(slots.inputs.values()) + list(slots.outputs.values())
        views = variant_pack.operands(self.indices)

        # Resolved first: every copy and fill below is issued on it.
        stream = ctx.stream if ctx.stream else torch.cuda.current_stream().cuda_stream
        Workspace.over(variant_pack, self.get_workspace_size(), self.plan_name)
        base = int(variant_pack.workspace)
        ws, _, _ = cuda_bwd_host._layout(base, self.total, self.h, self.n_seqs, **self.flags)
        staging = {port: buffers.DeviceView(ws[port], shape, dtype, variant_pack.device) for port, (dtype, shape) in self.staging.items()}

        # Raw device addresses -- no DLPack conversion on this route. A DECLARED
        # padded input is repacked through its staging carve on the execution
        # stream; an undeclared one raises, as does a padded output.
        addr = marshal.resolve_inputs(self.names, views, "KdaHopperCudaEngine", stream, self._WANT_BWD, staging)
        view_of = dict(zip(self.names, views))

        # initial_state and d_final_state are optional; absent means "zero", and
        # the carve is re-zeroed on the execution stream every call (R4).
        initial_state = addr.get("initial_state")
        if initial_state is None:
            initial_state = ws["s0_zero"]
            buffers.memset_zero_async(initial_state, self.state_bytes, stream)
        d_final_state = addr.get("d_final_state")
        if d_final_state is None:
            d_final_state = ws["dfs_zero"]
            buffers.memset_zero_async(d_final_state, self.state_bytes, stream)

        # The kernel always writes d_initial_state; a graph that did not ask for
        # it gets a workspace carve whose result is dropped.
        staged_dis = caller_dis = None
        if "d_initial_state" in addr:
            d_initial_state, staged_dis, caller_dis = marshal.stage_output(
                view_of["d_initial_state"], "float32", stream, staging.get("d_initial_state"), port="d_initial_state"
            )
        else:
            d_initial_state = ws["dinit_scratch"]

        cuda_bwd_host.launch(
            device=variant_pack.device,
            stream=int(stream),
            workspace=base,
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
        # A bf16 d_initial_state was written through an fp32 staging buffer.
        marshal.write_back(staged_dis, caller_dis, stream)


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
                ("d_initial_state", facts.d_initial_state_dtype, (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16)),
                ("d_final_state", facts.d_final_state_dtype, (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16)),
            ):
                allowed = want if isinstance(want, tuple) else (want,)
                if dtype is not None and dtype not in allowed:
                    raise NotImplementedError(f"KdaHopperCudaEngine: '{label}' must be {want}, got {dtype}")
        if facts.checkpoint_every_n_tokens:
            raise NotImplementedError("KdaHopperCudaEngine: state_checkpoints are not produced by the Hopper kernel")
        # Pool-addressed state (PR #1002). getattr, not attribute access: this
        # engine must decline these both before that PR lands, when the facts do
        # not exist yet, and after, when they do. Silently ignoring either would
        # read and write the wrong rows of the caller's pool -- and since the
        # KDA heuristics make this the default engine on sm90, it would be the
        # one that did so.
        # Declared layouts. This is the routing gate: a graph with a padded
        # operand is declined here so another KDA engine serves it, rather than
        # reaching a kernel that indexes it as packed. The op layer declares no
        # strides (the IR packs them), so this only fires for a graph built on
        # the public API -- which is how FlashInfer drives cuDNN.
        (node,) = graph.nodes
        # Outputs only: a DECLARED padded input is repacked at execute through
        # staging the plan sizes at build (see marshal.py), so declining one
        # would cost capability without buying safety.
        reason = declared_layout_reason(node, "KdaHopperCudaEngine", inputs_too=False)
        if reason is not None:
            raise NotImplementedError(reason)
        if getattr(facts, "has_state_indices", False):
            raise NotImplementedError("KdaHopperCudaEngine: state_indices (pool-addressed state) is unsupported")
        if getattr(facts, "overwrite_initial_state", False):
            raise NotImplementedError("KdaHopperCudaEngine: overwrite_initial_state is unsupported")
        if facts.is_bwd and (facts.safe_gate or facts.has_a_log or facts.has_dt_bias):
            raise NotImplementedError("KdaHopperCudaEngine: the kernel takes log-space g directly; " "safe_gate/a_log/dt_bias are unsupported")
        # The forward kernel applies the q/k L2 norm, the safe gate and the beta
        # sigmoid to the staged tile in shared memory -- that is FlashInfer's
        # default KDA contract. The backward kernel has no such path.
        if facts.is_bwd and (facts.use_beta_sigmoid or facts.use_qk_l2norm):
            raise NotImplementedError("KdaHopperCudaEngine: the backward kernel takes pre-normalized q/k and post-sigmoid beta")
        # beta in (0, 2) instead of (0, 1). The kernel's seed-truncation bound
        # assumes the delta-rule factors are non-expansive, which holds only for
        # beta in (0, 1), so a negative eigenvalue silently invalidates it. This
        # used to be unreachable because it requires use_beta_sigmoid, which the
        # engine declined; accepting the sigmoid exposed it.
        if facts.allow_neg_eigval:
            raise NotImplementedError("KdaHopperCudaEngine: allow_neg_eigval needs beta in (0, 1); the seed-truncation bound assumes it")
        if facts.beta_guard:
            raise NotImplementedError("KdaHopperCudaEngine: beta_guard is not applied by the Hopper kernel")
        if facts.safe_gate and not (facts.has_a_log and facts.has_dt_bias):
            raise NotImplementedError("KdaHopperCudaEngine: safe_gate needs both a_log and dt_bias")
        if (facts.has_a_log or facts.has_dt_bias) and not facts.safe_gate:
            raise NotImplementedError("KdaHopperCudaEngine: a_log/dt_bias are only read under safe_gate")
        if getattr(facts, "gate_domain", "log") != "log":
            raise NotImplementedError("KdaHopperCudaEngine: gate_domain='linear' is unsupported")

        # The FORWARD kernel takes the query scale as an argument, so it serves
        # any scale. The backward still bakes in 1/sqrt(D), and a mismatched
        # scale there is a silently wrong dq -- caught by test_bwd_scale when the
        # forward relaxation was applied to both directions.
        if facts.is_bwd and facts.scale is not None and abs(facts.scale - HEAD_DIM**-0.5) > 1e-9:
            raise NotImplementedError(f"KdaHopperCudaEngine: the backward kernel bakes in the default scale 1/sqrt({HEAD_DIM}), got {facts.scale}")
        # int64 cu_seqlens is converted at execute through workspace staging
        # sized at build (N + 1 int32 elements); see marshal.py.
        if facts.cu_dtype not in (cudnn.data_type.INT32, cudnn.data_type.INT64, None):
            raise NotImplementedError(f"KdaHopperCudaEngine: cu_seqlens must be int32 or int64, got {facts.cu_dtype}")
        if facts.d_qk != HEAD_DIM or facts.d_v != HEAD_DIM:
            raise NotImplementedError(f"KdaHopperCudaEngine: head dims must be {HEAD_DIM}, got K={facts.d_qk} V={facts.d_v}")
        # Grouped VALUE attention: more value heads than query heads, each query
        # head shared by a contiguous group. The forward kernel runs its head
        # grid at HO = max(H_q, H_v) and remaps q/k; the backward does not.
        if facts.h_q != facts.h_k:
            raise NotImplementedError(f"KdaHopperCudaEngine: q and k head counts must match (got {facts.h_q}/{facts.h_k})")
        if facts.h_v < facts.h_q or (facts.h_q and facts.h_v % facts.h_q):
            raise NotImplementedError(f"KdaHopperCudaEngine: value heads must be a whole multiple of query heads (got {facts.h_q}/{facts.h_v})")
        if facts.is_bwd and facts.h_v != facts.h_q:
            raise NotImplementedError(f"KdaHopperCudaEngine: the backward kernel needs equal head counts (got {facts.h_q}/{facts.h_v})")
        if facts.io_dtype not in (cudnn.data_type.BFLOAT16, None):
            raise NotImplementedError(f"KdaHopperCudaEngine: q/k/v must be bf16, got {facts.io_dtype}")
        # The kernel reads the gate and beta in fp32. FlashInfer passes both at
        # q's dtype, so on the FORWARD they are converted at execute (through
        # workspace staging sized at build, an interim Rule 2 exception recorded
        # in marshal.py) rather than declined -- one extra pass over g against
        # losing an engine that is 3.5x the alternative.
        #
        # Forward only, deliberately. A 16-bit gate on a backward node means
        # 16-bit dG and dBeta as well, and the kernel writes those in fp32;
        # supporting it would mean staging both gradients through fp32 buffers,
        # which is real work for a path FlashInfer does not use (its KDA is
        # prefill only). So the backward stays fp32-strict and such a graph goes
        # to an engine that serves it.
        sixteen_bit = () if facts.is_bwd else (cudnn.data_type.BFLOAT16, cudnn.data_type.HALF)
        if facts.g_dtype not in (cudnn.data_type.FLOAT, *sixteen_bit, None):
            raise NotImplementedError(f"KdaHopperCudaEngine: 'g' must be fp32{'' if facts.is_bwd else ', bf16 or fp16'}, got {facts.g_dtype}")
        if facts.beta_dtype not in (cudnn.data_type.FLOAT, *sixteen_bit, None):
            raise NotImplementedError(f"KdaHopperCudaEngine: 'beta' must be fp32{'' if facts.is_bwd else ', bf16 or fp16'}, got {facts.beta_dtype}")
        # The kernel carries the state in fp32; a bf16 state is converted at
        # execute through build-sized staging, which is what FlashInfer and
        # vLLM's Kimi path hand cuDNN.
        if facts.final_state_dtype not in (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16, None):
            raise NotImplementedError(f"KdaHopperCudaEngine: 'final_state' must be fp32 or bf16, got {facts.final_state_dtype}")
        if facts.state_dtype not in (cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16, None):
            raise NotImplementedError(f"KdaHopperCudaEngine: 'initial_state' must be fp32 or bf16, got {facts.state_dtype}")
        if not facts.thd_layout:
            raise NotImplementedError("KdaHopperCudaEngine: q/k/v must be THD [total_T, heads, dim]")

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        (node,) = graph.nodes
        if node.node_type == NodeType.KDA_BWD:
            return KdaHopperCudaBwdPlan(graph)
        return KdaHopperCudaPlan(graph)
