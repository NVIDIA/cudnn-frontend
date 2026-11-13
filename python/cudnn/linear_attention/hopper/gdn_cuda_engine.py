# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hopper (sm90) GDN engine on the fused CUDA C++ delta-rule kernels, both directions.

Gated DeltaNet is Kimi Delta Attention with a channel-constant gate: KDA decays
every key channel by its own ``exp(g[t, h, d])`` where GDN decays the whole key
row by one ``exp(g[t, h])``. The sm90 kernels under ``cuda_kernels/`` implement
the KDA recurrence and have a scalar-gate build (``-DKDA_FUSED_SCALAR_GATE``,
``-DKDA_BWD_SCALAR_GATE``) that this engine runs. It differs from the
per-channel build in two ways, and both matter:

* ``g`` is read and ``dG`` written at ``[T, H]``. The gate is never expanded:
  it is broadcast inside the kernels' shared-memory gate plane, and its
  gradient is reduced over the channels before the one store it gets.
* nothing is ever divided by the decay, so the gate needs no floor. The
  per-channel kernels factor the intra-chunk decay ``exp(G_i - G_j)`` into two
  operand scalings, one of which grows like ``exp(-G)`` and is bounded only by
  KDA's gate floor of -5 per token. GDN's gate has none --
  ``-exp(A_log) * softplus(.)`` with ``exp(A_log)`` up to 16 is what a freshly
  initialised model produces -- and on such a gate the per-channel kernels
  return NaN. A scalar gate commutes with the channel contraction, so the
  scalar build takes its Gram products on the raw operands and applies
  ``exp(G_i - G_j)`` to the small results. A gate below -128, where ``exp`` is
  already exactly zero in fp32, is read as -128, so a hard reset (-1e8, -inf)
  does not swamp the in-chunk prefix sums those differences are taken from.
  Subtracting inclusive fp32 prefixes is what the decay-factorisation section
  of ``python/cudnn/AGENTS.md`` says not to do. The -128 floor keeps a chunk
  prefix under 2^12 forward and 2^14 backward, where fp32 resolves about 1e-3
  and 2e-3; that limits the loss and does not remove it.

So this is not ``kda_hopper_cuda`` behind a gate expansion, and must not be
turned into one: that path is both slower and wrong for GDN's gate range.

Measured on H100 (bf16, K = V = 128, 4 x 4096 tokens x 32 heads, a freshly
initialised gate), forward + backward through the op runs at about 1.1x the
speed of FLA's Triton kernels and 2.8x that of ``gdn_cutile``, with about 12%
less transient memory than FLA.

Known limit, inherited from the forward kernel: it cuts each sequence into
``264 // pairs`` pieces, so a launch with 132 or fewer (sequence, head) pairs
splits, each piece warming up from a truncated history, and a piece that cannot
accumulate enough decay walks back to the start of its sequence and replays it.
The truncation assumes L2-normalised keys and beta in (0, 1), which
``check_support`` cannot see. The replay computes the same recurrence, but on a
slow gate (a per-chunk log decay above about -2.4) every piece does it, and the
forward kernel then costs more than not splitting at all; on one 32K-token
sequence the forward takes about 2.6x as long as FLA's. The piece count is
chosen on the host, which cannot see the gate; choosing it per (sequence,
head) from the gate, on the device, would remove the replay but not reach FLA
there (about 2.2x FLA's forward unsplit), and is not done.

Scope: THD q/k/v in bf16 with equal head counts and head dim 128, pre-normalised
q/k, a log-space fp32 gate and a post-sigmoid fp32 beta, with or without carried
state. Everything else is declined in ``check_support`` so another GDN engine
serves it.
"""

from typing import TYPE_CHECKING

from cudnn import behavior_note
from cudnn._torch_stream import launch_stream
from cudnn.engines.base import BaseEngine, CompiledPlan, bind_ports
from cudnn.frost import buffers
from cudnn.frost.workspace import Workspace
from cudnn.graph_types import NodeType

from ..graph_analyzer import analyze
from . import marshal
from .layout import declared_layout_reason

if TYPE_CHECKING:
    from cudnn._pygraph import pygraph

HOPPER_SM = 90
HEAD_DIM = 128
ENGINE = "GdnHopperCudaEngine"
STATE_ITEM = HEAD_DIM * HEAD_DIM * 4  # one fp32 [128, 128] state, bytes
MAX_GRID_Y = 65535  # CUDA's limit on a launch grid's y extent

# (device, is_bwd) -> why the kernels could not be compiled or loaded there. The op builds a graph
# per distinct token count, so without this an environment that cannot compile or cache would
# repeat the whole NVRTC compile for every new shape only to decline again.
_LOAD_FAILURES = {}


class _GdnHopperCudaPlanBase(CompiledPlan):
    takes_variant_pack = True
    plan_name = ENGINE

    def __init__(self, graph, device=None):
        (node,) = graph.nodes
        q, cu = (node.inputs[p] for p in ("q", "cu_seqlens"))
        self.total, self.h, self.k = (int(d) for d in q.dim)
        self.n_seqs = int(cu.dim[0]) - 1
        self.state_bytes = self.n_seqs * self.h * STATE_ITEM
        self.ports = None
        # The kernels were compiled and loaded for this device in build_plan.
        self._build_device = device
        self._device = None

    def _bind(self, graph, variant_pack):
        import torch

        if self.ports is None:
            self.ports = bind_ports(graph, variant_pack)
            (slots,) = self.ports.values()
            (node,) = graph.nodes
            self.names = list(slots.inputs) + list(slots.outputs)
            self.indices = list(slots.inputs.values()) + list(slots.outputs.values())
            declared = {**node.inputs, **node.outputs}
            self.dims = [tuple(int(d) for d in declared[name].dim) for name in self.names]
        views = variant_pack.operands(self.indices)
        # The launches take every extent from the graph (T, N and H are kernel arguments), so a buffer
        # of another shape -- an execute-time override that changes it, or a buffer smaller than its
        # declaration -- would be read or written past what it holds. Raise before anything launches.
        for name, view, want in zip(self.names, views, self.dims):
            if tuple(view.shape) != want:
                raise ValueError(
                    f"{ENGINE}: {name} has shape {tuple(view.shape)}, but the plan was built for {want}; execute-time shape changes are not supported"
                )
        if self._device is None:
            self._device = torch.from_dlpack(views[0]).device
        if self._build_device is not None and self._device.index != self._build_device:
            raise ValueError(f"{ENGINE}: the operands are on cuda:{self._device.index}, but the plan was built for cuda:{self._build_device}")
        return views

    def _slab(self, cursor, stream, zero):
        """One ``[N, H, 128, 128]`` fp32 state carved from the caller's workspace at ``cursor``.

        The kernels always read a seed and always write a state; a graph without
        the port gets a slab here. It is per-call scratch on purpose: a buffer
        owned by the plan would stay allocated for every graph the op has ever
        cached (one per distinct token count) and would be shared by two threads
        executing the same graph. A slab standing in for an INPUT is zeroed on
        the execution stream every call, which is what both a foreign stream and
        a CUDA graph replay need.
        """
        if zero:
            buffers.memset_zero_async(cursor, self.state_bytes, stream)
        return cursor, cursor + self.state_bytes

    def _workspace(self, variant_pack):
        """The caller's workspace address, checked: present, large enough and aligned.

        Everything carved below is written through a raw address, so a missing
        or short buffer would be a device fault or a silent overrun rather than
        an error. A plan that needs no scratch (a forward with both state ports
        bound) accepts a call without one.
        """
        need = self.get_workspace_size()
        if need:
            Workspace.over(variant_pack, need, ENGINE)
        return int(variant_pack.workspace)


class GdnHopperCudaPlan(_GdnHopperCudaPlanBase):
    """Forward: one fused kernel; ``O`` and ``final_state`` are written in place."""

    _WANT = {"cu_seqlens": "int32", "g": "float32", "beta": "float32", "initial_state": "float32"}

    def __init__(self, graph, device=None):
        super().__init__(graph, device)
        from . import cuda_host

        (node,) = graph.nodes
        scale = node.params.get("scale", None)
        self.q_scale = cuda_host.DEFAULT_Q_SCALE if scale is None else float(scale)
        self.has_seed = node.inputs.get("initial_state") is not None
        self.has_final = node.outputs.get("final_state") is not None

    def get_workspace_size(self) -> int:
        # The kernel itself needs none; these are the stand-ins for absent state ports.
        return self.state_bytes * ((not self.has_seed) + (not self.has_final))

    def execute(self, graph, variant_pack, ctx) -> None:
        import torch

        from . import cuda_host

        if ctx.context is not None:
            raise ValueError(f"{ENGINE}: this plan has no context (get_context_size() is 0)")
        views = self._bind(graph, variant_pack)
        # Resolved first: every copy and fill below is issued on it.
        # A handle's stream 0 is the legacy default stream, never "torch's current stream" (Rule 5, R1).
        stream = launch_stream(ctx.stream, self._device).cuda_stream
        addr, _keepalive = marshal.resolve_inputs(self.names, views, ENGINE, stream, self._WANT)

        # The kernel always reads a seed and always writes a final state.
        cursor = self._workspace(variant_pack)
        if self.has_seed:
            initial_state = addr["initial_state"]
        else:
            initial_state, cursor = self._slab(cursor, stream, zero=True)
        staged_fs = caller_fs = None
        if self.has_final:
            view = views[self.names.index("final_state")]
            final_state, staged_fs, caller_fs = marshal.stage_output(view, "float32", stream)
        else:
            final_state, cursor = self._slab(cursor, stream, zero=False)

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
            q_scale=self.q_scale,
            scalar_gate=True,
        )
        # A bf16 final_state was written through an fp32 staging buffer.
        marshal.write_back(staged_fs, caller_fs, stream)


class GdnHopperCudaBwdPlan(_GdnHopperCudaPlanBase):
    """Backward: ``k_meta`` (varlen only), ``k_prep``, ``k_scan_t``, ``k_bwd`` over a carved workspace."""

    _WANT = {"cu_seqlens": "int32", "g": "float32", "beta": "float32", "initial_state": "float32", "d_final_state": "float32"}

    def __init__(self, graph, device=None):
        super().__init__(graph, device)
        from . import cuda_bwd_host, cuda_host

        (node,) = graph.nodes
        # The backward takes the query scale as an argument, as the forward does.
        scale = node.params.get("scale", None)
        self.q_scale = cuda_host.DEFAULT_Q_SCALE if scale is None else float(scale)
        self.has_seed = node.inputs.get("initial_state") is not None
        self.has_dfs = node.inputs.get("d_final_state") is not None
        self.has_dis = node.outputs.get("d_initial_state") is not None
        # The kernels' own arena (a multiple of 256 bytes), then the stand-ins for absent state ports.
        self.arena_bytes = cuda_bwd_host.workspace_bytes(self.total, self.h, self.n_seqs, scalar_gate=True)

    def get_workspace_size(self) -> int:
        return self.arena_bytes + self.state_bytes * ((not self.has_seed) + (not self.has_dfs) + (not self.has_dis))

    def execute(self, graph, variant_pack, ctx) -> None:
        import torch

        from . import cuda_bwd_host

        if ctx.context is not None:
            raise ValueError(f"{ENGINE}: this plan has no context (get_context_size() is 0)")
        views = self._bind(graph, variant_pack)
        # A handle's stream 0 is the legacy default stream, never "torch's current stream" (Rule 5, R1).
        stream = launch_stream(ctx.stream, self._device).cuda_stream
        addr, _keepalive = marshal.resolve_inputs(self.names, views, ENGINE, stream, self._WANT)

        # initial_state and d_final_state are optional, absent meaning zero; the
        # kernel always writes d_initial_state.
        workspace = self._workspace(variant_pack)
        cursor = workspace + self.arena_bytes
        if self.has_seed:
            initial_state = addr["initial_state"]
        else:
            initial_state, cursor = self._slab(cursor, stream, zero=True)
        if self.has_dfs:
            d_final_state = addr["d_final_state"]
        else:
            d_final_state, cursor = self._slab(cursor, stream, zero=True)
        staged_dis = caller_dis = None
        if self.has_dis:
            view = views[self.names.index("d_initial_state")]
            d_initial_state, staged_dis, caller_dis = marshal.stage_output(view, "float32", stream)
        else:
            d_initial_state, cursor = self._slab(cursor, stream, zero=False)

        cuda_bwd_host.launch(
            device=self._device.index if self._device.index is not None else 0,
            stream=int(stream),
            workspace=workspace,
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
            scale=self.q_scale,
            scalar_gate=True,
        )
        marshal.write_back(staged_dis, caller_dis, stream)


class GdnHopperCudaEngine(BaseEngine):
    """Hopper (sm90) fused-CUDA backend for single-node GDN graphs, both directions (THD)."""

    name = "gdn_hopper_cuda"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph: "pygraph") -> None:
        import cudnn

        facts = graph._facts_for(analyze)
        if facts is None or facts.op != "GDN":
            raise NotImplementedError(f"{ENGINE} supports exactly one GDN/GDN_BWD node")
        if facts.invalid:
            raise NotImplementedError(f"{ENGINE}: {facts.invalid}")

        sm = buffers.current_sm()
        if sm != HOPPER_SM:
            raise NotImplementedError(f"{ENGINE} is the Hopper path and requires SM90 (found {sm})")

        # NVRTC and a CUDA include tree are the hard dependency. Probe them here so
        # that an environment which cannot compile declines instead of failing at
        # the first execute.
        try:
            from ..cake import compiler

            if facts.is_bwd:
                from . import cuda_bwd_host  # noqa: F401
            else:
                from . import cuda_host  # noqa: F401

            compiler.cuda_include_dirs()
            compiler.nvrtc_version()  # forces the libnvrtc load; importing the bindings does not
        except Exception as exc:  # noqa: BLE001 -- any import/toolkit failure is a decline
            raise NotImplementedError(f"{ENGINE} needs NVRTC and a CUDA include tree: {exc}") from exc

        fp32, bf16 = cudnn.data_type.FLOAT, cudnn.data_type.BFLOAT16
        (node,) = graph.nodes

        # --- geometry: the kernels are built for one head dim and index packed THD operands.
        if not facts.thd_layout:
            raise NotImplementedError(f"{ENGINE}: q/k/v must be THD [total_T, heads, dim]")
        if facts.d_qk != HEAD_DIM or facts.d_v != HEAD_DIM:
            raise NotImplementedError(f"{ENGINE}: head dims must be {HEAD_DIM}, got K={facts.d_qk} V={facts.d_v}")
        if not (facts.h_q == facts.h_k == facts.h_v):
            raise NotImplementedError(f"{ENGINE}: q, k and v head counts must match (got {facts.h_q}/{facts.h_k}/{facts.h_v})")
        if facts.gate_channels:
            raise NotImplementedError(f"{ENGINE}: the GDN gate is one value per (token, head), got {facts.gate_channels} channels")
        if not facts.gates_at_ho:
            raise NotImplementedError(f"{ENGINE}: g and beta must carry one row per head")
        if facts.total_t <= 0 or facts.n_seq <= 0:
            raise NotImplementedError(f"{ENGINE}: needs at least one token and one sequence (got {facts.total_t} tokens, {facts.n_seq} sequences)")
        # The backward scan launches one grid row per (sequence, head) and the forward one per
        # (sequence, piece). Declined in both directions, so one engine serves a graph and its gradient.
        if facts.n_seq * facts.h_q > MAX_GRID_Y:
            raise NotImplementedError(f"{ENGINE}: {facts.n_seq} sequences x {facts.h_q} heads exceeds a launch grid ({MAX_GRID_Y})")
        # Outputs only: a padded input is repacked at execute (see layout.py).
        reason = declared_layout_reason(node, ENGINE, inputs_too=False)
        if reason is not None:
            raise NotImplementedError(reason)

        # --- dtypes. q/k/v and their gradients are bf16; the gate, beta and their gradients
        # fp32; a state is carried in fp32 and converted at execute when it arrives as bf16.
        if not facts.uniform_io or facts.io_dtype not in (bf16, None):
            raise NotImplementedError(f"{ENGINE}: q/k/v must all be bf16, got {facts.io_dtype}")
        if facts.g_dtype not in (fp32, None):
            raise NotImplementedError(f"{ENGINE}: 'g' must be fp32, got {facts.g_dtype}")
        if facts.beta_dtype not in (fp32, None):
            raise NotImplementedError(f"{ENGINE}: 'beta' must be fp32, got {facts.beta_dtype}")
        if facts.cu_dtype not in (cudnn.data_type.INT32, cudnn.data_type.INT64, None):
            raise NotImplementedError(f"{ENGINE}: cu_seqlens must be int32 or int64, got {facts.cu_dtype}")
        state = (fp32, bf16)
        ports = [("initial_state", facts.state_dtype, state)]
        if facts.is_bwd:
            ports += [(p, d, (bf16,)) for p, d in (("dO", facts.do_dtype), ("dQ", facts.dq_dtype), ("dK", facts.dk_dtype), ("dV", facts.dv_dtype))]
            ports += [("dG", facts.dg_dtype, (fp32,)), ("dBeta", facts.dbeta_dtype, (fp32,))]
            ports += [("d_initial_state", facts.d_initial_state_dtype, state), ("d_final_state", facts.d_final_state_dtype, state)]
        else:
            ports += [("O", facts.o_dtype, (bf16,)), ("final_state", facts.final_state_dtype, state)]
        for label, dtype, allowed in ports:
            if dtype is not None and dtype not in allowed:
                raise NotImplementedError(f"{ENGINE}: '{label}' must be one of {allowed}, got {dtype}")

        # --- state features the kernels have no path for.
        if facts.has_state_indices or facts.overwrite_initial_state:
            raise NotImplementedError(f"{ENGINE}: state_indices / overwrite_initial_state are unsupported")
        if facts.checkpoint_every_n_tokens or facts.wants_state_checkpoints:
            raise NotImplementedError(f"{ENGINE}: state_checkpoints are not produced")
        if facts.wants_transition:
            raise NotImplementedError(f"{ENGINE}: the transition output is not produced")
        # Context-parallel ports.
        if facts.has_composed_summaries or facts.has_composed_state or facts.output_context or facts.input_context:
            raise NotImplementedError(f"{ENGINE}: context-parallel ports (composed summaries/state, context) are unsupported")

        # --- attributes. The kernels take a log-space gate, post-sigmoid beta and pre-normalised q/k.
        if facts.safe_gate or facts.has_a_log or facts.has_dt_bias or facts.gate_lower_bound is not None:
            raise NotImplementedError(f"{ENGINE}: safe_gate/a_log/dt_bias are unsupported; pass the log-space gate")
        if facts.gate_domain != "log":
            raise NotImplementedError(f"{ENGINE}: gate_domain='linear' is unsupported")
        if facts.use_qk_l2norm or facts.use_beta_sigmoid:
            raise NotImplementedError(f"{ENGINE}: q/k must be pre-normalised and beta post-sigmoid")
        if facts.allow_neg_eigval or facts.beta_guard:
            raise NotImplementedError(f"{ENGINE}: allow_neg_eigval/beta_guard are unsupported; beta must lie in (0, 1)")
        if facts.num_householder != 1:
            raise NotImplementedError(f"{ENGINE}: num_householder must be 1, got {facts.num_householder}")
        if facts.batch_invariant:
            # The forward splits a sequence into pieces by how many (sequence, head) pairs the launch has.
            raise NotImplementedError(f"{ENGINE}: batch_invariant is not guaranteed by the sm90 kernels")

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        (node,) = graph.nodes
        is_bwd = node.node_type == NodeType.GDN_BWD
        # Compile (or load the cached cubin) here rather than at the first execute. build_plans()
        # walks on to the next engine when a build declines, whereas a compile or cache-write error
        # at execute would be a hard failure on a graph gdn_cutile could have served.
        # The device is the caller's when the build was triggered from execute(..., handle=...);
        # otherwise the thread's current one, which is also what check_support judged.
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        try:
            if device is None:
                from cuda.bindings import runtime

                err, device = runtime.cudaGetDevice()
                if int(err) != 0:
                    raise RuntimeError(f"cudaGetDevice failed ({err})")
            key = (int(device), is_bwd)
            if key in _LOAD_FAILURES:
                raise RuntimeError(_LOAD_FAILURES[key])
            try:
                if is_bwd:
                    from . import cuda_bwd_host

                    cuda_bwd_host.library(int(device), scalar_gate=True)
                else:
                    from . import cuda_host

                    cuda_host._library(int(device), 0, False, True)
            except Exception as exc:  # noqa: BLE001
                _LOAD_FAILURES[key] = f"{type(exc).__name__}: {exc}"
                raise
        except Exception as exc:  # noqa: BLE001 -- whatever stops the kernels loading, the engine cannot serve
            raise NotImplementedError(f"{ENGINE}: the sm90 kernels could not be compiled or loaded: {exc}") from exc
        return GdnHopperCudaBwdPlan(graph, int(device)) if is_bwd else GdnHopperCudaPlan(graph, int(device))
