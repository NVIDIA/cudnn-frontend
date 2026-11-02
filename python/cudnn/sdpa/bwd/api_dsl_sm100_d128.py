# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Graph adapter for dense and packed d128 backward with prepared launches."""

import math
from contextlib import nullcontext

import torch

from cudnn.sdpa.bwd.api_dsl import SdpaBwdDsl
from .d128_contract import ACCUMULATION, RECOMPUTE, select_implementation, validate_dense_bounds, workspace_layout


class SdpaBwdDslSm100D128(SdpaBwdDsl):
    """Dense/THD MHA and BF16 GQA/MQA, with top-left causal or unmasked attention."""

    _NAME = "sdpa_bwd_sm100_d128"
    # DIAGNOSTIC lever (default False = the shipped kernels, rendered PTX-identical; pinned by
    # test_sdpa_bwd_d128_diag_dq_reduce_add_sm100.py): the store warp of the d128 accumulation bodies skips the fp32 dQ
    # ``tma_reduce_add`` issues per q tile -- every barrier arrive / wait and the store-group waits stay, so the barrier
    # table is unchanged -- and the OUTPUT IS WRONG BY DESIGN (dQ stays zero).  A timing-only ceiling for the L2 reduce-add
    # term, set on the adapter INSTANCE before ``compile()`` by a probe driver; no graph path, knob or environment variable
    # reaches it, and the compiled-plan cache keys the armed build apart.  Not a knob (it changes the function), never shipped armed.
    diag_delete_dq_reduce_add: bool = False
    # The per-S selector of the grouped THD route: which body serves a BF16 grouped THD request -- the one-CTA body
    # (``bprop_d128_accum_grouped_f16``, implementation ``accumulation_grouped_thd``) or the cta_group::2 PAIR body
    # (``bprop_d128_accum_grouped_2cta_stail_f16``, implementation ``accumulation_grouped_thd_2cta``; five launches either way).  A typed op-level
    # field (``d128_grouped_2cta_contract.CTA_PAIR_MODES``): "auto" = THE default = the per-S rule -- the pair iff both declared S_q and S_kv >=
    # ``CTA_PAIR_MIN_SEQ`` (8192: the smallest S where the pair body won the measured backward step on cc 10.0), cc in ``CTA_PAIR_CCS`` (the
    # SM100 line, cc 10.0 / 10.3; the compile target follows the device as for the one-CTA body), a cluster
    # of two admitted by the device's occupancy query, and every support predicate of the pair body; else the one-CTA body.  "off" = the one-CTA
    # body always; "on" = the pair body at any S, declining BY NAME when a predicate fails.  Resolved once in ``check_support``
    # (``prepared_sm100_d128_grouped_2cta.resolve_cta_pair_body``) and recorded on the instance (``cta_pair_resolution``: the token
    # ``pair=<mode>:<body>``).  No environment variable; set on the INSTANCE before ``check_support()``; the plan cache keys the two chains apart
    # by body (the one-CTA rendering is untouched: its md5 pins hold).
    cta_pair: str = "auto"
    # DIAGNOSTIC probes of the pair body: a ``CFG.DIAG_2CTA_NAMES`` name ("dq_reduce_add_off" = the store warp skips the four fp32 dQ reduce-add
    # issues; "exchange_off" = the dS exchange ship is skipped and its barrier armed with 0 bytes; "stats_one_chunk" = one stats / delta chunk per
    # quarter reused for all; "sdq_sts_off" = the dQ staging STS skipped) or ``None`` = THE default (no probe; rendered PTX-identical, pinned).
    # Every barrier arrive site stays (the barrier table is unchanged); the OUTPUT IS WRONG BY DESIGN -- a timing ceiling set on the adapter
    # INSTANCE by a probe driver, never a knob, never shipped armed, refused by the driver's `--validate`; read only on the 2-CTA route; typed
    # (ValueError by name); keyed apart in the plan cache.
    diag_2cta: "str | None" = None
    # Timing stamps of the pair body: a typed bool, default False = the shipped rendering (byte-identical; pinned).  True = every mbarrier wait of
    # ONE representative warp per role is bracketed by %clock64 and accumulated per (role, barrier) in SMEM, plus the epilogue's per-tile hand-off
    # segments and the ship -> recv handshake, dumped per CTA at exit into a workspace region the adapter carves only under the lever
    # (`scratch_workspace_bytes()` grows; `prepared_sm100_d128_grouped_2cta.debug_clk_records(api, workspace)` reads it).  Output CORRECT (dK / dV
    # bitwise, the test module's twin); a timing DIAGNOSTIC, never a knob (it changes the rendering), read only on the 2-CTA route under the
    # default role shape (declines by name under "16w_fa4"); the compile key carries it.
    debug_clk: bool = False
    # The pair body's softmax MASK FORM -- "fsel" (``None`` = the same, THE default: the keep-word mask applied as one select per cell on EVERY q
    # tile; rendering byte-identical when unset) | "predicated" (the same mask body under a CTA-uniform runtime branch that runs it only on a tile
    # that can hold a masked cell: the causal diagonal straddle, the last partial q tile, kv rows at or past seq_k; interior tiles pack the exp2
    # values straight; measured neutral against "fsel" on GB200 at 16K / 32K, kept selectable) | "diag_mask_off" (DIAGNOSTIC: the mask body
    # skipped on EVERY tile = the lever's ceiling; OUTPUT WRONG BY DESIGN on the diagonal / edge tiles, set on the adapter INSTANCE by a probe
    # driver, never a knob, never shipped armed, refused by the driver's `--validate`).  Bitwise on dQ / dK / dV between fsel and predicated (a
    # masked P / dS cell is exactly 0 in both forms).  Typed (``config_sm100_d128_accum_2cta_stail.CFG.MASK_FORMS``); read only on the 2-CTA
    # route; the compile key carries it.
    mask_form: "str | None" = None
    # How the GQA group's dK / dV are accumulated on the grouped THD chain -- "bf16_partials" (``None`` = the same, THE default: one bf16 partial
    # per Q head + the fp32 ``dkv_reduce_thd_origins`` fold launch, a fixed summation order; the rendering is byte-identical when unset) |
    # "fp32_reduce" (a per-KV-head fp32 dK / dV workspace carved inside the partial regions and zero-filled by ``delta_zero_thd``; every Q-head
    # owner reduce-adds its fp32 tile with ``cp.reduce.async.bulk.tensor`` add.f32; the fold launch gone; the one bf16 rounding folded into
    # ``cast_dq_thd``: FOUR launches).  A ROUNDING CHANGE -- fp32 accumulation across the q heads of a kv head in hardware order with one bf16
    # rounding at the output, versus 8 bf16-rounded partials summed in a fixed order; dK / dV are NOT bitwise reproducible across runs under it
    # (dQ is untouched; `use_deterministic_algorithm` is already declined on this route) -- hence an op ATTRIBUTE the caller sets, never a knob.
    # Served by the one-CTA grouped THD body and the pair body under the default role shape; the dense grouped chain, the MHA routes and
    # ``warp_roles = "16w_fa4"`` decline it BY NAME.  Typed (``d128_grouped_contract.DKV_ACCUM_FORMS``); the compile key carries it.
    dkv_accum: "str | None" = None
    # The pair body's ROLE SHAPE -- "16w" (``None`` = the same, THE default: the epilogue warps 8-11 stage dS from TMEM, ship the exchange and
    # read dQ out; rendering byte-identical when unset) | "16w_fa4" (an alternate shape kept for follow-up work: the SOFTMAX writes the dS halves
    # straight from its registers and its lead warp ships; warps 8-11 become the REDUCE group (dQ readout + the fp32 slice rounds + the item-end
    # dV / dK staging); two barrier rows deleted, one added; measured slower on GB200 at 8K-64K).  Perf-only on dK / dV (bitwise under both
    # values: the same accumulation chain per GEMM); dQ by the fp64 gate.  Under "16w_fa4" ``debug_clk`` and ``dkv_accum = "fp32_reduce"``
    # decline by name.  Typed (``config_sm100_d128_accum_2cta_stail.CFG.WARP_ROLES``); read only on the 2-CTA route; the compile key carries it.
    warp_roles: "str | None" = None
    # The pair body's STATS FORM -- how the softmax gets each q tile's LSE / delta rows: "tma_ring" (``None`` = the same under the default role
    # shape; THE default since the day-4 measurement, +1.8 / +1.9 % at the step at 16K / 32K and +3.3 % on the graph replay against "ldg": the
    # loader warp TMA-loads both rows one tile ahead into a two-stage SMEM ring behind a full / empty barrier pair; the chain's delta launch
    # writes the two rows pre-scaled with the SAME two multiplies, so every stat word is bitwise the ldg form's and dQ / dK / dV are bitwise
    # between the forms; the workspace carries the q-tile-padded pre-scaled stats for the LSE copy) | "ldg" (the previous default, selectable;
    # the form "16w_fa4" resolves to when unset: the softmax fetches them per tile with one global load per lane, scales, stores them into one
    # SMEM slab and fences the 8 warps with a named-barrier pair).  Perf-only; "tma_ring" NAMED declines by name under "16w_fa4"; served
    # together with ``debug_clk``.  Typed (``config_sm100_d128_accum_2cta_stail.CFG.STATS_FORMS``); read only on the 2-CTA route; the compile
    # key carries it.
    stats_form: "str | None" = None

    def __init__(
        self,
        sample_q,
        sample_k,
        sample_v,
        sample_o,
        sample_do,
        sample_stats,
        sample_dq,
        sample_dk,
        sample_dv,
        sample_sink=None,
        sample_dsink=None,
        sample_bias=None,
        sample_dbias=None,
        is_causal=False,
        causal_bottom_right=False,
        window_size_left=None,
        window_size_right=None,
        deterministic=False,
        scale_softmax=None,
        tile_m=None,
        tile_n=None,
        seq_kv_lens_present=False,
        seq_q_lens_present=False,
        thd=False,
        max_total_seq_len_q=None,
        max_total_seq_len_kv=None,
        thd_stats_token_major=False,
        thd_stats_head_stride=None,
        thd_ragged_offsets=None,
    ):
        self._thd_ro_spec = dict(thd_ragged_offsets or {})
        super().__init__(
            sample_q,
            sample_k,
            sample_v,
            sample_o,
            sample_do,
            sample_stats,
            sample_dq,
            sample_dk,
            sample_dv,
            sample_sink,
            sample_dsink,
            sample_bias,
            sample_dbias,
            is_causal,
            causal_bottom_right,
            window_size_left,
            window_size_right,
            deterministic,
            scale_softmax,
            tile_m,
            tile_n,
            seq_kv_lens_present,
            seq_q_lens_present,
            thd,
            max_total_seq_len_q,
            max_total_seq_len_kv,
            thd_stats_token_major,
            thd_stats_head_stride,
        )

    def _initialize_implementation(self):
        self._prepared = None
        self._implementation = None
        self._cta_pair = None  # the recorded per-S heuristic resolution (grouped THD only; prepared_sm100_d128_grouped_2cta.cta_pair_resolution)

    def _uses_grouped_heads(self):
        return self.q_desc.ndim == 4 and self.k_desc.ndim == 4 and self.q_desc.shape[1] != self.k_desc.shape[1]

    def check_support(self):
        if self._uses_grouped_heads():
            from .prepared_sm100_d128_grouped_2cta import check_support_selected

            return check_support_selected(self)
        from .d128_grouped_contract import DKV_ACCUM_DEFAULT, resolve_dkv_accum
        from .prepared_sm100_d128_grouped_2cta import cta_pair

        if resolve_dkv_accum(getattr(self, "dkv_accum", None)) != DKV_ACCUM_DEFAULT:
            # the field names a GQA group accumulation: the MHA routes have no group to accumulate -- a typed decline, by name
            raise NotImplementedError(
                f"{self._NAME}: dkv_accum 'fp32_reduce' requires grouped heads (Hq > Hkv) on the THD chain; the MHA routes have no group to accumulate"
            )
        if cta_pair(self) == "on":
            # the forced pair never falls through to a 1-CTA body: a typed decline, by name (auto / off take the MHA path: no heuristic there)
            raise NotImplementedError(f"{self._NAME}: cta_pair 'on' requires grouped heads (Hq > Hkv); the MHA path has no pair body")
        if self.thd:
            from .prepared_sm100_d128_thd import check_support

            return check_support(self)
        if self._thd_ro_spec:
            raise NotImplementedError(f"{self._NAME}: ragged offsets require THD")
        from cudnn import _pybind_module
        from cudnn.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old
        from cudnn.frost.device import compute_capability, resolve_device
        from .prepared import ROLES, _same_geometry

        def require(condition, reason):
            if not condition:
                raise NotImplementedError(f"{self._NAME}: {reason}")

        installed, version = cutedsl_state()
        require(installed and not cutedsl_too_old(version), f"requires nvidia-cutlass-dsl >= {'.'.join(map(str, CUTEDSL_MIN_VERSION))}; found {version}")
        require(self.q_desc.device.type == "cuda", "Q must reside on CUDA")
        cc = compute_capability(resolve_device(self.q_desc.device))
        require(100 <= cc[0] * 10 + cc[1] <= 106, f"requires SM100-106; got {cc}")
        arch_error = cutedsl_arch_requirement_error(cc)
        require(arch_error is None, arch_error)
        binder = getattr(_pybind_module, "_SdpaBwdBinder", None)
        require(getattr(binder, "supports_zero_workspace", False), "requires a frontend extension with zero-workspace backward binding support")
        require(not self.thd, "THD / ragged is not supported")
        require(not self.seq_q_lens_present and not self.seq_kv_lens_present, "padding masks are not supported")
        require(not self.causal_bottom_right, "bottom-right causal is not supported")
        require(self.window_size_right is None, "right-band widening is not supported")
        require(
            self.window_size_left is None or (type(self.window_size_left) is int and self.window_size_left >= 0),
            "sliding window radius must be a nonnegative integer",
        )
        require(all(d is None for d in (self.sink_desc, self.dsink_desc, self.bias_desc, self.dbias_desc)), "sink / dSink and bias / dBias are not supported")
        require(self.tile_m is None and self.tile_n is None, "the kernel has fixed tile geometry")
        for role in ROLES[:9]:
            desc = getattr(self, role + "_desc")
            require(desc.ndim == 4 and all(int(n) > 0 for n in desc.shape), f"{role} requires positive rank-four dimensions")
            require(desc.device == self.q_desc.device, f"{role} must reside on {self.q_desc.device}")
        b, h, sq, d = self.q_desc.shape
        bk, hk, sk, dk = self.k_desc.shape
        require(d == dk and d in (64, 128), "D_QK and D_V must equal 64 or 128")
        require(b == bk and h == hk, "Q and KV require equal batch and head counts (dense MHA)")
        try:
            validate_dense_bounds(b, h, sq, sk, self.window_size_left)
        except ValueError as error:
            require(False, str(error))
        self.dtype = self.q_desc.dtype
        require(self.dtype in (torch.bfloat16, torch.float16), "only BF16 and FP16 are supported")
        for role in ROLES[:9]:
            desc = getattr(self, role + "_desc")
            if role == "stats":
                require(tuple(desc.shape) == (b, h, sq, 1) and desc.dtype == torch.float32, "Stats must be FP32 (B, H, S_q, 1)")
                require(desc.is_contiguous(), "Stats must be contiguous")
            else:
                s = sk if role in ("k", "v", "dk", "dv") else sq
                expected = (b, h, s, d)
                require(tuple(desc.shape) == expected and desc.dtype == self.dtype, f"{role} must have shape {expected} and Q's dtype")
                require(_same_geometry((desc.shape, desc.stride), (expected, (s * h * d, d, h * d, 1))), f"{role} must use compact BSHD storage")
        self.batch_size, self.h_q, self.h_kv = b, h, h
        self.s_q_max, self.s_k_max = sq, sk
        self.head_dim_qk = self.head_dim_v = d
        self.scale_softmax = d**-0.5 if self.scale_softmax is None else float(self.scale_softmax)
        require(math.isfinite(self.scale_softmax) and self.scale_softmax != 0, "attn_scale must be finite and nonzero")
        window_in_envelope = self.window_size_left is None or (d == 64 and self.window_size_left + 1 + int(sk) + 2 * 128 <= 2**31 - 1)
        self._implementation = (
            RECOMPUTE
            if self.deterministic or not window_in_envelope
            else select_implementation(cc, version, int(b), int(h), int(sq), int(sk), self.scale_softmax, d_qk=int(d), d_v=int(d))
        )
        self._is_supported = True
        return True

    def compile(self):
        if self._uses_grouped_heads():
            from .prepared_sm100_d128_grouped_2cta import compile_plan_selected

            self._ensure_support_checked()  # resolves cta_pair and records the chosen body first
            self._prepared = compile_plan_selected(self)
            self._compiled_kernel = self._prepared.artifact
            return
        if self.thd:
            from .prepared_sm100_d128_thd import compile_plan

            self._ensure_support_checked()
            self._prepared = compile_plan(self)
            self._compiled_kernel = self._prepared.artifact
            return
        from .prepared_sm100_d128 import compile_plan

        self._ensure_support_checked()
        self._prepared = compile_plan(self)
        self._compiled_kernel = self._prepared.artifact

    def scratch_workspace_bytes(self):
        self._ensure_support_checked()
        if self.h_q != self.h_kv:
            return self._grouped_workspace_bytes
        if self.thd:
            return self._thd_workspace_bytes
        if self._implementation == ACCUMULATION:
            return workspace_layout(self.batch_size, self.h_q, self.s_q_max, self.head_dim_qk)[1]
        return 0

    def execute(
        self,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        do_tensor,
        stats_tensor,
        dq_tensor,
        dk_tensor,
        dv_tensor,
        scale_softmax=None,
        workspace=None,
        current_stream=None,
        **optional,
    ):
        from cudnn._device import ensure_current_context
        from cudnn.sdpa.fwd.prepared import facts_of_tensor
        from .prepared import ROLES, execute

        if self._prepared is None:
            raise RuntimeError(f"{self._NAME}: compile the plan before execute")
        if self.thd:
            from .prepared_sm100_d128_thd import execute_standalone

            execute_standalone(
                self,
                (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, stats_tensor, dq_tensor, dk_tensor, dv_tensor),
                workspace,
                current_stream,
                scale_softmax,
                optional,
            )
            return
        if any(value is not None for value in optional.values()):
            raise ValueError(f"{self._NAME}: optional operands were not compiled into this plan")
        tensors = (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, stats_tensor, dq_tensor, dk_tensor, dv_tensor)
        facts = dict(zip(ROLES[:9], map(facts_of_tensor, tensors)))
        geometry = tuple((getattr(self, role + "_desc").shape, getattr(self, role + "_desc").stride) for role in ROLES[:9])
        spec = self._prepared
        workspace_ptr = 0
        if spec.workspace_bytes:
            ws = facts_of_tensor(workspace)
            if ws is None or ws.dtype != "uint8" or not ws.contiguous or ws.span < spec.workspace_bytes or ws.device != (2, spec.device_index):
                raise ValueError(f"{self._NAME} requires {spec.workspace_bytes} bytes of contiguous uint8 workspace on CUDA device {spec.device_index}")
            workspace_ptr = ws.ptr
        stream = int(torch.cuda.current_stream(self.q_desc.device).cuda_stream if current_stream is None else current_stream)
        context = nullcontext() if torch.cuda.current_device() == spec.device_index else torch.cuda.device(spec.device_index)
        with context:
            ensure_current_context(stream, spec.device_index)
            execute(spec, facts, workspace_ptr, stream, scale=scale_softmax, geometry=geometry)
