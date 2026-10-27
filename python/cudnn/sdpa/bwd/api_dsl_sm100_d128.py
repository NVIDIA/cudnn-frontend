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
    # The per-S HEURISTIC of the grouped THD route (2026-10-27): which body serves a BF16 grouped THD request -- the one-CTA body
    # (``bprop_d128_accum_grouped_f16``, implementation ``accumulation_grouped_thd``) or the cta_group::2 PAIR body (``tmem_map``'s body,
    # implementation ``accumulation_grouped_thd_2cta``; five launches either way).  A typed op-level field (``d128_grouped_2cta_contract``:
    # ``CTA_PAIR_MODES``): "auto" = THE default = the per-S rule -- the pair iff both declared S_q and S_kv >= ``CTA_PAIR_MIN_SEQ`` (8192: the
    # smallest S where the pair won the in-session step on cc 10.0), cc in ``CTA_PAIR_CCS``, a cluster of two admitted by the device's occupancy
    # query, and every support predicate of the pair body; else the one-CTA body -- with the heuristic's pair body ``CTA_PAIR_BODY`` (s_tail)
    # materialized as ``tmem_map`` when that field is unset; "off" = the one-CTA body always; "on" = the pair body at any S, declining BY NAME
    # when a predicate fails (``tmem_map`` keeps its own default there).  Resolved once in ``check_support`` (``prepared_sm100_d128_grouped_2cta``
    # .resolve_cta_pair_body) and recorded on the instance (``cta_pair_resolution``: the token ``pair=<mode>:<body>``); the day-1 bool
    # ``datapath_2cta`` is folded into it.  No environment variable; set on the INSTANCE before ``check_support()``; the plan cache keys the
    # two chains apart by body (the one-CTA rendering is untouched: its md5 pins hold).
    cta_pair: str = "auto"
    # ARM 1 bring-up A/B field (2026-10-26, code-writer step 2): the pair body's leader MMA issue order, a ``CFG.PAIR_MMA_ORDERS`` name
    # (``config_sm100_d128_accum_2cta``: "s_first" = THE default, measured faster 2026-10-27; "dk_first" = S^T(t+1) after dK(t), the A/B arm); ``None`` = the
    # config default.  Same numerics under both (pinned bitwise on dK / dV); read only on the 2-CTA route; typed (ValueError by name).
    # On the s_tail map (``tmem_map = "s_tail"``, code-writer step 6, 2026-10-27): "fa4" = THE default | "fa4b" = dQ(t-1) issued after dV(t)
    # (the exchange-cover order; the same barrier table; bitwise on dK / dV); resolved PER MAP, s_first / dk_first decline there by name.
    pair_mma_order: "str | None" = None
    # ARM 1 bring-up A/B field (2026-10-27, code-writer step 3): the pair body's cga2 TMA load form, a ``CFG.TMA_LOAD_FORMS`` name
    # ("multicast" = THE default, the validated one-CTA-mask `.multicast::cluster.cta_group::2` load; "plain" = the same load without
    # a mask, FAv4's `.cta_group::2` form, the A/B arm ``frost_thd_2cta_plain_tma``); ``None`` = the config default.  Same bytes on the
    # same leader mbarrier under both (bitwise on dK / dV); read only on the 2-CTA route; typed (ValueError by name).  UNMEASURED.
    tma_load_form: "str | None" = None
    # DIAGNOSTIC probe of the pair body (2026-10-27, code-writer step 4; optimizer day-2 plan rank 1): a ``CFG.DIAG_2CTA_NAMES`` name
    # ("dq_reduce_add_off" = D2-pair: the store warp skips the four fp32 dQ reduce-add issues; "dp_empty_wait_off" = D7: the leader MMA
    # warp does not wait dp_empty before dP^T(t+1); "ds_gates_off" = D8: the leader does not wait the three dS gates before dQ(t)) or
    # ``None`` = THE default (no probe; rendered PTX-identical, pinned).  Every barrier arrive site stays (the barrier table is unchanged);
    # the OUTPUT IS WRONG BY DESIGN -- a timing ceiling set on the adapter INSTANCE by the probe driver, never a knob, never shipped
    # armed, refused by the driver's `--validate`; read only on the 2-CTA route; typed (ValueError by name); keyed apart in the plan cache.
    diag_2cta: "str | None" = None
    # KNOBS of the pair body (2026-10-27, code-writer step 4; perf-only, the same function BITWISE under every value; read only on the 2-CTA
    # route; typed, ValueError by name; ``None`` = the config default = the shipped rendering, PTX-identical; UNMEASURED until their A/B/A row):
    #   store_wait_form: a ``CFG.STORE_WAIT_FORMS`` name -- "retry" (default: the store warp's six mbarrier waits in the declared try_wait
    #     retry form) | "two_instr" (the hint-less 2-instruction spin the MMA warp's waits render as; the arm ``frost_thd_2cta_two_instr_wait``)
    #   reg_split: a ``CFG.REG_SPLIT_NAMES`` name -- "144/144/80" (default) | "136/136/104" (FA4's working point, M5'; the arm
    #     ``frost_thd_2cta_regs_136``); the three setmaxnreg counts, the entry formula asserted in the config.
    store_wait_form: "str | None" = None
    reg_split: "str | None" = None
    # ARM (ii) selector (2026-10-27, code-writer step 5): the pair BODY by its TMEM map -- "dp_tail" (default; the first body
    # ``bprop_d128_accum_grouped_2cta_f16``: dQ image in dP^T's dead half, s_first order; rendering byte-identical when unset) |
    # "s_tail" (``bprop_d128_accum_grouped_2cta_stail_f16``: dQ image in S^T's dead tail, FAv4's rotated order `fa4`, the dedicated sDQ
    # ring, 34 barriers).  Typed (``config_sm100_d128_accum_2cta_stail.TMEM_MAPS``); ``pair_mma_order`` / ``diag_2cta`` are resolved PER
    # MAP (fa4 | fa4b only on s_tail; dp_empty_wait_off declined there); the compile key carries it; the driver arms ``frost_thd_2cta_stail`` /
    # ``frost_thd_2cta_stail_fa4b``.
    tmem_map: "str | None" = None
    # ARM (ii) dQ ROUTE field (2026-10-27, code-writer step 8; optimizer 05:15 rank 1 "#3-dry"): how the fp32 dQ partial leaves the s_tail pair
    # body -- "tma" (default; the sDQ ring STS + the store warp's four reduce-adds per tile; rendering byte-identical when unset) | "red_global"
    # (`red.global.add.v4.f32` straight from the epilogue's dQ tcgen05.ld registers; the ring PROTOCOL kept, no barrier row moves) |
    # "red_global_pair" (the same reds with lane pairs forming full 32-B L2 sectors through shfl.sync).  Perf-only on dK / dV (bitwise under
    # every value); dQ is a non-deterministic fp32 sum under every route (the fp64 gate).  Typed (``config_sm100_d128_accum_2cta_stail.CFG.DQ_ROUTES_PAIR``,
    # s_tail only: declines by name on dp_tail); read only on the 2-CTA route; the compile key carries it; the driver arms
    # ``frost_thd_2cta_stail_redq`` / ``frost_thd_2cta_stail_redq_pair``.  UNMEASURED until their A/B/A row.
    dq_route_pair: "str | None" = None
    # STAMPS lever of the s_tail pair body (2026-10-27, code-writer day 3; optimizer 07:45 rank 0; the d512 bodies' `debug_clk` pattern): a typed
    # bool, default False = the shipped rendering (byte-identical; pinned).  True = every mbarrier wait of ONE representative warp per role is
    # bracketed by %clock64 and accumulated per (role, barrier) in SMEM, plus the epilogue's per-tile hand-off segments and the ship -> recv
    # handshake, dumped per CTA at exit into a workspace region the adapter carves only under the lever (`scratch_workspace_bytes()` grows;
    # `prepared_sm100_d128_grouped_2cta.debug_clk_records(api, workspace)` reads it).  Output CORRECT (dK / dV bitwise, the test module's twin);
    # a timing DIAGNOSTIC, never a knob (it changes the rendering), read only on the 2-CTA route with `tmem_map = "s_tail"` (declines by name
    # elsewhere); the compile key carries it.  The driver arm `frost_thd_2cta_stail_clk` + its decoder.
    debug_clk: bool = False
    # Lever L5 field (2026-10-27, code-writer D; optimizer day-3 rank 6): the s_tail body's softmax MASK FORM -- "fsel" (default; today's rendering:
    # the keep-word mask applied as one select per cell on EVERY q tile; rendering byte-identical when unset) | "predicated" (the same mask body
    # under a CTA-uniform runtime branch that runs it only on a tile that can hold a masked cell: the causal diagonal straddle, the last partial q
    # tile, kv rows at or past seq_k; interior tiles pack the exp2 values straight) | "diag_mask_off" (DIAGNOSTIC: the mask body skipped on EVERY
    # tile = the lever's ceiling; OUTPUT WRONG BY DESIGN on the diagonal / edge tiles, set on the adapter INSTANCE by the probe driver, never a
    # knob, never shipped armed, refused by the driver's `--validate`).  Bitwise on dQ / dK / dV between fsel and predicated (a masked P / dS cell
    # is exactly 0 in both forms).  Typed (``config_sm100_d128_accum_2cta_stail.CFG.MASK_FORMS``, s_tail only: declines by name on dp_tail); read
    # only on the 2-CTA route; the compile key carries it; the driver arms ``frost_thd_2cta_stail_l5`` / ``frost_thd_2cta_stail_l5off``.  UNMEASURED
    # until its A/B/A row.
    mask_form: "str | None" = None
    # C2b op-level field (2026-10-27, code-writer C; optimizer day-3 rank 5): how the GQA group's dK / dV are accumulated on the grouped THD chain --
    # "bf16_partials" (``None`` = the same, THE default: one bf16 partial per Q head + the fp32 ``dkv_reduce_thd_origins`` fold launch, a fixed
    # summation order; the rendering is byte-identical when unset) | "fp32_reduce" (a per-KV-head fp32 dK / dV workspace carved inside the partial
    # regions and zero-filled by ``delta_zero_thd``; every Q-head owner reduce-adds its fp32 tile with ``cp.reduce.async.bulk.tensor`` add.f32 from
    # the same alias staging as the dQ slices, two rounds per item (dV, then dK); the fold launch gone; the one bf16 rounding folded into
    # ``cast_dq_thd``: FOUR launches).  A ROUNDING CHANGE (fp32 accumulation across the group in hardware order vs 8 bf16-rounded partials summed
    # in a fixed order; dQ is untouched), hence an op ATTRIBUTE the caller sets, never a knob; `use_deterministic_algorithm` is already declined on
    # this route.  Both bodies serve it (the 1-CTA body and the s_tail pair body; the dp_tail pair body, the dense grouped chain and the MHA routes
    # decline it BY NAME).  Typed (``d128_grouped_contract.DKV_ACCUM_FORMS``); the compile key carries it; the driver arms
    # ``frost_thd_adapter_fp32`` / ``frost_thd_2cta_stail_fp32``.  UNMEASURED until their A/B/A row.
    dkv_accum: "str | None" = None
    # X-FA4 ROLE SHAPE field (2026-10-27, code-writer B on optimizer day-3 rank 1): FAv4's role shape on the s_tail pair body -- "16w" (default; today's
    # rendering byte-identical when unset: the epilogue warps 8-11 stage dS from TMEM, ship the exchange and read dQ out) | "16w_fa4" (the SOFTMAX
    # writes sDS own + XCHG straight from its dS^T registers and its lead warp ships; warps 8-11 become the REDUCE group (dQ readout + the fp32 slice
    # rounds + the item-end dV / dK staging); ds_tmem + dp_empty deleted (32 barriers), a softmax dq_done WAR wait added; TMEM / SMEM / the register split
    # unchanged).  Perf-only on dK / dV (bitwise under both values: the same accumulation chain per GEMM); dQ by the fp64 gate.  Typed
    # (``config_sm100_d128_accum_2cta_stail.CFG.WARP_ROLES``, s_tail only: declines by name on dp_tail; under "16w_fa4" the red dQ routes and
    # ``diag_2cta="ds_gates_off"`` decline by name); read only on the 2-CTA route; the compile key carries it; the driver arm
    # ``frost_thd_2cta_stail_xfa4``.  UNMEASURED until its A/B/A row.
    warp_roles: "str | None" = None

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
        # The recompute body owns each gradient tile in one CTA and visits
        # reduction tiles in a fixed order. The accumulation body uses unordered
        # FP32 dQ reductions, so it cannot serve a deterministic request.
        self._implementation = (
            RECOMPUTE
            if self.deterministic or d != 128 or self.window_size_left is not None
            else select_implementation(cc, version, int(b), int(h), int(sq), int(sk), self.scale_softmax)
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
            return workspace_layout(self.batch_size, self.h_q, self.s_q_max)[1]
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
