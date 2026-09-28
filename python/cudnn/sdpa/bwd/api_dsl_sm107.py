# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""cuDNN-frontend adapters over the FROST SM107 (Rubin) d=256 SDPA backward chains.

Two rows share this module -- ``sdpa_bwd_sm107`` (bf16 / fp16, :class:`SdpaBwdDslSm107`)
and ``sdpa_bwd_sm107_fp8`` (per-tensor FP8 E4M3, :class:`SdpaBwdDslSm107Fp8`) -- because
they share one chain shape.  Each is a TWO-kernel backward around a bf16 / fp16 dS
workspace, followed by the two gradient GEMMs and a fold:

    stage 1  delta = rowsum(dO * O)                      bprop_chain_common.dot_do_o{,_scaled}_host
    stage 2  dV (in TMEM, stored per Q head) + dS -> a   sm107/bprop_d256_{f16,fp8}.py
             [B, H_chunk, S_kv, S_q] GMEM workspace
    stage 3  dK = dS . Q,  dQ = dS^T . K                 bprop_matmul_blackwell.py (two renderings)
    stage 4  GQA fold of the per-Q-head dK / dV partials  dkv_reduce_host (half) /
             (+ descale, amax, scale, cast on the fp8 row)  fold_quant_host (fp8)

The workspace is KV-MAJOR (``[.., S_kv, S_q]``, q contiguous) -- the layout the stage-2
kernel writes without a transpose -- so the stage-3 operand majors are the OPPOSITE of
the SM100 chain's ``[S_q, S_kv]`` workspace: dK reads dS as ``A[kv, q]`` K-major
(``a_is_m_major=False``) and dQ reads dS^T as ``A[q, kv]`` M-major (``a_is_m_major=True``).
The causal K-trim MODES do not flip with the layout (they follow which axis is the
output row): dK trims the low q tiles (``CAUSAL_K_LO``), dQ the high kv blocks
(``CAUSAL_K_HI``), rounded to the kernel's 256-row kv block.  Under any mask the trim
is an optimization only: the adapter zero-fills the workspace once, so a tile the
stage-2 kernel skips reads as zero (``_zero_ws``).

Shapes: the kernels compile with EVERY extent concrete and TMA-strided, and require
``S_q % 128 == 0`` and ``S_kv % 256 == 0``.  A graph that is not a multiple is served by
padding: Q / dO (and the LSE, with ``+inf`` so ``P = 0``) are staged into zero-filled
padded copies carved from the workspace; K / V likewise, and a padded S_kv selects the
kernels' padded-mask specialization with the uniform real length, so every padded kv
row's dS / dV is exactly zero.  The GEMMs then read real-extent slices and write the
caller's tensors directly; only a padded-kv GQA graph folds through a padded staging
copy.  Everything the chain needs is carved from the caller's workspace in one fixed
order (:meth:`_scratch_plan`), so ``scratch_workspace_bytes()`` is a build-time function.

One exception, declined rather than served wrong: **bottom-right causal on the fp8 row
needs ``S_q % 128 == 0``.**  The bottom-right diagonal is ``S_kv - S_q`` in REAL rows.
The f16 body takes the real lengths (``sq_real`` / ``skv_real`` on its ``compile()``,
``SQ_REAL`` in its problem_size); the fp8 body's ABI has no ``seqlen_q_real`` -- it
derives the diagonal and the q-tile trim from the PADDED compile extent ``SQ`` (its kv
term IS the real length, the runtime ``seqlen_kv_real``), so a ragged S_q would shift
both by ``pad - S_q`` rows: finite, wrong dQ / dK / dV near the diagonal, no crash.  The
row declines it at eligibility (``Capabilities.bottom_right_s_q_multiple``) and
:meth:`SdpaBwdDslSm107Fp8._check_support_family` backstops.  A ragged S_kv is fine on
both rows.  Follow-up: thread ``seqlen_q_real`` through the fp8 body like the f16 one
and drop the decline.

The dS workspace is the dominant allocation (``B * H * S_kv * S_q * 2`` bytes): heads
(and, on the half row, batches -- the fp8 body has no ``batch_base``) are chunked to
fit ``_SM107_WS_BUDGET_BYTES`` and the chain loops over chunks with ``head_base`` /
``batch_base``.

Both rows are PREPARED launches (``bwd/prepared.py``; ``bwd/prepared_sm107.py`` +
``kernels/sm107/prepared_host.py``): ``compile()`` builds ONE pointer-host artifact that
runs the whole chain -- the padding copies, the ``seq_kv`` fill and the dS zero-fill,
``dot``, every chunk's main kernel + stage-3 GEMMs, the fold (and the fp8 upcast / fold +
quantize passes) -- from device pointers and the caller's workspace, and records it as
``self._prepared`` (a ``BwdLaunchSpec``).  The graph plan binds the normalized variant pack
straight into it (``engines.lower_dsl_bwd*`` -> ``PreparedBwdLaunch``); :meth:`execute` is
the standalone twin over torch tensors.  No torch op runs on the execute path.

FP8 (cuDNN ``sdpa_fp8_backward``): the twelve scalar descales / scales are 1-element
fp32 DEVICE tensors, read by the kernels -- never folded on the host.  Stage 2 consumes
descale_q/k/v/dO/s and scale_s, publishes dS in TRUE units (attn_scale folded) to the
bf16 workspace and its per-Q-head dV in bf16 (``dtype_o = BF16``: the pre-quantization
value), and folds ``amax_dP`` in-kernel.  The GEMMs run at bf16 over Q / K upcast EXACTLY
from e4m3, so their outputs still carry the operand's descale: stage 4 applies
``descale_q`` (dK) / ``descale_k`` (dQ), folds ``amax_dQ / dK / dV`` over the fp32
value, applies ``scale_dQ / dK / dV`` and casts to the graph's gradient dtype (E4M3,
or bf16 / fp16 with scale 1.0).  ``descale_dP`` / ``scale_dP`` are accepted and unused:
dS is never quantized to fp8 on this chain (plan Q2(b)).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import TensorDesc
from cudnn.frost.template_loader import load_template
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_FP16
from cudnn.sdpa.bwd import config_sm107 as _cfg
from cudnn.sdpa.bwd import prepared_sm107 as _prepared
from cudnn.sdpa.bwd.api_dsl import SdpaBwdDsl, _SM100_MATMUL_FILE, _SM100_WS_BUDGET_BYTES, _sm100_kernel_path
from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, CAUSAL_K_NONE, MatmulTemplateParams, vec_bytes_epi_for
from cudnn.sdpa.fwd.api_dsl import ws_align

_SM107_D = 256
# The bodies' compile geometry: the q loop walks 128-row q tiles, a cga2 pair owns a 256-row kv block
# (``config_sm107.q_pad_rows`` / ``kv_pad_rows``).  The adapter pads S_q / S_kv up to these and stages
# the padded operands (module doc).  ``_SM107_Q_PAD`` is also the fp8 row's bottom-right alignment
# claim (``engines.Capabilities.bottom_right_s_q_multiple``; pinned equal by the fp8 suite).
_SM107_Q_PAD = 128
_SM107_KV_PAD = 256
_SM107_KERNEL_FILES = {_cfg.FAMILY_F16: "sm107/bprop_d256_f16.py", _cfg.FAMILY_FP8: "sm107/bprop_d256_fp8.py"}
_SM107_TEMPLATE_TAGS = {_cfg.FAMILY_F16: "sdpa_bwd_sm107_main_f16", _cfg.FAMILY_FP8: "sdpa_bwd_sm107_main_fp8"}
_SM107_MM_TAGS = {"dk": "sdpa_bwd_sm107_mm_dk", "dq": "sdpa_bwd_sm107_mm_dq"}
# Same budget as the SM100 chain: above it the chunk shrinks and the chain runs more
# launches over the same total work.
_SM107_WS_BUDGET_BYTES = _SM100_WS_BUDGET_BYTES
# Stage-3 causal K-trim.  True = trimmed renderings (what ships).  False = both GEMMs
# rendered ``CAUSAL_K_NONE`` -- the correctness pin's twin (bitwise-equal gradients,
# since the zero-filled workspace makes the trim an optimization) and the A/B for its
# perf value on Rubin.  A module constant, not a knob: it must never differ per plan.
STAGE3_CAUSAL_TRIM: bool = True
_DTYPE_CODE = {torch.bfloat16: DTYPE_BF16, torch.float16: DTYPE_FP16, torch.float8_e4m3fn: DTYPE_E4M3}


def _sm107_chunks(b: int, h_q: int, group: int, s_q_pad: int, s_kv_pad: int, bpe_ds: int, budget: int = _SM107_WS_BUDGET_BYTES, batch_chunking: bool = True):
    """``(b_chunk, qh_chunk)`` for one stage-2 launch: the LARGEST (batch, head) chunk whose dS
    workspace ``b_chunk * qh_chunk * S_kv * S_q * bpe`` fits ``budget``.

    Divisors, not floors, so no launch has a ragged tail and one artifact serves every
    chunk; the head chunk is a multiple of the GQA group so a chunk's Q heads map onto
    whole KV heads (``config_sm107.validate_head_chunk``).  Heads shrink first (a
    smaller head chunk keeps the batch loop out of the way); the batch is chunked only
    when even ``group`` heads at the full batch do not fit, and only on the half row
    (``batch_chunking``): the fp8 body walks the whole batch in-grid.  When nothing
    fits the smallest legal chunk is returned and the size is still honest.
    """
    per = s_q_pad * s_kv_pad * bpe_ds
    heads = [c for c in range(h_q, 0, -1) if h_q % c == 0 and c % group == 0]
    batches = [c for c in range(b, 0, -1) if b % c == 0] if batch_chunking else [b]
    for bc in batches:
        for hc in heads:
            if bc * hc * per <= budget:
                return bc, hc
    return batches[-1], group


def _stage3_params(dtype_code: int, causal: bool, shift: int, gran: int, trim: Optional[bool] = None):
    """The two stage-3 renderings ``(dK, dQ)`` for the KV-MAJOR ``[S_kv, S_q]`` workspace.

    dK = dS . Q  : A = dS[kv, q]   -- M = kv, K = q, q contiguous -> K-major; K starts at kv's block (LO)
    dQ = dS^T . K: A = dS^T[q, kv] -- M = q,  K = kv, q contiguous -> M-major; K ends after q's block (HI)

    ``shift`` is how far the written band extends past the plain ``kv <= q`` diagonal
    (bottom-right: ``S_kv - S_q``); ``gran`` the kernel's kv write block (256).  ``trim``
    defaults to the module constant, read at CALL time so the bitwise pin can flip it.
    """
    if trim is None:
        trim = STAGE3_CAUSAL_TRIM
    lo = CAUSAL_K_LO if (causal and trim) else CAUSAL_K_NONE
    hi = CAUSAL_K_HI if (causal and trim) else CAUSAL_K_NONE
    common = dict(b_is_n_major=True, causal_gran=gran, causal_shift=shift, vec_bytes_epi=vec_bytes_epi_for(_SM107_D, 2), dtype_qkv=dtype_code)
    return (
        MatmulTemplateParams(a_is_m_major=False, causal_mode=lo, **common),
        MatmulTemplateParams(a_is_m_major=True, causal_mode=hi, **common),
    )


def _bshd_physical_ok(desc: TensorDesc) -> bool:
    """True when a logical-BHSD desc sits on compact BSHD storage (the rows' layout claim)."""
    b, h, s, d = (int(x) for x in desc.shape)
    return tuple(int(x) for x in desc.stride) == (s * h * d, d, h * d, 1)


class SdpaBwdDslSm107(SdpaBwdDsl):
    """``sdpa_bwd_sm107``: d_qk = d_v = 256, bf16 / fp16, on the Rubin line (cc 10.7-11.9)."""

    _FAMILY = _cfg.FAMILY_F16
    _NAME = "sdpa_bwd_sm107"
    _BATCH_CHUNKING = True
    _IO_DTYPES = (torch.bfloat16, torch.float16)

    # --- geometry ------------------------------------------------------------------
    def _initialize_implementation(self) -> None:
        q_shape = tuple(int(x) for x in self.q_desc.shape)  # logical BHSD
        k_shape = tuple(int(x) for x in self.k_desc.shape)
        self.batch_size, self.h_q, self.s_q_max, self.head_dim_qk = q_shape
        self.h_kv, self.s_k_max = int(k_shape[1]), int(k_shape[2])
        self.head_dim_v = int(tuple(self.v_desc.shape)[3])
        self.dtype = self.q_desc.dtype
        self.grad_dtype = self.dq_desc.dtype
        self._bpe = self.dtype.itemsize
        # dS workspace dtype: the io dtype on the half chain, bf16 on the fp8 chain (plan Q2(b)).
        self._ds_dtype = self.dtype if self._FAMILY == _cfg.FAMILY_F16 else torch.bfloat16
        self._bpe_ds = 2
        # attn_scale is OPTIONAL on the graph (see the SM100 adapter for the story).
        if self.scale_softmax is None or self.scale_softmax == 0.0:
            self.scale_softmax = 1.0 / math.sqrt(self.head_dim_qk)
        # Tile-rounded COMPILE shape: the q loop walks 128-row q tiles, a cga2 pair owns a
        # 256-row kv block.  The padded operands are staged (zero-filled), see the module doc.
        self._sq_pad = -(-self.s_q_max // _SM107_Q_PAD) * _SM107_Q_PAD
        self._skv_pad = -(-self.s_k_max // _SM107_KV_PAD) * _SM107_KV_PAD
        self._q_padded = self._sq_pad != self.s_q_max
        self._kv_padded = self._skv_pad != self.s_k_max
        self._gqa_group = self.h_q // max(self.h_kv, 1)
        self._b_chunk, self._qh_chunk = _sm107_chunks(
            self.batch_size, self.h_q, self._gqa_group, self._sq_pad, self._skv_pad, self._bpe_ds, batch_chunking=self._BATCH_CHUNKING
        )
        # Under ANY mask the stage-2 kernel skips the q tiles a kv block does not attend
        # (causal: below the block; SWA: above it), and the stage-3 GEMMs read those
        # tiles (the dQ / dK trim is an optimization, not a bound -- and SWA has none).
        # Zero-filled once per execute, the skipped tiles contribute exactly 0.
        self._zero_ws = bool(self.is_causal or self.window_size_left is not None)
        self._compiled = None
        self._prepared = None

    # --- capability backstop ---------------------------------------------------------
    def check_support(self) -> bool:
        """Re-check what the Capabilities row promised (backstops, never the gate)."""
        n = self._NAME
        self._value_error_if(
            self.head_dim_qk != _SM107_D or self.head_dim_v != _SM107_D, f"{n}: d_qk = d_v = {_SM107_D} exactly; got {self.head_dim_qk} / {self.head_dim_v}"
        )
        self._value_error_if(self.h_kv < 1 or self.h_q % self.h_kv != 0, f"{n}: h_q ({self.h_q}) must be a multiple of h_kv ({self.h_kv})")
        self._value_error_if(self.dtype not in self._IO_DTYPES, f"{n}: Q/K/V/O/dO dtype {self.dtype} not served")
        self._value_error_if(self.s_q_max < 1 or self.s_k_max < 1, f"{n}: empty sequences are not served (S_q={self.s_q_max}, S_kv={self.s_k_max})")
        self._value_error_if(self.s_q_max == 1, f"{n}: s_q == 1 (decode) is out of scope for the prefill bodies")
        for name, desc in (
            ("q", self.q_desc),
            ("k", self.k_desc),
            ("v", self.v_desc),
            ("o", self.o_desc),
            ("dO", self.do_desc),
            ("dQ", self.dq_desc),
            ("dK", self.dk_desc),
            ("dV", self.dv_desc),
        ):
            self._value_error_if(not _bshd_physical_ok(desc), f"{n}: {name} must be BSHD-physical (stride order 3,1,2,0); got stride {tuple(desc.stride)}")
        st_shape, st_stride = tuple(int(x) for x in self.stats_desc.shape), tuple(int(x) for x in self.stats_desc.stride)
        self._value_error_if(
            st_shape != (self.batch_size, self.h_q, self.s_q_max, 1) or st_stride != (self.h_q * self.s_q_max, self.s_q_max, 1, 1),
            f"{n}: stats must be contiguous (B, H_q, S_q, 1); got dim {st_shape} stride {st_stride}",
        )
        # v1 declines (plan Q4): each flips together with its accept test and tracker line.
        self._value_error_if(self.thd, f"{n}: THD / ragged is not implemented")
        self._value_error_if(self.seq_kv_lens_present or self.seq_q_lens_present, f"{n}: padding masks (seq lens) are not implemented")
        self._value_error_if(
            self.deterministic, f"{n}: use_deterministic_algorithm is not claimed yet (the chain has no atomics; the two-run bitwise test decides)"
        )
        self._value_error_if(self.sink_desc is not None or self.dsink_desc is not None, f"{n}: sink / dSink are not implemented")
        self._value_error_if(self.bias_desc is not None or self.dbias_desc is not None, f"{n}: bias / dBias are not implemented")
        self._value_error_if(
            self.window_size_right not in (None, 0), f"{n}: causal right-band widening (window_right={self.window_size_right}) is not implemented"
        )
        self._value_error_if(
            self.window_size_left is not None and self.window_size_left <= 0, f"{n}: sliding window needs window_left > 0; got {self.window_size_left}"
        )
        self._value_error_if(self.causal_bottom_right and not self.is_causal, f"{n}: bottom-right alignment requires a causal mask")
        self._check_support_family()
        self._is_supported = True
        return True

    def _check_support_family(self) -> None:
        self._value_error_if(self.grad_dtype != self.dtype, f"{self._NAME}: dQ/dK/dV dtype {self.grad_dtype} must match the io dtype {self.dtype}")

    # --- workspace: ONE ordered plan, carved identically by the prepared host ----------
    def _scratch_shapes(self):
        """``[(name, shape, dtype)]`` in carve order -- every buffer the chain touches that is
        not a caller tensor, as the compact region the pointer host views it as
        (``prepared_sm107._regions``), so :meth:`scratch_workspace_bytes` and the artifact
        cannot disagree."""
        b, h, hkv, sq, skv, d = self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max, _SM107_D
        sqp, skvp = self._sq_pad, self._skv_pad
        kv_rows = skvp if self._kv_padded else skv
        gqa = self._gqa_group > 1
        plan = [
            # stage 1's delta: [B, H_q, ceil128(S_q)] fp32 -- dot_do_o writes the rounded extent, zeros past S_q
            ("delta", (b, h, sqp), torch.float32),
            # the dS workspace of one launch: [b_chunk, qh_chunk, S_kv_pad, S_q_pad], kv-major
            ("ds_ws", (self._b_chunk, self._qh_chunk, skvp, sqp), self._ds_dtype),
            # stage 2's per-batch kv lengths (read only under the padded arm) + stage 3's dead THD ABI slot
            ("seq_kv", (b,), torch.int32),
            ("desc_words", (1,), torch.int64),
        ]
        if self._q_padded:
            plan += [("q_pad", (b, sqp, h, d), self.dtype), ("do_pad", (b, sqp, h, d), self.dtype), ("lse_pad", (b, h, sqp), torch.float32)]
        if self._kv_padded:
            plan += [("k_pad", (b, skvp, hkv, d), self.dtype), ("v_pad", (b, skvp, hkv, d), self.dtype)]
        plan += self._family_scratch_shapes(kv_rows, gqa)
        return plan

    def _family_scratch_shapes(self, kv_rows: int, gqa: bool):
        b, h, hkv, d = self.batch_size, self.h_q, self.h_kv, _SM107_D
        plan = []
        # stage 2's dV per Q head: the caller's dV only when MHA and no kv padding
        if gqa or self._kv_padded:
            plan.append(("dv_part", (b, kv_rows, h, d), self.dtype))
        # stage 3's dK per Q head: the caller's dK when MHA (real rows written)
        if gqa:
            plan.append(("dk_part", (b, kv_rows, h, d), self.dtype))
            if self._kv_padded:
                # dkv_reduce folds at the workspace's row extent; the real rows are copied out
                plan += [("dk_fold", (b, kv_rows, hkv, d), self.dtype), ("dv_fold", (b, kv_rows, hkv, d), self.dtype)]
        return plan

    def _scratch_plan(self):
        """``[(name, numel, dtype)]`` -- :meth:`_scratch_shapes` flattened (the byte sum's and the tests' form)."""
        return [(name, math.prod(shape), dtype) for name, shape, dtype in self._scratch_shapes()]

    def scratch_workspace_bytes(self) -> int:
        """A pure function of the compile geometry (delta + one dS chunk + padded staging +
        GQA partials): the artifact carves all of it from the caller's buffer."""
        return sum(ws_align(numel * dtype.itemsize) for _name, numel, dtype in self._scratch_plan())

    # --- compilation -------------------------------------------------------------------
    def _template_params(self):
        return _cfg.TemplateParams(
            dtype_qkv=_DTYPE_CODE[self.dtype],
            window_right=0 if self.is_causal else None,  # set => causal; right-band widening is declined
            window_left=self.window_size_left,
            bottom_right=self.causal_bottom_right,
            # A padded S_kv selects the padded-mask arm with the UNIFORM real length, so the
            # zero-filled pad rows produce P = 0 -> dS = dV = 0 there (and, on the fp8 row,
            # stay out of the amax folds).  Dense otherwise: the arm folds out.
            seq_kv_lens_present=self._kv_padded,
            dtype_o=self._dtype_o_code(),
        )

    def _dtype_o_code(self) -> int:
        return -1  # inherit the io dtype

    def _compile_plan(self, mod, mm_dk, mm_dq):
        return _prepared.compile_plan(self, mod, mm_dk, mm_dq)

    def compile(self) -> None:
        """Plan-time JIT: the stage-2 template specialized on this graph's masks / dtype, the
        two stage-3 GEMM renderings, and the ONE pointer-host artifact that launches the
        whole chain over them (``prepared_sm107.compile_plan``), recorded as ``_prepared``."""
        self._ensure_support_checked()
        if self._compiled is not None:
            return self._compiled
        mod = load_template(_sm100_kernel_path(_SM107_KERNEL_FILES[self._FAMILY]), self._template_params(), tag=_SM107_TEMPLATE_TAGS[self._FAMILY])
        # The bodies' own geometry guards (tile multiples, chunk divisors) -- cheap, and the plan is wrong if they fire.
        _cfg.validate_head_chunk(self.h_q, self.h_kv, self._qh_chunk)
        # Stage 3 reads the dS workspace in ITS dtype and writes the io dtype of the GEMM
        # outputs -- the io dtype on the half chain, bf16 on the fp8 chain (upcast Q / K,
        # bf16 dK / dQ partials for stage 4).
        shift = (self.s_k_max - self.s_q_max) if (self.is_causal and self.causal_bottom_right) else 0
        p_dk, p_dq = _stage3_params(_DTYPE_CODE[self._ds_dtype], bool(self.is_causal), shift, _cfg.kv_pad_rows(mod.CFG))
        mm_dk = load_template(_sm100_kernel_path(_SM100_MATMUL_FILE), p_dk, tag=_SM107_MM_TAGS["dk"])
        mm_dq = load_template(_sm100_kernel_path(_SM100_MATMUL_FILE), p_dq, tag=_SM107_MM_TAGS["dq"])
        self._prepared = self._compile_plan(mod, mm_dk, mm_dq)
        self._compiled = self._prepared.artifact
        return self._compiled

    # --- execution ---------------------------------------------------------------------
    def _refuse_unclaimed(self, seq_q_lens, seq_kv_lens, sink_tensor, dsink_tensor, bias_tensor, dbias_tensor) -> None:
        for name, t in (("sink", sink_tensor), ("dSink", dsink_tensor), ("bias", bias_tensor), ("dBias", dbias_tensor)):
            self._value_error_if(t is not None, f"{self._NAME}: {name} is not implemented")
        self._value_error_if(seq_q_lens is not None or seq_kv_lens is not None, f"{self._NAME}: padding masks (seq lens) are not implemented")

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        do_tensor: torch.Tensor,
        stats_tensor: torch.Tensor,
        dq_tensor: torch.Tensor,
        dk_tensor: torch.Tensor,
        dv_tensor: torch.Tensor,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        sink_tensor: Optional[torch.Tensor] = None,
        dsink_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        dbias_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """The standalone twin of the graph plan: the same prepared artifact, bound from torch
        tensors (``prepared_sm107.execute_standalone``).  Every operand must carry the plan's
        geometry; the workspace is the caller's (``scratch_workspace_bytes()`` bytes)."""
        self._refuse_unclaimed(seq_q_lens, seq_kv_lens, sink_tensor, dsink_tensor, bias_tensor, dbias_tensor)
        self.compile()
        tensors = (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, stats_tensor, dq_tensor, dk_tensor, dv_tensor)
        _prepared.execute_standalone(self, tensors, workspace, current_stream, scale_softmax)


_FP8_GRAD_DTYPES = (torch.float8_e4m3fn, torch.bfloat16, torch.float16)


class SdpaBwdDslSm107Fp8(SdpaBwdDslSm107):
    """``sdpa_bwd_sm107_fp8``: cuDNN ``sdpa_fp8_backward`` at d = 256 on the Rubin line.

    FP8 E4M3 Q / K / V / O / dO with scalar descales in, fp32 Stats, gradients in the
    graph's dtype (E4M3 with ``scale_dQ / dK / dV``, or bf16 / fp16) plus the requested
    ``amax_dQ / dK / dV / dP`` out.  See the module doc for where each scalar is applied.
    """

    _FAMILY = _cfg.FAMILY_FP8
    _NAME = "sdpa_bwd_sm107_fp8"
    _BATCH_CHUNKING = False  # the fp8 body has no batch_base: the whole batch is in-grid
    _IO_DTYPES = (torch.float8_e4m3fn,)

    def __init__(self, *args, amax_requested=(), **kwargs) -> None:
        """``amax_requested``: the subset of ``("amax_dQ", "amax_dK", "amax_dV", "amax_dP")`` the
        graph declared as real outputs.  A plan fact: it decides which amax pointers the
        artifact binds (an unrequested one is None-specialized and its fold atomics fold
        out), so the lowering passes it at construction."""
        names = tuple(amax_requested)
        unknown = [n for n in names if n not in _prepared.FP8_AMAX]
        if unknown:
            raise ValueError(f"{self._NAME}: unknown amax outputs {unknown}; expected a subset of {_prepared.FP8_AMAX}")
        self.amax_requested = frozenset(names)
        super().__init__(*args, **kwargs)

    def _check_support_family(self) -> None:
        n = self._NAME
        self._value_error_if(self.grad_dtype not in _FP8_GRAD_DTYPES, f"{n}: dQ/dK/dV dtype {self.grad_dtype} not in {_FP8_GRAD_DTYPES}")
        self._value_error_if(
            self.dk_desc.dtype != self.grad_dtype or self.dv_desc.dtype != self.grad_dtype,
            f"{n}: dQ/dK/dV must share one dtype; got {self.dq_desc.dtype}/{self.dk_desc.dtype}/{self.dv_desc.dtype}",
        )
        for name, desc in (("k", self.k_desc), ("v", self.v_desc), ("o", self.o_desc), ("dO", self.do_desc)):
            self._value_error_if(desc.dtype != self.dtype, f"{n}: {name} is an FP8 payload and must share Q's dtype {self.dtype}; got {desc.dtype}")
        # The fp8 body derives the bottom-right diagonal (S_kv - S_q) and the q-tile trim from the PADDED
        # S_q (no seqlen_q_real in its ABI; the f16 body threads it), so a ragged S_q would be served
        # silently wrong by (pad - S_q) rows.  The row already declines this at eligibility
        # (Capabilities.bottom_right_s_q_multiple); this is the backstop.  A ragged S_kv is fine: the
        # kernel's kv term is the runtime real length.
        self._value_error_if(
            self.causal_bottom_right and self._q_padded,
            f"{n}: bottom-right causal needs S_q % {_SM107_Q_PAD} == 0 (the fp8 body derives the diagonal from the padded S_q; "
            f"its ABI has no seqlen_q_real); got S_q={self.s_q_max}; follow-up: thread seqlen_q_real like the f16 body",
        )

    def _dtype_o_code(self) -> int:
        # Stage 2 publishes dV in bf16 (the pre-quantization value) for the fold + quantize pass.
        return DTYPE_BF16

    def _family_scratch_shapes(self, kv_rows: int, gqa: bool):
        b, h, hkv, sq, skv, d = self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max, _SM107_D
        return [
            ("dv_part", (b, kv_rows, h, d), torch.bfloat16),  # stage 2's per-Q-head dV_true, bf16
            ("dk_part", (b, kv_rows, h, d), torch.bfloat16),  # stage 3's per-Q-head dS . Q8 (descale_q pending)
            ("dq_ws", (b, sq, h, d), torch.bfloat16),  # stage 3's dS^T . K8 (descale_k pending)
            ("q_bf16", (b, sq, h, d), torch.bfloat16),  # Q8 upcast EXACTLY (the bf16 GEMM's B operand)
            ("k_bf16", (b, skv, hkv, d), torch.bfloat16),  # K8 upcast EXACTLY
            ("amax_scratch", (8,), torch.float32),  # stage 2's dV amax (recomputed by stage 4) + any amax the graph left virtual
        ]

    def _compile_plan(self, mod, mm_dk, mm_dq):
        return _prepared.compile_plan_fp8(self, mod, mm_dk, mm_dq)

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        do_tensor: torch.Tensor,
        stats_tensor: torch.Tensor,
        dq_tensor: torch.Tensor,
        dk_tensor: torch.Tensor,
        dv_tensor: torch.Tensor,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        sink_tensor: Optional[torch.Tensor] = None,
        dsink_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        dbias_tensor: Optional[torch.Tensor] = None,
        # sdpa_fp8_backward operands (append-only extension of the shared signature)
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        descale_o: Optional[torch.Tensor] = None,
        descale_dO: Optional[torch.Tensor] = None,
        descale_s: Optional[torch.Tensor] = None,
        descale_dP: Optional[torch.Tensor] = None,
        scale_s: Optional[torch.Tensor] = None,
        scale_dQ: Optional[torch.Tensor] = None,
        scale_dK: Optional[torch.Tensor] = None,
        scale_dV: Optional[torch.Tensor] = None,
        scale_dP: Optional[torch.Tensor] = None,
        amax_dQ: Optional[torch.Tensor] = None,
        amax_dK: Optional[torch.Tensor] = None,
        amax_dV: Optional[torch.Tensor] = None,
        amax_dP: Optional[torch.Tensor] = None,
    ) -> None:
        """The standalone twin of the graph plan over the ``sdpa_fp8_backward`` operand set: the
        twelve scalars are 1-element fp32 DEVICE tensors (read in-kernel), an amax tensor is
        given exactly when the plan was built with it requested (``amax_requested``)."""
        self._refuse_unclaimed(seq_q_lens, seq_kv_lens, sink_tensor, dsink_tensor, bias_tensor, dbias_tensor)
        given = dict(
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            descale_s=descale_s,
            scale_s=scale_s,
            descale_o=descale_o,
            descale_dO=descale_dO,
            descale_dP=descale_dP,
            scale_dQ=scale_dQ,
            scale_dK=scale_dK,
            scale_dV=scale_dV,
            scale_dP=scale_dP,
        )
        for name in _prepared.FP8_SCALARS:
            self._value_error_if(given[name] is None, f"{self._NAME}: {name} is required by sdpa_fp8_backward")
        amaxes = dict(amax_dQ=amax_dQ, amax_dK=amax_dK, amax_dV=amax_dV, amax_dP=amax_dP)
        for name in _prepared.FP8_AMAX:
            self._value_error_if(
                (amaxes[name] is not None) != (name in self.amax_requested),
                f"{self._NAME}: {name} must be given exactly when the plan requested it (requested: {sorted(self.amax_requested)})",
            )
        self.compile()
        tensors = (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, stats_tensor, dq_tensor, dk_tensor, dv_tensor)
        tensors += tuple(given[name] for name in _prepared.FP8_SCALARS) + tuple(amaxes[name] for name in _prepared.FP8_AMAX)
        _prepared.execute_standalone(self, tensors, workspace, current_stream, scale_softmax)


__all__ = ["SdpaBwdDslSm107", "SdpaBwdDslSm107Fp8", "STAGE3_CAUSAL_TRIM"]
