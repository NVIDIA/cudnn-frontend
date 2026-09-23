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

The dS workspace is the dominant allocation (``B * H * S_kv * S_q * 2`` bytes): heads
(and, on the half row, batches -- the fp8 body has no ``batch_base``) are chunked to
fit ``_SM107_WS_BUDGET_BYTES`` and the chain loops over chunks with runtime
``head_base`` / ``batch_base``; one compiled artifact serves every launch.

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
from cudnn.sdpa.bwd.api_dsl import (
    SdpaBwdDsl,
    _SM100_DOT_CHUNK_ELEMS,
    _SM100_DOT_Q_TILE,
    _SM100_MATMUL_FILE,
    _SM100_WS_BUDGET_BYTES,
    _sm100_kernel_path,
)
from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, CAUSAL_K_NONE, MatmulTemplateParams, vec_bytes_epi_for
from cudnn.sdpa.fwd.api_dsl import WorkspaceCarver, _torch_stream_context, ws_align

_SM107_D = 256
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
_LOG2E = math.log2(math.e)
_DTYPE_CODE = {torch.bfloat16: DTYPE_BF16, torch.float16: DTYPE_FP16, torch.float8_e4m3fn: DTYPE_E4M3}
_INF = float("inf")


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
        self._sq_pad = -(-self.s_q_max // 128) * 128
        self._skv_pad = -(-self.s_k_max // 256) * 256
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
        self._dot_fn = None
        self._reduce_fn = None
        self._fold_fns = {}

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

    # --- workspace: ONE ordered plan, carved identically at execute -------------------
    def _scratch_plan(self):
        """``[(name, numel, dtype)]`` in carve order.  Every buffer the chain touches that
        is not a caller tensor is here, so :meth:`scratch_workspace_bytes` and
        :meth:`execute` cannot disagree."""
        b, h, hkv, sq, skv, d = self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max, _SM107_D
        sqp, skvp = self._sq_pad, self._skv_pad
        kv_rows = skvp if self._kv_padded else skv
        gqa = self._gqa_group > 1
        plan = [
            # stage 1's delta: [B, H_q, ceil128(S_q)] fp32 -- dot_do_o writes the rounded extent, zeros past S_q
            ("delta", b * h * sqp, torch.float32),
            # the dS workspace of one launch: [b_chunk, qh_chunk, S_kv_pad, S_q_pad], kv-major
            ("ds_ws", self._b_chunk * self._qh_chunk * skvp * sqp, self._ds_dtype),
            # stage 2's per-batch kv lengths (read only under the padded arm) + stage 3's dead THD ABI slot
            ("seq_kv", b, torch.int32),
            ("desc_words", 1, torch.int64),
        ]
        if self._q_padded:
            plan += [("q_pad", b * sqp * h * d, self.dtype), ("do_pad", b * sqp * h * d, self.dtype), ("lse_pad", b * h * sqp, torch.float32)]
        if self._kv_padded:
            plan += [("k_pad", b * skvp * hkv * d, self.dtype), ("v_pad", b * skvp * hkv * d, self.dtype)]
        plan += self._family_scratch_plan(kv_rows, gqa)
        return plan

    def _family_scratch_plan(self, kv_rows: int, gqa: bool):
        b, h, hkv, d = self.batch_size, self.h_q, self.h_kv, _SM107_D
        plan = []
        # stage 2's dV per Q head: the caller's dV only when MHA and no kv padding
        if gqa or self._kv_padded:
            plan.append(("dv_part", b * kv_rows * h * d, self.dtype))
        # stage 3's dK per Q head: the caller's dK when MHA (real rows written)
        if gqa:
            plan.append(("dk_part", b * kv_rows * h * d, self.dtype))
            if self._kv_padded:
                # dkv_reduce folds at the workspace's row extent; the real rows are copied out
                plan += [("dk_fold", b * kv_rows * hkv * d, self.dtype), ("dv_fold", b * kv_rows * hkv * d, self.dtype)]
        return plan

    def scratch_workspace_bytes(self) -> int:
        """A pure function of the compile geometry (delta + one dS chunk + padded staging +
        GQA partials): the executor carves all of it from the caller's buffer."""
        return sum(ws_align(numel * dtype.itemsize) for _name, numel, dtype in self._scratch_plan())

    def _carve(self, carver: WorkspaceCarver) -> dict:
        return {name: carver.take(numel, dtype) for name, numel, dtype in self._scratch_plan()}

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

    def _compile_main(self, mod):
        return mod.compile(
            b=self.batch_size,
            qh=self.h_q,
            kh=self.h_kv,
            sq=self._sq_pad,
            skv=self._skv_pad,
            qh_chunk=self._qh_chunk,
            b_chunk=self._b_chunk,
            sq_real=self.s_q_max,
            skv_real=self.s_k_max,
        )

    def compile(self) -> None:
        """Plan-time JIT for the chain: the stage-2 template specialized on this graph's
        masks / dtype, its per-shape ``compile()``, and the two stage-3 GEMM renderings.
        Stage 1 and stage 4 compile at the first execute, where the real tensors are in hand."""
        self._ensure_support_checked()
        if self._compiled is not None:
            return self._compiled
        mod = load_template(_sm100_kernel_path(_SM107_KERNEL_FILES[self._FAMILY]), self._template_params(), tag=_SM107_TEMPLATE_TAGS[self._FAMILY])
        main = self._compile_main(mod)
        # Stage 3 reads the dS workspace in ITS dtype and writes the io dtype of the GEMM
        # outputs -- the io dtype on the half chain, bf16 on the fp8 chain (upcast Q / K,
        # bf16 dK / dQ partials for stage 4).
        shift = (self.s_k_max - self.s_q_max) if (self.is_causal and self.causal_bottom_right) else 0
        p_dk, p_dq = _stage3_params(_DTYPE_CODE[self._ds_dtype], bool(self.is_causal), shift, _cfg.kv_pad_rows(mod.CFG))
        mm_dk = load_template(_sm100_kernel_path(_SM100_MATMUL_FILE), p_dk, tag=_SM107_MM_TAGS["dk"])
        mm_dq = load_template(_sm100_kernel_path(_SM100_MATMUL_FILE), p_dq, tag=_SM107_MM_TAGS["dq"])
        self._compiled = (mod, main, mm_dk, mm_dq)
        return self._compiled

    # --- execution ---------------------------------------------------------------------
    @staticmethod
    def _padded_rows(dst: torch.Tensor, src: torch.Tensor, rows: int) -> torch.Tensor:
        """``dst`` [B, S_pad, H, D] := ``src`` [B, S, H, D] over the first ``rows``, zeros past them."""
        dst[:, :rows].copy_(src)
        dst[:, rows:].zero_()
        return dst

    def _cute_tensor(self, t):
        from cutlass.cute.runtime import from_dlpack

        return from_dlpack(t, assumed_align=16, enable_tvm_ffi=True)

    def _stage1(self, o, do, delta, stream):
        """delta = rowsum(dO * O), hoisted out of the chunk loop; the artifact is cached on the adapter."""
        import cutlass
        from cutlass.cute.runtime import make_fake_stream

        from cudnn.sdpa.bwd.kernels.bprop_chain_common import dot_do_o_host

        _t = self._cute_tensor
        if self._dot_fn is None:
            self._dot_fn = cutlass.cute.compile(
                dot_do_o_host,
                _t(o),
                _t(do),
                _t(delta),
                None,
                None,
                _SM100_DOT_Q_TILE,
                _SM107_D,
                _SM107_D,
                _SM100_DOT_CHUNK_ELEMS,
                False,
                False,
                make_fake_stream(use_tvm_ffi_env_stream=False),
                options="--enable-tvm-ffi",
            )
        self._dot_fn(_t(o), _t(do), _t(delta), None, None, stream)

    def _stage3(self, mm_dk, mm_dq, ds, q, k, dk_out, dq_out, bs, hs, hb, hc, bc, ws, stream):
        """dK = dS . Q into ``dk_out[bs, :, hs]`` and dQ = dS^T . K into ``dq_out[bs, :, hs]``
        for one (batch, head) chunk.  ``ds`` is the chunk's REAL-extent workspace slice
        ``[bc, hc, S_kv, S_q]``; every operand is a permuted view (no copies)."""
        group, gqa = self._gqa_group, self._gqa_group > 1
        # dK = dS . Q: A = dS[kv, q] (M, K, H, B) K-major; B = Q (D, q, H, B); out (kv, D, H, B)
        mm_dk.matmul_bh(
            ds.permute(2, 3, 1, 0),
            q[bs, :, hs, :].permute(3, 1, 2, 0),
            dk_out[bs, :, hs, :].permute(1, 3, 2, 0),
            n_head=hc,
            n_batch=bc,
            stream=stream,
            meta=ws["seq_kv"],
            desc_words=ws["desc_words"],
        )
        # dQ = dS^T . K: A = dS^T[q, kv] (M, K, H, B) M-major; B = K (D, kv, H_kv, B); out (q, D, H, B).
        # Under GQA the K head is shared by `group` Q heads, so the GEMM runs once per group
        # MEMBER over every `group`-th Q head, which lines A and the output up with the KV heads.
        kv_lo, kv_n = hb // group, hc // group
        kvs = slice(kv_lo, kv_lo + kv_n)
        for gi in range(group):
            a_g = ds[:, gi::group] if gqa else ds
            o_g = dq_out[bs, :, hs, :][:, :, gi::group, :] if gqa else dq_out[bs, :, hs, :]
            mm_dq.matmul_bh(
                a_g.permute(3, 2, 1, 0),
                k[bs, :, kvs, :].permute(3, 1, 2, 0),
                o_g.permute(1, 3, 2, 0),
                n_head=kv_n,
                n_batch=bc,
                stream=stream,
                meta=ws["seq_kv"],
                desc_words=ws["desc_words"],
            )

    def _refuse_unclaimed(self, seq_q_lens, seq_kv_lens, sink_tensor, dsink_tensor, bias_tensor, dbias_tensor) -> None:
        for name, t in (("sink", sink_tensor), ("dSink", dsink_tensor), ("bias", bias_tensor), ("dBias", dbias_tensor)):
            self._value_error_if(t is not None, f"{self._NAME}: {name} is not implemented")
        self._value_error_if(seq_q_lens is not None or seq_kv_lens is not None, f"{self._NAME}: padding masks (seq lens) are not implemented")

    def _stage2_inputs(self, ws, q, do, k, v, stats_tensor):
        """The kernel-facing (padded where needed) Q / dO / K / V / LSE views and the dS slice."""
        b, h, sq, skv = self.batch_size, self.h_q, self.s_q_max, self.s_k_max
        sqp, skvp, d = self._sq_pad, self._skv_pad, _SM107_D
        lse = stats_tensor.reshape(b, h, sq)
        if self._q_padded:
            q_k = self._padded_rows(ws["q_pad"].view(b, sqp, h, d), q, sq)
            do_k = self._padded_rows(ws["do_pad"].view(b, sqp, h, d), do, sq)
            # A q row past the real length: Q / dO zero and LSE = +inf -> P = exp2(S - inf) = 0.
            lse_k = ws["lse_pad"].view(b, h, sqp)
            lse_k[:, :, :sq].copy_(lse)
            lse_k[:, :, sq:].fill_(_INF)
        else:
            q_k, do_k, lse_k = q, do, lse
        if self._kv_padded:
            k_k = self._padded_rows(ws["k_pad"].view(b, skvp, self.h_kv, d), k, skv)
            v_k = self._padded_rows(ws["v_pad"].view(b, skvp, self.h_kv, d), v, skv)
        else:
            k_k, v_k = k, v
        # Read by the padded arm only (uniform real kv length); carved regardless (fixed ABI shape).
        ws["seq_kv"].fill_(skv)
        ds_full = ws["ds_ws"].view(self._b_chunk, self._qh_chunk, skvp, sqp)
        if self._zero_ws:
            ds_full.zero_()
        return q_k, do_k, k_k, v_k, lse_k, ds_full

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
        self._refuse_unclaimed(seq_q_lens, seq_kv_lens, sink_tensor, dsink_tensor, bias_tensor, dbias_tensor)
        _mod, main, mm_dk, mm_dq = self.compile()
        b, h, hkv, sq, skv, d = self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max, _SM107_D
        sqp, skvp = self._sq_pad, self._skv_pad
        bc, hc = self._b_chunk, self._qh_chunk
        gqa = self._gqa_group > 1
        scale = self.scale_softmax if scale_softmax is None or scale_softmax == 0.0 else float(scale_softmax)
        # Logical BHSD over BSHD storage -> the compact [B, S, H, D] view the kernels declare (a permute, never a copy).
        as_bshd = lambda t: t.permute(0, 2, 1, 3)  # noqa: E731
        q, k, v, o, do = (as_bshd(t) for t in (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor))
        dq, dk, dv = (as_bshd(t) for t in (dq_tensor, dk_tensor, dv_tensor))

        stream = self._get_default_stream(current_stream)
        with _torch_stream_context(current_stream, q_tensor.device):
            ws = self._carve(WorkspaceCarver(workspace, self.scratch_workspace_bytes(), self._NAME))
            delta = ws["delta"].view(b, h, sqp)
            q_k, do_k, k_k, v_k, lse_k, ds_full = self._stage2_inputs(ws, q, do, k, v, stats_tensor)
            kv_rows = skvp if self._kv_padded else skv
            dv_k = ws["dv_part"].view(b, kv_rows, h, d) if (gqa or self._kv_padded) else dv
            dk_tgt = ws["dk_part"].view(b, kv_rows, h, d) if gqa else dk
            ds = ds_full[:, :, :skv, :sq]  # what stage 3 reads: the real extents (padded rows / cols never reach a GEMM)

            # STAGE 1, hoisted out of the chunk loop: one streaming pass over O and dO.
            self._stage1(o, do, delta, stream)

            for bi in range(b // bc):
                bb = bi * bc
                bs = slice(bb, bb + bc)
                for ci in range(h // hc):
                    hb = ci * hc
                    hs = slice(hb, hb + hc)
                    # STAGE 2: head_base / batch_base offset every full-tensor read; dS stays chunk-local.
                    main(q_k, do_k, k_k, v_k, dv_k, ds_full, lse_k, delta, ws["seq_kv"], (b, h, hkv, sqp, skvp, hc, bc, sq, skv), float(scale), hb, bb, stream)
                    # STAGE 3: consume the chunk's workspace, write the outputs' (batch, head) slice.
                    self._stage3(mm_dk, mm_dq, ds, q, k, dk_tgt[:, :skv], dq, bs, hs, hb, hc, bc, ws, stream)

            # STAGE 4: fold the per-Q-head partials onto the KV heads (fixed order, deterministic).
            if gqa:
                dk_out = ws["dk_fold"].view(b, kv_rows, hkv, d) if self._kv_padded else dk
                dv_out = ws["dv_fold"].view(b, kv_rows, hkv, d) if self._kv_padded else dv
                self._fold_half(dk_tgt, dv_k, dk_out, dv_out, stream)
                if self._kv_padded:
                    dk.copy_(dk_out[:, :skv])
                    dv.copy_(dv_out[:, :skv])
            elif self._kv_padded:
                dv.copy_(dv_k[:, :skv])

    def _fold_half(self, dk_part, dv_part, dk_out, dv_out, stream):
        import cutlass
        from cutlass.cute.runtime import make_fake_stream

        from cudnn.sdpa.bwd.kernels.bprop_chain_common import dkv_reduce_host

        _t = self._cute_tensor
        io_dt = cutlass.BFloat16 if self.dtype == torch.bfloat16 else cutlass.Float16
        if self._reduce_fn is None:
            self._reduce_fn = cutlass.cute.compile(
                dkv_reduce_host,
                _t(dk_part),
                _t(dv_part),
                _t(dk_out),
                _t(dv_out),
                _SM107_D,
                _SM107_D,
                self._gqa_group,
                io_dt,
                False,
                make_fake_stream(use_tvm_ffi_env_stream=False),
                options="--enable-tvm-ffi",
            )
        self._reduce_fn(_t(dk_part), _t(dv_part), _t(dk_out), _t(dv_out), stream)


# The per-tensor FP8 backward's scalar set, in the order the fp8 kernel binds its share of it.
_FP8_KERNEL_SCALARS = ("descale_q", "descale_k", "descale_v", "descale_dO", "descale_s", "scale_s", "scale_dV")
_FP8_ALL_SCALARS = _FP8_KERNEL_SCALARS + ("descale_o", "descale_dP", "scale_dQ", "scale_dK", "scale_dP")
_FP8_GRAD_DTYPES = (torch.float8_e4m3fn, torch.bfloat16, torch.float16)
_CUTLASS_DTYPE = {"torch.float8_e4m3fn": "Float8E4M3FN", "torch.bfloat16": "BFloat16", "torch.float16": "Float16"}


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

    def _check_support_family(self) -> None:
        n = self._NAME
        self._value_error_if(self.grad_dtype not in _FP8_GRAD_DTYPES, f"{n}: dQ/dK/dV dtype {self.grad_dtype} not in {_FP8_GRAD_DTYPES}")
        self._value_error_if(
            self.dk_desc.dtype != self.grad_dtype or self.dv_desc.dtype != self.grad_dtype,
            f"{n}: dQ/dK/dV must share one dtype; got {self.dq_desc.dtype}/{self.dk_desc.dtype}/{self.dv_desc.dtype}",
        )
        for name, desc in (("k", self.k_desc), ("v", self.v_desc), ("o", self.o_desc), ("dO", self.do_desc)):
            self._value_error_if(desc.dtype != self.dtype, f"{n}: {name} is an FP8 payload and must share Q's dtype {self.dtype}; got {desc.dtype}")

    def _dtype_o_code(self) -> int:
        # Stage 2 publishes dV in bf16 (the pre-quantization value) for the fold + quantize pass.
        return DTYPE_BF16

    def _family_scratch_plan(self, kv_rows: int, gqa: bool):
        b, h, hkv, sq, skv, d = self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max, _SM107_D
        return [
            ("dv_part", b * kv_rows * h * d, torch.bfloat16),  # stage 2's per-Q-head dV_true, bf16
            ("dk_part", b * kv_rows * h * d, torch.bfloat16),  # stage 3's per-Q-head dS . Q8 (descale_q pending)
            ("dq_ws", b * sq * h * d, torch.bfloat16),  # stage 3's dS^T . K8 (descale_k pending)
            ("q_bf16", b * sq * h * d, torch.bfloat16),  # Q8 upcast EXACTLY (the bf16 GEMM's B operand)
            ("k_bf16", b * skv * hkv * d, torch.bfloat16),  # K8 upcast EXACTLY
            ("amax_scratch", 8, torch.float32),  # stage 2's dV amax (recomputed by stage 4) + any amax the graph left virtual
        ]

    def _compile_main(self, mod):
        return mod.compile(b=self.batch_size, qh=self.h_q, kh=self.h_kv, sq=self._sq_pad, skv=self._skv_pad, qh_chunk=self._qh_chunk, has_amax=True)

    def _scalar(self, t: Optional[torch.Tensor], name: str) -> torch.Tensor:
        """A per-tensor scale as the kernels' 1-element fp32 device view (never read back)."""
        self._value_error_if(t is None, f"{self._NAME}: {name} is required by sdpa_fp8_backward")
        self._value_error_if(not isinstance(t, torch.Tensor) or t.device.type != "cuda", f"{self._NAME}: {name} must be a CUDA tensor; got {type(t).__name__}")
        self._value_error_if(
            t.dtype != torch.float32 or t.numel() < 1, f"{self._NAME}: {name} must be a 1-element fp32 tensor; got dtype={t.dtype} numel={t.numel()}"
        )
        return t.reshape(-1)[:1]

    def _stage1_scaled(self, o, do, delta, descale_o, descale_do, stream):
        import cutlass
        from cutlass.cute.runtime import make_fake_stream

        from cudnn.sdpa.bwd.kernels.bprop_chain_common import dot_do_o_scaled_host

        _t = self._cute_tensor
        if self._dot_fn is None:
            self._dot_fn = cutlass.cute.compile(
                dot_do_o_scaled_host,
                _t(o),
                _t(do),
                _t(delta),
                _t(descale_o),
                _t(descale_do),
                _SM100_DOT_Q_TILE,
                _SM107_D,
                _SM100_DOT_CHUNK_ELEMS,
                make_fake_stream(use_tvm_ffi_env_stream=False),
                options="--enable-tvm-ffi",
            )
        self._dot_fn(_t(o), _t(do), _t(delta), _t(descale_o), _t(descale_do), stream)

    def _fold_quant(self, key: str, part, out, descale, scale, amax, group: int, stream):
        """``out = (fold_g(part) * descale) * scale`` in the graph's gradient dtype, ``amax`` the
        max |fold * descale| (atomicMax; zeroed by the caller).  One artifact per output."""
        import cutlass
        from cutlass.cute.runtime import make_fake_stream

        from cudnn.sdpa.bwd.kernels.bprop_chain_common import fold_quant_host

        _t = self._cute_tensor
        _opt = lambda x: None if x is None else _t(x)  # noqa: E731
        fn = self._fold_fns.get(key)
        if fn is None:
            out_dt = getattr(cutlass, _CUTLASS_DTYPE[str(self.grad_dtype)])
            fn = self._fold_fns[key] = cutlass.cute.compile(
                fold_quant_host,
                _t(part),
                _t(out),
                _opt(descale),
                _opt(scale),
                _opt(amax),
                _SM107_D,
                group,
                out_dt,
                make_fake_stream(use_tvm_ffi_env_stream=False),
                options="--enable-tvm-ffi",
            )
        fn(_t(part), _t(out), _opt(descale), _opt(scale), _opt(amax), stream)

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
        self._refuse_unclaimed(seq_q_lens, seq_kv_lens, sink_tensor, dsink_tensor, bias_tensor, dbias_tensor)
        _mod, main, mm_dk, mm_dq = self.compile()
        b, h, hkv, sq, skv, d = self.batch_size, self.h_q, self.h_kv, self.s_q_max, self.s_k_max, _SM107_D
        sqp, skvp = self._sq_pad, self._skv_pad
        bc, hc = self._b_chunk, self._qh_chunk
        scale = self.scale_softmax if scale_softmax is None or scale_softmax == 0.0 else float(scale_softmax)
        as_bshd = lambda t: t.permute(0, 2, 1, 3)  # noqa: E731
        q, k, v, o, do = (as_bshd(t) for t in (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor))
        dq, dk, dv = (as_bshd(t) for t in (dq_tensor, dk_tensor, dv_tensor))
        given = dict(
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            descale_o=descale_o,
            descale_dO=descale_dO,
            descale_s=descale_s,
            descale_dP=descale_dP,
            scale_s=scale_s,
            scale_dQ=scale_dQ,
            scale_dK=scale_dK,
            scale_dV=scale_dV,
            scale_dP=scale_dP,
        )
        sc = {name: self._scalar(given[name], name) for name in _FP8_ALL_SCALARS}

        stream = self._get_default_stream(current_stream)
        with _torch_stream_context(current_stream, q_tensor.device):
            ws = self._carve(WorkspaceCarver(workspace, self.scratch_workspace_bytes(), self._NAME))
            delta = ws["delta"].view(b, h, sqp)
            q_k, do_k, k_k, v_k, lse_k, ds_full = self._stage2_inputs(ws, q, do, k, v, stats_tensor)
            kv_rows = skvp if self._kv_padded else skv
            dv_part = ws["dv_part"].view(b, kv_rows, h, d)
            dk_part = ws["dk_part"].view(b, kv_rows, h, d)
            dq_ws = ws["dq_ws"].view(b, sq, h, d)
            q_bf16 = ws["q_bf16"].view(b, sq, h, d)
            k_bf16 = ws["k_bf16"].view(b, skv, hkv, d)
            ds = ds_full[:, :, :skv, :sq]
            # amax targets: the graph's outputs where requested, scratch otherwise -- all atomicMax'd
            # from zero.  Stage 2's dV amax is over the per-Q-head partials, so stage 4 recomputes it
            # over the folded value; the kernel's copy lands in scratch.
            scratch = ws["amax_scratch"]
            amax = {}
            for i, (name, t) in enumerate((("dQ", amax_dQ), ("dK", amax_dK), ("dV", amax_dV), ("dP", amax_dP))):
                amax[name] = self._scalar(t, f"amax_{name}") if t is not None else scratch[i : i + 1]
                amax[name].zero_()
            amax_dv_kernel = scratch[4:5]
            amax_dv_kernel.zero_()
            # The bf16 GEMM operands: e4m3 -> bf16 is exact (3 mantissa bits into 7, exponent range covered).
            q_bf16.copy_(q)
            k_bf16.copy_(k)

            # STAGE 1: delta in TRUE units = rowsum(dO8 * O8) * descale_o * descale_dO.
            self._stage1_scaled(o, do, delta, sc["descale_o"], sc["descale_dO"], stream)

            for ci in range(h // hc):
                hb = ci * hc
                hs = slice(hb, hb + hc)
                # STAGE 2 (whole batch in-grid; head_base walks the chunks).
                main(
                    q_k,
                    do_k,
                    k_k,
                    v_k,
                    dv_part,
                    ds_full,
                    lse_k,
                    delta,
                    *(sc[name] for name in _FP8_KERNEL_SCALARS),
                    amax_dv_kernel,
                    amax["dP"],
                    (b, h, hkv, sqp, skvp, hc),
                    float(scale),
                    float(scale * _LOG2E),
                    hb,
                    skv,  # the REAL kv length: the padded arm's mask bound and the amax row gate
                    stream=stream,
                )
                # STAGE 3 at bf16 over the exact upcasts; the partials still carry descale_q / descale_k.
                self._stage3(mm_dk, mm_dq, ds, q_bf16, k_bf16, dk_part[:, :skv], dq_ws, slice(0, b), hs, hb, hc, bc, ws, stream)

            # STAGE 4: fold (GQA) + the per-tensor FP8 epilogue into the caller's gradients.
            group = self._gqa_group
            self._fold_quant("dV", dv_part, dv, None, sc["scale_dV"], amax["dV"], group, stream)
            self._fold_quant("dK", dk_part, dk, sc["descale_q"], sc["scale_dK"], amax["dK"], group, stream)
            self._fold_quant("dQ", dq_ws, dq, sc["descale_k"], sc["scale_dQ"], amax["dQ"], 1, stream)


__all__ = ["SdpaBwdDslSm107", "SdpaBwdDslSm107Fp8", "STAGE3_CAUSAL_TRIM"]
