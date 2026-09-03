# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 GQA-substrate sparse-attention forward kernel, FP8-per-tensor -- PR4
roadmap item, precision slice.

Envelope: ``G = H_kv`` (one ``topk_idxs`` row per KV-head group, shared by
every Q head in that group), ``index_granularity in (4, 64, 128)`` (QSA / MSA
block shapes), separate (non-aliased) K/V, ``Q/K/V = Float8E4M3FN``
per-tensor-scaled, SM100/GB300/GR100-class Blackwell only. Companion to
``gqa_prefill_bf16_sm100.py`` (the previous subtask's BF16 kernel): this file
is that kernel's mainloop, scheduling, host-launch and dtype-dispatch
plumbing re-expressed for FP8E4M3FN storage plus the four device-scale
inputs, sharing ``_common_sm100.py``'s ``GqaPrefillConfig`` /
``resolve_entry_window`` / ``lane_group_sum`` gather-and-config logic
unchanged (see that module's docstring for the house-style rationale: warp-
per-row gather rather than tcgen05-tile-MMA, because the frozen contract's
per-query-token ``topk_idxs`` breaks the shared-KV-tile-per-M-tile batching
assumption tcgen05 MMA needs -- see ``_common_sm100.py`` for the full
argument).

Device-scale pattern (mirrors ``cudnn.sdpa.fwd.kernels.prefill_d128_fp8_sm100``
*exactly*, per Rule 3 in ``feedback_no_hidden_kernel_launches``: no host
readback of any scale, ever):

* ``descale_q`` / ``descale_k`` / ``descale_v`` / ``scale_o`` are 1-element
  FP32 **device** tensors. Every thread loads all four via
  ``cutlass.make_array_view(...)`` (same four addresses across the whole
  grid -> L2-broadcast, negligible traffic) -- there is no host-side
  ``.item()`` anywhere in this module.
* They are folded into ``scale`` (the plain ``1/sqrt(d_k)`` -- or caller
  -supplied ``softmax_scale`` -- QK scale; this mainloop's natural-log
  online softmax has no ``log2(e)``-folded ``scale_softmax_log2`` the way
  the dense tcgen05 kernel does, since it never issues an ``exp2``
  instruction) and ``o_scale`` (``descale_v * scale_o``, applied once at the
  very end of each row's epilogue division, mirroring ``o_scale_fused``)
  *before* the mainloop starts, once per warp -- not per gathered KV token.
  This is safe for exactly the reason ``prefill_d128_fp8_sm100`` folds it
  once: per-tensor scale is scale-only and commutes with which KV tokens end
  up in a row's online-softmax accumulation, so folding it before the gather
  loop (instead of per-token) changes nothing about the result.
* **P is never cast to FP8 at all in this mainloop, scaled or unscaled** --
  a real, load-bearing difference from ``prefill_d128_fp8_sm100`` worth
  calling out explicitly rather than silently matching that kernel's
  documented "P cast unscaled" behavior line-for-line. The dense kernel
  casts P to FP8 because P is the tcgen05 MMA's second-GEMM (``P @ V``)
  *operand* -- an FP8 MMA needs an FP8 operand. This mainloop (per
  ``_common_sm100.py``'s house-style note) does not use tensor cores at all:
  it is a warp-lane-parallel scalar reduction, so P (``p = exp(score -
  row_max)``) lives in an FP32 register the whole time and is consumed by a
  plain FMA against an FP32-promoted V element -- there is no operand-dtype
  constraint forcing a P cast, and introducing one would only cost precision
  for no throughput benefit. So the "P cast unscaled" clause from the
  subtask's brief does not apply to this file as written; if a future round
  migrates this envelope to a tcgen05-tile-MMA mainloop (the round-2 item
  ``_common_sm100.py`` names), a P-cast-to-FP8 step re-enters the design at
  that point, and the caller-invariance-of-per-tensor-descale argument above
  would still hold for it (research finding (d)) as long as gathered tiles
  stay contiguous-per-entry, which they do at these granularities (4/64/128
  each cover one contiguous ``[i*g, i*g+g)`` window).
* **Index-driven gather does not change the fold timing.** Same reasoning
  as the dense kernel and as this envelope's BF16 sibling: each gathered
  ``index_granularity``-token window is itself contiguous, and the
  per-tensor scale is a single scalar for the whole K/V tensor regardless of
  which physical tokens a row's top-k selected -- so the fold-once-per-row
  -before-the-gather-loop timing used below is exact, not an approximation.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import cuda.bindings.driver as _cuda_driver  # noqa: F401  (cute.compile pulls cuda)
import cutlass
import cutlass.cute as cute
import torch

from ._common_sm100 import WARP_LANES, GqaPrefillConfig, lane_group_sum, resolve_entry_window

NEG_INF = float("-inf")

__all__ = [
    "STORAGE_DTYPE",
    "sparse_attention_forward_wrapper",
]

# Q/K/V storage dtype for this file -- Float8E4M3FN only (per the PR4
# precision slice's stated scope: BF16 + FP8-per-tensor, no MXFP8/NVFP4;
# E5M2 is not part of this envelope, matching the dense d128 kernel's
# E4M3-primary path). Output stays BF16 (this envelope does not require an
# FP8 O; see the module docstring's P-cast note for why no FP8-operand
# constraint forces an FP8 O either).
STORAGE_DTYPE = cutlass.Float8E4M3FN
OUT_DTYPE = cutlass.BFloat16


# === Device kernel ===
#
# Mirrors ``gqa_prefill_bf16_sm100._make_kernel`` structurally (same warp
# -per-(query row, KV-head group, batch) loop nest, same
# ``GqaPrefillConfig``-captured-by-closure discipline -- see that module's
# comment for why); the only additions are the four descale/scale_o device
# tensors and the two device-scale folds (``scale``, ``o_scale``) applied
# once before the gather loop starts.


def _make_kernel(cfg: GqaPrefillConfig):
    @cute.kernel
    def kernel_fn(
        q: cute.Tensor,  # (T_q, H_q, D_k) fp8_e4m3
        k: cute.Tensor,  # (T_kv, H_kv, D_k) fp8_e4m3
        v: cute.Tensor,  # (T_kv, H_kv, D_v) fp8_e4m3
        topk_idxs: cute.Tensor,  # (T_q, H_kv, topk_max) int32
        topk_length: Optional[cute.Tensor],  # (T_q, H_kv) int32, or None (-> topk_max)
        attn_sink: Optional[cute.Tensor],  # (H_q,) fp32, or None
        out: cute.Tensor,  # (T_q, H_q, D_v) bf16
        lse: cute.Tensor,  # (T_q, H_q) fp32
        kv_bound: cutlass.Int32,  # T_kv (THD) or S_kv (BSHD)
        s_q: cutlass.Int32,  # rows per batch (BSHD) or T_q (THD, so row // s_q == 0 always)
        scale: cutlass.Float32,  # attn_scale (pre-descale)
        topk_max: cutlass.Int32,
        # 1-element fp32 device scales (Rule 3 -- see module docstring).
        descale_q_t: cute.Tensor,
        descale_k_t: cute.Tensor,
        descale_v_t: cute.Tensor,
        scale_o_t: cute.Tensor,
    ) -> None:
        lane, _, _ = cute.arch.thread_idx()
        row = cute.arch.block_idx()[0]
        kv_head = cute.arch.block_idx()[1]
        batch = cute.arch.block_idx()[2]

        t_q = row + batch * s_q

        kv_base = cutlass.Int32(0)
        if cutlass.const_expr(cfg.is_bshd):
            kv_base = (t_q // s_q) * kv_bound

        # --- device-scale fold (once per warp, before the gather loop) ---
        dsc_q = cutlass.Float32(cutlass.make_array_view(descale_q_t)[0])
        dsc_k = cutlass.Float32(cutlass.make_array_view(descale_k_t)[0])
        dsc_v = cutlass.Float32(cutlass.make_array_view(descale_v_t)[0])
        scl_o = cutlass.Float32(cutlass.make_array_view(scale_o_t)[0])
        scale = scale * dsc_q * dsc_k
        o_scale = dsc_v * scl_o

        q_v = cutlass.make_array_view(q)
        k_v = cutlass.make_array_view(k)
        v_v = cutlass.make_array_view(v)
        idx_v = cutlass.make_array_view(topk_idxs)
        out_v = cutlass.make_array_view(out)
        lse_v = cutlass.make_array_view(lse)
        len_v = cutlass.make_array_view(topk_length) if cutlass.const_expr(topk_length is not None) else None
        sink_v = cutlass.make_array_view(attn_sink) if cutlass.const_expr(attn_sink is not None) else None

        H = cfg.heads_per_kv
        V_CHUNKS = cfg.v_chunks_per_lane

        row_max = [cutlass.Float32(NEG_INF) for _ in range(H)]
        row_sum = [cutlass.Float32(0.0) for _ in range(H)]
        o_acc = [[cutlass.Float32(0.0) for _ in range(V_CHUNKS)] for _ in range(H)]

        n_entries = topk_max
        if cutlass.const_expr(len_v is not None):
            n_entries = cutlass.Int32(len_v[t_q, kv_head])

        for j in cutlass.range(0, topk_max, 1, unroll=1):
            if j < n_entries:
                entry = cutlass.Int32(idx_v[t_q, kv_head, j])
                tile_start, num_valid, is_valid = resolve_entry_window(entry, cfg.granularity, kv_bound)
                if is_valid:
                    for local in cutlass.range(0, cfg.granularity, 1, unroll=1):
                        if local < num_valid:
                            token = kv_base + tile_start + local

                            # --- lane-parallel QK^T reduction (raw fp8-promoted-to-fp32
                            # dot product, descaled by the folded scale above), per Q
                            # head in the group ---
                            for h in cutlass.range_constexpr(H):
                                q_head = kv_head * cutlass.Int32(H) + cutlass.Int32(h)
                                partial = cutlass.Float32(0.0)
                                for d in cutlass.range(lane, cfg.d_k, WARP_LANES, unroll=1):
                                    partial = partial + cutlass.Float32(q_v[t_q, q_head, d]) * cutlass.Float32(k_v[token, kv_head, d])
                                score = lane_group_sum(partial, lanes=WARP_LANES) * scale

                                # --- online-softmax update (init-then-conditionally-overwrite,
                                # matching the proven ``pointwise.softplus`` idiom for a
                                # dynamic-condition value merge) ---
                                old_max = row_max[h]
                                new_max = cute.math.max(old_max, score, ftz=True)
                                correction = cutlass.Float32(0.0)
                                if old_max > cutlass.Float32(NEG_INF):
                                    correction = cute.math.exp(old_max - new_max, fastmath=True)
                                p = cute.math.exp(score - new_max, fastmath=True)
                                row_sum[h] = row_sum[h] * correction + p
                                row_max[h] = new_max

                                for c in cutlass.range_constexpr(V_CHUNKS):
                                    d = lane + c * WARP_LANES
                                    v_val = cutlass.Float32(0.0)
                                    if d < cfg.d_v:
                                        v_val = cutlass.Float32(v_v[token, kv_head, d])
                                    o_acc[h][c] = o_acc[h][c] * correction + v_val * p

        # === epilogue: LSE / attn_sink / dead-row, per Q head ===
        # ``o_scale`` (descale_v * scale_o) is applied once, here, at the
        # same point the dense fp8 kernel's ``o_scale_fused`` is applied to
        # its correction-warp-group's ``inv_sum`` multiply -- P was never
        # rescaled in-loop (see module docstring), so O's accumulated units
        # are still "raw fp8-promoted V" the whole way through and need
        # exactly one descale_v (for V's storage scale) times scale_o (for
        # the caller's desired O scale) at the very end, same as the BF16
        # sibling's un-scaled equivalent (``o_scale == 1.0`` there).
        for h in cutlass.range_constexpr(H):
            q_head = kv_head * H + h
            if row_max[h] == cutlass.Float32(NEG_INF):
                lse_v[t_q, q_head] = cutlass.Float32(NEG_INF)
                for c in cutlass.range_constexpr(V_CHUNKS):
                    d = lane + c * WARP_LANES
                    if d < cfg.d_v:
                        out_v[t_q, q_head, d] = cutlass.Float32(0.0).to(out.element_type)
            else:
                sink_term = cutlass.Float32(0.0)
                if cutlass.const_expr(sink_v is not None):
                    sink_term = cute.math.exp(cutlass.Float32(sink_v[q_head]) - row_max[h], fastmath=True)
                denom = row_sum[h] + sink_term
                inv_denom = cutlass.Float32(1.0) / denom
                lse_v[t_q, q_head] = row_max[h] + cute.math.log(row_sum[h], fastmath=True)
                for c in cutlass.range_constexpr(V_CHUNKS):
                    d = lane + c * WARP_LANES
                    if d < cfg.d_v:
                        out_v[t_q, q_head, d] = (o_acc[h][c] * inv_denom * o_scale).to(out.element_type)

    return kernel_fn


# === Host launch ===
#
# Same discipline as ``gqa_prefill_bf16_sm100._make_host``: ``rows_per_batch``
# / ``n_batch`` are dynamic ``cutlass.Int32`` kernel arguments, one compiled
# artifact per ``GqaPrefillConfig`` shape point (never per problem size).


def _make_host(cfg: GqaPrefillConfig):
    kernel_fn = _make_kernel(cfg)

    @cute.jit
    def host_fn(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        topk_idxs: cute.Tensor,
        topk_length: Optional[cute.Tensor],
        attn_sink: Optional[cute.Tensor],
        out: cute.Tensor,
        lse: cute.Tensor,
        kv_bound: cutlass.Int32,
        s_q: cutlass.Int32,
        scale: cutlass.Float32,
        topk_max: cutlass.Int32,
        descale_q_t: cute.Tensor,
        descale_k_t: cute.Tensor,
        descale_v_t: cute.Tensor,
        scale_o_t: cute.Tensor,
        rows_per_batch: cutlass.Int32,
        n_batch: cutlass.Int32,
        stream: _cuda_driver.CUstream = None,
    ) -> None:
        kernel_fn(
            q,
            k,
            v,
            topk_idxs,
            topk_length,
            attn_sink,
            out,
            lse,
            kv_bound,
            s_q,
            scale,
            topk_max,
            descale_q_t,
            descale_k_t,
            descale_v_t,
            scale_o_t,
        ).launch(
            grid=(rows_per_batch, cfg.h_kv, n_batch),
            block=[WARP_LANES, 1, 1],
            stream=stream,
        )

    return host_fn


def _gpu_arch_flag(device: torch.device) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("gqa_prefill_fp8_sm100 compilation requires CUDA")
    major, minor = torch.cuda.get_device_capability(device)
    if major != 10:
        raise RuntimeError(f"gqa_prefill_fp8_sm100 requires an SM100-family GPU, found SM{major}{minor}")
    return {0: "sm_100a", 3: "sm_103a", 7: "sm_100f"}.get(minor, "sm_100a")


@lru_cache(maxsize=None)
def _compile(
    d_k: int,
    d_v: int,
    h_q: int,
    h_kv: int,
    granularity: int,
    is_bshd: bool,
    has_topk_length: bool,
    has_attn_sink: bool,
    arch: str,
):
    """One compiled artifact per (dtype/head-shape) ``GqaPrefillConfig`` --
    see ``gqa_prefill_bf16_sm100._compile``'s docstring for the compile-key
    discipline this mirrors (``T_q``/``T_kv``/``topk_max`` stay symbolic)."""
    cfg = GqaPrefillConfig(
        d_k=d_k,
        d_v=d_v,
        h_q=h_q,
        h_kv=h_kv,
        granularity=granularity,
        is_bshd=is_bshd,
        has_topk_length=has_topk_length,
        has_attn_sink=has_attn_sink,
    )
    fp8 = STORAGE_DTYPE
    bf16 = OUT_DTYPE
    t_q_sym = cute.sym_int(divisibility=1)
    t_kv_sym = cute.sym_int(divisibility=1)
    topk_max_sym = cute.sym_int(divisibility=1)

    fake_q = cute.runtime.make_fake_compact_tensor(fp8, (t_q_sym, h_q, d_k), stride_order=(2, 1, 0), assumed_align=16)
    fake_k = cute.runtime.make_fake_compact_tensor(fp8, (t_kv_sym, h_kv, d_k), stride_order=(2, 1, 0), assumed_align=16)
    fake_v = cute.runtime.make_fake_compact_tensor(fp8, (t_kv_sym, h_kv, d_v), stride_order=(2, 1, 0), assumed_align=16)
    fake_idx = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (t_q_sym, h_kv, topk_max_sym), stride_order=(2, 1, 0), assumed_align=4)
    fake_len = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (t_q_sym, h_kv), stride_order=(1, 0), assumed_align=4) if has_topk_length else None
    fake_sink = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (h_q,), stride_order=(0,), assumed_align=4) if has_attn_sink else None
    fake_out = cute.runtime.make_fake_compact_tensor(bf16, (t_q_sym, h_q, d_v), stride_order=(2, 1, 0), assumed_align=16)
    fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (t_q_sym, h_q), stride_order=(1, 0), assumed_align=4)
    fake_scale1 = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), stride_order=(0,), assumed_align=4)

    host_fn = _make_host(cfg)
    return cute.compile(
        host_fn,
        fake_q,
        fake_k,
        fake_v,
        fake_idx,
        fake_len,
        fake_sink,
        fake_out,
        fake_lse,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        fake_scale1,
        fake_scale1,
        fake_scale1,
        fake_scale1,
        cutlass.Int32(0),
        cutlass.Int32(0),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch {arch} --opt-level 2",
    )


def _flatten_leading(t: Optional[torch.Tensor], keep_trailing: int) -> Optional[torch.Tensor]:
    if t is None:
        return None
    lead = t.shape[: t.ndim - keep_trailing]
    trail = t.shape[t.ndim - keep_trailing :]
    return t.reshape((math.prod(lead),) + trail) if len(lead) > 1 else t


def _require_device_scale(t: Optional[torch.Tensor], name: str, device: torch.device) -> torch.Tensor:
    """Validate (never read) a 1-element FP32 device scale tensor.

    Rule 3 (``feedback_no_hidden_kernel_launches``): this function must not
    call ``.item()``/``.cpu()`` on ``t`` -- only shape/dtype/device checks,
    which are metadata reads, not data transfers.
    """
    if t is None:
        raise ValueError(f"gqa_prefill_fp8_sm100 requires {name} (1-element FP32 device tensor) for FP8 Q/K/V")
    if t.numel() != 1:
        raise ValueError(f"{name} must be a 1-element tensor, got shape {tuple(t.shape)}")
    if t.dtype != torch.float32:
        raise ValueError(f"{name} must be FP32, got {t.dtype}")
    if t.device != device:
        raise ValueError(f"{name} must live on {device} (Q's device), got {t.device}")
    return t.contiguous()


def sparse_attention_forward_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    index_granularity: int = 4,
    softmax_scale: Optional[float] = None,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    scale_o: Optional[torch.Tensor] = None,
    stream=None,
) -> dict:
    """PR4 GQA-substrate kernel entry point, FP8-per-tensor cell
    (``G == H_kv``, granularity in ``(4, 64, 128)``, ``Q/K/V =
    Float8E4M3FN``, separate K/V). Matches the ``fwd/api.py`` GQA dispatch
    call contract plus four required keyword-only device-scale tensors (see
    module docstring); returns ``{'out', 'lse'}`` with ``out`` in BF16.
    """
    if q.dtype != torch.float8_e4m3fn:
        raise ValueError(f"gqa_prefill_fp8_sm100 is Float8E4M3FN-only in this round, got Q dtype {q.dtype}")
    if k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
        raise ValueError("gqa_prefill_fp8_sm100 requires K/V in Float8E4M3FN to match Q")
    is_thd = q.ndim == 3
    if is_thd and cu_seqlens_q is None:
        raise ValueError("THD (3-D) Q requires cu_seqlens_q")
    if index_granularity not in (4, 64, 128):
        raise ValueError(f"gqa_prefill_fp8_sm100 serves index_granularity in (4, 64, 128), got {index_granularity}")

    device = q.device
    if device.type != "cuda":
        raise ValueError(f"Q must live on CUDA, got {device}")

    descale_q = _require_device_scale(descale_q, "descale_q", device)
    descale_k = _require_device_scale(descale_k, "descale_k", device)
    descale_v = _require_device_scale(descale_v, "descale_v", device)
    scale_o = _require_device_scale(scale_o, "scale_o", device)

    with torch.cuda.device(device):
        arch = _gpu_arch_flag(device)

        if is_thd:
            t_q, h_q, d_k = q.shape
            t_kv, h_kv, d_k_kv = k.shape
            _, _, d_v = v.shape
            rows_per_batch, n_batch = t_q, 1
            kv_bound = t_kv
            s_q = t_q
            q_flat, k_flat, v_flat = q, k, v
            idx_flat = topk_idxs
            len_flat = topk_length
        else:
            b, s_q_, h_q, d_k = q.shape
            _, s_kv, h_kv, d_k_kv = k.shape
            _, _, _, d_v = v.shape
            rows_per_batch, n_batch = s_q_, b
            kv_bound = s_kv
            s_q = s_q_
            q_flat = _flatten_leading(q, 2)
            k_flat = _flatten_leading(k, 2)
            v_flat = _flatten_leading(v, 2)
            idx_flat = _flatten_leading(topk_idxs, 2)
            len_flat = _flatten_leading(topk_length, 1)

        if d_k_kv != d_k:
            raise ValueError(f"K head dim ({d_k_kv}) must match Q ({d_k})")
        if h_q % h_kv != 0 or h_kv <= 1:
            raise ValueError(f"gqa_prefill_fp8_sm100 requires H_q % H_kv == 0 and H_kv > 1, got H_q={h_q} H_kv={h_kv}")
        if topk_idxs.shape[-2] != h_kv:
            raise ValueError(f"topk_idxs group dim must be H_kv ({h_kv}) for this kernel's envelope, got {topk_idxs.shape}")

        q_flat = q_flat.contiguous()
        k_flat = k_flat.contiguous()
        v_flat = v_flat.contiguous()
        idx_flat = idx_flat.contiguous()
        if len_flat is not None:
            len_flat = len_flat.contiguous()
        if attn_sink is not None:
            attn_sink = attn_sink.contiguous()

        total_q = rows_per_batch * n_batch
        out = torch.empty((total_q, h_q, d_v), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, h_q), dtype=torch.float32, device=device)

        scale = 1.0 / math.sqrt(d_k) if softmax_scale is None else float(softmax_scale)
        topk_max = idx_flat.shape[-1]

        compiled = _compile(
            int(d_k),
            int(d_v),
            int(h_q),
            int(h_kv),
            int(index_granularity),
            not is_thd,
            len_flat is not None,
            attn_sink is not None,
            arch,
        )

        cu_stream = stream if stream is not None else cuda_current_stream(device)
        compiled(
            q_flat,
            k_flat,
            v_flat,
            idx_flat,
            len_flat,
            attn_sink,
            out,
            lse,
            cutlass.Int32(int(kv_bound)),
            cutlass.Int32(int(s_q)),
            cutlass.Float32(scale),
            cutlass.Int32(int(topk_max)),
            descale_q,
            descale_k,
            descale_v,
            scale_o,
            cutlass.Int32(int(rows_per_batch)),
            cutlass.Int32(int(n_batch)),
            cu_stream,
        )

    if is_thd:
        return {"out": out, "lse": lse}
    return {"out": out.reshape(b, s_q_, h_q, d_v), "lse": lse.reshape(b, s_q_, h_q)}


def cuda_current_stream(device: torch.device):
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
