# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 GQA-substrate sparse-attention forward kernel -- PR4 roadmap item.

Envelope: ``G = H_kv`` (one ``topk_idxs`` row per KV-head group, shared by
every Q head in that group), ``index_granularity in (4, 64, 128)`` (QSA /
MSA block shapes), separate (non-aliased) K/V, BF16, SM100/GB300/GR100-class
Blackwell only. See ``_common_sm100.py``'s module docstring for the design
rationale (in particular: why this mainloop is warp-per-row register-tile
gather rather than the tcgen05-tile-MMA / TMA-tensor-map mainloop
``prefill_d128_f16_sm100.py`` uses, and why that is a deliberate, documented
round-1 scope call rather than a silent style regression).

Algorithm (one warp per (query row ``t``, KV-head group ``kv_head``, batch
``b``); grid = ``(rows_per_batch, H_kv, B)``, block = ``(32, 1, 1)``):

  for h in [0, heads_per_kv):                        # Q heads sharing this group
      row_max[h], row_sum[h] = -inf, 0
      O[h] = 0                                        # per-lane strided D_v accumulator
  for j in [0, topk_length[t, kv_head]):
      entry = topk_idxs[t, kv_head, j]
      (start, num_valid, is_valid) = resolve_entry_window(entry, g, kv_bound)
      if not is_valid: continue
      for local in [0, num_valid):                    # sequential token within the block
          token = start + local                       # contiguous storage-native id
          for h in [0, heads_per_kv):
              score = scale * dot(Q[t, h], K[token, kv_head])   # lane-parallel reduction
              online-softmax update row_max[h]/row_sum[h]/O[h] with V[token, kv_head]
  for h in [0, heads_per_kv):
      finalize row_max[h]/row_sum[h] (+ attn_sink) -> lse, out (dead row -> -inf / 0)

Every lane recomputes ``score``/``row_max``/``row_sum`` identically (the QK
dot product is a lane-parallel *reduction*, broadcast back to every lane by
the final butterfly shuffle step) and owns a disjoint, stride-32 slice of
each head's ``D_v`` output -- so there is no cross-lane O traffic and the
result is bit-identical regardless of warp scheduling, which is what the
frozen contract's determinism requirement needs.

BSHD and THD share one compiled kernel body: both Q and K/V are flattened to
their leading token axis on the host (``(B, S, H, D) -> (B*S, H, D)``, a
free view) before the launch; ``IS_BSHD`` (compile-time) selects how a row
recovers its KV addressing base -- ``kv_base = 0`` for THD (``topk_idxs``
already carries *global* flat KV ids, so no batch offset applies) or
``kv_base = (row // S_q) * S_kv`` for BSHD (``topk_idxs`` carries
*within-sequence* ids). ``kv_bound`` is passed as a single scalar
(``T_kv`` for THD, ``S_kv`` for BSHD) since neither layout ragged-splits the
KV axis in this contract.

FP8-per-tensor and the tcgen05/TMA throughput mainloop are round-2 items
(see ``_common_sm100.py``); this file is BF16-only, matching the subtask
scope.

Round-3 update: this module remains the correctness-reference / general
fallback (any per-row selection, any of ``index_granularity in (4, 64,
128)``). ``gqa_prefill_bf16_tile_sm100.py`` adds a fast path for
``index_granularity == 128`` **block-uniform** selection (``G == H_kv`` and
every row in a 32-row Q tile agreeing on ``topk_idxs``, e.g. MSA/NSA-style
block attention) that amortizes the KV gather across the whole tile instead
of re-reading it per row; see that module's docstring for why it ships a
``cp.async``-gather + FFMA mainloop rather than the tcgen05/``mma.sync``
Tensor Core mainloop originally targeted (a concrete, reproducible
toolchain finding: ``mma.sync`` does not compile for ``sm_100a`` on this
box). ``dispatch.py`` routes to it only when a caller opts in via
``uniform_within_tile=True`` -- this file (and its scalar warp-per-row
mainloop) is still what every other caller gets.
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


# === Device kernel ===
#
# ``cfg`` (all-Python-scalar ``GqaPrefillConfig``) is captured by closure --
# built fresh per compiled shape in ``_build_kernel_and_host`` below -- rather
# than threaded through the traced kernel signature, matching how every other
# SM100 DSL kernel in this tree (``prefill_d128_f16_sm100.CFG``,
# ``gdn_prefill_f16.CFG``, ...) references its shape config as a captured
# module/closure constant, not a runtime or Constexpr *parameter*.
#
# The per-head O accumulator is a plain Python list of ``cutlass.Float32``
# registers (one ``ceil(D_v/32)``-length list per Q head in the group) rather
# than a ``cudnn.frost.tile_dsl.regtile.RegTile``: every other use of
# ``RegTile`` in this tree only ever accumulates via ``+``/``-`` between two
# ``RegTile``s (never a bare-scalar broadcast multiply), and this kernel's
# per-lane O update is a rescale-by-scalar-then-add-scalar-times-vector each
# step -- exercising exactly the one ``RegTile.__mul__``/``__rmul__`` path
# (``Vector * bare-Float32`` broadcast) nothing else in the codebase relies
# on. Elementwise ``cutlass.Float32`` arithmetic over the fixed-size Python
# list sidesteps that unverified path entirely without changing the actual
# per-lane register footprint (a Python list of scalar SSA values compiles to
# exactly the same registers a same-length ``RegTile`` would).


def _make_kernel(cfg: GqaPrefillConfig):
    @cute.kernel
    def kernel_fn(
        q: cute.Tensor,  # (T_q, H_q, D_k)
        k: cute.Tensor,  # (T_kv, H_kv, D_k)
        v: cute.Tensor,  # (T_kv, H_kv, D_v)
        topk_idxs: cute.Tensor,  # (T_q, H_kv, topk_max) int32
        topk_length: Optional[cute.Tensor],  # (T_q, H_kv) int32, or None (-> topk_max)
        attn_sink: Optional[cute.Tensor],  # (H_q,) fp32, or None
        out: cute.Tensor,  # (T_q, H_q, D_v)
        lse: cute.Tensor,  # (T_q, H_q) fp32
        kv_bound: cutlass.Int32,  # T_kv (THD) or S_kv (BSHD)
        s_q: cutlass.Int32,  # rows per batch (BSHD) or T_q (THD, so row // s_q == 0 always)
        scale: cutlass.Float32,
        topk_max: cutlass.Int32,
    ) -> None:
        lane, _, _ = cute.arch.thread_idx()
        row = cute.arch.block_idx()[0]
        kv_head = cute.arch.block_idx()[1]
        batch = cute.arch.block_idx()[2]

        t_q = row + batch * s_q

        kv_base = cutlass.Int32(0)
        if cutlass.const_expr(cfg.is_bshd):
            kv_base = (t_q // s_q) * kv_bound

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

                            # --- lane-parallel QK^T reduction, per Q head in the group ---
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
                        out_v[t_q, q_head, d] = (o_acc[h][c] * inv_denom).to(out.element_type)

    return kernel_fn


# === Host launch ===
#
# ``rows_per_batch``/``n_batch`` are dynamic ``cutlass.Int32`` kernel
# arguments (not baked-in Python ints) -- like
# ``csa/compressor/compressor_sm100.py``'s ``gx = (nb_total + ...) // ...``
# grid, computed from a runtime scalar inside the ``@cute.jit`` host body --
# so one compiled artifact per ``GqaPrefillConfig`` (the *dtype-shape*
# constants: d_k/d_v/h_q/h_kv/granularity/layout-kind, never the per-call
# row/KV/top-k *extents*) serves every problem size the benchmark or test
# suite throws at it, matching
# ``deepseek_sparse_attention/sparse_attention_forward/_interface_sm100.py``'s
# compile-key discipline (shape-changing dtype/head config only, not runtime
# extents) rather than ``sdpa/fwd/kernels/split_combine_sm100.py``'s
# fixed-fake-shape-per-call style.


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
        ).launch(
            grid=(rows_per_batch, cfg.h_kv, n_batch),
            block=[WARP_LANES, 1, 1],
            stream=stream,
        )

    return host_fn


def _gpu_arch_flag(device: torch.device) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("gqa_prefill_bf16_sm100 compilation requires CUDA")
    major, minor = torch.cuda.get_device_capability(device)
    if major != 10:
        raise RuntimeError(f"gqa_prefill_bf16_sm100 requires an SM100-family GPU, found SM{major}{minor}")
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
    """One compiled artifact per (dtype/head-shape) ``GqaPrefillConfig``.

    ``T_q`` (rows), ``T_kv``, and ``topk_max`` are all traced as symbolic
    (``cute.sym_int``) extents -- none of them changes the generated code
    (only ``kv_bound``/``s_q``/``topk_max``/``rows_per_batch``/``n_batch`` as
    *scalar* kernel arguments, and each row's own token math, matter -- see
    ``kernel_fn``/``host_fn`` above) -- so the cache key is dtype/head-shape
    only, matching
    ``deepseek_sparse_attention/sparse_attention_forward/_interface_sm100.py``'s
    compile-key discipline (bake in only what changes the traced kernel),
    rather than ``sdpa/fwd/kernels/split_combine_sm100.py``'s
    fixed-fake-shape-per-call style.
    """
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
    bf16 = cutlass.BFloat16
    t_q_sym = cute.sym_int(divisibility=1)
    t_kv_sym = cute.sym_int(divisibility=1)
    topk_max_sym = cute.sym_int(divisibility=1)

    fake_q = cute.runtime.make_fake_compact_tensor(bf16, (t_q_sym, h_q, d_k), stride_order=(2, 1, 0), assumed_align=16)
    fake_k = cute.runtime.make_fake_compact_tensor(bf16, (t_kv_sym, h_kv, d_k), stride_order=(2, 1, 0), assumed_align=16)
    fake_v = cute.runtime.make_fake_compact_tensor(bf16, (t_kv_sym, h_kv, d_v), stride_order=(2, 1, 0), assumed_align=16)
    fake_idx = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (t_q_sym, h_kv, topk_max_sym), stride_order=(2, 1, 0), assumed_align=4)
    fake_len = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (t_q_sym, h_kv), stride_order=(1, 0), assumed_align=4) if has_topk_length else None
    fake_sink = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (h_q,), stride_order=(0,), assumed_align=4) if has_attn_sink else None
    fake_out = cute.runtime.make_fake_compact_tensor(bf16, (t_q_sym, h_q, d_v), stride_order=(2, 1, 0), assumed_align=16)
    fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (t_q_sym, h_q), stride_order=(1, 0), assumed_align=4)

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
    stream=None,
) -> dict:
    """PR4 GQA-substrate kernel entry point (``G == H_kv``, granularity in
    ``(4, 64, 128)``, BF16, separate K/V). Matches the ``fwd/api.py`` GQA
    dispatch call contract; returns ``{'out', 'lse'}``.
    """
    if q.dtype != torch.bfloat16:
        raise ValueError(f"gqa_prefill_bf16_sm100 is BF16-only in this round, got Q dtype {q.dtype}")
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("gqa_prefill_bf16_sm100 requires K/V in BF16 to match Q")
    is_thd = q.ndim == 3
    if is_thd and cu_seqlens_q is None:
        raise ValueError("THD (3-D) Q requires cu_seqlens_q")
    if index_granularity not in (4, 64, 128):
        raise ValueError(f"gqa_prefill_bf16_sm100 serves index_granularity in (4, 64, 128), got {index_granularity}")

    device = q.device
    if device.type != "cuda":
        raise ValueError(f"Q must live on CUDA, got {device}")

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
            raise ValueError(f"gqa_prefill_bf16_sm100 requires H_q % H_kv == 0 and H_kv > 1, got H_q={h_q} H_kv={h_kv}")
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
