# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""B5 + B6 fused -- the norm + RoPE BACKWARD of the gated attention block (plus the V band copy).

The forward's stages (2)+(3) are ``y = RoPE(RMSNorm(x) * w)`` on Q and K
(``qk_norm_rope.py``). Their backward, given ``dQ / dK / dV`` compact from the
SDPA backward (B4), writes the Q / K / V bands of the ``[T, N]`` ``dqkvg``
scratch (token stride ``N``, head stride ``D``). Per row, fp32, ONE rounding at
the store::

    g      = RoPE^T(dy)  on [0, rope_dim):  dx_r = dy*cos - rotate_half(dy*sin)     (exact adjoint, ANY table)
             dy          on [rope_dim, D)
    x_hat  = x * rstd                       rstd: the forward's SAVED fp32 rsqrt(mean(x^2) + eps)
    m      = mean_i(g_i * w_i * x_hat_i)    one lane_group_sum (5 shuffles) per row
    dx     = rstd * (g*w - x_hat*m)         == (rstd/D) * (D*dy*w - x_hat*sum(...)) to 2.2e-16
    dW[c]  += g * x_hat                     per-CTA fp32 partial, no 1/D
    V rows:  dx = dy                        a copy, bit-exact (raw 16-B words, no unpack)

**The RoPE adjoint form is the EXACT one.** Lane form, the same 8
shuffles per lane as the forward: ``t[i] = dy[i]*sin[i]``, ``partner =
shfl.bfly(t[i], half_lanes)``, ``signed = +partner if lane < half_lanes else
-partner`` (the forward's sign FLIPPED), ``dx[i] = dy[i]*cos[i] + signed``. The
naive ``dy*cos - rotate_half(dy)*sin`` is exact ONLY for duplicated-half tables
and off by O(1) for arbitrary ones; the exact form costs nothing extra, so it is
the one shipped, pinned by ``test_norm_rope_bwd_exact_adjoint_on_random_tables``.

**``qk_norm=False`` arm** (``apply_norm=False``): ``x / rstd / w / dW`` are
traced as ``None``; ``dx = RoPE^T(dy)`` on ``[0, rope_dim)`` and ``[rope_dim, D)``
is copied with NO fp32 op (widen + narrow of the same value: ``torch.equal``).
The host refuses weights / rstd / x / dW on that recipe and their absence on the
norm-on one, both typed (``check_norm_weights_match_recipe`` precedent).

**Grid: persistent SINGLE-CLASS CTAs.** ``grid = n_ctas_q + n_ctas_k + n_ctas_v``;
CTA ``c < n_ctas_q`` strides over Q row groups, the next ``n_ctas_k`` over K row
groups, the rest over V row groups, with ``n_ctas_x = min(row_groups_x, cap)``
PER CLASS (``cap = SMs * CTAS_PER_SM`` under ``n_ctas_policy="sm_fill"``,
``N`` under ``"fixed:N"``). So the class is per-CTA uniform (no per-row base /
stride select), a CTA never straddles the Q/K seam, ONE ``[D]`` dW accumulator
per lane suffices, and the partials plane is ``(n_ctas_q + n_ctas_k) x D x 4 B``
(3.47 MB at 397B S=32K on 212 SMs). ``n_ctas_for(r, t)`` is the plane geometry
the caller allocates; the reduce launch consumes exactly the plane's rows.

**Determinism by construction.** No atomics anywhere: per-lane fp32
accumulation in row order, the four lane groups combined in the fixed order
``g0 + g1 + g2 + g3`` through SMEM at CTA end, and ``run_dw_norm_reduce`` sums
the partials in a second launch on the same stream (``REDUCE_LANES`` residue
classes of rows, each ascending, then the classes ascending). Two runs
are bitwise equal PER KNOB; equality across ``rows_per_group`` / ``n_ctas`` values
is NOT a property (the summation order changes with the row-to-CTA assignment).

**Dead rows.** With ``seq_lens`` bound, ``dead = (token % s) >= seq_lens[token // s]``:
``dx := 0`` SELECTED and STORED in all three bands, and the row contributes 0 to
the dW partial -- the select sits BEFORE the FMA, so a NaN dead row cannot poison
dW (the same row-validity gate the SDPA epilogues put on amax). Folded out
when ``seq_lens`` is None.

SMEM buffer table (``rows_per_group in {1, 2}`` leaves it unchanged):

    buffer       dtype/elems/bytes            writer, how                          reader, how                       per-lane stride   swizzle + why
    sDwPartial   fp32, groups_per_cta x D     lane groups 1..3 at CTA end, each     lane group 0: the same addresses  16 B within a     NONE -- the interleave IS the bank
                 (4 x 256 = 1024 elems,       lane its 8 fp32 (its 16-B chunk of    for group in 1..3, summed in      (lanes*16)-B      spread: 8 consecutive lanes of a v4
                 4 KiB at D=256, 128 thr)     the [D] row) as TWO 16-B vectors,     fixed order g0+g1+g2+g3, then     granule column    st/ld.shared phase cover ONE 128-B
                                              j in {0,1} = quads 0-3 / 4-7, at      one 16-B store per lane per j     (a lane's two     line (32 banks x 4 B): 4 wavefronts
                                              elem off = group*D + c*lanes*8 +      into mDW*[cta, :] (natural        quads sit          per 512 B = the ideal. REJECTED:
                                              j*lanes*4 + lane*4                    [D] order)                        lanes*16 B apart) lane-contiguous 32 B (2-way conflict)

``sDwPartial`` is allocated ONCE, in ``frost_qk_norm_rope_bwd`` (under
``want_dw``), and handed to ``_row_class``: that body is inlined once per class
arm, so an allocation inside it ships one instance PER ARM (Q and K = 8 KiB,
read off the pre-fix artifact's MLIR ``constant(8192 : i64)``) -- the table
above is the contract, 4 KiB at D=256.

Barrier table:

    barrier                       producer / issuing lanes                          consumer                          count            when
    nvvm.barrier_cta_sync()       ALL threads_per_cta (=128) threads of the CTA,    lane group 0's ld.shared of       ONE per CTA,     after every group's sDwPartial
    (bar.sync 0)                  4 warps; SUM(issuing lanes) = 128 == the CTA's    groups 1..3                       at the tail      stores, before group 0's loads;
                                  thread count (the persistent loop bound is                                          after the        generic proxy on both sides:
                                  CTA-uniform, so every thread reaches it)                                            row loop         no fence_proxy

No mbarrier, no TMA. The barrier and the SMEM exist only under ``want_dw``.

**Bytes.** ``moved_bytes``: Q/K rows read dy and x and write dx (3 passes) plus
4 B rstd per row; V rows read + write (2 passes): 54408 B per token at 397B
(bf16, H_q=32, H_kv=2, D=256) -> S=32K 1.78 GB -> 139 us at the 12846 GB/s pin;
``apply_norm=False`` 36 KiB/token -> 1.21 GB -> 94 us. cos / sin and the ``[D]``
weights are L2-resident and not counted (the forward's convention).

**rstd is a scalar 4-B ``ld.global`` per row**: a v2/v4 load would need the K
side's offset ``R*4``-B aligned, the trap the forward's vectorised rstd STORE
fell into at ``T=3, h_q=3, R=2`` (``qk_norm_rope.py:489-524``); rstd is 1/128 of
the row traffic.

**Footguns inherited from the forward kernels.** (1) Never ``.view()`` a column
slice of the ``[T, N]`` slab on the kernel path: the caller hands ``[T, H, D]``
band views. (2) Every operand is traced with ``assumed_align=16``: a band base
``qkvg_offsets[i] * 2 B`` must be 16-B aligned -- ``QKVG_TILE_ALIGN = 64``
elements (128 B) guarantees it. ``run_qk_norm_rope_bwd`` refuses, on the host,
any ``[T, H, D]`` operand off that 16-B grid (``_check_row_layout``, shared with
the gate backward: token stride a multiple of 8 elements, head stride D, unit
element stride, 16-B base -- the token stride is SYMBOLIC in the artifact, so
nothing past the host would) and any operand off the launch device
(``_check_one_cuda_device``). (3) Strides are ``Int32`` at the tvm-ffi
boundary (the 2^27-element caveat is the DSL's, not this kernel's).
"""

from typing import NamedTuple, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.experimental import primitives as nvvm

from cudnn.frost.device import current_device, multiprocessor_count
from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp16, lane_group_sum, opaque_f32_zero
from cudnn.frost.tile_dsl.tma import ld_global, ld_global_v4, ld_shared_v4, st_global, st_global_v4, st_shared_v4

from .qk_norm_rope import ACCESS_BYTES, ELEMS_PER_ACCESS, _fake, fake_rowmajor_dynamic_token_stride, lanes_per_row, validate_shape, vec_chunks
from .sigmoid_gate_bwd import _check_one_cuda_device, _check_row_layout

DEFAULT_THREADS_PER_CTA = 128
# rows_per_group is PER ARM, from its own A/B (Rubin, cc 10.7, 204 SMs, SM clock
# locked at 2376 MHz; bf16 397B geometry H_q=32 H_kv=2 D=256 rope 64, B=1; ONE
# process, median of 20 L2-flushed launches per cell, no control pair;
# 2026-09-29, logs retained internally -- a re-run agrees per cell to <= 2.5 %,
# far below the 20-33 % deltas the decision rests on):
#
#     arm         S       rpg=1                 rpg=2
#     norm        8K      118.0 us / 3779 GB/s  142.0 us / 3140 GB/s   -> 1 (+20 %)
#     norm        32K     280.7 us / 6351 GB/s  371.9 us / 4794 GB/s   -> 1 (+32 %)
#     rope-only   8K       97.9 us / 3084 GB/s   82.3 us / 3670 GB/s   -> 2 (+19 %)
#     rope-only   32K     257.7 us / 4688 GB/s  199.2 us / 6064 GB/s   -> 2 (+29 %)
#
# Those cells are IDLE-stream event pairs: each includes that launch's host
# submission gap (measured 2026-09-30 on the same Rubin part: ~59 us for the
# 17-argument norm launch, ~40 us RoPE-only, ~22 us gate, ~30 us reduce), so
# the true kernel-time deltas are LARGER than the table's. Shadow-timed (kernel
# enqueued behind a 512 MiB memset, host gap hidden) shipped defaults, same part:
# norm rpg=1 59.6 / 222.1 us at 8K / 32K = 7484 / 8029 GB/s (58.3 / 62.5 % pin),
# RoPE-only rpg=2 42.6 / 160.4 us = 7093 / 7529 GB/s (55.2 / 58.6 % pin);
# logs retained internally.
#
# The norm arm carries +8 live fp32 for the dW accumulator, +8 for x and the
# 5-shuffle reduction per row, and like the forward (`qk_norm_rope.py:177-196`)
# it is register-bound: a second row in flight makes it worse. The RoPE-only arm
# has none of that state and wants the memory-level parallelism. `rows_per_group=None`
# (the default) resolves per arm; an explicit value is honoured as given.
DEFAULT_ROWS_PER_GROUP = 1
DEFAULT_ROWS_PER_GROUP_ROPE_ONLY = 2
DEFAULT_CONST_HEAD_COUNTS = True
# SM-fill cap per class: SMs x CTAS_PER_SM persistent CTAs of 128 threads.
# 8 is the MEASURED residency of this body, not a guess (2026-09-30, Rubin cc 10.7,
# 204 SMs): the sm_107a cubins carry REG 64 (norm arm) / 57
# (RoPE-only; allocated as 64), LOCAL 0 -- 128 thr x 64 regs = 8192 of the SM's
# 65536 = exactly 8 co-resident CTAs -- and the cap sweep over k x SMs, k in
# {1,2,3,4,6,8,12,16} (3 rounds x 20, logs retained internally) has 8
# fastest in all four cells (norm 8K/32K 116.5/278.6 us, RoPE-only 80.8/196.9),
# 6 slower by 6-9 %, 12 slower by 6-9 % (1.5 waves), 16 = 8 within 0.5 % (2 full
# waves). A part with a different register file re-measures this constant.
CTAS_PER_SM = 8
DEFAULT_N_CTAS_POLICY = "sm_fill"
_BPE = 2
# The reduce block: REDUCE_COLS consecutive columns of one partial row per
# row-lane (REDUCE_COLS x 4 B contiguous: whole 32-B sectors); REDUCE_LANES
# row-lanes take the rows in residue classes; REDUCE_UNROLL is the LLVM unroll
# hint on each lane's row loop (the summation ORDER is unchanged by it -- the
# adds stay sequential, only the loads get hoisted). REDUCE_COLS x REDUCE_LANES
# threads per block, d / REDUCE_COLS blocks per class (validate_shape's
# ``d % 8 == 0`` makes REDUCE_COLS=8 always divide d).
#
# MEASURED (Rubin cc 10.7, 204 SMs, 3 rounds x 20, SHADOW-timed = the launch
# enqueued behind a 512 MiB memset so the ~30 us host submission gap of an
# idle-stream event pair is hidden; 2026-09-30, logs retained internally),
# D=256, planes of 1632 rows (the SM-fill cap on 204 SMs) / 816 rows (cap 4 x SMs):
#
#     (cols, lanes, unroll)   grid      threads   1632 rows   816 rows
#     (32,  16, 0)  first     8 x 2     512        7.7 us      6.4 us
#     (32,  32, 0)            8 x 2     1024       6.7         4.9
#     (16,  64, 0)            16 x 2    1024       5.4         4.0
#     ( 8,  64, 0)            32 x 2    512        5.4         3.8
#     ( 8, 128, 0)  SHIPPED   32 x 2    1024       4.4         4.5
#     ( 4, 128, 0)            64 x 2    512        4.6         4.5
#     ( 4, 256, 0)            64 x 2    1024       5.4         4.2
#     unroll 2 / 4 on any of the above: SLOWER every time (e.g. (8,128,2) 6.5,
#     (8,128,4) 5.1, (32,16,4) 16.6 us at 1632 rows) -- do not retry; the knob
#     stays at 0 and exists to record the result.
#
# So the reduce is 4.4 us on the default plane (+75 % on the launch vs the first
# geometry's 7.7 us), 5.1 -> ~3 us IN SITU behind the 59.6 us norm launch at S=8K.
# The "34 us fixed cost" the first probe reported was the host submission gap of
# an idle-stream event pair, not kernel time.
REDUCE_COLS = 8
REDUCE_LANES = 128
REDUCE_UNROLL = 0

_FAKE_STREAM = None

__all__ = [
    "CTAS_PER_SM",
    "DEFAULT_CONST_HEAD_COUNTS",
    "DEFAULT_N_CTAS_POLICY",
    "DEFAULT_ROWS_PER_GROUP",
    "DEFAULT_ROWS_PER_GROUP_ROPE_ONLY",
    "REDUCE_COLS",
    "REDUCE_LANES",
    "REDUCE_UNROLL",
    "DEFAULT_THREADS_PER_CTA",
    "QkNormRopeBwdRecipe",
    "compile_qk_norm_rope_bwd",
    "moved_bytes",
    "n_ctas_for",
    "run_dw_norm_reduce",
    "run_qk_norm_rope_bwd",
    "validate_shape",
]


@cute.jit
def _row_class(
    dy_base: cutlass.Int64,
    dy_tok: cutlass.Int64,  # token strides in ELEMENTS (head stride is D, elem stride 1: baked)
    x_base: cutlass.Int64,
    x_tok: cutlass.Int64,
    rstd_base: cutlass.Int64,
    w_base: cutlass.Int64,
    cos_base: cutlass.Int64,
    cos_tok: cutlass.Int64,
    sin_base: cutlass.Int64,
    sin_tok: cutlass.Int64,
    out_base: cutlass.Int64,
    out_tok: cutlass.Int64,
    dw_base: cutlass.Int64,  # THIS CTA's [D] partial row (already offset)
    sl_base: cutlass.Int64,
    n_class_rows: cutlass.Int32,
    n_iters: cutlass.Int32,
    cta_in_class: cutlass.Int32,
    n_ctas_class: cutlass.Int32,
    h: cutlass.Int32,
    s: cutlass.Int32,
    io_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    apply_norm: cutlass.Constexpr[bool],
    want_dw: cutlass.Constexpr[bool],
    has_seq_lens: cutlass.Constexpr[bool],
    is_copy: cutlass.Constexpr[bool],
    sdw,  # the ONE sDwPartial SMEM Array of the kernel (None without want_dw)
) -> None:
    """The persistent loop of ONE row class (Q, K or V) for one CTA.

    Takes base addresses and element strides as scalars (the tensors' facts,
    read off ``.iterator.toint()`` / ``.stride`` in the kernel) so the three
    class arms specialise on ``h_ct`` / ``apply_norm`` / ``is_copy`` with one
    body. Row ``row`` of the class is token ``row // h``, head ``row % h``;
    iteration ``it`` of CTA ``cta_in_class`` covers row group
    ``cta_in_class + it * n_ctas_class`` (``rows_per_cta`` rows). ``sdw`` is
    the kernel's single ``sDwPartial`` allocation (SMEM table): allocating it
    here would instantiate it once per inlined arm.
    """
    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(vec_chunks(d))
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    rope_lanes = cutlass.const_expr(rope_dim // ELEMS_PER_ACCESS)
    half_lanes = cutlass.const_expr(rope_lanes // 2)
    n_acc = cutlass.const_expr(chunks * ELEMS_PER_ACCESS)

    _h = cutlass.Int32(h_ct) if cutlass.const_expr(const_head_count) else h
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    lane_off = lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
    bpe = cutlass.Int64(_BPE)
    d64 = cutlass.Int64(d)
    in_rope = lane < cutlass.Int32(rope_lanes) if cutlass.const_expr(rope_dim > 0) else cutlass.Boolean(False)
    zero = cutlass.Float32(0.0)
    inv_d = cutlass.Float32(1.0 / d)

    # ONE [D] dW accumulator per lane (8 fp32 per chunk), live across the whole
    # persistent loop: the class is CTA-uniform, so it never mixes Q and K.
    dw_acc = [cutlass.Float32(0.0) for _ in range(n_acc)]

    for it in cutlass.range(n_iters):
        g_idx = cta_in_class + it * n_ctas_class
        row0 = (g_idx * cutlass.Int32(groups_per_cta) + grp) * cutlass.Int32(rows_per_group)

        # --- PASS 1: every load this thread makes for its rows -----------------
        rows = []
        deads = []
        tokens = []
        out_addrs = []
        dys = []
        xs = []
        rstds = []
        for r in cutlass.range_constexpr(rows_per_group):
            row = row0 + cutlass.Int32(r)
            row_r = row if row < n_class_rows else n_class_rows - cutlass.Int32(1)
            token = row_r // _h
            head = row_r % _h
            tok64 = token.to(cutlass.Int64)
            head64 = head.to(cutlass.Int64)
            dead = cutlass.Boolean(False)
            if cutlass.const_expr(has_seq_lens):
                bidx = token // s
                pos = token - bidx * s
                slen = ld_global(sl_base + bidx.to(cutlass.Int64) * cutlass.Int64(4), cutlass.Int32)
                dead = pos >= slen
            dy_addr = dy_base + (tok64 * dy_tok + head64 * d64) * bpe
            out_addr = out_base + (tok64 * out_tok + head64 * d64) * bpe
            row_dy = []
            row_x = []
            for c in cutlass.range_constexpr(chunks):
                off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                words = ld_global_v4(dy_addr + off, cutlass.Int32)
                if cutlass.const_expr(is_copy):
                    row_dy.append(list(words))  # raw 16-B words: the copy never unpacks
                else:
                    row_dy.append([v for w in words for v in f16x2_to_f32(w, dtype=io_dtype)])
            rstd = cutlass.Float32(1.0)
            if cutlass.const_expr(apply_norm):
                x_addr = x_base + (tok64 * x_tok + head64 * d64) * bpe
                for c in cutlass.range_constexpr(chunks):
                    off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                    row_x.append([v for w in ld_global_v4(x_addr + off, cutlass.Int32) for v in f16x2_to_f32(w, dtype=io_dtype)])
                # scalar 4-B load: a v2/v4 needs an R*4-B aligned K-side offset (see the docstring)
                rstd = ld_global(rstd_base + row_r.to(cutlass.Int64) * cutlass.Int64(4), cutlass.Float32)
            rows.append(row)
            deads.append(dead)
            tokens.append(token)
            out_addrs.append(out_addr)
            dys.append(row_dy)
            xs.append(row_x)
            rstds.append(rstd)

        # --- PASS 2: adjoint RoPE, RMSNorm backward, dW fold, store ----------------
        for r in cutlass.range_constexpr(rows_per_group):
            valid = rows[r] < n_class_rows
            if cutlass.const_expr(is_copy):
                # V class: dx = dy, bit-exact. The dead-row select is on the raw words
                # (0x00000000 == two +0.0 halves in either io dtype).
                for c in cutlass.range_constexpr(chunks):
                    words = list(dys[r][c])
                    if cutlass.const_expr(has_seq_lens):
                        words = [(cutlass.Int32(0) if deads[r] else w) for w in words]
                    if valid:
                        off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                        st_global_v4(out_addrs[r] + off, words, cutlass.Int32)
            else:
                g = [list(dys[r][c]) for c in range(chunks)]
                if cutlass.const_expr(rope_dim > 0):
                    # cos/sin at their point of use, on the rope lanes only (the
                    # forward's deferred-load lesson: both are L2 hits by construction).
                    cos_v = [zero] * ELEMS_PER_ACCESS
                    sin_v = [zero] * ELEMS_PER_ACCESS
                    if in_rope:
                        tok64 = tokens[r].to(cutlass.Int64)
                        cb = cos_base + tok64 * cos_tok * bpe + lane_off
                        sb = sin_base + tok64 * sin_tok * bpe + lane_off
                        cos_v = [v for w in ld_global_v4(cb, cutlass.Int32) for v in f16x2_to_f32(w, dtype=io_dtype)]
                        sin_v = [v for w in ld_global_v4(sb, cutlass.Int32) for v in f16x2_to_f32(w, dtype=io_dtype)]
                    # Exact adjoint: dx = dy*cos - rotate_half(dy*sin). The shuffle is
                    # unconditional (every lane of the warp must reach shfl.sync; the
                    # non-rope lanes carry sin = 0) and only the rope lanes keep the result.
                    rotated = []
                    for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                        t_i = g[0][i] * sin_v[i]
                        partner = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, t_i, cutlass.Int32(half_lanes), 31, kind=nvvm.Shfl.BFLY))
                        signed = partner if lane < cutlass.Int32(half_lanes) else -partner  # the forward's sign, FLIPPED
                        rotated.append(g[0][i] * cos_v[i] + signed)
                    for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                        g[0][i] = rotated[i] if in_rope else g[0][i]

                dx = []
                if cutlass.const_expr(apply_norm):
                    # weights at their point of use ([D], shared by every row of the class)
                    ws = []
                    for c in cutlass.range_constexpr(chunks):
                        off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                        ws.append([v for w in ld_global_v4(w_base + off, cutlass.Int32) for v in f16x2_to_f32(w, dtype=io_dtype)])
                    rstd = rstds[r]
                    x_hat = [[xs[r][c][i] * rstd for i in range(ELEMS_PER_ACCESS)] for c in range(chunks)]
                    gw = [[g[c][i] * ws[c][i] for i in range(ELEMS_PER_ACCESS)] for c in range(chunks)]
                    acc = zero
                    for c in cutlass.range_constexpr(chunks):
                        for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                            acc = acc + gw[c][i] * x_hat[c][i]
                    m = lane_group_sum(acc, lanes) * inv_d
                    for c in cutlass.range_constexpr(chunks):
                        dx.append([rstd * (gw[c][i] - x_hat[c][i] * m) for i in range(ELEMS_PER_ACCESS)])
                    if cutlass.const_expr(want_dw):
                        # dead / tail rows contribute EXACTLY 0: select BEFORE the add (a
                        # NaN dy on a dead row must not poison dW), never `* 0`.
                        keep = valid & (~deads[r]) if cutlass.const_expr(has_seq_lens) else valid
                        for c in cutlass.range_constexpr(chunks):
                            for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                                contrib = g[c][i] * x_hat[c][i]
                                dw_acc[c * ELEMS_PER_ACCESS + i] = dw_acc[c * ELEMS_PER_ACCESS + i] + (contrib if keep else zero)
                else:
                    # RoPE-only: dx IS g; [rope_dim, D) goes back through fp32_to_fp16
                    # untouched by any fp32 op -> a bit-exact copy (asserted by the tests).
                    dx = g
                if cutlass.const_expr(has_seq_lens):
                    for c in cutlass.range_constexpr(chunks):
                        for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                            dx[c][i] = zero if deads[r] else dx[c][i]
                if valid:
                    for c in cutlass.range_constexpr(chunks):
                        off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                        st_global_v4(
                            out_addrs[r] + off,
                            [fp32_to_fp16(dx[c][2 * i], dx[c][2 * i + 1], dtype=io_dtype) for i in range(ELEMS_PER_ACCESS // 2)],
                            cutlass.Int32,
                        )

    # --- CTA end: combine the four lane groups' dW in FIXED order, store the partial ----
    if cutlass.const_expr(want_dw):
        # sDwPartial -- see the module docstring's SMEM table: interleaved so 8
        # consecutive lanes of a v4 phase cover one 128-B line (4 wavefronts / 512 B).
        # Allocated once by the kernel and passed in (one instance, not one per arm).
        sDwPartial = sdw
        quads = cutlass.const_expr(ELEMS_PER_ACCESS // 4)
        if grp > cutlass.Int32(0):
            for c in cutlass.range_constexpr(chunks):
                for j in cutlass.range_constexpr(quads):
                    p = sDwPartial.data_ptr(grp * cutlass.Int32(d) + cutlass.Int32(c * lanes * ELEMS_PER_ACCESS + j * lanes * 4) + lane * cutlass.Int32(4))
                    st_shared_v4(p, dw_acc[c * ELEMS_PER_ACCESS + 4 * j : c * ELEMS_PER_ACCESS + 4 * j + 4], cutlass.Float32)
        # ONE named barrier per CTA (barrier table row): all threads_per_cta threads
        # arrive -- the persistent loop bound is CTA-uniform. Generic proxy both sides.
        nvvm.barrier_cta_sync()
        if grp == cutlass.Int32(0):
            tot = list(dw_acc)
            for gi in cutlass.range_constexpr(1, groups_per_cta):
                for c in cutlass.range_constexpr(chunks):
                    for j in cutlass.range_constexpr(quads):
                        p = sDwPartial.data_ptr(cutlass.Int32(gi * d + c * lanes * ELEMS_PER_ACCESS + j * lanes * 4) + lane * cutlass.Int32(4))
                        v = ld_shared_v4(p, cutlass.Float32)
                        for q in cutlass.range_constexpr(4):
                            tot[c * ELEMS_PER_ACCESS + 4 * j + q] = tot[c * ELEMS_PER_ACCESS + 4 * j + q] + v[q]
            for c in cutlass.range_constexpr(chunks):
                for j in cutlass.range_constexpr(quads):
                    elem = cutlass.Int64(c * lanes * ELEMS_PER_ACCESS + 4 * j) + lane.to(cutlass.Int64) * cutlass.Int64(ELEMS_PER_ACCESS)
                    st_global_v4(dw_base + elem * cutlass.Int64(4), tot[c * ELEMS_PER_ACCESS + 4 * j : c * ELEMS_PER_ACCESS + 4 * j + 4], cutlass.Float32)


@cute.kernel
def frost_qk_norm_rope_bwd(
    mDQ: cute.Tensor,  # [T, H_q,  D] compact: B4's dQ
    mDK: cute.Tensor,  # [T, H_kv, D] compact: B4's dK
    mDV: cute.Tensor,  # [T, H_kv, D] compact: B4's dV
    mXq: Optional[cute.Tensor],  # q_pre: proj_slab band (token stride N) or compact; None under apply_norm=False
    mXk: Optional[cute.Tensor],  # k_pre
    mRstdQ: Optional[cute.Tensor],  # [T, H_q]  fp32 (saved); None under apply_norm=False
    mRstdK: Optional[cute.Tensor],  # [T, H_kv] fp32
    mWq: Optional[cute.Tensor],  # [D] norm weight; None under apply_norm=False (presence switch)
    mWk: Optional[cute.Tensor],  # [D]
    mCos: cute.Tensor,  # [T, rope_dim]
    mSin: cute.Tensor,  # [T, rope_dim]
    mOutQ: cute.Tensor,  # the dqkvg Q band (token stride N, head stride D); MAY alias mDQ
    mOutK: cute.Tensor,  # the dqkvg K band; MAY alias mDK
    mOutV: cute.Tensor,  # the dqkvg V band
    mDWq: Optional[cute.Tensor],  # [n_ctas_q, D] fp32 per-CTA partials (None: no dW)
    mDWk: Optional[cute.Tensor],  # [n_ctas_k, D]
    mSeqLens: Optional[cute.Tensor],  # [B] int32, or None (dense: the select is folded out)
    n_ctas_q: cutlass.Int32,
    n_ctas_k: cutlass.Int32,
    n_ctas_v: cutlass.Int32,
    n_q_rows: cutlass.Int32,
    n_k_rows: cutlass.Int32,  # == the V row count (both H_kv)
    s: cutlass.Int32,
    h_q: cutlass.Int32,
    h_kv: cutlass.Int32,
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    const_head_counts: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """Three CTA classes over one grid (Q | K | V), each a persistent row loop; see the module docstring."""
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    apply_norm = cutlass.const_expr(mWq is not None)
    want_dw = cutlass.const_expr(mDWq is not None)
    has_seq_lens = cutlass.const_expr(mSeqLens is not None)
    lanes = cutlass.const_expr(lanes_per_row(d))
    rows_per_iter = cutlass.const_expr((threads_per_cta // lanes) * rows_per_group)
    io_dtype = mDQ.element_type
    i64_0 = cutlass.Int64(0)

    cta = cutlass.Int32(cute.arch.block_idx()[0])
    sl_base = mSeqLens.iterator.toint() if cutlass.const_expr(has_seq_lens) else i64_0
    cos_base = mCos.iterator.toint()
    cos_tok = cutlass.Int64(mCos.stride[0])
    sin_base = mSin.iterator.toint()
    sin_tok = cutlass.Int64(mSin.stride[0])
    d_bytes = cutlass.Int64(d * 4)
    rpi = cutlass.Int32(rows_per_iter)
    # The ONE sDwPartial (SMEM table: fp32 groups_per_cta x D, 4 KiB at D=256 /
    # 128 threads). Allocated here, not in _row_class: that body is inlined per
    # class arm and an allocation inside it ships once per arm (8 KiB).
    sdw = cutlass.Array(cutlass.Float32, (threads_per_cta // lanes) * d, alignment=16, space=cutlass.AddressSpace.smem) if cutlass.const_expr(want_dw) else None

    if cta < n_ctas_q:
        # ---- Q class -------------------------------------------------------------
        n_groups = (n_q_rows + rpi - cutlass.Int32(1)) // rpi
        n_iters = (n_groups - cta + n_ctas_q - cutlass.Int32(1)) // n_ctas_q
        x_base = mXq.iterator.toint() if cutlass.const_expr(apply_norm) else i64_0
        x_tok = cutlass.Int64(mXq.stride[0]) if cutlass.const_expr(apply_norm) else i64_0
        rstd_base = mRstdQ.iterator.toint() if cutlass.const_expr(apply_norm) else i64_0
        w_base = mWq.iterator.toint() if cutlass.const_expr(apply_norm) else i64_0
        dw_base = (mDWq.iterator.toint() + cta.to(cutlass.Int64) * d_bytes) if cutlass.const_expr(want_dw) else i64_0
        _row_class(
            mDQ.iterator.toint(),
            cutlass.Int64(mDQ.stride[0]),
            x_base,
            x_tok,
            rstd_base,
            w_base,
            cos_base,
            cos_tok,
            sin_base,
            sin_tok,
            mOutQ.iterator.toint(),
            cutlass.Int64(mOutQ.stride[0]),
            dw_base,
            sl_base,
            n_q_rows,
            n_iters,
            cta,
            n_ctas_q,
            h_q,
            s,
            io_dtype,
            h_q_ct,
            const_head_counts,
            d,
            rope_dim,
            threads_per_cta,
            rows_per_group,
            apply_norm,
            want_dw,
            has_seq_lens,
            False,
            sdw,
        )
    else:
        if cta < n_ctas_q + n_ctas_k:
            # ---- K class ---------------------------------------------------------
            cta_k = cta - n_ctas_q
            n_groups = (n_k_rows + rpi - cutlass.Int32(1)) // rpi
            n_iters = (n_groups - cta_k + n_ctas_k - cutlass.Int32(1)) // n_ctas_k
            x_base = mXk.iterator.toint() if cutlass.const_expr(apply_norm) else i64_0
            x_tok = cutlass.Int64(mXk.stride[0]) if cutlass.const_expr(apply_norm) else i64_0
            rstd_base = mRstdK.iterator.toint() if cutlass.const_expr(apply_norm) else i64_0
            w_base = mWk.iterator.toint() if cutlass.const_expr(apply_norm) else i64_0
            dw_base = (mDWk.iterator.toint() + cta_k.to(cutlass.Int64) * d_bytes) if cutlass.const_expr(want_dw) else i64_0
            _row_class(
                mDK.iterator.toint(),
                cutlass.Int64(mDK.stride[0]),
                x_base,
                x_tok,
                rstd_base,
                w_base,
                cos_base,
                cos_tok,
                sin_base,
                sin_tok,
                mOutK.iterator.toint(),
                cutlass.Int64(mOutK.stride[0]),
                dw_base,
                sl_base,
                n_k_rows,
                n_iters,
                cta_k,
                n_ctas_k,
                h_kv,
                s,
                io_dtype,
                h_kv_ct,
                const_head_counts,
                d,
                rope_dim,
                threads_per_cta,
                rows_per_group,
                apply_norm,
                want_dw,
                has_seq_lens,
                False,
                sdw,
            )
        else:
            # ---- V class: the band copy (no norm, no RoPE, no dW) --------------------
            cta_v = cta - n_ctas_q - n_ctas_k
            n_groups = (n_k_rows + rpi - cutlass.Int32(1)) // rpi
            n_iters = (n_groups - cta_v + n_ctas_v - cutlass.Int32(1)) // n_ctas_v
            _row_class(
                mDV.iterator.toint(),
                cutlass.Int64(mDV.stride[0]),
                i64_0,
                i64_0,
                i64_0,
                i64_0,
                cos_base,
                cos_tok,
                sin_base,
                sin_tok,
                mOutV.iterator.toint(),
                cutlass.Int64(mOutV.stride[0]),
                i64_0,
                sl_base,
                n_k_rows,
                n_iters,
                cta_v,
                n_ctas_v,
                h_kv,
                s,
                io_dtype,
                h_kv_ct,
                const_head_counts,
                d,
                0,
                threads_per_cta,
                rows_per_group,
                False,
                False,
                has_seq_lens,
                True,
                sdw,
            )

    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()


@cute.kernel
def frost_dw_norm_reduce(
    mPq: cute.Tensor,  # [n_q, D] fp32 partials
    mPk: cute.Tensor,  # [n_k, D]
    mDWq: cute.Tensor,  # [D] fp32 OUT (overwritten)
    mDWk: cute.Tensor,  # [D]
    n_q: cutlass.Int32,
    n_k: cutlass.Int32,
    d: cutlass.Constexpr[int],
    cols: cutlass.Constexpr[int],
    lanes: cutlass.Constexpr[int],
    unroll: cutlass.Constexpr[int],
) -> None:
    """The SECOND launch, class ``blockIdx.y`` (0 = Q, 1 = K), columns
    ``[blockIdx.x * cols, +cols)``: a FIXED summation order, no atomics.

    Thread ``(l, j)`` sums ``partials[c, col]`` for ``c = l, l + lanes, ...``
    ASCENDING (its residue class), the ``lanes`` per-column partials go through
    SMEM and row-lane 0 adds them for ``l`` ASCENDING, then OVERWRITES
    ``dw_*_norm[col]`` (fp32). ``lanes``-way parallel over the rows instead of one
    thread per column walking every row; the shipped ``(cols, lanes) = (8, 128)``
    is the measured optimum on the SM-fill plane (the table above the module
    constants). Every partial was produced by the preceding launch on the same
    stream. Host mirror of the order:
    ``test_norm_rope_bwd_fixed_n_ctas_policy_reduces_the_same_sum``.

    SMEM: ``sPart`` fp32 ``lanes x cols`` (4 KiB at (8, 128)), written once per
    thread at ``l * cols + j`` (a warp's 32 threads write 32 consecutive fp32 =
    one 128-B line: conflict-free), read by row-lane 0. Barrier: ONE
    ``barrier_cta_sync`` per block, all ``cols * lanes`` (= 1024) threads arrive
    (no divergent path reaches it). Global reads: a warp covers ``32 / cols``
    rows x ``cols * 4`` B -- whole 32-B sectors at every ``cols >= 8``.
    """
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    j = tidx % cutlass.Int32(cols)
    l = tidx // cutlass.Int32(cols)
    col = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(cols) + j
    is_q = cutlass.Int32(cute.arch.block_idx()[1]) == cutlass.Int32(0)
    p_base = mPq.iterator.toint() if is_q else mPk.iterator.toint()
    o_base = mDWq.iterator.toint() if is_q else mDWk.iterator.toint()
    n = n_q if is_q else n_k
    sPart = cutlass.Array(cutlass.Float32, lanes * cols, alignment=16, space=cutlass.AddressSpace.smem)
    acc = opaque_f32_zero()
    if col < cutlass.Int32(d):
        # rows of row-lane l's residue class: l, l + lanes, ... < n (0 of them when l >= n)
        n_mine = (n - l + cutlass.Int32(lanes - 1)) // cutlass.Int32(lanes)
        col64 = col.to(cutlass.Int64)
        for k in cutlass.range(n_mine, unroll=unroll):
            c = l + k * cutlass.Int32(lanes)
            acc = acc + ld_global(p_base + (c.to(cutlass.Int64) * cutlass.Int64(d) + col64) * cutlass.Int64(4), cutlass.Float32)
    sPart.subview(l * cutlass.Int32(cols) + j).store(acc)
    nvvm.barrier_cta_sync()
    if l == cutlass.Int32(0):
        if col < cutlass.Int32(d):
            tot = acc
            # lanes ASCENDING, one sequential fp32 chain (the documented order); a
            # fully unrolled dynamic range, not range_constexpr: 127 static
            # iterations trip the DSL's compile-time warning
            for ll in cutlass.range(1, lanes, 1, unroll_full=True):
                tot = tot + sPart.subview(ll * cutlass.Int32(cols) + j).load()
            st_global(o_base + col.to(cutlass.Int64) * cutlass.Int64(4), tot, cutlass.Float32)


@cute.jit
def qk_norm_rope_bwd_launch(
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    xq: Optional[cute.Tensor],
    xk: Optional[cute.Tensor],
    rstd_q: Optional[cute.Tensor],
    rstd_k: Optional[cute.Tensor],
    w_q: Optional[cute.Tensor],
    w_k: Optional[cute.Tensor],
    cos: cute.Tensor,
    sin: cute.Tensor,
    out_q: cute.Tensor,
    out_k: cute.Tensor,
    out_v: cute.Tensor,
    dw_q: Optional[cute.Tensor],
    dw_k: Optional[cute.Tensor],
    seq_lens: Optional[cute.Tensor],
    n_ctas_q: cutlass.Int32,
    n_ctas_k: cutlass.Int32,
    n_ctas_v: cutlass.Int32,
    n_q_rows: cutlass.Int32,
    n_k_rows: cutlass.Int32,
    s: cutlass.Int32,
    h_q: cutlass.Int32,
    h_kv: cutlass.Int32,
    n_blocks: cutlass.Int32,
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    const_head_counts: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_qk_norm_rope_bwd(
        dq,
        dk,
        dv,
        xq,
        xk,
        rstd_q,
        rstd_k,
        w_q,
        w_k,
        cos,
        sin,
        out_q,
        out_k,
        out_v,
        dw_q,
        dw_k,
        seq_lens,
        n_ctas_q,
        n_ctas_k,
        n_ctas_v,
        n_q_rows,
        n_k_rows,
        s,
        h_q,
        h_kv,
        h_q_ct,
        h_kv_ct,
        const_head_counts,
        d,
        rope_dim,
        threads_per_cta,
        rows_per_group,
        use_pdl,
    ).launch(grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl)


@cute.jit
def dw_norm_reduce_launch(
    p_q: cute.Tensor,
    p_k: cute.Tensor,
    dw_q: cute.Tensor,
    dw_k: cute.Tensor,
    n_q: cutlass.Int32,
    n_k: cutlass.Int32,
    n_blocks: cutlass.Int32,
    d: cutlass.Constexpr[int],
    cols: cutlass.Constexpr[int],
    lanes: cutlass.Constexpr[int],
    unroll: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    frost_dw_norm_reduce(p_q, p_k, dw_q, dw_k, n_q, n_k, d, cols, lanes, unroll).launch(grid=(n_blocks, 2, 1), block=(cols * lanes, 1, 1), stream=stream)


compiled_cache = {}
reduce_cache = {}


class QkNormRopeBwdRecipe(NamedTuple):
    """Build-time facts of one norm+RoPE backward launch pair.

    Everything here is derivable from the DECLARATION (a legal compile key,
    AGENTS.md Rule 4); the token count enters as a ``cute.sym_int`` and the row
    / CTA counts ride in as runtime ``Int32``. ``n_ctas_cap`` is the per-class
    persistent-CTA cap the policy resolved to at compile time (SM-fill is
    bitwise-stable per device; ``fixed:N`` is cross-device stable).
    ``apply_norm`` / ``want_dw`` / ``has_seq_lens`` record what the artifact
    TRACED; :func:`run_qk_norm_rope_bwd` checks the operands against them in
    BOTH directions (Rule 1). ``eps`` is a recorded FACT of the forward, not a
    kernel input: the backward consumes the forward's SAVED fp32 ``rstd`` and
    never recomputes ``rsqrt(mean(x^2) + eps)``, so ``eps`` is deliberately NOT
    in the compile key (an ``eps`` kernel parameter would be dead)."""

    compiled: object
    reduce_compiled: object
    h_q: int
    h_kv: int
    d: int
    rope_dim: int
    eps: float
    rows_per_cta: int
    apply_norm: bool
    want_dw: bool
    has_seq_lens: bool
    n_ctas_cap: int
    dtype: object
    # Appended (defaulted): the rows_per_group the artifact was traced with,
    # after the per-arm default resolution.
    rows_per_group: int = DEFAULT_ROWS_PER_GROUP


def _resolve_n_ctas_cap(policy: str, device: int) -> int:
    if policy == "sm_fill":
        return int(multiprocessor_count(device)) * CTAS_PER_SM
    if policy.startswith("fixed:"):
        try:
            n = int(policy[len("fixed:") :])
        except ValueError:
            n = 0
        if n < 1:
            raise ValueError(f"n_ctas_policy='fixed:N' needs N >= 1, got {policy!r}")
        return n
    raise ValueError(f"n_ctas_policy must be 'sm_fill' or 'fixed:N', got {policy!r}")


def compile_qk_norm_rope_bwd(
    *,
    dtype,
    h_q: int,
    h_kv: int,
    d: int,
    rope_dim: int,
    eps: float,
    apply_norm: bool,
    want_dw: bool,
    has_seq_lens: bool,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    rows_per_group: Optional[int] = None,
    const_head_counts: bool = DEFAULT_CONST_HEAD_COUNTS,
    use_pdl: bool = False,
    n_ctas_policy: str = DEFAULT_N_CTAS_POLICY,
) -> QkNormRopeBwdRecipe:
    """Build both artifacts from SHAPES ALONE -- no allocation, no launch.

    ``apply_norm=False`` (the block's ``qk_norm=False``) traces the RoPE-only
    adjoint: ``x / rstd / w / dW`` slots traced as ``None``. ``want_dw`` needs
    the norm (there is no dW without one). ``rows_per_group=None`` resolves per
    arm (``DEFAULT_ROWS_PER_GROUP`` with the norm, ``..._ROPE_ONLY`` without --
    the measured optimum of each, see the module constants). Every knob, the
    policy and the device are in the cache key.
    """
    global _FAKE_STREAM
    if rows_per_group is None:
        rows_per_group = DEFAULT_ROWS_PER_GROUP if apply_norm else DEFAULT_ROWS_PER_GROUP_ROPE_ONLY
    # A row count per lane group: 0 would trace lane groups that own no rows and record
    # rows_per_cta == 0 (n_ctas_for then divides by it), a negative value a wrong CTA count,
    # a non-int a float rows_per_cta -- refused before the device is queried or anything traced.
    if not isinstance(rows_per_group, int) or rows_per_group < 1:
        raise ValueError(f"rows_per_group must be an int >= 1, got {rows_per_group!r}")
    validate_shape(d, rope_dim, threads_per_cta)
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"qk_norm_rope_bwd serves bf16/f16 only, got {dtype}")
    if want_dw and not apply_norm:
        raise ValueError("apply_norm=False (RoPE-only adjoint) computes no RMSNorm and therefore no dW_norm; want_dw must be False")
    device = current_device()
    n_ctas_cap = _resolve_n_ctas_cap(str(n_ctas_policy), device)
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    key = (
        str(dtype),
        int(h_q),
        int(h_kv),
        int(d),
        int(rope_dim),
        int(threads_per_cta),
        int(rows_per_group),
        bool(const_head_counts),
        bool(use_pdl),
        bool(apply_norm),
        bool(want_dw),
        bool(has_seq_lens),
        str(n_ctas_policy),
        device,
    )
    if key not in compiled_cache:
        tok = cute.sym_int()
        # Every [T, H, D] operand carries its OWN symbolic token stride: dq/dk/dv
        # are compact, x_q/x_k are proj_slab bands, out_* are dqkvg bands.
        dense = [fake_rowmajor_dynamic_token_stride(dtype, tok, h, d) for h in (h_q, h_kv, h_kv)]
        xs = [fake_rowmajor_dynamic_token_stride(dtype, tok, h, d) for h in (h_q, h_kv)] if apply_norm else [None, None]
        rstd = [_fake(torch.float32, (tok, h), (1, 0)) for h in (h_q, h_kv)] if apply_norm else [None, None]
        weights = [_fake(dtype, (d,), (0,)) for _ in range(2)] if apply_norm else [None, None]
        tables = [_fake(dtype, (tok, rope_dim if rope_dim else 1), (1, 0)) for _ in range(2)]
        outs = [fake_rowmajor_dynamic_token_stride(dtype, tok, h, d) for h in (h_q, h_kv, h_kv)]
        # Two DISTINCT symbolic row counts: the Q and K planes differ in size.
        dws = [_fake(torch.float32, (cute.sym_int(), d), (1, 0)) for _ in range(2)] if want_dw else [None, None]
        seq_lens = (
            cute.runtime.make_fake_compact_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(),), stride_order=(0,), assumed_align=4) if has_seq_lens else None
        )
        compiled_cache[key] = cute.compile(
            qk_norm_rope_bwd_launch,
            *dense,
            *xs,
            *rstd,
            *weights,
            *tables,
            *outs,
            *dws,
            seq_lens,
            cutlass.Int32(0),  # n_ctas_q ) all runtime values; the zeros only pin
            cutlass.Int32(0),  # n_ctas_k ) their TYPE at trace time, the real ones
            cutlass.Int32(0),  # n_ctas_v ) are bound per launch.
            cutlass.Int32(0),  # n_q_rows
            cutlass.Int32(0),  # n_k_rows
            cutlass.Int32(0),  # s
            cutlass.Int32(h_q),
            cutlass.Int32(h_kv),
            cutlass.Int32(0),  # n_blocks
            int(h_q),
            int(h_kv),
            bool(const_head_counts),
            int(d),
            int(rope_dim),
            int(threads_per_cta),
            int(rows_per_group),
            bool(use_pdl),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    reduce_key = (int(d), device)
    if want_dw and reduce_key not in reduce_cache:
        planes = [_fake(torch.float32, (cute.sym_int(), d), (1, 0)) for _ in range(2)]
        outs32 = [_fake(torch.float32, (d,), (0,)) for _ in range(2)]
        reduce_cache[reduce_key] = cute.compile(
            dw_norm_reduce_launch,
            *planes,
            *outs32,
            cutlass.Int32(0),  # n_q
            cutlass.Int32(0),  # n_k
            cutlass.Int32(0),  # n_blocks
            int(d),
            REDUCE_COLS,
            REDUCE_LANES,
            REDUCE_UNROLL,
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return QkNormRopeBwdRecipe(
        compiled=compiled_cache[key],
        reduce_compiled=reduce_cache[reduce_key] if want_dw else None,
        h_q=int(h_q),
        h_kv=int(h_kv),
        d=int(d),
        rope_dim=int(rope_dim),
        eps=float(eps),
        rows_per_cta=(threads_per_cta // lanes_per_row(d)) * rows_per_group,
        apply_norm=bool(apply_norm),
        want_dw=bool(want_dw),
        has_seq_lens=bool(has_seq_lens),
        n_ctas_cap=int(n_ctas_cap),
        dtype=dtype,
        rows_per_group=int(rows_per_group),
    )


def n_ctas_for(r: QkNormRopeBwdRecipe, t: int) -> tuple:
    """``(n_ctas_q, n_ctas_k, n_ctas_v)`` for ``t`` tokens: ``min(row_groups_x, cap)``
    PER CLASS, at least 1. ``dw_partials_q`` / ``dw_partials_k`` must be exactly
    ``[n_ctas_q, D]`` / ``[n_ctas_k, D]`` fp32 for that ``t`` -- slice the workspace
    plane to it (the reduce sums every row of the plane it is handed)."""
    t = int(t)
    out = []
    for n_rows in (t * r.h_q, t * r.h_kv, t * r.h_kv):
        groups = (n_rows + r.rows_per_cta - 1) // r.rows_per_cta
        out.append(max(1, min(groups, r.n_ctas_cap)))
    return tuple(out)


def _check_dense(r: QkNormRopeBwdRecipe, name: str, ten, t: int, h: int) -> None:
    if ten.dtype != r.dtype:
        raise ValueError(f"{name} is {ten.dtype} but this artifact was compiled for {r.dtype}; dtype is fixed per artifact")
    if ten.dim() != 3 or int(ten.shape[1]) != h:
        raise ValueError(f"{name} has shape {tuple(ten.shape)} but this artifact was compiled for H={h}; H is fixed per artifact (operands are [T, H, D])")
    if int(ten.shape[2]) != r.d:
        raise ValueError(f"{name} has D={int(ten.shape[2])} but this artifact was compiled for D={r.d}; D is fixed per artifact")
    if int(ten.shape[0]) != t:
        raise ValueError(f"{name} has T={int(ten.shape[0])} but dq has T={t}; every operand covers the same tokens")
    _check_row_layout(name, ten, r.d)


def _check_fp32_plane(name: str, ten, shape: tuple) -> None:
    want = "[" + ", ".join("*" if x is None else str(x) for x in shape) + "]"
    ok = (
        ten is not None
        and ten.dtype == torch.float32
        and ten.is_contiguous()
        and ten.dim() == len(shape)
        and all(x is None or int(a) == x for a, x in zip(ten.shape, shape))
    )
    if not ok:
        got = "None" if ten is None else f"{ten.dtype} of shape {tuple(ten.shape)} (contiguous={ten.is_contiguous()})"
        raise ValueError(f"{name} must be a contiguous fp32 {want} tensor, got {got}")


def _check_presence(r: QkNormRopeBwdRecipe, xq, xk, rstd_q, rstd_k, w_q, w_k, dw_q, dw_k, seq_lens) -> None:
    """Both directions for every presence switch, typed, before any compile / launch (Rule 1)."""
    for label, a, b in (
        ("x_q / x_k", xq, xk),
        ("rstd_q / rstd_k", rstd_q, rstd_k),
        ("w_q_norm / w_k_norm", w_q, w_k),
        ("dw_partials_q / dw_partials_k", dw_q, dw_k),
    ):
        if (a is None) != (b is None):
            raise ValueError(f"{label} must be given together or both be None; got {'None' if a is None else 'tensor'} / {'None' if b is None else 'tensor'}")
    norm_bound = [xq is not None, rstd_q is not None, w_q is not None]
    if r.apply_norm and not all(norm_bound):
        raise ValueError(
            "this artifact was compiled WITH the RMSNorm backward (apply_norm=True); x_q/x_k, rstd_q/rstd_k and both [D] norm weights must be bound at execute (Rule 1: no silent fallback)"
        )
    if not r.apply_norm and any(norm_bound):
        raise ValueError(
            "this artifact was compiled WITHOUT the RMSNorm (apply_norm=False, RoPE-only adjoint); pass x_q/x_k, rstd_q/rstd_k and the norm weights as None -- they would be silently ignored (Rule 1)"
        )
    if r.want_dw and dw_q is None:
        raise ValueError(
            "this artifact was compiled WITH dW partials (want_dw=True); both dw_partials_q / dw_partials_k must be bound at execute (Rule 1: no silent fallback)"
        )
    if not r.want_dw and dw_q is not None:
        raise ValueError(
            "this artifact was compiled WITHOUT dW partials (want_dw=False); passing dw_partials_q / dw_partials_k would silently ignore them (Rule 1)"
        )
    if r.has_seq_lens and seq_lens is None:
        raise ValueError("this artifact was compiled WITH seq_lens (the dead-row select); it must be bound at execute (Rule 1: no silent fallback)")
    if not r.has_seq_lens and seq_lens is not None:
        raise ValueError("this artifact was compiled WITHOUT seq_lens (dense, the select folded out); passing one would silently ignore it (Rule 1)")


def run_qk_norm_rope_bwd(
    r: QkNormRopeBwdRecipe,
    dq,
    dk,
    dv,
    xq,
    xk,
    rstd_q,
    rstd_k,
    w_q,
    w_k,
    cos,
    sin,
    out_q,
    out_k,
    out_v,
    dw_partials_q=None,
    dw_partials_k=None,
    seq_lens=None,
    *,
    s: Optional[int] = None,
    stream,
) -> None:
    """The FIRST launch: dx into the Q / K / V bands, per-CTA dW partials.

    ``dq / dk / dv`` compact ``[T, H, D]`` (``T = B*S``); ``xq / xk`` bands or
    compact; ``rstd_*`` ``[T, H]`` fp32 contiguous; ``w_*`` ``[D]``; ``cos / sin``
    ``[T, rope_dim]``; ``out_*`` the dqkvg bands (``out_q`` may alias ``dq``,
    ``out_k`` may alias ``dk``); ``dw_partials_*`` exactly ``[n_ctas_x, D]`` fp32
    per :func:`n_ctas_for`; ``seq_lens`` (``[B]`` int32) needs ``s``. Host-only
    checks, no allocation, no key build.
    """
    _check_presence(r, xq, xk, rstd_q, rstd_k, w_q, w_k, dw_partials_q, dw_partials_k, seq_lens)
    t = int(dq.shape[0])
    for name, ten, h in (
        ("dq", dq, r.h_q),
        ("dk", dk, r.h_kv),
        ("dv", dv, r.h_kv),
        ("out_q", out_q, r.h_q),
        ("out_k", out_k, r.h_kv),
        ("out_v", out_v, r.h_kv),
    ):
        _check_dense(r, name, ten, t, h)
    if r.apply_norm:
        _check_dense(r, "x_q", xq, t, r.h_q)
        _check_dense(r, "x_k", xk, t, r.h_kv)
        _check_fp32_plane("rstd_q", rstd_q, (t, r.h_q))
        _check_fp32_plane("rstd_k", rstd_k, (t, r.h_kv))
        for name, w in (("w_q_norm", w_q), ("w_k_norm", w_k)):
            if w.dtype != r.dtype or tuple(w.shape) != (r.d,) or not w.is_contiguous():
                raise ValueError(f"{name} must be a contiguous [{r.d}] {r.dtype} vector, got {w.dtype} of shape {tuple(w.shape)}")
    tab_cols = r.rope_dim if r.rope_dim else 1
    for name, tab in (("cos", cos), ("sin", sin)):
        if tab.dtype != r.dtype or tab.dim() != 2 or int(tab.shape[0]) != t or int(tab.shape[1]) != tab_cols or tab.stride(1) != 1:
            raise ValueError(f"{name} must be a [{t}, {tab_cols}] {r.dtype} table with unit column stride, got {tab.dtype} of shape {tuple(tab.shape)}")
    n_ctas_q, n_ctas_k, n_ctas_v = n_ctas_for(r, t)
    if r.want_dw:
        _check_fp32_plane("dw_partials_q", dw_partials_q, (n_ctas_q, r.d))
        _check_fp32_plane("dw_partials_k", dw_partials_k, (n_ctas_k, r.d))
    if seq_lens is not None:
        if s is None:
            raise ValueError("seq_lens needs s (the per-batch sequence length, T == B * s) to map a token to its batch entry")
        if seq_lens.dtype != torch.int32 or seq_lens.dim() != 1:
            raise ValueError(f"seq_lens must be a 1-D int32 tensor, got {seq_lens.dtype} of shape {tuple(seq_lens.shape)}")
        if int(s) <= 0 or t % int(s) != 0:
            raise ValueError(f"s={s} must divide T={t} (T == B * s)")
        if int(seq_lens.numel()) != t // int(s):
            raise ValueError(f"seq_lens has {int(seq_lens.numel())} entries but T // s = {t // int(s)} batches")
    _check_one_cuda_device(
        "dq",
        dq,
        (
            ("dk", dk),
            ("dv", dv),
            ("x_q", xq),
            ("x_k", xk),
            ("rstd_q", rstd_q),
            ("rstd_k", rstd_k),
            ("w_q_norm", w_q),
            ("w_k_norm", w_k),
            ("cos", cos),
            ("sin", sin),
            ("out_q", out_q),
            ("out_k", out_k),
            ("out_v", out_v),
            ("dw_partials_q", dw_partials_q),
            ("dw_partials_k", dw_partials_k),
            ("seq_lens", seq_lens),
        ),
    )
    # The optional slots stay in the ABI even when they traced to None (the
    # artifact folded out the loads / stores, not the parameters).
    r.compiled(
        dq,
        dk,
        dv,
        xq,
        xk,
        rstd_q,
        rstd_k,
        w_q,
        w_k,
        cos,
        sin,
        out_q,
        out_k,
        out_v,
        dw_partials_q,
        dw_partials_k,
        seq_lens,
        cutlass.Int32(n_ctas_q),
        cutlass.Int32(n_ctas_k),
        cutlass.Int32(n_ctas_v),
        cutlass.Int32(t * r.h_q),
        cutlass.Int32(t * r.h_kv),
        cutlass.Int32(int(s) if s is not None else 0),
        cutlass.Int32(r.h_q),
        cutlass.Int32(r.h_kv),
        cutlass.Int32(n_ctas_q + n_ctas_k + n_ctas_v),
        cuda.CUstream(int(stream)),
    )


def run_dw_norm_reduce(r: QkNormRopeBwdRecipe, dw_partials_q, dw_partials_k, dw_q_norm, dw_k_norm, *, stream, t: Optional[int] = None) -> None:
    """The SECOND launch: for every column, ``REDUCE_LANES`` row-lanes each sum
    their residue class of partial rows ascending, then the lanes are added
    ascending (fixed order, no atomics); OVERWRITES ``dw_*_norm`` (fp32 ``[D]``,
    contiguous). The planes' row counts ARE the partial counts (slice the
    workspace to :func:`n_ctas_for`). ``t`` (appended, optional): the token count
    of the launch that wrote the planes -- when given, each plane must have
    EXACTLY ``n_ctas_for(r, t)`` rows, so a full workspace plane handed to the
    reduce for a smaller ``t`` is a typed error instead of a sum over unwritten
    rows (residue, NaN under poisoning). Pass it."""
    if not r.want_dw:
        raise ValueError("this artifact was compiled WITHOUT dW partials (want_dw=False); there is nothing to reduce (Rule 1)")
    if t is not None:
        n_ctas_q, n_ctas_k, _ = n_ctas_for(r, t)
        _check_fp32_plane("dw_partials_q", dw_partials_q, (n_ctas_q, r.d))
        _check_fp32_plane("dw_partials_k", dw_partials_k, (n_ctas_k, r.d))
    _check_fp32_plane("dw_partials_q", dw_partials_q, (None, r.d))
    _check_fp32_plane("dw_partials_k", dw_partials_k, (None, r.d))
    _check_fp32_plane("dw_q_norm", dw_q_norm, (r.d,))
    _check_fp32_plane("dw_k_norm", dw_k_norm, (r.d,))
    n_q, n_k = int(dw_partials_q.shape[0]), int(dw_partials_k.shape[0])
    if n_q < 1 or n_k < 1:
        raise ValueError(f"the partial planes need at least one row each, got {n_q} / {n_k}")
    _check_one_cuda_device("dw_partials_q", dw_partials_q, (("dw_partials_k", dw_partials_k), ("dw_q_norm", dw_q_norm), ("dw_k_norm", dw_k_norm)))
    n_blocks = (r.d + REDUCE_COLS - 1) // REDUCE_COLS
    r.reduce_compiled(
        dw_partials_q, dw_partials_k, dw_q_norm, dw_k_norm, cutlass.Int32(n_q), cutlass.Int32(n_k), cutlass.Int32(n_blocks), cuda.CUstream(int(stream))
    )


def moved_bytes(t: int, h_q: int, h_kv: int, d: int, *, elem_bytes: int = 2, apply_norm: bool = True) -> int:
    """HBM traffic of the main launch -- the denominator for an SOL number.

    Q and K rows: read ``dy``, read ``x`` (norm only), write ``dx``, plus one fp32
    ``rstd`` per row (norm only). V rows: read + write. cos / sin, the ``[D]``
    weights, ``seq_lens`` and the dW partials plane (``n_ctas x D x 4 B``, ~3.5 MB
    at SM-fill) are excluded, the forward's convention.
    """
    qk_rows = t * (h_q + h_kv)
    qk = (3 if apply_norm else 2) * qk_rows * d * elem_bytes + (qk_rows * 4 if apply_norm else 0)
    v = 2 * t * h_kv * d * elem_bytes
    return qk + v


frost_qk_norm_rope_bwd.set_name_prefix("cudnn", remove_cutlass_symbol=True)
frost_dw_norm_reduce.set_name_prefix("cudnn", remove_cutlass_symbol=True)
