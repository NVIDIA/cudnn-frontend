# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This kernel is derived from cuDNN, NVIDIA Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exact intra-tile decay for steep key channels in the GDN-2 BT = 16 kernels.

The chunk kernels split the intra-tile decay of key channel d between tokens t >= s,
``2^(g[t, d] - g[s, d])`` with ``g`` the inclusive log2 gate prefix inside the 16-token
tile, across the two operands of one register MMA: K decay / Q decay carry
``exp2(+g[t, d])`` and K inv carries ``rcp(exp2(g[s, d]))``.  Once a prefix falls below
-126, ``exp2(+g)`` leaves the fp32 normal range, the approximate reciprocal returns inf,
and ``0 * inf`` turns KK, A and the state update NaN for the rest of the sequence.

A channel needs exact decay when its prefix falls below the dtype threshold (``-120`` for bf16, ``-14`` for fp16),
or its raw key magnitude makes the inverse operand unsafe. Normalized keys have magnitude at most one;
``channel_needs_exact_decay`` also bounds unnormalized keys before selecting the path.

Dispatch.  The steep path costs registers and code even where it never runs, so each kernel's ``host`` launches it
twice (``cfg.steep_mode``):

- ``STEEP_FAST``: the original arithmetic plus a range check and one ballot per scan warp; a warp that finds a steep channel stores 1 into
  the second int32 of the kernel's scheduler slot, which the prologue zeroes every call.  A steep tile's outputs are
  garbage at this point.
- ``STEEP_EXACT``: launched right after, from the same host, with the steep path compiled in.  Every thread exits on
  entry while the flag is 0, so a call without a steep tile pays one empty launch; otherwise the kernel redoes the whole
  launch exactly (pulling tiles off the flag int, past its 1), overwriting every output before the next kernel reads it.
  No host synchronisation is involved, so the pair is safe under CUDA graphs and ``torch.compile``.
A launch whose output aliases a state it reads (``overwrite_initial_state``, a ``state_indices`` pool, the backward's
``d_initial_state`` over ``d_final_state``) reads that state from a workspace snapshot (common/state_snapshot.py), so
the exact launch never reads the fast launch's garbage.

The steep path (``STEEP_EXACT``):

- the gate scan retains individual log2 gates for steep channels on exchange rows ``0 .. BT - 2``;
  the final token's log2 gate goes to a per-channel array. The last exchange row retains ``exp2(prefix)``
  for state decay. Ballot bits select this representation, including zero gates and padding;
- exact decay sums only the intervening token logs. Subtracting inclusive fp32 prefixes would lose small
  gates after a large negative spike;
- the row stage selects valid scales before constructing operands (``row_operand_scales``): K decay / Q decay use
  ``exp2(g[t])``, which may underflow to zero harmlessly; K inv and K restore use
  ``k[t] * exp2(g[BT - 1] - g[t])``, whose exponent is never positive;
- ``steep_row_corrections`` adds the steep channels' exact KK and A entries, every
  exponent ``g[t] - g[s] <= 0``, minus the products the register MMAs formed with
  their K inv column; the register-MMA warps add the corrections to their
  accumulators;
- the backward picks bounded per-channel scales before it assembles gradients (``gradient_operand_scales``), so it
  never reads token logs as prefix exponentials and repairs those lanes afterwards; ``steep_grad_corrections`` adds the
  exact intra-tile terms;
- ``common/gdn2_state_grad.py`` retains the steep channels' original state gradients in FP32 scratch. Their tensor-core
  operand is zero; bounded FP32 projections supply dU, restored dK and dGate. Decay is applied before the gate dot
  product. This avoids overflowing a finite FP32 seed or carry when it is packed into FP16.

Synchronisation invariants of the steep path:

- per-group scan ballots alternate between two slots where a mild tile skips the exact path's final group barrier, so
  its next iteration can publish before every warp has read the previous ballots; the next scan's publication barrier
  keeps a group from running two iterations ahead. Last-token and norm scratch stay behind that final barrier. The
  ``alternating`` gate pattern covers mild/steep transitions in every scan group;
- every correction writer arrives on the K/Q-ready barriers itself: the async-proxy fence orders only the issuing
  thread's accesses and does not join the other lanes that wrote correction rows. ``STEEP_FAST`` keeps its original
  elected-lane arrivals and warp-sized counts. Readers join before an elected lane releases a correction stage;
- the exact backward's dA and dM writers both arrive on the existing diagonal-ready barrier after waiting for its slot,
  counted directly. ``STEEP_FAST`` keeps the single-producer count because it does not consume the dM copy;
- the public summary ops have no overwrite attribute, so their state outputs never alias and their wrappers launch
  ``STEEP_FAST``.
"""

import dataclasses

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm

from cudnn.frost.tile_dsl.pointwise import sigmoid
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b

STEEP_LOG2_PREFIX: float = -120.0
"""bf16 operands: a key channel whose inclusive log2 gate prefix inside a tile falls below this is steep for the tile.
``exp2`` of the prefix reaches the fp32 (and bf16) normal floor, ``2^-126``, six binades further down."""

STEEP_LOG2_PREFIX_FP16: float = -14.0
"""fp16 operands: K inv = k * 2^-g reaches the fp16 maximum, ``65504 ~ 2^16``, two binades further down."""


def steep_log2_prefix(io_dtype) -> float:
    """The steep threshold for the operand dtype (a trace-time constant)."""
    return STEEP_LOG2_PREFIX_FP16 if io_dtype == cutlass.Float16 else STEEP_LOG2_PREFIX


STEEP_FAST: int = 0
"""Kernel ``cfg.steep_mode``: the original arithmetic plus the steep vote; a steep tile raises the call's steep flag.
``host`` follows the launch with a ``STEEP_EXACT`` launch of the same kernel."""

STEEP_EXACT: int = 1
"""The exact kernel of a ``STEEP_FAST`` launch: every thread exits on entry unless the flag is raised, else it redoes
the whole launch with the steep path, pulling tiles from its own ticket."""

STEEP_FLAG_SLOT: int = 1
"""The steep flag and the exact launch's ticket share the second int32 of the kernel's scheduler slot, which the
prologue zeroes with the first: 0 means no steep tile, and the exact launch's tickets start at the 1 the fast launch
stored."""


def has_exact_path(cfg) -> bool:
    """Whether ``cfg``'s kernel compiles the steep path in (``STEEP_EXACT``; a trace-time constant)."""
    return cfg.steep_mode != STEEP_FAST


def steep_launch_cfg(cfg, launch_idx):
    """Compile-time launch order: the requested specialization, then its exact twin."""
    return cfg if launch_idx == 0 else dataclasses.replace(cfg, steep_mode=STEEP_EXACT)


@cute.jit
def steep_exit_unless_flagged(cfg: cutlass.Constexpr, flag_slot):
    """``STEEP_EXACT`` kernel entry, after the grid-dependency wait: leave unless the fast launch raised the flag.

    Do not trigger dependents here. An early trigger lets the next persistent kernel's CTAs be dispatched while this
    grid's CTAs are still being placed, and that kernel then ran about 12% slower (B300, mild gates). Its dependents
    launch when this grid completes, which they wait for anyway."""
    if cutlass.const_expr(cfg.steep_mode == STEEP_EXACT):
        if flag_slot[STEEP_FLAG_SLOT] == cutlass.Int32(0):
            nvvm.exit()


@cute.jit
def steep_raise_flag(cfg: cutlass.Constexpr, flag_slot, steep_word, lane_idx):
    """``STEEP_FAST`` scan: a warp whose ballot found a steep channel raises the flag (idempotent, never read back)."""
    if cutlass.const_expr(cfg.steep_mode == STEEP_FAST):
        if steep_word != 0:
            if lane_idx == 0:
                flag_slot[STEEP_FLAG_SLOT] = cutlass.Int32(1)


@cute.jit
def steep_ticket(cfg: cutlass.Constexpr, mScheduler):
    """The next ticket off the scheduler slot: the first int32, or under ``STEEP_EXACT`` the flag slot past its 1."""
    fetched = cutlass.Int32(0)
    if cutlass.const_expr(cfg.steep_mode == STEEP_EXACT):
        fetched = cutlass.Int32(nvvm.atomicrmw("add", mScheduler.iterator + STEEP_FLAG_SLOT, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu")) - 1
    else:
        fetched = cutlass.Int32(nvvm.atomicrmw("add", mScheduler.iterator, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu"))
    return fetched


@cute.jit
def exchange_offset(cfg: cutlass.Constexpr, row, channel):
    """Element offset of ``(row, channel)`` in one ``[BT, DK]`` fp32 gate-exchange stage."""
    seg = channel // 32
    return seg * (cfg.b_t * 32) + row * 32 + swizzle_xor_128b(row ^ seg, channel - seg * 32, elem_bytes=4)


@cute.jit
def raw_offset(cfg: cutlass.Constexpr, row, channel):
    """Element offset of ``(row, channel)`` in one ``[BT, DK]`` 16-bit row-major operand tile."""
    seg = channel // 64
    return seg * (cfg.b_t * 64) + row * 64 + swizzle_xor_128b(row, channel - seg * 64, elem_bytes=2)


@cute.jit
def channel_needs_exact_decay(cfg: cutlass.Constexpr, prefix, raw_k_ptr, channel):
    """Bound the inverse operand over a channel before selecting its decay path.

    Normalized keys have magnitude at most one. Otherwise scan the raw tile
    and reserve one binade below the operand maximum for reciprocal rounding.
    The last prefix is the minimum, so this bounds every row's inverse.
    """
    threshold = cutlass.Float32(steep_log2_prefix(cfg.io_dtype))
    if cutlass.const_expr(not cfg.l2norm):
        max_key = cutlass.Float32(1.0)
        for row in cutlass.range(cfg.b_t, unroll=1):
            key = (raw_k_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32)
            max_key = cute.math.max(max_key, cute.math.abs(key))
        limit = cutlass.Float32(15.0 if cfg.io_dtype == cutlass.Float16 else 127.0)
        threshold = cute.math.max(threshold, cute.math.log2(max_key) - limit)
    return prefix < threshold


@cute.jit
def classify_steep_channel(cfg: cutlass.Constexpr, prefix, raw_k_ptr, channel, channel_rows: cutlass.Constexpr, flag_slot, lane_idx):
    """Classify a scan lane, mask duplicate lanes, and flag a FAST relaunch.

    Callers retain ownership of stage offsets, scratch stores and barriers.
    """
    channel_steep = channel_needs_exact_decay(cfg, prefix, raw_k_ptr, channel)
    word = cutlass.Int32(nvvm.vote_sync(0xFFFFFFFF, channel_steep, nvvm.VoteSync.BALLOT))
    if cutlass.const_expr(channel_rows < cfg.threads_per_warp):
        word = word & cutlass.Int32((1 << channel_rows) - 1)
    steep_raise_flag(cfg, flag_slot, word, lane_idx)
    return channel_steep, word


@cute.jit
def encode_gate_exchange(
    cfg: cutlass.Constexpr,
    prefix,
    raw_k_ptr,
    channel,
    channel_rows: cutlass.Constexpr,
    flag_slot,
    lane_idx,
    scan_warp,
    token_logs,
    values,
    words_ptr,
    log_last_ptr,
    channel_active=True,
):
    """Classify a channel and encode its exchange registers and scan metadata.

    ``values`` always holds inclusive log2 prefixes on entry. Exact steep
    channels retain token logs except in the last row,
    which always holds the state-decay exponential. Callers own exchange stores,
    stage selection and publication barriers; this helper never synchronizes.
    """
    channel_steep, word = classify_steep_channel(cfg, prefix, raw_k_ptr, channel, channel_rows, flag_slot, lane_idx)
    if cutlass.const_expr(has_exact_path(cfg)):
        if lane_idx == 0:
            (words_ptr + scan_warp).store(word)
        if word != 0:
            if channel_steep:
                if channel_active:
                    (log_last_ptr + channel).store(token_logs[cfg.b_t - 1])
    for row in cutlass.range_constexpr(cfg.b_t):
        values[row] = cute.math.exp2(values[row], fastmath=True)
        if cutlass.const_expr(has_exact_path(cfg) and row < cfg.b_t - 1):
            if channel_steep:
                values[row] = token_logs[row]
    return word


@cute.jit
def steep_tile_mask(cfg: cutlass.Constexpr, words_ptr):
    """Read the tile's four scan-warp ballots; FAST never reads exact-path scratch."""
    mask = cutlass.Int32(0)
    if cutlass.const_expr(has_exact_path(cfg)):
        mask = (words_ptr.load() | (words_ptr + 1).load()) | ((words_ptr + 2).load() | (words_ptr + 3).load())
    return mask


@cute.jit
def lane_steep_bits(cfg: cutlass.Constexpr, words_ptr, lane_in_row_group, channel_rows: cutlass.Constexpr[int]):
    """The steep channels a row-stage lane owns: bit ``h * 8 + j`` is channel ``h * 64 + lane_in_row_group * 8 + j``.

    ``words_ptr`` holds one ballot word per scan warp; channel ``c`` is bit ``c % channel_rows`` of word
    ``c // channel_rows``."""
    bits = cutlass.Int32(0)
    for dim_half in cutlass.range_constexpr(cfg.d_k // 64):
        first = dim_half * 64 + lane_in_row_group * 8
        word = (words_ptr + first // channel_rows).load()
        bits = bits | (((word >> (first % channel_rows)) & 0xFF) << (dim_half * 8))
    return bits


@cute.jit
def steep_channel_bit(words_ptr, channel, channel_rows: cutlass.Constexpr[int]):
    """1 if ``channel`` is steep in the ballot words at ``words_ptr`` (one word per scan warp of ``channel_rows`` channels), else 0."""
    return ((words_ptr + channel // channel_rows).load() >> (channel % channel_rows)) & 1


@cute.jit
def steep_beta(cfg: cutlass.Constexpr, beta_ptr, row, channel):
    """The erase gate the operands used, from the chunk's raw beta tile (beta_eff under the guard, written back)."""
    beta = (beta_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32)
    if cutlass.const_expr(cfg.beta_sigmoid and not cfg.beta_guard):
        beta = (sigmoid(beta) * (2.0 if cfg.allow_neg_eigval else 1.0)).to(cfg.io_dtype).to(cutlass.Float32)
    return beta


@cute.jit
def steep_token_log(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, row, channel):
    """Read a steep channel's conditioned per-token log2 gate, including zero padding."""
    value = (log_last_ptr + channel).load()
    if row < cfg.b_t - 1:
        value = (exchange_ptr + exchange_offset(cfg, row, channel)).load()
    return value


@cute.jit
def steep_interval_log(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, start, end, channel):
    """Sum [start, end) directly: prefix subtraction loses mild gates after large spikes."""
    total = cutlass.Float32(0.0)
    for token in cutlass.range(start, end, unroll=1):
        total = total + steep_token_log(cfg, exchange_ptr, log_last_ptr, token, channel)
    return total


@cute.jit
def steep_decay_scale(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, row, channel):
    """A steep channel's K decay / Q decay scale on ``row``, ``exp2(g[row])``, at most 1 (may underflow to zero)."""
    return cute.math.exp2(steep_interval_log(cfg, exchange_ptr, log_last_ptr, 0, row + 1, channel), fastmath=True)


@cute.jit
def steep_restore_scale(cfg: cutlass.Constexpr, exchange_ptr, log_last_ptr, row, channel):
    """A steep channel's K inv / K restore scale on ``row``, ``exp2(g[BT - 1] - g[row])``, at most 1."""
    return cute.math.exp2(steep_interval_log(cfg, exchange_ptr, log_last_ptr, row + 1, cfg.b_t, channel), fastmath=True)


@cute.jit
def gradient_operand_scales(cfg: cutlass.Constexpr, channel_steep, row, channel, value, exchange_ptr, log_last_ptr):
    """Return decay, inverse-gradient and restore scales before gradient assembly.

    Steep channels use bounded prefix/suffix factors. Their inverse-gradient
    contribution comes from the exact intra-tile correction, so its scale is zero.
    """
    decay = value
    inverse = cutlass.Float32(0.0)
    restore = cutlass.Float32(0.0)
    if channel_steep != 0:
        decay = steep_decay_scale(cfg, exchange_ptr, log_last_ptr, row, channel)
        restore = steep_restore_scale(cfg, exchange_ptr, log_last_ptr, row, channel)
    else:
        inverse = cute.math.rcp(value, approx=True, ftz=True)
        restore = inverse
    return decay, inverse, restore


@cute.jit
def row_operand_scales(cfg: cutlass.Constexpr, bits, reg_idx, row, channel, value, last_value, exchange_ptr, log_last_ptr):
    """Return decay, inverse and restore-multiplier scales before operand construction.

    The fast specialization retains the original factorization. Exact channels
    put the bounded restore scale in K inv; K restore then uses multiplier one.
    All other channels retain the original operation order and rounding.
    """
    decay = value
    inverse = cute.math.rcp(value, approx=True, ftz=True)
    restore_multiplier = last_value
    if cutlass.const_expr(has_exact_path(cfg)):
        if ((bits >> reg_idx) & 1) != 0:
            decay = steep_decay_scale(cfg, exchange_ptr, log_last_ptr, row, channel)
            inverse = steep_restore_scale(cfg, exchange_ptr, log_last_ptr, row, channel)
            restore_multiplier = cutlass.Float32(1.0)
    return decay, inverse, restore_multiplier


@cute.jit
def steep_row_corrections(
    cfg: cutlass.Constexpr,
    bits,
    row,
    lane_in_row_group,
    k_norm,
    q_scale,
    exchange_ptr,
    log_last_ptr,
    raw_k_ptr,
    raw_q_ptr,
    beta_ptr,
    k_norm_ptr,
    k_inv_ptr,
    k_decay_ptr,
    q_decay_ptr,
    corr_ptr,
):
    """Add the steep channels' KK and A entries of tile row ``row`` to ``corr_ptr``.

    Called by all lanes of the row stage after its K inv / K decay stores.  ``bits`` is the lane's
    :func:`lane_steep_bits`; ``k_norm`` / ``q_scale`` the row's K inverse norm and Q factor (the inverse norm, times
    the attention scale where the A tile carries it); ``raw_q_ptr`` is ``None`` without an A tile.  Every operand is
    read from the raw and operand tiles. ``corr_ptr`` is ``[2, BT, BT]`` fp32, the KK rows then the A rows.
    Each row group reduces in a fixed order and one lane stores each entry, for
    ``s <= row`` and each steep channel,

        A[row, s]  += q_n[row] k_n[s] 2^(g[row] - g[s]) - Q decay[row] K inv[s]
        KK[row, s] += beta[row] k_n[row] k_n[s] 2^(g[row] - g[s]) - K decay[row] K inv[s]   (s < row)

    using the stored K decay / Q decay / K inv values for the second terms. The correction does not duplicate
    operand rounding. The row-stage barrier must make all operand rows visible before this call."""
    with_a = cutlass.const_expr(raw_q_ptr is not None)
    kk_regs = cutlass.Array(cutlass.Float32, cfg.b_t)
    attn_regs = cutlass.Array(cutlass.Float32, cfg.b_t)
    for source_row in cutlass.range_constexpr(cfg.b_t):
        kk_regs[source_row] = cutlass.Float32(0.0)
        attn_regs[source_row] = cutlass.Float32(0.0)
    for j in cutlass.range(cfg.d_k // 64 * 8, unroll=1):
        if ((bits >> j) & 1) != 0:
            channel = (j // 8) * 64 + lane_in_row_group * 8 + (j % 8)
            x = ((raw_k_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32) * k_norm) * steep_beta(cfg, beta_ptr, row, channel)
            x_decay = (k_decay_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32)
            if cutlass.const_expr(with_a):
                z = (raw_q_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32) * q_scale
                z_decay = (q_decay_ptr + raw_offset(cfg, row, channel)).load().to(cutlass.Float32)
            interval = cutlass.Float32(0.0)
            # Grow an interval backward from its destination. Never subtract prefixes.
            for step in cutlass.range_constexpr(cfg.b_t):
                source_row = cfg.b_t - 1 - step
                if source_row <= row:
                    if source_row < row:
                        interval = interval + steep_token_log(cfg, exchange_ptr, log_last_ptr, source_row + 1, channel)
                    k_s = (raw_k_ptr + raw_offset(cfg, source_row, channel)).load().to(cutlass.Float32)
                    if cutlass.const_expr(k_norm_ptr is not None):
                        k_s = k_s * (k_norm_ptr + source_row).load()
                    k_inv_s = (k_inv_ptr + raw_offset(cfg, source_row, channel)).load().to(cutlass.Float32)
                    k_decayed = k_s * cute.math.exp2(interval, fastmath=True)
                    if source_row < row:
                        kk_regs[source_row] = kk_regs[source_row] + (x * k_decayed - x_decay * k_inv_s)
                    if cutlass.const_expr(with_a):
                        attn_regs[source_row] = attn_regs[source_row] + (z * k_decayed - z_decay * k_inv_s)
    for source_row in cutlass.range_constexpr(cfg.b_t):
        kk = kk_regs[source_row]
        attn = attn_regs[source_row]
        for step in cutlass.range_constexpr(3):
            distance = 4 >> step
            kk = kk + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, kk, distance, 31, kind=nvvm.Shfl.BFLY))
            if cutlass.const_expr(with_a):
                attn = attn + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, attn, distance, 31, kind=nvvm.Shfl.BFLY))
        if lane_in_row_group == 0:
            (corr_ptr + row * cfg.b_t + source_row).store(kk)
            if cutlass.const_expr(with_a):
                (corr_ptr + cfg.b_t * cfg.b_t + row * cfg.b_t + source_row).store(attn)


@cute.jit
def add_steep_corrections(cfg: cutlass.Constexpr, acc, corr_ptr, lane_idx):
    """Add a ``[BT, BT]`` fp32 correction tile to an m16n16 register-MMA accumulator fragment (8 values per lane)."""
    for accum_idx in cutlass.range_constexpr(8):
        row = lane_idx // 4
        if cutlass.const_expr(accum_idx % 4 >= 2):
            row = row + cutlass.Int32(8)
        col = (accum_idx // 4) * 8 + 2 * (lane_idx % 4)
        if cutlass.const_expr(accum_idx % 2 == 1):
            col = col + cutlass.Int32(1)
        acc[accum_idx] = acc[accum_idx] + (corr_ptr + row * cfg.b_t + col).load()
    # Callers release this stage through an elected lane. Join every reader
    # before that lane can let the producer overwrite the correction tile.
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def steep_grad_corrections(
    cfg: cutlass.Constexpr, channel, t, scale, exchange_ptr, log_last_ptr, raw_q_ptr, raw_k_ptr, beta_ptr, q_norm_ptr, k_norm_ptr, dadm_ptr
):
    """The backward's intra-tile terms of steep channel ``channel`` at token ``t`` (bprop compute group 2).

    ``dadm_ptr`` holds the chunk's strict-lower dA and dM tiles, ``[2, BT, BT]`` in the io dtype, exactly as the
    tcgen05 GEMMs read them.  With ``F(t, s) = 2^(g[t] - g[s])`` and ``K inv[s] = k_n[s] * 2^(g[BT - 1] - g[s])``
    rounded to the io dtype (the steep column of the K inv operand), forms

        dq_attn  = sum_{s<t} dA[t, s] k_n[s] F(t, s)          dq_mma  = sum_{s<t} dA[t, s] K inv[s]
        dkd_attn = sum_{s<t} dM[t, s] k_n[s] F(t, s)          dkd_mma = sum_{s<t} dM[t, s] K inv[s]
        dk_inv   = sum_{s>t} (scale q_n[s] dA[s, t] - beta[s] k_n[s] dM[s, t]) F(s, t)

    Every exponent of F is at most 0. ``dq_mma`` and ``dkd_mma`` contain products for the steep column.
    The dQ attn and dK decay GEMMs have already accumulated these products.
    Return the dQ correction, the inverse-side dK term, the beta correction and the
    decay-side dK correction (also used for dgate). Callers only map these onto their register layouts;
    they add the two dK terms in their original accumulation order."""
    dq_attn = cutlass.Float32(0.0)
    dq_mma = cutlass.Float32(0.0)
    dkd_attn = cutlass.Float32(0.0)
    dkd_mma = cutlass.Float32(0.0)
    dk_inv = cutlass.Float32(0.0)
    tile = cutlass.const_expr(cfg.b_t * cfg.b_t)
    # Grow bounded intervals directly. Never subtract cumulative prefixes: a
    # large negative spike can erase later mild gates in FP32.
    interval = cutlass.Float32(0.0)
    restore = steep_interval_log(cfg, exchange_ptr, log_last_ptr, t + 1, cfg.b_t, channel)
    for distance in cutlass.range(1, t + 1, unroll=1):
        s = t - distance
        gate = steep_token_log(cfg, exchange_ptr, log_last_ptr, s + 1, channel)
        interval = interval + gate
        restore = restore + gate
        k_s = (raw_k_ptr + raw_offset(cfg, s, channel)).load().to(cutlass.Float32)
        if cutlass.const_expr(cfg.l2norm):
            k_s = k_s * (k_norm_ptr + s).load()
        k_inv_s = (k_s * cute.math.exp2(restore, fastmath=True)).to(cfg.io_dtype).to(cutlass.Float32)
        decay = cute.math.exp2(interval, fastmath=True)
        da = (dadm_ptr + t * cfg.b_t + s).load().to(cutlass.Float32)
        dm = (dadm_ptr + tile + t * cfg.b_t + s).load().to(cutlass.Float32)
        dq_attn = dq_attn + da * (k_s * decay)
        dq_mma = dq_mma + da * k_inv_s
        dkd_attn = dkd_attn + dm * (k_s * decay)
        dkd_mma = dkd_mma + dm * k_inv_s
    interval = cutlass.Float32(0.0)
    for s in cutlass.range(t + 1, cfg.b_t, unroll=1):
        interval = interval + steep_token_log(cfg, exchange_ptr, log_last_ptr, s, channel)
        k_s = (raw_k_ptr + raw_offset(cfg, s, channel)).load().to(cutlass.Float32)
        q_s = (raw_q_ptr + raw_offset(cfg, s, channel)).load().to(cutlass.Float32)
        if cutlass.const_expr(cfg.l2norm):
            k_s = k_s * (k_norm_ptr + s).load()
            q_s = q_s * (q_norm_ptr + s).load()
        decay = cute.math.exp2(interval, fastmath=True)
        da = (dadm_ptr + s * cfg.b_t + t).load().to(cutlass.Float32)
        dm = (dadm_ptr + tile + s * cfg.b_t + t).load().to(cutlass.Float32)
        dk_inv = dk_inv + ((q_s * scale) * da - (k_s * steep_beta(cfg, beta_ptr, s, channel)) * dm) * decay
    exp_t = steep_decay_scale(cfg, exchange_ptr, log_last_ptr, t, channel)
    k_t = (raw_k_ptr + raw_offset(cfg, t, channel)).load().to(cutlass.Float32)
    if cutlass.const_expr(cfg.l2norm):
        k_t = k_t * (k_norm_ptr + t).load()
    beta_t = steep_beta(cfg, beta_ptr, t, channel)
    dq_delta = scale * (dq_attn - exp_t * dq_mma)
    dkd_delta = exp_t * dkd_mma - dkd_attn
    return dq_delta, dk_inv, k_t * dkd_delta, beta_t * dkd_delta


@cute.jit
def store_steep_beta(cfg: cutlass.Constexpr, bits, beta_regs, beta_ptr, row, lane_in_row_group):
    """Write a row's guarded beta_eff back over the raw beta tile on its steep channels, for steep_beta to read."""
    for reg_idx in cutlass.range_constexpr(cfg.d_k // 64 * 8):
        if ((bits >> reg_idx) & 1) != 0:
            channel = (reg_idx // 8) * 64 + lane_in_row_group * 8 + reg_idx % 8
            (beta_ptr + raw_offset(cfg, row, channel)).store(beta_regs[reg_idx].to(cfg.io_dtype))
