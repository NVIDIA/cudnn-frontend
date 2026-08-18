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


import logging
from types import SimpleNamespace

import cuda.tile as ct
from cuda.tile.tune import exhaustive_search

from .common import (
    autotuned_launch,
    cdiv,
    ct_min,
    dummy,
    exp,
    exp2,
    fused_beta_sigmoid,
    fused_beta_sigmoid_bwd,
    head_group_sum,
    l2norm_bwd,
    l2norm_fwd,
    launch_hint_cache,
    next_power_of_2,
    opt,
    softplus,
    sum_leading,
    tf32,
    zero_fill,
)
from cudnn.frost.buffers import current_device_id, dtype_name as dtname

logger = logging.getLogger(__name__)

ConstInt = ct.Constant[int]

RCP_LN2 = 1.4426950216  # 1/ln(2)

# chunk size (BT tile) of these kernels; the engine's carve layout imports it
BT_CHUNK = 64


# --- Host helpers ---------------------------------------------------------------------------------


def cast(bufs, name, src, ref, stream=None):
    """Dtype cast at the gradient boundary, through the ``bufs[name]`` carve."""
    if str(src.dtype).split(".")[-1] == str(ref.dtype).split(".")[-1]:
        return src
    from .common import cast_copy

    dst = bufs[name]
    n = 1
    for s_ in src.shape:
        n *= int(s_)
    cast_copy(dst.reshape((n,)), src.reshape((n,)), n, stream=0 if stream is None else stream)
    return dst


# --- Launch tuning --------------------------------------------------------------------------------


def autotuned_launch_bv(kernel, cache_key, bv_choices, grid_fn, args_fn, timeout=40, stream=None):
    """Launch ``kernel`` autotuning over BV (the V-tile block width) only.

    BV is a kernel ConstInt that drives the grid V-fan-out and the V-tile shapes
    (the K axis is always split into fixed 64-wide blocks, so BV never changes
    the numerics -- it is pure V-tiling). It therefore cannot be set via
    replace_hints; it is swept as a real config dimension. ``grid_fn(bv)`` /
    ``args_fn(bv)`` rebuild the BV-dependent grid and argument tuple. The winning
    BV is cached and the BV-specialized kernel re-launched on every subsequent
    call. ``bv_choices[0]`` is the safe fallback when tuning fails.

    Why the sweep is BV-ONLY: the state scan is register-bound (255 reg/thread
    -> ~1 block/SM) with a serial per-CTA NT inter-chunk loop, so the only
    parallelism lever is the V-tile CTA count (grid = (cdiv(V, BV), N*HV)).
    The optimum is NOT monotone in BV (a smaller BV multiplies redundant K/W/G
    reloads once the grid already fills the SMs), and the occupancy x
    num_worker_warps hint is nearly inert per BV -- do NOT cross occ x nww into
    this search (the larger sweep mis-ranks BV).
    """
    stream = 0 if stream is None else stream
    if cache_key not in launch_hint_cache:
        chosen = None
        try:
            configs = [SimpleNamespace(BV=bv) for bv in bv_choices]
            with ct.compiler_timeout(timeout):
                result = exhaustive_search(
                    configs,
                    stream,
                    lambda cfg: grid_fn(cfg.BV),
                    kernel,
                    lambda cfg: args_fn(cfg.BV),
                    lambda cfg: {},
                )
            chosen = result.best.config.BV
        except Exception:
            chosen = None
        launch_hint_cache[cache_key] = chosen

    chosen = launch_hint_cache[cache_key]
    bv = bv_choices[0] if chosen is None else chosen
    ct.launch(stream, grid_fn(bv), kernel, args_fn(bv))


# --- Device helpers -------------------------------------------------------------------------------


def mma_operands(a, b):
    # ct.mma/ct.matmul require matching operand dtypes. Forward callers always
    # pass matched dtypes (left unchanged here); some backward callers mix bf16
    # with fp32, so promote both to fp32 in that case.
    if a.dtype != b.dtype:
        a = ct.astype(a, ct.float32)
        b = ct.astype(b, ct.float32)
    return tf32(a), tf32(b)


def safe_matmul(a, b):
    # tf32 is only the multiply precision; ct.matmul returns a tf32 tile which
    # is a restricted dtype (no elementwise arithmetic). Cast the result back to
    # fp32 so downstream adds work (fp32 dot-accumulator semantics).
    aa, bb = mma_operands(a, b)
    out = ct.matmul(aa, bb)
    return ct.astype(out, ct.float32) if out.dtype != ct.float32 else out


def safe_mma(a, b, acc):
    aa, bb = mma_operands(a, b)
    return ct.mma(aa, bb, acc)


def bf16_mma(a, b, acc):
    # Like safe_mma but stages the MMA operands as bf16 (2 bytes) instead of
    # tf32 (4 bytes). cuTile stages tiny MMA operands into static SMEM; tf32
    # staging doubles that footprint vs bf16, which on the SMEM-bound
    # compute-pairs kernel pins it to 1 block/SM.
    return ct.mma(ct.astype(a, ct.bfloat16), ct.astype(b, ct.bfloat16), acc)


def reg_matmul(a, b):
    # Register-resident matmul for TINY products (e.g. [16,16]@[16,16] /
    # [16,16]@[16,K]). ct.matmul/ct.mma lower even tiny tf32 dots through an
    # SMEM-staged HMMA path (~83KB static smem/block -> 1 block/SM, ~12% occ).
    # Keep these dots in registers instead, via a
    # broadcast-multiply-reduce over the contraction dim: out[i,j] =
    # sum_k a[i,k]*b[k,j]. Inputs are rounded to tf32 first.
    aa = tf32(a)
    bb = tf32(b)
    aa = ct.astype(aa, ct.float32)
    bb = ct.astype(bb, ct.float32)
    return ct.sum(aa[:, :, None] * bb[None, :, :], axis=1)


def load_bc_bk(arr, base, row0, col0, stride_row, BC: ConstInt, BK: ConstInt, T_eff, K: ConstInt):
    # Boundary-checked (BC,BK) block load at (row0,col0) of the (T,K) view
    # (row stride stride_row), cast to fp32, on the flattened 1-D view.
    o_r = ct.arange(BC, dtype=ct.int32)
    o_c = ct.arange(BK, dtype=ct.int32)
    rows = row0 + o_r
    cols = col0 + o_c
    off = base + rows[:, None] * stride_row + cols[None, :]
    mask = (rows < T_eff)[:, None] & (cols < K)[None, :]
    return ct.astype(ct.gather(arr, off, mask=mask, check_bounds=False, padding_value=0.0), ct.float32)


def store_bc_bc(arr, base, row0, col0, stride_row, blk, BC: ConstInt, BT_or_BC: ConstInt, T_eff):
    o_r = ct.arange(BC, dtype=ct.int32)
    o_c = ct.arange(BC, dtype=ct.int32)
    rows = row0 + o_r
    cols = col0 + o_c
    off = base + rows[:, None] * stride_row + cols[None, :]
    mask = (rows < T_eff)[:, None] & (cols < BT_or_BC)[None, :]
    ct.scatter(arr, off, ct.astype(blk, arr.dtype), mask=mask, check_bounds=False)


# --- Kernels: normalization and gates -------------------------------------------------------------


@ct.kernel
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    H: ConstInt,
    S: ConstInt,
    BT: ConstInt,
    BS: ConstInt,
    REVERSE: ConstInt,
    HAS_SCALE: ConstInt,
):
    i_s = ct.bid(0)
    i_t = ct.bid(1)
    i_h = ct.bid(2)

    i_n = ct.load(chunk_indices, (i_t * 2,), shape=()).item()
    i_t = ct.load(chunk_indices, (i_t * 2 + 1,), shape=()).item()
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T_eff = eos - bos

    o_t = ct.arange(BT, dtype=ct.int32)
    o_s = ct.arange(BS, dtype=ct.int32)
    row = i_t * BT + o_t
    col = i_s * BS + o_s
    m_t = row < T_eff
    m_s = col < S
    m = m_t[:, None] & m_s[None, :]

    base = (bos * H + i_h) * S
    off = base + row[:, None] * (H * S) + col[None, :]

    b_s = ct.astype(ct.gather(s, off, mask=m, check_bounds=False, padding_value=0.0), ct.float32)
    b_o = ct.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = ct.sum(b_s, axis=0)
        b_o = -b_o + b_z[None, :] + b_s
    if HAS_SCALE:
        b_o = b_o * scale
    ct.scatter(o, off, ct.astype(b_o, o.dtype), mask=m, check_bounds=False)


@ct.kernel
def kda_gate_chunk_cumsum_vector_kernel(
    s,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    lower_bound,
    H: ConstInt,
    S: ConstInt,
    BT: ConstInt,
    BS: ConstInt,
    REVERSE: ConstInt,
    HAS_BIAS: ConstInt,
    HAS_SCALE: ConstInt,
    USE_LOWER_BOUND: ConstInt,
):
    i_s = ct.bid(0)
    i_t = ct.bid(1)
    i_h = ct.bid(2)

    i_n = ct.load(chunk_indices, (i_t * 2,), shape=()).item()
    i_t = ct.load(chunk_indices, (i_t * 2 + 1,), shape=()).item()
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T_eff = eos - bos

    o_t = ct.arange(BT, dtype=ct.int32)
    o_s = ct.arange(BS, dtype=ct.int32)
    row = i_t * BT + o_t
    col = i_s * BS + o_s
    m_t = row < T_eff
    m_s = col < S
    m = m_t[:, None] & m_s[None, :]

    base = (bos * H + i_h) * S
    off = base + row[:, None] * (H * S) + col[None, :]
    b_s = ct.astype(ct.gather(s, off, mask=m, check_bounds=False, padding_value=0.0), ct.float32)

    if HAS_BIAS:
        bias_off = i_h * S + col
        b_bias = ct.astype(ct.gather(dt_bias, bias_off, mask=m_s, check_bounds=False, padding_value=0.0), ct.float32)
        b_s = b_s + b_bias[None, :]

    b_A = ct.astype(ct.load(A_log, (i_h,), shape=()).item(), ct.float32)
    if USE_LOWER_BOUND:
        b_gate = lower_bound * (1.0 / (1.0 + ct.exp(-(exp(b_A) * b_s))))
    else:
        b_gate = -exp(b_A) * softplus(b_s)

    b_o = ct.cumsum(b_gate, axis=0)
    if REVERSE:
        b_z = ct.sum(b_gate, axis=0)
        b_o = -b_o + b_z[None, :] + b_gate
    if HAS_SCALE:
        b_o = b_o * scale
    ct.scatter(o, off, ct.astype(b_o, o.dtype), mask=m, check_bounds=False)


@ct.kernel
def kda_gate_bwd_kernel(
    g,
    A_log,
    dt_bias,
    dyg,
    dg,
    dA,
    db,
    lower_bound,
    T,
    H: ConstInt,
    D: ConstInt,
    BT: ConstInt,
    BD: ConstInt,
    HAS_BIAS: ConstInt,
    USE_LOWER_BOUND: ConstInt,
):
    # G/dYg/dG layout [T, H, D] -> view (T, D) stride (H*D, 1) at base i_h*D.
    i_t = ct.bid(0)
    i_h = ct.bid(1)

    b_A = ct.astype(ct.load(A_log, (i_h,), shape=()).item(), ct.float32)

    o_t = ct.arange(BT, dtype=ct.int32)
    o_d = ct.arange(BD, dtype=ct.int32)
    row = i_t * BT + o_t
    m_t = row < T
    m_d = o_d < D
    m = m_t[:, None] & m_d[None, :]
    base = i_h * D
    off = base + row[:, None] * (H * D) + o_d[None, :]

    b_g = ct.astype(ct.gather(g, off, mask=m, check_bounds=False, padding_value=0.0), ct.float32)
    b_dyg = ct.astype(ct.gather(dyg, off, mask=m, check_bounds=False, padding_value=0.0), ct.float32)
    if HAS_BIAS:
        bias_off = i_h * D + o_d
        b_bias = ct.astype(ct.gather(dt_bias, bias_off, mask=m_d, check_bounds=False, padding_value=0.0), ct.float32)
        b_g = b_g + b_bias[None, :]

    if USE_LOWER_BOUND:
        b_eA = exp(b_A)
        b_inner = b_eA * b_g
        b_sig = 1.0 / (1.0 + ct.exp(-b_inner))
        b_dsig = b_sig * (1.0 - b_sig)
        b_d_inner_term = b_dyg * (lower_bound * b_dsig)
        b_dg = b_d_inner_term * b_eA
        b_dA = ct.sum(ct.sum(b_dg * b_g, axis=1), axis=0)
    else:
        b_negeA = -exp(b_A)
        b_yg = b_negeA * softplus(b_g)
        b_sig = 1.0 / (1.0 + ct.exp(-b_g))
        b_dg = b_negeA * (b_dyg * b_sig)
        b_dA = ct.sum(ct.sum(b_dyg * b_yg, axis=1), axis=0)

    ct.scatter(dg, off, ct.astype(b_dg, dg.dtype), mask=m, check_bounds=False)
    ct.scatter(dA, i_t * H + i_h, b_dA)
    if HAS_BIAS:
        # b_dg is zero on masked lanes (b_dyg gathers with padding_value=0)
        b_db = ct.sum(b_dg, axis=0)
        ct.scatter(db, (i_t * H + i_h) * D + o_d, b_db, mask=m_d, check_bounds=False)


# --- Kernels: WY representation -------------------------------------------------------------------


@ct.kernel
def recompute_w_u_fwd_kda_kernel(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    STORE_U: ConstInt,
    STORE_QG: ConstInt,
    STORE_KG: ConstInt,
):
    # Arrays arrive pre-flattened to 1-D from the host (cuTile cannot reshape a
    # rank-4 dynamic array in-kernel). Use flat element offsets row*stride+col.
    k_flat = k
    q_flat = q
    qg_flat = qg
    kg_flat = kg
    w_flat = w
    gk_flat = gk
    v_flat = v
    u_flat = u
    beta_flat = beta
    A_flat = A

    i_t = ct.bid(0)
    i_hv = ct.bid(1)
    i_h = i_hv // (HV // H)

    i_n = ct.load(chunk_indices, (i_t * 2,), shape=()).item()
    i_t_loc = ct.load(chunk_indices, (i_t * 2 + 1,), shape=()).item()
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    Tloc = eos - bos

    t_off = i_t_loc * BT + ct.arange(BT, dtype=ct.int32)
    m_t = t_off < Tloc

    b_idx = (bos + t_off) * HV + i_hv
    b_b = ct.astype(ct.gather(beta_flat, b_idx, mask=m_t, check_bounds=False, padding_value=0.0), ct.float32)

    a_rows = ((bos + t_off) * HV + i_hv)[:, None]
    a_cols = ct.arange(BT, dtype=ct.int32)[None, :]
    a_off = ct.broadcast_to(a_rows, (BT, BT)) * BT + ct.broadcast_to(a_cols, (BT, BT))
    b_A = ct.gather(
        A_flat,
        a_off,
        mask=ct.broadcast_to(m_t[:, None], (BT, BT)),
        check_bounds=False,
        padding_value=0.0,
    )

    if STORE_U:
        v_off = ct.arange(BV, dtype=ct.int32)
        for i_v in range(ct.cdiv(V, BV)):
            vcols = (i_v * BV + v_off)[None, :]
            vrows = ((bos + t_off) * HV + i_hv)[:, None]
            m_v = m_t[:, None] & ((i_v * BV + v_off) < V)[None, :]
            v_offset = ct.broadcast_to(vrows, (BT, BV)) * V + ct.broadcast_to(vcols, (BT, BV))
            b_v = ct.gather(
                v_flat,
                v_offset,
                mask=m_v,
                check_bounds=False,
                padding_value=0.0,
            )
            b_vb = ct.astype(ct.astype(b_v, ct.float32) * b_b[:, None], b_v.dtype)
            b_u = safe_matmul(b_A, b_vb)
            ct.scatter(
                u_flat,
                v_offset,
                ct.astype(b_u, u_flat.dtype),
                mask=m_v,
                check_bounds=False,
            )

    last_idx = ct_min(i_t_loc * BT + BT, Tloc) - 1
    k_off = ct.arange(BK, dtype=ct.int32)
    for i_k in range(ct.cdiv(K, BK)):
        kcols = (i_k * BK + k_off)[None, :]
        m_kcol = ((i_k * BK + k_off) < K)[None, :]
        m_k = m_t[:, None] & m_kcol
        krows = ((bos + t_off) * H + i_h)[:, None]
        gkrows = ((bos + t_off) * HV + i_hv)[:, None]
        bk_col = ct.broadcast_to(kcols, (BT, BK))
        k_offset = ct.broadcast_to(krows, (BT, BK)) * K + bk_col
        gk_offset = ct.broadcast_to(gkrows, (BT, BK)) * K + bk_col
        b_k = ct.gather(
            k_flat,
            k_offset,
            mask=m_k,
            check_bounds=False,
            padding_value=0.0,
        )
        b_gk = ct.astype(
            ct.gather(
                gk_flat,
                gk_offset,
                mask=m_k,
                check_bounds=False,
                padding_value=0.0,
            ),
            ct.float32,
        )
        b_egk = exp2(b_gk)
        b_kb = ct.astype(b_k, ct.float32) * b_b[:, None] * b_egk

        if STORE_QG:
            qrows = ((bos + t_off) * H + i_h)[:, None]
            q_offset = ct.broadcast_to(qrows, (BT, BK)) * K + bk_col
            b_q = ct.gather(
                q_flat,
                q_offset,
                mask=m_k,
                check_bounds=False,
                padding_value=0.0,
            )
            b_qg = ct.astype(b_q, ct.float32) * b_egk
            qgrows = ((bos + t_off) * HV + i_hv)[:, None]
            qg_offset = ct.broadcast_to(qgrows, (BT, BK)) * K + bk_col
            ct.scatter(
                qg_flat,
                qg_offset,
                ct.astype(b_qg, qg_flat.dtype),
                mask=m_k,
                check_bounds=False,
            )
        if STORE_KG:
            gn_rows = ct.broadcast_to(((bos + last_idx) * HV + i_hv), (1, BK))
            gn_off = gn_rows * K + kcols
            b_gn = ct.astype(
                ct.gather(
                    gk_flat,
                    gn_off,
                    mask=m_kcol,
                    check_bounds=False,
                    padding_value=0.0,
                ),
                ct.float32,
            )
            decay = ct.where(m_t[:, None], exp2(b_gn - b_gk), ct.zeros((BT, BK), dtype=ct.float32))
            b_kg = ct.astype(b_k, ct.float32) * decay
            kgrows = ((bos + t_off) * HV + i_hv)[:, None]
            kg_offset = ct.broadcast_to(kgrows, (BT, BK)) * K + bk_col
            ct.scatter(
                kg_flat,
                kg_offset,
                ct.astype(b_kg, kg_flat.dtype),
                mask=m_k,
                check_bounds=False,
            )

        b_w = safe_matmul(b_A, ct.astype(b_kb, b_k.dtype))
        wrows = ((bos + t_off) * HV + i_hv)[:, None]
        w_offset = ct.broadcast_to(wrows, (BT, BK)) * K + bk_col
        ct.scatter(
            w_flat,
            w_offset,
            ct.astype(b_w, w_flat.dtype),
            mask=m_k,
            check_bounds=False,
        )


@ct.kernel
def chunk_kda_fwd_kernel_intra_token_parallel(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    N,
    T,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BC: ConstInt,
    BH: ConstInt,
    BK: ConstInt,
):
    # BK = next_power_of_2(K) is passed by host
    # (cuTile tile shapes must be compile-time constants).
    i_tg = ct.bid(0)
    i_hg = ct.bid(1)

    bos = (i_tg // T) * T
    i_t = i_tg % T
    T_eff = T
    left = 0
    right = N
    # Unrolled binary search to find i_n s.t. cu[i_n] <= i_tg < cu[i_n+1]
    for _ in range(20):
        if left < right:
            mid = (left + right) // 2
            cmid = ct.load(cu_seqlens, (mid + 1,), shape=()).item()
            if i_tg < cmid:
                right = mid
            else:
                left = mid + 1
    i_n = left
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T_eff = eos - bos
    i_t = i_tg - bos

    if i_t >= T_eff:
        return

    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC

    G = HV // H

    Aqk_base = bos * HV * BT
    Akk_base = bos * HV * BC
    beta_base = bos * HV

    # cuTile gather/scatter require the index tuple rank to match the array rank
    # (no raw pointer arithmetic). Arrays arrive pre-flattened from the
    # host: Q/K -> (B*T*H, K), G -> (B*T*HV, K), Beta/Aqk/Akk -> 1-D.
    o_hv = i_hg * BH + ct.arange(BH, dtype=ct.int32)
    o_h = o_hv // G
    o_k = ct.arange(BK, dtype=ct.int32)
    m_hv = o_hv < HV
    m_k = o_k < K
    m_hk = m_hv[:, None] & m_k[None, :]

    col = ct.broadcast_to(o_k[None, :], (BH, BK))

    # Q/K: row = (bos + token) * H + head; col = key
    qk_row = ct.broadcast_to(((bos + i_t) * H + o_h)[:, None], (BH, BK))
    b_q = ct.astype(ct.gather(q, (qk_row, col), mask=m_hk, check_bounds=False, padding_value=0.0), ct.float32)
    b_k = ct.astype(ct.gather(k, (qk_row, col), mask=m_hk, check_bounds=False, padding_value=0.0), ct.float32)

    # G: row = (bos + token) * HV + head; Beta: idx = (bos + token) * HV + head
    g_row = ct.broadcast_to(((bos + i_t) * HV + o_hv)[:, None], (BH, BK))
    b_g = ct.astype(ct.gather(g, (g_row, col), mask=m_hk, check_bounds=False, padding_value=0.0), ct.float32)
    b_beta = ct.astype(ct.gather(beta, beta_base + i_t * HV + o_hv, mask=m_hv, check_bounds=False, padding_value=0.0), ct.float32)
    b_k = b_k * b_beta[:, None]

    # Counted loop over the (static) BC-wide sub-chunk window. A runtime-bounded
    # `for j in range(i_ts, j_hi)` lowers to per-iteration branches that tileiras
    # cannot unroll/predicate; a counted `for jj in range(BC)` with a runtime
    # guard derives `j` from `jj` and stays fully unrolled.
    for jj in range(BC):
        j = i_ts + jj
        if j < i_t + 1 and j < T_eff and j < i_ts + BC:
            kj_row = ct.broadcast_to(((bos + j) * H + o_h)[:, None], (BH, BK))
            gj_row = ct.broadcast_to(((bos + j) * HV + o_hv)[:, None], (BH, BK))
            b_kj = ct.astype(ct.gather(k, (kj_row, col), mask=m_hk, check_bounds=False, padding_value=0.0), ct.float32)
            b_gj = ct.astype(ct.gather(g, (gj_row, col), mask=m_hk, check_bounds=False, padding_value=0.0), ct.float32)

            b_kgj = ct.where(m_k[None, :], b_kj * exp2(b_g - b_gj), ct.zeros((BH, BK), dtype=ct.float32))
            b_Aqk = ct.sum(b_q * b_kgj, axis=1) * scale
            b_Akk = ct.sum(b_k * b_kgj, axis=1) * (1.0 if j < i_t else 0.0)

            ct.scatter(
                Aqk,
                Aqk_base + i_t * HV * BT + o_hv * BT + (j % BT),
                ct.astype(b_Aqk, Aqk.dtype),
                mask=m_hv,
                check_bounds=False,
            )
            ct.scatter(
                Akk,
                Akk_base + i_t * HV * BC + o_hv * BC + (j - i_ts),
                ct.astype(b_Akk, Akk.dtype),
                mask=m_hv,
                check_bounds=False,
            )


@ct.kernel
def chunk_kda_fwd_kernel_inter_diag_compute_solve(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BC: ConstInt,
    BK: ConstInt,
):
    i_t = ct.bid(0)
    i_i = ct.bid(1)
    i_hv = ct.bid(2)
    i_h = i_hv // (HV // H)

    i_n = ct.load(chunk_indices, (i_t * 2,), shape=()).item()
    i_t = ct.load(chunk_indices, (i_t * 2 + 1,), shape=()).item()
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T_eff = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T_eff:
        return

    o_bc = ct.arange(BC, dtype=ct.int32)
    o_bk = ct.arange(BK, dtype=ct.int32)
    o_c = i_ti + o_bc
    m_c = o_c < T_eff
    m_k = o_bk < K

    q_base = (bos * H + i_h) * K
    k_base = (bos * H + i_h) * K
    g_base = (bos * HV + i_hv) * K
    beta_base = bos * HV + i_hv
    Aqk_base = (bos * HV + i_hv) * BT
    Akk_base = (bos * HV + i_hv) * BC

    qk_rows = o_c[:, None]
    qk_cols = o_bk[None, :]
    m_qk = m_c[:, None] & m_k[None, :]
    q_off = q_base + qk_rows * (H * K) + qk_cols
    k_off = k_base + qk_rows * (H * K) + qk_cols
    g_off = g_base + qk_rows * (HV * K) + qk_cols
    b_q = ct.gather(q, q_off, mask=m_qk, check_bounds=False, padding_value=0.0)
    b_k = ct.gather(k, k_off, mask=m_qk, check_bounds=False, padding_value=0.0)
    b_g = ct.gather(g, g_off, mask=m_qk, check_bounds=False, padding_value=0.0)
    b_beta = ct.gather(beta, beta_base + o_c * HV, mask=m_c, check_bounds=False, padding_value=0.0)

    gn_row = i_ti + ct_min(BC // 2, T_eff - i_ti - 1)
    b_gn = ct.gather(g, g_base + gn_row * (HV * K) + o_bk, mask=m_k, check_bounds=False, padding_value=0.0)
    b_gn = ct.astype(b_gn, ct.float32)[None, :]

    b_gm = ct.astype(b_g, ct.float32) - b_gn
    b_gq = ct.where(m_c[:, None], exp2(b_gm), ct.zeros((BC, BK), dtype=ct.float32))
    b_gk = ct.where(m_c[:, None], exp2(-b_gm), ct.zeros((BC, BK), dtype=ct.float32))

    if K < 256:
        b_kgt = ct.transpose(ct.astype(ct.astype(b_k, ct.float32) * b_gk, b_k.dtype))
        b_qg = ct.astype(ct.astype(b_q, ct.float32) * b_gq, b_q.dtype)
        b_kg = ct.astype(ct.astype(b_k, ct.float32) * b_gq, b_k.dtype)
    else:
        b_kgt = ct.transpose(ct.astype(b_k, ct.float32) * b_gk)
        b_qg = ct.astype(b_q, ct.float32) * b_gq
        b_kg = ct.astype(b_k, ct.float32) * b_gq

    b_Aqk = safe_matmul(b_qg, b_kgt) * scale
    b_Akk = safe_matmul(b_kg, b_kgt) * ct.astype(b_beta, ct.float32)[:, None]

    o_i = o_bc
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = ct.where(m_Aqk, b_Aqk, ct.zeros((BC, BC), dtype=ct.float32))
    b_Akk = ct.where(m_Akk, b_Akk, ct.zeros((BC, BC), dtype=ct.float32))

    # store Aqk (Akk for this kernel writes to the fp32 diagonal buffer)
    aqk_rows = o_c[:, None]
    aqk_cols = (i_i * BC + o_bc)[None, :]
    aqk_off = Aqk_base + aqk_rows * (HV * BT) + aqk_cols
    m_aqk = m_c[:, None] & ((i_i * BC + o_bc) < BT)[None, :]
    ct.scatter(Aqk, aqk_off, ct.astype(b_Aqk, Aqk.dtype), mask=m_aqk, check_bounds=False)

    # diagonal Akk -> inverse via Neumann series by squaring, written to Akk(diag buf).
    #
    # Akk is strictly-lower-triangular and nilpotent (Akk^BC = 0); with
    # N := -Akk, (I + Akk)^-1 = (I - N)^-1 = sum_{k=0}^{BC-1} N^k
    # = prod_{j=0}^{log2(BC)-1} (I + N^(2^j)) -- log2(BC) block matmuls, no
    # serial row dependency. Rows beyond T_eff have N = 0 (b_gq/b_gk were
    # zeroed by m_c) and converge to the identity.
    #
    # Precision: the squarings run at tf32 via safe_matmul (fp32 operands
    # would fall back to SIMT even at M=32).
    b_N = -b_Akk  # N := -Akk, so (I + Akk)^-1 = (I - N)^-1 = sum_k N^k
    b_Ai = ct.astype(m_I, ct.float32) + b_N  # (I + N)
    # Squaring stages: `range(2, BC)` traces as a compile-time-bounded loop with
    # concrete Python `i` during unroll (BC is a ConstInt). We do one squaring
    # each time `i` is a power of two (i = 2, 4, 8, 16 for BC=32), giving exactly
    # ceil(log2(BC)) - 1 stages (factors N^2, N^4, N^8, N^16). All the branch
    # decisions are host-side (Python int `i`), so nothing lowers to a device
    # branch -- the body is fully unrolled into log2(BC) block matmuls.
    for i in range(2, BC):
        if (i & (i - 1)) == 0:  # i is a power of two (host-time test)
            b_N = safe_matmul(b_N, b_N)  # N -> N^2 -> N^4 ...
            b_Ai = b_Ai + safe_matmul(b_Ai, b_N)  # (prod so far) @ (I + N^(2^j))

    akk_rows = o_c[:, None]
    akk_cols = o_bc[None, :]
    akk_off = Akk_base + akk_rows * (HV * BC) + akk_cols
    m_akk = m_c[:, None] & (o_bc < BC)[None, :]
    ct.scatter(Akk, akk_off, ct.astype(b_Ai, Akk.dtype), mask=m_akk, check_bounds=False)


@ct.kernel
def chunk_kda_fwd_kernel_inter_solve_fused(
    q,
    k,
    g,
    beta,
    Aqk,
    Akkd,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BC: ConstInt,
    NC: ConstInt,
    BK: ConstInt,
    USE_SAFE_GATE: ConstInt,
):
    i_t = ct.bid(0)
    i_hv = ct.bid(1)
    i_h = i_hv // (HV // H)

    i_n = ct.load(chunk_indices, (i_t * 2,), shape=()).item()
    i_t = ct.load(chunk_indices, (i_t * 2 + 1,), shape=()).item()
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T_eff = eos - bos

    if i_t * BT >= T_eff:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q_base = (bos * H + i_h) * K
    k_base = (bos * H + i_h) * K
    g_base = (bos * HV + i_hv) * K
    Aqk_base = (bos * HV + i_hv) * BT
    Akk_base = (bos * HV + i_hv) * BT
    Akkd_base = (bos * HV + i_hv) * BC
    beta_base = bos * HV + i_hv

    o_i = ct.arange(BC, dtype=ct.int32)
    o_k = ct.arange(BK, dtype=ct.int32)
    m_tc1 = (i_tc1 + o_i) < T_eff
    m_tc2 = (i_tc2 + o_i) < T_eff
    m_tc3 = (i_tc3 + o_i) < T_eff

    z = ct.zeros((BC, BC), dtype=ct.float32)
    b_Aqk10 = z
    b_Akk10 = z
    b_Aqk20 = z
    b_Akk20 = z
    b_Aqk21 = z
    b_Akk21 = z
    b_Aqk30 = z
    b_Akk30 = z
    b_Aqk31 = z
    b_Akk31 = z
    b_Aqk32 = z
    b_Akk32 = z

    # ---- off-diagonal blocks -----------------------------------------------------
    num_k = (K + BK - 1) // BK
    for i_k in range(num_k):
        kk = i_k * BK + o_k
        m_k = kk < K
        b_k0 = load_bc_bk(k, k_base, i_tc0, i_k * BK, H * K, BC, BK, T_eff, K)
        b_g0 = load_bc_bk(g, g_base, i_tc0, i_k * BK, HV * K, BC, BK, T_eff, K)

        if i_tc1 < T_eff:
            b_q1 = load_bc_bk(q, q_base, i_tc1, i_k * BK, H * K, BC, BK, T_eff, K)
            b_k1 = load_bc_bk(k, k_base, i_tc1, i_k * BK, H * K, BC, BK, T_eff, K)
            b_g1 = load_bc_bk(g, g_base, i_tc1, i_k * BK, HV * K, BC, BK, T_eff, K)
            b_gn1 = ct.astype(
                ct.gather(g, g_base + i_tc1 * (HV * K) + kk, mask=m_k, check_bounds=False, padding_value=0.0),
                ct.float32,
            )
            b_gqn = ct.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), ct.zeros((BC, BK), dtype=ct.float32))
            b_kgt = ct.transpose(b_k0 * exp2(b_gn1[None, :] - b_g0))
            b_Aqk10 = safe_mma(b_q1 * b_gqn, b_kgt, b_Aqk10)
            b_Akk10 = safe_mma(b_k1 * b_gqn, b_kgt, b_Akk10)

            if NC >= 3 and i_tc2 < T_eff:
                b_q2 = load_bc_bk(q, q_base, i_tc2, i_k * BK, H * K, BC, BK, T_eff, K)
                b_k2 = load_bc_bk(k, k_base, i_tc2, i_k * BK, H * K, BC, BK, T_eff, K)
                b_g2 = load_bc_bk(g, g_base, i_tc2, i_k * BK, HV * K, BC, BK, T_eff, K)
                b_gn2 = ct.astype(
                    ct.gather(g, g_base + i_tc2 * (HV * K) + kk, mask=m_k, check_bounds=False, padding_value=0.0),
                    ct.float32,
                )
                b_gqn2 = ct.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), ct.zeros((BC, BK), dtype=ct.float32))
                b_qg2 = b_q2 * b_gqn2
                b_kg2 = b_k2 * b_gqn2
                b_kgt = ct.transpose(b_k0 * exp2(b_gn2[None, :] - b_g0))
                b_Aqk20 = safe_mma(b_qg2, b_kgt, b_Aqk20)
                b_Akk20 = safe_mma(b_kg2, b_kgt, b_Akk20)
                b_kgt = ct.transpose(b_k1 * exp2(b_gn2[None, :] - b_g1))
                b_Aqk21 = safe_mma(b_qg2, b_kgt, b_Aqk21)
                b_Akk21 = safe_mma(b_kg2, b_kgt, b_Akk21)

                if NC >= 4 and i_tc3 < T_eff:
                    b_q3 = load_bc_bk(q, q_base, i_tc3, i_k * BK, H * K, BC, BK, T_eff, K)
                    b_k3 = load_bc_bk(k, k_base, i_tc3, i_k * BK, H * K, BC, BK, T_eff, K)
                    b_g3 = load_bc_bk(g, g_base, i_tc3, i_k * BK, HV * K, BC, BK, T_eff, K)
                    b_gn3 = ct.astype(
                        ct.gather(g, g_base + i_tc3 * (HV * K) + kk, mask=m_k, check_bounds=False, padding_value=0.0),
                        ct.float32,
                    )
                    b_gqn3 = ct.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), ct.zeros((BC, BK), dtype=ct.float32))
                    b_qg3 = b_q3 * b_gqn3
                    b_kg3 = b_k3 * b_gqn3
                    b_kgt = ct.transpose(b_k0 * exp2(b_gn3[None, :] - b_g0))
                    b_Aqk30 = safe_mma(b_qg3, b_kgt, b_Aqk30)
                    b_Akk30 = safe_mma(b_kg3, b_kgt, b_Akk30)
                    b_kgt = ct.transpose(b_k1 * exp2(b_gn3[None, :] - b_g1))
                    b_Aqk31 = safe_mma(b_qg3, b_kgt, b_Aqk31)
                    b_Akk31 = safe_mma(b_kg3, b_kgt, b_Akk31)
                    b_kgt = ct.transpose(b_k2 * exp2(b_gn3[None, :] - b_g2))
                    b_Aqk32 = safe_mma(b_qg3, b_kgt, b_Aqk32)
                    b_Akk32 = safe_mma(b_kg3, b_kgt, b_Akk32)

    # ---- save off-diagonal Aqk blocks, scale Akk by Beta -------------------------
    if i_tc1 < T_eff:
        store_bc_bc(Aqk, Aqk_base, i_tc1, 0, HV * BT, b_Aqk10 * scale, BC, BT, T_eff)
        b_b1 = ct.astype(
            ct.gather(beta, beta_base + (i_tc1 + o_i) * HV, mask=m_tc1, check_bounds=False, padding_value=0.0),
            ct.float32,
        )
        b_Akk10 = b_Akk10 * b_b1[:, None]
    if NC >= 3 and i_tc2 < T_eff:
        store_bc_bc(Aqk, Aqk_base, i_tc2, 0, HV * BT, b_Aqk20 * scale, BC, BT, T_eff)
        store_bc_bc(Aqk, Aqk_base, i_tc2, BC, HV * BT, b_Aqk21 * scale, BC, BT, T_eff)
        b_b2 = ct.astype(
            ct.gather(beta, beta_base + (i_tc2 + o_i) * HV, mask=m_tc2, check_bounds=False, padding_value=0.0),
            ct.float32,
        )
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if NC >= 4 and i_tc3 < T_eff:
        store_bc_bc(Aqk, Aqk_base, i_tc3, 0, HV * BT, b_Aqk30 * scale, BC, BT, T_eff)
        store_bc_bc(Aqk, Aqk_base, i_tc3, BC, HV * BT, b_Aqk31 * scale, BC, BT, T_eff)
        store_bc_bc(Aqk, Aqk_base, i_tc3, 2 * BC, HV * BT, b_Aqk32 * scale, BC, BT, T_eff)
        b_b3 = ct.astype(
            ct.gather(beta, beta_base + (i_tc3 + o_i) * HV, mask=m_tc3, check_bounds=False, padding_value=0.0),
            ct.float32,
        )
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    # ---- load diagonal inverse blocks from Akkd (fp32) ---------------------------
    b_Ai00 = load_bc_bk(Akkd, Akkd_base, i_tc0, 0, HV * BC, BC, BC, T_eff, BC)
    b_Ai11 = load_bc_bk(Akkd, Akkd_base, i_tc1, 0, HV * BC, BC, BC, T_eff, BC)
    b_Ai22 = load_bc_bk(Akkd, Akkd_base, i_tc2, 0, HV * BC, BC, BC, T_eff, BC) if NC >= 3 else z
    b_Ai33 = load_bc_bk(Akkd, Akkd_base, i_tc3, 0, HV * BC, BC, BC, T_eff, BC) if NC >= 4 else z

    # ---- forward substitution on diagonals (only when gate not pre-solved) -------
    if not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]
        b_Ai00 = -ct.where(m_A, b_Ai00, z)
        b_Ai11 = -ct.where(m_A, b_Ai11, z)
        if NC >= 3:
            b_Ai22 = -ct.where(m_A, b_Ai22, z)
        if NC >= 4:
            b_Ai33 = -ct.where(m_A, b_Ai33, z)

        # Counted loops over the static [BC] forward-substitution window with a
        # runtime guard (i < T_eff - i_tc0). A runtime upper bound (ct_min(...,
        # T_eff - i_tc0)) lowers to per-iteration branches tileiras can't
        # unroll/predicate; the counted form stays fully unrolled.
        for i in range(2, BC):
            if i < T_eff - i_tc0:
                b_a00 = -ct.astype(
                    ct.gather(Akkd, Akkd_base + (i_tc0 + i) * (HV * BC) + o_i, check_bounds=False, padding_value=0.0),
                    ct.float32,
                )
                b_a00 = ct.where(o_i < i, b_a00, ct.zeros((BC,), dtype=ct.float32))
                b_a00 = b_a00 + ct.sum(b_a00[:, None] * b_Ai00, axis=0)
                b_Ai00 = ct.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, 2 * BC):
            if i < T_eff - i_tc0:
                b_a11 = -ct.astype(
                    ct.gather(Akkd, Akkd_base + (i_tc0 + i) * (HV * BC) + o_i, check_bounds=False, padding_value=0.0),
                    ct.float32,
                )
                b_a11 = ct.where(o_i < i - BC, b_a11, ct.zeros((BC,), dtype=ct.float32))
                b_a11 = b_a11 + ct.sum(b_a11[:, None] * b_Ai11, axis=0)
                b_Ai11 = ct.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        if NC >= 3:
            for i in range(2 * BC + 2, 3 * BC):
                if i < T_eff - i_tc0:
                    b_a22 = -ct.astype(
                        ct.gather(Akkd, Akkd_base + (i_tc0 + i) * (HV * BC) + o_i, check_bounds=False, padding_value=0.0),
                        ct.float32,
                    )
                    b_a22 = ct.where(o_i < i - 2 * BC, b_a22, ct.zeros((BC,), dtype=ct.float32))
                    b_a22 = b_a22 + ct.sum(b_a22[:, None] * b_Ai22, axis=0)
                    b_Ai22 = ct.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
        if NC >= 4:
            for i in range(3 * BC + 2, 4 * BC):
                if i < T_eff - i_tc0:
                    b_a33 = -ct.astype(
                        ct.gather(Akkd, Akkd_base + (i_tc0 + i) * (HV * BC) + o_i, check_bounds=False, padding_value=0.0),
                        ct.float32,
                    )
                    b_a33 = ct.where(o_i < i - 3 * BC, b_a33, ct.zeros((BC,), dtype=ct.float32))
                    b_a33 = b_a33 + ct.sum(b_a33[:, None] * b_Ai33, axis=0)
                    b_Ai33 = ct.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

        b_Ai00 = b_Ai00 + ct.astype(m_I, ct.float32)
        b_Ai11 = b_Ai11 + ct.astype(m_I, ct.float32)
        if NC >= 3:
            b_Ai22 = b_Ai22 + ct.astype(m_I, ct.float32)
        if NC >= 4:
            b_Ai33 = b_Ai33 + ct.astype(m_I, ct.float32)

    # ---- merged inverse using off-diagonals (tf32) -------------------------------
    b_Ai10 = -safe_matmul(safe_matmul(b_Ai11, b_Akk10), b_Ai00)
    b_Ai20 = z
    b_Ai21 = z
    b_Ai30 = z
    b_Ai31 = z
    b_Ai32 = z
    if NC >= 3:
        b_Ai21 = -safe_matmul(safe_matmul(b_Ai22, b_Akk21), b_Ai11)
        b_Ai20 = -safe_matmul(b_Ai22, safe_matmul(b_Akk20, b_Ai00) + safe_matmul(b_Akk21, b_Ai10))
    if NC >= 4:
        b_Ai32 = -safe_matmul(safe_matmul(b_Ai33, b_Akk32), b_Ai22)
        b_Ai31 = -safe_matmul(b_Ai33, safe_matmul(b_Akk31, b_Ai11) + safe_matmul(b_Akk32, b_Ai21))
        b_Ai30 = -safe_matmul(b_Ai33, safe_matmul(b_Akk30, b_Ai00) + safe_matmul(b_Akk31, b_Ai10) + safe_matmul(b_Akk32, b_Ai20))

    # ---- store full Akk_inv to Akk -----------------------------------------------
    store_bc_bc(Akk, Akk_base, i_tc0, 0, HV * BT, b_Ai00, BC, BT, T_eff)
    store_bc_bc(Akk, Akk_base, i_tc1, 0, HV * BT, b_Ai10, BC, BT, T_eff)
    store_bc_bc(Akk, Akk_base, i_tc1, BC, HV * BT, b_Ai11, BC, BT, T_eff)
    if NC >= 3:
        store_bc_bc(Akk, Akk_base, i_tc2, 0, HV * BT, b_Ai20, BC, BT, T_eff)
        store_bc_bc(Akk, Akk_base, i_tc2, BC, HV * BT, b_Ai21, BC, BT, T_eff)
        store_bc_bc(Akk, Akk_base, i_tc2, 2 * BC, HV * BT, b_Ai22, BC, BT, T_eff)
    if NC >= 4:
        store_bc_bc(Akk, Akk_base, i_tc3, 0, HV * BT, b_Ai30, BC, BT, T_eff)
        store_bc_bc(Akk, Akk_base, i_tc3, BC, HV * BT, b_Ai31, BC, BT, T_eff)
        store_bc_bc(Akk, Akk_base, i_tc3, 2 * BC, HV * BT, b_Ai32, BC, BT, T_eff)
        store_bc_bc(Akk, Akk_base, i_tc3, 3 * BC, HV * BT, b_Ai33, BC, BT, T_eff)


# --- Kernels: state scan --------------------------------------------------------------------------


@ct.kernel
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BV: ConstInt,
    BK: ConstInt,  # next_pow2(K) -- full-width state/K tile (general for any K)
    USE_G: ConstInt,
    USE_GK: ConstInt,
    USE_INITIAL_STATE: ConstInt,
    STORE_FINAL_STATE: ConstInt,
    SAVE_NEW_VALUE: ConstInt,
    STATE_V_FIRST: ConstInt,
):
    # The KV state is carried as a SINGLE full-width tile ((BV, BK) when
    # STATE_V_FIRST else (BK, BV), BK=next_pow2(K)), so the kernel is general
    # for ANY K (no K<=256 cap). Every K-axis load zero-pads cols [K:BK] and
    # every K-axis store masks them off, so the tail is 0 on load, contributes
    # 0 to every MMA, and stays 0 in the state; a padded Gk tail loads as 0 so
    # exp2(0)=1 leaves those zero cols unchanged.
    i_v = ct.bid(0)
    i_nh = ct.bid(1)
    i_n = i_nh // HV
    i_h = i_nh % HV

    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T = eos - bos
    NT = ct.cdiv(T, BT)
    boh = ct.load(chunk_offsets, (i_n,), shape=()).item()
    if STATE_V_FIRST:
        b_h = ct.zeros((BV, BK), dtype=ct.float32)
    else:
        b_h = ct.zeros((BK, BV), dtype=ct.float32)

    h_base = (boh * HV + i_h) * K * V
    v_base = (bos * HV + i_h) * V
    k_base = (bos * H + i_h // (HV // H)) * K
    w_base = (bos * HV + i_h) * K
    vnew_base = (bos * HV + i_h) * V
    h0_base = i_nh * K * V
    ht_base = i_nh * K * V

    o_bk = ct.arange(BK, dtype=ct.int32)
    o_bt = ct.arange(BT, dtype=ct.int32)
    o_bv = ct.arange(BV, dtype=ct.int32)

    # K-tile / V-tile validity masks. The kernel carries a single BK-wide K tile
    # (BK=next_pow2(K)) and BV-wide V tiles; when K or V is not a multiple of the
    # tile width the extra lanes alias the neighbouring head's H slot. The matmul
    # state rows are zeroed via K-masked K/Gk loads, but the raw H/H0/Ht
    # gather/scatter are also masked here -> no cross-head corruption when
    # K % BK != 0 (i.e. K < BK) or V % BV != 0.
    mkh = o_bk < K
    mvh = (i_v * BV + o_bv) < V

    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            row = (i_v * BV + o_bv)[:, None]
            b_h = b_h + ct.astype(
                ct.gather(
                    h0,
                    h0_base + row * K + o_bk[None, :],
                    mask=mvh[:, None] & mkh[None, :],
                    check_bounds=True,
                    padding_value=0.0,
                ),
                ct.float32,
            )
        else:
            col = (i_v * BV + o_bv)[None, :]
            b_h = b_h + ct.astype(
                ct.gather(
                    h0,
                    h0_base + o_bk[:, None] * V + col,
                    mask=mkh[:, None] & mvh[None, :],
                    check_bounds=True,
                    padding_value=0.0,
                ),
                ct.float32,
            )

    for i_t in range(NT):
        h_chunk = h_base + i_t * HV * K * V

        if STATE_V_FIRST:
            row = (i_v * BV + o_bv)[:, None]
            ct.scatter(
                h,
                h_chunk + row * K + o_bk[None, :],
                ct.astype(b_h, h.dtype),
                mask=mvh[:, None] & mkh[None, :],
                check_bounds=True,
            )
        else:
            col = (i_v * BV + o_bv)[None, :]
            ct.scatter(
                h,
                h_chunk + o_bk[:, None] * V + col,
                ct.astype(b_h, h.dtype),
                mask=mkh[:, None] & mvh[None, :],
                check_bounds=True,
            )

        w_row = (i_t * BT + o_bt)[:, None]
        wmask_r = (i_t * BT + o_bt) < T
        # Full-width K: padded cols [K:BK] gather OOB (masked to 0) and row-mask
        # zeros the partial-chunk tail, so both contribute 0 to the MMA.
        bw = ct.gather(w, w_base + w_row * (HV * K) + o_bk[None, :], mask=mkh[None, :], check_bounds=True, padding_value=0.0)
        bw = ct.where(wmask_r[:, None], bw, ct.zeros((BT, BK), dtype=bw.dtype))
        b_v = ct.zeros((BT, BV), dtype=ct.float32)
        bmat = ct.astype(ct.transpose(b_h), bw.dtype) if STATE_V_FIRST else ct.astype(b_h, bw.dtype)
        b_v = safe_mma(bw, bmat, b_v)

        v_col = (i_v * BV + o_bv)[None, :]
        vmask_c = (i_v * BV + o_bv) < V
        v_off = v_base + (i_t * BT + o_bt)[:, None] * (HV * V) + v_col
        v_full = (wmask_r[:, None]) & (vmask_c[None, :])
        b_v_load = ct.gather(v, v_off, check_bounds=True, padding_value=0.0)
        b_v_load = ct.where(v_full, b_v_load, ct.zeros((BT, BV), dtype=b_v_load.dtype))
        b_v = ct.astype(b_v_load, ct.float32) - b_v

        if SAVE_NEW_VALUE:
            vn_off = vnew_base + (i_t * BT + o_bt)[:, None] * (HV * V) + v_col
            vn_oob = ct.full((BT, BV), v.shape[0], dtype=ct.int32)
            vn_off = ct.where(v_full, vn_off, vn_oob)
            ct.scatter(v_new, vn_off, ct.astype(b_v, v_new.dtype), check_bounds=True)

        last_idx = ct_min((i_t + 1) * BT, T) - 1

        if USE_G:
            m_t = (i_t * BT + o_bt) < T
            b_g_last = ct.astype(ct.gather(g, (bos * HV + last_idx * HV + i_h,), check_bounds=True, padding_value=0.0), ct.float32)
            g_off = (bos * HV + i_h) + (i_t * BT + o_bt) * HV
            b_g = ct.astype(ct.gather(g, g_off, mask=m_t, check_bounds=True, padding_value=0.0), ct.float32)
            decay = ct.where(m_t, exp2(b_g_last - b_g), ct.zeros((BT,), dtype=ct.float32))
            b_v = b_v * decay[:, None]
            b_g_last = exp2(b_g_last)
            b_h = b_h * b_g_last

        if USE_GK:
            # Padded tail [K:BK] loads as 0 -> exp2(0)=1 leaves zero state cols unchanged.
            gk_base = (bos + last_idx) * HV * K + i_h * K
            b_gk = ct.astype(ct.gather(gk, gk_base + o_bk, mask=o_bk < K, check_bounds=True, padding_value=0.0), ct.float32)
            if STATE_V_FIRST:
                b_h = b_h * exp2(b_gk)[None, :]
            else:
                b_h = b_h * exp2(b_gk)[:, None]

        b_v = ct.astype(b_v, k.dtype)

        k_row = o_bk[:, None]
        k_time = (i_t * BT + o_bt)[None, :]
        kmask = (o_bk[:, None] < K) & ((i_t * BT + o_bt)[None, :] < T)
        bk = ct.gather(k, k_base + k_row * 1 + k_time * (H * K), check_bounds=True, padding_value=0.0)
        bk = ct.where(kmask, bk, ct.zeros((BK, BT), dtype=bk.dtype))
        prod = safe_mma(bk, b_v, ct.zeros((BK, BV), dtype=ct.float32))
        if STATE_V_FIRST:
            b_h = b_h + ct.transpose(prod)
        else:
            b_h = b_h + prod

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            row = (i_v * BV + o_bv)[:, None]
            ct.scatter(
                ht,
                ht_base + row * K + o_bk[None, :],
                ct.astype(b_h, ht.dtype),
                mask=mvh[:, None] & mkh[None, :],
                check_bounds=True,
            )
        else:
            col = (i_v * BV + o_bv)[None, :]
            ct.scatter(
                ht,
                ht_base + o_bk[:, None] * V + col,
                ct.astype(b_h, ht.dtype),
                mask=mkh[:, None] & mvh[None, :],
                check_bounds=True,
            )


@ct.kernel
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    gk,
    dstate_in,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BV: ConstInt,
    BK: ConstInt,  # next_pow2(K) -- full-width state/K tile (general for any K)
    USE_G: ConstInt,
    USE_GK: ConstInt,
    USE_INITIAL_STATE: ConstInt,
    USE_FINAL_STATE_GRADIENT: ConstInt,
    STATE_V_FIRST: ConstInt,
):
    # dH state is a SINGLE full-width tile ((BV, BK) when STATE_V_FIRST else
    # (BK, BV), BK=next_pow2(K)), so the kernel is general for ANY K (no K<=256
    # cap). Every flat gather/scatter over the K axis uses oK=arange(BK) with a
    # `(oK < K)` mask, so rows/cols [K:BK] are zero on load, contribute 0 to
    # every MMA, and are masked out on store; a Gk padding tail loads as 0 so
    # exp2(0)=1 leaves those zero state cols unchanged.
    i_v = ct.bid(0)
    i_nh = ct.bid(1)
    i_n = i_nh // HV
    i_h = i_nh % HV
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T = eos - bos
    NT = ct.cdiv(T, BT)
    boh = ct.load(chunk_offsets, (i_n,), shape=()).item()
    if STATE_V_FIRST:
        b_dh = ct.zeros((BV, BK), dtype=ct.float32)
    else:
        b_dh = ct.zeros((BK, BV), dtype=ct.float32)

    q_base = (bos * H + i_h // (HV // H)) * K
    k_base = (bos * H + i_h // (HV // H)) * K
    w_base = (bos * HV + i_h) * K
    do_base = (bos * HV + i_h) * V
    dv_base = (bos * HV + i_h) * V
    dv2_base = (bos * HV + i_h) * V
    dh_base = (boh * HV + i_h) * K * V
    gk_base = (bos * HV + i_h) * K
    dh0_base = i_nh * K * V
    dht_base = i_nh * K * V

    o_bk = ct.arange(BK, dtype=ct.int32)
    o_bt = ct.arange(BT, dtype=ct.int32)
    o_bv = ct.arange(BV, dtype=ct.int32)

    # V-boundary mask for the state-gradient (dH/dH0) scatters: a partial
    # trailing V tile (V not a multiple of BV) must not write rows/cols >= V,
    # else the store spills into the neighbouring chunk/head's state-gradient
    # (boundary-checked store semantics).
    m_v = (i_v * BV + o_bv) < V

    # --- Load final state gradient dHt -> b_dh (single full-width tile; [K:BK] loads 0) ---
    if USE_FINAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            row = (i_v * BV + o_bv)[:, None]
            b_dh = b_dh + ct.gather(
                dstate_in,
                dht_base + row * K + o_bk[None, :],
                mask=(o_bk < K)[None, :],
                check_bounds=True,
                padding_value=0.0,
            )
        else:
            col = (i_v * BV + o_bv)[None, :]
            b_dh = b_dh + ct.gather(
                dstate_in,
                dht_base + o_bk[:, None] * V + col,
                mask=(o_bk < K)[:, None],
                check_bounds=True,
                padding_value=0.0,
            )

    # cuTile range() requires a positive step; iterate forward and reverse the
    # index to preserve the backward (NT-1 .. 0) chunk traversal.
    for _i_t in range(NT):
        i_t = NT - 1 - _i_t
        dh_chunk = dh_base + i_t * HV * K * V

        # Store current b_dh to dH[i_t] (single full-width tile; [K:BK] masked out)
        if STATE_V_FIRST:
            row = (i_v * BV + o_bv)[:, None]
            ct.scatter(
                dh,
                dh_chunk + row * K + o_bk[None, :],
                ct.astype(b_dh, dh.dtype),
                mask=m_v[:, None] & (o_bk < K)[None, :],
                check_bounds=False,
            )
        else:
            col = (i_v * BV + o_bv)[None, :]
            ct.scatter(
                dh,
                dh_chunk + o_bk[:, None] * V + col,
                ct.astype(b_dh, dh.dtype),
                mask=(o_bk < K)[:, None] & m_v[None, :],
                check_bounds=False,
            )

        last_idx = ct_min((i_t + 1) * BT, T) - 1

        m_t = (i_t * BT + o_bt) < T
        if USE_G:
            bg_last = ct.astype(ct.gather(g, ((bos + last_idx) * HV + i_h,), check_bounds=True, padding_value=0.0), ct.float32)
            g_off = (bos * HV + i_h) + (i_t * BT + o_bt) * HV
            b_g = ct.astype(ct.gather(g, g_off, mask=m_t, check_bounds=True, padding_value=0.0), ct.float32)
            bg_last_exp = exp2(bg_last)
            b_g_exp = exp2(b_g)
        else:
            bg_last_exp = ct.astype(0.0, ct.float32)
            b_g_exp = ct.zeros((BT,), dtype=ct.float32)

        # dO, dV, dV2 tiles
        v_col = (i_v * BV + o_bv)[None, :]
        vmask_c = (i_v * BV + o_bv) < V
        v_full = (m_t[:, None]) & (vmask_c[None, :])
        do_off = do_base + (i_t * BT + o_bt)[:, None] * (HV * V) + v_col
        b_do = ct.gather(do, do_off, check_bounds=True, padding_value=0.0)
        b_do = ct.where(v_full, b_do, ct.zeros((BT, BV), dtype=b_do.dtype))

        # b_dv = b_k @ b_dh (single full-width K MMA). b_k tile is (BT, BK); [K:BK]
        # is masked to 0 so it contributes nothing.
        bk = ct.gather(
            k,
            k_base + (i_t * BT + o_bt)[:, None] * (H * K) + o_bk[None, :],
            mask=(o_bk < K)[None, :],
            check_bounds=True,
            padding_value=0.0,
        )
        bk = ct.where(m_t[:, None], bk, ct.zeros((BT, BK), dtype=bk.dtype))
        if USE_GK:
            # Gk offset: base (bos*HV+i_h)*K ; then + last_idx*HV*K + o_k.
            # Padded tail [K:BK] loads as 0 -> exp2(0)=1 leaves those zero cols unchanged.
            gkl = (bos * HV + i_h) * K + last_idx * HV * K
            b_gk_last = ct.astype(ct.gather(gk, gkl + o_bk, mask=o_bk < K, check_bounds=True, padding_value=0.0), ct.float32)
        bmat = ct.astype(ct.transpose(b_dh), bk.dtype) if STATE_V_FIRST else ct.astype(b_dh, bk.dtype)
        b_dv = safe_mma(bk, bmat, ct.zeros((BT, BV), dtype=ct.float32))

        if USE_G:
            decay = ct.where(m_t, exp2(bg_last - b_g), ct.zeros((BT,), dtype=ct.float32))
            b_dv = b_dv * decay[:, None]

        # b_dv += dV ; store to dV2
        dv_off = dv_base + (i_t * BT + o_bt)[:, None] * (HV * V) + v_col
        b_dv_load = ct.gather(dv, dv_off, check_bounds=True, padding_value=0.0)
        b_dv_load = ct.where(v_full, b_dv_load, ct.zeros((BT, BV), dtype=b_dv_load.dtype))
        b_dv = b_dv + ct.astype(b_dv_load, ct.float32)
        dv2_off = dv2_base + (i_t * BT + o_bt)[:, None] * (HV * V) + v_col
        dv2_oob = ct.full((BT, BV), dv2.shape[0], dtype=ct.int32)
        dv2_off = ct.where(v_full, dv2_off, dv2_oob)
        ct.scatter(dv2, dv2_off, ct.astype(b_dv, dv2.dtype), check_bounds=True)

        # b_dh += trans(b_q@b_do*scale - b_w@b_dv)  (b_q,b_w are (BK,BT) transposed;
        # rows [K:BK] are masked to 0 so contribute nothing to the update)
        b_dv_c = ct.astype(b_dv, do.dtype)
        time = (i_t * BT + o_bt)[None, :]
        tmask = (i_t * BT + o_bt)[None, :] < T
        kr = o_bk[:, None]
        wmask = (o_bk[:, None] < K) & tmask
        b_w = ct.gather(w, w_base + kr * 1 + time * (HV * K), check_bounds=True, padding_value=0.0)
        b_w = ct.where(wmask, b_w, ct.zeros((BK, BT), dtype=b_w.dtype))
        b_q = ct.gather(q, q_base + kr * 1 + time * (H * K), check_bounds=True, padding_value=0.0)
        b_q = ct.where(wmask, b_q, ct.zeros((BK, BT), dtype=b_q.dtype))
        if USE_G:
            b_dh = b_dh * bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if USE_GK:
            if STATE_V_FIRST:
                b_dh = b_dh * exp2(b_gk_last)[None, :]
            else:
                b_dh = b_dh * exp2(b_gk_last[:, None])
        term = safe_matmul(b_q, b_do) * scale - safe_matmul(b_w, b_dv_c)
        if STATE_V_FIRST:
            b_dh = b_dh + ct.transpose(term)
        else:
            b_dh = b_dh + term

    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            row = (i_v * BV + o_bv)[:, None]
            ct.scatter(
                dh0,
                dh0_base + row * K + o_bk[None, :],
                ct.astype(b_dh, dh0.dtype),
                mask=m_v[:, None] & (o_bk < K)[None, :],
                check_bounds=False,
            )
        else:
            col = (i_v * BV + o_bv)[None, :]
            ct.scatter(
                dh0,
                dh0_base + o_bk[:, None] * V + col,
                ct.astype(b_dh, dh0.dtype),
                mask=(o_bk < K)[:, None] & m_v[None, :],
                check_bounds=False,
            )


# --- Kernels: attention and gradients -------------------------------------------------------------


@ct.kernel
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    STATE_V_FIRST: ConstInt,
):
    i_v = ct.bid(0)
    i_t = ct.bid(1)
    i_hv = ct.bid(2)
    i_h = i_hv // (HV // H)

    # grid dim-1 is the GLOBAL chunk index; H is laid out per global chunk
    # (H-state kernel writes slot chunk_offsets[i_n] + local). Capture it
    # before i_t is reassigned to the per-sequence (local) chunk index.
    i_tg = i_t
    i_n = ct.load(chunk_indices, (i_t * 2,), shape=()).item()
    i_t = ct.load(chunk_indices, (i_t * 2 + 1,), shape=()).item()
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T = eos - bos
    NT = ct.cdiv(T, BT)
    o_bt = ct.arange(BT, dtype=ct.int32)
    o_bk = ct.arange(BK, dtype=ct.int32)
    o_bv = ct.arange(BV, dtype=ct.int32)
    m_s = o_bt[:, None] >= o_bt[None, :]

    q_base = (bos * H + i_h) * K
    g_base = (bos * HV + i_hv) * K
    v_base = (bos * HV + i_hv) * V
    o_base = (bos * HV + i_hv) * V
    h_base = (i_tg * HV + i_hv) * K * V
    A_base = (bos * HV + i_hv) * BT

    t_row = i_t * BT + o_bt
    m_t = t_row < T

    b_o = ct.zeros((BT, BV), dtype=ct.float32)
    num_k = (K + BK - 1) // BK
    for i_k in range(num_k):
        k_col = i_k * BK + o_bk
        m_k = k_col < K
        v_col = i_v * BV + o_bv
        m_v = v_col < V

        # b_h: STATE_V_FIRST -> view (V,K) block (BV,BK) at (i_v*BV, i_k*BK);
        #      else -> view (K,V) block (BK,BV) at (i_k*BK, i_v*BV).
        if STATE_V_FIRST:
            h_off = h_base + v_col[:, None] * K + k_col[None, :]
            h_mask = m_v[:, None] & m_k[None, :]
            b_h = ct.gather(h, h_off, mask=h_mask, check_bounds=True, padding_value=0.0)
        else:
            h_off = h_base + k_col[:, None] * V + v_col[None, :]
            h_mask = m_k[:, None] & m_v[None, :]
            b_h = ct.gather(h, h_off, mask=h_mask, check_bounds=True, padding_value=0.0)

        q_off = q_base + t_row[:, None] * (H * K) + k_col[None, :]
        g_off = g_base + t_row[:, None] * (HV * K) + k_col[None, :]
        qg_mask = m_t[:, None] & m_k[None, :]
        b_q = ct.gather(q, q_off, mask=qg_mask, check_bounds=True, padding_value=0.0)
        b_g = ct.astype(ct.gather(g, g_off, mask=qg_mask, check_bounds=True, padding_value=0.0), ct.float32)
        b_qg = ct.astype(b_q * exp2(b_g), b_q.dtype)

        if STATE_V_FIRST:
            b_o = safe_mma(b_qg, ct.astype(ct.transpose(b_h), b_qg.dtype), b_o)
        else:
            b_o = safe_mma(b_qg, ct.astype(b_h, b_qg.dtype), b_o)

    b_o = b_o * scale

    v_col = i_v * BV + o_bv
    m_v = v_col < V
    v_off = v_base + t_row[:, None] * (HV * V) + v_col[None, :]
    v_mask = m_t[:, None] & m_v[None, :]
    b_v = ct.gather(v, v_off, mask=v_mask, check_bounds=True, padding_value=0.0)

    A_col = o_bt
    A_off = A_base + t_row[:, None] * (HV * BT) + A_col[None, :]
    A_mask = m_t[:, None] & (A_col[None, :] < BT)
    b_A = ct.gather(A, A_off, mask=A_mask, check_bounds=True, padding_value=0.0)
    b_A = ct.astype(ct.where(m_s, b_A, ct.zeros((BT, BT), dtype=b_A.dtype)), b_v.dtype)
    b_o = safe_mma(b_A, b_v, b_o)

    o_off = o_base + t_row[:, None] * (HV * V) + v_col[None, :]
    ct.scatter(o, o_off, ct.astype(b_o, o.dtype), mask=v_mask, check_bounds=True)


@ct.kernel
def chunk_kda_bwd_kernel_dAv(
    q,
    k,
    v,
    A,
    do,
    dv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
):
    # NOTE: q/k params unused in the body (kept for signature parity).
    i_t = ct.bid(0)
    i_hv = ct.bid(1)

    i_n = ct.load(chunk_indices, (i_t * 2,), shape=()).item()
    i_t = ct.load(chunk_indices, (i_t * 2 + 1,), shape=()).item()
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T_eff = eos - bos

    v_base = (bos * HV + i_hv) * V
    do_base = (bos * HV + i_hv) * V
    dv_base = (bos * HV + i_hv) * V
    dA_base = (bos * HV + i_hv) * BT
    A_base = (bos * HV + i_hv) * BT

    o_bt = ct.arange(BT, dtype=ct.int32)
    o_bv = ct.arange(BV, dtype=ct.int32)
    o_t = i_t * BT + o_bt
    m_t = o_t < T_eff

    # b_A: view (BT, T) stride (1, HV*BT), block (BT, BT) at (0, i_t*BT) ->
    # element (r,c) = A_base + r*1 + (i_t*BT + c)*(HV*BT)
    A_off = A_base + o_bt[:, None] * 1 + (i_t * BT + o_bt)[None, :] * (HV * BT)
    A_mask = (o_bt[:, None] < BT) & ((i_t * BT + o_bt)[None, :] < T_eff)
    b_A = ct.gather(A, A_off, mask=A_mask, check_bounds=False, padding_value=0.0)
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_A = ct.astype(ct.where(m_A, b_A, ct.zeros((BT, BT), dtype=b_A.dtype)), do.dtype)

    b_dA = ct.zeros((BT, BT), dtype=ct.float32)
    for i_v in range(ct.cdiv(V, BV)):
        v_col = i_v * BV + o_bv
        m_v = v_col < V
        # b_v: view (V, T) stride (1, HV*V), block (BV, BT) at (i_v*BV, i_t*BT)
        v_off = v_base + v_col[:, None] * 1 + o_t[None, :] * (HV * V)
        v_mask = m_v[:, None] & m_t[None, :]
        b_v = ct.gather(v, v_off, mask=v_mask, check_bounds=False, padding_value=0.0)

        do_off = do_base + o_t[:, None] * (HV * V) + v_col[None, :]
        do_mask = m_t[:, None] & m_v[None, :]
        b_do = ct.gather(do, do_off, mask=do_mask, check_bounds=False, padding_value=0.0)

        b_dA = safe_mma(b_do, b_v, b_dA)
        b_dv = safe_matmul(b_A, b_do)
        ct.scatter(dv, do_off, ct.astype(b_dv, dv.dtype), mask=do_mask, check_bounds=False)

    b_dA = ct.where(o_t[:, None] >= o_t[None, :], b_dA * scale, ct.zeros((BT, BT), dtype=ct.float32))
    dA_off = dA_base + o_t[:, None] * (HV * BT) + o_bt[None, :]
    dA_mask = m_t[:, None] & (o_bt[None, :] < BT)
    ct.scatter(dA, dA_off, ct.astype(b_dA, dA.dtype), mask=dA_mask, check_bounds=False)


@ct.kernel
def chunk_kda_bwd_kernel_wy_dqkg_fused(
    q,
    k,
    v,
    v_new,
    g,
    beta,
    A,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    dv2,
    dg,
    db,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    STATE_V_FIRST: ConstInt,
):
    i_t = ct.bid(0)
    i_hv = ct.bid(1)
    i_h = i_hv // (HV // H)

    # grid dim-0 is the GLOBAL chunk index; H/dH are laid out per global
    # chunk (written with chunk_offsets+local). Capture it before i_t is
    # reassigned to the per-sequence (local) chunk index.
    i_tg = i_t
    i_n = ct.load(chunk_indices, (i_t * 2,), shape=()).item()
    i_t = ct.load(chunk_indices, (i_t * 2 + 1,), shape=()).item()
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T = eos - bos
    NT = ct.cdiv(T, BT)
    off_t = i_t * BT
    o_bt = ct.arange(BT, dtype=ct.int32)
    o_t = off_t + o_bt
    m_t = o_t < T
    m_last = o_t == (ct_min(T, off_t + BT) - 1)

    q_base = (bos * H + i_h) * K
    k_base = (bos * H + i_h) * K
    v_base = (bos * HV + i_hv) * V
    vnew_base = (bos * HV + i_hv) * V
    g_base = (bos * HV + i_hv) * K
    beta_base = bos * HV + i_hv
    A_base = (bos * HV + i_hv) * BT
    h_base = (i_tg * HV + i_hv) * K * V
    do_base = (bos * HV + i_hv) * V
    dh_base = (i_tg * HV + i_hv) * K * V
    dq_base = (bos * HV + i_hv) * K
    dk_base = (bos * HV + i_hv) * K
    dv_base = (bos * HV + i_hv) * V
    dv2_base = (bos * HV + i_hv) * V
    dg_base = (bos * HV + i_hv) * K
    db_base = bos * HV + i_hv
    dA_base = (bos * HV + i_hv) * BT

    b_beta = ct.gather(beta, beta_base + o_t * HV, mask=m_t, check_bounds=False, padding_value=0.0)

    # A transposed: view (BT, T) stride (1, HV*BT), block (BT, BT) at (0, off_t)
    A_row = o_bt[:, None]
    A_time = (off_t + o_bt)[None, :]
    A_m = ct.broadcast_to((o_bt < BT)[:, None], (BT, BT)) & ((off_t + o_bt)[None, :] < T)
    b_A = ct.gather(A, A_base + A_row * 1 + A_time * (HV * BT), mask=A_m, check_bounds=False, padding_value=0.0)

    b_dA = ct.zeros((BT, BT), dtype=ct.float32)
    b_db = ct.zeros((BT,), dtype=ct.float32)

    o_bk = ct.arange(BK, dtype=ct.int32)
    o_bv = ct.arange(BV, dtype=ct.int32)
    NK = (K + BK - 1) // BK
    NV = (V + BV - 1) // BV
    for i_k in range(NK):
        off_k = i_k * BK
        o_k = off_k + o_bk
        m_k = o_k < K

        k_cols = o_k[None, :]
        kg_m = m_t[:, None] & m_k[None, :]
        b_k = ct.gather(k, k_base + (off_t + o_bt)[:, None] * (H * K) + k_cols, mask=kg_m, check_bounds=False, padding_value=0.0)
        b_g = ct.astype(
            ct.gather(
                g,
                g_base + (off_t + o_bt)[:, None] * (HV * K) + k_cols,
                mask=kg_m,
                check_bounds=False,
                padding_value=0.0,
            ),
            ct.float32,
        )

        gn_idx = ct_min(T, off_t + BT) - 1
        b_gn = ct.astype(ct.gather(g, g_base + gn_idx * HV * K + o_k, mask=m_k, check_bounds=False, padding_value=0.0), ct.float32)

        b_dq = ct.zeros((BT, BK), dtype=ct.float32)
        b_dk = ct.zeros((BT, BK), dtype=ct.float32)
        b_dw = ct.zeros((BT, BK), dtype=ct.float32)
        b_dgk = ct.zeros((BK,), dtype=ct.float32)

        for i_v in range(NV):
            off_v = i_v * BV
            v_cols = (off_v + o_bv)[None, :]
            if STATE_V_FIRST:
                # H/dH: view (V, K) stride (K, 1), block (BV, BK) at (off_v, off_k)
                h_rows = (off_v + o_bv)[:, None]
                h_m = ct.broadcast_to((off_v + o_bv)[:, None] < V, (BV, BK)) & ct.broadcast_to(m_k[None, :], (BV, BK))
                b_h = ct.gather(h, h_base + h_rows * K + o_k[None, :], mask=h_m, check_bounds=False, padding_value=0.0)
                b_dh = ct.gather(dh, dh_base + h_rows * K + o_k[None, :], mask=h_m, check_bounds=False, padding_value=0.0)
            else:
                # H/dH transposed: view (V, K) stride (1, V), block (BV, BK) at (off_v, off_k)
                h_row = (off_v + o_bv)[:, None]
                h_col = o_k[None, :]
                h_m = ct.broadcast_to((off_v + o_bv)[:, None] < V, (BV, BK)) & ct.broadcast_to(m_k[None, :], (BV, BK))
                b_h = ct.gather(h, h_base + h_row * 1 + h_col * V, mask=h_m, check_bounds=False, padding_value=0.0)
                b_dh = ct.gather(dh, dh_base + h_row * 1 + h_col * V, mask=h_m, check_bounds=False, padding_value=0.0)

            v_m = m_t[:, None] & ((off_v + o_bv) < V)[None, :]
            b_v_new = ct.gather(
                v_new,
                vnew_base + (off_t + o_bt)[:, None] * (HV * V) + v_cols,
                mask=v_m,
                check_bounds=False,
                padding_value=0.0,
            )
            b_do = ct.gather(
                do,
                do_base + (off_t + o_bt)[:, None] * (HV * V) + v_cols,
                mask=v_m,
                check_bounds=False,
                padding_value=0.0,
            )
            b_dv = ct.gather(
                dv,
                dv_base + (off_t + o_bt)[:, None] * (HV * V) + v_cols,
                mask=v_m,
                check_bounds=False,
                padding_value=0.0,
            )

            b_dgk = b_dgk + ct.sum(ct.astype(b_h, ct.float32) * ct.astype(b_dh, ct.float32), axis=0)
            b_dq = safe_mma(b_do, ct.astype(b_h, b_do.dtype), b_dq)
            b_dk = safe_mma(b_v_new, ct.astype(b_dh, b_v_new.dtype), b_dk)
            b_dw = safe_mma(ct.astype(b_dv, b_v_new.dtype), ct.astype(b_h, b_v_new.dtype), b_dw)
            # the b_v reuse below relies on program order within the block.
            if i_k == 0:
                b_v = ct.gather(
                    v,
                    v_base + (off_t + o_bt)[:, None] * (HV * V) + v_cols,
                    mask=v_m,
                    check_bounds=False,
                    padding_value=0.0,
                )
                b_dA = safe_mma(b_dv, ct.transpose(b_v), b_dA)
                b_dvb = safe_matmul(b_A, b_dv)
                b_dv2 = b_dvb * ct.astype(b_beta, b_dvb.dtype)[:, None]
                b_db = b_db + ct.sum(ct.astype(b_dvb, ct.float32) * ct.astype(b_v, ct.float32), axis=1)
                ct.scatter(
                    dv2,
                    dv2_base + (off_t + o_bt)[:, None] * (HV * V) + v_cols,
                    ct.astype(b_dv2, dv2.dtype),
                    mask=v_m,
                    check_bounds=False,
                )

        b_gk_exp = exp2(b_g)
        b_gb = b_gk_exp * ct.astype(b_beta, ct.float32)[:, None]
        b_dgk = b_dgk * exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        b_dk = b_dk * ct.where(m_t[:, None], exp2(b_gn[None, :] - b_g), ct.zeros((BT, BK), dtype=ct.float32))

        b_kg = ct.astype(b_k, ct.float32) * b_gk_exp

        b_dw = ct.astype(-b_dw, b_A.dtype)
        b_dA = safe_mma(b_dw, ct.transpose(ct.astype(b_kg, b_A.dtype)), b_dA)

        b_dkgb = safe_matmul(b_A, b_dw)
        b_db = b_db + ct.sum(ct.astype(b_dkgb, ct.float32) * b_kg, axis=1)

        b_q = ct.gather(q, q_base + (off_t + o_bt)[:, None] * (H * K) + k_cols, mask=kg_m, check_bounds=False, padding_value=0.0)
        b_kdk = ct.astype(b_k, ct.float32) * b_dk
        b_dgk = b_dgk + ct.sum(b_kdk, axis=0)
        b_dg = (
            ct.astype(b_q, ct.float32) * b_dq
            - b_kdk
            + ct.astype(m_last, ct.float32)[:, None] * b_dgk[None, :]
            + b_kg * ct.astype(b_dkgb, ct.float32) * ct.astype(b_beta, ct.float32)[:, None]
        )
        b_dk = b_dk + ct.astype(b_dkgb, ct.float32) * b_gb

        ct.scatter(
            dq,
            dq_base + (off_t + o_bt)[:, None] * (HV * K) + k_cols,
            ct.astype(b_dq, dq.dtype),
            mask=kg_m,
            check_bounds=False,
        )
        ct.scatter(
            dk,
            dk_base + (off_t + o_bt)[:, None] * (HV * K) + k_cols,
            ct.astype(b_dk, dk.dtype),
            mask=kg_m,
            check_bounds=False,
        )
        ct.scatter(
            dg,
            dg_base + (off_t + o_bt)[:, None] * (HV * K) + k_cols,
            ct.astype(b_dg, dg.dtype),
            mask=kg_m,
            check_bounds=False,
        )

    m_A = (o_bt[:, None] > o_bt[None, :]) & (m_t[:, None] & m_t[None, :])
    b_dA = ct.where(m_A, b_dA * ct.astype(b_beta, ct.float32)[None, :], ct.zeros((BT, BT), dtype=ct.float32))
    b_dA = safe_matmul(ct.astype(b_dA, b_A.dtype), b_A)
    b_dA = safe_matmul(b_A, ct.astype(b_dA, b_A.dtype))
    b_dA = ct.where(m_A, -b_dA, ct.zeros((BT, BT), dtype=ct.float32))

    ct.scatter(db, db_base + o_t * HV, ct.astype(b_db, db.dtype), mask=m_t, check_bounds=False)
    dA_rows = (off_t + o_bt)[:, None]
    dA_cols = o_bt[None, :]
    dA_m = ((off_t + o_bt)[:, None] < T) & ct.broadcast_to((o_bt < BT)[None, :], (BT, BT))
    ct.scatter(dA, dA_base + dA_rows * (HV * BT) + dA_cols, ct.astype(b_dA, dA.dtype), mask=dA_m, check_bounds=False)


@ct.kernel
def chunk_kda_bwd_kernel_intra(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dg,
    dg2,
    db,
    cu_seqlens,
    chunk_indices,
    T,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BC: ConstInt,
    BK: ConstInt,
    NC: ConstInt,
    SAFE_GATE: ConstInt,
):
    i_kc = ct.bid(0)
    i_t = ct.bid(1)
    i_hv = ct.bid(2)
    i_h = i_hv // (HV // H)
    i_k = i_kc // NC
    i_i = i_kc % NC

    all_bt = T
    i_n = ct.load(chunk_indices, (i_t * 2,), shape=()).item()
    i_t = ct.load(chunk_indices, (i_t * 2 + 1,), shape=()).item()
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T_eff = eos - bos
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T_eff:
        return

    o_bc = ct.arange(BC, dtype=ct.int32)
    o_bk = ct.arange(BK, dtype=ct.int32)
    o_k = i_k * BK + o_bk
    m_k = o_k < K

    q_base = (bos * H + i_h) * K
    k_base = (bos * H + i_h) * K
    g_base = (bos * HV + i_hv) * K
    beta_base = bos * HV + i_hv
    dAqk_base = (bos * HV + i_hv) * BT
    dAkk_base = (bos * HV + i_hv) * BT
    dq_base = (bos * HV + i_hv) * K
    dq2_base = (bos * HV + i_hv) * K
    dk_base = (bos * HV + i_hv) * K
    dk2_base = (bos * HV + i_hv) * K
    dg_base = (bos * HV + i_hv) * K
    dg2_base = (bos * HV + i_hv) * K
    db_base = (i_k * all_bt + bos) * HV + i_hv

    # b_g (current sub-chunk)
    cur_rows = (i_ti + o_bc)[:, None]
    cur_cols = o_k[None, :]
    m_cur = ((i_ti + o_bc) < T_eff)[:, None] & m_k[None, :]
    b_g = ct.astype(
        ct.gather(g, g_base + cur_rows * (HV * K) + cur_cols, mask=m_cur, check_bounds=False, padding_value=0.0),
        ct.float32,
    )
    b_b = ct.astype(
        ct.gather(beta, beta_base + (i_ti + o_bc) * HV, mask=(i_ti + o_bc) < T_eff, check_bounds=False, padding_value=0.0),
        ct.float32,
    )

    b_dq2 = ct.zeros((BC, BK), dtype=ct.float32)
    b_dk2 = ct.zeros((BC, BK), dtype=ct.float32)
    if i_i > 0:
        b_gn = ct.astype(ct.gather(g, g_base + i_ti * (HV * K) + o_k, mask=m_k, check_bounds=False, padding_value=0.0), ct.float32)[None, :]
        for i_j in range(0, i_i):
            jrow0 = i_t * BT + i_j * BC
            j_rows = (jrow0 + o_bc)[:, None]
            m_j = ((jrow0 + o_bc) < T_eff)[:, None] & m_k[None, :]
            b_k = ct.gather(k, k_base + j_rows * (H * K) + cur_cols, mask=m_j, check_bounds=False, padding_value=0.0)
            b_gk = ct.gather(g, g_base + j_rows * (HV * K) + cur_cols, mask=m_j, check_bounds=False, padding_value=0.0)
            # dAqk/dAkk: view (T, BT) stride (HV*BT, 1), block (BC, BC) at (i_ti, i_j*BC)
            dA_rows = (i_ti + o_bc)[:, None]
            dA_cols = (i_j * BC + o_bc)[None, :]
            m_dA = ((i_ti + o_bc) < T_eff)[:, None] & ((i_j * BC + o_bc) < BT)[None, :]
            b_dAqk = ct.gather(dAqk, dAqk_base + dA_rows * (HV * BT) + dA_cols, mask=m_dA, check_bounds=False, padding_value=0.0)
            b_dAkk = ct.gather(dAkk, dAkk_base + dA_rows * (HV * BT) + dA_cols, mask=m_dA, check_bounds=False, padding_value=0.0)
            b_kg = ct.astype(b_k, ct.float32) * exp2(b_gn - ct.astype(b_gk, ct.float32))
            b_dq2 = safe_mma(b_dAqk, ct.astype(b_kg, b_dAqk.dtype), b_dq2)
            b_dk2 = safe_mma(b_dAkk, ct.astype(b_kg, b_dAkk.dtype), b_dk2)
        b_gqn = exp2(b_g - b_gn)
        b_dq2 = b_dq2 * b_gqn
        b_dk2 = b_dk2 * b_gqn

    o_i = o_bc
    # current block Q, K
    b_q = ct.gather(q, q_base + cur_rows * (H * K) + cur_cols, mask=m_cur, check_bounds=False, padding_value=0.0)
    b_k = ct.gather(k, k_base + cur_rows * (H * K) + cur_cols, mask=m_cur, check_bounds=False, padding_value=0.0)

    if SAFE_GATE:
        gn_row = i_ti + ct_min(BC // 2, T_eff - i_ti - 1)
        b_gn2 = ct.astype(ct.gather(g, g_base + gn_row * (HV * K) + o_k, mask=m_k, check_bounds=False, padding_value=0.0), ct.float32)[None, :]
        dA_rows = (i_ti + o_bc)[:, None]
        dA_cols = (i_i * BC + o_bc)[None, :]
        m_dA = ((i_ti + o_bc) < T_eff)[:, None] & ((i_i * BC + o_bc) < BT)[None, :]
        b_dAqk_diag = ct.astype(
            ct.gather(dAqk, dAqk_base + dA_rows * (HV * BT) + dA_cols, mask=m_dA, check_bounds=False, padding_value=0.0),
            ct.float32,
        )
        b_dAkk_diag = ct.astype(
            ct.gather(dAkk, dAkk_base + dA_rows * (HV * BT) + dA_cols, mask=m_dA, check_bounds=False, padding_value=0.0),
            ct.float32,
        )

        m_i_diag = (o_i[:, None] >= o_i[None, :]) & ((i_ti + o_i[:, None]) < T_eff) & ((i_ti + o_i[None, :]) < T_eff)
        m_j_diag = (i_ti + o_i[:, None]) < T_eff
        b_dAqk_diag = ct.where(m_i_diag, b_dAqk_diag, ct.zeros((BC, BC), dtype=ct.float32))
        b_dAkk_diag = ct.where(m_i_diag, b_dAkk_diag, ct.zeros((BC, BC), dtype=ct.float32))
        b_g_diag = ct.where(m_j_diag, b_g - b_gn2, ct.zeros((BC, BK), dtype=ct.float32))
        exp_g = ct.where(m_j_diag, exp2(b_g_diag), ct.zeros((BC, BK), dtype=ct.float32))
        exp_ng = ct.where(m_j_diag, exp2(-b_g_diag), ct.zeros((BC, BK), dtype=ct.float32))

        b_k_exp = ct.astype(b_k, ct.float32) * exp_ng
        b_dq2 = b_dq2 + safe_matmul(b_dAqk_diag, ct.astype(b_k_exp, b_dAqk_diag.dtype)) * exp_g
        b_dk2 = b_dk2 + safe_matmul(b_dAkk_diag, ct.astype(b_k_exp, b_dAkk_diag.dtype)) * exp_g
    else:
        nj = ct_min(BC, T_eff - i_t * BT - i_i * BC)
        for j in range(0, nj):
            b_dAqk = ct.gather(
                dAqk,
                dAqk_base + (i_ti + o_bc) * (HV * BT) + (i_i * BC) + j,
                mask=(i_ti + o_bc) < T_eff,
                check_bounds=False,
                padding_value=0.0,
            )
            b_dAkk = ct.gather(
                dAkk,
                dAkk_base + (i_ti + o_bc) * (HV * BT) + (i_i * BC) + j,
                mask=(i_ti + o_bc) < T_eff,
                check_bounds=False,
                padding_value=0.0,
            )
            b_kj = ct.astype(
                ct.gather(k, k_base + (i_ti + j) * (H * K) + o_k, mask=m_k, check_bounds=False, padding_value=0.0),
                ct.float32,
            )
            b_gkj = ct.astype(
                ct.gather(g, g_base + (i_ti + j) * (HV * K) + o_k, mask=m_k, check_bounds=False, padding_value=0.0),
                ct.float32,
            )
            m_i = o_i[:, None] >= j
            b_gqk = exp2(b_g - b_gkj[None, :])
            b_dq2 = b_dq2 + ct.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, ct.zeros((BC, BK), dtype=ct.float32))
            b_dk2 = b_dk2 + ct.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, ct.zeros((BC, BK), dtype=ct.float32))

    b_db = ct.sum(b_dk2 * ct.astype(b_k, ct.float32), axis=1)
    b_dk2 = b_dk2 * b_b[:, None]

    b_dg2 = ct.astype(b_q, ct.float32) * b_dq2

    # dQ2 = b_dq2 + dQ
    dq_off = dq_base + cur_rows * (HV * K) + cur_cols
    b_dq_prev = ct.astype(ct.gather(dq, dq_off, mask=m_cur, check_bounds=False, padding_value=0.0), ct.float32)
    b_dq2_out = b_dq2 + b_dq_prev
    ct.scatter(dq2, dq2_base + cur_rows * (HV * K) + cur_cols, ct.astype(b_dq2_out, dq2.dtype), mask=m_cur, check_bounds=False)
    ct.scatter(db, db_base + (i_ti + o_bc) * HV, ct.astype(b_db, db.dtype), mask=(i_ti + o_bc) < T_eff, check_bounds=False)

    # --- second half: b_dkt accumulation (transposed dAqk/dAkk views) ---
    b_dkt = ct.zeros((BC, BK), dtype=ct.float32)
    NCl = ct_min(NC, ct.cdiv(T_eff - i_t * BT, BC))
    if i_i < NCl - 1:
        gn_row2 = ct_min(i_ti + BC, T_eff) - 1
        b_gn3 = ct.astype(ct.gather(g, g_base + gn_row2 * (HV * K) + o_k, mask=m_k, check_bounds=False, padding_value=0.0), ct.float32)[None, :]
        for i_j in range(i_i + 1, NCl):
            jrow0 = i_t * BT + i_j * BC
            j_idx = jrow0 + o_bc
            m_jr = j_idx < T_eff
            b_bj = ct.astype(ct.gather(beta, beta_base + j_idx * HV, mask=m_jr, check_bounds=False, padding_value=0.0), ct.float32)
            j_rows = j_idx[:, None]
            m_jrk = m_jr[:, None] & m_k[None, :]
            b_qj = ct.gather(q, q_base + j_rows * (H * K) + cur_cols, mask=m_jrk, check_bounds=False, padding_value=0.0)
            b_kbj = (
                ct.astype(
                    ct.gather(k, k_base + j_rows * (H * K) + cur_cols, mask=m_jrk, check_bounds=False, padding_value=0.0),
                    ct.float32,
                )
                * b_bj[:, None]
            )
            b_gkj = ct.astype(
                ct.gather(g, g_base + j_rows * (HV * K) + cur_cols, mask=m_jrk, check_bounds=False, padding_value=0.0),
                ct.float32,
            )
            # dAqk/dAkk transposed: view (BT, T) stride (1, HV*BT), block (BC, BC) at (i_i*BC, jrow0)
            tr_rows = (i_i * BC + o_bc)[:, None]
            tr_cols = (jrow0 + o_bc)[None, :]
            m_tr = ((i_i * BC + o_bc) < BT)[:, None] & ((jrow0 + o_bc) < T_eff)[None, :]
            b_dAqk = ct.gather(dAqk, dAqk_base + tr_rows * 1 + tr_cols * (HV * BT), mask=m_tr, check_bounds=False, padding_value=0.0)
            b_dAkk = ct.gather(dAkk, dAkk_base + tr_rows * 1 + tr_cols * (HV * BT), mask=m_tr, check_bounds=False, padding_value=0.0)

            b_gkn = exp2(b_gkj - b_gn3)
            decay = ct.where(m_jr[:, None], b_gkn, ct.zeros((BC, BK), dtype=ct.float32))
            b_qg = ct.astype(b_qj, ct.float32) * decay
            b_kbg = b_kbj * decay
            b_dkt = safe_mma(b_dAqk, ct.astype(b_qg, b_dAqk.dtype), b_dkt)
            b_dkt = safe_mma(b_dAkk, ct.astype(b_kbg, b_dAkk.dtype), b_dkt)
        b_dkt = b_dkt * exp2(b_gn3 - b_g)

    if SAFE_GATE:
        gn_row = i_ti + ct_min(BC // 2, T_eff - i_ti - 1)
        b_gn4 = ct.astype(ct.gather(g, g_base + gn_row * (HV * K) + o_k, mask=m_k, check_bounds=False, padding_value=0.0), ct.float32)[None, :]
        # transposed dAqk/dAkk at (i_i*BC, i_ti)
        tr_rows = (i_i * BC + o_bc)[:, None]
        tr_cols = (i_ti + o_bc)[None, :]
        m_tr = ((i_i * BC + o_bc) < BT)[:, None] & ((i_ti + o_bc) < T_eff)[None, :]
        b_dAqk_kk = ct.astype(
            ct.gather(dAqk, dAqk_base + tr_rows * 1 + tr_cols * (HV * BT), mask=m_tr, check_bounds=False, padding_value=0.0),
            ct.float32,
        )
        b_dAkk_kk = ct.astype(
            ct.gather(dAkk, dAkk_base + tr_rows * 1 + tr_cols * (HV * BT), mask=m_tr, check_bounds=False, padding_value=0.0),
            ct.float32,
        )

        m_i_kk = (o_i[:, None] <= o_i[None, :]) & ((i_ti + o_i[:, None]) < T_eff) & ((i_ti + o_i[None, :]) < T_eff)
        m_j_kk = (i_ti + o_i[:, None]) < T_eff
        b_dAqk_kk = ct.where(m_i_kk, b_dAqk_kk, ct.zeros((BC, BC), dtype=ct.float32))
        b_dAkk_kk = ct.where(m_i_kk, b_dAkk_kk, ct.zeros((BC, BC), dtype=ct.float32))
        b_g_kk = ct.where(m_j_kk, b_g - b_gn4, ct.zeros((BC, BK), dtype=ct.float32))
        exp_g_kk = ct.where(m_j_kk, exp2(b_g_kk), ct.zeros((BC, BK), dtype=ct.float32))
        exp_ng_kk = ct.where(m_j_kk, exp2(-b_g_kk), ct.zeros((BC, BK), dtype=ct.float32))

        b_q_exp = ct.astype(b_q, ct.float32) * exp_g_kk
        b_kb_exp = ct.astype(b_k, ct.float32) * b_b[:, None] * exp_g_kk
        b_dkt = b_dkt + safe_matmul(b_dAqk_kk, ct.astype(b_q_exp, b_dAqk_kk.dtype)) * exp_ng_kk
        b_dkt = b_dkt + safe_matmul(b_dAkk_kk, ct.astype(b_kb_exp, b_dAkk_kk.dtype)) * exp_ng_kk
    else:
        nj = ct_min(BC, T_eff - i_t * BT - i_i * BC)
        for j in range(0, nj):
            # transposed scalar reads: dAqk + (i_ti*HV*BT + i_i*BC + o_i) + j*HV*BT
            base_o = i_ti * (HV * BT) + i_i * BC + o_i
            # A maskless gather still emits a default `other` (padding_value=0)
            # operand; without a matching `mask` the tileiras LDGSTS lowering
            # asserts (llOthers non-empty, llMasks empty). Supply an explicit
            # row-validity mask.
            m_row = (i_ti + o_bc) < T_eff
            b_dAqk = ct.gather(dAqk, dAqk_base + base_o + j * (HV * BT), mask=m_row, check_bounds=False, padding_value=0.0)
            b_dAkk = ct.gather(dAkk, dAkk_base + base_o + j * (HV * BT), mask=m_row, check_bounds=False, padding_value=0.0)
            b_qj = ct.astype(
                ct.gather(q, q_base + (i_ti + j) * (H * K) + o_k, mask=m_k, check_bounds=False, padding_value=0.0),
                ct.float32,
            )
            b_bj = ct.astype(
                ct.gather(beta, beta_base + (i_ti + j) * HV, mask=(i_ti + j) < T_eff, check_bounds=False, padding_value=0.0),
                ct.float32,
            ).item()
            b_kbj = (
                ct.astype(
                    ct.gather(k, k_base + (i_ti + j) * (H * K) + o_k, mask=m_k, check_bounds=False, padding_value=0.0),
                    ct.float32,
                )
                * b_bj
            )
            b_gkj = ct.astype(
                ct.gather(g, g_base + (i_ti + j) * (HV * K) + o_k, mask=m_k, check_bounds=False, padding_value=0.0),
                ct.float32,
            )
            m_i = o_i[:, None] <= j
            b_gkq = exp2(b_gkj[None, :] - b_g)
            b_dkt = b_dkt + ct.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, ct.zeros((BC, BK), dtype=ct.float32))
            b_dkt = b_dkt + ct.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * b_gkq, ct.zeros((BC, BK), dtype=ct.float32))

    dk_off = dk_base + cur_rows * (HV * K) + cur_cols
    dg_off = dg_base + cur_rows * (HV * K) + cur_cols
    b_dg_prev = ct.astype(ct.gather(dg, dg_off, mask=m_cur, check_bounds=False, padding_value=0.0), ct.float32)
    b_dk_prev = ct.astype(ct.gather(dk, dk_off, mask=m_cur, check_bounds=False, padding_value=0.0), ct.float32)
    b_dg2 = b_dg2 + (b_dk2 - b_dkt) * ct.astype(b_k, ct.float32) + b_dg_prev
    b_dk2_out = b_dk2 + b_dk_prev + b_dkt
    ct.scatter(dk2, dk2_base + cur_rows * (HV * K) + cur_cols, ct.astype(b_dk2_out, dk2.dtype), mask=m_cur, check_bounds=False)
    ct.scatter(dg2, dg2_base + cur_rows * (HV * K) + cur_cols, ct.astype(b_dg2, dg2.dtype), mask=m_cur, check_bounds=False)


# --- Launchers: normalization and gates -----------------------------------------------------------


BS_LIST_DEFAULT = 32


def chunk_local_cumsum_vector(
    g,
    chunk_size,
    reverse=False,
    scale=None,
    cu_seqlens=None,
    chunk_indices=None,
    out=None,
    stream=None,
):
    stream = 0 if stream is None else stream
    T, H, S = g.shape
    BT = chunk_size
    NT = len(chunk_indices)
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    BS = min(BS_LIST_DEFAULT, next_power_of_2(S))
    g_org = g
    g_out = out.reshape((T, H, S))
    scale_val = float(scale) if scale is not None else 0.0
    has_scale = int(scale is not None)
    cu_arg = cu_seqlens.reshape(-1)
    ci_arg = chunk_indices.reshape(-1)
    grid = (cdiv(S, BS), NT, H)
    ct.launch(
        stream,
        grid,
        chunk_local_cumsum_vector_kernel,
        (
            g_org.reshape((-1,)),
            g_out.reshape(-1),
            scale_val,
            cu_arg,
            ci_arg,
            H,
            S,
            BT,
            BS,
            int(bool(reverse)),
            has_scale,
        ),
    )
    return g_out


def chunk_local_cumsum(
    g,
    chunk_size,
    reverse=False,
    scale=None,
    cu_seqlens=None,
    chunk_indices=None,
    out=None,
    stream=None,
):
    stream = 0 if stream is None else stream
    return chunk_local_cumsum_vector(
        g=g,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        out=out,
        stream=stream,
    )


def kda_gate_chunk_cumsum(
    g,
    A_log,
    chunk_size,
    scale=None,
    dt_bias=None,
    cu_seqlens=None,
    chunk_indices=None,
    lower_bound=None,
    out=None,
    bufs=None,
    stream=None,
):
    stream = 0 if stream is None else stream
    T, H, S = g.shape
    BT = chunk_size
    NT = len(chunk_indices)
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    BS = min(BS_LIST_DEFAULT, next_power_of_2(S))
    g_out = out.reshape((T, H, S))
    dt_arg = opt(dt_bias, bufs, dtname(A_log)).reshape((-1,))
    scale_val = float(scale) if scale is not None else 0.0
    lb_val = float(lower_bound) if lower_bound is not None else 0.0
    cu_arg = cu_seqlens.reshape(-1)
    ci_arg = chunk_indices.reshape(-1)
    grid = (cdiv(S, BS), NT, H)
    ct.launch(
        stream,
        grid,
        kda_gate_chunk_cumsum_vector_kernel,
        (
            g.reshape((-1,)),
            A_log.reshape((-1,)),
            dt_arg,
            g_out.reshape(-1),
            scale_val,
            cu_arg,
            ci_arg,
            lb_val,
            H,
            S,
            BT,
            BS,
            0,
            int(dt_bias is not None),
            int(scale is not None),
            int(lower_bound is not None),
        ),
    )
    return g_out


def kda_gate_bwd(g, A_log, dt_bias=None, dyg=None, lower_bound=None, dg_out=None, dA_out=None, dbias_out=None, bufs=None, stream=None):
    """Vector-gate backward. ``dg_out`` (g-shaped), ``dA_out`` (A_log-shaped) and —
    with ``dt_bias`` — ``dbias_out`` (H*K) are written in place; ``bufs['dA_gate']``
    / ``bufs['db_gate']`` hold the (NT, H) / (NT, H*K) fp32 chunk partials."""
    stream = 0 if stream is None else stream
    H, K = g.shape[-2:]
    T = g.numel() // (H * K)
    BT = 32
    NT = cdiv(T, BT)
    BD = next_power_of_2(K)
    dg = dg_out.reshape(tuple(g.shape))
    dA_nt = bufs["dA_gate"].reshape((NT, H))
    db_nt = bufs["db_gate"].reshape((NT, H * K)) if dt_bias is not None else None
    dt_arg = opt(dt_bias, bufs, dtname(A_log)).reshape((-1,))
    db_arg = opt(db_nt, bufs).reshape(-1)
    lb_val = float(lower_bound) if lower_bound is not None else 0.0
    grid = (NT, H)
    ct.launch(
        stream,
        grid,
        kda_gate_bwd_kernel,
        (
            g.reshape((-1,)),
            A_log.reshape((-1,)),
            dt_arg,
            dyg.reshape((-1,)),
            dg.reshape(-1),
            dA_nt.reshape(-1),
            db_arg,
            lb_val,
            T,
            H,
            K,
            BT,
            BD,
            int(dt_bias is not None),
            int(lower_bound is not None),
        ),
    )
    sum_leading(dA_out.reshape((H,)), dA_nt, NT, H, stream=stream)
    if dt_bias is not None:
        sum_leading(dbias_out.reshape((H * K,)), db_nt, NT, H * K, stream=stream)
    return dg, dA_out, (dbias_out if dt_bias is not None else None)


# --- Launchers: WY representation -----------------------------------------------------------------


def recompute_w_u_fwd(k, v, beta, A, gk, q=None, cu_seqlens=None, chunk_indices=None, bufs=None, stream=None):
    stream = 0 if stream is None else stream
    T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[1]
    BT = A.shape[-1]
    BK = 32
    BV = 32
    NT = len(chunk_indices)
    dev = current_device_id()

    w = bufs["w"].reshape((T, HV, K))
    u = bufs["u"].reshape((T, HV, V))
    qg = bufs["qg"].reshape((T, HV, K)) if q is not None else None
    kg = bufs["kg"].reshape((T, HV, K))

    cu_arg = cu_seqlens.reshape(-1)
    ci_arg = chunk_indices.reshape(-1)
    q_arg = opt(q, bufs, dtname(k))
    qg_arg = opt(qg, bufs, dtname(k))
    # Kernel uses flat element-offset gather/scatter; pass 1-D views so the
    # cuTile index-tuple rank (1) matches the array rank. reshape(-1) on these
    # contiguous tensors yields storage-aliasing views (outputs W/U/QG/KG too).
    _wu_grid = (NT, HV)
    _wu_args = (
        q_arg.reshape((-1,)),
        k.reshape((-1,)),
        qg_arg.reshape(-1),
        kg.reshape(-1),
        v.reshape((-1,)),
        beta.reshape((-1,)),
        w.reshape(-1),
        u.reshape(-1),
        A.reshape(-1),
        gk.reshape(-1),
        cu_arg,
        ci_arg,
        H,
        HV,
        K,
        V,
        BT,
        BK,
        BV,
        1,
        int(q is not None),
        1,
    )
    # Launch-hint autotune (occupancy x num_worker_warps) on this wy_fast
    # recompute kernel.
    _wu_key = (
        "recompute_w_u_fwd_kda_kernel",
        int(H),
        int(HV),
        int(K),
        int(V),
        int(BT),
        int(BK),
        int(BV),
        int(q is not None),
        1,
        str(k.dtype),
        str(dev),
    )
    autotuned_launch(recompute_w_u_fwd_kda_kernel, _wu_key, _wu_grid, _wu_args, occ_choices=(1, 2, 4, 8), nww_choices=(4,), stream=stream)
    return w, u, qg, kg


def chunk_kda_fwd_intra_token_parallel(q, k, gk, beta, Aqk, Akk, scale, cu_seqlens=None, chunk_size=64, sub_chunk_size=16, stream=None):
    stream = 0 if stream is None else stream
    T, H, K, HV = *q.shape, gk.shape[1]
    N = len(cu_seqlens) - 1
    BT = chunk_size
    BC = sub_chunk_size
    # BH heads per block: larger BH amortizes the per-block binary search + the
    # BC-wide j-loop (gathers reused across heads share the same row indexing).
    # BH in {1,2,4,8}; HV must be divisible for the grid split.
    BH = 4 if (HV % 4 == 0) else (2 if (HV % 2 == 0) else 1)
    BK = next_power_of_2(K)
    cu_arg = cu_seqlens.reshape(-1)
    grid = (T, cdiv(HV, BH))
    # cuTile gather/scatter index-tuple rank must match the array rank, so pass
    # pre-flattened views: Q/K -> (T*H, K), Gk -> (T*HV, K), Beta/Aqk/Akk -> 1-D.
    # Aqk/Akk are contiguous, so reshape(-1) is a view aliasing the original storage.
    ct.launch(
        stream,
        grid,
        chunk_kda_fwd_kernel_intra_token_parallel,
        (
            q.reshape((-1, K)),
            k.reshape((-1, K)),
            gk.reshape(-1, K),
            beta.reshape((-1,)),
            Aqk.reshape(-1),
            Akk.reshape(-1),
            float(scale),
            cu_arg,
            N,
            T,
            H,
            HV,
            K,
            BT,
            BC,
            BH,
            BK,
        ),
    )
    return Aqk, Akk


def chunk_kda_fwd_intra(
    q,
    k,
    v,
    gk=None,
    beta=None,
    scale=None,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
    safe_gate=False,
    disable_recompute=False,
    bufs=None,
    stream=None,
):
    stream = 0 if stream is None else stream
    T, H, K, HV = *k.shape, gk.shape[1]
    BT = chunk_size
    if BT not in (32, 64):
        raise ValueError(f"KDA intra chunk kernel only supports chunk_size 32 or 64, got {BT}.")
    # BC=32 (NC=2) for BT=64,K>=64. With NC=2 there
    # is exactly ONE off-diagonal pair -> only 2 live [32,32] accumulators (vs 12
    # [16,16] at NC=4), and its merged-inverse [32,32]@[32,32] ct.matmul lowers
    # to HMMA (M=32) instead of SIMT scalar FADD/FMUL (M=16).
    # K<64 falls back to BC=16/NC=4.
    BC = 32 if BT == 64 and K >= 64 else 16
    NT = len(chunk_indices)
    NC = cdiv(BT, BC)
    # use_split_diag_compute_solve: pre-solve diagonals in
    # inter_diag_compute_solve so inter_solve_fused can SKIP forward-substitution.
    use_split_diag_compute_solve = (not safe_gate) and BT == 64 and K >= 64
    use_solved_diagonal = safe_gate or use_split_diag_compute_solve
    dev = current_device_id()

    Aqk = bufs["Aqk"].reshape((T, HV, BT))
    Akk = bufs["Akk"].reshape((T, HV, BT))
    zero_fill(Akk, stream=stream)
    Akkd = bufs["Akkd"].reshape((T, HV, BC))

    cu_arg = cu_seqlens.reshape(-1)
    ci_arg = chunk_indices.reshape(-1)

    # Step 1: diagonal blocks into Akkd (fp32). When use_solved_diagonal is set
    # (safe_gate OR the split path) the diagonals are PRE-SOLVED
    # here so the inter-solve kernel can skip forward-substitution.
    if use_solved_diagonal:
        BK = next_power_of_2(K)
        # Kernel uses flat element-offset gather/scatter; pass 1-D views so the
        # cuTile index-tuple rank (1) matches the array rank.
        _diag_grid = (NT, NC, HV)
        _diag_args = (
            q.reshape((-1,)),
            k.reshape((-1,)),
            gk.reshape(-1),
            beta.reshape((-1,)),
            Aqk.reshape(-1),
            Akkd.reshape(-1),
            float(scale),
            cu_arg,
            ci_arg,
            H,
            HV,
            K,
            BT,
            BC,
            BK,
        )
        # Launch-hint autotune (occupancy x num_worker_warps) on this aux
        # kernel.
        _diag_key = (
            "chunk_kda_fwd_kernel_inter_diag_compute_solve",
            int(H),
            int(HV),
            int(K),
            int(BT),
            int(BC),
            int(BK),
            1,
            str(k.dtype),
            str(dev),
        )
        autotuned_launch(chunk_kda_fwd_kernel_inter_diag_compute_solve, _diag_key, _diag_grid, _diag_args, stream=stream)
    else:
        Aqk, Akkd = chunk_kda_fwd_intra_token_parallel(
            q=q, k=k, gk=gk, beta=beta, Aqk=Aqk, Akk=Akkd, scale=scale, cu_seqlens=cu_seqlens, chunk_size=BT, sub_chunk_size=BC, stream=stream
        )

    # Step 2: inter-subchunk blocks + merged solve_tril
    # (chunk_kda_fwd_intra): ALL NC use the single inter_solve_fused
    # kernel. The fused kernel handles NC>=3/NC>=4 internally (block-triangular
    # forward-substitution over all sub-chunk pairs). With use_solved_diagonal
    # the forward-substitution is skipped.
    BKf = next_power_of_2(K)
    # Launch-hint autotune (occupancy x num_worker_warps) on this fused
    # solve kernel.
    _isf_grid = (NT, HV)
    _isf_args = (
        q.reshape((-1,)),
        k.reshape((-1,)),
        gk.reshape(-1),
        beta.reshape((-1,)),
        Aqk.reshape(-1),
        Akkd.reshape(-1),
        Akk.reshape(-1),
        float(scale),
        cu_arg,
        ci_arg,
        H,
        HV,
        K,
        BT,
        BC,
        NC,
        BKf,
        int(use_solved_diagonal),
    )
    _isf_key = (
        "chunk_kda_fwd_kernel_inter_solve_fused",
        int(H),
        int(HV),
        int(K),
        int(BT),
        int(BC),
        int(NC),
        1,
        int(use_solved_diagonal),
        str(k.dtype),
        str(dev),
    )
    autotuned_launch(chunk_kda_fwd_kernel_inter_solve_fused, _isf_key, _isf_grid, _isf_args, occ_choices=(1, 2, 4), nww_choices=(4,), stream=stream)
    w, u, qg, kg = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=Akk, q=q if disable_recompute else None, gk=gk, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, bufs=bufs, stream=stream
    )
    return w, u, qg, kg, Aqk, Akk


# --- Launchers: state scan ------------------------------------------------------------------------


def chunk_gated_delta_rule_fwd_h(
    k,
    w,
    u,
    g=None,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_indices=None,
    bufs=None,
    stream=None,
):
    stream = 0 if stream is None else stream
    T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[1]
    BT = chunk_size
    N, NT = len(cu_seqlens) - 1, len(chunk_indices)
    chunk_offsets = bufs["chunk_offsets"]
    # Full-width state/K tile: BK = next_pow2(K). The blockdim64 kernel carries the
    # KV state as a single (BV, BK)/(BK, BV) tile (no K<=256 cap); K-axis loads
    # zero-pad [K:BK] and stores drop it, so any K is supported.
    BK = next_power_of_2(K)

    state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
    h = bufs["state_checkpoints"].reshape((NT, HV) + state_shape[2:])
    final_state = bufs["final_state"].reshape(state_shape) if output_final_state else None
    if final_state is not None:
        zero_fill(final_state, stream=stream)
    v_new = bufs["v_new"].reshape((T, HV, V)) if save_new_value else None

    dev = current_device_id()
    vnew_arg = opt(v_new, bufs, dtname(u))
    g_arg = opt(g, bufs)
    gk_arg = opt(gk, bufs)
    h0_arg = opt(initial_state, bufs)
    ht_arg = opt(final_state, bufs)
    cu_arg = cu_seqlens.reshape(-1)
    co_arg = chunk_offsets.reshape(-1)

    # Kernel uses flat element-offset gather/scatter; pass 1-D views so the
    # cuTile index-tuple rank (1) matches the array rank.
    _k_arg = k.reshape(-1)
    _u_arg = u.reshape(-1)
    _w_arg = w.reshape(-1)
    _vnew_arg = vnew_arg.reshape(-1)
    _h_arg = h.reshape(-1)
    _h01d = h0_arg.reshape((-1,))
    _ht1d = ht_arg.reshape((-1,))
    _g1d = g_arg.reshape(-1)
    _gk1d = gk_arg.reshape(-1)

    # BV is a V-tiling block width (drives the grid V-fan-out and the V-tile
    # shapes only; the K axis is always split into fixed 64-wide blocks, so BV
    # never changes the numerics). The tuner picks the per-shape grid fill that
    # best hides the inter-chunk latency.
    def _grid_fn(bv):
        return (cdiv(V, bv), N * HV)

    def _args_fn(bv):
        return (
            _k_arg,
            _u_arg,
            _w_arg,
            _vnew_arg,
            _g1d,
            _gk1d,
            _h_arg,
            _h01d,
            _ht1d,
            cu_arg,
            co_arg,
            H,
            HV,
            K,
            V,
            BT,
            bv,
            BK,
            int(g is not None),
            int(gk is not None),
            int(initial_state is not None),
            int(output_final_state),
            int(save_new_value),
            int(state_v_first),
        )

    _h_key = (
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        int(H),
        int(HV),
        int(K),
        int(V),
        int(BT),
        int(g is not None),
        int(gk is not None),
        int(initial_state is not None),
        int(output_final_state),
        int(save_new_value),
        int(state_v_first),
        str(k.dtype),
        str(dev),
    )
    # BV candidates: divisors of the V tile that keep N>=8 on the MMA V-axis,
    # capped at <= V so a single tile is never larger than V. The grid-fill
    # tradeoff is shape-dependent, not monotone: shrinking BV multiplies the
    # V-tile CTA count but each CTA is register-bound (255 reg -> ~1 block/SM)
    # and redundantly reloads K/W/G, so past the point where the grid already
    # fills the SMs a *larger* BV wins (fewer, fatter CTAs, higher warp
    # occupancy); 16 stays first as the safe fallback when tuning fails.
    _bv_choices = tuple(bv for bv in (16, 8, 32, 64) if bv <= max(V, 8))
    if not _bv_choices:
        _bv_choices = (min(32, V),)
    autotuned_launch_bv(chunk_gated_delta_rule_fwd_kernel_h_blockdim64, _h_key, _bv_choices, _grid_fn, _args_fn, stream=stream)
    return h, v_new, final_state


def chunk_gated_delta_rule_bwd_dhu(
    q,
    k,
    w,
    do,
    dv,
    g=None,
    gk=None,
    h0=None,
    dstate_in=None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens=None,
    chunk_size: int = 64,
    chunk_indices=None,
    bufs: dict | None = None,
    stream=None,
):
    stream = 0 if stream is None else stream
    T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[1]
    BT = chunk_size
    # Full-width state/K tile: BK = next_pow2(K). The blockdim64 dhu kernel carries
    # the dH state as a single (BV, BK)/(BK, BV) tile (no K<=256 cap); K-axis loads
    # zero-pad/mask [K:BK] and stores drop it, so any K is supported. dH/dH0
    # carves stay K-wide.
    BK = next_power_of_2(K)
    N, NT = len(cu_seqlens) - 1, len(chunk_indices)
    chunk_offsets = bufs["chunk_offsets"]

    dh = bufs["dstate"].reshape((NT, HV, V, K) if state_v_first else (NT, HV, K, V))
    dh0 = bufs["dstate0"].reshape(tuple(h0.shape)) if h0 is not None else None
    dv2 = bufs["dv_dstate_u"].reshape((T, HV, V))

    BV = 64
    grid = (cdiv(V, BV), N * HV)

    ct.launch(
        stream,
        grid,
        chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
        (
            q.reshape(-1),
            k.reshape(-1),
            w.reshape(-1),
            opt(g, bufs).reshape(-1),
            opt(gk, bufs).reshape(-1),
            opt(dstate_in, bufs).reshape((-1,)),
            opt(dh0, bufs).reshape((-1,)),
            do.reshape((-1,)),
            dh.reshape(-1),
            dv.reshape(-1),
            dv2.reshape(-1),
            cu_seqlens.reshape(-1),
            chunk_offsets.reshape(-1),
            float(scale),
            H,
            HV,
            K,
            V,
            BT,
            BV,
            BK,
            int(g is not None),
            int(gk is not None),
            int(h0 is not None),
            int(dstate_in is not None),
            int(state_v_first),
        ),
    )
    return dh, dh0, dv2


# --- Launchers: attention and gradients -----------------------------------------------------------


def chunk_gla_fwd_o_gk(q, v, g, A, h, scale, state_v_first=False, cu_seqlens=None, chunk_size=64, chunk_indices=None, bufs=None, stream=None):
    stream = 0 if stream is None else stream
    T, H, K, HV, V = *q.shape, v.shape[1], v.shape[-1]
    BT = chunk_size
    NT = len(chunk_indices)
    o = bufs["o"]
    zero_fill(o, stream=stream)
    BK = min(max(next_power_of_2(K), 16), 64)
    # BV=128 when V<=128 removes the V grid-split
    # (grid dim0 -> 1), doubling work/block but halving launched blocks.
    BV = 128 if V <= 128 else 64
    dev = current_device_id()
    cu_arg = cu_seqlens.reshape(-1)
    ci_arg = chunk_indices.reshape(-1)

    grid = (cdiv(V, BV), NT, HV)

    # Kernel uses flat element-offset gather/scatter; pass 1-D views so the
    # cuTile index-tuple rank (1) matches the array rank.
    _q_arg = q.reshape((-1,))
    _v_arg = v.reshape(-1)
    _g_arg = g.reshape(-1)
    _h_arg = h.reshape(-1)
    _o_arg = o.reshape((-1,))
    _A_arg = A.reshape(-1)
    _o_args = (
        _q_arg,
        _v_arg,
        _g_arg,
        _h_arg,
        _o_arg,
        _A_arg,
        cu_arg,
        ci_arg,
        float(scale),
        H,
        HV,
        K,
        V,
        BT,
        BK,
        BV,
        int(state_v_first),
    )
    # Launch-hint autotune (occupancy x num_worker_warps) on this
    # output-projection kernel.
    _o_key = (
        "chunk_gla_fwd_kernel_o",
        int(H),
        int(HV),
        int(K),
        int(V),
        int(BT),
        int(BK),
        int(BV),
        int(state_v_first),
        str(q.dtype),
        str(dev),
    )
    autotuned_launch(chunk_gla_fwd_kernel_o, _o_key, grid, _o_args, occ_choices=(1, 2, 4), nww_choices=(4,), stream=stream)
    return o


def chunk_kda_bwd_dAv(q, k, v, do, A=None, scale=None, cu_seqlens=None, chunk_size=64, chunk_indices=None, bufs=None, stream=None):
    stream = 0 if stream is None else stream
    T, H, K, HV, V = *k.shape, do.shape[1], do.shape[-1]
    BT = chunk_size
    CONST_TILING = 64
    BK = min(max(next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(next_power_of_2(V), 16), CONST_TILING)
    NT = len(chunk_indices)

    dA = bufs["dAqk"].reshape((T, HV, BT))
    dv = bufs["dv_dAv"].reshape((T, HV, V))
    cu_arg = cu_seqlens.reshape(-1)
    ci_arg = chunk_indices.reshape(-1)
    # Kernel uses flat element-offset gather/scatter; pass 1-D views.
    ct.launch(
        stream,
        (NT, HV),
        chunk_kda_bwd_kernel_dAv,
        (
            q.reshape((-1,)),
            k.reshape((-1,)),
            v.reshape(-1),
            A.reshape(-1),
            do.reshape((-1,)),
            dv.reshape(-1),
            dA.reshape(-1),
            cu_arg,
            ci_arg,
            float(scale),
            H,
            HV,
            K,
            V,
            BT,
            BK,
            BV,
        ),
    )
    return dA, dv


def chunk_kda_bwd_wy_dqkg_fused(
    q,
    k,
    v,
    v_new,
    g,
    beta,
    A,
    h,
    do,
    dh,
    dv,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens=None,
    chunk_size: int = 64,
    chunk_indices=None,
    bufs: dict | None = None,
    stream=None,
):
    stream = 0 if stream is None else stream
    T, H, K, HV, V = *k.shape, v.shape[1], v.shape[-1]
    BT = chunk_size
    NT = len(chunk_indices)
    CONST_TILING = 64
    BK = min(max(next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(next_power_of_2(V), 16), CONST_TILING)

    dq = bufs["dq"].reshape((T, HV, K))
    dk = bufs["dk"].reshape((T, HV, K))
    dv2 = bufs["dv2"].reshape((T, HV, V))
    dg = bufs["dg"].reshape((T, HV, K))
    db = bufs["db"].reshape((T, HV))
    dA = bufs["dAkk"].reshape((T, HV, BT))

    grid = (NT, HV)

    ct.launch(
        stream,
        grid,
        chunk_kda_bwd_kernel_wy_dqkg_fused,
        (
            q.reshape((-1,)),
            k.reshape((-1,)),
            v.reshape((-1,)),
            v_new.reshape(-1),
            g.reshape(-1),
            beta.reshape((-1,)),
            A.reshape(-1),
            h.reshape(-1),
            do.reshape((-1,)),
            dh.reshape(-1),
            dq.reshape(-1),
            dk.reshape(-1),
            dv.reshape(-1),
            dv2.reshape(-1),
            dg.reshape(-1),
            db.reshape(-1),
            dA.reshape(-1),
            cu_seqlens.reshape(-1),
            chunk_indices.reshape(-1),
            float(scale),
            H,
            HV,
            K,
            V,
            BT,
            BK,
            BV,
            int(state_v_first),
        ),
    )
    dv = dv2
    return dq, dk, dv, db, dg, dA


def chunk_kda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens=None, chunk_indices=None, chunk_size=64, safe_gate=False, bufs=None, stream=None):
    stream = 0 if stream is None else stream
    T, H, K, HV = *k.shape, g.shape[1]
    BT = chunk_size
    # Fast path: for BT >= 64 use larger
    # BC=32 / BK=64 sub-tiles and route through the SAFE_GATE matmul branch
    # instead of the BC-iteration scalar `for j` loops. The scalar path is the
    # bwd bottleneck.
    use_fast_path = BT >= 64
    BC = 32 if use_fast_path else min(16, BT)
    BK = min(64 if use_fast_path else 32, next_power_of_2(K))
    safe_gate = safe_gate or use_fast_path
    NT = len(chunk_indices)
    NC = cdiv(BT, BC)
    NK = cdiv(K, BK)

    dq2 = bufs["dq2"].reshape((T, HV, K))
    dk2 = bufs["dk2"].reshape((T, HV, K))
    db2 = bufs["db2"].reshape((NK, T, HV))
    dg2 = bufs["dg2"].reshape((T, HV, K))

    cu_arg = cu_seqlens.reshape(-1)
    ci_arg = chunk_indices.reshape(-1)
    grid = (NK * NC, NT, HV)
    # Kernel uses flat element-offset gather/scatter; pass 1-D views.
    ct.launch(
        stream,
        grid,
        chunk_kda_bwd_kernel_intra,
        (
            q.reshape((-1,)),
            k.reshape((-1,)),
            g.reshape(-1),
            beta.reshape((-1,)),
            dAqk.reshape(-1),
            dAkk.reshape(-1),
            dq.reshape(-1),
            dq2.reshape(-1),
            dk.reshape(-1),
            dk2.reshape(-1),
            dg.reshape(-1),
            dg2.reshape(-1),
            db2.reshape(-1),
            cu_arg,
            ci_arg,
            T,
            H,
            HV,
            K,
            BT,
            BC,
            BK,
            NC,
            int(safe_gate),
        ),
    )
    dq = dq2
    dk = dk2
    # dBeta += sum_nk dBeta2 (fp32 acc); the fan-in NK is a compile-time constant
    sum_leading(db.reshape((T * HV,)), db2.reshape((NK, T * HV)), NK, T * HV, stream=stream, accumulate=True)
    dg = dg2
    return dq, dk, db, dg


# --- Pipelines ------------------------------------------------------------------------------------


def chunk_kda_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_indices=None,
    chunk_size=64,
    safe_gate=False,
    lower_bound=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    disable_recompute=False,
    return_intermediate_states=False,
    compute_o=True,
    cp_context=None,
    bufs=None,
    stream=None,
):
    stream = 0 if stream is None else stream
    g_org = None
    if use_gate_in_kernel:
        g_org = g
        g = kda_gate_chunk_cumsum(
            g=g_org,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
            out=bufs["g_cum"],
            bufs=bufs,
            stream=stream,
        )
    else:
        g = chunk_local_cumsum(
            g=g,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            out=bufs["g_cum"],
            stream=stream,
        )

    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute,
        bufs=bufs,
        stream=stream,
    )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        state_v_first=state_v_first,
        bufs=bufs,
        stream=stream,
    )

    o = None
    if compute_o:
        o = chunk_gla_fwd_o_gk(
            q=q,
            v=v_new,
            g=g,
            A=Aqk,
            h=h,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=state_v_first,
            bufs=bufs,
            stream=stream,
        )
    if disable_recompute is False:
        w, u, qg, kg, v_new = None, None, None, None, None
        if not return_intermediate_states:
            h = None
        if use_gate_in_kernel:
            g = None
    return o, final_state, g, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state


def chunk_kda_bwd(
    q,
    k,
    v,
    beta,
    Aqk,
    Akk,
    scale,
    initial_state,
    do,
    dstate_in,
    g=None,
    g_org=None,
    state_v_first=False,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=64,
    safe_gate=False,
    lower_bound=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    disable_recompute=False,
    cp_context=None,
    bufs=None,
    stream=None,
    **kwargs,
):
    stream = 0 if stream is None else stream
    H, HV = q.shape[1], v.shape[1]
    G = HV // H

    if disable_recompute is False:
        if use_gate_in_kernel:
            g = kda_gate_chunk_cumsum(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=RCP_LN2,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=lower_bound,
                out=bufs["g_cum"],
                bufs=bufs,
                stream=stream,
            )
        w, u, qg, kg = recompute_w_u_fwd(k=k, v=v, beta=beta, A=Akk, q=q, gk=g, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, bufs=bufs, stream=stream)
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            gk=g,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            state_v_first=state_v_first,
            bufs=bufs,
            stream=stream,
        )
    else:
        w, u, qg, kg, v_new, h = kwargs["w"], kwargs["u"], kwargs["qg"], kwargs["kg"], kwargs["v_new"], kwargs["h"]

    dAqk, dv = chunk_kda_bwd_dAv(
        q=q, k=k, v=v_new, do=do, A=Aqk, scale=scale, cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices, bufs=bufs, stream=stream
    )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=qg,
        k=kg,
        w=w,
        gk=g,
        h0=initial_state,
        dstate_in=dstate_in,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        bufs=bufs,
        stream=stream,
    )

    dq, dk, dv, db, dg, dAkk = chunk_kda_bwd_wy_dqkg_fused(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=Akk,
        h=h,
        do=do,
        dh=dh,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        bufs=bufs,
        stream=stream,
    )

    dq, dk, db, dg = chunk_kda_bwd_intra(
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        db=db,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        bufs=bufs,
        stream=stream,
    )

    # For GVA, reduce dQ and dK from [T, HV, K] back to [T, H, K]
    if HV > H:
        T_, K_ = dq.shape[0], dq.shape[-1]
        dq_r = bufs["dq_hred"].reshape((T_, H, K_))
        dk_r = bufs["dk_hred"].reshape((T_, H, K_))
        head_group_sum(dq_r, dq, T_, H, G, K_, stream=stream)
        head_group_sum(dk_r, dk, T_, H, G, K_, stream=stream)
        dq, dk = dq_r, dk_r

    dA, dbias = None, None
    dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, out=bufs["dg_cum"], stream=stream)
    if use_gate_in_kernel:
        dg, dA, dbias = kda_gate_bwd(
            g=g_org,
            A_log=A_log,
            dt_bias=dt_bias,
            dyg=dg,
            lower_bound=lower_bound,
            dg_out=bufs["dg_gate"],
            dA_out=bufs["dA_log"],
            dbias_out=bufs["ddt_bias"] if dt_bias is not None else None,
            bufs=bufs,
            stream=stream,
        )

    return dq, dk, dv, db, dg, dh0, dA, dbias


def chunk_kda_grad(
    q,
    k,
    v,
    g,
    beta,
    do,
    dstate_in=None,
    scale=None,
    initial_state=None,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=None,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_indices=None,
    chunk_size=64,
    A_log=None,
    dt_bias=None,
    bufs=None,
    stream=None,
):
    r"""KDA backward as a plain pipeline over explicit THD arguments.

    ``q``/``k``/``v``/``do`` are ``[total_T, H, D]``, ``g`` is
    ``[total_T, HV, K]``, ``beta`` is ``[total_T, HV]``; ``cu_seqlens`` and
    ``chunk_indices`` are required. Recomputes the forward's prep
    (L2-normalized Q/K + rstd, cumulative gate, WY factors, per-chunk
    states) from the inputs, then runs the backward kernels; intermediates
    live in the ``bufs`` carves when provided.

    Returns ``(dq, dk, dv, dg, dbeta, dh0, dA_log, ddt_bias)`` in THD layout
    (the last three ``None`` unless the corresponding input/feature is
    present).
    """
    stream = 0 if stream is None else stream
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q_rstd, k_rstd = None, None
    q_in, k_in = q, k
    if use_qk_l2norm_in_kernel:
        q_in, q_rstd = l2norm_fwd(q, out=bufs["q_norm"], rstd_out=bufs["q_rstd"], stream=stream)
        k_in, k_rstd = l2norm_fwd(k, out=bufs["k_norm"], rstd_out=bufs["k_rstd"], stream=stream)
    beta_raw = beta
    if use_beta_sigmoid_in_kernel:
        beta = fused_beta_sigmoid(beta_raw, scale=2.0 if allow_neg_eigval else 1.0, out=bufs["beta_sig"], stream=stream)
    _o, _fs, g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state = chunk_kda_fwd(
        q=q_in,
        k=k_in,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        chunk_size=chunk_size,
        state_v_first=state_v_first,
        disable_recompute=True,
        compute_o=False,
        bufs=bufs,
        stream=stream,
    )
    dq, dk, dv, db, dg, dh0, dA, dbias = chunk_kda_bwd(
        q=q_in,
        k=k_in,
        v=v,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        initial_state=initial_state,
        do=do,
        dstate_in=dstate_in,
        g=g_cumsum,
        g_org=g if use_gate_in_kernel else None,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        disable_recompute=True,
        bufs=bufs,
        w=w,
        u=u,
        qg=qg,
        kg=kg,
        v_new=v_new,
        h=h,
        stream=stream,
    )
    if use_qk_l2norm_in_kernel:
        dq = l2norm_bwd(q_in, q_rstd, dq, out=bufs["dq_l2"], bufs=bufs, stream=stream)
        dk = l2norm_bwd(k_in, k_rstd, dk, out=bufs["dk_l2"], bufs=bufs, stream=stream)
    if use_beta_sigmoid_in_kernel:
        db = fused_beta_sigmoid_bwd(beta_raw, db, scale=2.0 if allow_neg_eigval else 1.0, out=db, stream=stream)
    return (
        cast(bufs, "dq_cast", dq, q),
        cast(bufs, "dk_cast", dk, k),
        cast(bufs, "dv_cast", dv, v),
        cast(bufs, "dg_cast", dg, g),
        cast(bufs, "db_cast", db, beta_raw),
        dh0,
        dA,
        dbias,
    )


def chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=None,
    disable_recompute=False,
    return_intermediate_states=False,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_indices=None,
    cp_context=None,
    bufs=None,
    stream=None,
    **kwargs,
):
    r"""Chunked Kimi Delta Attention (KDA) over THD (token-packed) inputs.

    ``q``/``k``/``v`` are ``[total_T, H, D]``, ``g`` is ``[total_T, HV, K]``,
    ``beta`` is ``[total_T, HV]``; ``cu_seqlens``, ``chunk_indices`` and the
    pre-carved ``bufs`` views are required. Returns ``(o, final_state)`` in
    THD layout, written into ``bufs['o']`` / ``bufs['fs']``."""
    stream = 0 if stream is None else stream
    if "transpose_state_layout" in kwargs:
        state_v_first = kwargs.pop("transpose_state_layout")

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")

    chunk_size = kwargs.pop("chunk_size", 64)

    if safe_gate and use_gate_in_kernel:
        if not (-5 <= lower_bound < 0):
            raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")

    T, H, K, HV = *q.shape, v.shape[1]

    if scale is None:
        scale = K**-0.5

    q_in, k_in = q, k
    if use_qk_l2norm_in_kernel:
        q_in, _q_rstd = l2norm_fwd(q, out=bufs["q_norm"], rstd_out=bufs["q_rstd"], stream=stream)
        k_in, _k_rstd = l2norm_fwd(k, out=bufs["k_norm"], rstd_out=bufs["k_rstd"], stream=stream)
    if use_beta_sigmoid_in_kernel:
        beta = fused_beta_sigmoid(beta, scale=2.0 if allow_neg_eigval else 1.0, out=bufs["beta_sig"], stream=stream)
    o, final_state, _gc, _Aqk, _Akk, _w, _u, _qg, _kg, _vn, h, _s0 = chunk_kda_fwd(
        q=q_in,
        k=k_in,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        chunk_size=chunk_size,
        disable_recompute=disable_recompute,
        return_intermediate_states=return_intermediate_states,
        cp_context=cp_context,
        state_v_first=state_v_first,
        bufs=bufs,
        stream=stream,
    )
    if return_intermediate_states:
        return o, final_state, h
    return o, final_state
