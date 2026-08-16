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


import cuda.tile as ct

from .common import (
    add_inplace,
    autotuned_launch,
    cdiv,
    ct_min,
    dummy,
    exp,
    exp2,
    fused_beta_sigmoid,
    head_group_sum,
    l2norm_bwd,
    l2norm_fwd,
    next_power_of_2,
    opt,
    softplus,
    sum_leading,
    tf32,
    zero_fill,
)
from cudnn.frost.buffers import current_device_id, dtype_name as dtname

ConstInt = ct.Constant[int]

RCP_LN2 = 1.4426950216  # 1/ln(2)

# chunk size (BT tile) of these kernels; the engine's carve layout imports it
BT_CHUNK = 64


# --- Host helpers ---------------------------------------------------------------------------------


def device_attrs():
    """(sm_count, cc_major) of the current device via the runtime API (which
    auto-inits the primary context, so this works before any framework call)."""
    try:
        from cuda.bindings import runtime as rt

        err, dev = rt.cudaGetDevice()
        if int(err) != 0:
            return 0, 0
        err, sm = rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrMultiProcessorCount, dev)
        err2, major = rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, dev)
        return (int(sm) if int(err) == 0 else 0), (int(major) if int(err2) == 0 else 0)
    except Exception:  # noqa: BLE001
        return 0, 0


# --- Launch tuning --------------------------------------------------------------------------------


# occupancy=2 is EXCLUDED: on tileiras 13.2 a freshly compiled occ=2
# chunk_bwd_kernel_dqkwg deadlocks on its first launch (occ=1/4 are fine).
TUNE_OCC = (1, 4)


# --- Device helpers -------------------------------------------------------------------------------


def safe_matmul(a, b):
    return ct.matmul(tf32(a), tf32(b))


safe_dot = safe_matmul  # alias: non-accumulating dot with fp32->tf32 guard


def safe_mma(a, b, acc):
    return ct.mma(tf32(a), tf32(b), acc)


def gather_flat(raw, off, *, mask=None, padding_value=0, check_bounds=False):
    return raw.load_offset(off, mask=mask, padding_value=padding_value)


def scatter_flat(raw, off, value, *, mask=None, check_bounds=False):
    raw.store_offset(off, value, mask=mask)


def array_numel(arr):
    # Total element count; rank statically unrolled (<=5) since a Python for over
    # arr.shape is captured as a device loop and rejected. K<=256 bounds the rank.
    s = arr.shape
    nd = arr.ndim
    n = s[0]
    if nd > 1:
        n = n * s[1]
    if nd > 2:
        n = n * s[2]
    if nd > 3:
        n = n * s[3]
    if nd > 4:
        n = n * s[4]
    return n


# Bounds-checked flat access: 1-D bounds mask `0 <= off < numel` AND-ed with any
# caller mask. Pass `numel = array_numel(arr)`.
def gather_flat_cb(raw, off, numel, *, mask=None, padding_value=0):
    inb = (off >= 0) & (off < numel)
    m = inb if mask is None else (mask & inb)
    return raw.load_offset(off, mask=m, padding_value=padding_value)


def scatter_flat_cb(raw, off, value, numel, *, mask=None):
    inb = (off >= 0) & (off < numel)
    m = inb if mask is None else (mask & inb)
    raw.store_offset(off, value, mask=m)


# --- Kernels: normalization and gates -------------------------------------------------------------


@ct.kernel
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    H: ConstInt,
    BT: ConstInt,
    REVERSE: ConstInt,
    HAS_SCALE: ConstInt,
):
    i_t = ct.bid(0)
    i_h = ct.bid(1)

    i_n = ct.load(chunk_indices, (i_t, 0), shape=()).item()
    i_t = ct.load(chunk_indices, (i_t, 1), shape=()).item()
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T_eff = eos - bos

    offs = ct.arange(BT, dtype=ct.int32)
    row = i_t * BT + offs
    mask = row < T_eff
    t_idx = bos + row

    b_s = ct.astype(ct.gather(s, (t_idx, i_h), mask=mask, check_bounds=False, padding_value=0.0), ct.float32)
    b_o = ct.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = ct.sum(b_s, axis=0)
        b_o = -b_o + b_z + b_s
    if HAS_SCALE:
        b_o = b_o * scale
    ct.scatter(o, (t_idx, i_h), ct.astype(b_o, o.dtype), mask=mask, check_bounds=False)


@ct.kernel
def gdn_gate_chunk_cumsum_scalar_kernel(
    g,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    H: ConstInt,
    BT: ConstInt,
    REVERSE: ConstInt,
    HAS_BIAS: ConstInt,
    HAS_SCALE: ConstInt,
):
    i_t = ct.bid(0)
    i_h = ct.bid(1)

    i_n = ct.load(chunk_indices, (i_t, 0), shape=()).item()
    i_t = ct.load(chunk_indices, (i_t, 1), shape=()).item()
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T_eff = eos - bos

    offs = ct.arange(BT, dtype=ct.int32)
    row = i_t * BT + offs
    mask = row < T_eff
    t_idx = bos + row

    b_g = ct.astype(ct.gather(g, (t_idx, i_h), mask=mask, check_bounds=False, padding_value=0.0), ct.float32)
    b_A = ct.astype(ct.load(A_log, (i_h,), shape=()).item(), ct.float32)
    if HAS_BIAS:
        b_bias = ct.astype(ct.load(dt_bias, (i_h,), shape=()).item(), ct.float32)
        b_g = b_g + b_bias

    b_gate = -exp(b_A) * softplus(b_g)
    b_o = ct.cumsum(b_gate, axis=0)
    if REVERSE:
        b_z = ct.sum(b_gate, axis=0)
        b_o = -b_o + b_z + b_gate
    if HAS_SCALE:
        b_o = b_o * scale
    ct.scatter(o, (t_idx, i_h), ct.astype(b_o, o.dtype), mask=mask, check_bounds=False)


@ct.kernel
def gdn_gate_bwd_kernel(g, A_log, dt_bias, dyg, dg, dA, db, T, H: ConstInt, BT: ConstInt, HAS_BIAS: ConstInt):
    i_t = ct.bid(0)
    i_h = ct.bid(1)

    b_A = ct.astype(ct.load(A_log, (i_h,), shape=()).item(), ct.float32)
    offs = ct.arange(BT, dtype=ct.int32)
    row = i_t * BT + offs
    mask = row < T
    idx = i_h + row * H

    b_g = ct.astype(ct.gather(g, idx, mask=mask, check_bounds=False, padding_value=0.0), ct.float32)
    b_dyg = ct.astype(ct.gather(dyg, idx, mask=mask, check_bounds=False, padding_value=0.0), ct.float32)
    if HAS_BIAS:
        b_bias = ct.astype(ct.load(dt_bias, (i_h,), shape=()).item(), ct.float32)
        b_g = b_g + b_bias

    b_neg_expA = -exp(b_A)
    b_yg = b_neg_expA * softplus(b_g)
    b_sig = 1.0 / (1.0 + ct.exp(-b_g))
    b_dg = b_neg_expA * (b_dyg * b_sig)
    b_dA = ct.sum(b_dyg * b_yg, axis=0)

    ct.scatter(dg, idx, ct.astype(b_dg, dg.dtype), mask=mask, check_bounds=False)
    ct.scatter(dA, i_t * H + i_h, b_dA)
    if HAS_BIAS:
        # b_dg is zero on masked lanes (b_dyg gathers with padding_value=0)
        b_db = ct.sum(b_dg, axis=0)
        ct.scatter(db, i_t * H + i_h, b_db)


# --- Kernels: WY representation -------------------------------------------------------------------


@ct.kernel
def chunk_gated_delta_rule_fwd_kkt_solve_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BC: ConstInt,
    BK: ConstInt,
    USE_G: ConstInt,
):
    """Fused: Beta * K @ K^T (lower triangular) + solve_tril (I+A)^{-1} in one pass."""
    i_t = ct.bid(0)
    i_h = ct.bid(1)

    i_n = ct.gather(chunk_indices, (i_t, 0), check_bounds=False).item()
    i_t = ct.gather(chunk_indices, (i_t, 1), check_bounds=False).item()
    bos = ct.gather(cu_seqlens, (i_n,), check_bounds=False).item()
    eos = ct.gather(cu_seqlens, (i_n + 1,), check_bounds=False).item()
    T = eos - bos

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    # Layouts: k (T,H,K)  beta/g (T,HV)  A (T,HV,BT), token-packed THD.
    i_kh = i_h // (HV // H)
    t_base = bos

    o_i = ct.arange(BC, dtype=ct.int32)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_b0 = ct.astype(
        ct.gather(beta, (t_base + i_tc0 + o_i, i_h), mask=m_tc0, check_bounds=False, padding_value=0.0),
        ct.float32,
    )
    b_b1 = ct.astype(
        ct.gather(beta, (t_base + i_tc1 + o_i, i_h), mask=m_tc1, check_bounds=False, padding_value=0.0),
        ct.float32,
    )
    b_b2 = ct.astype(
        ct.gather(beta, (t_base + i_tc2 + o_i, i_h), mask=m_tc2, check_bounds=False, padding_value=0.0),
        ct.float32,
    )
    b_b3 = ct.astype(
        ct.gather(beta, (t_base + i_tc3 + o_i, i_h), mask=m_tc3, check_bounds=False, padding_value=0.0),
        ct.float32,
    )

    b_g0 = ct.zeros((BC,), dtype=ct.float32)
    b_g1 = ct.zeros((BC,), dtype=ct.float32)
    b_g2 = ct.zeros((BC,), dtype=ct.float32)
    b_g3 = ct.zeros((BC,), dtype=ct.float32)
    if USE_G:
        b_g0 = ct.astype(
            ct.gather(g, (t_base + i_tc0 + o_i, i_h), mask=m_tc0, check_bounds=False, padding_value=0.0),
            ct.float32,
        )
        b_g1 = ct.astype(
            ct.gather(g, (t_base + i_tc1 + o_i, i_h), mask=m_tc1, check_bounds=False, padding_value=0.0),
            ct.float32,
        )
        b_g2 = ct.astype(
            ct.gather(g, (t_base + i_tc2 + o_i, i_h), mask=m_tc2, check_bounds=False, padding_value=0.0),
            ct.float32,
        )
        b_g3 = ct.astype(
            ct.gather(g, (t_base + i_tc3 + o_i, i_h), mask=m_tc3, check_bounds=False, padding_value=0.0),
            ct.float32,
        )

    # -- Step 1: 10 lower-tri [BC,BC] blocks of K @ K^T --
    b_A00 = ct.zeros((BC, BC), dtype=ct.float32)
    b_A11 = ct.zeros((BC, BC), dtype=ct.float32)
    b_A22 = ct.zeros((BC, BC), dtype=ct.float32)
    b_A33 = ct.zeros((BC, BC), dtype=ct.float32)
    b_A10 = ct.zeros((BC, BC), dtype=ct.float32)
    b_A20 = ct.zeros((BC, BC), dtype=ct.float32)
    b_A21 = ct.zeros((BC, BC), dtype=ct.float32)
    b_A30 = ct.zeros((BC, BC), dtype=ct.float32)
    b_A31 = ct.zeros((BC, BC), dtype=ct.float32)
    b_A32 = ct.zeros((BC, BC), dtype=ct.float32)

    o_bc = ct.arange(BC, dtype=ct.int32)
    num_k = ct.cdiv(K, BK)
    for i_k in range(num_k):
        off_k = i_k * BK
        cols = off_k + ct.arange(BK, dtype=ct.int32)
        m_cols = cols < K
        rows0 = i_tc0 + o_bc
        m0 = ct.expand_dims(rows0 < T, 1) & ct.expand_dims(m_cols, 0)
        b_k0 = ct.gather(
            k,
            (ct.expand_dims(t_base + rows0, 1), i_kh, ct.expand_dims(cols, 0)),
            mask=m0,
            check_bounds=False,
            padding_value=0.0,
        )
        b_A00 = ct.mma(b_k0, ct.transpose(b_k0), b_A00)

        if i_tc1 < T:
            rows1 = i_tc1 + o_bc
            m1 = ct.expand_dims(rows1 < T, 1) & ct.expand_dims(m_cols, 0)
            b_k1 = ct.gather(
                k,
                (ct.expand_dims(t_base + rows1, 1), i_kh, ct.expand_dims(cols, 0)),
                mask=m1,
                check_bounds=False,
                padding_value=0.0,
            )
            b_A11 = ct.mma(b_k1, ct.transpose(b_k1), b_A11)
            b_A10 = ct.mma(b_k1, ct.transpose(b_k0), b_A10)

            if i_tc2 < T:
                rows2 = i_tc2 + o_bc
                m2 = ct.expand_dims(rows2 < T, 1) & ct.expand_dims(m_cols, 0)
                b_k2 = ct.gather(
                    k,
                    (ct.expand_dims(t_base + rows2, 1), i_kh, ct.expand_dims(cols, 0)),
                    mask=m2,
                    check_bounds=False,
                    padding_value=0.0,
                )
                b_A22 = ct.mma(b_k2, ct.transpose(b_k2), b_A22)
                b_A20 = ct.mma(b_k2, ct.transpose(b_k0), b_A20)
                b_A21 = ct.mma(b_k2, ct.transpose(b_k1), b_A21)

                if i_tc3 < T:
                    rows3 = i_tc3 + o_bc
                    m3 = ct.expand_dims(rows3 < T, 1) & ct.expand_dims(m_cols, 0)
                    b_k3 = ct.gather(
                        k,
                        (ct.expand_dims(t_base + rows3, 1), i_kh, ct.expand_dims(cols, 0)),
                        mask=m3,
                        check_bounds=False,
                        padding_value=0.0,
                    )
                    b_A33 = ct.mma(b_k3, ct.transpose(b_k3), b_A33)
                    b_A30 = ct.mma(b_k3, ct.transpose(b_k0), b_A30)
                    b_A31 = ct.mma(b_k3, ct.transpose(b_k1), b_A31)
                    b_A32 = ct.mma(b_k3, ct.transpose(b_k2), b_A32)

    # -- Step 2: Gate + Beta scaling --
    m_d = ct.expand_dims(o_i, 1) > ct.expand_dims(o_i, 0)
    m_I = ct.expand_dims(o_i, 1) == ct.expand_dims(o_i, 0)

    if USE_G:
        b_A00 = b_A00 * ct.where(
            m_d & ct.expand_dims(m_tc0, 1) & ct.expand_dims(m_tc0, 0),
            ct.exp2(ct.astype(ct.expand_dims(b_g0, 1) - ct.expand_dims(b_g0, 0), ct.float32)),
            0.0,
        )
        b_A11 = b_A11 * ct.where(
            m_d & ct.expand_dims(m_tc1, 1) & ct.expand_dims(m_tc1, 0),
            ct.exp2(ct.astype(ct.expand_dims(b_g1, 1) - ct.expand_dims(b_g1, 0), ct.float32)),
            0.0,
        )
        b_A22 = b_A22 * ct.where(
            m_d & ct.expand_dims(m_tc2, 1) & ct.expand_dims(m_tc2, 0),
            ct.exp2(ct.astype(ct.expand_dims(b_g2, 1) - ct.expand_dims(b_g2, 0), ct.float32)),
            0.0,
        )
        b_A33 = b_A33 * ct.where(
            m_d & ct.expand_dims(m_tc3, 1) & ct.expand_dims(m_tc3, 0),
            ct.exp2(ct.astype(ct.expand_dims(b_g3, 1) - ct.expand_dims(b_g3, 0), ct.float32)),
            0.0,
        )

        b_A10 = b_A10 * ct.where(
            ct.expand_dims(m_tc1, 1) & ct.expand_dims(m_tc0, 0),
            ct.exp2(ct.astype(ct.expand_dims(b_g1, 1) - ct.expand_dims(b_g0, 0), ct.float32)),
            0.0,
        )
        b_A20 = b_A20 * ct.where(
            ct.expand_dims(m_tc2, 1) & ct.expand_dims(m_tc0, 0),
            ct.exp2(ct.astype(ct.expand_dims(b_g2, 1) - ct.expand_dims(b_g0, 0), ct.float32)),
            0.0,
        )
        b_A21 = b_A21 * ct.where(
            ct.expand_dims(m_tc2, 1) & ct.expand_dims(m_tc1, 0),
            ct.exp2(ct.astype(ct.expand_dims(b_g2, 1) - ct.expand_dims(b_g1, 0), ct.float32)),
            0.0,
        )
        b_A30 = b_A30 * ct.where(
            ct.expand_dims(m_tc3, 1) & ct.expand_dims(m_tc0, 0),
            ct.exp2(ct.astype(ct.expand_dims(b_g3, 1) - ct.expand_dims(b_g0, 0), ct.float32)),
            0.0,
        )
        b_A31 = b_A31 * ct.where(
            ct.expand_dims(m_tc3, 1) & ct.expand_dims(m_tc1, 0),
            ct.exp2(ct.astype(ct.expand_dims(b_g3, 1) - ct.expand_dims(b_g1, 0), ct.float32)),
            0.0,
        )
        b_A32 = b_A32 * ct.where(
            ct.expand_dims(m_tc3, 1) & ct.expand_dims(m_tc2, 0),
            ct.exp2(ct.astype(ct.expand_dims(b_g3, 1) - ct.expand_dims(b_g2, 0), ct.float32)),
            0.0,
        )
    else:
        b_A00 = ct.where(m_d, b_A00, 0.0)
        b_A11 = ct.where(m_d, b_A11, 0.0)
        b_A22 = ct.where(m_d, b_A22, 0.0)
        b_A33 = ct.where(m_d, b_A33, 0.0)

    b_A00 = b_A00 * ct.expand_dims(b_b0, 1)
    b_A11 = b_A11 * ct.expand_dims(b_b1, 1)
    b_A22 = b_A22 * ct.expand_dims(b_b2, 1)
    b_A33 = b_A33 * ct.expand_dims(b_b3, 1)
    b_A10 = b_A10 * ct.expand_dims(b_b1, 1)
    b_A20 = b_A20 * ct.expand_dims(b_b2, 1)
    b_A21 = b_A21 * ct.expand_dims(b_b2, 1)
    b_A30 = b_A30 * ct.expand_dims(b_b3, 1)
    b_A31 = b_A31 * ct.expand_dims(b_b3, 1)
    b_A32 = b_A32 * ct.expand_dims(b_b3, 1)

    # -- Step 3: forward substitution on diagonal blocks --
    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33

    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = ct.sum(ct.where(ct.expand_dims(o_i == i, 1), -b_A00, 0.0), axis=0)
        b_a00 = ct.where(o_i < i, b_a00, 0.0)
        b_a00 = b_a00 + ct.sum(ct.expand_dims(b_a00, 1) * b_Ai00, axis=0)
        b_Ai00 = ct.where(ct.expand_dims(o_i == i, 1), b_a00, b_Ai00)
    for i in range(2, min(BC, T - i_tc1)):
        b_a11 = ct.sum(ct.where(ct.expand_dims(o_i == i, 1), -b_A11, 0.0), axis=0)
        b_a11 = ct.where(o_i < i, b_a11, 0.0)
        b_a11 = b_a11 + ct.sum(ct.expand_dims(b_a11, 1) * b_Ai11, axis=0)
        b_Ai11 = ct.where(ct.expand_dims(o_i == i, 1), b_a11, b_Ai11)
    for i in range(2, min(BC, T - i_tc2)):
        b_a22 = ct.sum(ct.where(ct.expand_dims(o_i == i, 1), -b_A22, 0.0), axis=0)
        b_a22 = ct.where(o_i < i, b_a22, 0.0)
        b_a22 = b_a22 + ct.sum(ct.expand_dims(b_a22, 1) * b_Ai22, axis=0)
        b_Ai22 = ct.where(ct.expand_dims(o_i == i, 1), b_a22, b_Ai22)
    for i in range(2, min(BC, T - i_tc3)):
        b_a33 = ct.sum(ct.where(ct.expand_dims(o_i == i, 1), -b_A33, 0.0), axis=0)
        b_a33 = ct.where(o_i < i, b_a33, 0.0)
        b_a33 = b_a33 + ct.sum(ct.expand_dims(b_a33, 1) * b_Ai33, axis=0)
        b_Ai33 = ct.where(ct.expand_dims(o_i == i, 1), b_a33, b_Ai33)

    b_Ai00 = b_Ai00 + ct.astype(m_I, ct.float32)
    b_Ai11 = b_Ai11 + ct.astype(m_I, ct.float32)
    b_Ai22 = b_Ai22 + ct.astype(m_I, ct.float32)
    b_Ai33 = b_Ai33 + ct.astype(m_I, ct.float32)

    # -- Step 4: block merge --
    b_Ai10 = -ct.matmul(ct.matmul(b_Ai11, b_A10), b_Ai00)
    b_Ai21 = -ct.matmul(ct.matmul(b_Ai22, b_A21), b_Ai11)
    b_Ai32 = -ct.matmul(ct.matmul(b_Ai33, b_A32), b_Ai22)
    b_Ai20 = -ct.matmul(b_Ai22, ct.matmul(b_A20, b_Ai00) + ct.matmul(b_A21, b_Ai10))
    b_Ai31 = -ct.matmul(b_Ai33, ct.matmul(b_A31, b_Ai11) + ct.matmul(b_A32, b_Ai21))
    b_Ai30 = -ct.matmul(
        b_Ai33,
        ct.matmul(b_A30, b_Ai00) + ct.matmul(b_A31, b_Ai10) + ct.matmul(b_A32, b_Ai20),
    )

    # -- Step 5: store full (I+A)^{-1} --
    cBC = ct.arange(BC, dtype=ct.int32)
    rA0 = i_tc0 + cBC
    rA1 = i_tc1 + cBC
    rA2 = i_tc2 + cBC
    rA3 = i_tc3 + cBC

    msk_A00 = ct.expand_dims(rA0 < T, 1) & ct.expand_dims(cBC < BT, 0)
    ct.scatter(
        A,
        (ct.expand_dims(t_base + rA0, 1), i_h, ct.expand_dims(cBC, 0)),
        ct.astype(b_Ai00, A.dtype),
        mask=msk_A00,
        check_bounds=False,
    )

    msk_A10 = ct.expand_dims(rA1 < T, 1) & ct.expand_dims(cBC < BT, 0)
    ct.scatter(
        A,
        (ct.expand_dims(t_base + rA1, 1), i_h, ct.expand_dims(cBC, 0)),
        ct.astype(b_Ai10, A.dtype),
        mask=msk_A10,
        check_bounds=False,
    )

    msk_A11 = ct.expand_dims(rA1 < T, 1) & ct.expand_dims((BC + cBC) < BT, 0)
    ct.scatter(
        A,
        (ct.expand_dims(t_base + rA1, 1), i_h, ct.expand_dims(BC + cBC, 0)),
        ct.astype(b_Ai11, A.dtype),
        mask=msk_A11,
        check_bounds=False,
    )

    msk_A20 = ct.expand_dims(rA2 < T, 1) & ct.expand_dims(cBC < BT, 0)
    ct.scatter(
        A,
        (ct.expand_dims(t_base + rA2, 1), i_h, ct.expand_dims(cBC, 0)),
        ct.astype(b_Ai20, A.dtype),
        mask=msk_A20,
        check_bounds=False,
    )

    msk_A21 = ct.expand_dims(rA2 < T, 1) & ct.expand_dims((BC + cBC) < BT, 0)
    ct.scatter(
        A,
        (ct.expand_dims(t_base + rA2, 1), i_h, ct.expand_dims(BC + cBC, 0)),
        ct.astype(b_Ai21, A.dtype),
        mask=msk_A21,
        check_bounds=False,
    )

    msk_A22 = ct.expand_dims(rA2 < T, 1) & ct.expand_dims((2 * BC + cBC) < BT, 0)
    ct.scatter(
        A,
        (ct.expand_dims(t_base + rA2, 1), i_h, ct.expand_dims(2 * BC + cBC, 0)),
        ct.astype(b_Ai22, A.dtype),
        mask=msk_A22,
        check_bounds=False,
    )

    msk_A30 = ct.expand_dims(rA3 < T, 1) & ct.expand_dims(cBC < BT, 0)
    ct.scatter(
        A,
        (ct.expand_dims(t_base + rA3, 1), i_h, ct.expand_dims(cBC, 0)),
        ct.astype(b_Ai30, A.dtype),
        mask=msk_A30,
        check_bounds=False,
    )

    msk_A31 = ct.expand_dims(rA3 < T, 1) & ct.expand_dims((BC + cBC) < BT, 0)
    ct.scatter(
        A,
        (ct.expand_dims(t_base + rA3, 1), i_h, ct.expand_dims(BC + cBC, 0)),
        ct.astype(b_Ai31, A.dtype),
        mask=msk_A31,
        check_bounds=False,
    )

    msk_A32 = ct.expand_dims(rA3 < T, 1) & ct.expand_dims((2 * BC + cBC) < BT, 0)
    ct.scatter(
        A,
        (ct.expand_dims(t_base + rA3, 1), i_h, ct.expand_dims(2 * BC + cBC, 0)),
        ct.astype(b_Ai32, A.dtype),
        mask=msk_A32,
        check_bounds=False,
    )

    msk_A33 = ct.expand_dims(rA3 < T, 1) & ct.expand_dims((3 * BC + cBC) < BT, 0)
    ct.scatter(
        A,
        (ct.expand_dims(t_base + rA3, 1), i_h, ct.expand_dims(3 * BC + cBC, 0)),
        ct.astype(b_Ai33, A.dtype),
        mask=msk_A33,
        check_bounds=False,
    )


@ct.kernel
def recompute_w_u_fwd_kernel(
    k,  # [T, H, K]
    v,  # [T, HV, V]
    beta,  # [T, HV]
    w,  # [T, HV, K] (out)
    u,  # [T, HV, V] (out)
    A,  # [T, HV, BT]
    g,  # [T, HV] (or dummy)
    cu_seqlens,
    chunk_indices,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    USE_G: ConstInt,
):
    i_t = ct.bid(0)
    i_h = ct.bid(1)
    i_kh = i_h // (HV // H)

    i_n = ct.astype(ct.load(chunk_indices, index=(i_t, 0), shape=()), ct.int32)
    i_t_loc = ct.astype(ct.load(chunk_indices, index=(i_t, 1), shape=()), ct.int32)
    bos = ct.astype(ct.load(cu_seqlens, index=(i_n,), shape=()), ct.int32)
    eos = ct.astype(ct.load(cu_seqlens, index=(i_n + 1,), shape=()), ct.int32)

    # Per-sequence contiguous slabs -> block-indexed (TMA) loads.
    k_seg = k.slice(axis=0, start=bos, stop=eos)
    v_seg = v.slice(axis=0, start=bos, stop=eos)
    w_seg = w.slice(axis=0, start=bos, stop=eos)
    u_seg = u.slice(axis=0, start=bos, stop=eos)
    A_seg = A.slice(axis=0, start=bos, stop=eos)
    beta_seg = beta.slice(axis=0, start=bos, stop=eos)

    Z = ct.PaddingMode.ZERO
    # Keep Beta native bf16: scaling in bf16 avoids 2 extra ftof
    # converts/MMA-input that an f32 cast would force.
    b_b = ct.load(beta_seg, index=(i_t_loc, i_h), shape=(BT, 1), padding_mode=Z).reshape((BT,))
    b_A = ct.load(A_seg, index=(i_t_loc, i_h, 0), shape=(BT, 1, BT), padding_mode=Z).reshape((BT, BT))

    # U = A @ (V * Beta). latency=3 -> deeper pipeline of the per-tile TMA loads.
    for i_v in range(ct.cdiv(V, BV)):
        b_v = ct.load(v_seg, index=(i_t_loc, i_h, i_v), shape=(BT, 1, BV), padding_mode=Z, latency=3).reshape((BT, BV))
        # bf16 * bf16 -> bf16.
        b_vb = ct.astype(b_v * b_b[:, None], b_v.dtype)
        acc = ct.zeros((BT, BV), dtype=ct.float32)
        b_u = safe_mma(b_A, b_vb, acc)
        ct.store(u_seg, index=(i_t_loc, i_h, i_v), tile=ct.astype(b_u, u.dtype).reshape((BT, 1, BV)))

    b_g = ct.ones((BT,), dtype=ct.float32)
    if USE_G:
        b_g_val = ct.astype(
            ct.load(g.slice(axis=0, start=bos, stop=eos), index=(i_t_loc, i_h), shape=(BT, 1), padding_mode=Z).reshape((BT,)),
            ct.float32,
        )
        b_g = exp2(b_g_val)

    # W = A @ (K * Beta * Gate)
    for i_k in range(ct.cdiv(K, BK)):
        b_k = ct.load(k_seg, index=(i_t_loc, i_kh, i_k), shape=(BT, 1, BK), padding_mode=Z, latency=3).reshape((BT, BK))
        # bf16 * bf16 -> bf16; only the g-decay path promotes to f32.
        b_kb = b_k * b_b[:, None]
        if USE_G:
            b_kb = ct.astype(b_kb, ct.float32) * b_g[:, None]
        acc = ct.zeros((BT, BK), dtype=ct.float32)
        b_w = safe_mma(b_A, ct.astype(b_kb, b_k.dtype), acc)
        ct.store(w_seg, index=(i_t_loc, i_h, i_k), tile=ct.astype(b_w, w.dtype).reshape((BT, 1, BK)))


@ct.kernel
def prepare_wy_repr_bwd_kernel(
    k,
    v,
    beta,
    g,
    A,
    dw,
    du,
    dk,
    dv,
    db,
    dg,
    cu_seqlens,
    chunk_indices,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    USE_G: ConstInt,
    # NK/NV = cdiv(K,BK)/cdiv(V,BV) as RUNTIME ints so the K/V tile loops roll
    # instead of unrolling (unrolled 4-trip loops spill regs at K=256). K/V/BK/BV
    # stay ConstInt so offset arithmetic and tile shapes stay specialized.
    NK,
    NV,
):
    k_flat = k.get_raw_memory()
    v_flat = v.get_raw_memory()
    dw_flat = dw.get_raw_memory()
    du_flat = du.get_raw_memory()
    dk_flat = dk.get_raw_memory()
    dv_flat = dv.get_raw_memory()
    beta_flat = beta.get_raw_memory()
    g_flat = g.get_raw_memory()
    A_flat = A.get_raw_memory()
    db_flat = db.get_raw_memory()
    dg_flat = dg.get_raw_memory()

    i_t = ct.bid(0)
    i_h = ct.bid(1)
    i_kh = i_h // (HV // H)

    i_n = ct.load(chunk_indices, (i_t, 0), shape=()).item()
    i_t_loc = ct.load(chunk_indices, (i_t, 1), shape=()).item()
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    Tloc = eos - bos

    t_off = i_t_loc * BT + ct.arange(BT, dtype=ct.int32)
    m_t = t_off < Tloc
    row_hv = (bos + t_off) * HV + i_h
    row_h = (bos + t_off) * H + i_kh

    b_b = ct.astype(gather_flat(beta_flat, row_hv, mask=m_t, padding_value=0.0), ct.float32)

    a_rows = row_hv[:, None]
    a_cols = ct.arange(BT, dtype=ct.int32)[None, :]
    b_A_nat = gather_flat(
        A_flat,
        ct.broadcast_to(a_rows, (BT, BT)) * BT + ct.broadcast_to(a_cols, (BT, BT)),
        mask=ct.broadcast_to(m_t[:, None], (BT, BT)),
        padding_value=0.0,
    )
    b_A = ct.transpose(b_A_nat)

    b_db = ct.zeros((BT,), dtype=ct.float32)
    b_dA = ct.zeros((BT, BT), dtype=ct.float32)
    b_g = ct.zeros((BT,), dtype=ct.float32)
    b_g_exp = ct.ones((BT,), dtype=ct.float32)
    b_dg = ct.zeros((BT,), dtype=ct.float32)
    if USE_G:
        b_g = ct.astype(gather_flat(g_flat, row_hv, mask=m_t, padding_value=0.0), ct.float32)
        b_g_exp = exp2(b_g)

    k_off = ct.arange(BK, dtype=ct.int32)
    for i_k in range(NK):
        kcols = (i_k * BK + k_off)[None, :]
        m_k = m_t[:, None] & ((i_k * BK + k_off) < K)[None, :]
        krows = row_h[:, None]
        kvrows = row_hv[:, None]
        b_k = gather_flat(
            k_flat,
            ct.broadcast_to(krows, (BT, BK)) * K + ct.broadcast_to(kcols, (BT, BK)),
            mask=m_k,
            padding_value=0.0,
        )
        b_dw = gather_flat(
            dw_flat,
            ct.broadcast_to(kvrows, (BT, BK)) * K + ct.broadcast_to(kcols, (BT, BK)),
            mask=m_k,
            padding_value=0.0,
        )
        if USE_G:
            b_kbg = ct.astype(b_k, ct.float32) * (b_b * b_g_exp)[:, None]
        else:
            b_kbg = ct.astype(b_k, ct.float32) * b_b[:, None]
        b_dA = ct.mma(b_dw, ct.astype(ct.transpose(b_kbg), b_dw.dtype), b_dA)
        b_dkbg = ct.astype(safe_matmul(b_A, b_dw), ct.float32)
        if USE_G:
            b_dk = b_dkbg * (b_g_exp * b_b)[:, None]
            b_db = b_db + ct.sum(b_dkbg * ct.astype(b_k, ct.float32) * b_g_exp[:, None], axis=1)
            b_dg = b_dg + ct.sum(b_dkbg * b_kbg, axis=1)
        else:
            b_dk = b_dkbg * b_b[:, None]
            b_db = b_db + ct.sum(b_dkbg * ct.astype(b_k, ct.float32), axis=1)
        scatter_flat(
            dk_flat,
            ct.broadcast_to(kvrows, (BT, BK)) * K + ct.broadcast_to(kcols, (BT, BK)),
            ct.astype(b_dk, dk_flat.dtype),
            mask=m_k,
        )

    v_off = ct.arange(BV, dtype=ct.int32)
    for i_v in range(NV):
        vcols = (i_v * BV + v_off)[None, :]
        m_v = m_t[:, None] & ((i_v * BV + v_off) < V)[None, :]
        vrows = row_hv[:, None]
        b_v = gather_flat(
            v_flat,
            ct.broadcast_to(vrows, (BT, BV)) * V + ct.broadcast_to(vcols, (BT, BV)),
            mask=m_v,
            padding_value=0.0,
        )
        b_du = gather_flat(
            du_flat,
            ct.broadcast_to(vrows, (BT, BV)) * V + ct.broadcast_to(vcols, (BT, BV)),
            mask=m_v,
            padding_value=0.0,
        )
        b_vb = ct.astype(ct.astype(b_v, ct.float32) * b_b[:, None], b_v.dtype)
        b_dA = ct.mma(b_du, ct.transpose(b_vb), b_dA)
        b_dvb = ct.astype(safe_matmul(b_A, b_du), ct.float32)
        b_dv = b_dvb * b_b[:, None]
        b_db = b_db + ct.sum(b_dvb * ct.astype(b_v, ct.float32), axis=1)
        scatter_flat(
            dv_flat,
            ct.broadcast_to(vrows, (BT, BV)) * V + ct.broadcast_to(vcols, (BT, BV)),
            ct.astype(b_dv, dv_flat.dtype),
            mask=m_v,
        )

    o_t = i_t_loc * BT + ct.arange(BT, dtype=ct.int32)
    m_tt = o_t < Tloc
    m_lt = (o_t[:, None] > o_t[None, :]) & (m_tt[:, None] & m_tt)
    zero_bt = ct.zeros((BT, BT), dtype=ct.float32)
    b_dA = ct.where(m_lt, b_dA, zero_bt)
    b_dA = safe_matmul(ct.astype(b_dA, b_A.dtype), b_A)
    b_dA = safe_matmul(b_A, ct.astype(b_dA, b_A.dtype))
    if USE_G:
        b_dA = b_dA * exp2(b_g[:, None] - b_g[None, :])
    b_dA = ct.astype(ct.where(m_lt, -b_dA, zero_bt), k.dtype)

    b_A2 = ct.zeros((BT, BT), dtype=ct.float32)
    for i_k in range(NK):
        kcols = (i_k * BK + k_off)[None, :]
        m_k = m_t[:, None] & ((i_k * BK + k_off) < K)[None, :]
        krows = row_h[:, None]
        kvrows = row_hv[:, None]
        b_k = gather_flat(
            k_flat,
            ct.broadcast_to(krows, (BT, BK)) * K + ct.broadcast_to(kcols, (BT, BK)),
            mask=m_k,
            padding_value=0.0,
        )
        b_dk_prev = gather_flat(
            dk_flat,
            ct.broadcast_to(kvrows, (BT, BK)) * K + ct.broadcast_to(kcols, (BT, BK)),
            mask=m_k,
            padding_value=0.0,
        )
        b_kt = ct.transpose(b_k)
        b_kb = ct.astype(b_k, ct.float32) * b_b[:, None]
        b_A2 = ct.mma(b_k, b_kt, b_A2)
        b_dkb = ct.astype(safe_matmul(b_dA, b_k), ct.float32)
        b_db = b_db + ct.sum(b_dkb * ct.astype(b_k, ct.float32), axis=1)
        b_dk = b_dkb * b_b[:, None] + ct.astype(safe_matmul(ct.transpose(b_dA), ct.astype(b_kb, b_dA.dtype)), ct.float32)
        b_dk = b_dk + ct.astype(b_dk_prev, ct.float32)
        scatter_flat(
            dk_flat,
            ct.broadcast_to(kvrows, (BT, BK)) * K + ct.broadcast_to(kcols, (BT, BK)),
            ct.astype(b_dk, dk_flat.dtype),
            mask=m_k,
        )

    scatter_flat(db_flat, row_hv, ct.astype(b_db, db_flat.dtype), mask=m_t)

    b_A2 = b_A2 * b_b[:, None]
    if USE_G:
        b_AdA = b_dA * b_A2
        b_dg = b_dg + (ct.sum(b_AdA, axis=1) - ct.sum(b_AdA, axis=0))
        scatter_flat(dg_flat, row_hv, ct.astype(b_dg, dg_flat.dtype), mask=m_t)


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
    USE_G: ConstInt,
    USE_GK: ConstInt,
    USE_INITIAL_STATE: ConstInt,
    STORE_FINAL_STATE: ConstInt,
    SAVE_NEW_VALUE: ConstInt,
    STATE_V_FIRST: ConstInt,
):
    i_v = ct.bid(0)
    i_nh = ct.bid(1)
    i_n = i_nh // HV
    i_h = i_nh % HV

    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T = eos - bos
    NT = ct.cdiv(T, BT)
    boh = ct.load(chunk_offsets, (i_n,), shape=()).item()

    # State tiles b_h1..b_h4: (BV, 64) when V-first else (64, BV).
    if STATE_V_FIRST:
        b_h1 = ct.zeros((BV, 64), dtype=ct.float32)
        b_h2 = ct.zeros((BV, 64), dtype=ct.float32)
        b_h3 = ct.zeros((BV, 64), dtype=ct.float32)
        b_h4 = ct.zeros((BV, 64), dtype=ct.float32)
    else:
        b_h1 = ct.zeros((64, BV), dtype=ct.float32)
        b_h2 = ct.zeros((64, BV), dtype=ct.float32)
        b_h3 = ct.zeros((64, BV), dtype=ct.float32)
        b_h4 = ct.zeros((64, BV), dtype=ct.float32)

    # Layouts: k (T,H,K)  v/w/v_new (T,HV,*)  g (T,HV)  gk (T,HV,K)
    #   h (NT_total,HV,*)  h0/ht (N,HV,*), all token/chunk-packed THD;
    #   h0/ht indexed by the sequence i_n.
    i_kh = i_h // (HV // H)
    t_base = bos

    o64 = ct.arange(64, dtype=ct.int32)
    o_bt = ct.arange(BT, dtype=ct.int32)
    o_bv = ct.arange(BV, dtype=ct.int32)
    m_v_lane = (i_v * BV + o_bv) < V

    # --- Load initial state H0 -> b_h1..b_h4 ---
    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            row = (i_v * BV + o_bv)[:, None]
            m1 = ct.broadcast_to(m_v_lane[:, None] & ((0 + o64) < K)[None, :], (BV, 64))
            b_h1 = b_h1 + ct.astype(
                ct.gather(h0, (i_n, i_h, row, (0 + o64)[None, :]), mask=m1, check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            if K > 64:
                m2 = ct.broadcast_to(m_v_lane[:, None] & ((64 + o64) < K)[None, :], (BV, 64))
                b_h2 = b_h2 + ct.astype(
                    ct.gather(h0, (i_n, i_h, row, (64 + o64)[None, :]), mask=m2, check_bounds=True, padding_value=0.0),
                    ct.float32,
                )
            if K > 128:
                m3 = ct.broadcast_to(m_v_lane[:, None] & ((128 + o64) < K)[None, :], (BV, 64))
                b_h3 = b_h3 + ct.astype(
                    ct.gather(h0, (i_n, i_h, row, (128 + o64)[None, :]), mask=m3, check_bounds=True, padding_value=0.0),
                    ct.float32,
                )
            if K > 192:
                m4 = ct.broadcast_to(m_v_lane[:, None] & ((192 + o64) < K)[None, :], (BV, 64))
                b_h4 = b_h4 + ct.astype(
                    ct.gather(h0, (i_n, i_h, row, (192 + o64)[None, :]), mask=m4, check_bounds=True, padding_value=0.0),
                    ct.float32,
                )
        else:
            col = (i_v * BV + o_bv)[None, :]
            m1 = ct.broadcast_to(((0 + o64) < K)[:, None] & m_v_lane[None, :], (64, BV))
            b_h1 = b_h1 + ct.astype(
                ct.gather(h0, (i_n, i_h, (0 + o64)[:, None], col), mask=m1, check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            if K > 64:
                m2 = ct.broadcast_to(((64 + o64) < K)[:, None] & m_v_lane[None, :], (64, BV))
                b_h2 = b_h2 + ct.astype(
                    ct.gather(h0, (i_n, i_h, (64 + o64)[:, None], col), mask=m2, check_bounds=True, padding_value=0.0),
                    ct.float32,
                )
            if K > 128:
                m3 = ct.broadcast_to(((128 + o64) < K)[:, None] & m_v_lane[None, :], (64, BV))
                b_h3 = b_h3 + ct.astype(
                    ct.gather(h0, (i_n, i_h, (128 + o64)[:, None], col), mask=m3, check_bounds=True, padding_value=0.0),
                    ct.float32,
                )
            if K > 192:
                m4 = ct.broadcast_to(((192 + o64) < K)[:, None] & m_v_lane[None, :], (64, BV))
                b_h4 = b_h4 + ct.astype(
                    ct.gather(h0, (i_n, i_h, (192 + o64)[:, None], col), mask=m4, check_bounds=True, padding_value=0.0),
                    ct.float32,
                )

    # --- Recurrent state-carry loop over chunks (kept as Python for) ---
    for i_t in range(NT):
        hc_idx = boh + i_t

        # Store current state b_h* to h[.., i_t, ..]
        if STATE_V_FIRST:
            row = (i_v * BV + o_bv)[:, None]
            hm1 = ct.broadcast_to(m_v_lane[:, None] & ((0 + o64) < K)[None, :], (BV, 64))
            ct.scatter(
                h,
                (hc_idx, i_h, row, (0 + o64)[None, :]),
                ct.astype(b_h1, h.dtype),
                mask=hm1,
                check_bounds=True,
            )
            if K > 64:
                hm2 = ct.broadcast_to(m_v_lane[:, None] & ((64 + o64) < K)[None, :], (BV, 64))
                ct.scatter(
                    h,
                    (hc_idx, i_h, row, (64 + o64)[None, :]),
                    ct.astype(b_h2, h.dtype),
                    mask=hm2,
                    check_bounds=True,
                )
            if K > 128:
                hm3 = ct.broadcast_to(m_v_lane[:, None] & ((128 + o64) < K)[None, :], (BV, 64))
                ct.scatter(
                    h,
                    (hc_idx, i_h, row, (128 + o64)[None, :]),
                    ct.astype(b_h3, h.dtype),
                    mask=hm3,
                    check_bounds=True,
                )
            if K > 192:
                hm4 = ct.broadcast_to(m_v_lane[:, None] & ((192 + o64) < K)[None, :], (BV, 64))
                ct.scatter(
                    h,
                    (hc_idx, i_h, row, (192 + o64)[None, :]),
                    ct.astype(b_h4, h.dtype),
                    mask=hm4,
                    check_bounds=True,
                )
        else:
            col = (i_v * BV + o_bv)[None, :]
            hm1 = ct.broadcast_to(((0 + o64) < K)[:, None] & m_v_lane[None, :], (64, BV))
            ct.scatter(
                h,
                (hc_idx, i_h, (0 + o64)[:, None], col),
                ct.astype(b_h1, h.dtype),
                mask=hm1,
                check_bounds=True,
            )
            if K > 64:
                hm2 = ct.broadcast_to(((64 + o64) < K)[:, None] & m_v_lane[None, :], (64, BV))
                ct.scatter(
                    h,
                    (hc_idx, i_h, (64 + o64)[:, None], col),
                    ct.astype(b_h2, h.dtype),
                    mask=hm2,
                    check_bounds=True,
                )
            if K > 128:
                hm3 = ct.broadcast_to(((128 + o64) < K)[:, None] & m_v_lane[None, :], (64, BV))
                ct.scatter(
                    h,
                    (hc_idx, i_h, (128 + o64)[:, None], col),
                    ct.astype(b_h3, h.dtype),
                    mask=hm3,
                    check_bounds=True,
                )
            if K > 192:
                hm4 = ct.broadcast_to(((192 + o64) < K)[:, None] & m_v_lane[None, :], (64, BV))
                ct.scatter(
                    h,
                    (hc_idx, i_h, (192 + o64)[:, None], col),
                    ct.astype(b_h4, h.dtype),
                    mask=hm4,
                    check_bounds=True,
                )

        # --- V = W @ H (accumulated over K blocks) ---
        w_time = (t_base + i_t * BT + o_bt)[:, None]
        wmask_r = (i_t * BT + o_bt) < T
        bw1 = ct.gather(
            w,
            (w_time, i_h, (0 + o64)[None, :]),
            mask=ct.broadcast_to(wmask_r[:, None] & ((0 + o64) < K)[None, :], (BT, 64)),
            check_bounds=True,
            padding_value=0.0,
        )
        bw1 = ct.where(wmask_r[:, None], bw1, ct.zeros((BT, 64), dtype=bw1.dtype))
        b_v = ct.zeros((BT, BV), dtype=ct.float32)
        if STATE_V_FIRST:
            a = bw1
            bmat = ct.astype(ct.transpose(b_h1), bw1.dtype)
        else:
            a = bw1
            bmat = ct.astype(b_h1, bw1.dtype)
        a = ct.astype(a, ct.tfloat32) if a.dtype == ct.float32 else a
        bmat = ct.astype(bmat, ct.tfloat32) if bmat.dtype == ct.float32 else bmat
        b_v = ct.mma(a, bmat, b_v)
        if K > 64:
            bw2 = ct.gather(
                w,
                (w_time, i_h, (64 + o64)[None, :]),
                mask=ct.broadcast_to(wmask_r[:, None] & ((64 + o64) < K)[None, :], (BT, 64)),
                check_bounds=True,
                padding_value=0.0,
            )
            bw2 = ct.where(wmask_r[:, None], bw2, ct.zeros((BT, 64), dtype=bw2.dtype))
            bmat = ct.astype(ct.transpose(b_h2), bw2.dtype) if STATE_V_FIRST else ct.astype(b_h2, bw2.dtype)
            a = ct.astype(bw2, ct.tfloat32) if bw2.dtype == ct.float32 else bw2
            bmat = ct.astype(bmat, ct.tfloat32) if bmat.dtype == ct.float32 else bmat
            b_v = ct.mma(a, bmat, b_v)
        if K > 128:
            bw3 = ct.gather(
                w,
                (w_time, i_h, (128 + o64)[None, :]),
                mask=ct.broadcast_to(wmask_r[:, None] & ((128 + o64) < K)[None, :], (BT, 64)),
                check_bounds=True,
                padding_value=0.0,
            )
            bw3 = ct.where(wmask_r[:, None], bw3, ct.zeros((BT, 64), dtype=bw3.dtype))
            bmat = ct.astype(ct.transpose(b_h3), bw3.dtype) if STATE_V_FIRST else ct.astype(b_h3, bw3.dtype)
            a = ct.astype(bw3, ct.tfloat32) if bw3.dtype == ct.float32 else bw3
            bmat = ct.astype(bmat, ct.tfloat32) if bmat.dtype == ct.float32 else bmat
            b_v = ct.mma(a, bmat, b_v)
        if K > 192:
            bw4 = ct.gather(
                w,
                (w_time, i_h, (192 + o64)[None, :]),
                mask=ct.broadcast_to(wmask_r[:, None] & ((192 + o64) < K)[None, :], (BT, 64)),
                check_bounds=True,
                padding_value=0.0,
            )
            bw4 = ct.where(wmask_r[:, None], bw4, ct.zeros((BT, 64), dtype=bw4.dtype))
            bmat = ct.astype(ct.transpose(b_h4), bw4.dtype) if STATE_V_FIRST else ct.astype(b_h4, bw4.dtype)
            a = ct.astype(bw4, ct.tfloat32) if bw4.dtype == ct.float32 else bw4
            bmat = ct.astype(bmat, ct.tfloat32) if bmat.dtype == ct.float32 else bmat
            b_v = ct.mma(a, bmat, b_v)

        # b_v = load(v) - b_v
        v_col = (i_v * BV + o_bv)[None, :]
        vmask_c = (i_v * BV + o_bv) < V
        v_time = (t_base + i_t * BT + o_bt)[:, None]
        v_full = (wmask_r[:, None]) & (vmask_c[None, :])
        b_v_load = ct.gather(
            v,
            (v_time, i_h, v_col),
            mask=ct.broadcast_to(v_full, (BT, BV)),
            check_bounds=True,
            padding_value=0.0,
        )
        b_v_load = ct.where(v_full, b_v_load, ct.zeros((BT, BV), dtype=b_v_load.dtype))
        b_v = ct.astype(b_v_load, ct.float32) - b_v

        if SAVE_NEW_VALUE:
            ct.scatter(
                v_new,
                (v_time, i_h, v_col),
                ct.astype(b_v, v_new.dtype),
                mask=ct.broadcast_to(v_full, (BT, BV)),
                check_bounds=True,
            )

        last_idx = ct_min((i_t + 1) * BT, T) - 1

        # --- USE_G gate decay ---
        if USE_G:
            m_t = (i_t * BT + o_bt) < T
            b_g_last = ct.astype(ct.gather(g, (t_base + last_idx, i_h), check_bounds=True, padding_value=0.0), ct.float32)
            b_g = ct.astype(
                ct.gather(g, (t_base + i_t * BT + o_bt, i_h), mask=m_t, check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            decay = ct.where(m_t, exp2(b_g_last - b_g), ct.zeros((BT,), dtype=ct.float32))
            b_v = b_v * decay[:, None]
            b_g_last = exp2(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last

        # --- USE_GK per-key gate decay ---
        if USE_GK:
            gk_time = t_base + last_idx
            mk1 = o64 < K
            b_gk1 = ct.astype(
                ct.gather(gk, (gk_time, i_h, 0 + o64), mask=mk1, check_bounds=True, padding_value=0.0),
                ct.float32,
            )
            if STATE_V_FIRST:
                b_h1 = b_h1 * exp2(b_gk1)[None, :]
            else:
                b_h1 = b_h1 * exp2(b_gk1)[:, None]
            if K > 64:
                ok2 = 64 + o64
                mk2 = ok2 < K
                b_gk2 = ct.astype(
                    ct.gather(gk, (gk_time, i_h, ok2), mask=mk2, check_bounds=True, padding_value=0.0),
                    ct.float32,
                )
                if STATE_V_FIRST:
                    b_h2 = b_h2 * exp2(b_gk2)[None, :]
                else:
                    b_h2 = b_h2 * exp2(b_gk2)[:, None]
            if K > 128:
                ok3 = 128 + o64
                mk3 = ok3 < K
                b_gk3 = ct.astype(
                    ct.gather(gk, (gk_time, i_h, ok3), mask=mk3, check_bounds=True, padding_value=0.0),
                    ct.float32,
                )
                if STATE_V_FIRST:
                    b_h3 = b_h3 * exp2(b_gk3)[None, :]
                else:
                    b_h3 = b_h3 * exp2(b_gk3)[:, None]
            if K > 192:
                ok4 = 192 + o64
                mk4 = ok4 < K
                b_gk4 = ct.astype(
                    ct.gather(gk, (gk_time, i_h, ok4), mask=mk4, check_bounds=True, padding_value=0.0),
                    ct.float32,
                )
                if STATE_V_FIRST:
                    b_h4 = b_h4 * exp2(b_gk4)[None, :]
                else:
                    b_h4 = b_h4 * exp2(b_gk4)[:, None]

        b_v = ct.astype(b_v, k.dtype)

        # --- H += K^T @ V (K loaded transposed as (64, BT)) ---
        k_time = (t_base + i_t * BT + o_bt)[None, :]
        kmask = ((0 + o64)[:, None] < K) & ((i_t * BT + o_bt)[None, :] < T)
        bk1 = ct.gather(
            k,
            (k_time, i_kh, (0 + o64)[:, None]),
            mask=ct.broadcast_to(kmask, (64, BT)),
            check_bounds=True,
            padding_value=0.0,
        )
        bk1 = ct.where(kmask, bk1, ct.zeros((64, BT), dtype=bk1.dtype))
        a = ct.astype(bk1, ct.tfloat32) if bk1.dtype == ct.float32 else bk1
        bvm = ct.astype(b_v, ct.tfloat32) if b_v.dtype == ct.float32 else b_v
        prod1 = ct.mma(a, bvm, ct.zeros((64, BV), dtype=ct.float32))
        if STATE_V_FIRST:
            b_h1 = b_h1 + ct.transpose(prod1)
        else:
            b_h1 = b_h1 + prod1
        if K > 64:
            kmask2 = ((64 + o64)[:, None] < K) & ((i_t * BT + o_bt)[None, :] < T)
            bk2 = ct.gather(
                k,
                (k_time, i_kh, (64 + o64)[:, None]),
                mask=ct.broadcast_to(kmask2, (64, BT)),
                check_bounds=True,
                padding_value=0.0,
            )
            bk2 = ct.where(kmask2, bk2, ct.zeros((64, BT), dtype=bk2.dtype))
            a = ct.astype(bk2, ct.tfloat32) if bk2.dtype == ct.float32 else bk2
            prod2 = ct.mma(a, bvm, ct.zeros((64, BV), dtype=ct.float32))
            if STATE_V_FIRST:
                b_h2 = b_h2 + ct.transpose(prod2)
            else:
                b_h2 = b_h2 + prod2
        if K > 128:
            kmask3 = ((128 + o64)[:, None] < K) & ((i_t * BT + o_bt)[None, :] < T)
            bk3 = ct.gather(
                k,
                (k_time, i_kh, (128 + o64)[:, None]),
                mask=ct.broadcast_to(kmask3, (64, BT)),
                check_bounds=True,
                padding_value=0.0,
            )
            bk3 = ct.where(kmask3, bk3, ct.zeros((64, BT), dtype=bk3.dtype))
            a = ct.astype(bk3, ct.tfloat32) if bk3.dtype == ct.float32 else bk3
            prod3 = ct.mma(a, bvm, ct.zeros((64, BV), dtype=ct.float32))
            if STATE_V_FIRST:
                b_h3 = b_h3 + ct.transpose(prod3)
            else:
                b_h3 = b_h3 + prod3
        if K > 192:
            kmask4 = ((192 + o64)[:, None] < K) & ((i_t * BT + o_bt)[None, :] < T)
            bk4 = ct.gather(
                k,
                (k_time, i_kh, (192 + o64)[:, None]),
                mask=ct.broadcast_to(kmask4, (64, BT)),
                check_bounds=True,
                padding_value=0.0,
            )
            bk4 = ct.where(kmask4, bk4, ct.zeros((64, BT), dtype=bk4.dtype))
            a = ct.astype(bk4, ct.tfloat32) if bk4.dtype == ct.float32 else bk4
            prod4 = ct.mma(a, bvm, ct.zeros((64, BV), dtype=ct.float32))
            if STATE_V_FIRST:
                b_h4 = b_h4 + ct.transpose(prod4)
            else:
                b_h4 = b_h4 + prod4

    # --- Store final state Ht <- b_h*  (Ht (N, HV, *), per-sequence state) ---
    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            row = (i_v * BV + o_bv)[:, None]
            tm1 = ct.broadcast_to(m_v_lane[:, None] & ((0 + o64) < K)[None, :], (BV, 64))
            ct.scatter(ht, (i_n, i_h, row, (0 + o64)[None, :]), ct.astype(b_h1, ht.dtype), mask=tm1, check_bounds=True)
            if K > 64:
                tm2 = ct.broadcast_to(m_v_lane[:, None] & ((64 + o64) < K)[None, :], (BV, 64))
                ct.scatter(ht, (i_n, i_h, row, (64 + o64)[None, :]), ct.astype(b_h2, ht.dtype), mask=tm2, check_bounds=True)
            if K > 128:
                tm3 = ct.broadcast_to(m_v_lane[:, None] & ((128 + o64) < K)[None, :], (BV, 64))
                ct.scatter(ht, (i_n, i_h, row, (128 + o64)[None, :]), ct.astype(b_h3, ht.dtype), mask=tm3, check_bounds=True)
            if K > 192:
                tm4 = ct.broadcast_to(m_v_lane[:, None] & ((192 + o64) < K)[None, :], (BV, 64))
                ct.scatter(ht, (i_n, i_h, row, (192 + o64)[None, :]), ct.astype(b_h4, ht.dtype), mask=tm4, check_bounds=True)
        else:
            col = (i_v * BV + o_bv)[None, :]
            tm1 = ct.broadcast_to(((0 + o64) < K)[:, None] & m_v_lane[None, :], (64, BV))
            ct.scatter(ht, (i_n, i_h, (0 + o64)[:, None], col), ct.astype(b_h1, ht.dtype), mask=tm1, check_bounds=True)
            if K > 64:
                tm2 = ct.broadcast_to(((64 + o64) < K)[:, None] & m_v_lane[None, :], (64, BV))
                ct.scatter(ht, (i_n, i_h, (64 + o64)[:, None], col), ct.astype(b_h2, ht.dtype), mask=tm2, check_bounds=True)
            if K > 128:
                tm3 = ct.broadcast_to(((128 + o64) < K)[:, None] & m_v_lane[None, :], (64, BV))
                ct.scatter(ht, (i_n, i_h, (128 + o64)[:, None], col), ct.astype(b_h3, ht.dtype), mask=tm3, check_bounds=True)
            if K > 192:
                tm4 = ct.broadcast_to(((192 + o64) < K)[:, None] & m_v_lane[None, :], (64, BV))
                ct.scatter(ht, (i_n, i_h, (192 + o64)[:, None], col), ct.astype(b_h4, ht.dtype), mask=tm4, check_bounds=True)


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
    USE_G: ConstInt,
    USE_GK: ConstInt,
    USE_INITIAL_STATE: ConstInt,
    USE_FINAL_STATE_GRADIENT: ConstInt,
    STATE_V_FIRST: ConstInt,
    occupancy: ConstInt = 1,
):
    # `occupancy` kept for launch-tuning signature parity but unused in body.
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
        b_dh1 = ct.zeros((BV, 64), dtype=ct.float32)
        b_dh2 = ct.zeros((BV, 64), dtype=ct.float32)
        b_dh3 = ct.zeros((BV, 64), dtype=ct.float32)
        b_dh4 = ct.zeros((BV, 64), dtype=ct.float32)
    else:
        b_dh1 = ct.zeros((64, BV), dtype=ct.float32)
        b_dh2 = ct.zeros((64, BV), dtype=ct.float32)
        b_dh3 = ct.zeros((64, BV), dtype=ct.float32)
        b_dh4 = ct.zeros((64, BV), dtype=ct.float32)

    q_base = (bos * H + i_h // (HV // H)) * K
    k_base = (bos * H + i_h // (HV // H)) * K
    w_base = (bos * HV + i_h) * K
    do_base = (bos * HV + i_h) * V
    dv_base = (bos * HV + i_h) * V
    dv2_base = (bos * HV + i_h) * V
    dh_base = (boh * HV + i_h) * K * V
    gk_base0 = (bos * HV + i_h) * K
    dh0_base = i_nh * K * V
    dht_base = i_nh * K * V

    # Raw flat handles + element counts for 1-D flat gather/scatter (cuTile 1.4.0).
    dhtf = dstate_in.get_raw_memory()
    dht_n = array_numel(dstate_in)
    dhf = dh.get_raw_memory()
    dh_n = array_numel(dh)
    dh0f = dh0.get_raw_memory()
    dh0_n = array_numel(dh0)
    gf = g.get_raw_memory()
    g_n = array_numel(g)
    gkf = gk.get_raw_memory()
    gk_n = array_numel(gk)
    dof = do.get_raw_memory()
    do_n = array_numel(do)
    kf = k.get_raw_memory()
    k_n = array_numel(k)
    qf = q.get_raw_memory()
    q_n = array_numel(q)
    wf = w.get_raw_memory()
    w_n = array_numel(w)
    dvf = dv.get_raw_memory()
    dv_n = array_numel(dv)
    dv2f = dv2.get_raw_memory()
    dv2_n = array_numel(dv2)

    o64 = ct.arange(64, dtype=ct.int32)
    o_bt = ct.arange(BT, dtype=ct.int32)
    o_bv = ct.arange(BV, dtype=ct.int32)

    # --- Load final-state gradient dHt -> b_dh* ---
    if USE_FINAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            row = (i_v * BV + o_bv)[:, None]
            b_dh1 = b_dh1 + gather_flat_cb(dhtf, dht_base + row * K + (0 + o64)[None, :], dht_n, padding_value=0.0)
            if K > 64:
                b_dh2 = b_dh2 + gather_flat_cb(dhtf, dht_base + row * K + (64 + o64)[None, :], dht_n, padding_value=0.0)
            if K > 128:
                b_dh3 = b_dh3 + gather_flat_cb(dhtf, dht_base + row * K + (128 + o64)[None, :], dht_n, padding_value=0.0)
            if K > 192:
                b_dh4 = b_dh4 + gather_flat_cb(dhtf, dht_base + row * K + (192 + o64)[None, :], dht_n, padding_value=0.0)
        else:
            col = (i_v * BV + o_bv)[None, :]
            b_dh1 = b_dh1 + gather_flat_cb(dhtf, dht_base + (0 + o64)[:, None] * V + col, dht_n, padding_value=0.0)
            if K > 64:
                b_dh2 = b_dh2 + gather_flat_cb(dhtf, dht_base + (64 + o64)[:, None] * V + col, dht_n, padding_value=0.0)
            if K > 128:
                b_dh3 = b_dh3 + gather_flat_cb(dhtf, dht_base + (128 + o64)[:, None] * V + col, dht_n, padding_value=0.0)
            if K > 192:
                b_dh4 = b_dh4 + gather_flat_cb(dhtf, dht_base + (192 + o64)[:, None] * V + col, dht_n, padding_value=0.0)

    v_col = (i_v * BV + o_bv)[None, :]
    vmask_c = (i_v * BV + o_bv) < V

    # --- Reverse recurrent loop over chunks ---
    for i_rev in range(NT):
        i_t = NT - 1 - i_rev
        dh_chunk = dh_base + i_t * HV * K * V

        # Store b_dh* -> dh[i_t]. Mask 64-wide K rows and BV-wide V cols; flat numel
        # check alone would let over-range K rows spill into the next chunk's dh slot.
        if STATE_V_FIRST:
            row = (i_v * BV + o_bv)[:, None]
            vmask = ((i_v * BV + o_bv) < V)[:, None]
            scatter_flat_cb(
                dhf,
                dh_chunk + row * K + (0 + o64)[None, :],
                ct.astype(b_dh1, dh.dtype),
                dh_n,
                mask=vmask & ((0 + o64) < K)[None, :],
            )
            if K > 64:
                scatter_flat_cb(
                    dhf,
                    dh_chunk + row * K + (64 + o64)[None, :],
                    ct.astype(b_dh2, dh.dtype),
                    dh_n,
                    mask=vmask & ((64 + o64) < K)[None, :],
                )
            if K > 128:
                scatter_flat_cb(
                    dhf,
                    dh_chunk + row * K + (128 + o64)[None, :],
                    ct.astype(b_dh3, dh.dtype),
                    dh_n,
                    mask=vmask & ((128 + o64) < K)[None, :],
                )
            if K > 192:
                scatter_flat_cb(
                    dhf,
                    dh_chunk + row * K + (192 + o64)[None, :],
                    ct.astype(b_dh4, dh.dtype),
                    dh_n,
                    mask=vmask & ((192 + o64) < K)[None, :],
                )
        else:
            col = (i_v * BV + o_bv)[None, :]
            vmask = ((i_v * BV + o_bv) < V)[None, :]
            scatter_flat_cb(
                dhf,
                dh_chunk + (0 + o64)[:, None] * V + col,
                ct.astype(b_dh1, dh.dtype),
                dh_n,
                mask=((0 + o64) < K)[:, None] & vmask,
            )
            if K > 64:
                scatter_flat_cb(
                    dhf,
                    dh_chunk + (64 + o64)[:, None] * V + col,
                    ct.astype(b_dh2, dh.dtype),
                    dh_n,
                    mask=((64 + o64) < K)[:, None] & vmask,
                )
            if K > 128:
                scatter_flat_cb(
                    dhf,
                    dh_chunk + (128 + o64)[:, None] * V + col,
                    ct.astype(b_dh3, dh.dtype),
                    dh_n,
                    mask=((128 + o64) < K)[:, None] & vmask,
                )
            if K > 192:
                scatter_flat_cb(
                    dhf,
                    dh_chunk + (192 + o64)[:, None] * V + col,
                    ct.astype(b_dh4, dh.dtype),
                    dh_n,
                    mask=((192 + o64) < K)[:, None] & vmask,
                )

        last_idx = ct_min((i_t + 1) * BT, T) - 1

        bg_last_exp = ct.zeros((), dtype=ct.float32)
        b_g_exp = ct.zeros((BT,), dtype=ct.float32)
        bg_last = ct.zeros((), dtype=ct.float32)
        b_g = ct.zeros((BT,), dtype=ct.float32)
        m_t = (i_t * BT + o_bt) < T
        if USE_G:
            bg_last = ct.astype(gather_flat_cb(gf, (bos + last_idx) * HV + i_h, g_n, padding_value=0.0), ct.float32)
            g_off = (bos * HV + i_h) + (i_t * BT + o_bt) * HV
            b_g = ct.astype(gather_flat_cb(gf, g_off, g_n, mask=m_t, padding_value=0.0), ct.float32)
            bg_last_exp = exp2(bg_last)
            b_g_exp = exp2(b_g)

        do_off = do_base + (i_t * BT + o_bt)[:, None] * (HV * V) + v_col
        do_full = (m_t[:, None]) & (vmask_c[None, :])
        b_do = gather_flat_cb(dof, do_off, do_n, padding_value=0.0)
        b_do = ct.where(do_full, b_do, ct.zeros((BT, BV), dtype=b_do.dtype))

        # dV = sum_n K_n @ dH_n
        k_row = (i_t * BT + o_bt)[:, None]
        kmask_r = (i_t * BT + o_bt) < T
        bk1 = gather_flat_cb(kf, k_base + k_row * (H * K) + (0 + o64)[None, :], k_n, padding_value=0.0)
        bk1 = ct.where(kmask_r[:, None], bk1, ct.zeros((BT, 64), dtype=bk1.dtype))

        b_gk1 = ct.zeros((64,), dtype=ct.float32)
        b_gk2 = ct.zeros((64,), dtype=ct.float32)
        b_gk3 = ct.zeros((64,), dtype=ct.float32)
        b_gk4 = ct.zeros((64,), dtype=ct.float32)
        if USE_GK:
            mk1 = o64 < K
            b_gk1 = ct.astype(gather_flat_cb(gkf, gk_base0 + last_idx * HV * K + o64, gk_n, mask=mk1, padding_value=0.0), ct.float32)

        b_dv = ct.zeros((BT, BV), dtype=ct.float32)
        a = ct.astype(bk1, ct.tfloat32) if bk1.dtype == ct.float32 else bk1
        if STATE_V_FIRST:
            bmat = ct.astype(ct.transpose(b_dh1), bk1.dtype)
        else:
            bmat = ct.astype(b_dh1, bk1.dtype)
        bmat = ct.astype(bmat, ct.tfloat32) if bmat.dtype == ct.float32 else bmat
        b_dv = ct.mma(a, bmat, b_dv)

        if K > 64:
            bk2 = gather_flat_cb(kf, k_base + k_row * (H * K) + (64 + o64)[None, :], k_n, padding_value=0.0)
            bk2 = ct.where(kmask_r[:, None], bk2, ct.zeros((BT, 64), dtype=bk2.dtype))
            if USE_GK:
                ok2 = 64 + o64
                mk2 = ok2 < K
                b_gk2 = ct.astype(
                    gather_flat_cb(gkf, gk_base0 + last_idx * HV * K + ok2, gk_n, mask=mk2, padding_value=0.0),
                    ct.float32,
                )
            a = ct.astype(bk2, ct.tfloat32) if bk2.dtype == ct.float32 else bk2
            bmat = ct.astype(ct.transpose(b_dh2), bk2.dtype) if STATE_V_FIRST else ct.astype(b_dh2, bk2.dtype)
            bmat = ct.astype(bmat, ct.tfloat32) if bmat.dtype == ct.float32 else bmat
            b_dv = ct.mma(a, bmat, b_dv)
        if K > 128:
            bk3 = gather_flat_cb(kf, k_base + k_row * (H * K) + (128 + o64)[None, :], k_n, padding_value=0.0)
            bk3 = ct.where(kmask_r[:, None], bk3, ct.zeros((BT, 64), dtype=bk3.dtype))
            if USE_GK:
                ok3 = 128 + o64
                mk3 = ok3 < K
                b_gk3 = ct.astype(
                    gather_flat_cb(gkf, gk_base0 + last_idx * HV * K + ok3, gk_n, mask=mk3, padding_value=0.0),
                    ct.float32,
                )
            a = ct.astype(bk3, ct.tfloat32) if bk3.dtype == ct.float32 else bk3
            bmat = ct.astype(ct.transpose(b_dh3), bk3.dtype) if STATE_V_FIRST else ct.astype(b_dh3, bk3.dtype)
            bmat = ct.astype(bmat, ct.tfloat32) if bmat.dtype == ct.float32 else bmat
            b_dv = ct.mma(a, bmat, b_dv)
        if K > 192:
            bk4 = gather_flat_cb(kf, k_base + k_row * (H * K) + (192 + o64)[None, :], k_n, padding_value=0.0)
            bk4 = ct.where(kmask_r[:, None], bk4, ct.zeros((BT, 64), dtype=bk4.dtype))
            if USE_GK:
                ok4 = 192 + o64
                mk4 = ok4 < K
                b_gk4 = ct.astype(
                    gather_flat_cb(gkf, gk_base0 + last_idx * HV * K + ok4, gk_n, mask=mk4, padding_value=0.0),
                    ct.float32,
                )
            a = ct.astype(bk4, ct.tfloat32) if bk4.dtype == ct.float32 else bk4
            bmat = ct.astype(ct.transpose(b_dh4), bk4.dtype) if STATE_V_FIRST else ct.astype(b_dh4, bk4.dtype)
            bmat = ct.astype(bmat, ct.tfloat32) if bmat.dtype == ct.float32 else bmat
            b_dv = ct.mma(a, bmat, b_dv)

        if USE_G:
            decay = ct.where(m_t, exp2(bg_last - b_g), ct.zeros((BT,), dtype=ct.float32))
            b_dv = b_dv * decay[:, None]

        dv_off = dv_base + (i_t * BT + o_bt)[:, None] * (HV * V) + v_col
        dv_loaded = gather_flat_cb(dvf, dv_off, dv_n, padding_value=0.0)
        dv_loaded = ct.where(do_full, dv_loaded, ct.zeros((BT, BV), dtype=dv_loaded.dtype))
        b_dv = b_dv + ct.astype(dv_loaded, ct.float32)
        dv2_off = dv2_base + (i_t * BT + o_bt)[:, None] * (HV * V) + v_col
        # masked lanes park at numel: scatter_flat_cb's bounds check drops them
        dv2_oob = ct.full((BT, BV), dv2_n, dtype=ct.int32)
        dv2_off = ct.where(do_full, dv2_off, dv2_oob)
        scatter_flat_cb(dv2f, dv2_off, ct.astype(b_dv, dv2.dtype), dv2_n)

        # --- dH += trans?(Q @ dO * scale - W @ dV) per K block ---
        time = (i_t * BT + o_bt)[None, :]
        tmask = (i_t * BT + o_bt)[None, :] < T
        b_dv_cast = ct.astype(b_dv, ct.float32)

        # block 1 (rows 0..63)
        kr = (0 + o64)[:, None]
        wmask = ((0 + o64)[:, None] < K) & tmask
        b_w = gather_flat_cb(wf, w_base + kr * 1 + time * (HV * K), w_n, padding_value=0.0)
        b_w = ct.where(wmask, b_w, ct.zeros((64, BT), dtype=b_w.dtype))
        b_q = gather_flat_cb(qf, q_base + kr * 1 + time * (H * K), q_n, padding_value=0.0)
        b_q = ct.where(wmask, b_q, ct.zeros((64, BT), dtype=b_q.dtype))
        if USE_G:
            b_dh1 = b_dh1 * bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if USE_GK:
            if STATE_V_FIRST:
                b_dh1 = b_dh1 * exp2(b_gk1)[None, :]
            else:
                b_dh1 = b_dh1 * exp2(b_gk1)[:, None]
        qa = ct.astype(b_q, ct.tfloat32)
        doa = ct.astype(b_do, ct.tfloat32)
        qdo1 = ct.mma(qa, doa, ct.zeros((64, BV), dtype=ct.float32))
        wa = ct.astype(b_w, ct.tfloat32)
        dva = ct.astype(b_dv_cast, ct.tfloat32)
        wdv1 = ct.mma(wa, dva, ct.zeros((64, BV), dtype=ct.float32))
        upd1 = qdo1 * scale - wdv1
        if STATE_V_FIRST:
            b_dh1 = b_dh1 + ct.transpose(upd1)
        else:
            b_dh1 = b_dh1 + upd1

        if K > 64:
            kr = (64 + o64)[:, None]
            wmask = ((64 + o64)[:, None] < K) & tmask
            b_w = gather_flat_cb(wf, w_base + kr * 1 + time * (HV * K), w_n, padding_value=0.0)
            b_w = ct.where(wmask, b_w, ct.zeros((64, BT), dtype=b_w.dtype))
            b_q = gather_flat_cb(qf, q_base + kr * 1 + time * (H * K), q_n, padding_value=0.0)
            b_q = ct.where(wmask, b_q, ct.zeros((64, BT), dtype=b_q.dtype))
            if USE_G:
                b_dh2 = b_dh2 * bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh2 = b_dh2 * exp2(b_gk2)[None, :]
                else:
                    b_dh2 = b_dh2 * exp2(b_gk2)[:, None]
            qa = ct.astype(b_q, ct.tfloat32)
            qdo2 = ct.mma(qa, doa, ct.zeros((64, BV), dtype=ct.float32))
            wa = ct.astype(b_w, ct.tfloat32)
            wdv2 = ct.mma(wa, dva, ct.zeros((64, BV), dtype=ct.float32))
            upd2 = qdo2 * scale - wdv2
            if STATE_V_FIRST:
                b_dh2 = b_dh2 + ct.transpose(upd2)
            else:
                b_dh2 = b_dh2 + upd2
        if K > 128:
            kr = (128 + o64)[:, None]
            wmask = ((128 + o64)[:, None] < K) & tmask
            b_w = gather_flat_cb(wf, w_base + kr * 1 + time * (HV * K), w_n, padding_value=0.0)
            b_w = ct.where(wmask, b_w, ct.zeros((64, BT), dtype=b_w.dtype))
            b_q = gather_flat_cb(qf, q_base + kr * 1 + time * (H * K), q_n, padding_value=0.0)
            b_q = ct.where(wmask, b_q, ct.zeros((64, BT), dtype=b_q.dtype))
            if USE_G:
                b_dh3 = b_dh3 * bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh3 = b_dh3 * exp2(b_gk3)[None, :]
                else:
                    b_dh3 = b_dh3 * exp2(b_gk3)[:, None]
            qa = ct.astype(b_q, ct.tfloat32)
            qdo3 = ct.mma(qa, doa, ct.zeros((64, BV), dtype=ct.float32))
            wa = ct.astype(b_w, ct.tfloat32)
            wdv3 = ct.mma(wa, dva, ct.zeros((64, BV), dtype=ct.float32))
            upd3 = qdo3 * scale - wdv3
            if STATE_V_FIRST:
                b_dh3 = b_dh3 + ct.transpose(upd3)
            else:
                b_dh3 = b_dh3 + upd3
        if K > 192:
            kr = (192 + o64)[:, None]
            wmask = ((192 + o64)[:, None] < K) & tmask
            b_w = gather_flat_cb(wf, w_base + kr * 1 + time * (HV * K), w_n, padding_value=0.0)
            b_w = ct.where(wmask, b_w, ct.zeros((64, BT), dtype=b_w.dtype))
            b_q = gather_flat_cb(qf, q_base + kr * 1 + time * (H * K), q_n, padding_value=0.0)
            b_q = ct.where(wmask, b_q, ct.zeros((64, BT), dtype=b_q.dtype))
            if USE_G:
                b_dh4 = b_dh4 * bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh4 = b_dh4 * exp2(b_gk4)[None, :]
                else:
                    b_dh4 = b_dh4 * exp2(b_gk4)[:, None]
            qa = ct.astype(b_q, ct.tfloat32)
            qdo4 = ct.mma(qa, doa, ct.zeros((64, BV), dtype=ct.float32))
            wa = ct.astype(b_w, ct.tfloat32)
            wdv4 = ct.mma(wa, dva, ct.zeros((64, BV), dtype=ct.float32))
            upd4 = qdo4 * scale - wdv4
            if STATE_V_FIRST:
                b_dh4 = b_dh4 + ct.transpose(upd4)
            else:
                b_dh4 = b_dh4 + upd4

    # --- Store initial-state gradient dH0 <- b_dh* ---
    if USE_INITIAL_STATE:
        # Mask both the 64-wide K-block rows and BV-wide V cols; flat numel check
        # alone would let an over-range K row spill into the next head's dh0 slot.
        if STATE_V_FIRST:
            row = (i_v * BV + o_bv)[:, None]
            vmask = ((i_v * BV + o_bv) < V)[:, None]
            scatter_flat_cb(
                dh0f,
                dh0_base + row * K + (0 + o64)[None, :],
                ct.astype(b_dh1, dh0.dtype),
                dh0_n,
                mask=vmask & ((0 + o64) < K)[None, :],
            )
            if K > 64:
                scatter_flat_cb(
                    dh0f,
                    dh0_base + row * K + (64 + o64)[None, :],
                    ct.astype(b_dh2, dh0.dtype),
                    dh0_n,
                    mask=vmask & ((64 + o64) < K)[None, :],
                )
            if K > 128:
                scatter_flat_cb(
                    dh0f,
                    dh0_base + row * K + (128 + o64)[None, :],
                    ct.astype(b_dh3, dh0.dtype),
                    dh0_n,
                    mask=vmask & ((128 + o64) < K)[None, :],
                )
            if K > 192:
                scatter_flat_cb(
                    dh0f,
                    dh0_base + row * K + (192 + o64)[None, :],
                    ct.astype(b_dh4, dh0.dtype),
                    dh0_n,
                    mask=vmask & ((192 + o64) < K)[None, :],
                )
        else:
            col = (i_v * BV + o_bv)[None, :]
            vmask = ((i_v * BV + o_bv) < V)[None, :]
            scatter_flat_cb(
                dh0f,
                dh0_base + (0 + o64)[:, None] * V + col,
                ct.astype(b_dh1, dh0.dtype),
                dh0_n,
                mask=((0 + o64) < K)[:, None] & vmask,
            )
            if K > 64:
                scatter_flat_cb(
                    dh0f,
                    dh0_base + (64 + o64)[:, None] * V + col,
                    ct.astype(b_dh2, dh0.dtype),
                    dh0_n,
                    mask=((64 + o64) < K)[:, None] & vmask,
                )
            if K > 128:
                scatter_flat_cb(
                    dh0f,
                    dh0_base + (128 + o64)[:, None] * V + col,
                    ct.astype(b_dh3, dh0.dtype),
                    dh0_n,
                    mask=((128 + o64) < K)[:, None] & vmask,
                )
            if K > 192:
                scatter_flat_cb(
                    dh0f,
                    dh0_base + (192 + o64)[:, None] * V + col,
                    ct.astype(b_dh4, dh0.dtype),
                    dh0_n,
                    mask=((192 + o64) < K)[:, None] & vmask,
                )


# --- Kernels: attention and gradients -------------------------------------------------------------


@ct.kernel
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
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
    USE_G: ConstInt,
    USE_G_GAMMA: ConstInt,
    STATE_V_FIRST: ConstInt,
):
    i_v = ct.bid(0)
    i_t = ct.bid(1)
    i_h = ct.bid(2)

    i_tg = i_t
    i_n = ct.astype(ct.load(chunk_indices, index=(i_t, 0), shape=()), ct.int32)
    i_t = ct.astype(ct.load(chunk_indices, index=(i_t, 1), shape=()), ct.int32)
    bos = ct.astype(ct.load(cu_seqlens, index=(i_n,), shape=()), ct.int32)
    eos = ct.astype(ct.load(cu_seqlens, index=(i_n + 1,), shape=()), ct.int32)
    T = eos - bos

    # q/k/v/o are (T,*,D) token-packed slabs, h is (NT_total*HV,*,*);
    # block-indexed ct.load -> TMA.
    i_kh = i_h // (HV // H)
    i_hc = i_tg * HV + i_h

    q_seg = q.slice(axis=0, start=bos, stop=eos)
    k_seg = k.slice(axis=0, start=bos, stop=eos)
    v_seg = v.slice(axis=0, start=bos, stop=eos)
    o_seg = o.slice(axis=0, start=bos, stop=eos)
    Z = ct.PaddingMode.ZERO

    r_bt = ct.arange(BT, dtype=ct.int32)
    o_t = i_t * BT + r_bt
    m_t = o_t < T

    b_o = ct.zeros((BT, BV), dtype=ct.float32)
    b_A = ct.zeros((BT, BT), dtype=ct.float32)

    for i_k in range(ct.cdiv(K, BK)):
        b_q = ct.load(q_seg, index=(i_t, i_kh, i_k), shape=(BT, 1, BK), padding_mode=Z, latency=2).reshape((BT, BK))
        b_k = ct.load(k_seg, index=(i_t, i_kh, i_k), shape=(BT, 1, BK), padding_mode=Z, latency=2).reshape((BT, BK))

        if STATE_V_FIRST:
            # b_h (BV, BK), O += Q @ H^T
            b_h = ct.load(h, index=(i_hc, i_v, i_k), shape=(1, BV, BK), padding_mode=Z, latency=2).reshape((BV, BK))
            b_o = safe_mma(b_q, ct.transpose(b_h), b_o)
        else:
            # b_h (BK, BV), O += Q @ H
            b_h = ct.load(h, index=(i_hc, i_k, i_v), shape=(1, BK, BV), padding_mode=Z, latency=2).reshape((BK, BV))
            b_o = safe_mma(b_q, b_h, b_o)
        b_A = safe_mma(b_q, ct.transpose(b_k), b_A)

    if USE_G:
        b_g = ct.load(g.slice(axis=0, start=bos, stop=eos), index=(i_t, i_h), shape=(BT, 1), padding_mode=Z).reshape((BT,))
        b_o = b_o * exp2(b_g)[:, None]
        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
    if USE_G_GAMMA:
        b_gamma = ct.load(g_gamma, (i_h,), shape=()).item()
        b_g = b_gamma * ct.astype(r_bt + 1, ct.float32)
        b_o = b_o * exp2(b_g)[:, None]
        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])

    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_A = ct.where(m_A, b_A, ct.zeros((BT, BT), dtype=ct.float32))

    b_v = ct.load(v_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV), padding_mode=Z).reshape((BT, BV))

    b_o = b_o * scale + safe_dot(ct.astype(b_A, b_v.dtype), b_v) * scale

    ct.store(o_seg, index=(i_t, i_h, i_v), tile=ct.astype(b_o, o.dtype).reshape((BT, 1, BV)))


@ct.kernel
def chunk_bwd_kernel_dqkwg(
    q,
    k,
    v,
    g,
    g_gamma,
    h,
    do,
    dh,
    dq,
    dk,
    dw,
    dv,
    dg,
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
    USE_G: ConstInt,
    USE_G_GAMMA: ConstInt,
    USE_DW: ConstInt,
    STATE_V_FIRST: ConstInt,
    NV,  # runtime cdiv(V,BV) -> rolled V-loop (avoids unroll reg-spill at K=256)
):
    # q/k/v/do/dv/dq/dk are (T,*,D) token-packed slabs, h/dh (NT_total*HV,*,*),
    # dg (NK,T,HV); block-indexed ct.load -> TMA.
    i_k = ct.bid(0)
    i_t = ct.bid(1)
    i_h = ct.bid(2)

    i_tg = i_t
    i_n = ct.load(chunk_indices, (i_t, 0), shape=()).item()
    i_t = ct.load(chunk_indices, (i_t, 1), shape=()).item()
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T = eos - bos

    i_kh = i_h // (HV // H)
    i_hc = i_tg * HV + i_h

    q_seg = q.slice(axis=0, start=bos, stop=eos)
    k_seg = k.slice(axis=0, start=bos, stop=eos)
    v_seg = v.slice(axis=0, start=bos, stop=eos)
    do_seg = do.slice(axis=0, start=bos, stop=eos)
    dq_seg = dq.slice(axis=0, start=bos, stop=eos)
    dk_seg = dk.slice(axis=0, start=bos, stop=eos)
    Z = ct.PaddingMode.ZERO

    r_bt = ct.arange(BT, dtype=ct.int32)
    o_t = i_t * BT + r_bt
    m_t = o_t < T

    b_g_gamma = ct.zeros((BT,), dtype=ct.float32)
    b_g_last_gamma = 0.0
    if USE_G_GAMMA:
        b_gamma = ct.load(g_gamma, (i_h,), shape=()).item()
        b_g_gamma = b_gamma * ct.astype(r_bt + 1, ct.float32)
        rem = T - i_t * BT
        bt_clip = BT if BT < rem else rem
        b_g_last_gamma = b_gamma * bt_clip

    b_dg_last = ct.zeros((1,), dtype=ct.float32)
    b_dq = ct.zeros((BT, BK), dtype=ct.float32)
    b_dk = ct.zeros((BT, BK), dtype=ct.float32)
    b_ds = ct.zeros((BT, BT), dtype=ct.float32)
    b_dw = ct.zeros((BT, BK), dtype=ct.float32)

    for i_v in range(NV):
        # h/dh state tiles: (BV,BK) when V-first; else load (BK,BV) and transpose.
        if STATE_V_FIRST:
            b_h = ct.load(h, index=(i_hc, i_v, i_k), shape=(1, BV, BK), padding_mode=Z, latency=2).reshape((BV, BK))
            b_dh = ct.load(dh, index=(i_hc, i_v, i_k), shape=(1, BV, BK), padding_mode=Z, latency=2).reshape((BV, BK))
        else:
            b_h = ct.transpose(ct.load(h, index=(i_hc, i_k, i_v), shape=(1, BK, BV), padding_mode=Z, latency=2).reshape((BK, BV)))
            b_dh = ct.transpose(ct.load(dh, index=(i_hc, i_k, i_v), shape=(1, BK, BV), padding_mode=Z, latency=2).reshape((BK, BV)))

        b_v = ct.load(v_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV), padding_mode=Z, latency=2).reshape((BT, BV))
        b_do = ct.load(do_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV), padding_mode=Z, latency=2).reshape((BT, BV))

        if USE_G:
            b_dg_last = b_dg_last + ct.sum(ct.astype(b_h, ct.float32) * ct.astype(b_dh, ct.float32))
        b_ds = safe_mma(b_do, ct.transpose(b_v), b_ds)
        b_dq = safe_mma(b_do, ct.astype(b_h, b_do.dtype), b_dq)
        b_dk = safe_mma(b_v, ct.astype(b_dh, b_v.dtype), b_dk)
        if USE_DW:
            b_dv = ct.load(
                dv.slice(axis=0, start=bos, stop=eos),
                index=(i_t, i_h, i_v),
                shape=(BT, 1, BV),
                padding_mode=Z,
                latency=2,
            ).reshape((BT, BV))
            b_dw = safe_mma(ct.astype(b_dv, b_v.dtype), ct.astype(b_h, b_v.dtype), b_dw)

    if USE_DW:
        dw_seg = dw.slice(axis=0, start=bos, stop=eos)
        ct.store(dw_seg, index=(i_t, i_h, i_k), tile=ct.astype(-b_dw, dw.dtype).reshape((BT, 1, BK)))

    b_q = ct.load(q_seg, index=(i_t, i_kh, i_k), shape=(BT, 1, BK), padding_mode=Z, latency=2).reshape((BT, BK))
    b_k = ct.load(k_seg, index=(i_t, i_kh, i_k), shape=(BT, 1, BK), padding_mode=Z, latency=2).reshape((BT, BK))

    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :])

    if USE_G:
        g_seg = g.slice(axis=0, start=bos, stop=eos)
        b_g = ct.load(g_seg, index=(i_t, i_h), shape=(BT, 1), padding_mode=Z).reshape((BT,))
        end_idx = i_t * BT + BT
        end_clip = end_idx if end_idx < T else T
        b_g_last = ct.load(g_seg, (end_clip - 1, i_h), shape=()).item()

        b_dg_last = b_dg_last * exp2(b_g_last)
        b_dq = b_dq * exp2(b_g)[:, None] * scale
        decay_k = ct.where(m_t, exp2(-b_g + b_g_last), ct.zeros((BT,), dtype=ct.float32))
        b_dk = b_dk * decay_k[:, None]
        b_dg_last = b_dg_last + ct.sum(b_dk * ct.astype(b_k, ct.float32))
        b_ds = ct.where(m_A, b_ds * exp2(b_g[:, None] - b_g[None, :]), ct.zeros((BT, BT), dtype=ct.float32)) * scale
        b_ds_c = ct.astype(b_ds, b_k.dtype)
        b_dq = safe_mma(b_ds_c, b_k, b_dq)
        b_dk = safe_mma(ct.transpose(b_ds_c), b_q, b_dk)
        b_dg = ct.sum(b_dq * ct.astype(b_q, ct.float32), axis=1) - ct.sum(b_dk * ct.astype(b_k, ct.float32), axis=1)
        last_row = end_clip - 1
        b_dg = ct.where(o_t < last_row, b_dg, b_dg + b_dg_last)
        ct.store(dq_seg, index=(i_t, i_h, i_k), tile=ct.astype(b_dq, dq.dtype).reshape((BT, 1, BK)))
        ct.store(dk_seg, index=(i_t, i_h, i_k), tile=ct.astype(b_dk, dk.dtype).reshape((BT, 1, BK)))
        dg_seg = dg.slice(axis=1, start=bos, stop=eos)
        ct.store(dg_seg, index=(i_k, i_t, i_h), tile=ct.astype(b_dg, dg.dtype).reshape((1, BT, 1)))
    elif USE_G_GAMMA:
        b_g = b_g_gamma
        b_g_last = b_g_last_gamma
        b_dq = b_dq * exp2(b_g)[:, None] * scale
        decay_k = ct.where(m_t, exp2(-b_g + b_g_last), ct.zeros((BT,), dtype=ct.float32))
        b_dk = b_dk * decay_k[:, None]
        b_ds = ct.where(m_A, b_ds * exp2(b_g[:, None] - b_g[None, :]), ct.zeros((BT, BT), dtype=ct.float32)) * scale
        b_ds_c = ct.astype(b_ds, b_k.dtype)
        b_dq = safe_mma(b_ds_c, b_k, b_dq)
        b_dk = safe_mma(ct.transpose(b_ds_c), b_q, b_dk)
        ct.store(dq_seg, index=(i_t, i_h, i_k), tile=ct.astype(b_dq, dq.dtype).reshape((BT, 1, BK)))
        ct.store(dk_seg, index=(i_t, i_h, i_k), tile=ct.astype(b_dk, dk.dtype).reshape((BT, 1, BK)))
    else:
        b_ds = ct.where(m_A, b_ds, ct.zeros((BT, BT), dtype=ct.float32))
        b_ds_c = ct.astype(b_ds, b_k.dtype)
        b_dq = safe_mma(b_ds_c, b_k, b_dq)
        b_dk = safe_mma(ct.transpose(b_ds_c), b_q, b_dk) * scale
        b_dq = b_dq * scale
        ct.store(dq_seg, index=(i_t, i_h, i_k), tile=ct.astype(b_dq, dq.dtype).reshape((BT, 1, BK)))
        ct.store(dk_seg, index=(i_t, i_h, i_k), tile=ct.astype(b_dk, dk.dtype).reshape((BT, 1, BK)))


@ct.kernel
def chunk_bwd_kernel_dv_local(
    q,
    k,
    g,
    g_gamma,
    A,
    do,
    dv,
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
    USE_G: ConstInt,
    USE_G_GAMMA: ConstInt,
    USE_A: ConstInt,
):
    i_t = ct.bid(0)
    i_h = ct.bid(1)

    qf = q.get_raw_memory()
    kf = k.get_raw_memory()
    dof = do.get_raw_memory()
    dvf = dv.get_raw_memory()
    Af = A.get_raw_memory()
    gf = g.get_raw_memory()

    i_n = ct.load(chunk_indices, (i_t, 0), shape=()).item()
    i_t = ct.load(chunk_indices, (i_t, 1), shape=()).item()
    bos = ct.load(cu_seqlens, (i_n,), shape=()).item()
    eos = ct.load(cu_seqlens, (i_n + 1,), shape=()).item()
    T = eos - bos

    q_base = (bos * H + i_h // (HV // H)) * K
    k_base = (bos * H + i_h // (HV // H)) * K
    do_base = (bos * HV + i_h) * V
    dv_base = (bos * HV + i_h) * V

    r_bt = ct.arange(BT, dtype=ct.int32)
    r_bk = ct.arange(BK, dtype=ct.int32)
    r_bv = ct.arange(BV, dtype=ct.int32)

    o_t = i_t * BT + r_bt
    m_t = o_t < T

    b_A = ct.zeros((BT, BT), dtype=ct.float32)
    b_g = ct.zeros((BT,), dtype=ct.float32)
    b_gamma = 0.0

    if USE_A:
        A_base = (bos * HV + i_h) * BT
        a_row = r_bt
        a_col = i_t * BT + r_bt
        A_off = A_base + a_row[:, None] * 1 + a_col[None, :] * (HV * BT)
        A_mask = (a_row[:, None] < BT) & (a_col[None, :] < T)
        b_A = gather_flat(Af, A_off, mask=A_mask, padding_value=0)
    else:
        if USE_G:
            g_base = bos * HV + i_h
            g_off = g_base + o_t * HV
            b_g = gather_flat(gf, g_off, mask=m_t, padding_value=0)
        if USE_G_GAMMA:
            b_gamma = ct.load(g_gamma, (i_h,), shape=()).item()
            b_g = b_gamma * ct.astype(r_bt + 1, ct.float32)

        for i_k in range(ct.cdiv(K, BK)):
            k_col = i_k * BK + r_bk
            q_t = i_t * BT + r_bt
            q_off = q_base + k_col[:, None] * 1 + q_t[None, :] * (H * K)
            q_mask = (k_col[:, None] < K) & (q_t[None, :] < T)
            b_q = gather_flat(qf, q_off, mask=q_mask, padding_value=0)
            k_row = i_t * BT + r_bt
            k_off = k_base + k_row[:, None] * (H * K) + k_col[None, :]
            k_mask = (k_row[:, None] < T) & (k_col[None, :] < K)
            b_k = gather_flat(kf, k_off, mask=k_mask, padding_value=0)
            b_A = b_A + safe_dot(b_k, b_q) * scale

        if USE_G or USE_G_GAMMA:
            if H <= 16:
                b_g_ref = ct.max(b_g, axis=0)
                b_A = b_A * (exp2(b_g[None, :] - b_g_ref) * exp2(b_g_ref - b_g[:, None]))
            elif USE_G:
                g_base = bos * HV + i_h
                b_g_ref = ct.astype(gf.load_offset(g_base + i_t * BT * HV).item(), ct.float32)
                b_A = b_A * (exp2(b_g[None, :] - b_g_ref) * exp2(b_g_ref - b_g[:, None]))
            else:
                b_g_ref = b_gamma
                b_A = b_A * (exp2(b_g[None, :] - b_g_ref) * exp2(b_g_ref - b_g[:, None]))

    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_A = ct.astype(ct.where(m_A, b_A, ct.zeros((BT, BT), dtype=ct.float32)), do.dtype)

    for i_v in range(ct.cdiv(V, BV)):
        v_idx = i_v * BV + r_bv
        do_row = i_t * BT + r_bt
        do_off = do_base + do_row[:, None] * (HV * V) + v_idx[None, :]
        do_mask = (do_row[:, None] < T) & (v_idx[None, :] < V)
        b_do = gather_flat(dof, do_off, mask=do_mask, padding_value=0)
        b_dv = safe_dot(ct.astype(b_A, b_do.dtype), b_do)
        dv_off = dv_base + do_row[:, None] * (HV * V) + v_idx[None, :]
        scatter_flat(dvf, dv_off, ct.astype(b_dv, dv.dtype), mask=do_mask)


# --- Launchers: normalization and gates -----------------------------------------------------------


def chunk_local_cumsum_scalar(
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
    T, H = g.shape
    BT = chunk_size
    NT = len(chunk_indices)
    g_out = out.reshape((T, H))
    scale_val = float(scale) if scale is not None else 0.0
    has_scale = int(scale is not None)
    cu_arg = cu_seqlens
    ci_arg = chunk_indices
    ct.launch(
        stream,
        (NT, H),
        chunk_local_cumsum_scalar_kernel,
        (
            g,
            g_out,
            scale_val,
            cu_arg,
            ci_arg,
            H,
            BT,
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
    return chunk_local_cumsum_scalar(
        g=g,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        out=out,
        stream=stream,
    )


def gdn_gate_chunk_cumsum(
    g,
    A_log,
    chunk_size,
    scale=None,
    dt_bias=None,
    cu_seqlens=None,
    chunk_indices=None,
    out=None,
    bufs=None,
    stream=None,
):
    stream = 0 if stream is None else stream
    T, H = g.shape
    BT = chunk_size
    NT = len(chunk_indices)
    o = out.reshape((T, H))
    dt_arg = opt(dt_bias, bufs, dtname(A_log)).reshape((-1,))
    scale_val = float(scale) if scale is not None else 0.0
    cu_arg = cu_seqlens
    ci_arg = chunk_indices
    ct.launch(
        stream,
        (NT, H),
        gdn_gate_chunk_cumsum_scalar_kernel,
        (
            g,
            A_log.reshape((-1,)),
            dt_arg,
            o,
            scale_val,
            cu_arg,
            ci_arg,
            H,
            BT,
            0,
            int(dt_bias is not None),
            int(scale is not None),
        ),
    )
    return o


def gdn_gate_bwd(g, A_log, dt_bias, dyg, dg_out=None, dA_out=None, dbias_out=None, bufs=None, stream=None):
    """Gate backward. ``dg_out`` (g-shaped, any dtype), ``dA_out`` (A_log-shaped)
    and — with ``dt_bias`` — ``dbias_out`` (H) are written in place;
    ``bufs['dA_gate']``/``bufs['db_gate']`` hold the (NT, H) fp32 chunk partials."""
    stream = 0 if stream is None else stream
    H = g.shape[-1]
    T = g.numel() // H
    BT = 32
    NT = cdiv(T, BT)
    dg = dg_out.reshape(tuple(g.shape))
    dA_nt = bufs["dA_gate"].reshape((NT, H))
    db_nt = bufs["db_gate"].reshape((NT, H)) if dt_bias is not None else None
    dt_arg = opt(dt_bias, bufs, dtname(A_log)).reshape((-1,))
    db_arg = opt(db_nt, bufs).reshape(-1)
    ct.launch(
        stream,
        (NT, H),
        gdn_gate_bwd_kernel,
        (
            g.reshape((-1,)),
            A_log.reshape((-1,)),
            dt_arg,
            dyg.reshape((-1,)),
            dg.reshape(-1),
            dA_nt.reshape(-1),
            db_arg,
            T,
            H,
            BT,
            int(dt_bias is not None),
        ),
    )
    sum_leading(dA_out.reshape((H,)), dA_nt, NT, H, stream=stream)
    if dt_bias is not None:
        sum_leading(dbias_out.reshape((H,)), db_nt, NT, H, stream=stream)
    return dg, dA_out, (dbias_out if dt_bias is not None else None)


# --- Launchers: WY representation -----------------------------------------------------------------


def recompute_w_u_fwd(k, v, beta, A, g=None, cu_seqlens=None, chunk_indices=None, bufs=None, stream=None):
    stream = 0 if stream is None else stream
    T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[1]
    BT = A.shape[-1]
    BK = 64
    BV = 64
    NT = len(chunk_indices)
    w = bufs["w"].reshape((T, HV, K))
    u = bufs["u"].reshape((T, HV, V))
    beta2 = beta.reshape((T, HV))
    A3 = A.reshape(T, HV, BT)
    g_arg = g.reshape(T, HV) if g is not None else dummy("float32", bufs)
    cu_arg = cu_seqlens
    ci_arg = chunk_indices
    autotuned_launch(
        recompute_w_u_fwd_kernel,
        (H, HV, K, V, BT, BK, BV, int(g is not None), str(k.dtype), current_device_id()),
        (NT, HV),
        (
            k,
            v,
            beta2,
            w,
            u,
            A3,
            g_arg,
            cu_arg,
            ci_arg,
            H,
            HV,
            K,
            V,
            BT,
            BK,
            BV,
            int(g is not None),
        ),
        occ_choices=TUNE_OCC,
        stream=stream,
    )
    return w, u


def prepare_wy_repr_bwd(k, v, beta, A, dw, du, g=None, cu_seqlens=None, chunk_indices=None, bufs=None, stream=None):
    stream = 0 if stream is None else stream
    T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[1]
    BT = 64
    NT = len(chunk_indices)
    CONST_TILING = 64
    BK = min(max(next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(next_power_of_2(V), 16), CONST_TILING)
    dk = bufs["wy_dk"].reshape((T, HV, K))
    dv = bufs["wy_dv"].reshape((T, HV, V))
    dg = bufs["wy_dg"].reshape((T, HV)) if g is not None else None
    db = bufs["db"].reshape((T, HV))
    g_arg = opt(g, bufs)
    dg_arg = opt(dg, bufs)
    cu_arg = cu_seqlens
    ci_arg = chunk_indices
    autotuned_launch(
        prepare_wy_repr_bwd_kernel,
        (H, HV, K, V, BT, BK, BV, int(g is not None), str(k.dtype), current_device_id()),
        (NT, HV),
        (
            k,
            v,
            beta,
            g_arg,
            A,
            dw,
            du,
            dk,
            dv,
            db,
            dg_arg,
            cu_arg,
            ci_arg,
            H,
            HV,
            K,
            V,
            BT,
            BK,
            BV,
            int(g is not None),
            cdiv(K, BK),
            cdiv(V, BV),
        ),
        occ_choices=TUNE_OCC,
        stream=stream,
    )
    if H != HV:
        dk_r = bufs["wy_dk_hred"].reshape((T, H, K))
        head_group_sum(dk_r, dk, T, H, HV // H, K, stream=stream)
        dk = dk_r
    return dk, dv, db, dg


def chunk_gated_delta_rule_fwd_intra(k, v, g=None, beta=None, cu_seqlens=None, chunk_size=64, chunk_indices=None, bufs=None, compute_wu=True, stream=None):
    stream = 0 if stream is None else stream
    T, H, K, HV = *k.shape, beta.shape[1]
    BT = chunk_size
    NT = len(chunk_indices)

    A = bufs["A"].reshape((T, HV, BT))
    zero_fill(A, stream=stream)
    BK = 64
    g_arg = opt(g, bufs)
    cu_arg = cu_seqlens
    ci_arg = chunk_indices
    # Masked BC=16 4-sub-block kernel: masks the partial last chunk's tail.
    BC = 16
    autotuned_launch(
        chunk_gated_delta_rule_fwd_kkt_solve_kernel,
        (H, HV, K, BT, BC, BK, int(g is not None), str(k.dtype), current_device_id()),
        (NT, HV),
        (
            k,
            g_arg,
            beta,
            A,
            cu_arg,
            ci_arg,
            H,
            HV,
            K,
            BT,
            BC,
            BK,
            int(g is not None),
        ),
        occ_choices=TUNE_OCC,
        stream=stream,
    )
    if not compute_wu:
        return None, None, A
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, bufs=bufs, stream=stream)
    return w, u, A


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

    state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
    h = bufs["state_checkpoints"].reshape((NT, HV) + state_shape[2:])
    final_state = bufs["final_state"].reshape(state_shape) if output_final_state else None
    if final_state is not None:
        zero_fill(final_state, stream=stream)
    v_new = bufs["v_new"].reshape((T, HV, V)) if save_new_value else None

    vnew_arg = opt(v_new, bufs, dtname(u))
    g_arg = opt(g, bufs)
    gk_arg = opt(gk, bufs)
    h0_arg = opt(initial_state, bufs)
    ht_arg = opt(final_state, bufs)
    cu_arg = cu_seqlens
    co_arg = chunk_offsets

    # BV = V-tile width; chosen up front (not swept) to avoid a tileiras compile stall.
    BV = 64
    if V % 32 == 0:
        try:
            sm_count, cc_major = device_attrs()
        except Exception:  # noqa: BLE001
            sm_count = 0
            cc_major = 0
        grid_blocks = cdiv(V, 64) * (N * HV)
        if sm_count and grid_blocks * 2 <= sm_count:
            # Grid-saturation split (all arches).
            BV = 32
        elif cc_major == 8 and K >= 192:
            # Register-spill split (sm_80 only): at K>=192 the 4 fp32 (64,BV) state
            # accumulators overflow the 255-reg budget at BV=64 -> heavy spill.
            BV = 32
    grid = (cdiv(V, BV), N * HV)

    # Multi-dim arrays passed as-is (per-dim indices + real strides).
    autotuned_launch(
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
        (
            H,
            HV,
            K,
            V,
            BT,
            BV,
            int(g is not None),
            int(gk is not None),
            int(initial_state is not None),
            int(output_final_state),
            int(save_new_value),
            int(state_v_first),
            str(k.dtype),
            current_device_id(),
        ),
        grid,
        (
            k,
            u,
            w,
            vnew_arg,
            g_arg,
            gk_arg,
            h,
            h0_arg,
            ht_arg,
            cu_arg,
            co_arg,
            H,
            HV,
            K,
            V,
            BT,
            BV,
            int(g is not None),
            int(gk is not None),
            int(initial_state is not None),
            int(output_final_state),
            int(save_new_value),
            int(state_v_first),
        ),
        occ_choices=TUNE_OCC,
        stream=stream,
    )
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
    scale=None,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
    bufs=None,
    stream=None,
):
    stream = 0 if stream is None else stream
    T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[1]
    BT = chunk_size
    N, NT = len(cu_seqlens) - 1, len(chunk_indices)
    chunk_offsets = bufs["chunk_offsets"]

    dh = bufs["dstate"].reshape((NT, HV, V, K) if state_v_first else (NT, HV, K, V))
    dh0 = bufs["dstate0"].reshape(tuple(h0.shape)) if h0 is not None else None
    dv2 = bufs["dv2"].reshape((T, HV, V))

    # For K>128 the blockdim64 kernel's 4 (64xBV) fp32 dH accumulators overflow
    # tileiras allocation at BV=64; shrink BV to keep the live footprint in range.
    BV = 16 if K > 128 else 64
    grid = (cdiv(V, BV), N * HV)
    g_arg = opt(g, bufs)
    gk_arg = opt(gk, bufs)
    dht_arg = opt(dstate_in, bufs)
    dh0_arg = opt(dh0, bufs)
    cu_arg = cu_seqlens
    co_arg = chunk_offsets

    autotuned_launch(
        chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
        (
            H,
            HV,
            K,
            V,
            BT,
            BV,
            int(g is not None),
            int(gk is not None),
            int(state_v_first),
            str(q.dtype),
            current_device_id(),
        ),
        grid,
        (
            q,
            k,
            w,
            g_arg,
            gk_arg,
            dht_arg,
            dh0_arg,
            do,
            dh,
            dv,
            dv2,
            cu_arg,
            co_arg,
            float(scale),
            H,
            HV,
            K,
            V,
            BT,
            BV,
            int(g is not None),
            int(gk is not None),
            int(h0 is not None),
            int(dstate_in is not None),
            int(state_v_first),
            1,
        ),
        occ_choices=TUNE_OCC,
        stream=stream,
    )
    return dh, dh0, dv2


# --- Launchers: attention and gradients -----------------------------------------------------------


def chunk_fwd_o(
    q,
    k,
    v,
    h,
    g=None,
    g_gamma=None,
    scale=None,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
    bufs=None,
    stream=None,
):
    stream = 0 if stream is None else stream
    T, H, K, V, HV = *q.shape, v.shape[-1], v.shape[1]
    BT = chunk_size
    NT = len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o = bufs["o"].reshape((T, HV, V))
    BK = min(max(next_power_of_2(K), 16), 64)
    BV = 64
    grid = (cdiv(V, BV), NT, HV)
    g_arg = g.reshape(T, HV) if g is not None else dummy("float32", bufs)
    gg_arg = opt(g_gamma, bufs)
    cu_arg = cu_seqlens
    ci_arg = chunk_indices
    # h flattened to (NT*HV,*,*) slabs for block-indexed TMA.
    if state_v_first:
        h3 = h.reshape(NT * HV, V, K)
    else:
        h3 = h.reshape(NT * HV, K, V)
    autotuned_launch(
        chunk_fwd_kernel_o,
        (
            H,
            HV,
            K,
            V,
            BT,
            BK,
            BV,
            int(g is not None),
            int(g_gamma is not None),
            int(state_v_first),
            str(q.dtype),
            current_device_id(),
        ),
        grid,
        (
            q,
            k,
            v,
            h3,
            g_arg,
            gg_arg,
            o,
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
            int(g is not None),
            int(g_gamma is not None),
            int(state_v_first),
        ),
        occ_choices=TUNE_OCC,
        stream=stream,
    )
    return o


def chunk_bwd_dqkwg(
    q,
    k,
    v,
    do,
    h,
    dh,
    w=None,
    g=None,
    g_gamma=None,
    dv=None,
    scale=None,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
    bufs=None,
    stream=None,
):
    stream = 0 if stream is None else stream
    T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[1]
    BT = chunk_size
    NT = len(chunk_indices)
    CONST_TILING = 64
    BK = min(max(next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(next_power_of_2(V), 16), CONST_TILING)
    NK = cdiv(K, BK)
    dq = bufs["dq"].reshape((T, HV, K))
    dk = bufs["dk"].reshape((T, HV, K))
    dg = bufs["dg_nk"].reshape((NK, T, HV)) if g is not None else None
    dw = bufs["dw"].reshape((T, HV, K)) if w is not None else None
    grid = (NK, NT, HV)
    # h/dh flattened to (NT*HV,*,*) slabs for block-indexed TMA.
    if state_v_first:
        h3 = h.reshape(NT * HV, V, K)
        dh3 = dh.reshape(NT * HV, V, K)
    else:
        h3 = h.reshape(NT * HV, K, V)
        dh3 = dh.reshape(NT * HV, K, V)
    g_arg = g.reshape(T, HV) if g is not None else dummy(dtname(q), bufs)
    gg_arg = opt(g_gamma, bufs)
    dw3 = opt(dw, bufs, dtname(k))
    dv3 = dv.reshape(T, HV, V) if dv is not None else dummy(dtname(k), bufs)
    dg3 = opt(dg, bufs)
    cu_arg = cu_seqlens
    ci_arg = chunk_indices
    use_dw = int(dw is not None and dv is not None)
    autotuned_launch(
        chunk_bwd_kernel_dqkwg,
        (
            H,
            HV,
            K,
            V,
            BT,
            BK,
            BV,
            int(g is not None),
            int(g_gamma is not None),
            use_dw,
            int(state_v_first),
            str(q.dtype),
            current_device_id(),
        ),
        grid,
        (
            q,
            k,
            v,
            g_arg,
            gg_arg,
            h3,
            do,
            dh3,
            dq,
            dk,
            dw3,
            dv3,
            dg3,
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
            int(g is not None),
            int(g_gamma is not None),
            use_dw,
            int(state_v_first),
            cdiv(V, BV),
        ),
        occ_choices=TUNE_OCC,
        stream=stream,
    )
    if H != HV:
        dq_r = bufs["dq_hred"].reshape((T, H, K))
        dk_r = bufs["dk_hred"].reshape((T, H, K))
        head_group_sum(dq_r, dq, T, H, HV // H, K, stream=stream)
        head_group_sum(dk_r, dk, T, H, HV // H, K, stream=stream)
        dq, dk = dq_r, dk_r
    if dg is not None:
        dg_r = bufs["dg"].reshape((T, HV))
        sum_leading(dg_r.reshape((T * HV,)), dg.reshape((NK, T * HV)), NK, T * HV, stream=stream)
        dg = dg_r
    return dq, dk, dw, dg


def chunk_bwd_dv_local(q, k, do, g=None, g_gamma=None, A=None, scale=None, cu_seqlens=None, chunk_size=64, chunk_indices=None, bufs=None, stream=None):
    stream = 0 if stream is None else stream
    T, H, K, V, HV = *k.shape, do.shape[-1], do.shape[1]
    BT = chunk_size
    CONST_TILING = 64
    BK = min(max(next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(next_power_of_2(V), 16), CONST_TILING)
    NT = len(chunk_indices)
    dv = bufs["dv"].reshape((T, HV, V))
    grid = (NT, HV)
    g_arg = opt(g, bufs)
    gg_arg = opt(g_gamma, bufs)
    A_arg = opt(A, bufs)
    cu_arg = cu_seqlens
    ci_arg = chunk_indices
    scale_val = float(scale) if scale is not None else 0.0
    autotuned_launch(
        chunk_bwd_kernel_dv_local,
        (
            H,
            HV,
            K,
            V,
            BT,
            BK,
            BV,
            int(g is not None),
            int(g_gamma is not None),
            int(A is not None),
            str(q.dtype),
            current_device_id(),
        ),
        grid,
        (
            q,
            k,
            g_arg,
            gg_arg,
            A_arg,
            do,
            dv,
            cu_arg,
            ci_arg,
            scale_val,
            H,
            HV,
            K,
            V,
            BT,
            BK,
            BV,
            int(g is not None),
            int(g_gamma is not None),
            int(A is not None),
        ),
        occ_choices=TUNE_OCC,
        stream=stream,
    )
    return dv


# --- Pipelines ------------------------------------------------------------------------------------


def chunk_gated_delta_rule_fwd(
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
    cp_context=None,
    chunk_indices=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    bufs=None,
    stream=None,
):
    stream = 0 if stream is None else stream
    g_input = g if use_gate_in_kernel else None
    if use_gate_in_kernel:
        g = gdn_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            chunk_size=64,
            scale=RCP_LN2,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            out=bufs["g_cum"],
            stream=stream,
        )
    else:
        g = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, out=bufs["g_cum"], stream=stream)

    w, u, A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, bufs=bufs, stream=stream)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        bufs=bufs,
        stream=stream,
    )
    o = chunk_fwd_o(
        q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, state_v_first=state_v_first, bufs=bufs, stream=stream
    )
    return g, o, A, final_state, initial_state, g_input


def chunk_gated_delta_rule_bwd(
    q,
    k,
    v,
    g,
    beta,
    A,
    scale,
    initial_state,
    do,
    dstate_in,
    state_v_first=False,
    cu_seqlens=None,
    cp_context=None,
    chunk_indices=None,
    use_gate_in_kernel=False,
    g_input=None,
    A_log=None,
    dt_bias=None,
    bufs=None,
    stream=None,
):
    stream = 0 if stream is None else stream
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, bufs=bufs, stream=stream)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        bufs=bufs,
        stream=stream,
    )
    dv = chunk_bwd_dv_local(q=q, k=k, g=g, do=do, scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, bufs=bufs, stream=stream)
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dstate_in=dstate_in,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        bufs=bufs,
        stream=stream,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        bufs=bufs,
        stream=stream,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k, v=v, beta=beta, g=g, A=A, dw=dw, du=dv, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, bufs=bufs, stream=stream
    )
    m_dg = 1
    for s_ in dg.shape:
        m_dg *= int(s_)
    add_inplace(dg.reshape((m_dg,)), dg2.reshape((m_dg,)), m_dg, stream=stream)
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, out=bufs["dg_cum"], stream=stream)
    dA_log, ddt_bias = None, None
    if use_gate_in_kernel:
        dg, dA_log, ddt_bias = gdn_gate_bwd(
            g=g_input,
            A_log=A_log,
            dt_bias=dt_bias,
            dyg=dg,
            dg_out=bufs["dg_gate"],
            dA_out=bufs["dA_log"],
            dbias_out=bufs["ddt_bias"] if dt_bias is not None else None,
            bufs=bufs,
            stream=stream,
        )
    return dq, dk, dk2, dv, db, dg, dh0, dA_log, ddt_bias


def chunk_gated_delta_rule_grad(
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
    state_v_first=False,
    cu_seqlens=None,
    chunk_indices=None,
    bufs=None,
    stream=None,
):
    r"""GDN backward over THD (token-packed) inputs.

    ``q``/``k``/``v``/``do`` are ``[total_T, H, D]``, ``g``/``beta`` are
    ``[total_T, HV]``; ``cu_seqlens``, ``chunk_indices`` and the pre-carved
    ``bufs`` views are required. Recomputes the forward's prep (L2-normalized
    q/k + rstd, cumulative gate, intra-chunk WY matrix) from the inputs, then
    runs the backward kernels. Gradients land in the caller's planted carves;
    the returned handles are those same buffers."""
    stream = 0 if stream is None else stream
    if scale is None:
        scale = k.shape[-1] ** -0.5
    q_in, k_in, q_rstd, k_rstd = q, k, None, None
    if use_qk_l2norm_in_kernel:
        q_in, q_rstd = l2norm_fwd(q, out=bufs["q_norm"], rstd_out=bufs["q_rstd"], stream=stream)
        k_in, k_rstd = l2norm_fwd(k, out=bufs["k_norm"], rstd_out=bufs["k_rstd"], stream=stream)
    g_cum = chunk_local_cumsum(g, chunk_size=BT_CHUNK, scale=RCP_LN2, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, out=bufs["g_cum"], stream=stream)
    _w, _u, A = chunk_gated_delta_rule_fwd_intra(
        k=k_in, v=v, g=g_cum, beta=beta, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, bufs=bufs, compute_wu=False, stream=stream
    )
    dq, dk, dk2, dv, db, dg, dh0, _dA_log, _ddt_bias = chunk_gated_delta_rule_bwd(
        q=q_in,
        k=k_in,
        v=v,
        g=g_cum,
        beta=beta,
        A=A,
        scale=scale,
        initial_state=initial_state,
        do=do,
        dstate_in=dstate_in,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        bufs=bufs,
        stream=stream,
    )
    if use_qk_l2norm_in_kernel:
        dq = l2norm_bwd(q_in, q_rstd, dq, out=bufs["dq_l2"], bufs=bufs, stream=stream)
        dk = l2norm_bwd(k_in, k_rstd, dk, dy2=dk2, out=bufs["dk_l2"], bufs=bufs, stream=stream)
    else:
        # dk/dk2 are the head-reduced finals for GVA, HV-head for MHA
        n_dk = k.shape[0] * k.shape[1] * k.shape[2]
        add_inplace(dk.reshape(-1), dk2.reshape(-1), n_dk, stream=stream)
    return dq, dk, dv, db, dg, dh0


def chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_indices=None,
    cp_context=None,
    bufs=None,
    stream=None,
    **kwargs,
):
    r"""Chunked Gated DeltaNet (GDN) over THD (token-packed) inputs.

    ``q``/``k``/``v`` are ``[total_T, H, D]``, ``g``/``beta`` are ``[total_T, H]``;
    ``cu_seqlens``, ``chunk_indices`` and the pre-carved ``bufs`` views are
    required. Returns ``(o, final_state)`` in THD layout, written into
    ``bufs['o']`` / ``bufs['final_state']``."""
    stream = 0 if stream is None else stream

    if "transpose_state_layout" in kwargs:
        if state_v_first:
            raise ValueError("Cannot pass both `state_v_first` and the deprecated `transpose_state_layout`.")
        state_v_first = kwargs.pop("transpose_state_layout")

    if q.shape[1] != k.shape[1]:
        raise ValueError(f"q and k must have the same number of heads, got q.shape[1]={q.shape[1]} and k.shape[1]={k.shape[1]}")
    H, HV = q.shape[1], v.shape[1]
    if HV % H != 0:
        raise ValueError(f"For GVA, HV ({HV}) must be divisible by H ({H}), got HV % H = {HV % H}")

    if "head_first" in kwargs:
        raise DeprecationWarning("head_first has been removed. Inputs must be in THD `[total_T, H, ...]` format.")

    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required (THD layout)")
    if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
        raise ValueError(f"#initial states must equal #sequences ({len(cu_seqlens) - 1}), got {initial_state.shape[0]}.")
    use_gate_in_kernel = kwargs.get("use_gate_in_kernel", False)
    A_log = kwargs.get("A_log")
    dt_bias = kwargs.get("dt_bias")
    if use_gate_in_kernel:
        assert A_log is not None, "A_log must be provided when use_gate_in_kernel=True."
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")

    if scale is None:
        scale = k.shape[-1] ** -0.5

    q_in, k_in = q, k
    if use_qk_l2norm_in_kernel:
        q_in, _q_rstd = l2norm_fwd(q, out=bufs["q_norm"], rstd_out=bufs["q_rstd"], stream=stream)
        k_in, _k_rstd = l2norm_fwd(k, out=bufs["k_norm"], rstd_out=bufs["k_rstd"], stream=stream)
    if use_beta_sigmoid_in_kernel:
        beta = fused_beta_sigmoid(beta, scale=2.0 if allow_neg_eigval else 1.0, out=bufs["beta_sig"], stream=stream)
    if cu_seqlens is not None and chunk_indices is None:
        raise ValueError("varlen (cu_seqlens) requires chunk_indices — callers build the (seq, intra) table")
    _g_cum, o, _A, final_state, _s0, _g_input = chunk_gated_delta_rule_fwd(
        q=q_in,
        k=k_in,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cp_context=cp_context,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        bufs=bufs,
        stream=stream,
    )
    return o, final_state
