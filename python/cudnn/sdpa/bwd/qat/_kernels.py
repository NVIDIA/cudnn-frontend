# SPDX-License-Identifier: Apache-2.0

"""Triton kernels for NVFP4 quantization-aware SDPA backward.

The split dQ and dK/dV topology is ported from FastVideo commit
e9bbaca07d511b2ee7e16474dae6f923426223dc. The port keeps the production
64x64 SM100 path and uses the same kernels with 32x32 tiles for the remaining
supported Blackwell configurations.
"""

import triton
import triton.language as tl

from ._nvfp4 import fake_quantize_nvfp4

RCP_LN2 = tl.constexpr(1.4426950408889634)


@triton.jit
def attention_backward_preprocess(
    high_precision_o,
    grad_o,
    delta,
    seqlen_q,
    block_m: tl.constexpr,
    head_dim: tl.constexpr,
):
    row_offsets = tl.program_id(0) * block_m + tl.arange(0, block_m)
    batch_head = tl.program_id(1)
    col_offsets = tl.arange(0, head_dim)
    valid = row_offsets < seqlen_q
    tensor_offsets = batch_head * head_dim * seqlen_q + row_offsets[:, None] * head_dim + col_offsets[None, :]
    o = tl.load(high_precision_o + tensor_offsets, mask=valid[:, None], other=0.0).to(tl.float32)
    do = tl.load(grad_o + tensor_offsets, mask=valid[:, None], other=0.0).to(tl.float32)
    tl.store(delta + batch_head * seqlen_q + row_offsets, tl.sum(o * do, axis=1), mask=valid)


@triton.jit
def attention_backward_dq(
    q,
    k,
    v,
    softmax_scale,
    grad_o,
    grad_q,
    lse,
    delta,
    q_stride_b,
    kv_stride_b,
    q_stride_h,
    kv_stride_h,
    q_stride_s,
    kv_stride_s,
    q_stride_d,
    kv_stride_d,
    num_heads,
    seqlen_q,
    seqlen_kv,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    head_dim: tl.constexpr,
    causal: tl.constexpr,
):
    batch_head = tl.program_id(1)
    q_batch_head_offset = q_stride_h * (batch_head % num_heads) + q_stride_b * (batch_head // num_heads)
    kv_batch_head_offset = kv_stride_h * (batch_head % num_heads) + kv_stride_b * (batch_head // num_heads)
    q += q_batch_head_offset
    k += kv_batch_head_offset
    v += kv_batch_head_offset
    grad_o += q_batch_head_offset
    grad_q += q_batch_head_offset
    lse += batch_head * seqlen_q
    delta += batch_head * seqlen_q

    start_m = tl.program_id(0) * block_m
    row_offsets = start_m + tl.arange(0, block_m)
    col_offsets = tl.arange(0, head_dim)
    q_valid = row_offsets < seqlen_q
    q_tile = tl.load(
        q + row_offsets[:, None] * q_stride_s + col_offsets[None, :] * q_stride_d,
        mask=q_valid[:, None],
        other=0.0,
    )
    do_tile = tl.load(
        grad_o + row_offsets[:, None] * q_stride_s + col_offsets[None, :] * q_stride_d,
        mask=q_valid[:, None],
        other=0.0,
    )
    lse_log2 = tl.load(lse + row_offsets, mask=q_valid, other=0.0)[:, None] * RCP_LN2
    delta_tile = tl.load(delta + row_offsets, mask=q_valid, other=0.0)
    dq = tl.zeros([block_m, head_dim], dtype=tl.float32)
    qk_scale_log2 = softmax_scale * RCP_LN2

    for start_n in tl.range(0, seqlen_kv, block_n):
        key_offsets = start_n + tl.arange(0, block_n)
        kv_valid = key_offsets < seqlen_kv
        k_tile = tl.load(
            k + key_offsets[:, None] * kv_stride_s + col_offsets[None, :] * kv_stride_d,
            mask=kv_valid[:, None],
            other=0.0,
        )
        v_tile = tl.load(
            v + key_offsets[:, None] * kv_stride_s + col_offsets[None, :] * kv_stride_d,
            mask=kv_valid[:, None],
            other=0.0,
        )

        valid_probability = q_valid[:, None] & kv_valid[None, :]
        if causal:
            valid_probability = valid_probability & (row_offsets[:, None] >= key_offsets[None, :])
        logits = tl.dot(q_tile, tl.trans(k_tile)) * qk_scale_log2
        probability = tl.where(valid_probability, tl.math.exp2(logits - lse_log2), 0.0)
        grad_probability = tl.dot(do_tile, tl.trans(v_tile))
        grad_score = probability * (grad_probability - delta_tile[:, None])
        dq += tl.dot(grad_score.to(tl.bfloat16), k_tile)

    dq *= softmax_scale
    tl.store(
        grad_q + row_offsets[:, None] * q_stride_s + col_offsets[None, :] * q_stride_d,
        dq,
        mask=q_valid[:, None],
    )


@triton.jit
def attention_backward_dkdv(
    q,
    k,
    v,
    softmax_scale,
    grad_o,
    grad_k,
    grad_v,
    lse,
    delta,
    q_stride_b,
    kv_stride_b,
    q_stride_h,
    kv_stride_h,
    q_stride_s,
    kv_stride_s,
    q_stride_d,
    kv_stride_d,
    num_heads,
    seqlen_q,
    seqlen_kv,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    head_dim: tl.constexpr,
    causal: tl.constexpr,
):
    batch_head = tl.program_id(1)
    q_batch_head_offset = q_stride_h * (batch_head % num_heads) + q_stride_b * (batch_head // num_heads)
    kv_batch_head_offset = kv_stride_h * (batch_head % num_heads) + kv_stride_b * (batch_head // num_heads)
    q += q_batch_head_offset
    k += kv_batch_head_offset
    v += kv_batch_head_offset
    grad_o += q_batch_head_offset
    grad_k += kv_batch_head_offset
    grad_v += kv_batch_head_offset
    lse += batch_head * seqlen_q
    delta += batch_head * seqlen_q

    start_n = tl.program_id(0) * block_n
    key_offsets = start_n + tl.arange(0, block_n)
    col_offsets = tl.arange(0, head_dim)
    kv_valid = key_offsets < seqlen_kv
    k_tile = tl.load(
        k + key_offsets[:, None] * kv_stride_s + col_offsets[None, :] * kv_stride_d,
        mask=kv_valid[:, None],
        other=0.0,
    )
    v_tile = tl.load(
        v + key_offsets[:, None] * kv_stride_s + col_offsets[None, :] * kv_stride_d,
        mask=kv_valid[:, None],
        other=0.0,
    )
    dk = tl.zeros([block_n, head_dim], dtype=tl.float32)
    dv = tl.zeros([block_n, head_dim], dtype=tl.float32)
    qk_scale_log2 = softmax_scale * RCP_LN2

    for start_m in tl.range(0, seqlen_q, block_m):
        row_offsets = start_m + tl.arange(0, block_m)
        q_valid = row_offsets < seqlen_q
        q_tile = tl.load(
            q + row_offsets[:, None] * q_stride_s + col_offsets[None, :] * q_stride_d,
            mask=q_valid[:, None],
            other=0.0,
        )
        do_tile = tl.load(
            grad_o + row_offsets[:, None] * q_stride_s + col_offsets[None, :] * q_stride_d,
            mask=q_valid[:, None],
            other=0.0,
        )
        lse_log2 = tl.load(lse + row_offsets, mask=q_valid, other=0.0) * RCP_LN2
        delta_tile = tl.load(delta + row_offsets, mask=q_valid, other=0.0)

        valid_probability = q_valid[:, None] & kv_valid[None, :]
        if causal:
            valid_probability = valid_probability & (row_offsets[:, None] >= key_offsets[None, :])
        logits = tl.dot(q_tile, tl.trans(k_tile)) * qk_scale_log2
        probability = tl.where(valid_probability, tl.math.exp2(logits - lse_log2[:, None]), 0.0)

        # dV models the NVFP4 probability path. dQ and dK retain the
        # high-precision probability for the straight-through estimator.
        fake_probability = fake_quantize_nvfp4(
            probability,
            valid_probability,
            block_m,
            block_n,
            tl.bfloat16,
        )
        dv += tl.dot(tl.trans(fake_probability), do_tile)

        grad_probability = tl.dot(do_tile, tl.trans(v_tile))
        grad_score = probability * (grad_probability - delta_tile[:, None])
        dk += tl.dot(tl.trans(grad_score.to(tl.bfloat16)), q_tile)

    dk *= softmax_scale
    tl.store(
        grad_k + key_offsets[:, None] * kv_stride_s + col_offsets[None, :] * kv_stride_d,
        dk,
        mask=kv_valid[:, None],
    )
    tl.store(
        grad_v + key_offsets[:, None] * kv_stride_s + col_offsets[None, :] * kv_stride_d,
        dv,
        mask=kv_valid[:, None],
    )
