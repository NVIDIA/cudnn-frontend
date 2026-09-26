# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudnn
import pytest
import torch
import math
from enum import IntEnum
from looseversion import LooseVersion

from .fp8_ref import (
    compute_ref,
    compute_ref_backward,
    gqa_kv_head,
)
from .helpers import (
    get_fp8_scale_factor,
    get_fp8_descale_factor,
    convert_to_cudnn_type,
    create_sparse_int_tensor,
    inject_negative_score_rows,
    print_tensor_stats,
    exact_equal,
    prefix_sum,
    convert_packed_to_uniform,
    convert_uniform_to_packed,
    time_execution,
    profile_execution,
    note_frost_routing,
)
from .random_config import packed_token_capacity

# fmt: off

class GraphFwdUid(IntEnum):
    q = 0
    k = 1
    v = 2
    q_descale = 5
    k_descale = 6
    v_descale = 7
    s_scale = 9
    s_descale = 8
    o_scale = 10
    o = 3
    stats = 4
    o_amax = 12
    kv_seq_len = 13
    q_seq_len = 14
    k_block_table = 15
    v_block_table = 16
    q_ragged_offset = 17
    k_ragged_offset = 18
    v_ragged_offset = 19
    o_ragged_offset = 20
    stats_ragged_offset = 21
    sink_token = 22
    cu_seq_len_q = 23
    cu_seq_len_kv = 24
    sf_o = 25

class GraphBwdUid(IntEnum):
    q = 100
    k = 101
    v = 102
    o = 103
    dO = 104
    stats = 105
    q_descale = 106
    k_descale = 107
    v_descale = 108
    o_descale = 109
    dO_descale = 110
    s_descale = 111
    dP_descale = 112
    s_scale = 113
    dQ_scale = 114
    dK_scale = 115
    dV_scale = 116
    dP_scale = 117
    dQ = 118
    dK = 119
    dV = 120
    dQ_amax = 121
    dK_amax = 122
    dV_amax = 123
    dP_amax = 124
    q_ragged_offset = 125
    k_ragged_offset = 126
    v_ragged_offset = 127
    o_ragged_offset = 128
    stats_ragged_offset = 129
    dO_ragged_offset = 130
    kv_seq_len = 131
    q_seq_len = 132
    sink_token = 133
    dSink_token = 134

def block_scaled_o_sf_dims(b, h_q, s_qo, d_vo, o_block_scale):
    """sf_o declared as per-(b, h) planes: [b, h_q, s padded to 128, d/block padded to 4], F8_128x4 atom order."""
    rows = -(-s_qo // 128) * 128
    cols = max(4, -(-(d_vo // o_block_scale) // 4) * 4)
    return (b, h_q, rows, cols)


def generate_graph_fwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, block_size, is_ragged=False, generate_stats=True, left_bound=None, right_bound=None, diag_align=None, with_sink_token=False, is_cu_seq_len=False, with_ragged_offset_multiplier=False, implementation=cudnn.attention_implementation.AUTO, max_total_seq_len_q=None, max_total_seq_len_kv=None, o_block_scale=0):
    graph_fwd = cudnn.pygraph(io_data_type=cudnn_itype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    use_padding_mask = None
    kv_seq_len = None
    q_seq_len = None
    cu_seq_len_q = None
    cu_seq_len_kv = None
    k_block_table = None
    v_block_table = None

    # BSHD stride order: (s*h*d, d, h*d, 1)
    stride_q = (s_qo * h_q * d_qk, d_qk, h_q * d_qk, 1)
    stride_k = (s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1)
    stride_v = (s_kv * h_v * d_vo, d_vo, h_v * d_vo, 1)

    if block_size == 0:
        q = graph_fwd.tensor(uid=GraphFwdUid.q, dim=(b, h_q, s_qo, d_qk), stride=stride_q, data_type=cudnn_itype)
        k = graph_fwd.tensor(uid=GraphFwdUid.k, dim=(b, h_k, s_kv, d_qk), stride=stride_k, data_type=cudnn_itype)
        v = graph_fwd.tensor(uid=GraphFwdUid.v, dim=(b, h_v, s_kv, d_vo), stride=stride_v, data_type=cudnn_itype)
    else:
        table_size = math.ceil(s_kv / block_size)
        num_blocks = table_size * b

        q = graph_fwd.tensor(uid=GraphFwdUid.q, dim=(b, h_q, s_qo, d_qk), stride=stride_q, data_type=cudnn_itype)
        k = graph_fwd.tensor(uid=GraphFwdUid.k, dim=(num_blocks, h_k, block_size, d_qk), stride=(block_size * h_k * d_qk, block_size * d_qk, d_qk, 1), data_type=cudnn_itype)
        v = graph_fwd.tensor(uid=GraphFwdUid.v, dim=(num_blocks, h_v, block_size, d_vo), stride=(block_size * h_v * d_vo, block_size * d_vo, d_vo, 1), data_type=cudnn_itype)

        use_padding_mask = True
        kv_seq_len = graph_fwd.tensor(uid=GraphFwdUid.kv_seq_len, dim=(b,), stride=(1,), data_type=cudnn.data_type.INT32)
        q_seq_len = graph_fwd.tensor(uid=GraphFwdUid.q_seq_len, dim=(b,), stride=(1,), data_type=cudnn.data_type.INT32)
        k_block_table = graph_fwd.tensor(uid=GraphFwdUid.k_block_table, dim=(b, 1, table_size, 1), stride=(table_size, table_size, 1, 1), data_type=cudnn.data_type.INT32)
        v_block_table = graph_fwd.tensor(uid=GraphFwdUid.v_block_table, dim=(b, 1, table_size, 1), stride=(table_size, table_size, 1, 1), data_type=cudnn.data_type.INT32)

    if is_ragged:
        use_padding_mask = True
        if is_cu_seq_len:
            cu_seq_len_q = graph_fwd.tensor(uid=GraphFwdUid.cu_seq_len_q, dim=(b + 1,), stride=(1,), data_type=cudnn.data_type.INT32)
            cu_seq_len_kv = graph_fwd.tensor(uid=GraphFwdUid.cu_seq_len_kv, dim=(b + 1,), stride=(1,), data_type=cudnn.data_type.INT32)
        else:
            q_seq_len = graph_fwd.tensor(uid=GraphFwdUid.q_seq_len, dim=(b,), stride=(1,), data_type=cudnn.data_type.INT32)
            kv_seq_len = graph_fwd.tensor(uid=GraphFwdUid.kv_seq_len, dim=(b,), stride=(1,), data_type=cudnn.data_type.INT32)

        q_ragged_offset = graph_fwd.tensor(uid=int(GraphFwdUid.q_ragged_offset), dim=(b + 1,), stride=(1,), data_type=cudnn.data_type.INT64)
        k_ragged_offset = graph_fwd.tensor(uid=int(GraphFwdUid.k_ragged_offset), dim=(b + 1,), stride=(1,), data_type=cudnn.data_type.INT64)
        v_ragged_offset = graph_fwd.tensor(uid=int(GraphFwdUid.v_ragged_offset), dim=(b + 1,), stride=(1,), data_type=cudnn.data_type.INT64)
        o_ragged_offset = graph_fwd.tensor(uid=int(GraphFwdUid.o_ragged_offset), dim=(b + 1,), stride=(1,), data_type=cudnn.data_type.INT64)
        stats_ragged_offset = graph_fwd.tensor(uid=int(GraphFwdUid.stats_ragged_offset), dim=(b + 1,), stride=(1,), data_type=cudnn.data_type.INT64) if generate_stats else None
        q.set_ragged_offset(q_ragged_offset)
        k.set_ragged_offset(k_ragged_offset)
        v.set_ragged_offset(v_ragged_offset)
        if with_ragged_offset_multiplier:
            # Offsets are stored in coarser units (divided out in the allocation);
            # the engine multiplies back to element offsets.
            q.set_ragged_offset_multiplier(d_qk)
            k.set_ragged_offset_multiplier(d_qk)
            v.set_ragged_offset_multiplier(d_vo)

    q_descale = graph_fwd.tensor(uid=GraphFwdUid.q_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    k_descale = graph_fwd.tensor(uid=GraphFwdUid.k_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    v_descale = graph_fwd.tensor(uid=GraphFwdUid.v_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    s_scale = graph_fwd.tensor(uid=GraphFwdUid.s_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    s_descale = graph_fwd.tensor(uid=GraphFwdUid.s_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    o_scale = graph_fwd.tensor(uid=GraphFwdUid.o_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    sink_token = None
    if with_sink_token:
        sink_token = graph_fwd.tensor(uid=GraphFwdUid.sink_token, dim=(1, h_q, 1, 1), stride=(h_q, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    sdpa_kwargs = dict(
        q=q, k=k, v=v,
        descale_q=q_descale, descale_k=k_descale, descale_v=v_descale,
        scale_s=s_scale, descale_s=s_descale, scale_o=o_scale,
        generate_stats=generate_stats, attn_scale=attn_scale, use_causal_mask=False,
        use_padding_mask=use_padding_mask, seq_len_kv=kv_seq_len, seq_len_q=q_seq_len,
        cu_seq_len_q=cu_seq_len_q, cu_seq_len_kv=cu_seq_len_kv,
        paged_attention_k_table=k_block_table, paged_attention_v_table=v_block_table,
        paged_attention_max_seq_len_kv=s_kv,
        left_bound=left_bound, right_bound=right_bound,
        sink_token=sink_token,
        implementation=implementation,
        max_total_seq_len_q=max_total_seq_len_q,
        max_total_seq_len_kv=max_total_seq_len_kv,
    )
    # Only pass diagonal_alignment if it's not None (pybind11 doesn't accept None for enum types)
    if diag_align is not None:
        sdpa_kwargs['diagonal_alignment'] = diag_align
    if o_block_scale:
        # Block-scaled O: the sf_o output (per-(b,h) planes) rides sdpa_fp8 like
        # rng_dump; FP4 O carries E4M3 scales, E4M3 O carries UE8M0 scales.
        sf_dims = block_scaled_o_sf_dims(b, h_q, s_qo, d_vo, o_block_scale)
        sf_stride = (sf_dims[1] * sf_dims[2] * sf_dims[3], sf_dims[2] * sf_dims[3], sf_dims[3], 1)
        sf_dtype = cudnn.data_type.FP8_E4M3 if o_block_scale == 16 else cudnn.data_type.FP8_E8M0
        sdpa_kwargs['sf_o'] = graph_fwd.tensor(uid=GraphFwdUid.sf_o, dim=sf_dims, stride=sf_stride, data_type=sf_dtype)
        cudnn_otype = cudnn.data_type.FP4_E2M1 if o_block_scale == 16 else cudnn.data_type.FP8_E4M3
    # Amax_S is not requested: nothing downstream consumes it, and the FROST
    # engines no longer produce it (they decline graphs that declare it).
    o, stats, _amax_s_unused, amax_o = graph_fwd.sdpa_fp8(**sdpa_kwargs)

    stride_o = (s_qo * h_q * d_vo, d_vo, h_q * d_vo, 1)
    o.set_uid(GraphFwdUid.o).set_output(True).set_dim((b, h_q, s_qo, d_vo)).set_stride(stride_o).set_data_type(cudnn_otype)
    if is_ragged:
        o.set_ragged_offset(o_ragged_offset)
        if with_ragged_offset_multiplier:
            o.set_ragged_offset_multiplier(d_vo)

    if generate_stats:
        stats_stride = (s_qo * h_q, 1, h_q, 1) if is_ragged else (s_qo * h_q, s_qo, 1, 1)
        stats.set_uid(GraphFwdUid.stats).set_output(True).set_dim((b, h_q, s_qo, 1)).set_stride(stats_stride).set_data_type(cudnn.data_type.FLOAT)
        if is_ragged:
            stats.set_ragged_offset(stats_ragged_offset)

    amax_o.set_uid(GraphFwdUid.o_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

    return graph_fwd

def generate_graph_bwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, deterministic, is_ragged=False, left_bound=None, right_bound=None, diag_align=None, with_sink_token=False):
    graph_bwd = cudnn.pygraph(io_data_type=cudnn_itype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    stride_q = (s_qo * h_q * d_qk, d_qk, h_q * d_qk, 1)
    stride_k = (s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1)
    stride_v = (s_kv * h_v * d_vo, d_vo, h_v * d_vo, 1)
    stride_o = (s_qo * h_q * d_vo, d_vo, h_q * d_vo, 1)
    stats_stride = (s_qo * h_q, 1, h_q, 1) if is_ragged else (s_qo * h_q, s_qo, 1, 1)

    q = graph_bwd.tensor(uid=GraphBwdUid.q, dim=(b, h_q, s_qo, d_qk), stride=stride_q, data_type=cudnn_itype)
    k = graph_bwd.tensor(uid=GraphBwdUid.k, dim=(b, h_k, s_kv, d_qk), stride=stride_k, data_type=cudnn_itype)
    v = graph_bwd.tensor(uid=GraphBwdUid.v, dim=(b, h_v, s_kv, d_vo), stride=stride_v, data_type=cudnn_itype)
    o = graph_bwd.tensor(uid=GraphBwdUid.o, dim=(b, h_q, s_qo, d_vo), stride=stride_o, data_type=cudnn_otype)
    dO = graph_bwd.tensor(uid=GraphBwdUid.dO, dim=(b, h_q, s_qo, d_vo), stride=stride_o, data_type=cudnn_itype)
    stats = graph_bwd.tensor(uid=GraphBwdUid.stats, dim=(b, h_q, s_qo, 1), stride=stats_stride, data_type=cudnn.data_type.FLOAT)

    use_padding_mask = False
    seq_len_q = None
    seq_len_kv = None

    if is_ragged:
        use_padding_mask = True
        seq_len_q = graph_bwd.tensor(uid=int(GraphBwdUid.q_seq_len), dim=(b,), stride=(1,), data_type=cudnn.data_type.INT32)
        seq_len_kv = graph_bwd.tensor(uid=int(GraphBwdUid.kv_seq_len), dim=(b,), stride=(1,), data_type=cudnn.data_type.INT32)

        q_ragged_offset = graph_bwd.tensor(uid=int(GraphBwdUid.q_ragged_offset), dim=(b + 1,), stride=(1,), data_type=cudnn.data_type.INT64)
        k_ragged_offset = graph_bwd.tensor(uid=int(GraphBwdUid.k_ragged_offset), dim=(b + 1,), stride=(1,), data_type=cudnn.data_type.INT64)
        v_ragged_offset = graph_bwd.tensor(uid=int(GraphBwdUid.v_ragged_offset), dim=(b + 1,), stride=(1,), data_type=cudnn.data_type.INT64)
        o_ragged_offset = graph_bwd.tensor(uid=int(GraphBwdUid.o_ragged_offset), dim=(b + 1,), stride=(1,), data_type=cudnn.data_type.INT64)
        stats_ragged_offset = graph_bwd.tensor(uid=int(GraphBwdUid.stats_ragged_offset), dim=(b + 1,), stride=(1,), data_type=cudnn.data_type.INT64)
        dO_ragged_offset = graph_bwd.tensor(uid=int(GraphBwdUid.dO_ragged_offset), dim=(b + 1,), stride=(1,), data_type=cudnn.data_type.INT64)
        q.set_ragged_offset(q_ragged_offset)
        k.set_ragged_offset(k_ragged_offset)
        v.set_ragged_offset(v_ragged_offset)
        o.set_ragged_offset(o_ragged_offset)
        stats.set_ragged_offset(stats_ragged_offset)
        dO.set_ragged_offset(dO_ragged_offset)

    q_descale = graph_bwd.tensor(uid=GraphBwdUid.q_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    k_descale = graph_bwd.tensor(uid=GraphBwdUid.k_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    v_descale = graph_bwd.tensor(uid=GraphBwdUid.v_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    o_descale = graph_bwd.tensor(uid=GraphBwdUid.o_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dO_descale = graph_bwd.tensor(uid=GraphBwdUid.dO_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    s_descale = graph_bwd.tensor(uid=GraphBwdUid.s_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dP_descale = graph_bwd.tensor(uid=GraphBwdUid.dP_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    s_scale = graph_bwd.tensor(uid=GraphBwdUid.s_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dQ_scale = graph_bwd.tensor(uid=GraphBwdUid.dQ_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dK_scale = graph_bwd.tensor(uid=GraphBwdUid.dK_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dV_scale = graph_bwd.tensor(uid=GraphBwdUid.dV_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dP_scale = graph_bwd.tensor(uid=GraphBwdUid.dP_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    sink_token = None
    dSink_token = None
    if with_sink_token:
        sink_token = graph_bwd.tensor(uid=GraphBwdUid.sink_token, dim=(1, h_q, 1, 1), stride=(h_q, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
        dSink_token = graph_bwd.tensor(uid=GraphBwdUid.dSink_token, dim=(1, h_q, 1, 1), stride=(h_q, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP = graph_bwd.sdpa_fp8_backward(
        q=q, k=k, v=v, o=o, dO=dO, stats=stats,
        descale_q=q_descale, descale_k=k_descale, descale_v=v_descale,
        descale_o=o_descale, descale_dO=dO_descale, descale_s=s_descale, descale_dP=dP_descale,
        scale_s=s_scale, scale_dQ=dQ_scale, scale_dK=dK_scale, scale_dV=dV_scale, scale_dP=dP_scale,
        attn_scale=attn_scale, use_padding_mask=use_padding_mask,
        diagonal_alignment=diag_align if diag_align is not None else cudnn.diagonal_alignment.TOP_LEFT,
        left_bound=left_bound,
        right_bound=right_bound,
        use_deterministic_algorithm=deterministic,
        seq_len_q=seq_len_q, seq_len_kv=seq_len_kv,
        sink_token=sink_token,
        dSink_token=dSink_token,
    )

    dQ.set_uid(GraphBwdUid.dQ).set_output(True).set_dim((b, h_q, s_qo, d_qk)).set_stride(stride_q).set_data_type(cudnn_otype)
    dK.set_uid(GraphBwdUid.dK).set_output(True).set_dim((b, h_k, s_kv, d_qk)).set_stride(stride_k).set_data_type(cudnn_otype)
    dV.set_uid(GraphBwdUid.dV).set_output(True).set_dim((b, h_v, s_kv, d_vo)).set_stride(stride_v).set_data_type(cudnn_otype)

    if is_ragged:
        dQ.set_ragged_offset(q_ragged_offset)
        dK.set_ragged_offset(k_ragged_offset)
        dV.set_ragged_offset(v_ragged_offset)

    amax_dQ.set_uid(GraphBwdUid.dQ_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dK.set_uid(GraphBwdUid.dK_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dV.set_uid(GraphBwdUid.dV_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dP.set_uid(GraphBwdUid.dP_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

    if with_sink_token:
        dSink_token.set_uid(GraphBwdUid.dSink_token).set_output(True).set_dim((1, h_q, 1, 1)).set_stride((h_q, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

    return graph_bwd

# A kernel-vs-reference fp8 flip is two fp32 computations of the SAME intermediate straddling a rounding
# midpoint.  They differ by ~1e-6..1e-5 relative (exp2/FMA vs torch.exp; exact fp32 sums of small ints
# otherwise), while adjacent fp8 codes are >= 2^-3 (e4m3) / 2^-2 (e5m2) apart relative -- so the reference's
# value must sit within a small fraction of one code spacing of the midpoint.  1/32 of the spacing is ~3e-3
# relative for e4m3, two orders of magnitude above the largest observed straddle and far below where a
# coincidence starts to look like evidence.  One constant, one place to tighten.
_MIDPOINT_WINDOW = 1 / 32

# The kernel's row and the reference's row recomputed with the one flipped code are compared as OUTPUT codes:
# each side rounded its fp32 row once to the output grid, so an element may sit half an output code spacing from
# the exact prediction on either side (one whole code, with the wider spacing when a rounding crosses a binade).
# That bound cannot tell one flip from two at a binade whose spacing equals one flip -- the residual is one code
# either way -- but a genuine row's rounding residuals are unbiased, while two flips move EVERY element one way.
# The least-squares number of flips fitted to the row separates them: 1 +- 1/sqrt(d) for one flip, 2 for two.
# One flip must fit within this band of 1 (1.5 and 0.5 are out: two flips fit at 2, half a flip is no flip).
_ONE_FLIP_BAND = 0.5

# tag / kind -> (the reduction runs over q rows i (True) or kv columns j (False), the intermediate it sums)
_FLIP_KINDS = {"O": (True, "p_scaled"), "dQ": (True, "ds_scaled"), "dK": (False, "ds_scaled"), "dV": (False, "p_scaled")}


def _fp8_finite_magnitudes(fp8_dtype, device, dtype):
    """Every finite non-negative value of ``fp8_dtype``, ascending (0 once; e5m2's inf and both NaN blocks excluded)."""
    codes = torch.arange(256, dtype=torch.uint8, device=device).view(fp8_dtype).float()
    return codes[torch.isfinite(codes) & (codes >= 0)].unique().to(dtype)


def _fp8_codes_around(x, fp8_dtype):
    """For scaled intermediates ``x`` (any float dtype, any shape): ``c_ref``, the code the reference rounds x to --
    torch's own conversion, i.e. exactly the reference's ``.to(torch_itype)``: round-to-nearest-even, e4m3fn
    saturating at 448, e5m2 overflowing to inf; ``c_alt``, the ADJACENT representable code on x's side of c_ref
    (the one the other computation could have rounded to); and ``u = c_alt - c_ref``, 0 where no flip is possible
    (x exactly on a code, x = 0, or the neighbour is not finite: past 448 / at inf / NaN)."""
    mags = _fp8_finite_magnitudes(fp8_dtype, x.device, x.dtype)
    c_ref = x.to(fp8_dtype).to(x.dtype)
    finite = torch.isfinite(c_ref) & torch.isfinite(x)
    mag = c_ref.abs()
    idx = torch.searchsorted(mags, mag.masked_fill(~finite, 0.0).contiguous())
    away = x.abs() > mag  # x lies beyond its code (farther from zero) -> the neighbour is the next code up
    alt_idx = torch.where(away, idx + 1, idx - 1)
    possible = finite & (x.abs() != mag) & (alt_idx >= 0) & (alt_idx < mags.numel())
    sign = torch.where(x < 0, -1.0, 1.0).to(x.dtype)
    c_ref = sign * mag
    c_alt = torch.where(possible, sign * mags[alt_idx.clamp(0, mags.numel() - 1)], c_ref)
    return c_ref, c_alt, torch.where(possible, c_alt - c_ref, torch.zeros_like(c_ref))


def _grid_spacing(y, dtype):
    """Spacing of ``dtype``'s value grid at |y|, for values that are ``dtype`` codes times a power-of-two descale
    (every scale factor in this harness is one, so the grid keeps its relative structure): ``eps(dtype) x
    2^floor(log2 |y|)``, the spacing from |y| to the next code away from zero (at an exact power of two, the wider
    one); 0 at y = 0.  Below the dtype's normal range the real spacing is constant and larger than this -- there
    the atol term of any bound built on it dominates by orders of magnitude."""
    _, exp = torch.frexp(y)  # |y| = m x 2^exp with m in [0.5, 1): floor(log2 |y|) = exp - 1
    sp = torch.ldexp(torch.full_like(y, torch.finfo(dtype).eps), exp - 1)
    return torch.where(y == 0, torch.zeros_like(y), sp)


def _unravel(flat, shape):
    return tuple(int(x) for x in torch.unravel_index(torch.as_tensor(flat), tuple(shape)))


def _flip_position(by_q_row, row, c, g, s_op):
    """The (b, q_head, i, j) intermediate a candidate index ``c`` of output d-row ``row`` = (b, s, h) names."""
    b, s_, h_ = row
    if by_q_row:
        return (b, h_, s_, c)  # candidate = key j
    return (b, h_ * g + c // s_op, c % s_op, s_)  # candidate = (q head of the group, query i)


def _flip_evidence(actual, expected, bad_rows, atol, rtol, *, tag, kind, operand, flip_unit, fp8_dtype, out_dtype, intermediates):
    """Attribute every bad d-row to ONE kernel-vs-reference fp8 rounding flip, with the reference's own
    intermediates as evidence; return the attributions, or None after printing why the first row that is not one
    is not one.

    A d-row of O / dQ / dK / dV is LINEAR in the fp8 P or dS codes it reduces over (``fp8_ref``, BSHD indices):
        O[b, i, hq]  = sum_j                    P[b, hq, i, j]  * gain * s_descale  * (V[b, j, hk] * v_descale)
        dQ[b, i, hq] = sum_j                    dS[b, hq, i, j] * dP_descale        * (K[b, j, hk] * k_descale)
        dK[b, j, hk] = sum_{hq in group(hk), i} dS[b, hq, i, j] * dP_descale        * (Q[b, i, hq] * q_descale)
        dV[b, j, hk] = sum_{hq in group(hk), i} P[b, hq, i, j]  * s_descale         * (dO[b, i, hq] * dO_descale)
    with ``hk = gqa_kv_head(hq)``.  So a row is one flip iff for SOME position of ITS OWN reduction -- the same
    batch, a q head of the same GQA group, a key / query it actually sums over -- that is VALID in the reference
    (unmasked, inside the sequence, not a padded q row; everywhere else P = dS = 0 by construction and nothing can
    round differently), the reference's scaled fp32 intermediate ``x`` sits at the rounding boundary of two
    adjacent fp8 codes ``c_ref -> c_alt`` (within ``_MIDPOINT_WINDOW`` of their midpoint), and the row equals
    ``expected + u * flip_unit * gain * operand_row`` -- the result recomputed with that one code flipped -- three
    ways at once: within the ordinary tolerance; per element within the OUTPUT's own rounding (``atol`` plus half
    an ``out_dtype`` code spacing at |actual| and at |expected|: each side rounded its fp32 row once to the output
    grid, and both sides are compared as dequantized output codes); and as a whole, the least-squares number of
    such flips fitted to the row being 1 within ``_ONE_FLIP_BAND`` (two flips fit at 2, a residual that merely
    stays inside a wide tolerance fits at whatever multiple it is).  ``flip_unit`` is the intermediate's descale,
    ``operand`` the DEQUANTIZED operand tensor in BSHD (the generator's sparse ints, ``= fp8 * descale`` exactly),
    ``gain`` 1 for the gradients and the forward's ``2^rescale_threshold * exp(m_block - m_final) / l`` for O.
    ``intermediates(selection)`` re-runs the reference on the bad rows only (``fp8_ref._Intermediates``): a
    power-of-two multiple of an operand row proves nothing by itself -- it does not name the position, cannot
    see a mask, a batch or a GQA group, and has no notion of which spacing the codes at that value actually have.
    What this cannot see: a deviation between half and one-and-a-half flips of the same intermediate, hidden
    inside the output's rounding -- the outputs are fp8 codes, and nothing finer than their spacing is observable."""
    by_q_row, x_key = _FLIP_KINDS[kind]
    b_, s_out, h_out, d = actual.shape
    ops = operand.detach().to(device=actual.device, dtype=actual.dtype)
    if ops.dim() != 4 or ops.shape[0] != b_ or ops.shape[-1] != d:
        print(f"%%%% '{tag}': flip attribution skipped: operand {tuple(ops.shape)} is not [b={b_}, s, h, d={d}]")
        return None
    s_op, h_op = ops.shape[1], ops.shape[2]
    h_q, h_kv = (h_out, h_op) if by_q_row else (h_op, h_out)
    if h_q % h_kv:
        print(f"%%%% '{tag}': flip attribution skipped: h_q={h_q} is not a multiple of h_kv={h_kv}")
        return None
    g = h_q // h_kv
    rows = bad_rows.nonzero().flatten()
    coords = torch.stack(torch.unravel_index(rows, actual.shape[:-1]), dim=1)  # [n, 3] = (b, s, h)
    mode = "q_rows" if by_q_row else "kv_cols"
    ref = intermediates({mode: coords[:, [0, 2, 1]]})  # (b, q_head, i) or (b, kv_head, j)
    if ref.get("mode") != mode or ref.get("h_q") != h_q or ref.get("h_kv") != h_kv:
        raise ValueError(f"intermediates for '{tag}' describe mode={ref.get('mode')} h_q={ref.get('h_q')} h_kv={ref.get('h_kv')}; expected {mode} {h_q} {h_kv}")
    want = (rows.numel(), s_op) if by_q_row else (rows.numel(), g, s_op)
    x = ref[x_key].to(actual.device)
    if tuple(x.shape) != want or tuple(ref["valid"].shape) != want:
        raise ValueError(f"intermediates for '{tag}' have shape {tuple(x.shape)}; expected {want}")
    valid = ref["valid"].to(actual.device).reshape(rows.numel(), -1)
    gain = ref.get("gain")
    gain = torch.ones_like(x) if gain is None else gain.to(actual.device)
    c_ref, c_alt, u = _fp8_codes_around(x, fp8_dtype)
    off_mid = (x - (c_ref + c_alt) / 2).abs()
    near = (u != 0) & (off_mid <= u.abs() * _MIDPOINT_WINDOW)
    step = (u * float(flip_unit) * gain).to(actual.dtype)  # dequantized deviation of the intermediate
    x, c_ref, c_alt, u, off_mid, near, step = (t.reshape(rows.numel(), -1) for t in (x, c_ref, c_alt, u, off_mid, near, step))
    fits = []
    for n_ in range(rows.numel()):
        row = tuple(coords[n_].tolist())
        bb, s_, h_ = row
        a_row, e_row = actual[bb, s_, h_], expected[bb, s_, h_]
        rvec = a_row - e_row
        tol = atol + rtol * e_row.abs()
        # each side rounded its fp32 row once to the output grid: half a code spacing from either side
        tol_out = atol + (_grid_spacing(a_row, out_dtype) + _grid_spacing(e_row, out_dtype)) / 2
        if by_q_row:
            op_rows = ops[bb, :, gqa_kv_head(h_, h_q, h_kv), :]  # [s_kv, d]: candidate c = key j
        else:
            op_rows = ops[bb, :, h_ * g : (h_ + 1) * g, :].permute(1, 0, 2).reshape(g * s_op, d)  # c = gi * s_q + i
        pred = step[n_][:, None] * op_rows  # [candidates, d]: the row recomputed with that one code flipped, minus the reference
        res = rvec[None, :] - pred
        worst = (res.abs() / tol[None, :]).amax(dim=-1)  # per candidate, against the ordinary tolerance
        rounding = (res.abs() / tol_out[None, :]).amax(dim=-1)  # ... against the output's own rounding
        pp = (pred * pred).sum(dim=-1)
        flips = torch.where(pp > 0, (pred * rvec[None, :]).sum(dim=-1) / pp.clamp_min(1e-30), torch.zeros_like(pp))  # least-squares flip count
        ok = valid[n_] & near[n_] & (worst <= 1.0) & (rounding <= 1.0) & ((flips - 1.0).abs() < _ONE_FLIP_BAND)
        if not bool(ok.any()):
            _explain_unfit_row(tag, kind, row, rvec, tol, ops, float(flip_unit), by_q_row, h_q, h_kv, g, s_op, fp8_dtype, out_dtype,
                               x[n_], c_ref[n_], c_alt[n_], u[n_], off_mid[n_], valid[n_], near[n_], worst, rounding, flips)
            return None
        score = torch.maximum(rounding, torch.maximum(worst, (flips - 1.0).abs() / _ONE_FLIP_BAND))
        c = int(torch.where(ok, score, torch.full_like(score, float("inf"))).argmin().item())
        fits.append(dict(row=row, position=_flip_position(by_q_row, row, c, g, s_op), x=x[n_][c].item(), c_ref=c_ref[n_][c].item(),
                         c_alt=c_alt[n_][c].item(), u=u[n_][c].item(), off_mid_spacings=(off_mid[n_][c] / u[n_][c].abs()).item(),
                         step=step[n_][c].item(), gain=gain.reshape(rows.numel(), -1)[n_][c].item(), worst=worst[c].item(),
                         rounding=rounding[c].item(), flips=flips[c].item(), n_tied=int(ok.sum().item()) - 1,
                         operand_row=(bb, c, gqa_kv_head(h_, h_q, h_kv)) if by_q_row else (bb, c % s_op, h_ * g + c // s_op)))
    return fits


def _closest_scalar_multiple(flat, rvec, tol, chunk=1 << 16):
    """Per operand row of ``flat`` [N, d]: the least-squares multiple ``alpha`` of the row closest to ``rvec`` and
    ``max |rvec - alpha * row| / tol`` -- in row chunks, so the only [*, d] temporaries are one chunk's (the operand
    is the whole generator tensor, 512 MB at the suite's largest shapes)."""
    alpha = torch.empty(flat.shape[0], device=flat.device, dtype=flat.dtype)
    fit = torch.empty_like(alpha)
    for i in range(0, flat.shape[0], chunk):
        f = flat[i : i + chunk]
        norms = (f * f).sum(dim=-1)
        a = torch.where(norms > 0, (f @ rvec) / norms.clamp_min(1e-30), torch.zeros_like(norms))
        alpha[i : i + chunk] = a
        fit[i : i + chunk] = ((rvec[None, :] - a[:, None] * f).abs() / tol[None, :]).amax(dim=-1)
    return alpha, fit


def _explain_unfit_row(tag, kind, row, rvec, tol, ops, flip_unit, by_q_row, h_q, h_kv, g, s_op, fp8_dtype, out_dtype,
                       x_n, c_ref_n, c_alt_n, u_n, off_n, valid_n, near_n, worst, rounding, flips):
    """Print why d-row ``row`` is not one flip: the closest scalar multiple of ANY operand row (the shape-only
    test), and which piece of evidence that candidate lacks."""
    bb, s_, h_ = row
    flat = ops.reshape(-1, ops.shape[-1])
    alpha, fit = _closest_scalar_multiple(flat, rvec, tol)
    hk = gqa_kv_head(h_, h_q, h_kv) if by_q_row else h_
    ob_all, os_all, oh_all = torch.unravel_index(torch.arange(flat.shape[0], device=flat.device), ops.shape[:-1])
    in_reduction = (ob_all == bb) & ((oh_all == hk) if by_q_row else ((oh_all >= h_ * g) & (oh_all < (h_ + 1) * g)))
    # Identical operand rows (the suite's negative-score q rows) tie: prefer the one inside the row's reduction whose
    # intermediate sits at a midpoint (it carries the evidence), then any inside the reduction, then the rest.
    c_of = os_all if by_q_row else ((oh_all - h_ * g) * s_op + os_all)
    at_midpoint = torch.zeros_like(in_reduction)
    at_midpoint[in_reduction] = (valid_n & near_n)[c_of[in_reduction]]
    best = int((fit + torch.where(at_midpoint, 0.0, torch.where(in_reduction, 1e-6, 2e-6))).argmin().item())
    ob, os_, oh = _unravel(best, ops.shape[:-1])
    a = alpha[best].item()
    k = math.log2(abs(a) / flip_unit) if a != 0 else float("nan")
    msg = (f"%%%% '{tag}': d-row {row} is NOT one flipped fp8 intermediate. The closest scalar multiple of ANY operand row is "
           f"row {(ob, os_, oh)} x {a:+.5f} (= 2^{k:.2f} x flip_unit {flip_unit:.5g}), max |residual| / tol = {fit[best].item():.2f}")
    mags = _fp8_finite_magnitudes(fp8_dtype, rvec.device, torch.float32)
    widest = (mags[1:] - mags[:-1]).max().item()
    if a != 0 and abs(a) / flip_unit > widest:
        msg += f"; no two adjacent {fp8_dtype} codes are more than {widest:g} apart, so no single flip can be a {abs(a) / flip_unit:g}-code step (before gain)"

    def verdict(c):
        """Why candidate ``c`` (valid, at a midpoint) does not reproduce the row."""
        if worst[c] > 1.0:
            return f"which predicts the row to max |residual| / tol = {worst[c].item():.2f}"
        if rounding[c] > 1.0:
            return (f"which predicts the row only to {rounding[c].item():.2f} x the output's own rounding "
                    f"(atol + half a {out_dtype} code spacing from each side; inside the ordinary tolerance, max |residual| / tol = {worst[c].item():.2f})")
        return (f"which fits the row as {flips[c].item():.2f} flips of itself (one flip fits within 1 +- {_ONE_FLIP_BAND:g}; "
                f"per element it stays inside the output's rounding, {rounding[c].item():.2f} x, and the tolerance, {worst[c].item():.2f} x)")

    if ob != bb:
        msg += f"; but that row is in batch {ob}, and {kind}{(bb, s_, h_)} reduces over batch {bb} only"
    elif by_q_row and oh != hk:
        msg += f"; but that row is kv head {oh}, and q head {h_} reads kv head {hk} only"
    elif not by_q_row and not (h_ * g <= oh < (h_ + 1) * g):
        msg += f"; but that row is q head {oh}, outside the GQA group of kv head {h_} (q heads {h_ * g}..{(h_ + 1) * g - 1})"
    else:
        c = os_ if by_q_row else (oh - h_ * g) * s_op + os_
        pos = _flip_position(by_q_row, row, c, g, s_op)
        xv, cr, ca, uv = x_n[c].item(), c_ref_n[c].item(), c_alt_n[c].item(), u_n[c].item()
        if not bool(valid_n[c]):
            msg += f"; but position (b, hq, i, j)={pos} is masked / out of range in the reference (P = dS = 0 there by construction, nothing rounds)"
        elif uv == 0:
            msg += f"; but the reference's intermediate there, (b, hq, i, j)={pos} x={xv:+.7g}, is exactly the {fp8_dtype} code {cr:+g}: no adjacent code could have been rounded to instead"
        elif not bool(near_n[c]):
            msg += (f"; but the reference's intermediate there, (b, hq, i, j)={pos} x={xv:+.7g}, lies {(off_n[c] / abs(uv)).item():.4f} code spacings from the "
                    f"midpoint of {fp8_dtype} codes {cr:+g} and {ca:+g} (a flip needs <= {_MIDPOINT_WINDOW:g})")
        else:
            msg += (f"; the adjacent {fp8_dtype} codes there, (b, hq, i, j)={pos} x={xv:+.7g}, are {cr:+g} -> {ca:+g}: a step of {uv:+g} x flip_unit x gain, "
                    f"{verdict(c)}, not the fitted {a:+.5f}")
    cand = valid_n & near_n
    if bool(cand.any()):
        score = torch.maximum(rounding, torch.maximum(worst, (flips - 1.0).abs() / _ONE_FLIP_BAND))
        c = int(torch.where(cand, score, torch.full_like(score, float("inf"))).argmin().item())
        msg += (f". The best VALID near-midpoint candidate, (b, hq, i, j)={_flip_position(by_q_row, row, c, g, s_op)} x={x_n[c].item():+.7g} "
                f"({c_ref_n[c].item():+g} -> {c_alt_n[c].item():+g}), leaves max |residual| / tol = {worst[c].item():.2f}, "
                f"{rounding[c].item():.2f} x the output's rounding, and fits as {flips[c].item():.2f} flips")
    else:
        msg += ". No valid position of the row's reduction has an intermediate within the midpoint window"
    print(msg)


def assert_close_fp8_grad(actual, expected, atol, rtol, tag, budget=1e-5, keys=None, operand=None, flip_unit=None, intermediates=None, kind=None, fp8_dtype=None, out_dtype=None):
    """assert_close for fp8 SDPA outputs/gradients with a small mismatch budget.

    Returns None, or -- when the row cap was lifted on the evidence of ``intermediates`` -- the list of flip
    attributions (dicts with ``row``, ``position`` = (b, q head, i, j), ``x``, ``c_ref``, ``c_alt``, ``u``, ...).

    The kernel and the reference each quantize P (with s_scale) and dS (with dP_scale) to fp8
    independently, from fp32 values that differ by ~1e-6 relative (exp2/FMA vs torch.exp). When
    such a value lands on an e4m3 rounding midpoint (seen: P*s_scale = 25.00008 between 24 and
    26; dS*dP_scale = 15.0 / -13.0), the two sides round to different fp8 codes and every
    gradient element fed by that value moves by one e4m3 step * |dO| (or |K|, |Q|): 0.09-0.26 for
    the suite's data, more than atol + rtol*|ref| for near-cancelling elements. That is not a
    kernel defect and it hits a handful of elements out of 10^6-10^7, so a budget of 1e-5 of the
    elements (at least 1) is tolerated. NaN/Inf are never budgeted, and a real bug (a tile,
    >= 128*d elements) is orders of magnitude above the budget.

    One flipped P or dS value at (i, j) feeds a whole d-row of the outputs -- O/dQ row i, dK/dV
    row j -- so when that row's Q/K/V/dO is dense the flip costs d elements, not a handful. The
    negative-score q rows (inject_negative_score_rows) all share one dense q vector, so their
    dS(i, j) terms into dK row j add coherently: on GB300 CI (test310, e4m3, d192) a single
    dS*dP_scale within 1e-5 relative of an e4m3 midpoint moved all 192 elements of one dK row by
    0.25 (the reference recomputed with P*(1+1e-5) reproduces the kernel's row exactly). The
    same holds for O with e5m2 P (2 mantissa bits: one step is 25% of P, 0.15-0.3 in O for
    |v|~2). The budget therefore also counts affected d-rows.

    A flip is an event per P (or dS) VALUE, so the number of rows it can touch scales with
    rows * keys, not with rows alone: ``keys`` is the reduction length feeding each d-row (s_kv
    for O/dQ, s_q for dK/dV) and the row budget is 1e-5 of rows * keys (at least 1). Measured
    rate on B200: e5m2 d256 no-mask s=1093 (test_sdpa_fp8_fwd_L0[test136]) flipped 79 of 61,208
    O rows = 1.2e-6 per P value, 2 elements per row (only the largest-|v| dims clear atol). The
    row budget is only honoured while every deviation stays within 4*atol -- one fp8 P step
    times |v| for this suite's sparse-int data (measured flips: 0.13-0.375).  The cap alone is
    not a physical bound: the negative-score q rows have amplitude m = 8 at d192 / attn_scale
    1/8, so one dS step of 2^-4 * dP_descale moves a dK row by 0.5 -- the sm107 212-SM lane's
    test310 (2026-09-15: dK row 1464 uniformly +-0.5, dQ row 575 by <= 0.25 = the same flip at
    (575, 1464) seen through K).  So a bad row above the cap may instead be PROVED to be one
    flip, from the reference's own intermediates: the caller passes ``intermediates`` (a
    callable re-running the reference with ``return_intermediates=<selection>`` -- evaluated only
    when the cap fails, on the bad rows only), ``fp8_dtype`` (the intermediate's dtype),
    ``operand`` (the dequantized BSHD operand: V for O, K for dQ, Q for dK, dO for dV) and
    ``flip_unit`` (the intermediate's descale: s_descale for O/dV, dP_descale for dQ/dK), and
    ``out_dtype`` (the dtype ``actual`` / ``expected`` were quantized to before being dequantized,
    ``torch_otype``; default ``fp8_dtype``).  ``kind`` names the output (``O`` / ``dQ`` / ``dK`` /
    ``dV``; default ``tag``).  The evidence is consulted only when the row budget holds and no
    value is non-finite -- it replaces the magnitude cap, never the row count.  Each bad row must
    then have, at a VALID position of its own reduction (same batch, a q head of the same GQA
    group, unmasked), a reference intermediate within 1/32 of a code spacing of the midpoint
    between its fp8 code and the adjacent one, whose flip -- ``(c_alt - c_ref) * flip_unit *
    gain * operand_row`` -- reproduces the row (``_flip_evidence``): within the ordinary
    tolerance; per element within the output's own rounding (``atol`` plus half an ``out_dtype``
    code spacing at |actual| and at |expected| -- both sides are dequantized output codes, each
    rounded once, so one flip may show as 0 or 2 flips on an element whose output spacing is
    twice the flip); and as a whole, the least-squares number of such flips fitted to the row
    being 1 within 1/2 (a genuine row's rounding residuals are unbiased; two flips fit at 2, and
    a deviation that merely stays inside a tolerance several flips wide -- ``rtol * |expected|``
    on a row of large gradients -- fits at whatever multiple it is).  Below the output's rounding
    nothing is observable: a deviation between 1/2 and 3/2 flips of the SAME intermediate cannot
    be told from one flip, because the outputs are fp8 codes.  Only a dense [b, s, h, d] layout
    supports this; a packed (ragged) output keeps the plain cap, its [T, h, d] rows do not name
    (b, s) and the reference's intermediates are indexed by (b, h, s).  The cap is what rejects a
    real defect: the GitHub
    #981 d256 fp8 corruption (garbage weights on a 16-key band) measured on the unfixed kernel
    as 11,048 elements / 253 rows / max 2.0 (MHA), 1,591 / 37 / 0.42 (e4m3) and 562 / 15 / 0.69
    (s=1024) -- the row counts sit inside the rows * keys budget, the magnitudes never do, and a
    band of wrong weights is a sum over many positions, so no single code flip reproduces it.
    A masked key, another batch's or another head's row, two rows each one flip away, three
    flips on a row whose ``rtol * |expected|`` is two flips wide, or a step no adjacent pair of
    codes has (8192 in e4m3) all fit a scalar-multiple test and none fits this one.  Do not
    raise the cap for a "slightly" worse flip; pass the intermediates and let the evidence
    decide.
    """
    actual = actual.detach().float()
    expected = expected.detach().float()
    diff = (actual - expected).abs()
    # NaN compares false against the tolerance, so non-finite values on either side are flagged explicitly.
    nonfinite = ~torch.isfinite(actual) | ~torch.isfinite(expected)
    bad = (diff > atol + rtol * expected.abs()) | nonfinite
    n_bad = int(bad.sum().item())
    if n_bad == 0:
        return
    allowed = max(1, int(actual.numel() * budget))
    bad_rows = bad.reshape(-1, bad.shape[-1]).any(dim=-1)
    n_bad_rows = int(bad_rows.sum().item())
    allowed_rows = max(1, int(bad_rows.numel() * (keys or 1) * budget))
    max_diff = diff.max().item()
    idx = tuple(bad.nonzero()[0].tolist())
    print(
        f"%%%% '{tag}': {n_bad:,} of {actual.numel():,} elements outside atol={atol} rtol={rtol} (budget {allowed}) "
        f"in {n_bad_rows:,} of {bad_rows.numel():,} d-rows (budget {allowed_rows} for keys={keys}); "
        f"first at {idx}: actual={actual[idx].item():+.5f} expected={expected[idx].item():+.5f}; max |diff|={max_diff:.4f} (row-budget cap {4 * atol})"
    )
    within_budget = n_bad <= allowed or (n_bad_rows <= allowed_rows and max_diff <= 4 * atol)
    fits = None
    if not within_budget and intermediates is not None and n_bad_rows <= allowed_rows and not bool(nonfinite.any()):
        kind = kind or tag
        if kind not in _FLIP_KINDS:
            raise ValueError(f"kind={kind!r} (from tag={tag!r}) must be one of {sorted(_FLIP_KINDS)} to attribute flips")
        if operand is None or flip_unit is None or fp8_dtype is None:
            raise TypeError("intermediates= needs operand=, flip_unit= and fp8_dtype= as well")
        out_dtype = out_dtype or fp8_dtype
        if actual.dim() != 4:
            print(f"%%%% '{tag}': flip attribution needs a dense [b, s, h, d] output (got {tuple(actual.shape)}); the row-budget cap applies")
        else:
            fits = _flip_evidence(actual, expected, bad_rows, atol, rtol, tag=tag, kind=kind, operand=operand, flip_unit=flip_unit,
                                  fp8_dtype=fp8_dtype, out_dtype=out_dtype, intermediates=intermediates)
            for f in fits or ():
                print(f"%%%% '{tag}': d-row {f['row']} is ONE flipped fp8 intermediate at (b, hq, i, j)={f['position']}: x={f['x']:+.7g} rounds to "
                      f"{fp8_dtype} code {f['c_ref']:+g}, the kernel to {f['c_alt']:+g} ({f['off_mid_spacings']:.5f} spacings from their midpoint); "
                      f"step {f['u']:+g} x flip_unit {float(flip_unit):.5g} x gain {f['gain']:.5g} = {f['step']:+.5g} x operand row {f['operand_row']} "
                      f"reproduces the row to max |residual| / tol = {f['worst']:.2f}, {f['rounding']:.2f} x the output's own rounding, "
                      f"and fits it as {f['flips']:.3f} flips"
                      + (f" ({f['n_tied']} other valid near-midpoint candidate(s) of the reduction, identical operand rows, fit equally)" if f["n_tied"] else ""))
            within_budget = fits is not None
    if not within_budget or bool(nonfinite.any()):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
    return fits


def create_paged_container_and_block_table(tensor, block_size, seq_lens=None):
    """Page a dense [B, H, S, D] tensor: container [B*blocks, H, block_size, D] (page p of
    batch b at pool index p*B + b) + a row-major (B, 1, blocks, 1) int32 table.

    ``seq_lens`` (per-batch lengths, the padding-mask values bound alongside) is OPT-IN
    poison: when given, every page at or past ``ceil(seq_lens[b] / block_size)`` -- a page
    no length reaches -- is NaN-filled, so an engine that dereferences a dead table slot
    poisons its O through 0 * NaN and fails the compare. Only engines that promise to
    skip dead slots may be tested this way: the FROST paged kernels issue a TMA-OOB page
    -1 there, while the backend engine loads whole tile-rounded page ranges through the
    table and masks the scores, so it needs finite data in every table slot (its default
    ``exec_sdpa_fp8`` path passes ``seq_lens=None``; see ``ExecConfig.paged_nan_dead_pages``).
    Rows INSIDE the last live page but past the length keep their finite data either way:
    the paged-attention contract lets a kernel load and mask them."""
    B, H, S, D = tensor.shape
    blocks_per_batch = math.ceil(S / block_size)

    padding_seq = blocks_per_batch * block_size - S
    if padding_seq > 0:
        zeros = torch.zeros(B, H, padding_seq, D, device="cuda", dtype=tensor.dtype)
        cat_tensor = torch.cat((tensor, zeros), dim=2)
    else:
        cat_tensor = tensor

    container = torch.cat(cat_tensor.chunk(blocks_per_batch, dim=2), dim=0)
    if seq_lens is not None:
        for b, length in enumerate(seq_lens):
            live_pages = math.ceil(int(length) / block_size)
            for p in range(live_pages, blocks_per_batch):
                container[p * B + b] = float("nan")

    table_size = math.ceil(S / block_size)
    block_table_temp = torch.linspace(0, B * table_size - 1, B * table_size, device="cuda", dtype=torch.int32).reshape(table_size, 1, B, 1)
    block_table_temp = torch.transpose(block_table_temp, 0, 2)

    block_table = (torch.zeros(blocks_per_batch * B, device="cuda", dtype=torch.int32).as_strided((B, 1, blocks_per_batch, 1), (blocks_per_batch, blocks_per_batch, 1, 1)))
    block_table.copy_(block_table_temp)

    return (container, block_table)

def exec_sdpa_fp8(cfg, request, cudnn_handle):
    """Build, run and validate one fp8 SDPA forward (and backward when cfg.is_train) against fp8_ref."""
    if request.config.option.dryrun:
        pytest.skip("dryrun")
    perf = request.config.getoption("--perf")

    # The conftest binds the handle to its own torch.cuda.Stream(); every tensor below
    # (inputs, page tables, the NaN-prefilled outputs) is produced on torch's current
    # stream. Run the graphs on that same stream, like fp16.py does, or the two are
    # unordered: under GPU contention (xdist -n 8, CI) the prefill lands after cuDNN's
    # memset/kernel -> NaN dSink/O, and a page table can be read before it is written.
    cudnn.set_stream(handle=cudnn_handle, stream=torch.cuda.current_stream().cuda_stream)

    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.14.0":
        pytest.skip("SDPA FP8 requires cuDNN 9.14.0 or higher")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("SDPA FP8 requires Hopper or higher")

    is_cu_seq_len = bool(getattr(cfg, 'is_cu_seq_len', False))
    with_ragged_offset_multiplier = bool(getattr(cfg, 'with_ragged_offset_multiplier', False))
    if (is_cu_seq_len or with_ragged_offset_multiplier) and cudnn_version < "9.25.0":
        pytest.skip("cu_seq_len / ragged offset multiplier for FP8 requires cuDNN 9.25.0 or higher (unified engine)")
    if is_cu_seq_len:
        assert cfg.is_infer, "is_cu_seq_len=True is forward-only (cu_seq_len is not plumbed for backward)"

    torch_itype = cfg.data_type
    torch_otype = cfg.output_type if hasattr(cfg, 'output_type') and cfg.output_type else cfg.data_type
    cudnn_itype = convert_to_cudnn_type(torch_itype)
    cudnn_otype = convert_to_cudnn_type(torch_otype)

    b = cfg.batches
    h_q, h_k, h_v = cfg.h_q, cfg.h_k, cfg.h_v
    s_qo, s_kv = cfg.s_q, cfg.s_kv
    d_qk, d_vo = cfg.d_qk, cfg.d_v
    block_size = cfg.block_size if cfg.is_paged else 0
    # Block-scaled O (sf_o): the FROST d128 epilogue (SM100 and newer) serves
    # dense, unpaged, non-ragged d_qk = d_v = 128 forward graphs; fold the knob
    # to 0 elsewhere so the drawn config still runs as a plain fp8 forward
    # instead of skipping on "unsupported forward graph".
    o_block_scale = int(getattr(cfg, 'o_block_scale', 0) or 0)
    block_scaled_o_arch = torch.cuda.get_device_capability()[0] >= 10
    if o_block_scale and not (
        block_scaled_o_arch and cfg.is_infer and not cfg.is_paged and not getattr(cfg, 'is_ragged', False) and d_qk == 128 and d_vo == 128
    ):
        o_block_scale = 0
    if o_block_scale == 16 and not hasattr(torch, "float4_e2m1fn_x2"):
        o_block_scale = 0  # the packed FP4 dtype arrived in torch 2.8; older builds run the draw as plain fp8
    if o_block_scale:
        # The quantized O container is E4M3 (mxfp8) or the E2M1 byte container (nvfp4);
        # the reference stays fp32 and is compared after dequantization.
        torch_otype = torch.float8_e4m3fn
    deterministic = cfg.is_determin if hasattr(cfg, 'is_determin') else False
    is_ragged = cfg.is_ragged if hasattr(cfg, 'is_ragged') else False
    left_bound = cfg.left_bound if hasattr(cfg, 'left_bound') else None
    right_bound = cfg.right_bound if hasattr(cfg, 'right_bound') else None
    diag_align = cfg.diag_align if hasattr(cfg, 'diag_align') else None
    with_sink_token = cfg.with_sink_token if hasattr(cfg, 'with_sink_token') else False
    rescale_threshold = cfg.rescale_threshold if hasattr(cfg, 'rescale_threshold') and cfg.rescale_threshold is not None else 4.0

    attn_scale = 0.125

    is_paged = block_size > 0

    seq_len_q_list = cfg.seq_len_q if hasattr(cfg, 'seq_len_q') and cfg.seq_len_q else []
    seq_len_kv_list = cfg.seq_len_kv if hasattr(cfg, 'seq_len_kv') and cfg.seq_len_kv else []

    if is_ragged and torch.cuda.get_device_capability()[0] == 9:
        # GitHub #1009: the SM90 FP8 forward kernel hangs on query tiles with no valid
        # keys -- a batch with seq_len_kv == 0, or a TOP_LEFT left window bound that
        # ends past the batch's keys. Per-batch lengths are runtime data, so the
        # frontend cannot decline this at build time; skip instead of hanging CI.
        top_left = diag_align is None or diag_align == cudnn.diagonal_alignment.TOP_LEFT
        for q_len, kv_len in zip(seq_len_q_list, seq_len_kv_list):
            past_keys = top_left and left_bound is not None and q_len > kv_len + left_bound
            if q_len > 0 and (kv_len == 0 or past_keys):
                pytest.skip(f"SM90 FP8 forward hangs on query rows with no valid keys (seq_len_q={q_len}, seq_len_kv={kv_len}, left_bound={left_bound}; GitHub #1009)")

    if is_ragged:
        seq_len_q_gpu = torch.tensor(seq_len_q_list, dtype=torch.int32, device="cuda").view(-1)
        seq_len_kv_gpu = torch.tensor(seq_len_kv_list, dtype=torch.int32, device="cuda").view(-1)
        # Guaranteed capacity tail (> total tokens); convert_uniform_to_packed
        # NaN-fills it, so engines that read past the last ragged offset fail
        # deterministically (GitHub #624). First-class total_q/total_kv widen
        # the capacity further when set.
        max_t_q = getattr(cfg, "total_q", None) or packed_token_capacity(seq_len_q_list)
        max_t_kv = getattr(cfg, "total_kv", None) or packed_token_capacity(seq_len_kv_list)

        # With the ragged offset multiplier, offsets are stored in coarser units
        # (divided by the per-tensor multiplier; always divides evenly) and the
        # engine scales them back to element offsets.
        q_off_mult = d_qk if with_ragged_offset_multiplier else 1
        k_off_mult = d_qk if with_ragged_offset_multiplier else 1
        v_off_mult = d_vo if with_ragged_offset_multiplier else 1
        o_off_mult = d_vo if with_ragged_offset_multiplier else 1
        q_ragged_offset_gpu = (prefix_sum(seq_len_q_gpu) * h_q * d_qk // q_off_mult).to(torch.int64)
        k_ragged_offset_gpu = (prefix_sum(seq_len_kv_gpu) * h_k * d_qk // k_off_mult).to(torch.int64)
        v_ragged_offset_gpu = (prefix_sum(seq_len_kv_gpu) * h_v * d_vo // v_off_mult).to(torch.int64)
        o_ragged_offset_gpu = (prefix_sum(seq_len_q_gpu) * h_q * d_vo // o_off_mult).to(torch.int64)
        stats_ragged_offset_gpu = (prefix_sum(seq_len_q_gpu) * h_q * 1).to(torch.int64)
        if is_cu_seq_len:
            cu_seq_len_q_gpu = prefix_sum(seq_len_q_gpu).to(torch.int32).view(-1)
            cu_seq_len_kv_gpu = prefix_sum(seq_len_kv_gpu).to(torch.int32).view(-1)

    # Build forward graph (always needed)
    try:
        graph_fwd = generate_graph_fwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, block_size, is_ragged=is_ragged, left_bound=left_bound, right_bound=right_bound, diag_align=diag_align, with_sink_token=with_sink_token, is_cu_seq_len=is_cu_seq_len, with_ragged_offset_multiplier=with_ragged_offset_multiplier, implementation=cfg.implementation,
                                       max_total_seq_len_q=max_t_q if (is_ragged and getattr(cfg, "declare_total_seq_len", False)) else None,
                                       max_total_seq_len_kv=max_t_kv if (is_ragged and getattr(cfg, "declare_total_seq_len", False)) else None,
                                       o_block_scale=o_block_scale)
        graph_fwd.validate()
        graph_fwd.build_operation_graph()
        graph_fwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph_fwd.check_support()
        graph_fwd.build_plans()
        note_frost_routing(graph_fwd, label="fp8-fwd")
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"unsupported forward graph: {e}")
    except Exception as e:
        pytest.fail(f"Error building forward graph: {e}")

    rng_data = torch.Generator(device="cuda").manual_seed(cfg.rng_data_seed)

    # Use sparse small integers for better low-precision testing
    q_gen = create_sparse_int_tensor((b, s_qo, h_q, d_qk), torch.float, rng_data)
    k_gen = create_sparse_int_tensor((b, s_kv, h_k, d_qk), torch.float, rng_data)
    v_gen = create_sparse_int_tensor((b, s_kv, h_v, d_vo), torch.float, rng_data)
    if not perf:
        # keep at least a few q rows in the deeply-negative-score regime (see
        # inject_negative_score_rows); must run before the amax/descale computation.
        # for ragged, sample only rows the packing step keeps (s >= seq_len is dropped)
        valid_q_rows = (torch.arange(s_qo, device="cuda")[None, :, None] < seq_len_q_gpu[:, None, None]).expand(b, s_qo, h_q) if is_ragged else None
        inject_negative_score_rows(q_gen, k_gen, rng_data, attn_scale=attn_scale, head_axis=2, valid_rows=valid_q_rows)  # bshd

    q_amax = q_gen.abs().max().item()
    k_amax = k_gen.abs().max().item()
    v_amax = v_gen.abs().max().item()
    s_amax = 1.0

    q_fp8 = (q_gen * get_fp8_scale_factor(q_amax, torch_itype)).to(torch_itype)
    k_fp8 = (k_gen * get_fp8_scale_factor(k_amax, torch_itype)).to(torch_itype)
    v_fp8 = (v_gen * get_fp8_scale_factor(v_amax, torch_itype)).to(torch_itype)

    q_descale_gpu = torch.tensor([get_fp8_descale_factor(q_amax, torch_itype)], dtype=torch.float, device="cuda")
    k_descale_gpu = torch.tensor([get_fp8_descale_factor(k_amax, torch_itype)], dtype=torch.float, device="cuda")
    v_descale_gpu = torch.tensor([get_fp8_descale_factor(v_amax, torch_itype)], dtype=torch.float, device="cuda")
    s_scale_gpu = torch.tensor([get_fp8_scale_factor(s_amax, torch_itype)], dtype=torch.float, device="cuda")
    s_descale_gpu = torch.tensor([get_fp8_descale_factor(s_amax, torch_itype)], dtype=torch.float, device="cuda")

    # Create sink_token tensor if needed
    sink_token_gpu = None
    if with_sink_token:
        sink_token_gpu = torch.randn((1, h_q, 1, 1), dtype=torch.float, device="cuda", generator=rng_data) * 0.5

    # Paged: the per-batch lengths the padding mask binds. A "padded" config draws them
    # (partial last pages, zero-length sequences, dead pages past each length); a full
    # config binds every batch at its maximum, as before.
    paged_seq_len_q = list(seq_len_q_list) if (is_paged and seq_len_q_list) else [s_qo] * b
    paged_seq_len_kv = list(seq_len_kv_list) if (is_paged and seq_len_kv_list) else [s_kv] * b

    # Compute forward reference (also computes o_amax internally)
    if is_ragged:
        seq_len_q_ref = torch.tensor(seq_len_q_list, dtype=torch.int32, device="cuda")
        seq_len_kv_ref = torch.tensor(seq_len_kv_list, dtype=torch.int32, device="cuda")
        padding = (seq_len_q_ref, seq_len_kv_ref)
    elif is_paged:
        seq_len_q_ref = torch.tensor(paged_seq_len_q, dtype=torch.int32, device="cuda")
        seq_len_kv_ref = torch.tensor(paged_seq_len_kv, dtype=torch.int32, device="cuda")
        padding = (seq_len_q_ref, seq_len_kv_ref)
    else:
        padding = None

    if perf:
        o_amax = 1.0
    else:
        def ref_fwd(return_intermediates=False):
            # One closure, so a flip attribution re-runs the reference with exactly these arguments.
            return compute_ref(q_fp8, k_fp8, v_fp8, attn_scale=attn_scale,
                               q_descale=q_descale_gpu, k_descale=k_descale_gpu, v_descale=v_descale_gpu,
                               s_scale=s_scale_gpu, s_descale=s_descale_gpu, torch_itype=torch_itype,
                               torch_otype=torch_otype, padding=padding,
                               left_bound=left_bound, right_bound=right_bound, diag_align=diag_align,
                               sink_token=sink_token_gpu, rescale_threshold=rescale_threshold,
                               return_intermediates=return_intermediates)
        o_ref, stats_ref, o_amax = ref_fwd()

    if o_block_scale == 16:
        # NVFP4 global scale: put the tensor amax at the top of the E4M3 x E2M1
        # range (block scale <= 448 when the block amax / 6 is scaled by it).
        o_scale_val = (448.0 * 6.0) / max(o_amax, 1e-6) * 0.5
    elif o_block_scale == 32:
        o_scale_val = 1.0  # UE8M0 block scales absorb the range; no global scale needed
    else:
        o_scale_val = get_fp8_scale_factor(o_amax, torch_otype)
    o_scale_gpu = torch.tensor([o_scale_val], dtype=torch.float, device="cuda")

    # Prepare GPU input tensors (pack for ragged, page for paged)
    q_gpu = q_fp8
    k_gpu = k_fp8
    v_gpu = v_fp8

    if is_ragged:
        q_gpu = convert_uniform_to_packed(torch.einsum("bshd->bhsd", q_fp8), torch.tensor(seq_len_q_list, dtype=torch.int32, device="cuda"), max_t_q)
        k_gpu = convert_uniform_to_packed(torch.einsum("bshd->bhsd", k_fp8), torch.tensor(seq_len_kv_list, dtype=torch.int32, device="cuda"), max_t_kv)
        v_gpu = convert_uniform_to_packed(torch.einsum("bshd->bhsd", v_fp8), torch.tensor(seq_len_kv_list, dtype=torch.int32, device="cuda"), max_t_kv)

    if is_paged:
        k_gpu_bhsd = torch.einsum('bshd->bhsd', k_fp8).contiguous()
        v_gpu_bhsd = torch.einsum('bshd->bhsd', v_fp8).contiguous()
        # Dead-page NaN poison is opt-in (cfg.paged_nan_dead_pages): the FROST-pinned
        # paged tests set it; the default path serves the backend engine, whose
        # contract lets it read (and mask) every page the table names.
        poison_lens = paged_seq_len_kv if getattr(cfg, "paged_nan_dead_pages", False) else None
        container_k_gpu, k_block_table_gpu = create_paged_container_and_block_table(k_gpu_bhsd, block_size, seq_lens=poison_lens)
        container_v_gpu, v_block_table_gpu = create_paged_container_and_block_table(v_gpu_bhsd, block_size, seq_lens=poison_lens)

    # Allocate forward output tensors
    if is_ragged:
        o_gpu = torch.full((max_t_q, h_q, d_vo), float('nan'), dtype=torch_otype, device="cuda")
        stats_gpu = torch.full((max_t_q, h_q, 1), float('nan'), dtype=torch.float, device="cuda")
    elif o_block_scale == 16:
        o_gpu = torch.full((b, s_qo, h_q, d_vo // 2), 0x7F, dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
        stats_gpu = torch.full((b, h_q, s_qo, 1), float('nan'), dtype=torch.float, device="cuda")
    else:
        o_gpu = torch.full((b, s_qo, h_q, d_vo), float('nan'), dtype=torch_otype, device="cuda")
        stats_gpu = torch.full((b, h_q, s_qo, 1), float('nan'), dtype=torch.float, device="cuda")
    sf_o_gpu = None
    if o_block_scale:
        sf_o_gpu = torch.full(block_scaled_o_sf_dims(b, h_q, s_qo, d_vo, o_block_scale), 0xAA, dtype=torch.uint8, device="cuda")

    o_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")

    variant_pack = {
        int(GraphFwdUid.q): q_gpu,
        int(GraphFwdUid.k): k_gpu,
        int(GraphFwdUid.v): v_gpu,
        int(GraphFwdUid.q_descale): q_descale_gpu,
        int(GraphFwdUid.k_descale): k_descale_gpu,
        int(GraphFwdUid.v_descale): v_descale_gpu,
        int(GraphFwdUid.s_descale): s_descale_gpu,
        int(GraphFwdUid.s_scale): s_scale_gpu,
        int(GraphFwdUid.o_scale): o_scale_gpu,
        int(GraphFwdUid.o): o_gpu,
        int(GraphFwdUid.stats): stats_gpu,
        int(GraphFwdUid.o_amax): o_amax_gpu,
    }

    if is_paged:
        variant_pack[int(GraphFwdUid.k)] = container_k_gpu
        variant_pack[int(GraphFwdUid.v)] = container_v_gpu
        variant_pack[int(GraphFwdUid.kv_seq_len)] = torch.tensor(paged_seq_len_kv, device="cuda", dtype=torch.int32)
        variant_pack[int(GraphFwdUid.q_seq_len)] = torch.tensor(paged_seq_len_q, device="cuda", dtype=torch.int32)
        variant_pack[int(GraphFwdUid.k_block_table)] = k_block_table_gpu
        variant_pack[int(GraphFwdUid.v_block_table)] = v_block_table_gpu

    if is_ragged:
        if is_cu_seq_len:
            variant_pack[int(GraphFwdUid.cu_seq_len_q)] = cu_seq_len_q_gpu
            variant_pack[int(GraphFwdUid.cu_seq_len_kv)] = cu_seq_len_kv_gpu
        else:
            variant_pack[int(GraphFwdUid.q_seq_len)] = torch.tensor(seq_len_q_list, dtype=torch.int32, device="cuda").view(-1)
            variant_pack[int(GraphFwdUid.kv_seq_len)] = torch.tensor(seq_len_kv_list, dtype=torch.int32, device="cuda").view(-1)
        variant_pack[int(GraphFwdUid.q_ragged_offset)] = q_ragged_offset_gpu
        variant_pack[int(GraphFwdUid.k_ragged_offset)] = k_ragged_offset_gpu
        variant_pack[int(GraphFwdUid.v_ragged_offset)] = v_ragged_offset_gpu
        variant_pack[int(GraphFwdUid.o_ragged_offset)] = o_ragged_offset_gpu
        variant_pack[int(GraphFwdUid.stats_ragged_offset)] = stats_ragged_offset_gpu

    if with_sink_token:
        variant_pack[int(GraphFwdUid.sink_token)] = sink_token_gpu
    if o_block_scale:
        variant_pack[int(GraphFwdUid.sf_o)] = sf_o_gpu

    workspace = torch.empty(graph_fwd.get_workspace_size(), dtype=torch.uint8, device="cuda")
    if perf:
        times_ms = time_execution(graph_fwd.execute, variant_pack, workspace, cudnn_handle)
        print(f"@@@@ FP8 Fwd graph_fwd.execute avg_time_ms={times_ms.mean().item():.3f}")
        profile_execution(graph_fwd.execute, variant_pack, workspace, cudnn_handle)
    graph_fwd.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    # Compare forward output
    if not perf and o_block_scale:
        from .block_scale_o_ref import dequantize_e2m1, quantize_o_mxfp8, quantize_o_nvfp4, unpack_e2m1, unswizzle_128x4

        # o_ref came back quantized to the per-tensor container; rebuild the fp32
        # O (times the kernel's global scale) from the un-quantized reference.
        o_ref32, _, _ = compute_ref(q_fp8, k_fp8, v_fp8, attn_scale=attn_scale,
                                     q_descale=q_descale_gpu, k_descale=k_descale_gpu, v_descale=v_descale_gpu,
                                     s_scale=s_scale_gpu, s_descale=s_descale_gpu, torch_itype=torch_itype,
                                     torch_otype=torch.float32, padding=padding,
                                     left_bound=left_bound, right_bound=right_bound, diag_align=diag_align,
                                     sink_token=sink_token_gpu, rescale_threshold=rescale_threshold, quantize_o=False)
        o_ref_scaled = o_ref32.float() * o_scale_val  # (b, s, h, d)
        c_used = d_vo // o_block_scale
        sf_log = unswizzle_128x4(sf_o_gpu)[:, :, :s_qo, :c_used].permute(0, 2, 1, 3)  # (b, s, h, c)
        if o_block_scale == 16:
            codes = unpack_e2m1(o_gpu.view(torch.uint8)).reshape(b, s_qo, h_q, d_vo)
            o_deq = (dequantize_e2m1(codes).reshape(b, s_qo, h_q, c_used, 16) * sf_log.view(torch.float8_e4m3fn).float()[..., None]).reshape(b, s_qo, h_q, d_vo)
            _, _, ref_q = quantize_o_nvfp4(o_ref_scaled)
        else:
            o_deq = (o_gpu.float().reshape(b, s_qo, h_q, c_used, 32) * torch.pow(2.0, sf_log.float() - 127.0)[..., None]).reshape(b, s_qo, h_q, d_vo)
            _, _, ref_q = quantize_o_mxfp8(o_ref_scaled)
        assert not torch.isnan(o_deq).any(), "NaN in dequantized block-scaled O"
        assert bool((unswizzle_128x4(sf_o_gpu)[:, :, s_qo:, :] == 0).all().item()), "sf_o pad rows past s_q must be zero"
        floor = (ref_q - o_ref_scaled).abs().max().item()
        atol = max((0.125 if torch_itype == torch.float8_e5m2 else 0.08) * o_scale_val, 3.0 * floor)
        assert_close_fp8_grad(o_deq, o_ref_scaled, atol, 0.2, tag="O(block-scaled)", keys=s_kv)
    elif not perf:
        if is_ragged:
            o_ref_comp = convert_uniform_to_packed(torch.einsum("bshd->bhsd", o_ref), seq_len_q_ref, max_t_q)
        else:
            o_ref_comp = o_ref

        o_gpu_float = o_gpu.detach().float() * get_fp8_descale_factor(o_amax, torch_otype)
        o_ref_float = o_ref_comp.detach().float() * get_fp8_descale_factor(o_amax, torch_otype)

        if is_ragged:
            t_idx = sum(seq_len_q_list)
            o_gpu_float[t_idx:] = 0
            o_ref_float[t_idx:] = 0
        elif is_paged:
            # Padded (dead) query rows: the reference holds 0 there and the engines
            # write O := 0 (the FROST dense padded-Q trim) or leave the buffer; compare
            # the live rows only.  O is [b, s_qo, h_q, d_vo].
            dead_q = torch.arange(s_qo, device="cuda")[None, :] >= seq_len_q_ref[:, None]
            o_gpu_float[dead_q] = 0
            o_ref_float[dead_q] = 0

        # E5M2 is less precise than E4M3, so its P quantization needs one wider step.
        atol, rtol = (0.125 if torch_itype == torch.float8_e5m2 else 0.08), 0.2
        # A bad row above the row cap is attributed to one fp8 P flip against the reference's own intermediates,
        # re-run on the bad rows only.  Ragged keeps the plain cap: the packed [T, h, d] rows do not name (b, s)
        # and the reference's intermediates are indexed by (b, h, s).
        fwd_intermediates = None if is_ragged else (lambda selection: ref_fwd(return_intermediates=selection)[3])
        assert_close_fp8_grad(o_gpu_float, o_ref_float, atol, rtol, tag="O", keys=s_kv, operand=v_gen, flip_unit=s_descale_gpu.item(),
                              intermediates=fwd_intermediates, fp8_dtype=torch_itype, out_dtype=torch_otype)

    # Backward pass
    if not cfg.is_infer:
        dO_gen = create_sparse_int_tensor((b, s_qo, h_q, d_vo), torch.float, rng_data)
        dO_amax = dO_gen.abs().max().item()
        dO_fp8 = (dO_gen * get_fp8_scale_factor(dO_amax, torch_itype)).to(torch_itype)

        o_descale_gpu = torch.tensor([get_fp8_descale_factor(o_amax, torch_otype)], dtype=torch.float, device="cuda")
        dO_descale_gpu = torch.tensor([get_fp8_descale_factor(dO_amax, torch_itype)], dtype=torch.float, device="cuda")

        if perf:
            dP_amax = 1.0
            dQ_amax = 1.0
            dK_amax = 1.0
            dV_amax = 1.0
        else:
            # Get unpacked BSHD references for backward
            if is_ragged:
                q_ref_bwd = torch.einsum("bhsd->bshd", convert_packed_to_uniform(q_gpu, seq_len_q_ref, s_qo))
                k_ref_bwd = torch.einsum("bhsd->bshd", convert_packed_to_uniform(k_gpu, seq_len_kv_ref, s_kv))
                v_ref_bwd = torch.einsum("bhsd->bshd", convert_packed_to_uniform(v_gpu, seq_len_kv_ref, s_kv))
                o_ref_bwd = torch.einsum("bhsd->bshd", convert_packed_to_uniform(o_gpu, seq_len_q_ref, s_qo))
                dO_ref_bwd = dO_fp8
            else:
                q_ref_bwd = q_gpu
                k_ref_bwd = k_gpu
                v_ref_bwd = v_gpu
                o_ref_bwd = o_gpu
                dO_ref_bwd = dO_fp8

            padding_bwd = (seq_len_q_ref, seq_len_kv_ref) if is_ragged else None

            def ref_bwd(return_intermediates=False):
                # One closure, so a flip attribution re-runs the reference with exactly these arguments.
                return compute_ref_backward(
                    q_ref_bwd, k_ref_bwd, v_ref_bwd, o_ref_bwd, dO_ref_bwd, attn_scale=attn_scale,
                    q_descale=q_descale_gpu, k_descale=k_descale_gpu, v_descale=v_descale_gpu,
                    s_scale=s_scale_gpu, s_descale=s_descale_gpu, torch_itype=torch_itype,
                    o_descale=o_descale_gpu, dO_descale=dO_descale_gpu,
                    torch_otype=torch_otype, padding=padding_bwd,
                    left_bound=left_bound, right_bound=right_bound, diag_align=diag_align,
                    sink_token=sink_token_gpu,
                    # Ragged packs stats differently, so keep softmax there.
                    stats=(None if is_ragged else stats_gpu),
                    return_intermediates=return_intermediates,
                )
            dQ_ref, dK_ref, dV_ref, dSink_token_ref, dP_amax, dQ_amax, dK_amax, dV_amax = ref_bwd()

        dP_descale_gpu = torch.tensor([get_fp8_descale_factor(dP_amax, torch_itype)], dtype=torch.float, device="cuda")
        dQ_scale_gpu = torch.tensor([get_fp8_scale_factor(dQ_amax, torch_otype)], dtype=torch.float, device="cuda")
        dK_scale_gpu = torch.tensor([get_fp8_scale_factor(dK_amax, torch_otype)], dtype=torch.float, device="cuda")
        dV_scale_gpu = torch.tensor([get_fp8_scale_factor(dV_amax, torch_otype)], dtype=torch.float, device="cuda")
        dP_scale_gpu = torch.tensor([get_fp8_scale_factor(dP_amax, torch_otype)], dtype=torch.float, device="cuda")

        try:
            graph_bwd = generate_graph_bwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, deterministic, is_ragged=is_ragged, left_bound=left_bound, right_bound=right_bound, diag_align=diag_align, with_sink_token=with_sink_token)
            graph_bwd.validate()
            graph_bwd.build_operation_graph()
            graph_bwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            graph_bwd.check_support()
            graph_bwd.build_plans()
            note_frost_routing(graph_bwd, label="fp8-bwd")
        except cudnn.cudnnGraphNotSupportedError as e:
            pytest.skip(f"unsupported backward graph: {e}")
        except Exception as e:
            pytest.fail(f"Error building backward graph: {e}")

        if is_ragged:
            dO_gpu = convert_uniform_to_packed(torch.einsum("bshd->bhsd", dO_fp8), torch.tensor(seq_len_q_list, dtype=torch.int32, device="cuda"), max_t_q)
        else:
            dO_gpu = dO_fp8

        if is_ragged:
            dQ_gpu = torch.full((max_t_q, h_q, d_qk), float('nan'), dtype=torch_otype, device="cuda")
            dK_gpu = torch.full((max_t_kv, h_k, d_qk), float('nan'), dtype=torch_otype, device="cuda")
            dV_gpu = torch.full((max_t_kv, h_v, d_vo), float('nan'), dtype=torch_otype, device="cuda")
        else:
            dQ_gpu = torch.full((b, s_qo, h_q, d_qk), float('nan'), dtype=torch_otype, device="cuda")
            dK_gpu = torch.full((b, s_kv, h_k, d_qk), float('nan'), dtype=torch_otype, device="cuda")
            dV_gpu = torch.full((b, s_kv, h_v, d_vo), float('nan'), dtype=torch_otype, device="cuda")
        dQ_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
        dK_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
        dV_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
        dP_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
        dSink_token_gpu = None
        if with_sink_token:
            dSink_token_gpu = torch.full((1, h_q, 1, 1), float('nan'), dtype=torch.float, device="cuda")

        variant_pack_bwd = {
            int(GraphBwdUid.q): q_gpu, int(GraphBwdUid.k): k_gpu, int(GraphBwdUid.v): v_gpu,
            int(GraphBwdUid.o): o_gpu, int(GraphBwdUid.dO): dO_gpu, int(GraphBwdUid.stats): stats_gpu,
            int(GraphBwdUid.q_descale): q_descale_gpu, int(GraphBwdUid.k_descale): k_descale_gpu,
            int(GraphBwdUid.v_descale): v_descale_gpu, int(GraphBwdUid.o_descale): o_descale_gpu,
            int(GraphBwdUid.dO_descale): dO_descale_gpu, int(GraphBwdUid.s_descale): s_descale_gpu,
            int(GraphBwdUid.s_scale): s_scale_gpu, int(GraphBwdUid.dP_descale): dP_descale_gpu,
            int(GraphBwdUid.dP_scale): dP_scale_gpu, int(GraphBwdUid.dQ_scale): dQ_scale_gpu,
            int(GraphBwdUid.dK_scale): dK_scale_gpu, int(GraphBwdUid.dV_scale): dV_scale_gpu,
            int(GraphBwdUid.dQ): dQ_gpu, int(GraphBwdUid.dK): dK_gpu, int(GraphBwdUid.dV): dV_gpu,
            int(GraphBwdUid.dQ_amax): dQ_amax_gpu, int(GraphBwdUid.dK_amax): dK_amax_gpu,
            int(GraphBwdUid.dV_amax): dV_amax_gpu, int(GraphBwdUid.dP_amax): dP_amax_gpu,
        }

        if is_ragged:
            variant_pack_bwd[int(GraphBwdUid.q_seq_len)] = seq_len_q_gpu
            variant_pack_bwd[int(GraphBwdUid.kv_seq_len)] = seq_len_kv_gpu
            variant_pack_bwd[int(GraphBwdUid.q_ragged_offset)] = q_ragged_offset_gpu
            variant_pack_bwd[int(GraphBwdUid.k_ragged_offset)] = k_ragged_offset_gpu
            variant_pack_bwd[int(GraphBwdUid.v_ragged_offset)] = v_ragged_offset_gpu
            variant_pack_bwd[int(GraphBwdUid.o_ragged_offset)] = o_ragged_offset_gpu
            variant_pack_bwd[int(GraphBwdUid.stats_ragged_offset)] = stats_ragged_offset_gpu
            variant_pack_bwd[int(GraphBwdUid.dO_ragged_offset)] = o_ragged_offset_gpu

        if with_sink_token:
            variant_pack_bwd[int(GraphBwdUid.sink_token)] = sink_token_gpu
            variant_pack_bwd[int(GraphBwdUid.dSink_token)] = dSink_token_gpu

        workspace_bwd = torch.empty(graph_bwd.get_workspace_size(), dtype=torch.uint8, device="cuda")
        if perf:
            times_ms = time_execution(graph_bwd.execute, variant_pack_bwd, workspace_bwd, cudnn_handle)
            print(f"@@@@ FP8 Bwd graph.execute avg_time_ms={times_ms.mean().item():.3f}")
            profile_execution(graph_bwd.execute, variant_pack_bwd, workspace_bwd, cudnn_handle)
        graph_bwd.execute(variant_pack_bwd, workspace_bwd, handle=cudnn_handle)
        torch.cuda.synchronize()

        if deterministic:
            dQ_gpu_rerun = dQ_gpu.clone().detach()
            dK_gpu_rerun = dK_gpu.clone().detach()
            dV_gpu_rerun = dV_gpu.clone().detach()

            dQ_gpu = torch.fill_(dQ_gpu, float("nan"))
            dK_gpu = torch.fill_(dK_gpu, float("nan"))
            dV_gpu = torch.fill_(dV_gpu, float("nan"))
            torch.cuda.synchronize()
            graph_bwd.execute(variant_pack_bwd, workspace_bwd, handle=cudnn_handle)
            torch.cuda.synchronize()

            determin_err_count = 0
            determin_err_count += exact_equal(dQ_gpu, dQ_gpu_rerun, tag="dQ_determin", disp_elems=10)
            determin_err_count += exact_equal(dK_gpu, dK_gpu_rerun, tag="dK_determin", disp_elems=10)
            determin_err_count += exact_equal(dV_gpu, dV_gpu_rerun, tag="dV_determin", disp_elems=10)

            if determin_err_count != 0:
                print("@@@@ Overall result: FAILED, determinism check failed - outputs differ between runs.")
                pytest.fail("determinism check failed", pytrace=False)
            print("@@@@ Determinism check: PASSED, dQ, dK, dV bitwise match between runs.")

        if not perf:
            if is_ragged:
                dQ_ref = convert_uniform_to_packed(torch.einsum("bshd->bhsd", dQ_ref), seq_len_q_ref, max_t_q)
                dK_ref = convert_uniform_to_packed(torch.einsum("bshd->bhsd", dK_ref), seq_len_kv_ref, max_t_kv)
                dV_ref = convert_uniform_to_packed(torch.einsum("bshd->bhsd", dV_ref), seq_len_kv_ref, max_t_kv)

            dQ_out = dQ_gpu.detach().float() * get_fp8_descale_factor(dQ_amax, torch_otype)
            dK_out = dK_gpu.detach().float() * get_fp8_descale_factor(dK_amax, torch_otype)
            dV_out = dV_gpu.detach().float() * get_fp8_descale_factor(dV_amax, torch_otype)

            dQ_ref_float = dQ_ref.detach().float() * get_fp8_descale_factor(dQ_amax, torch_otype)
            dK_ref_float = dK_ref.detach().float() * get_fp8_descale_factor(dK_amax, torch_otype)
            dV_ref_float = dV_ref.detach().float() * get_fp8_descale_factor(dV_amax, torch_otype)

            if is_ragged:
                t_idx_q = sum(seq_len_q_list)
                dQ_out[t_idx_q:] = 0
                dQ_ref_float[t_idx_q:] = 0
                t_idx_kv = sum(seq_len_kv_list)
                dK_out[t_idx_kv:] = 0
                dK_ref_float[t_idx_kv:] = 0
                dV_out[t_idx_kv:] = 0
                dV_ref_float[t_idx_kv:] = 0

            # E5M2 is less precise than E4M3, so its P quantization needs one wider step.
            atol, rtol = (0.125 if torch_itype == torch.float8_e5m2 else 0.08), 0.2
            # As for O: a bad row above the cap must be ONE dS (dQ, dK) or P (dV) flip by the reference's own
            # intermediates; ragged keeps the plain cap (packed rows do not name (b, s)), and so does h_k != h_v
            # (compute_ref_backward's intermediates need one GQA group size for dK and dV; the suites draw h_k == h_v).
            bwd_intermediates = None if (is_ragged or h_k != h_v) else (lambda selection: ref_bwd(return_intermediates=selection)[8])
            assert_close_fp8_grad(dQ_out, dQ_ref_float, atol, rtol, tag="dQ", keys=s_kv, operand=k_gen, flip_unit=dP_descale_gpu.item(),
                                  intermediates=bwd_intermediates, fp8_dtype=torch_itype, out_dtype=torch_otype)
            assert_close_fp8_grad(dK_out, dK_ref_float, atol, rtol, tag="dK", keys=s_qo, operand=q_gen, flip_unit=dP_descale_gpu.item(),
                                  intermediates=bwd_intermediates, fp8_dtype=torch_itype, out_dtype=torch_otype)
            assert_close_fp8_grad(dV_out, dV_ref_float, atol, rtol, tag="dV", keys=s_qo, operand=dO_gen, flip_unit=s_descale_gpu.item(),
                                  intermediates=bwd_intermediates, fp8_dtype=torch_itype, out_dtype=torch_otype)

            if with_sink_token:
                torch.testing.assert_close(dSink_token_gpu, dSink_token_ref, atol=0.02, rtol=0.2)

    # Print hash and stats for determinism verification
    print_tensor_stats(o_gpu, tag="o_gpu")
    print_tensor_stats(o_amax_gpu, tag="o_amax_gpu")
    if not cfg.is_infer:
        print_tensor_stats(stats_gpu, tag="stats_gpu")
        print_tensor_stats(dQ_gpu, tag="dQ_gpu")
        print_tensor_stats(dK_gpu, tag="dK_gpu")
        print_tensor_stats(dV_gpu, tag="dV_gpu")
        print_tensor_stats(dQ_amax_gpu, tag="dQ_amax_gpu")
        print_tensor_stats(dK_amax_gpu, tag="dK_amax_gpu")
        print_tensor_stats(dV_amax_gpu, tag="dV_amax_gpu")
        print_tensor_stats(dP_amax_gpu, tag="dP_amax_gpu")
        if with_sink_token:
            print_tensor_stats(dSink_token_gpu, tag="dSink_token_gpu")
