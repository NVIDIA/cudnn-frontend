# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Try to import TransformerEngine (>= 2.12) for MXFP8 quantization
# NOTE: TE must be imported BEFORE cudnn to avoid library loading conflicts
from looseversion import LooseVersion

import cudnn
import pytest
import torch
import math
from enum import IntEnum

from .helpers import (
    exact_equal,
    fill_sparse_small_int,
    inject_negative_score_rows,
    time_execution,
    profile_execution,
    note_frost_routing,
)
from .mxfp8_ref import compute_ref, compute_ref_backward

# Torch-only MXFP8 block quantization + F8_128x4 swizzle (replicates the
# TransformerEngine MXFP8Quantizer / tex.swizzle_scales_for_gemm_ semantics —
# see mxfp8_quant.py for the source-level derivation).  TransformerEngine is
# no longer required to run the MXFP8 SDPA tests; TE-parity of these
# primitives is covered by test_mxfp8_quant.py (which only runs when TE is
# installed).
from .mxfp8_quant import (
    quantize_to_mxfp8,
)  # noqa: F401  (re-exported; mxfp8_ref imports it from here)

# Layout support: dense-full, plus a forward THD/ragged path
# (exec_sdpa_mxfp8_thd) following the engine contract from
# frost/test_sdpa_fwd_mxfp8_sm100.py::_run_thd — packed [T, H, D] tokens with
# per-operand ragged offsets, per-sequence 128-TILE-padded SF concatenated in
# cu_seqlens order (the engine derives the packed SF extent from the buffer's
# byte size), token-major TH1 ragged Stats, and optional
# sdpa_mxfp8(max_total_seq_len_q/kv=...) totals. Ragged capacity tails are
# NaN-poisoned (GitHub #624). Dense "padded" configs remain unsupported (the
# per-batch padding mask is untested on the dense mxfp8 surface), and there is
# no THD backward engine — cfg.is_train + is_ragged skips.

# fmt: off

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class GraphFwdUid(IntEnum):
    q = 0
    k = 1
    v = 2
    sf_q = 5
    sf_k = 6
    sf_v = 7
    o = 3
    stats = 4
    o_amax = 12
    sink_token = 13


class GraphBwdUid(IntEnum):
    q = 100
    q_t = 101
    k = 102
    k_t = 103
    v = 104
    o = 105
    dO = 106
    dO_t = 107
    dO_f16 = 108
    stats = 109

    sf_q = 110
    sf_q_t = 111
    sf_k = 112
    sf_k_t = 113
    sf_v = 114
    sf_dO = 115
    sf_dO_t = 116

    dQ = 117
    dK = 118
    dV = 119
    dQ_amax = 120
    dK_amax = 121
    dV_amax = 122
    sink_token = 123
    dSink_token = 124

# Helper to compare tensors with detailed output
def compare_tensors(actual, expected, atol, rtol, tag, disp_elems=10):
    actual_f32 = actual.float()
    mismatches = torch.where(torch.isclose(actual_f32, expected, rtol=rtol, atol=atol, equal_nan=True) == False)
    mismatch_cnt = mismatches[0].numel()
    num_elements = torch.numel(actual)

    if mismatch_cnt != 0:
        percentage = 100 * mismatch_cnt / num_elements
        print(f"\nComparing '{tag}' using rtol={rtol:.4e}, atol={atol:.4e}")
        combined = torch.stack(mismatches, dim=-1).tolist()
        for i, index in enumerate(combined[:disp_elems]):
            idx = tuple(index)
            gpu_val = actual_f32[idx].item()
            ref_val = expected[idx].item()
            diff = gpu_val - ref_val
            print(f"  idx{index}: {tag}_gpu={gpu_val:+.6e}, {tag}_ref={ref_val:+.6e}, diff={diff:+.2e}")
        print(f"Total {mismatch_cnt:,} mismatches ({percentage:.1f}%) for '{tag}'")
    else:
        print(f"'{tag}' within tolerance (rtol={rtol}, atol={atol})")

    return mismatch_cnt

def compare_amax(actual, expected, rtol=0.02, tag="amax"):
    amax_ref = torch.amax(torch.abs(expected)).item()
    amax_gpu = torch.amax(torch.abs(actual)).item()
    amax_diff = abs(amax_gpu - amax_ref)
    amax_atol = rtol * max(amax_ref, 1.0)
    print(f"amax: gpu={amax_gpu:.6e}, ref={amax_ref:.6e}, diff={amax_diff:.2e}, tol={amax_atol:.2e} for '{tag}'")
    return amax_diff < amax_atol

def compute_mxfp8_scale_dims(s, d, block_size=32):
    """
    Compute scale tensor dimensions for MXFP8.

    For Q/K: scale the d (hidden) dimension
    For V: scale the s (sequence) dimension (BMM2 contracts on s)

    F8_128x4 reordering requires:
    - Sequence dimension padded to multiple of 128
    - Scale dimension padded to multiple of 4
    """
    d_scale = ceil_div(d, block_size)
    s_scale = ceil_div(s, block_size)

    s_padded = ceil_div(s, 128) * 128
    d_scale_padded = ceil_div(d_scale, 4) * 4
    s_scale_padded = ceil_div(s_scale, 4) * 4
    d_padded = ceil_div(d, 128) * 128  # Must be multiple of 128 for F8_128x4

    return {
        "s_padded": s_padded,
        "d_scale": d_scale,
        "d_scale_padded": d_scale_padded,
        "s_scale": s_scale,
        "s_scale_padded": s_scale_padded,
        "d_padded": d_padded,
    }



def generate_graph_fwd(b, h_q, h_k, h_v,
                       s_qo, s_kv, d_qk, d_vo, attn_scale,
                       block_size=32,
                       cudnn_itype=cudnn.data_type.FP8_E4M3,
                       cudnn_otype=cudnn.data_type.HALF,
                       left_bound=None, right_bound=None, diag_align=None,
                       with_sink_token=False,
                       with_unfuse_fma=False,
                       implementation=cudnn.attention_implementation.AUTO):
    # Compute padded dimensions for F8_128x4 scale factors
    s_q_padded = ceil_div(s_qo, 128) * 128
    s_kv_padded = ceil_div(s_kv, 128) * 128
    d_qk_scale_padded = ceil_div(ceil_div(d_qk, block_size), 4) * 4
    d_vo_padded = ceil_div(d_vo, 128) * 128
    s_kv_scale_padded = ceil_div(ceil_div(s_kv, block_size), 4) * 4

    # Build graph
    graph = cudnn.pygraph(
        io_data_type=cudnn_itype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT
    )

    # Q, K, V tensors with BHSD layout
    # Stride: (s * h * d, d, h * d, 1) for interleaved layout
    q = graph.tensor(
        uid=GraphFwdUid.q,
        dim=(b, h_q, s_qo, d_qk),
        stride=(h_q * s_qo * d_qk, s_qo * d_qk, d_qk, 1),
        data_type=cudnn_itype
    )
    k = graph.tensor(
        uid=GraphFwdUid.k,
        dim=(b, h_k, s_kv, d_qk),
        stride=(h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1),
        data_type=cudnn_itype
    )
    v = graph.tensor(
        uid=GraphFwdUid.v,
        dim=(b, h_v, s_kv, d_vo),
        stride=(h_v * s_kv * d_vo, s_kv * d_vo, d_vo, 1),
        data_type=cudnn_itype
    )

    # Scale factor tensors (FP8_E8M0 with F8_128x4 reordering)
    # SF_Q: [B, H_q, S_q_padded, D_scale_padded], d_scale contiguous
    sf_q_dims = (b, h_q, s_q_padded, d_qk_scale_padded)
    sf_q_strides = (h_q * s_q_padded * d_qk_scale_padded, s_q_padded * d_qk_scale_padded, d_qk_scale_padded, 1)
    sf_q = graph.tensor(
        uid=GraphFwdUid.sf_q,
        dim=sf_q_dims,
        stride=sf_q_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # SF_K: [B, H_k, S_kv_padded, D_scale_padded], d_scale contiguous
    sf_k_dims = (b, h_k, s_kv_padded, d_qk_scale_padded)
    sf_k_strides = (h_k * s_kv_padded * d_qk_scale_padded, s_kv_padded * d_qk_scale_padded, d_qk_scale_padded, 1)
    sf_k = graph.tensor(
        uid=GraphFwdUid.sf_k,
        dim=sf_k_dims,
        stride=sf_k_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # SF_V: [B, H_v, S_scale_padded, D_v_padded], s_scale contiguous
    sf_v_dims = (b, h_v, s_kv_scale_padded, d_vo_padded)
    sf_v_strides = (h_v * s_kv_scale_padded * d_vo_padded, s_kv_scale_padded * d_vo_padded, d_vo_padded, 1)
    sf_v = graph.tensor(
        uid=GraphFwdUid.sf_v,
        dim=sf_v_dims,
        stride=sf_v_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # Create sink_token tensor if needed
    sink_token = None
    if with_sink_token:
        sink_token = graph.tensor(uid=GraphFwdUid.sink_token, dim=(1, h_q, 1, 1), stride=(h_q, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    # Call MXFP8 SDPA
    o, stats, amax_o = graph.sdpa_mxfp8(
        q=q, k=k, v=v,
        descale_q=sf_q, descale_k=sf_k, descale_v=sf_v,
        attn_scale=attn_scale,
        generate_stats=True,
        diagonal_alignment=diag_align if diag_align is not None else cudnn.diagonal_alignment.TOP_LEFT,
        diagonal_band_left_bound=left_bound,
        diagonal_band_right_bound=right_bound,
        sink_token=sink_token,
        unfuse_fma=with_unfuse_fma,
        implementation=implementation,
    )

    # Set output tensor properties
    o.set_uid(GraphFwdUid.o).set_output(True).set_dim((b, h_q, s_qo, d_vo)).set_stride((h_q * s_qo * d_vo, s_qo * d_vo, d_vo, 1)).set_data_type(cudnn_otype)
    stats.set_uid(GraphFwdUid.stats).set_output(True).set_dim((b, h_q, s_qo, 1)).set_stride((h_q * s_qo, s_qo, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_o.set_uid(GraphFwdUid.o_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

    return graph


def generate_graph_bwd(b, h_q, h_k, h_v,
                       s_qo, s_kv, d_qk, d_vo,
                       attn_scale, deterministic,
                       block_size=32,
                       cudnn_itype=cudnn.data_type.FP8_E4M3,
                       cudnn_otype=cudnn.data_type.HALF,
                       left_bound=None, right_bound=None, diag_align=None,
                       with_sink_token=False):
    # Compute padded dimensions for F8_128x4 scale factors
    s_qo_padded = ceil_div(s_qo, 128) * 128
    s_kv_padded = ceil_div(s_kv, 128) * 128
    d_qk_padded = ceil_div(d_qk, 128) * 128
    d_vo_padded = ceil_div(d_vo, 128) * 128
    s_qo_scale_padded = ceil_div(ceil_div(s_qo, block_size), 4) * 4
    s_kv_scale_padded = ceil_div(ceil_div(s_kv, block_size), 4) * 4
    d_qk_scale_padded = ceil_div(ceil_div(d_qk, block_size), 4) * 4
    d_vo_scale_padded = ceil_div(ceil_div(d_vo, block_size), 4) * 4

    # Create graph
    graph_bwd = cudnn.pygraph(
        io_data_type=cudnn_itype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT
    )

    # Create input tensors with BHSD contiguous layout
    q = graph_bwd.tensor(
        uid=GraphBwdUid.q,
        dim=(b, h_q, s_qo, d_qk),
        stride=(h_q * s_qo * d_qk, s_qo * d_qk, d_qk, 1),
        data_type=cudnn_itype
    )
    q_t = graph_bwd.tensor(
        uid=GraphBwdUid.q_t,
        dim=(b, h_q, s_qo, d_qk),
        stride=(h_q * s_qo * d_qk, s_qo * d_qk, d_qk, 1),
        data_type=cudnn_itype
    )
    k = graph_bwd.tensor(
        uid=GraphBwdUid.k,
        dim=(b, h_k, s_kv, d_qk),
        stride=(h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1),
        data_type=cudnn_itype
    )
    k_t = graph_bwd.tensor(
        uid=GraphBwdUid.k_t,
        dim=(b, h_k, s_kv, d_qk),
        stride=(h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1),
        data_type=cudnn_itype
    )
    v = graph_bwd.tensor(
        uid=GraphBwdUid.v,
        dim=(b, h_v, s_kv, d_vo),
        stride=(h_v * s_kv * d_vo, s_kv * d_vo, d_vo, 1),
        data_type=cudnn_itype
    )
    o = graph_bwd.tensor(
        uid=GraphBwdUid.o,
        dim=(b, h_q, s_qo, d_vo),
        stride=(h_q * s_qo * d_vo, s_qo * d_vo, d_vo, 1),
        data_type=cudnn.data_type.BFLOAT16
    )
    dO = graph_bwd.tensor(
        uid=GraphBwdUid.dO,
        dim=(b, h_q, s_qo, d_vo),
        stride=(h_q * s_qo * d_vo, s_qo * d_vo, d_vo, 1),
        data_type=cudnn_itype
    )
    dO_t = graph_bwd.tensor(
        uid=GraphBwdUid.dO_t,
        dim=(b, h_q, s_qo, d_vo),
        stride=(h_q * s_qo * d_vo, s_qo * d_vo, d_vo, 1),
        data_type=cudnn_itype
    )
    dO_f16 = graph_bwd.tensor(
        uid=GraphBwdUid.dO_f16,
        dim=(b, h_q, s_qo, d_vo),
        stride=(h_q * s_qo * d_vo, s_qo * d_vo, d_vo, 1),
        data_type=cudnn.data_type.BFLOAT16
    )
    stats = graph_bwd.tensor(
        uid=GraphBwdUid.stats,
        dim=(b, h_q, s_qo, 1),
        stride=(s_qo * h_q, s_qo, 1, 1),
        data_type=cudnn.data_type.FLOAT
    )

    # Create scale factor tensors with E8M0 dtype and F8_128x4 reordering
    # SF_Q: [B, H_q, S_qo_padded, D_qk_scale_padded]
    sf_q_dims = (b, h_q, s_qo_padded, d_qk_scale_padded)
    sf_q_strides = (h_q * s_qo_padded * d_qk_scale_padded, s_qo_padded * d_qk_scale_padded, d_qk_scale_padded, 1)
    sf_q = graph_bwd.tensor(
        uid=GraphBwdUid.sf_q,
        dim=sf_q_dims,
        stride=sf_q_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # SF_Q_T: [B, H_q, S_qo_scale_padded, D_qk_padded]
    sf_q_t_dims = (b, h_q, s_qo_scale_padded, d_qk_padded)
    sf_q_t_strides = (h_q * s_qo_scale_padded * d_qk_padded, s_qo_scale_padded * d_qk_padded, d_qk_padded, 1)
    sf_q_t = graph_bwd.tensor(
        uid=GraphBwdUid.sf_q_t,
        dim=sf_q_t_dims,
        stride=sf_q_t_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # SF_K: [B, H_k, S_kv_padded, D_qk_scale_padded]
    sf_k_dims = (b, h_k, s_kv_padded, d_qk_scale_padded)
    sf_k_strides = (h_k * s_kv_padded * d_qk_scale_padded, s_kv_padded * d_qk_scale_padded, d_qk_scale_padded, 1)
    sf_k = graph_bwd.tensor(
        uid=GraphBwdUid.sf_k,
        dim=sf_k_dims,
        stride=sf_k_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # SF_K_T: [B, H_k, S_kv_scale_padded, D_qk_padded]
    sf_k_t_dims = (b, h_k, s_kv_scale_padded, d_qk_padded)
    sf_k_t_strides = (h_k * s_kv_scale_padded * d_qk_padded, s_kv_scale_padded * d_qk_padded, d_qk_padded, 1)
    sf_k_t = graph_bwd.tensor(
        uid=GraphBwdUid.sf_k_t,
        dim=sf_k_t_dims,
        stride=sf_k_t_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # SF_V: [B, H_v, S_kv_padded, D_vo_scale_padded]
    sf_v_dims = (b, h_v, s_kv_padded, d_vo_scale_padded)
    sf_v_strides = (h_v * s_kv_padded * d_vo_scale_padded, s_kv_padded * d_vo_scale_padded, d_vo_scale_padded, 1)
    sf_v = graph_bwd.tensor(
        uid=GraphBwdUid.sf_v,
        dim=sf_v_dims,
        stride=sf_v_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # SF_dO: [B, H_q, S_qo_padded, D_vo_scale_padded]
    sf_dO_dims = (b, h_q, s_qo_padded, d_vo_scale_padded)
    sf_dO_strides = (h_q * s_qo_padded * d_vo_scale_padded, s_qo_padded * d_vo_scale_padded, d_vo_scale_padded, 1)
    sf_dO = graph_bwd.tensor(
        uid=GraphBwdUid.sf_dO,
        dim=sf_dO_dims,
        stride=sf_dO_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # SF_dO_T: [B, H_q, S_qo_scale_padded, D_vo_padded]
    sf_dO_t_dims = (b, h_q, s_qo_scale_padded, d_vo_padded)
    sf_dO_t_strides = (h_q * s_qo_scale_padded * d_vo_padded, s_qo_scale_padded * d_vo_padded, d_vo_padded, 1)
    sf_dO_t = graph_bwd.tensor(
        uid=GraphBwdUid.sf_dO_t,
        dim=sf_dO_t_dims,
        stride=sf_dO_t_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # Create sink_token and dSink_token tensors if needed
    sink_token = None
    dSink_token = None
    if with_sink_token:
        sink_token = graph_bwd.tensor(uid=GraphBwdUid.sink_token, dim=(1, h_q, 1, 1), stride=(h_q, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
        dSink_token = graph_bwd.tensor(uid=GraphBwdUid.dSink_token, dim=(1, h_q, 1, 1), stride=(h_q, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    dQ, dK, dV, amax_dQ, amax_dK, amax_dV = graph_bwd.sdpa_mxfp8_backward(
        q=q, q_T=q_t, k=k, k_T=k_t, v=v,
        o_f16=o, dO_f16=dO_f16, dO=dO, dO_T=dO_t,
        stats=stats,
        descale_q=sf_q, descale_q_T=sf_q_t, descale_k=sf_k, descale_k_T=sf_k_t, descale_v=sf_v,
        descale_dO=sf_dO, descale_dO_T=sf_dO_t,
        attn_scale=attn_scale,
        use_deterministic_algorithm=deterministic,
        diagonal_alignment=diag_align if diag_align is not None else cudnn.diagonal_alignment.TOP_LEFT,
        left_bound=left_bound,
        right_bound=right_bound,
        sink_token=sink_token,
        dSink_token=dSink_token,
    )

    # Set output tensor properties
    dQ.set_uid(GraphBwdUid.dQ).set_output(True).set_dim((b, h_q, s_qo, d_qk)).set_stride((h_q * s_qo * d_qk, s_qo * d_qk, d_qk, 1)).set_data_type(cudnn_otype)
    dK.set_uid(GraphBwdUid.dK).set_output(True).set_dim((b, h_k, s_kv, d_qk)).set_stride((h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1)).set_data_type(cudnn_otype)
    dV.set_uid(GraphBwdUid.dV).set_output(True).set_dim((b, h_v, s_kv, d_vo)).set_stride((h_v * s_kv * d_vo, s_kv * d_vo, d_vo, 1)).set_data_type(cudnn_otype)

    amax_dQ.set_uid(GraphBwdUid.dQ_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dK.set_uid(GraphBwdUid.dK_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dV.set_uid(GraphBwdUid.dV_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

    # Mark dSink_token as output if using sink_token
    if with_sink_token:
        dSink_token.set_output(True)

    return graph_bwd

def _quantize_seq(t_1hsd, h, s, d, torch_itype, block_size, *, columnwise):
    """Per-sequence MXFP8 quantization for the THD packing.

    Returns (fp8 data [1, h, s, d], per-elem dequant scale [1, h, s, d],
    SF tiles [h, n_tiles, tile_bytes] uint8) with n_tiles = ceil(s/128): the
    quantizer's F8_128x4 atom padding rounds the S extent up to a multiple of
    128, which is exactly the engine's per-sequence-TILE-padded SF layout."""
    if columnwise:
        # 4 scale rows per 128-token tile x padded d columns.
        tile_bytes = 4 * (ceil_div(d, 128) * 128)
    else:
        tile_bytes = 128 * (ceil_div(ceil_div(d, block_size), 4) * 4)
    if s == 0:
        empty = t_1hsd.new_zeros((1, h, 0, d))
        return empty.to(torch_itype), empty.float(), torch.zeros((h, 0, tile_bytes), dtype=torch.uint8, device=t_1hsd.device)
    data_d, dq_d, swz_d, data_s, dq_s, swz_s = quantize_to_mxfp8(t_1hsd, 1, h, s, d, block_size, torch_itype, with_ref=True)
    n_tiles = ceil_div(s, 128)
    # dq stays in the quantizer's raw (b*h, s, d) shape — that is what
    # mxfp8_ref.compute_ref consumes.
    if columnwise:
        return data_s, dq_s, swz_s.view(torch.uint8).reshape(h, n_tiles, tile_bytes)
    return data_d, dq_d, swz_d.view(torch.uint8).reshape(h, n_tiles, tile_bytes)


def exec_sdpa_mxfp8_thd(cfg, request, cudnn_handle):
    """Forward THD/ragged MXFP8 SDPA: packed tokens + ragged offsets + packed
    per-sequence-TILE-padded SF, per-batch lengths in plain or cu form, and
    the first-class total_q/total_kv capacities (NaN-poisoned tails)."""
    from .random_config import packed_token_capacity

    perf = request.config.getoption("--perf")
    if perf:
        pytest.skip("perf mode not wired for the mxfp8 THD path")
    if cfg.is_train:
        pytest.skip("MXFP8 SDPA not supported: no THD backward engine")
    # THD mxfp8 is served only by the opt-in FROST engine
    # (sdpa_fwd_prefill_sm100_mxfp8). The native backend passes check_support
    # for these graphs but cannot execute them (NaN output at normal shapes,
    # device hang on sub-tile shapes) — skip rather than exercise that
    # known support gap. This only READS the FE feature flag; the suite never
    # sets environment variables.
    import os
    if os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "0").strip().lower() not in ("1", "true", "yes", "on"):
        pytest.skip("MXFP8 THD requires the opt-in FROST engine (set CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1)")

    b = cfg.batches
    h_q, h_k, h_v = cfg.h_q, cfg.h_k, cfg.h_v
    s_q_max, s_kv_max = cfg.s_q, cfg.s_kv
    d_qk, d_vo = cfg.d_qk, cfg.d_v
    block_size = 32
    attn_scale = 1.0 / math.sqrt(d_qk)
    left_bound = getattr(cfg, 'left_bound', None)
    right_bound = getattr(cfg, 'right_bound', None)
    diag_align = getattr(cfg, 'diag_align', None)
    with_sink_token = getattr(cfg, 'with_sink_token', False)
    rescale_threshold = cfg.rescale_threshold if cfg.rescale_threshold is not None else 4.0

    torch_itype = cfg.data_type or torch.float8_e4m3fn
    torch_otype = cfg.output_type or torch.bfloat16
    if torch_itype == torch.float8_e4m3fn:
        cudnn_itype = cudnn.data_type.FP8_E4M3
    elif torch_itype == torch.float8_e5m2:
        cudnn_itype = cudnn.data_type.FP8_E5M2
    else:
        pytest.skip(f"Unsupported input type: {torch_itype}")
    cudnn_otype = cudnn.data_type.HALF if torch_otype == torch.float16 else cudnn.data_type.BFLOAT16

    seq_len_q, seq_len_kv = list(cfg.seq_len_q), list(cfg.seq_len_kv)
    # All-zero sides are legal configs (t_q == 0: nothing to compute; t_kv == 0:
    # every live row is dead and must come back exact 0) — no skipping.
    t_q, t_kv = sum(seq_len_q), sum(seq_len_kv)
    max_t_q = cfg.total_q or packed_token_capacity(seq_len_q)
    max_t_kv = cfg.total_kv or packed_token_capacity(seq_len_kv)

    cu_q, cu_kv = [0], [0]
    for s in seq_len_q:
        cu_q.append(cu_q[-1] + s)
    for s in seq_len_kv:
        cu_kv.append(cu_kv[-1] + s)

    # Per-sequence data gen + quantization; pack tokens and SF tiles in
    # cu_seqlens order (Q/K rowwise, V columnwise — same as the dense path).
    rng_data = torch.Generator(device="cuda").manual_seed(cfg.rng_data_seed)
    q8_seqs, k8_seqs, v8_seqs, dqq_seqs, dqk_seqs, dqv_seqs = [], [], [], [], [], []
    sfq_seqs, sfk_seqs, sfv_seqs = [], [], []
    for i in range(b):
        s_q_i, s_kv_i = seq_len_q[i], seq_len_kv[i]
        q_f32 = torch.empty(1, h_q, s_q_i, d_qk, dtype=torch.float32, device="cuda")
        k_f32 = torch.empty(1, h_k, s_kv_i, d_qk, dtype=torch.float32, device="cuda")
        v_f32 = torch.empty(1, h_v, s_kv_i, d_vo, dtype=torch.float32, device="cuda")
        if s_q_i:
            fill_sparse_small_int(q_f32, rng_data, sparsity=0.8, abs_max=2)
        if s_kv_i:
            fill_sparse_small_int(k_f32, rng_data, sparsity=0.8, abs_max=2)
            fill_sparse_small_int(v_f32, rng_data, sparsity=0.8, abs_max=2)
        if s_q_i and s_kv_i:
            # keep a few q rows in the deeply-negative-score regime; must run
            # before quantization (same contract as the dense path)
            inject_negative_score_rows(q_f32, k_f32, rng_data, attn_scale=attn_scale)
        q8, dqq, sfq = _quantize_seq(q_f32, h_q, s_q_i, d_qk, torch_itype, block_size, columnwise=False)
        k8, dqk, sfk = _quantize_seq(k_f32, h_k, s_kv_i, d_qk, torch_itype, block_size, columnwise=False)
        v8, dqv, sfv = _quantize_seq(v_f32, h_v, s_kv_i, d_vo, torch_itype, block_size, columnwise=True)
        q8_seqs.append(q8); k8_seqs.append(k8); v8_seqs.append(v8)
        dqq_seqs.append(dqq); dqk_seqs.append(dqk); dqv_seqs.append(dqv)
        sfq_seqs.append(sfq); sfk_seqs.append(sfk); sfv_seqs.append(sfv)

    def _pack_tokens(x8_seqs, h, d, capacity_tokens, dt):
        # [1,h,s,d] per sequence -> packed [T,h,d] tokens in a
        # capacity-tokens buffer whose tail is NaN-poisoned (GitHub #624).
        # With zero live tokens the whole buffer is poison — legal, since the
        # clamped descriptors must never read it.
        stor = torch.full((capacity_tokens * h * d,), float("nan"), device="cuda", dtype=torch.float32).to(dt)
        pieces = [x.squeeze(0).permute(1, 0, 2).reshape(-1) for x in x8_seqs if x.numel()]
        if pieces:
            packed = torch.cat(pieces)
            stor[: packed.numel()] = packed
        return stor

    q_stor = _pack_tokens(q8_seqs, h_q, d_qk, max_t_q, torch_itype)
    k_stor = _pack_tokens(k8_seqs, h_k, d_qk, max_t_kv, torch_itype)
    v_stor = _pack_tokens(v8_seqs, h_v, d_vo, max_t_kv, torch_itype)
    # Packed SF: [h, total_tiles, tile_bytes] — per head, the sequences' tiles
    # in cu_seqlens order. The buffer is EXACTLY the packed layout (the engine
    # derives the packed tile extent from its byte size).
    def _nonempty_sf(sf, h):
        # Zero live tiles (all sequences empty on this side) still needs a real
        # device allocation to bind; one zeroed TILE is never read because the
        # engine's packed extent follows cu[B] == 0.
        if sf.shape[1] == 0:
            return torch.zeros((h, 1, sf.shape[2]), dtype=torch.uint8, device="cuda")
        return sf

    sfq_pk = _nonempty_sf(torch.cat(sfq_seqs, dim=1).contiguous(), h_q)
    sfk_pk = _nonempty_sf(torch.cat(sfk_seqs, dim=1).contiguous(), h_k)
    sfv_pk = _nonempty_sf(torch.cat(sfv_seqs, dim=1).contiguous(), h_v)

    o_stor = torch.full((max_t_q * h_q * d_vo,), float("nan"), device="cuda", dtype=torch.float32).to(torch_otype)
    stats_stor = torch.full((max_t_q * h_q,), float("nan"), dtype=torch.float32, device="cuda")
    amax_o_gpu = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device="cuda")

    stride_q = (s_q_max * h_q * d_qk, d_qk, h_q * d_qk, 1)
    stride_k = (s_kv_max * h_k * d_qk, d_qk, h_k * d_qk, 1)
    stride_v = (s_kv_max * h_v * d_vo, d_vo, h_v * d_vo, 1)
    stride_o = (s_q_max * h_q * d_vo, d_vo, h_q * d_vo, 1)

    seq_len_q_gpu = torch.tensor(seq_len_q, dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    seq_len_kv_gpu = torch.tensor(seq_len_kv, dtype=torch.int32, device="cuda").view(b, 1, 1, 1)
    cu_q_gpu = torch.tensor(cu_q, dtype=torch.int32, device="cuda").view(b + 1, 1, 1, 1)
    cu_kv_gpu = torch.tensor(cu_kv, dtype=torch.int32, device="cuda").view(b + 1, 1, 1, 1)
    ro_q = (torch.tensor(cu_q, dtype=torch.int64, device="cuda") * h_q * d_qk).view(b + 1, 1, 1, 1)
    ro_k = (torch.tensor(cu_kv, dtype=torch.int64, device="cuda") * h_k * d_qk).view(b + 1, 1, 1, 1)
    ro_v = (torch.tensor(cu_kv, dtype=torch.int64, device="cuda") * h_v * d_vo).view(b + 1, 1, 1, 1)
    ro_o = (torch.tensor(cu_q, dtype=torch.int64, device="cuda") * h_q * d_vo).view(b + 1, 1, 1, 1)
    ro_stats = (torch.tensor(cu_q, dtype=torch.int64, device="cuda") * h_q).view(b + 1, 1, 1, 1)

    try:
        graph = cudnn.pygraph(io_data_type=cudnn_itype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        tq = graph.tensor(dim=(b, h_q, s_q_max, d_qk), stride=stride_q, data_type=cudnn_itype, name="q")
        tk = graph.tensor(dim=(b, h_k, s_kv_max, d_qk), stride=stride_k, data_type=cudnn_itype, name="k")
        tv = graph.tensor(dim=(b, h_v, s_kv_max, d_vo), stride=stride_v, data_type=cudnn_itype, name="v")
        len_q_t = graph.tensor_like(cu_q_gpu if cfg.is_cu_seq_len else seq_len_q_gpu)
        len_kv_t = graph.tensor_like(cu_kv_gpu if cfg.is_cu_seq_len else seq_len_kv_gpu)
        t_ro_q, t_ro_k, t_ro_v, t_ro_o = (graph.tensor_like(ro_q) for _ in range(4))
        tq.set_ragged_offset(t_ro_q)
        tk.set_ragged_offset(t_ro_k)
        tv.set_ragged_offset(t_ro_v)

        def _sf_tensor(dims):
            # Dense-capacity declaration; the bound buffer holds the PACKED layout.
            return graph.tensor(
                dim=dims,
                stride=(dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1),
                data_type=cudnn.data_type.FP8_E8M0,
                reordering_type=cudnn.tensor_reordering.F8_128x4,
            )

        d_qk_sc = ceil_div(ceil_div(d_qk, block_size), 4) * 4
        d_vo_pad = ceil_div(d_vo, 128) * 128
        sf_q_t = _sf_tensor((b, h_q, ceil_div(s_q_max, 128) * 128, d_qk_sc))
        sf_k_t = _sf_tensor((b, h_k, ceil_div(s_kv_max, 128) * 128, d_qk_sc))
        sf_v_t = _sf_tensor((b, h_v, ceil_div(s_kv_max, 128) * 4, d_vo_pad))

        sdpa_kwargs = dict(
            q=tq, k=tk, v=tv,
            descale_q=sf_q_t, descale_k=sf_k_t, descale_v=sf_v_t,
            attn_scale=attn_scale,
            generate_stats=True,
            use_padding_mask=True,
            diagonal_alignment=diag_align if diag_align is not None else cudnn.diagonal_alignment.TOP_LEFT,
            diagonal_band_left_bound=left_bound,
            diagonal_band_right_bound=right_bound,
            unfuse_fma=getattr(cfg, 'with_unfuse_fma', False),
            implementation=cfg.implementation,
        )
        if cfg.is_cu_seq_len:
            sdpa_kwargs.update(cu_seq_len_q=len_q_t, cu_seq_len_kv=len_kv_t)
        else:
            sdpa_kwargs.update(seq_len_q=len_q_t, seq_len_kv=len_kv_t)
        if cfg.declare_total_seq_len:
            sdpa_kwargs.update(max_total_seq_len_q=max_t_q, max_total_seq_len_kv=max_t_kv)

        sink_token_gpu = None
        if with_sink_token:
            rng_sink = torch.Generator(device="cuda").manual_seed(cfg.rng_data_seed + 1000)
            sink_token_gpu = torch.randn((1, h_q, 1, 1), dtype=torch.float32, device="cuda", generator=rng_sink) * 0.5
            sink_t = graph.tensor_like(sink_token_gpu)
            sdpa_kwargs["sink_token"] = sink_t

        o_t, stats_t, amax_o_t = graph.sdpa_mxfp8(**sdpa_kwargs)
        o_t.set_output(True).set_dim((b, h_q, s_q_max, d_vo)).set_stride(stride_o).set_data_type(cudnn_otype)
        o_t.set_ragged_offset(t_ro_o)
        amax_o_t.set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
        # Ragged Stats: packed token-major TH1 ([t, h]; offsets = cu_q * h_q).
        stats_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        stats_t.set_dim((b, h_q, s_q_max, 1)).set_stride((s_q_max * h_q, 1, h_q, 1))
        t_ro_stats = graph.tensor_like(ro_stats, name="stats_ro")
        stats_t.set_ragged_offset(t_ro_stats)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()
        note_frost_routing(graph, label="mxfp8-fwd")
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"MXFP8 SDPA not supported: {e}")
    except Exception as e:
        # NOT_SUPPORTED can also surface at build_plans/finalize AFTER
        # check_support accepted the graph (backend support-check gap);
        # that is a waive, not a harness error.
        if "CUDNN_STATUS_NOT_SUPPORTED" in str(e):
            pytest.skip(f"MXFP8 SDPA not supported (at finalize): {e}")
        pytest.fail(f"Error building MXFP8 THD SDPA graph: {e}")

    # If the FROST engine declined this config, auto-selection falls back to
    # the native backend — which check_support-accepts THD mxfp8 graphs it
    # cannot execute (NaN output / device hang). Never run that path.
    if getattr(graph, "selected_engine", None) is None:
        pytest.skip("MXFP8 THD graph fell back to the native backend (FROST engine declined); native cannot execute THD mxfp8")

    # Bind the flat packed storages directly: the graph reads only the base
    # pointer, and a dense (b, h, s_max, d) view need not fit in a
    # total-token-capacity buffer.
    variant_pack = {
        tq: q_stor, tk: k_stor, tv: v_stor,
        sf_q_t: sfq_pk, sf_k_t: sfk_pk, sf_v_t: sfv_pk,
        len_q_t: cu_q_gpu if cfg.is_cu_seq_len else seq_len_q_gpu,
        len_kv_t: cu_kv_gpu if cfg.is_cu_seq_len else seq_len_kv_gpu,
        t_ro_q: ro_q, t_ro_k: ro_k, t_ro_v: ro_v, t_ro_o: ro_o,
        t_ro_stats: ro_stats,
        o_t: o_stor,
        stats_t: stats_stor,
        amax_o_t: amax_o_gpu,
    }
    if with_sink_token:
        variant_pack[sink_t] = sink_token_gpu

    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    # Per-sequence reference through the shared mxfp8 reference; compare the
    # packed live tokens only (dead rows from zero-length KV must be exact 0,
    # their Stats are engine-conventional and skipped).
    err = 0
    o_out = o_stor[: t_q * h_q * d_vo].reshape(t_q, h_q, d_vo)
    lse_out = stats_stor[: t_q * h_q].reshape(t_q, h_q)
    amax_ref = 0.0
    for i in range(b):
        lo, hi = cu_q[i], cu_q[i + 1]
        if lo == hi:
            continue
        o_rows = o_out[lo:hi]
        if seq_len_kv[i] == 0:
            err += compare_tensors(o_rows, torch.zeros_like(o_rows, dtype=torch.float32), 0.0, 0.0, f"output[seq{i}, dead]")
            continue
        o_ref, stats_ref = compute_ref(
            q8_seqs[i], k8_seqs[i], v8_seqs[i],
            dqq_seqs[i], dqk_seqs[i], dqv_seqs[i], attn_scale,
            torch_itype=torch_itype, output_type=torch_otype,
            left_bound=left_bound, right_bound=right_bound, diag_align=diag_align,
            sink_token=sink_token_gpu, rescale_threshold=rescale_threshold)
        amax_ref = max(amax_ref, o_ref.abs().max().item())
        # [1,h,s,d] -> packed [s,h,d]; [1,h,s,1] -> [s,h]
        err += compare_tensors(o_rows, o_ref.squeeze(0).permute(1, 0, 2).float(), 0.12, 0.20, f"output[seq{i}]")
        err += compare_tensors(lse_out[lo:hi], stats_ref.squeeze(0).squeeze(-1).permute(1, 0), 0.05, 0.05, f"stats[seq{i}]")
    assert err == 0, f"THD mismatch: {err} elements differ"
    amax_diff = abs(amax_o_gpu.item() - amax_ref)
    assert amax_diff <= 0.02 * max(amax_ref, 1.0), f"amax mismatch: gpu={amax_o_gpu.item():.6e} ref={amax_ref:.6e}"


def exec_sdpa_mxfp8(cfg, request, cudnn_handle):
    """Execute MXFP8 SDPA test."""
    if request.config.option.dryrun:
        pytest.skip("dry run mode")
    if getattr(cfg, 'is_ragged', False):
        return exec_sdpa_mxfp8_thd(cfg, request, cudnn_handle)
    perf = request.config.getoption("--perf")

    cudnn_version = LooseVersion(cudnn.backend_version_string())

    # Extract config
    b = cfg.batches
    h_q, h_k, h_v = cfg.h_q, cfg.h_k, cfg.h_v
    s_qo, s_kv = cfg.s_q, cfg.s_kv
    d_qk, d_vo = cfg.d_qk, cfg.d_v
    block_size = 32

    attn_scale = 1.0 / math.sqrt(d_qk)
    deterministic = getattr(cfg, 'is_determin', False)
    left_bound  = getattr(cfg, 'left_bound', None)
    right_bound = getattr(cfg, 'right_bound', None)
    diag_align  = getattr(cfg, 'diag_align', None)
    with_sink_token = getattr(cfg, 'with_sink_token', False)
    with_unfuse_fma = getattr(cfg, 'with_unfuse_fma', False)
    rescale_threshold = cfg.rescale_threshold if hasattr(cfg, 'rescale_threshold') and cfg.rescale_threshold is not None else 4.0

    # Get input/output types from config
    torch_itype = cfg.data_type if hasattr(cfg, 'data_type') and cfg.data_type else torch.float8_e4m3fn
    torch_otype = cfg.output_type if hasattr(cfg, 'output_type') and cfg.output_type else torch.bfloat16

    # Map torch types to cudnn types
    if torch_itype == torch.float8_e4m3fn:
        cudnn_itype = cudnn.data_type.FP8_E4M3
    elif torch_itype == torch.float8_e5m2:
        cudnn_itype = cudnn.data_type.FP8_E5M2
    else:
        pytest.skip(f"Unsupported input type: {torch_itype}")
    cudnn_otype = cudnn.data_type.HALF if torch_otype == torch.float16 else cudnn.data_type.BFLOAT16

    # Build forward graph
    try:
        graph_fwd = generate_graph_fwd(
            b, h_q, h_k, h_v,
            s_qo, s_kv, d_qk, d_vo, attn_scale,
            block_size,
            cudnn_itype, cudnn_otype,
            left_bound=left_bound, right_bound=right_bound, diag_align=diag_align,
            with_sink_token=with_sink_token,
            with_unfuse_fma=with_unfuse_fma,
            implementation=cfg.implementation,
        )
        graph_fwd.validate()
        graph_fwd.build_operation_graph()
        graph_fwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph_fwd.check_support()
        graph_fwd.build_plans()
        note_frost_routing(graph_fwd, label="mxfp8-fwd")
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"MXFP8 SDPA not supported: {e}")
    except Exception as e:
        # NOT_SUPPORTED can also surface at build_plans/finalize AFTER
        # check_support accepted the graph (backend support-check gap);
        # that is a waive, not a harness error.
        if "CUDNN_STATUS_NOT_SUPPORTED" in str(e):
            pytest.skip(f"MXFP8 SDPA not supported (at finalize): {e}")
        pytest.fail(f"Error building MXFP8 SDPA graph: {e}")

    rng_data = torch.Generator(device="cuda").manual_seed(cfg.rng_data_seed)
    q_f32 = torch.empty(b, h_q, s_qo, d_qk, dtype=torch.float32, device="cuda")
    fill_sparse_small_int(q_f32, rng_data, sparsity=0.8, abs_max=2)
    k_f32 = torch.empty(b, h_k, s_kv, d_qk, dtype=torch.float32, device="cuda")
    fill_sparse_small_int(k_f32, rng_data, sparsity=0.8, abs_max=2)
    if not perf:
        # keep at least a few q rows in the deeply-negative-score regime (see
        # inject_negative_score_rows); must run before mxfp8 quantization
        inject_negative_score_rows(q_f32, k_f32, rng_data, attn_scale=attn_scale)
    v_f32 = torch.empty(b, h_v, s_kv, d_vo, dtype=torch.float32, device="cuda")
    fill_sparse_small_int(v_f32, rng_data, sparsity=0.8, abs_max=2)

    q_fp8_d, sf_q_d_ref, sf_q_d_swizzle, q_fp8_s, sf_q_s_ref, sf_q_s_swizzle = quantize_to_mxfp8(q_f32, b, h_q, s_qo, d_qk, block_size, torch_itype, with_ref=not perf)
    k_fp8_d, sf_k_d_ref, sf_k_d_swizzle, k_fp8_s, sf_k_s_ref, sf_k_s_swizzle = quantize_to_mxfp8(k_f32, b, h_k, s_kv, d_qk, block_size, torch_itype, with_ref=not perf)
    v_fp8_d, sf_v_d_ref, sf_v_d_swizzle, v_fp8_s, sf_v_s_ref, sf_v_s_swizzle = quantize_to_mxfp8(v_f32, b, h_v, s_kv, d_vo, block_size, torch_itype, with_ref=not perf)

    # Generate sink_token if needed
    sink_token_gpu = None
    if with_sink_token:
        rng_sink = torch.Generator(device="cuda").manual_seed(cfg.rng_data_seed + 1000)
        sink_token_gpu = torch.randn((1, h_q, 1, 1), dtype=torch.float32, device="cuda", generator=rng_sink) * 0.5

    # Allocate output tensors
    o_gpu = torch.empty(b, h_q, s_qo, d_vo, dtype=torch_otype, device="cuda")
    stats_gpu = torch.empty(b, h_q, s_qo, 1, dtype=torch.float32, device="cuda")
    amax_o_gpu = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device="cuda")

    # Build variant pack
    variant_pack = {
        int(GraphFwdUid.q): q_fp8_d,
        int(GraphFwdUid.k): k_fp8_d,
        int(GraphFwdUid.v): v_fp8_s,
        int(GraphFwdUid.sf_q): sf_q_d_swizzle,
        int(GraphFwdUid.sf_k): sf_k_d_swizzle,
        int(GraphFwdUid.sf_v): sf_v_s_swizzle,
        int(GraphFwdUid.o): o_gpu,
        int(GraphFwdUid.stats): stats_gpu,
        int(GraphFwdUid.o_amax): amax_o_gpu,
    }
    if with_sink_token:
        variant_pack[int(GraphFwdUid.sink_token)] = sink_token_gpu

    # Execute
    workspace = torch.empty(graph_fwd.get_workspace_size(), dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    if perf:
        times_ms = time_execution(graph_fwd.execute, variant_pack, workspace, cudnn_handle)
        print(f"@@@@ MXFP8 Fwd graph_fwd.execute avg_time_ms={times_ms.mean().item():.3f}")
        profile_execution(graph_fwd.execute, variant_pack, workspace, cudnn_handle)
    graph_fwd.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    o_f16 = o_gpu
    stats_bwd = stats_gpu
    if not perf:
        o_ref, stats_ref = compute_ref(q_fp8_d, k_fp8_d, v_fp8_s, sf_q_d_ref, sf_k_d_ref, sf_v_s_ref, attn_scale,
                                       torch_itype=torch_itype, output_type=torch_otype,
                                       left_bound=left_bound, right_bound=right_bound, diag_align=diag_align,
                                       sink_token=sink_token_gpu, rescale_threshold=rescale_threshold)
        o_f16 = o_ref.to(torch.bfloat16)
        stats_bwd = stats_ref
        for actual, expected, atol, rtol, name in (
            (o_gpu, o_ref, 0.12, 0.20, "output"),
            (stats_gpu, stats_ref, 0.05, 0.05, "stats"),
        ):
            error = compare_tensors(actual, expected, atol, rtol, name)
            assert error == 0, f"{name} mismatch: {error} elements differ"
        assert compare_amax(o_gpu, o_ref, rtol=0.05, tag="amax"), "Amax mismatch: 1 element differs"

    if not cfg.is_infer:
        dO_f32 = torch.empty(b, h_q, s_qo, d_vo, dtype=torch.float32, device="cuda")
        fill_sparse_small_int(dO_f32, rng_data, sparsity=0.8, abs_max=2)
        dO_fp8_d, sf_dO_d_ref, sf_dO_d_swizzle, dO_fp8_s, sf_dO_s_ref, sf_dO_s_swizzle = quantize_to_mxfp8(dO_f32, b, h_q, s_qo, d_vo, block_size, torch_itype, with_ref=not perf)

        dO_f16 = dO_f32.to(torch.bfloat16)

        # Build backward graph
        try:
            graph_bwd = generate_graph_bwd(
                b, h_q, h_k, h_v,
                s_qo, s_kv, d_qk, d_vo, attn_scale,
                deterministic, block_size,
                cudnn_itype, cudnn_otype,
                left_bound=left_bound, right_bound=right_bound, diag_align=diag_align,
                with_sink_token=with_sink_token,
            )
            graph_bwd.validate()
            graph_bwd.build_operation_graph()
            graph_bwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            graph_bwd.check_support()
            graph_bwd.build_plans()
            note_frost_routing(graph_bwd, label="mxfp8-bwd")
        except cudnn.cudnnGraphNotSupportedError as e:
            pytest.skip(f"MXFP8 SDPA not supported: {e}")
        except Exception as e:
            # NOT_SUPPORTED can also surface at build_plans/finalize AFTER
            # check_support accepted the graph (backend support-check gap);
            # that is a waive, not a harness error.
            if "CUDNN_STATUS_NOT_SUPPORTED" in str(e):
                pytest.skip(f"MXFP8 SDPA not supported (at finalize): {e}")
            pytest.fail(f"Error building MXFP8 SDPA graph: {e}")

        # Allocate backward output tensors
        dQ_gpu = torch.empty(b, h_q, s_qo, d_qk, dtype=torch_otype, device="cuda")
        dK_gpu = torch.empty(b, h_k, s_kv, d_qk, dtype=torch_otype, device="cuda")
        dV_gpu = torch.empty(b, h_v, s_kv, d_vo, dtype=torch_otype, device="cuda")
        dQ_amax_gpu = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device="cuda")
        dK_amax_gpu = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device="cuda")
        dV_amax_gpu = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device="cuda")
        dSink_token_gpu = None
        if with_sink_token:
            dSink_token_gpu = torch.zeros(1, h_q, 1, 1, dtype=torch.float32, device="cuda")

        # Build backward variant pack
        variant_pack_bwd = {
            int(GraphBwdUid.q): q_fp8_d,
            int(GraphBwdUid.q_t): q_fp8_s,
            int(GraphBwdUid.k): k_fp8_d,
            int(GraphBwdUid.k_t): k_fp8_s,
            int(GraphBwdUid.v): v_fp8_d,
            int(GraphBwdUid.o): o_f16,
            int(GraphBwdUid.dO): dO_fp8_d,
            int(GraphBwdUid.dO_t): dO_fp8_s,
            int(GraphBwdUid.dO_f16): dO_f16,
            int(GraphBwdUid.stats): stats_bwd,
            int(GraphBwdUid.sf_q): sf_q_d_swizzle,
            int(GraphBwdUid.sf_q_t): sf_q_s_swizzle,
            int(GraphBwdUid.sf_k): sf_k_d_swizzle,
            int(GraphBwdUid.sf_k_t): sf_k_s_swizzle,
            int(GraphBwdUid.sf_v): sf_v_d_swizzle,
            int(GraphBwdUid.sf_dO): sf_dO_d_swizzle,
            int(GraphBwdUid.sf_dO_t): sf_dO_s_swizzle,
            int(GraphBwdUid.dQ): dQ_gpu,
            int(GraphBwdUid.dK): dK_gpu,
            int(GraphBwdUid.dV): dV_gpu,
            int(GraphBwdUid.dQ_amax): dQ_amax_gpu,
            int(GraphBwdUid.dK_amax): dK_amax_gpu,
            int(GraphBwdUid.dV_amax): dV_amax_gpu,
        }
        if with_sink_token:
            variant_pack_bwd[int(GraphBwdUid.sink_token)] = sink_token_gpu
            variant_pack_bwd[int(GraphBwdUid.dSink_token)] = dSink_token_gpu

        # Execute backward graph
        workspace_bwd = torch.empty(graph_bwd.get_workspace_size(), dtype=torch.uint8, device="cuda")
        if perf:
            times_ms = time_execution(graph_bwd.execute, variant_pack_bwd, workspace_bwd, cudnn_handle)
            print(f"@@@@ MXFP8 Bwd graph_bwd.execute avg_time_ms={times_ms.mean().item():.3f}")
            profile_execution(graph_bwd.execute, variant_pack_bwd, workspace_bwd, cudnn_handle)
        torch.cuda.synchronize()
        graph_bwd.execute(variant_pack_bwd, workspace_bwd, handle=cudnn_handle)
        torch.cuda.synchronize()

        # Determinism check
        dQ_gpu_rerun = dQ_gpu.clone().detach()
        dK_gpu_rerun = dK_gpu.clone().detach()
        dV_gpu_rerun = dV_gpu.clone().detach()
        dQ_amax_gpu_rerun = dQ_amax_gpu.clone().detach()
        dK_amax_gpu_rerun = dK_amax_gpu.clone().detach()
        dV_amax_gpu_rerun = dV_amax_gpu.clone().detach()

        torch.fill_(dQ_gpu, float("nan"))
        torch.fill_(dK_gpu, float("nan"))
        torch.fill_(dV_gpu, float("nan"))
        torch.fill_(dQ_amax_gpu, float("nan"))
        torch.fill_(dK_amax_gpu, float("nan"))
        torch.fill_(dV_amax_gpu, float("nan"))
        torch.cuda.synchronize()
        graph_bwd.execute(variant_pack_bwd, workspace_bwd, handle=cudnn_handle)
        torch.cuda.synchronize()

        determin_err_count = 0
        determin_err_count += exact_equal(dQ_gpu, dQ_gpu_rerun, tag="dQ_determin", disp_elems=request.config.getoption("--diffs"))
        determin_err_count += exact_equal(dK_gpu, dK_gpu_rerun, tag="dK_determin", disp_elems=request.config.getoption("--diffs"))
        determin_err_count += exact_equal(dV_gpu, dV_gpu_rerun, tag="dV_determin", disp_elems=request.config.getoption("--diffs"))
        determin_err_count += exact_equal(dQ_amax_gpu, dQ_amax_gpu_rerun, tag="dQ_amax_determin", disp_elems=request.config.getoption("--diffs"))
        determin_err_count += exact_equal(dK_amax_gpu, dK_amax_gpu_rerun, tag="dK_amax_determin", disp_elems=request.config.getoption("--diffs"))
        determin_err_count += exact_equal(dV_amax_gpu, dV_amax_gpu_rerun, tag="dV_amax_determin", disp_elems=request.config.getoption("--diffs"))

        if determin_err_count != 0:
            print("@@@@ Overall result: FAILED, determinism check failed - outputs differ between runs.")
            pytest.fail("determinism check failed", pytrace=False)


        if not perf:
            dQ_ref, dK_ref, dV_ref, dSink_token_ref = compute_ref_backward(
                q_fp8_d, q_fp8_s, k_fp8_d, k_fp8_s, v_fp8_d,
                o_f16, dO_f16, dO_fp8_d, dO_fp8_s,
                attn_scale,
                sf_q_d_ref, sf_q_s_ref, sf_k_d_ref, sf_k_s_ref, sf_v_d_ref,
                sf_dO_d_ref, sf_dO_s_ref,
                torch_itype=torch_itype, torch_otype=torch_otype,
                left_bound=left_bound, right_bound=right_bound, diag_align=diag_align,
                sink_token=sink_token_gpu,
                # stats_bwd is what the backward graph is fed, so use it here too.
                stats=stats_bwd,
            )

            for actual, expected, name in (
                (dQ_gpu, dQ_ref, "dQ"),
                (dK_gpu, dK_ref, "dK"),
                (dV_gpu, dV_ref, "dV"),
            ):
                # dV carries the largest magnitudes, so it needs one wider step than fp8.
                error = compare_tensors(actual, expected, 0.125, 0.20, name)
                assert error == 0, f"{name} mismatch: {error} elements differ"

            if with_sink_token and dSink_token_ref is not None:
                dSink_err = compare_tensors(dSink_token_gpu, dSink_token_ref, 0.08, 0.20, "dSink_token")
                assert dSink_err == 0, f"dSink_token mismatch: {dSink_err} elements differ"

            for actual, expected, name in (
                (dQ_gpu, dQ_ref, "dQ"),
                (dK_gpu, dK_ref, "dK"),
                (dV_gpu, dV_ref, "dV"),
            ):
                assert compare_amax(actual, expected, rtol=0.04, tag=name), f"{name} amax mismatch: 1 element differs"
