# Try to import TransformerEngine (>= 2.12) for MXFP8 quantization
# NOTE: TE must be imported BEFORE cudnn to avoid library loading conflicts
from looseversion import LooseVersion

try:
    import transformer_engine

    if LooseVersion(transformer_engine.__version__) < LooseVersion("2.12.0"):
        raise ImportError(f"TransformerEngine >= 2.12.0 required, found {transformer_engine.__version__}")
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    import transformer_engine_torch as tex

    HAS_TE = True
except Exception:
    HAS_TE = False
    tex = None
    MXFP8Quantizer = None

import cudnn
import pytest
import torch
import math
from enum import IntEnum

from .helpers import exact_equal, fill_sparse_small_int, time_execution, profile_execution
from .mxfp8_ref import compute_ref, compute_ref_backward

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



def quantize_to_mxfp8(tensor, b, h, s, d, block_size=32, fp8_dtype=torch.float8_e4m3fn):
    l = b * h
    te_dtype = tex.DType.kFloat8E4M3 if fp8_dtype == torch.float8_e4m3fn else tex.DType.kFloat8E5M2

    d_scale = ceil_div(d, block_size)
    d_scale_padded = ceil_div(d_scale, 4) * 4
    d_padded = d_scale_padded * block_size

    s_scale = ceil_div(s, block_size)
    s_scale_padded = ceil_div(s_scale, 4) * 4
    s_padded = s_scale_padded * block_size

    tensor_3d = tensor.float().reshape(l, s, d)
    pad_d = d_padded - d
    pad_s = s_padded - s
    if pad_s > 0 or pad_d > 0:
        tensor_3d = torch.nn.functional.pad(tensor_3d, (0, pad_d, 0, pad_s))
    tensor_2d = tensor_3d.reshape(l * s_padded, d_padded)

    # without swizzle
    quantizer = MXFP8Quantizer(fp8_dtype=te_dtype, rowwise=True, columnwise=True)
    quantizer_swizzle = quantizer.copy()
    mxfp8_result = quantizer(tensor_2d)
    # --- Rowwise results (quantized along D dimension) ---
    fp8_data_d_flat = mxfp8_result._rowwise_data
    fp8_data_d = fp8_data_d_flat.reshape(l, s_padded, d_padded)[:, :s, :d].contiguous()
    fp8_data_d = fp8_data_d.view(fp8_dtype).reshape(b, h, s, d)

    scale_inv_d = mxfp8_result._rowwise_scale_inv
    scale_inv_d_f32 = scale_inv_d.view(torch.float8_e8m0fnu).float()
    sf_d_ref = torch.repeat_interleave(scale_inv_d_f32.reshape(l, s_padded, d_scale_padded), repeats=32, dim=2)[:, :s, :d].contiguous()

    # --- Columnwise results (quantized along S dimension) ---
    fp8_data_s_flat = mxfp8_result._columnwise_data
    fp8_data_s = fp8_data_s_flat.reshape(l, s_padded, d_padded)[:, :s, :d].contiguous()
    fp8_data_s = fp8_data_s.view(fp8_dtype).reshape(b, h, s, d)

    scale_inv_s = mxfp8_result._columnwise_scale_inv
    scale_inv_s_f32 = scale_inv_s.view(torch.float8_e8m0fnu).float()
    sf_s_ref = torch.repeat_interleave(scale_inv_s_f32.reshape(l, s_scale_padded, d_padded), repeats=32, dim=1)[:, :s, :d].contiguous()

    # with swizzle
    quantizer_swizzle.optimize_for_gemm = True
    mxfp8_result_swizzle = quantizer_swizzle(tensor_2d)
    # --- Rowwise results (quantized along D dimension) ---
    sf_d_swizzle = mxfp8_result_swizzle._rowwise_scale_inv

    # --- Columnwise results (quantized along S dimension) ---
    sf_s_swizzle = mxfp8_result_swizzle._columnwise_scale_inv

    return fp8_data_d, sf_d_ref, sf_d_swizzle, fp8_data_s, sf_s_ref, sf_s_swizzle


def generate_graph_fwd(b, h_q, h_k, h_v,
                       s_qo, s_kv, d_qk, d_vo, attn_scale,
                       block_size=32,
                       cudnn_itype=cudnn.data_type.FP8_E4M3,
                       cudnn_otype=cudnn.data_type.HALF,
                       left_bound=None, right_bound=None, diag_align=None,
                       with_sink_token=False):
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

def exec_sdpa_mxfp8(cfg, request, cudnn_handle):
    """Execute MXFP8 SDPA test."""
    if request.config.option.dryrun:
        pytest.skip("dry run mode")

    cudnn_version = LooseVersion(cudnn.backend_version_string())

    if not HAS_TE:
        pytest.skip("TransformerEngine is not installed; skipping MXFP8 tests.")

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
    rescale_threshold = getattr(cfg, 'rescale_threshold', 4.0)

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
        )
        graph_fwd.validate()
        graph_fwd.build_operation_graph()
        graph_fwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph_fwd.check_support()
        graph_fwd.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"MXFP8 SDPA not supported: {e}")
    except Exception as e:
        pytest.fail(f"Error building MXFP8 SDPA graph: {e}")

    rng_data = torch.Generator(device="cuda").manual_seed(cfg.rng_data_seed)
    q_f32 = torch.empty(b, h_q, s_qo, d_qk, dtype=torch.float32, device="cuda")
    fill_sparse_small_int(q_f32, rng_data, sparsity=0.8, abs_max=2)
    k_f32 = torch.empty(b, h_k, s_kv, d_qk, dtype=torch.float32, device="cuda")
    fill_sparse_small_int(k_f32, rng_data, sparsity=0.8, abs_max=2)
    v_f32 = torch.empty(b, h_v, s_kv, d_vo, dtype=torch.float32, device="cuda")
    fill_sparse_small_int(v_f32, rng_data, sparsity=0.8, abs_max=2)

    q_fp8_d, sf_q_d_ref, sf_q_d_swizzle, q_fp8_s, sf_q_s_ref, sf_q_s_swizzle = quantize_to_mxfp8(q_f32, b, h_q, s_qo, d_qk, block_size, torch_itype)
    k_fp8_d, sf_k_d_ref, sf_k_d_swizzle, k_fp8_s, sf_k_s_ref, sf_k_s_swizzle = quantize_to_mxfp8(k_f32, b, h_k, s_kv, d_qk, block_size, torch_itype)
    v_fp8_d, sf_v_d_ref, sf_v_d_swizzle, v_fp8_s, sf_v_s_ref, sf_v_s_swizzle = quantize_to_mxfp8(v_f32, b, h_v, s_kv, d_vo, block_size, torch_itype)

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
    if request.config.getoption("--perf"):
        times_ms = time_execution(graph_fwd.execute, variant_pack, workspace, cudnn_handle)
        print(f"@@@@ MXFP8 Fwd graph_fwd.execute avg_time_ms={times_ms.mean().item():.3f}")
        profile_execution(graph_fwd.execute, variant_pack, workspace, cudnn_handle)
    graph_fwd.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    # Compute reference
    o_ref, stats_ref = compute_ref(q_fp8_d, k_fp8_d, v_fp8_s, sf_q_d_ref, sf_k_d_ref, sf_v_s_ref, attn_scale,
                                   torch_itype=torch_itype, output_type=torch_otype,
                                   left_bound=left_bound, right_bound=right_bound, diag_align=diag_align,
                                   sink_token=sink_token_gpu, rescale_threshold=rescale_threshold)

    # Compare output
    o_atol, o_rtol = 0.12, 0.20
    o_err = compare_tensors(o_gpu, o_ref, o_atol, o_rtol, "output")

    # Compare stats (logsumexp) - tight tolerance
    stats_atol, stats_rtol = 0.05, 0.05
    stats_err = compare_tensors(stats_gpu, stats_ref, stats_atol, stats_rtol, "stats")

    # Compare amax
    amax_err = compare_amax(o_gpu, o_ref, rtol=0.05, tag="amax")

    # Assert all checks pass
    assert o_err == 0, f"Output mismatch: {o_err} elements differ"
    assert stats_err == 0, f"Stats mismatch: {stats_err} elements differ"
    assert amax_err, "Amax mismatch: 1 element differs"

    if not cfg.is_infer:
        dO_f32 = torch.empty(b, h_q, s_qo, d_vo, dtype=torch.float32, device="cuda")
        fill_sparse_small_int(dO_f32, rng_data, sparsity=0.8, abs_max=2)
        dO_fp8_d, sf_dO_d_ref, sf_dO_d_swizzle, dO_fp8_s, sf_dO_s_ref, sf_dO_s_swizzle = quantize_to_mxfp8(dO_f32, b, h_q, s_qo, d_vo, block_size, torch_itype)

        o_f16 = o_ref.to(torch.bfloat16)
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
        except cudnn.cudnnGraphNotSupportedError as e:
            pytest.skip(f"MXFP8 SDPA not supported: {e}")
        except Exception as e:
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
            int(GraphBwdUid.stats): stats_ref,
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
        torch.cuda.synchronize()
        workspace_bwd = torch.empty(graph_bwd.get_workspace_size(), dtype=torch.uint8, device="cuda")
        if request.config.getoption("--perf"):
            times_ms = time_execution(graph_bwd.execute, variant_pack_bwd, workspace_bwd, cudnn_handle)
            print(f"@@@@ MXFP8 Bwd graph_bwd.execute avg_time_ms={times_ms.mean().item():.3f}")
            profile_execution(graph_bwd.execute, variant_pack_bwd, workspace_bwd, cudnn_handle)
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


        # Compute reference backward
        dQ_ref, dK_ref, dV_ref, dSink_token_ref = compute_ref_backward(
            q_fp8_d, q_fp8_s, k_fp8_d, k_fp8_s, v_fp8_d,
            o_f16, dO_f16, dO_fp8_d, dO_fp8_s,
            attn_scale,
            sf_q_d_ref, sf_q_s_ref, sf_k_d_ref, sf_k_s_ref, sf_v_d_ref,
            sf_dO_d_ref, sf_dO_s_ref,
            torch_itype=torch_itype, torch_otype=torch_otype,
            left_bound=left_bound, right_bound=right_bound, diag_align=diag_align,
            sink_token=sink_token_gpu,
        )

        # Compare output
        grad_atol, grad_rtol = 0.08, 0.20

        dQ_err = compare_tensors(dQ_gpu, dQ_ref, grad_atol, grad_rtol, "dQ")
        dK_err = compare_tensors(dK_gpu, dK_ref, grad_atol, grad_rtol, "dK")
        dV_err = compare_tensors(dV_gpu, dV_ref, grad_atol, grad_rtol, "dV")

        assert dQ_err == 0, f"dQ mismatch: {dQ_err} elements differ"
        assert dK_err == 0, f"dK mismatch: {dK_err} elements differ"
        assert dV_err == 0, f"dV mismatch: {dV_err} elements differ"

        # Compare dSink_token if using sink_token
        if with_sink_token and dSink_token_ref is not None:
            dSink_err = compare_tensors(dSink_token_gpu, dSink_token_ref, grad_atol, grad_rtol, "dSink_token")
            assert dSink_err == 0, f"dSink_token mismatch: {dSink_err} elements differ"

        # Compare amax
        dQ_amax_err = compare_amax(dQ_gpu, dQ_ref, rtol=0.04, tag="dQ")
        dK_amax_err = compare_amax(dK_gpu, dK_ref, rtol=0.04, tag="dK")
        dV_amax_err = compare_amax(dV_gpu, dV_ref, rtol=0.04, tag="dV")
    
        assert dQ_amax_err, "dQ amax mismatch: 1 element differs"
        assert dK_amax_err, "dK amax mismatch: 1 element differs"
        assert dV_amax_err, "dV amax mismatch: 1 element differs"
