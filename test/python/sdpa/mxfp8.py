import cudnn
import pytest
import torch
import math
from enum import IntEnum
from looseversion import LooseVersion

from .helpers import fill_sparse_small_int, exact_equal, time_execution, profile_execution
from .mxfp8_ref import compute_ref, compute_ref_backward

# Try to import CUTLASS for scale factor conversion
try:
    import cutlass.cute as cute
    import cutlass
    from cutlass.cute.runtime import from_dlpack

    @cute.jit
    def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        sf_ref_tensor: cute.Tensor,
        sf_mma_tensor: cute.Tensor,
    ):
        """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
        sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
        sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
        for i in cutlass.range(cute.size(sf_ref_tensor)):
            mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
            sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]

    HAS_CUTLASS = True
except Exception:
    HAS_CUTLASS = False
    cute = None
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L = None

# Try to import TransformerEngine for MXFP8 quantization
try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    import transformer_engine_torch as tex

    HAS_TE = True
except Exception:
    HAS_TE = False
    tex = None
    MXFP8Quantizer = None

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


def create_sf_layout_tensor(l, mn, nk, sf_vec_size):
    """Create scale factor tensor with F8_128x4 layout."""
    sf_k = ceil_div(nk, sf_vec_size)

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )

    mma_permute_order = (3, 4, 1, 5, 2, 0)
    cute_f32_torch_tensor_cpu = torch.zeros(mma_shape, dtype=torch.float32).permute(mma_permute_order)

    return cute_f32_torch_tensor_cpu, sf_k


def create_scale_factor_tensor(l, mn, k, sf_vec_size, o_dtype=torch.float8_e8m0fnu, ref_values=None):
    """
    Create scale factor tensor for SDPA with F8_128x4 reordering.

    Args:
        l: batch dimension (b * h for SDPA)
        mn: non-contracting dimension (s for Q/K, d for V)
        k: contracting dimension to be scaled (d for Q/K, s for V)
        sf_vec_size: block size (32 for MXFP8)
        o_dtype: output dtype (torch.float8_e8m0fnu)
        ref_values: Optional float32 tensor of shape [mn, sf_k, l] containing per-block
                    scale factor values (e.g. derived from actual data). If None, random
                    values are generated.

    Returns:
        ref_tensor: reference tensor for computation [mn, sf_k, l] -> broadcast to [mn, k, l]
        cute_tensor: F8_128x4 reordered tensor for cuDNN
    """
    if not HAS_CUTLASS:
        pytest.skip("CUTLASS is not installed; skipping MXFP8 tests.")

    cute_f32_torch_tensor_cpu, sf_k = create_sf_layout_tensor(l, mn, k, sf_vec_size)
    ref_shape = (l, mn, sf_k)
    ref_permute_order = (1, 2, 0)

    if ref_values is not None:
        assert ref_values.shape == (mn, sf_k, l), \
            f"Expected ref_values shape {(mn, sf_k, l)}, got {ref_values.shape}"
        ref_f32_torch_tensor_cpu = ref_values.cpu().float().contiguous()
    else:
        # Create reference scale factors (small positive values for stability)
        ref_f32_torch_tensor_cpu = (
            torch.empty(ref_shape, dtype=torch.float32).uniform_(0.5, 2.0).permute(ref_permute_order).to(torch.int8).to(torch.float32)
        )

    # Convert ref f32 tensor to cute f32 tensor with F8_128x4 layout
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(ref_f32_torch_tensor_cpu),
        from_dlpack(cute_f32_torch_tensor_cpu),
    )

    # Expand scale factors to match the original k dimension
    ref_expanded = ref_f32_torch_tensor_cpu.permute(2, 0, 1).unsqueeze(-1).expand(l, mn, sf_k, sf_vec_size).reshape(l, mn, sf_k * sf_vec_size).permute(*ref_permute_order)
    ref_expanded = ref_expanded[:, :k, :]

    # Convert to E8M0 dtype
    cute_torch_tensor = cute_f32_torch_tensor_cpu.to(o_dtype).cuda()

    return ref_expanded.cuda(), cute_torch_tensor


def quantize_to_mxfp8(tensor, b, h, s, d, s_padded, block_size=32, fp8_dtype=torch.float8_e4m3fn):
    l = b * h
    te_dtype = tex.DType.kFloat8E4M3 if fp8_dtype == torch.float8_e4m3fn else tex.DType.kFloat8E5M2
    quantizer = MXFP8Quantizer(fp8_dtype=te_dtype, rowwise=True, columnwise=False)

    # Quantize along D dimension
    d_scale = ceil_div(d, block_size)
    d_scale_padded = ceil_div(d_scale, 4) * 4
    d_padded = d_scale_padded * block_size

    tensor_3d_d = tensor.float().reshape(l, s, d)
    pad_d = d_padded - d
    pad_s = s_padded - s
    if pad_s > 0 or pad_d > 0:
        tensor_3d_d = torch.nn.functional.pad(tensor_3d_d, (0, pad_d, 0, pad_s))
    tensor_2d_d = tensor_3d_d.reshape(l * s_padded, d_padded).contiguous()

    mxfp8_result_d = quantizer.quantize_impl(tensor_2d_d)

    # [l*s_padded, d_padded] -> [b, h, s, d]
    fp8_data_flat = mxfp8_result_d._rowwise_data
    fp8_data = fp8_data_flat.reshape(l, s_padded, d_padded)[:, :s, :d].contiguous()
    fp8_data = fp8_data.view(fp8_dtype).reshape(b, h, s, d)

    # [l*s_padded, d_scale_padded] -> [s_padded, d_scale, l]
    scale_inv_d = mxfp8_result_d._rowwise_scale_inv
    scale_inv_d_f32 = scale_inv_d.view(torch.float8_e8m0fnu).float()
    scale_3d_d = scale_inv_d_f32.reshape(l, s_padded, d_scale_padded)
    ref_values_d = scale_3d_d.permute(1, 2, 0)

    sf_d_ref, sf_d_cute = create_scale_factor_tensor(
        l=l, mn=s_padded, k=d_padded, sf_vec_size=block_size,
        ref_values=ref_values_d,
    )

    # Quantize along S dimension
    s_scale_padded = s_padded // block_size

    # Transpose: [l, s, d] -> [l, d, s]
    tensor_3d_s = tensor.float().reshape(l, s, d).permute(0, 2, 1).contiguous()
    pad_s = s_padded - s
    pad_d = d_padded - d
    if pad_d > 0 or pad_s > 0:
        tensor_3d_s = torch.nn.functional.pad(tensor_3d_s, (0, pad_s, 0, pad_d))
    tensor_2d_s = tensor_3d_s.reshape(l * d_padded, s_padded).contiguous()

    mxfp8_result_s = quantizer.quantize_impl(tensor_2d_s)

    # [l*d_padded, s_padded] -> [b, h, d, s]
    fp8_data_s_flat = mxfp8_result_s._rowwise_data
    fp8_data_s = fp8_data_s_flat.reshape(l, d_padded, s_padded)[:, :d, :s].contiguous()
    fp8_data_s = fp8_data_s.view(fp8_dtype).reshape(b, h, d, s)

    # [l*d_padded, s_scale_te] -> [d_padded, s_scale, l]
    scale_inv_s = mxfp8_result_s._rowwise_scale_inv
    scale_inv_s_f32 = scale_inv_s.view(torch.float8_e8m0fnu).float()
    scale_3d_s = scale_inv_s_f32.reshape(l, d_padded, s_scale_padded)
    ref_values_s = scale_3d_s.permute(1, 2, 0)

    sf_s_ref, sf_s_cute = create_scale_factor_tensor(
        l=l, mn=d_padded, k=s_padded, sf_vec_size=block_size,
        ref_values=ref_values_s,
    )

    return fp8_data, sf_d_ref, sf_d_cute, fp8_data_s, sf_s_ref, sf_s_cute


def generate_graph_fwd(b, h_q, h_k, h_v,
                       s_qo, s_kv, d_qk, d_vo, attn_scale,
                       block_size=32,
                       cudnn_itype=cudnn.data_type.FP8_E4M3,
                       cudnn_otype=cudnn.data_type.HALF,
                       left_bound=None, right_bound=None, diag_align=None):
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
    sf_v_strides = (h_v * s_kv_scale_padded * d_vo_padded, s_kv_scale_padded * d_vo_padded, 1, s_kv_scale_padded)
    sf_v = graph.tensor(
        uid=GraphFwdUid.sf_v,
        dim=sf_v_dims,
        stride=sf_v_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # Call MXFP8 SDPA
    o, stats, amax_o = graph.sdpa_mxfp8(
        q=q, k=k, v=v,
        descale_q=sf_q, descale_k=sf_k, descale_v=sf_v,
        attn_scale=attn_scale,
        generate_stats=True,
        diagonal_alignment=diag_align if diag_align is not None else cudnn.diagonal_alignment.TOP_LEFT,
        diagonal_band_left_bound=left_bound,
        diagonal_band_right_bound=right_bound,
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
                       left_bound=None, right_bound=None, diag_align=None):
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
    sf_q_t_strides = (h_q * s_qo_scale_padded * d_qk_padded, s_qo_scale_padded * d_qk_padded, 1, s_qo_scale_padded)
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
    sf_k_t_strides = (h_k * s_kv_scale_padded * d_qk_padded, s_kv_scale_padded * d_qk_padded, 1, s_kv_scale_padded)
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
    sf_dO_t_strides = (h_q * s_qo_scale_padded * d_vo_padded, s_qo_scale_padded * d_vo_padded, 1, s_qo_scale_padded)
    sf_dO_t = graph_bwd.tensor(
        uid=GraphBwdUid.sf_dO_t,
        dim=sf_dO_t_dims,
        stride=sf_dO_t_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

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
    )

    # Set output tensor properties
    dQ.set_uid(GraphBwdUid.dQ).set_output(True).set_dim((b, h_q, s_qo, d_qk)).set_stride((h_q * s_qo * d_qk, s_qo * d_qk, d_qk, 1)).set_data_type(cudnn_otype)
    dK.set_uid(GraphBwdUid.dK).set_output(True).set_dim((b, h_k, s_kv, d_qk)).set_stride((h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1)).set_data_type(cudnn_otype)
    dV.set_uid(GraphBwdUid.dV).set_output(True).set_dim((b, h_v, s_kv, d_vo)).set_stride((h_v * s_kv * d_vo, s_kv * d_vo, d_vo, 1)).set_data_type(cudnn_otype)

    amax_dQ.set_uid(GraphBwdUid.dQ_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dK.set_uid(GraphBwdUid.dK_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dV.set_uid(GraphBwdUid.dV_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

    return graph_bwd

def exec_sdpa_mxfp8(cfg, request, cudnn_handle):
    """Execute MXFP8 SDPA test."""
    if request.config.option.dryrun:
        pytest.skip("dry run mode")

    cudnn_version = LooseVersion(cudnn.backend_version_string())

    if not HAS_CUTLASS:
        pytest.skip("CUTLASS is not installed; skipping MXFP8 tests.")

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

    # Compute padded dimensions for F8_128x4 scale factors
    s_q_padded = ceil_div(s_qo, 128) * 128
    s_kv_padded = ceil_div(s_kv, 128) * 128
    d_qk_scale_padded = ceil_div(ceil_div(d_qk, block_size), 4) * 4
    d_vo_padded = ceil_div(d_vo, 128) * 128
    s_kv_scale_padded = ceil_div(ceil_div(s_kv, block_size), 4) * 4

    # Build forward graph
    try:
        graph_fwd = generate_graph_fwd(
            b, h_q, h_k, h_v,
            s_qo, s_kv, d_qk, d_vo, attn_scale,
            block_size,
            cudnn_itype, cudnn_otype,
            left_bound=left_bound, right_bound=right_bound, diag_align=diag_align,
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

    # Create FP8 input tensors using sparse small integers for better low-precision testing
    rng_data = torch.Generator(device="cuda").manual_seed(cfg.rng_data_seed)
    q_f32 = torch.empty(b, h_q, s_qo, d_qk, dtype=torch.float32, device="cuda")
    fill_sparse_small_int(q_f32, rng_data, sparsity=0.8, abs_max=2)
    k_f32 = torch.empty(b, h_k, s_kv, d_qk, dtype=torch.float32, device="cuda")
    fill_sparse_small_int(k_f32, rng_data, sparsity=0.8, abs_max=2)
    v_f32 = torch.empty(b, h_v, s_kv, d_vo, dtype=torch.float32, device="cuda")
    fill_sparse_small_int(v_f32, rng_data, sparsity=0.8, abs_max=2)

    q_fp8 = q_f32.to(torch_itype)
    k_fp8 = k_f32.to(torch_itype)
    v_fp8 = v_f32.to(torch_itype)

    # Create scale factor tensors using CUTLASS cute DSL for proper F8_128x4 layout
    # create_scale_factor_tensor returns: (ref_expanded [mn, k, l], cute_tensor with F8_128x4 layout)

    # SF_Q: [s_q_padded, d_qk, b*h_q]
    # sf_q_ref_raw: [s_q_padded, d_qk, b*h_q]
    # sf_q_cute: [32, 4, s_q_padded / 128, 4, d_qk_scale_padded / 4, b*h_q]
    sf_q_ref_raw, sf_q_cute = create_scale_factor_tensor(
        l=b * h_q, mn=s_q_padded, k=d_qk, sf_vec_size=block_size
    )

    # SF_K: [s_kv_padded, d_qk, b*h_k]
    # sf_k_ref_raw: [s_kv_padded, d_qk, b*h_k]
    # sf_k_cute: [32, 4, s_kv_padded / 128, 4, d_qk_scale_padded / 4, b*h_k]
    sf_k_ref_raw, sf_k_cute = create_scale_factor_tensor(
        l=b * h_k, mn=s_kv_padded, k=d_qk, sf_vec_size=block_size
    )

    # SF_V: [d_vo_padded, s_kv, b*h_v] - s is scaled (contiguous in BMM2 contraction)
    # sf_v_ref_raw: [d_vo_padded, s_kv, b*h_v]
    # sf_v_cute: [32, 4, d_vo_padded / 128, 4, s_kv_scale_padded / 4, b*h_v]
    sf_v_ref_raw, sf_v_cute = create_scale_factor_tensor(
        l=b * h_v, mn=d_vo_padded, k=s_kv, sf_vec_size=block_size
    )

    # Cute tensors are already in F8_128x4 layout, use directly
    sf_q_cudnn = sf_q_cute
    sf_k_cudnn = sf_k_cute
    sf_v_cudnn = sf_v_cute

    # sf_q_ref: [S_q, D, B*H_q] -> [B*H_q, S_q, D]
    sf_q_ref_raw = sf_q_ref_raw.permute(2, 0, 1)
    # sf_k_ref: [S_kv, D, B*H_k] -> [B*H_k, S_kv, D]
    sf_k_ref_raw = sf_k_ref_raw.permute(2, 0, 1)
    # sf_v_ref: [D, S_kv, B*H_v] -> [B*H_v, S_kv, D]
    sf_v_ref_raw = sf_v_ref_raw.permute(2, 1, 0)

    # Trim reference scale factors to actual dimensions for compute_ref
    sf_q_ref = sf_q_ref_raw[:, :s_qo, :d_qk]
    sf_k_ref = sf_k_ref_raw[:, :s_kv, :d_qk]
    sf_v_ref = sf_v_ref_raw[:, :s_kv, :d_vo]

    # Allocate output tensors
    o_gpu = torch.empty(b, h_q, s_qo, d_vo, dtype=torch_otype, device="cuda")
    stats_gpu = torch.empty(b, h_q, s_qo, 1, dtype=torch.float32, device="cuda")
    amax_o_gpu = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device="cuda")

    # Build variant pack
    variant_pack = {
        int(GraphFwdUid.q): q_fp8,
        int(GraphFwdUid.k): k_fp8,
        int(GraphFwdUid.v): v_fp8,
        int(GraphFwdUid.sf_q): sf_q_cudnn,
        int(GraphFwdUid.sf_k): sf_k_cudnn,
        int(GraphFwdUid.sf_v): sf_v_cudnn,
        int(GraphFwdUid.o): o_gpu,
        int(GraphFwdUid.stats): stats_gpu,
        int(GraphFwdUid.o_amax): amax_o_gpu,
    }

    # Execute
    workspace = torch.empty(graph_fwd.get_workspace_size(), dtype=torch.uint8, device="cuda")
    if request.config.getoption("--perf"):
        times_ms = time_execution(graph_fwd.execute, variant_pack, workspace, cudnn_handle)
        print(f"@@@@ MXFP8 Fwd graph_fwd.execute avg_time_ms={times_ms.mean().item():.3f}")
        profile_execution(graph_fwd.execute, variant_pack, workspace, cudnn_handle)
    graph_fwd.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    # Compute reference
    o_ref, stats_ref = compute_ref(q_fp8, k_fp8, v_fp8, sf_q_ref, sf_k_ref, sf_v_ref, attn_scale,
                                   torch_itype=torch_itype, output_type=torch_otype,
                                   left_bound=left_bound, right_bound=right_bound, diag_align=diag_align)

    # Compare output
    o_atol, o_rtol = 0.08, 0.20
    o_err = compare_tensors(o_gpu, o_ref, o_atol, o_rtol, "output")

    # Compare stats (logsumexp) - tight tolerance
    stats_atol, stats_rtol = 0.05, 0.05
    stats_err = compare_tensors(stats_gpu, stats_ref, stats_atol, stats_rtol, "stats")

    # Compare amax
    amax_err = compare_amax(o_gpu, o_ref, rtol=0.02, tag="amax")

    # Assert all checks pass
    assert o_err == 0, f"Output mismatch: {o_err} elements differ"
    assert stats_err == 0, f"Stats mismatch: {stats_err} elements differ"
    assert amax_err, "Amax mismatch: 1 element differs"

    if not cfg.is_infer:
        # Generate dO input
        dO_f32 = torch.empty(b, h_q, s_qo, d_vo, dtype=torch.float32, device="cuda")
        fill_sparse_small_int(dO_f32, rng_data, sparsity=0.8, abs_max=2)
        dO_fp8 = dO_f32.to(torch_itype)

        # Create scale factors for dO
        sf_dO_ref_raw, sf_dO_cute = create_scale_factor_tensor(
            l=b * h_q, mn=s_q_padded, k=d_vo, sf_vec_size=block_size
        )
        sf_dO_ref_raw = sf_dO_ref_raw.permute(2, 0, 1)
        sf_dO_ref = sf_dO_ref_raw[:, :s_qo, :d_vo]

        o_f16 = o_ref.to(torch.bfloat16)
        q_f32 = q_fp8.float()
        k_f32 = k_fp8.float()
        dO_f32 = dO_fp8.float()

        # Dequantize inputs to FP32
        q = q_f32.reshape(b * h_q, s_qo, d_qk)
        k = k_f32.reshape(b * h_k, s_kv, d_qk)
        dO = dO_f32.reshape(b * h_q, s_qo, d_vo)
        v = v_f32.reshape(b * h_v, s_kv, d_vo)

        q_dq = q * sf_q_ref
        k_dq = k * sf_k_ref
        dO_dq = dO * sf_dO_ref
        v_dq = v * sf_v_ref

        q_dq = q_dq.reshape(b, h_q, s_qo, d_qk)
        k_dq = k_dq.reshape(b, h_k, s_kv, d_qk)
        dO_dq = dO_dq.reshape(b, h_q, s_qo, d_vo)
        v_dq = v_dq.reshape(b, h_v, s_kv, d_vo)
        dO_f16 = dO_dq.to(torch.bfloat16)

        # Re-quantize using row-wise and column-wise quantization
        q_fp8_scaled, sf_q_ref_scaled, sf_q_cute_scaled, q_fp8_t_scaled, sf_q_t_ref_scaled, sf_q_t_cute_scaled = quantize_to_mxfp8(q_dq, b, h_q, s_qo, d_qk, s_q_padded, block_size, torch_itype)
        k_fp8_scaled, sf_k_ref_scaled, sf_k_cute_scaled, k_fp8_t_scaled, sf_k_t_ref_scaled, sf_k_t_cute_scaled = quantize_to_mxfp8(k_dq, b, h_k, s_kv, d_qk, s_kv_padded, block_size, torch_itype)
        dO_fp8_scaled, sf_dO_ref_scaled, sf_dO_cute_scaled, dO_fp8_t_scaled, sf_dO_t_ref_scaled, sf_dO_t_cute_scaled = quantize_to_mxfp8(dO_dq, b, h_q, s_qo, d_vo, s_q_padded, block_size, torch_itype)
        v_fp8_scaled, sf_v_ref_scaled, sf_v_cute_scaled, v_fp8_t_scaled, sf_v_t_ref_scaled, sf_v_t_cute_scaled = quantize_to_mxfp8(v_dq, b, h_v, s_kv, d_vo, s_kv_padded, block_size, torch_itype)

        q_fp8_t_scaled = q_fp8_t_scaled.transpose(-2, -1).contiguous()
        k_fp8_t_scaled = k_fp8_t_scaled.transpose(-2, -1).contiguous()
        dO_fp8_t_scaled = dO_fp8_t_scaled.transpose(-2, -1).contiguous()
        v_fp8_t_scaled = v_fp8_t_scaled.transpose(-2, -1).contiguous()

        sf_q_ref_scaled = sf_q_ref_scaled.permute(2, 0, 1)
        sf_k_ref_scaled = sf_k_ref_scaled.permute(2, 0, 1)
        sf_dO_ref_scaled = sf_dO_ref_scaled.permute(2, 0, 1)
        sf_v_ref_scaled = sf_v_ref_scaled.permute(2, 0, 1)
        sf_v_t_ref_scaled = sf_v_t_ref_scaled.permute(2, 1, 0)
        sf_q_t_ref_scaled = sf_q_t_ref_scaled.permute(2, 1, 0)
        sf_k_t_ref_scaled = sf_k_t_ref_scaled.permute(2, 1, 0)
        sf_dO_t_ref_scaled = sf_dO_t_ref_scaled.permute(2, 1, 0)

        sf_q_ref_scaled = sf_q_ref_scaled[:, :s_qo, :d_qk]
        sf_k_ref_scaled = sf_k_ref_scaled[:, :s_kv, :d_qk]
        sf_dO_ref_scaled = sf_dO_ref_scaled[:, :s_qo, :d_vo]
        sf_q_t_ref_scaled = sf_q_t_ref_scaled[:, :s_qo, :d_qk]
        sf_k_t_ref_scaled = sf_k_t_ref_scaled[:, :s_kv, :d_qk]
        sf_dO_t_ref_scaled = sf_dO_t_ref_scaled[:, :s_qo, :d_vo]
        sf_v_ref_scaled = sf_v_ref_scaled[:, :s_kv, :d_vo]
        sf_v_t_ref_scaled = sf_v_t_ref_scaled[:, :s_kv, :d_vo]

        # Build backward graph
        try:
            graph_bwd = generate_graph_bwd(
                b, h_q, h_k, h_v,
                s_qo, s_kv, d_qk, d_vo, attn_scale,
                deterministic, block_size,
                cudnn_itype, cudnn_otype,
                left_bound=left_bound, right_bound=right_bound, diag_align=diag_align,
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

        # Build backward variant pack
        variant_pack_bwd = {
            int(GraphBwdUid.q): q_fp8_scaled,
            int(GraphBwdUid.q_t): q_fp8_t_scaled,
            int(GraphBwdUid.k): k_fp8_scaled,
            int(GraphBwdUid.k_t): k_fp8_t_scaled,
            int(GraphBwdUid.v): v_fp8_scaled,
            int(GraphBwdUid.o): o_f16,
            int(GraphBwdUid.dO): dO_fp8_scaled,
            int(GraphBwdUid.dO_t): dO_fp8_t_scaled,
            int(GraphBwdUid.dO_f16): dO_f16,
            int(GraphBwdUid.stats): stats_ref,
            int(GraphBwdUid.sf_q): sf_q_cute_scaled,
            int(GraphBwdUid.sf_q_t): sf_q_t_cute_scaled,
            int(GraphBwdUid.sf_k): sf_k_cute_scaled,
            int(GraphBwdUid.sf_k_t): sf_k_t_cute_scaled,
            int(GraphBwdUid.sf_v): sf_v_cute_scaled,
            int(GraphBwdUid.sf_dO): sf_dO_cute_scaled,
            int(GraphBwdUid.sf_dO_t): sf_dO_t_cute_scaled,
            int(GraphBwdUid.dQ): dQ_gpu,
            int(GraphBwdUid.dK): dK_gpu,
            int(GraphBwdUid.dV): dV_gpu,
            int(GraphBwdUid.dQ_amax): dQ_amax_gpu,
            int(GraphBwdUid.dK_amax): dK_amax_gpu,
            int(GraphBwdUid.dV_amax): dV_amax_gpu,
        }

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
        dQ_ref, dK_ref, dV_ref = compute_ref_backward(
            q_fp8_scaled, q_fp8_t_scaled, k_fp8_scaled, k_fp8_t_scaled, v_fp8_scaled,
            o_f16, dO_f16, dO_fp8_scaled, dO_fp8_t_scaled,
            attn_scale,
            sf_q_ref_scaled, sf_q_t_ref_scaled, sf_k_ref_scaled, sf_k_t_ref_scaled, sf_v_ref_scaled,
            sf_dO_ref_scaled, sf_dO_t_ref_scaled,
            torch_itype=torch_itype, torch_otype=torch_otype,
            left_bound=left_bound, right_bound=right_bound, diag_align=diag_align,
        )

        # Compare output
        grad_atol, grad_rtol = 0.08, 0.20

        dQ_err = compare_tensors(dQ_gpu, dQ_ref, grad_atol, grad_rtol, "dQ")
        dK_err = compare_tensors(dK_gpu, dK_ref, grad_atol, grad_rtol, "dK")
        dV_err = compare_tensors(dV_gpu, dV_ref, grad_atol, grad_rtol, "dV")

        assert dQ_err == 0, f"dQ mismatch: {dQ_err} elements differ"
        assert dK_err == 0, f"dK mismatch: {dK_err} elements differ"
        assert dV_err == 0, f"dV mismatch: {dV_err} elements differ"

        # Compare amax
        dQ_amax_err = compare_amax(dQ_gpu, dQ_ref, rtol=0.02, tag="dQ")
        dK_amax_err = compare_amax(dK_gpu, dK_ref, rtol=0.02, tag="dK")
        dV_amax_err = compare_amax(dV_gpu, dV_ref, rtol=0.02, tag="dV")
    
        assert dQ_amax_err, "dQ amax mismatch: 1 element differs"
        assert dK_amax_err, "dK amax mismatch: 1 element differs"
        assert dV_amax_err, "dV amax mismatch: 1 element differs"
