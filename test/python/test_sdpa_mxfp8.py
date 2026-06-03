"""
Test for MXFP8 (Microscaling FP8) Scaled Dot Product Attention.

MXFP8 uses block-wise scale factors with E8M0 data type and F8_128x4 reordering.
Requirements:
- cuDNN 9.21.0 or later
- Blackwell GPU architecture or newer
"""

import cudnn
import pytest
import torch
import math
from looseversion import LooseVersion

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


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def compute_mxfp8_scale_dims(b, h, s, d, block_size=32):
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
    d_padded = ceil_div(d, 4) * 4

    return {
        "s_padded": s_padded,
        "d_scale": d_scale,
        "d_scale_padded": d_scale_padded,
        "s_scale": s_scale,
        "s_scale_padded": s_scale_padded,
        "d_padded": d_padded,
    }


def create_sf_layout_tensor_for_sdpa(l, mn, nk, sf_vec_size):
    """Create scale factor tensor with F8_128x4 layout for SDPA."""
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


def create_scale_factor_tensor_for_sdpa(l, mn, k, sf_vec_size, dtype):
    """
    Create scale factor tensor for SDPA with F8_128x4 reordering.

    Args:
        l: batch dimension (b * h for SDPA)
        mn: non-contracting dimension (s for Q/K, d for V)
        k: contracting dimension to be scaled (d for Q/K, s for V)
        sf_vec_size: block size (32 for MXFP8)
        dtype: output dtype (torch.float8_e8m0fnu)

    Returns:
        ref_tensor: reference tensor for computation [mn, sf_k, l] -> broadcast to [mn, k, l]
        cute_tensor: F8_128x4 reordered tensor for cuDNN
    """
    if not HAS_CUTLASS:
        pytest.skip("CUTLASS is not installed; skipping MXFP8 tests.")

    cute_f32_torch_tensor_cpu, sf_k = create_sf_layout_tensor_for_sdpa(l, mn, k, sf_vec_size)
    ref_shape = (l, mn, sf_k)
    ref_permute_order = (1, 2, 0)

    # Create reference scale factors (small positive values for stability)
    ref_f32_torch_tensor_cpu = torch.empty(ref_shape, dtype=torch.float32).uniform_(0.5, 2.0).permute(ref_permute_order).to(torch.int8).to(torch.float32)

    # Convert ref f32 tensor to cute f32 tensor with F8_128x4 layout
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(ref_f32_torch_tensor_cpu),
        from_dlpack(cute_f32_torch_tensor_cpu),
    )

    # Expand scale factors to match the original k dimension
    # ref shape: [mn, sf_k, l] -> expand to [mn, sf_k * sf_vec_size, l] -> trim to [mn, k, l]
    ref_expanded = (
        ref_f32_torch_tensor_cpu.permute(2, 0, 1).unsqueeze(-1).expand(l, mn, sf_k, sf_vec_size).reshape(l, mn, sf_k * sf_vec_size).permute(*ref_permute_order)
    )
    ref_expanded = ref_expanded[:, :k, :]

    # Convert to E8M0 dtype
    cute_torch_tensor = cute_f32_torch_tensor_cpu.to(torch.float8_e8m0fnu).cuda()

    return ref_expanded.cuda(), cute_torch_tensor


def create_fp8_tensor(shape, dtype=torch.float8_e4m3fn):
    """Create FP8 tensor with random values."""
    # Generate in float32, clamp to FP8 range, convert
    tensor_f32 = torch.randn(shape, dtype=torch.float32, device="cuda").clamp(-2.0, 2.0)
    tensor_fp8 = tensor_f32.to(dtype)
    return tensor_fp8, tensor_f32


def compute_sdpa_ref(q_f32, k_f32, v_f32, sf_q_ref, sf_k_ref, sf_v_ref, attn_scale, use_causal_mask=False):
    """
    Compute reference SDPA with MXFP8 dequantization.

    Args:
        q_f32: Query tensor [B, H, S_q, D] in float32
        k_f32: Key tensor [B, H, S_kv, D] in float32
        v_f32: Value tensor [B, H, S_kv, D] in float32
        sf_q_ref: Q scale factors [S_q, D, B*H] expanded to [S_q, D, B*H]
        sf_k_ref: K scale factors [S_kv, D, B*H] expanded to [S_kv, D, B*H]
        sf_v_ref: V scale factors [D, S_kv, B*H] expanded to [D, S_kv, B*H]
        attn_scale: attention scale factor
        use_causal_mask: whether to apply causal masking

    Returns:
        o_ref: Output tensor [B, H, S_q, D] in float32
    """
    b, h, s_q, d = q_f32.shape
    _, _, s_kv, _ = k_f32.shape

    # Reshape for batch processing: [B, H, S, D] -> [B*H, S, D]
    q = q_f32.reshape(b * h, s_q, d)
    k = k_f32.reshape(b * h, s_kv, d)
    v = v_f32.reshape(b * h, s_kv, d)

    # sf_q_ref: [S_q, D, B*H] -> [B*H, S_q, D]
    sf_q = sf_q_ref.permute(2, 0, 1)
    # sf_k_ref: [S_kv, D, B*H] -> [B*H, S_kv, D]
    sf_k = sf_k_ref.permute(2, 0, 1)
    # sf_v_ref: [D, S_kv, B*H] -> [B*H, S_kv, D]
    sf_v = sf_v_ref.permute(2, 1, 0)

    # Dequantize Q and K (scale factors apply to D dimension)
    q_dq = q * sf_q[:, :s_q, :d]
    k_dq = k * sf_k[:, :s_kv, :d]

    # BMM1: S = Q @ K^T, shape [B*H, S_q, S_kv]
    s = torch.bmm(q_dq, k_dq.transpose(1, 2)) * attn_scale

    # Apply causal mask if requested
    if use_causal_mask:
        mask = torch.triu(torch.ones(s_q, s_kv, device=s.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask.unsqueeze(0), float("-inf"))

    # Softmax
    p = torch.softmax(s, dim=-1)

    # Dequantize V (scale factors apply to S_kv dimension)
    v_dq = v * sf_v[:, :s_kv, :d]

    # BMM2: O = P @ V, shape [B*H, S_q, D]
    o = torch.bmm(p, v_dq)

    # Reshape back to [B, H, S_q, D]
    o_ref = o.reshape(b, h, s_q, d)

    return o_ref


@pytest.mark.L2
def test_sdpa_mxfp8_with_reference(request, cudnn_handle):
    """Test MXFP8 SDPA with reference computation."""
    if request.config.option.dryrun:
        pytest.skip("dry run mode")

    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.21.0":
        pytest.skip("MXFP8 SDPA requires cuDNN 9.21.0 or higher")

    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 SDPA requires Blackwell or higher")

    if not HAS_CUTLASS:
        pytest.skip("CUTLASS is not installed; skipping MXFP8 tests.")

    # Problem dimensions
    b = 2  # batch size
    h = 2  # number of heads
    s = 512  # sequence length (both s_q and s_kv for simplicity)
    d = 128  # hidden head dim
    block_size = 32

    attn_scale = 1.0 / math.sqrt(d)

    # Compute scale tensor dimensions
    dims = compute_mxfp8_scale_dims(b, h, s, d, block_size)

    # Create FP8 input tensors (BHSD layout)
    q_fp8, q_f32 = create_fp8_tensor((b, h, s, d))
    k_fp8, k_f32 = create_fp8_tensor((b, h, s, d))
    v_fp8, v_f32 = create_fp8_tensor((b, h, s, d))

    # Create scale factor tensors
    # SF_Q and SF_K: scale the D dimension, shape [S_padded, D_scale, B*H] for ref
    # The cute tensor will have F8_128x4 layout
    sf_q_ref, sf_q_cute = create_scale_factor_tensor_for_sdpa(b * h, dims["s_padded"], d, block_size, torch.float8_e8m0fnu)
    sf_k_ref, sf_k_cute = create_scale_factor_tensor_for_sdpa(b * h, dims["s_padded"], d, block_size, torch.float8_e8m0fnu)

    # SF_V: scale the S dimension, shape [D_padded, S_scale, B*H] for ref
    # Note: for V, the contracting dim is S, so we scale S
    sf_v_ref, sf_v_cute = create_scale_factor_tensor_for_sdpa(b * h, dims["d_padded"], s, block_size, torch.float8_e8m0fnu)

    # Compute reference output
    o_ref = compute_sdpa_ref(q_f32, k_f32, v_f32, sf_q_ref, sf_k_ref, sf_v_ref, attn_scale, use_causal_mask=True)

    # Build cuDNN graph
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    # Q, K, V tensors - BHSD layout, contiguous
    qkv_dims = (b, h, s, d)
    qkv_strides = (h * s * d, s * d, d, 1)

    q = graph.tensor(dim=qkv_dims, stride=qkv_strides, data_type=cudnn.data_type.FP8_E4M3)
    k = graph.tensor(dim=qkv_dims, stride=qkv_strides, data_type=cudnn.data_type.FP8_E4M3)
    v = graph.tensor(dim=qkv_dims, stride=qkv_strides, data_type=cudnn.data_type.FP8_E4M3)

    # Block scale tensors for Q, K (FP8_E8M0 with F8_128x4 reordering)
    # Shape: [B, H, S_padded, D_scale_padded], d_scale contiguous (stride[3]=1)
    sf_qk_dims = (b, h, dims["s_padded"], dims["d_scale_padded"])
    sf_qk_strides = (h * dims["s_padded"] * dims["d_scale_padded"], dims["s_padded"] * dims["d_scale_padded"], dims["d_scale_padded"], 1)

    sf_q_tensor = graph.tensor(dim=sf_qk_dims, stride=sf_qk_strides, data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)
    sf_k_tensor = graph.tensor(dim=sf_qk_dims, stride=sf_qk_strides, data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)

    # Block scale tensor for V (FP8_E8M0 with F8_128x4 reordering)
    # Shape: [B, H, S_scale_padded, D_padded], s_scale contiguous (stride[2]=1)
    sf_v_dims = (b, h, dims["s_scale_padded"], dims["d_padded"])
    sf_v_strides = (h * dims["s_scale_padded"] * dims["d_padded"], dims["s_scale_padded"] * dims["d_padded"], 1, dims["s_scale_padded"])

    sf_v_tensor = graph.tensor(dim=sf_v_dims, stride=sf_v_strides, data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)

    # Call MXFP8 SDPA
    o, stats, amax_o = graph.sdpa_mxfp8(
        q=q,
        k=k,
        v=v,
        descale_q=sf_q_tensor,
        descale_k=sf_k_tensor,
        descale_v=sf_v_tensor,
        attn_scale=attn_scale,
        use_causal_mask=True,
        generate_stats=True,
    )

    # Set output tensor properties
    o_strides = (h * s * d, s * d, d, 1)
    o.set_output(True).set_dim(qkv_dims).set_stride(o_strides).set_data_type(cudnn.data_type.BFLOAT16)
    amax_o.set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    # Validate and build the graph
    try:
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A])
        graph.check_support()
        graph.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"MXFP8 SDPA not supported: {e}")
    except Exception as e:
        pytest.fail(f"Error building MXFP8 SDPA graph: {e}")

    # Allocate output tensors
    o_gpu = torch.empty(b, h, s, d, dtype=torch.bfloat16, device="cuda")
    amax_o_gpu = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device="cuda")
    stats_gpu = torch.empty(b, h, s, 1, dtype=torch.float32, device="cuda")

    # Reshape cute tensors for cuDNN (they need to be contiguous in the right layout)
    # The cute tensor is already in F8_128x4 layout, reshape to [B, H, S_padded, D_scale_padded]
    sf_q_cudnn = sf_q_cute.reshape(b, h, dims["s_padded"], dims["d_scale_padded"]).contiguous()
    sf_k_cudnn = sf_k_cute.reshape(b, h, dims["s_padded"], dims["d_scale_padded"]).contiguous()
    # For V, reshape to [B, H, S_scale_padded, D_padded] with s_scale contiguous
    sf_v_cudnn = sf_v_cute.reshape(b, h, dims["s_scale_padded"], dims["d_padded"]).permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)

    # Build variant pack
    variant_pack = {
        q: q_fp8,
        k: k_fp8,
        v: v_fp8,
        sf_q_tensor: sf_q_cudnn,
        sf_k_tensor: sf_k_cudnn,
        sf_v_tensor: sf_v_cudnn,
        o: o_gpu,
        amax_o: amax_o_gpu,
        stats: stats_gpu,
    }

    # Execute
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    # Compare results
    o_gpu_f32 = o_gpu.float()
    atol, rtol = 0.1, 0.2  # FP8 has limited precision

    torch.testing.assert_close(o_gpu_f32, o_ref, atol=atol, rtol=rtol)


@pytest.mark.L2
def test_sdpa_mxfp8_graph_build(request):
    """Test that MXFP8 SDPA graph can be built on supported configurations."""
    if request.config.option.dryrun:
        pytest.skip("dry run mode")

    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.21.0":
        pytest.skip("MXFP8 SDPA requires cuDNN 9.21.0 or higher")

    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 SDPA requires Blackwell or higher")

    # Problem dimensions
    b = 2  # batch size
    h = 2  # number of heads
    s = 512  # sequence length
    d = 128  # hidden head dim

    attn_scale = 0.123

    # Compute scale tensor dimensions
    dims = compute_mxfp8_scale_dims(b, h, s, d)

    # Create graph
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    # Q, K, V tensors (FP8_E4M3) - BHSD layout with packed QKV (bs3hd interleaved)
    qkv_dims = (b, h, s, d)
    qkv_strides = (s * 3 * h * d, d, 3 * h * d, 1)  # bs3hd
    o_strides = (s * h * d, d, h * d, 1)  # bshd

    q = graph.tensor(dim=qkv_dims, stride=qkv_strides, data_type=cudnn.data_type.FP8_E4M3)
    k = graph.tensor(dim=qkv_dims, stride=qkv_strides, data_type=cudnn.data_type.FP8_E4M3)
    v = graph.tensor(dim=qkv_dims, stride=qkv_strides, data_type=cudnn.data_type.FP8_E4M3)

    # Block scale tensors for Q, K (FP8_E8M0 with F8_128x4 reordering)
    # Q and K scale the d (hidden) dimension since BMM1 contracts on d
    sf_qk_dims = (b, h, dims["s_padded"], dims["d_scale_padded"])
    sf_qk_strides = (h * dims["s_padded"] * dims["d_scale_padded"], dims["s_padded"] * dims["d_scale_padded"], dims["d_scale_padded"], 1)

    sf_q = graph.tensor(dim=sf_qk_dims, stride=sf_qk_strides, data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)

    sf_k = graph.tensor(dim=sf_qk_dims, stride=sf_qk_strides, data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)

    # Block scale tensor for V (FP8_E8M0 with F8_128x4 reordering)
    # V scales the s (sequence) dimension since BMM2 (S @ V) contracts on s_kv
    # The contracting dimension (s_scale) must be contiguous, so use COL_MAJOR-like strides
    sf_v_dims = (b, h, dims["s_scale_padded"], dims["d_padded"])
    sf_v_strides = (h * dims["s_scale_padded"] * dims["d_padded"], dims["s_scale_padded"] * dims["d_padded"], 1, dims["s_scale_padded"])  # s_scale contiguous

    sf_v = graph.tensor(dim=sf_v_dims, stride=sf_v_strides, data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)

    # Call MXFP8 SDPA
    o, stats, amax_o = graph.sdpa_mxfp8(
        q=q, k=k, v=v, descale_q=sf_q, descale_k=sf_k, descale_v=sf_v, attn_scale=attn_scale, use_causal_mask=True, generate_stats=True
    )

    # Set output tensor properties
    o.set_output(True).set_dim(qkv_dims).set_stride(o_strides).set_data_type(cudnn.data_type.BFLOAT16)
    amax_o.set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    # Validate and build the graph
    try:
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A])
        graph.check_support()
        graph.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"MXFP8 SDPA not supported: {e}")
    except Exception as e:
        pytest.fail(f"Error building MXFP8 SDPA graph: {e}")


@pytest.mark.L2
def test_sdpa_mxfp8_unsupported_version(request):
    """Test that MXFP8 SDPA returns GRAPH_NOT_SUPPORTED on old cuDNN versions."""
    if request.config.option.dryrun:
        pytest.skip("dry run mode")

    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version >= "9.21.0":
        pytest.skip("This test is for older cuDNN versions")

    # Problem dimensions
    b, h, s, d = 2, 2, 512, 128
    attn_scale = 0.123

    dims = compute_mxfp8_scale_dims(b, h, s, d)

    graph = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    qkv_dims = (b, h, s, d)
    qkv_strides = (s * 3 * h * d, d, 3 * h * d, 1)

    q = graph.tensor(dim=qkv_dims, stride=qkv_strides, data_type=cudnn.data_type.FP8_E4M3)
    k = graph.tensor(dim=qkv_dims, stride=qkv_strides, data_type=cudnn.data_type.FP8_E4M3)
    v = graph.tensor(dim=qkv_dims, stride=qkv_strides, data_type=cudnn.data_type.FP8_E4M3)

    sf_qk_dims = (b, h, dims["s_padded"], dims["d_scale_padded"])
    sf_qk_strides = (h * dims["s_padded"] * dims["d_scale_padded"], dims["s_padded"] * dims["d_scale_padded"], dims["d_scale_padded"], 1)

    sf_q = graph.tensor(dim=sf_qk_dims, stride=sf_qk_strides, data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)
    sf_k = graph.tensor(dim=sf_qk_dims, stride=sf_qk_strides, data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)

    sf_v_dims = (b, h, dims["s_scale_padded"], dims["d_padded"])
    sf_v_strides = (h * dims["s_scale_padded"] * dims["d_padded"], dims["s_scale_padded"] * dims["d_padded"], 1, dims["s_scale_padded"])

    sf_v = graph.tensor(dim=sf_v_dims, stride=sf_v_strides, data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)

    o, stats, amax_o = graph.sdpa_mxfp8(
        q=q, k=k, v=v, descale_q=sf_q, descale_k=sf_k, descale_v=sf_v, attn_scale=attn_scale, use_causal_mask=True, generate_stats=True
    )

    o_strides = (s * h * d, d, h * d, 1)
    o.set_output(True).set_dim(qkv_dims).set_stride(o_strides).set_data_type(cudnn.data_type.BFLOAT16)
    amax_o.set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    # On older cuDNN, validate() should return GRAPH_NOT_SUPPORTED
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        graph.validate()
