"""
Utilities and parameterization for Grouped GEMM SwiGLU tests.
Contains test configuration fixtures, tensor creation, and reference implementations.

Reference: continugous_blockscaled_grouped_gemm_swiglu_quant_fusion.py (lines 3518-4825)
"""

import torch
import pytest
from typing import Optional, Tuple, List, Dict, Any
from test_fe_api_utils import (
    ceil_div,
    compute_reference_amax,
    create_and_permute_tensor,
    create_scale_factor_tensor,
    create_sf_layout_tensor,
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L,
)
from test_low_precision_matmul import (
    _bfloat16_to_float4_e2m1fn_x2,
    float4_e2m1fn_x2_to_float32,
)

# =============================================================================
# Parameterization Marks
# =============================================================================

GROUPED_GEMM_SWIGLU_PARAM_MARKS_FP8 = [
    pytest.mark.parametrize(
        "ab_dtype",
        [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ],
    ),
    pytest.mark.parametrize(
        "c_dtype",
        [
            # torch.float8_e4m3fn,
            # torch.float8_e5m2,
            # torch.float16,
            torch.bfloat16,
            # torch.float32,
        ],
    ),
    pytest.mark.parametrize(
        "d_dtype",
        [
            torch.float8_e4m3fn,
            # torch.float8_e5m2,
            # torch.bfloat16,
        ],
    ),
    pytest.mark.parametrize("cd_major", ["n"]),
    pytest.mark.parametrize("acc_dtype", [torch.float32]),
    pytest.mark.parametrize(
        "mma_tiler_mn",
        [
            (256, 256),
        ],
    ),
    pytest.mark.parametrize(
        "cluster_shape_mn",
        [
            (2, 1),
            (1, 1),
        ],
    ),
    pytest.mark.parametrize("sf_vec_size", [32]),
    pytest.mark.parametrize(
        "sf_dtype",
        [
            torch.float8_e8m0fnu,
        ],
    ),
    pytest.mark.parametrize("vector_f32", [True, False]),
    pytest.mark.parametrize("discrete_col_sfd", [True, False]),
]

GROUPED_GEMM_SWIGLU_PARAM_MARKS_FP4 = [
    pytest.mark.parametrize(
        "ab_dtype",
        [
            torch.float4_e2m1fn_x2,
            # torch.uint8,
        ],
    ),
    pytest.mark.parametrize(
        "c_dtype",
        [
            # torch.float16,
            torch.bfloat16,
        ],
    ),
    pytest.mark.parametrize(
        "d_dtype",
        [
            torch.bfloat16,
            torch.float32,
        ],
    ),
    pytest.mark.parametrize("cd_major", ["n"]),
    pytest.mark.parametrize("acc_dtype", [torch.float32]),
    pytest.mark.parametrize(
        "mma_tiler_mn",
        [
            (256, 256),
            (128, 128),
        ],
    ),
    pytest.mark.parametrize(
        "cluster_shape_mn",
        [
            (2, 1),
            (1, 1),
        ],
    ),
    pytest.mark.parametrize("sf_vec_size", [16, 32]),
    pytest.mark.parametrize(
        "sf_dtype",
        [
            torch.float8_e8m0fnu,
            torch.float8_e4m3fn,
        ],
    ),
    pytest.mark.parametrize("vector_f32", [True, False]),
    pytest.mark.parametrize("discrete_col_sfd", [False]),
]


def with_grouped_gemm_swiglu_params_fp4(func):
    """Decorator to apply grouped GEMM SwiGLU FP4 test parameters."""
    for mark in reversed(GROUPED_GEMM_SWIGLU_PARAM_MARKS_FP4):
        func = mark(func)
    return func


def with_grouped_gemm_swiglu_params_fp8(func):
    """Decorator to apply grouped GEMM SwiGLU FP8 test parameters."""
    for mark in reversed(GROUPED_GEMM_SWIGLU_PARAM_MARKS_FP8):
        func = mark(func)
    return func


# =============================================================================
# Configuration Initialization
# =============================================================================


def grouped_gemm_swiglu_init(
    request,
    ab_dtype: torch.dtype,
    c_dtype: torch.dtype,
    d_dtype: torch.dtype,
    cd_major: str,
    acc_dtype: torch.dtype,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    sf_vec_size: int,
    sf_dtype: torch.dtype,
    vector_f32: bool = False,
    discrete_col_sfd: bool = False,
) -> Dict[str, Any]:
    """Initialize configuration for Grouped GEMM SwiGLU tests.

    :param request: pytest request object
    :param ab_dtype: Data type for A and B tensors
    :param c_dtype: Data type for intermediate C tensor (always bfloat16)
    :param d_dtype: Data type for output D tensor (fp8 when ab is fp8, bf16 when ab is fp4)
    :param cd_major: Major dimension for output C and D tensors
    :param acc_dtype: Accumulator data type
    :param mma_tiler_mn: MMA tiler shape
    :param cluster_shape_mn: Cluster shape
    :param sf_vec_size: Scale factor vector size
    :param sf_dtype: Scale factor data type
    :param vector_f32: Use vectorized f32 operations
    :param discrete_col_sfd: Generate discrete col-major scale factor tensor
    :return: Configuration dictionary
    """
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(f"Environment not supported: requires compute capability >= 10, found {major}")

    # Parse CLI options
    nkl_str = request.config.getoption("--grouped-gemm-nkl", default=None)
    group_m_str = request.config.getoption("--grouped-gemm-group-m", default=None)
    m_aligned_opt = request.config.getoption("--grouped-gemm-m-aligned", default=None)
    skip_ref = request.config.getoption("--grouped-gemm-skip-ref", default=False)

    # Default values
    if nkl_str is not None:
        n, k, l = [int(x.strip()) for x in nkl_str.split(",")]
    else:
        n, k, l = 512, 512, 4

    if group_m_str is not None:
        group_m_list = [int(x.strip()) for x in group_m_str.split(",")]
    else:
        # Default: equal M values per group
        group_m_list = [256] * l

    m_aligned = int(m_aligned_opt) if m_aligned_opt is not None else mma_tiler_mn[0]

    config = {
        "n": n,
        "k": k,
        "l": l,
        "group_m_list": group_m_list,
        "m_aligned": m_aligned,
        "mma_tiler_mn": mma_tiler_mn,
        "cluster_shape_mn": cluster_shape_mn,
        "ab_dtype": ab_dtype,
        "c_dtype": c_dtype,
        "d_dtype": d_dtype,
        "cd_major": cd_major,
        "acc_dtype": acc_dtype,
        "sf_vec_size": sf_vec_size,
        "sf_dtype": sf_dtype,
        "vector_f32": vector_f32,
        "skip_ref": skip_ref,
        "discrete_col_sfd": discrete_col_sfd,
    }

    return config


# =============================================================================
# Helper Functions
# =============================================================================


def get_dtype_rcp_limits(dtype: torch.dtype) -> float:
    """Get reciprocal of max value for quantization."""
    if dtype == torch.float8_e5m2:
        return 1 / 128.0
    elif dtype == torch.float8_e4m3fn:
        return 1 / 448.0
    elif dtype in {torch.float4_e2m1fn_x2, torch.uint8}:
        return 1 / 6.0
    return 1.0


def create_mask(
    group_m_list: List[int],
    cta_tile_m: int,
    m_aligned: int = 128,
    permuted_m: Optional[int] = None,
) -> Tuple[int, List[int], torch.Tensor, torch.Tensor]:
    """Create mask and group mapping for contiguous grouped GEMM.

    :param group_m_list: List of M values for each group (will be aligned to m_aligned)
    :param cta_tile_m: CTA tile size in M dimension (from mma_tiler_mn[0])
    :param m_aligned: Alignment requirement for group M dimension
    :param permuted_m: Optional padded M dimension for CUDA graph support

    Note: m_aligned should be a multiple of the CTA tile M dimension to prevent
          a single tile from spanning multiple groups, which would cause incorrect
          B matrix access.

    Note: For cuda_graph support, set permuted_m to the pre-calculated padded size:
          permuted_m = m * topK + num_local_experts * (256 - 1)
          Example: 4096*8 + (256/32)*255 = 34808
          Only the actual valid rows (aligned_groupm[0]+aligned_groupm[1]+...) contain
          valid data. The kernel will exit when tile_idx >= num_non_exiting_tiles.

    :return: Tuple of (valid_m, aligned_group_m_list, tile_idx_to_expert_idx, num_non_exiting_tiles, num_m_split_cumsum)
             - tile_idx_to_expert_idx: shape (permuted_m/cta_tile_m,) if permuted_m provided,
               else (valid_m/cta_tile_m,)
             - num_non_exiting_tiles: scalar value = valid_m/cta_tile_m
             - num_m_split_cumsum: cumulative sum of aligned_group_m_list
    """
    valid_m = 0
    aligned_group_m_list = []
    tile_idx_to_expert_idx = []
    m_split_cumsum = []
    m_split_cumsum.append(valid_m)

    for i, group_m in enumerate(group_m_list):
        aligned_group_m = ((group_m + m_aligned - 1) // m_aligned) * m_aligned
        valid_m += aligned_group_m
        aligned_group_m_list.append(aligned_group_m)

        # Calculate number of tiles for this group based on CTA tile M size
        # Each tile covers cta_tile_m rows in M dimension
        num_tiles_in_group = aligned_group_m // cta_tile_m
        # Add expert_idx for each tile in this group
        tile_idx_to_expert_idx.extend([i] * num_tiles_in_group)
        m_split_cumsum.append(valid_m)

    # Compute num_non_exiting_tiles (number of valid tiles in M dimension)
    num_non_exiting_tiles = len(tile_idx_to_expert_idx)

    # Apply padding if requested (for cuda_graph support)
    if permuted_m is not None:
        if permuted_m < valid_m:
            raise ValueError(f"permuted_m ({permuted_m}) must be >= valid_m ({valid_m}). " f"Cannot pad to a smaller size.")
        if permuted_m > valid_m:
            # Calculate how many padding tiles are needed based on CTA tile M size
            num_padding_tiles = (permuted_m - valid_m) // cta_tile_m
            # Pad with large negative value (these tiles won't be accessed due to
            # num_non_exiting_tiles check)
            tile_idx_to_expert_idx.extend([int(-2e9)] * num_padding_tiles)

    # Convert to tensors
    tile_idx_to_expert_idx_tensor = torch.tensor(tile_idx_to_expert_idx, device="cuda", dtype=torch.int32)
    num_non_exiting_tiles_tensor = torch.tensor([num_non_exiting_tiles], device="cuda", dtype=torch.int32)
    num_m_split_cumsum_tensor = torch.tensor(m_split_cumsum, device="cuda", dtype=torch.int32)

    return (
        valid_m,
        aligned_group_m_list,
        tile_idx_to_expert_idx_tensor,
        num_non_exiting_tiles_tensor,
        num_m_split_cumsum_tensor,
    )


# =============================================================================
# Tensor Allocation
# =============================================================================


def allocate_grouped_gemm_input_tensors(
    n: int,
    k: int,
    l: int,
    group_m_list: List[int],
    ab_dtype: torch.dtype,
    sf_dtype: torch.dtype,
    sf_vec_size: int,
    m_aligned: int,
    cta_tile_m: int,
    permuted_m: Optional[int] = None,
    norm_const: float = 1.0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Allocate input tensors for grouped GEMM SwiGLU.

    Matches the original create_tensors() implementation.

    :return: Dictionary containing all input tensors and metadata
    """

    (
        valid_m,
        aligned_group_m_list,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles,
        num_m_split_cumsum,
    ) = create_mask(group_m_list, cta_tile_m, m_aligned, permuted_m)

    tensor_m = permuted_m if permuted_m is not None else valid_m

    # Note: a and b tensors are always K-major
    a_ref, a_tensor = create_and_permute_tensor(1, tensor_m, k, False, ab_dtype)
    b_ref, b_tensor = create_and_permute_tensor(l, n, k, False, ab_dtype)

    sfa_ref, sfa_tensor = create_scale_factor_tensor(1, tensor_m, k, sf_vec_size, sf_dtype)
    sfb_ref, sfb_tensor = create_scale_factor_tensor(l, n, k, sf_vec_size, sf_dtype)

    alpha_tensor = torch.randint(-2, 2, (l,), dtype=torch.float32, device=device).float()

    prob_tensor = torch.randint(-2, 2, (tensor_m, 1, 1), dtype=torch.float32, device=device).float()

    result = {
        "a_tensor": a_tensor,
        "a_ref": a_ref,
        "b_tensor": b_tensor,
        "b_ref": b_ref,
        "sfa_tensor": sfa_tensor,
        "sfa_ref": sfa_ref,
        "sfb_tensor": sfb_tensor,
        "sfb_ref": sfb_ref,
        "alpha_tensor": alpha_tensor,
        "prob_tensor": prob_tensor,
        "tile_idx_to_expert_idx": tile_idx_to_expert_idx,
        "num_non_exiting_tiles": num_non_exiting_tiles,
        "num_m_split_cumsum_tensor": num_m_split_cumsum,
        "aligned_group_m_list": aligned_group_m_list,
        "valid_m": valid_m,
        "tensor_m": tensor_m,
        "norm_const_tensor": None,
    }

    # Norm constant tensor
    if ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and sf_dtype in [
        torch.float8_e8m0fnu,
        torch.float8_e4m3fn,
    ]:
        result["norm_const_tensor"] = torch.tensor([norm_const], dtype=torch.float32, device=device)

    return result


def allocate_grouped_gemm_output_tensors(
    tensor_m: int,
    n: int,
    l: int,
    ab_dtype: torch.dtype,
    c_dtype: torch.dtype,
    d_dtype: torch.dtype,
    cd_major: str,
    sf_dtype: torch.dtype,
    sf_vec_size: int = 16,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Allocate output tensors for grouped GEMM SwiGLU.

    Matches the original create_tensors() implementation.

    :param c_dtype: Data type for intermediate C tensor (always bfloat16)
    :param d_dtype: Data type for output D tensor (fp8 when ab is fp8, bf16 when ab is fp4)
    :return: Dictionary containing all output tensors
    """
    n_out = n // 2  # After SwiGLU

    _, c_tensor = create_and_permute_tensor(1, tensor_m, n, cd_major == "m", c_dtype)
    _, d_tensor = create_and_permute_tensor(1, tensor_m, n_out, cd_major == "m", d_dtype)
    _, d_col_tensor = create_and_permute_tensor(1, tensor_m, n_out, cd_major == "m", d_dtype)

    result = {
        "c_tensor": c_tensor,
        "d_tensor": d_tensor,
        "d_col_tensor": d_col_tensor,
        "sfd_row_tensor": None,
        "sfd_col_tensor": None,
    }

    if d_dtype in [torch.bfloat16, torch.float16]:
        result["amax_tensor"] = torch.full((l, 1), float("-inf"), dtype=torch.float32, device=device)

    if ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and sf_dtype in [
        torch.float8_e8m0fnu,
        torch.float8_e4m3fn,
    ]:  # generate_sfd
        sfd_row_ref, sfd_row_tensor = create_scale_factor_tensor(1, tensor_m, n_out, sf_vec_size, sf_dtype)
        result["sfd_row_tensor"] = sfd_row_tensor
        result["sfd_row_ref"] = sfd_row_ref

        sfd_col_ref, sfd_col_tensor = create_scale_factor_tensor(1, n_out, tensor_m, sf_vec_size, sf_dtype)
        result["sfd_col_tensor"] = sfd_col_tensor
        result["sfd_col_ref"] = sfd_col_ref

    return result


# =============================================================================
# Reference Implementations
# =============================================================================


def run_grouped_gemm_swiglu_ref(
    a_ref: torch.Tensor,
    b_ref: torch.Tensor,
    sfa_ref: torch.Tensor,
    sfb_ref: torch.Tensor,
    alpha_tensor: torch.Tensor,
    prob_tensor: torch.Tensor,
    aligned_group_m_list: List[int],
    valid_m: int,
    generate_amax: bool = False,
    generate_sfd: bool = False,
    norm_const_tensor: Optional[torch.Tensor] = None,
    c_dtype: torch.dtype = torch.bfloat16,
    d_dtype: torch.dtype = torch.float32,
    sf_vec_size: int = 16,
    sf_dtype: torch.dtype = torch.float8_e8m0fnu,
) -> torch.Tensor:
    """Run reference implementation for grouped GEMM SwiGLU.

    Matches the reference checking in continugous_blockscaled_grouped_gemm_swiglu_quant_fusion.py
    (lines 4113-4179)

    :param a_ref: A tensor (tensor_m, k, 1) in float32
    :param b_ref: B tensor (n, k, l) in float32
    :param sfa_ref: Scale factor A tensor (tensor_m, k, 1) in float32
    :param sfb_ref: Scale factor B tensor (n, k, l) in float32
    :param alpha_tensor: Per-group alpha scaling (l,)
    :param prob_tensor: Per-row probability scaling (tensor_m, 1, 1)
    :param aligned_group_m_list: Aligned M values per group
    :param valid_m: Total valid M dimension
    :param generate_amax: Generate AMAX tensor
    :param generate_sfd: Generate SFD tensor
    :param norm_const_tensor: Normalization constant tensor (1,)
    :param c_dtype: Intermediate C tensor dtype (always bfloat16)
    :param d_dtype: Output D tensor dtype
    :param sf_vec_size: Scale factor vector size
    :param sf_dtype: Scale factor dtype
    :return: Reference output tensor (valid_m, n_out, 1)
    """
    n, k, l = b_ref.shape
    n_out = n // 2
    ref_tensors = {}

    # Step 1: Compute GEMM per group with scale factors
    ref = torch.empty((1, valid_m, n), dtype=torch.float32, device=a_ref.device)
    start = 0
    for i, group_m in enumerate(aligned_group_m_list):
        end = start + group_m
        res_a = torch.einsum("mk,mk->mk", a_ref[start:end, :, 0], sfa_ref[start:end, :, 0])
        res_b = torch.einsum("nk,nk->nk", b_ref[:, :, i], sfb_ref[:, :, i])
        ref[0, start:end, :] = torch.einsum("mk,nk->mn", res_a, res_b)
        start = end
    ref = ref.permute((1, 2, 0))

    # Step 2: Apply alpha per group
    start = 0
    for i, group_m in enumerate(aligned_group_m_list):
        end = start + group_m
        ref[start:end, :, 0] = ref[start:end, :, 0] * alpha_tensor[i].item()
        start = end

    ref_tensors["c_ref"] = ref.clone()

    # Step 3: Apply SwiGLU with interleaved block layout
    group = 32
    assert n % group == 0, "N must be divisible by 32 for GLU block grouping"
    num_blocks = n // group
    assert num_blocks % 2 == 0, "Number of 32-col blocks must be even (pairs of input/gate)"

    cols = torch.arange(n, device=ref.device, dtype=torch.long)
    block_cols = cols.view(num_blocks, group)
    # up: blocks 0,2,4,6,... (even blocks)
    # gate: blocks 1,3,5,7,... (odd blocks)
    up_idx = block_cols[0::2].reshape(-1)
    gate_idx = block_cols[1::2].reshape(-1)
    ref_up = ref.index_select(1, up_idx)
    ref_gate = ref.index_select(1, gate_idx)

    # SwiGLU: up * (gate * sigmoid(gate))
    ref_gate = ref_gate * torch.sigmoid(ref_gate)
    ref_after_swiglu = ref_up * ref_gate

    # Step 4: Apply prob
    ref_after_swiglu = ref_after_swiglu * prob_tensor.expand(-1, n_out, -1)
    ref_tensors["d_ref"] = ref_after_swiglu.clone()

    if generate_amax:
        amax_ref = torch.empty((l,), dtype=torch.float32, device=a_ref.device)
        start = 0
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            amax_ref[i] = compute_reference_amax(ref_after_swiglu[start:end, :, 0].clone())
            start = end
        ref_tensors["amax_ref"] = amax_ref

    if generate_sfd:
        try:
            from cutlass.cute.runtime import from_dlpack
            import cutlass.cute as cute
            from cudnn.datatypes import _convert_to_cutlass_data_type
        except ImportError:
            pytest.skip("CUTLASS not available for scale factor conversion")

        norm_const = norm_const_tensor[0].item()

        # 1. Compute reference SFDRow (m, sfn, l) in fp32
        sfn = ceil_div(n_out, sf_vec_size)
        # Resahpe ref to (l, m, sfn, sf_vec_size)
        ref_for_sf = ref_after_swiglu.permute(2, 0, 1).contiguous()  # (l, m, n)
        # l is involved in valid_m
        ref_for_sf = ref_for_sf.view(1, valid_m, sfn, sf_vec_size)
        # Take abs max over sf_vec_size dimension
        ref_for_sf, _ = torch.abs(ref_for_sf).max(dim=3)  # (l, m, sfn)
        # Multiply by norm_const and rcp_limits
        ref_sfd_row_f32 = ref_for_sf * norm_const * get_dtype_rcp_limits(d_dtype)
        # Permute to (m, sfn, l)
        ref_sfd_row_f32 = ref_sfd_row_f32.permute(1, 2, 0)

        # Convert fp32 -> f8 -> fp32 for ref_sfd_row_f32
        ref_sfd_row_f8_torch = torch.empty(*(1, valid_m, sfn), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
        ref_sfd_row_f8 = from_dlpack(ref_sfd_row_f8_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        ref_sfd_row_f8.element_type = _convert_to_cutlass_data_type(sf_dtype)
        ref_sfd_row_f32_device = ref_sfd_row_f32.cuda()
        ref_sfd_row_f32_tensor = from_dlpack(ref_sfd_row_f32_device, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        cute.testing.convert(ref_sfd_row_f32_tensor, ref_sfd_row_f8)
        cute.testing.convert(ref_sfd_row_f8, ref_sfd_row_f32_tensor)
        ref_sfd_row_f32 = ref_sfd_row_f32_device.cpu()

        # 2. Convert ref_sfd_row_f32 to scale factor layout and compare with kernel sfd tensor
        ref_sfd_row_f32_cute_torch_tensor_cpu, _ = create_sf_layout_tensor(1, valid_m, n_out, sf_vec_size)

        # convert ref_after_swiglu f32 tensor to cute f32 tensor
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
            from_dlpack(ref_sfd_row_f32),
            from_dlpack(ref_sfd_row_f32_cute_torch_tensor_cpu),
        )
        ref_sfd_row_f32 = ref_sfd_row_f32.cuda()
        ref_tensors["sfd_row_ref"] = ref_sfd_row_f32_cute_torch_tensor_cpu.clone()

        # 3. Quantized output with scale factor
        # Compute reciprocal of ref_sfd_row_f32 and multiply by norm_const
        ref_sfd_row_rcp = norm_const * ref_sfd_row_f32.reciprocal()
        ref_sfd_row_rcp = torch.clamp(ref_sfd_row_rcp, max=3.40282346638528859812e38)
        # Expand the sfn dimension by repeating each value sf_vec_size times
        # ref_sfd_row_rcp: (m, sfn, l) -> (m, sfn, sf_vec_size, l) -> (m, n, l)
        ref_sfd_row_rcp_expanded = ref_sfd_row_rcp.unsqueeze(2).expand(valid_m, sfn, sf_vec_size, 1)
        ref_sfd_row_rcp_expanded = ref_sfd_row_rcp_expanded.reshape(valid_m, sfn * sf_vec_size, 1)
        # Trim to exact n dimension if needed
        ref_sfd_row_rcp_expanded = ref_sfd_row_rcp_expanded[:, :n_out, :]

        # Apply scale to reference output: ref = ref * ref_sfd_row_rcp
        ref_after_row_quant = torch.einsum("mnl,mnl->mnl", ref_after_swiglu, ref_sfd_row_rcp_expanded)
        ref_tensors["d_ref"] = ref_after_row_quant.clone()

        # Col Quantized SFD tensor
        # 1. Compute reference SFDCol (m, sfn, l) in fp32
        ref_after_swiglu = ref_after_swiglu.permute(2, 1, 0).contiguous().permute(1, 2, 0)
        n_after_swiglu = ref_after_swiglu.shape[1]
        sfn = ceil_div(n_after_swiglu, sf_vec_size)
        valid_m = ref_after_swiglu.shape[0]
        # Reshape ref to (l, m, sfn, sf_vec_size)
        ref_for_sf = ref_after_swiglu.permute(2, 0, 1).contiguous()  # (l, m, n)
        # l is involved in valid_m
        ref_for_sf = ref_for_sf.view(1, valid_m, sfn, sf_vec_size)
        # Take abs max over sf_vec_size dimension
        ref_for_sf, _ = torch.abs(ref_for_sf).max(dim=3)  # (l, m, sfn)
        # Multiply by norm_const and rcp_limits
        ref_sfd_row_f32 = ref_for_sf * norm_const * get_dtype_rcp_limits(d_dtype)
        # Permute to (m, sfn, l)
        ref_sfd_row_f32 = ref_sfd_row_f32.permute(1, 2, 0)

        # Convert fp32 -> f8 -> fp32 for ref_sfd_row_f32
        ref_sfd_row_f8_torch = torch.empty(*(1, valid_m, sfn), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
        ref_sfd_row_f8 = from_dlpack(ref_sfd_row_f8_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        ref_sfd_row_f8.element_type = _convert_to_cutlass_data_type(sf_dtype)
        ref_sfd_row_f32_device = ref_sfd_row_f32.cuda()
        ref_sfd_row_f32_tensor = from_dlpack(ref_sfd_row_f32_device, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        cute.testing.convert(ref_sfd_row_f32_tensor, ref_sfd_row_f8)
        cute.testing.convert(ref_sfd_row_f8, ref_sfd_row_f32_tensor)
        ref_sfd_row_f32 = ref_sfd_row_f32_device.cpu()

        # 2. Convert ref_sfd_row_f32 to scale factor layout and compare with kernel sfd tensor
        ref_sfd_row_f32_cute_torch_tensor_cpu, _ = create_sf_layout_tensor(1, valid_m, n_after_swiglu, sf_vec_size)

        # convert ref_after_swiglu f32 tensor to cute f32 tensor
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
            from_dlpack(ref_sfd_row_f32),
            from_dlpack(ref_sfd_row_f32_cute_torch_tensor_cpu),
        )
        ref_sfd_row_f32 = ref_sfd_row_f32.cuda()
        ref_tensors["sfd_col_ref"] = ref_sfd_row_f32_cute_torch_tensor_cpu.clone()

        # 3. Quantized output with scale factor
        # Compute reciprocal of ref_sfd_row_f32 and multiply by norm_const
        ref_sfd_row_rcp = norm_const * ref_sfd_row_f32.reciprocal()
        ref_sfd_row_rcp = torch.clamp(ref_sfd_row_rcp, max=3.40282346638528859812e38)
        # Expand the sfn dimension by repeating each value sf_vec_size times
        # ref_sfd_row_rcp: (m, sfn, l) -> (m, sfn, sf_vec_size, l) -> (m, n, l)
        ref_sfd_row_rcp_expanded = ref_sfd_row_rcp.unsqueeze(2).expand(valid_m, sfn, sf_vec_size, 1)
        ref_sfd_row_rcp_expanded = ref_sfd_row_rcp_expanded.reshape(valid_m, sfn * sf_vec_size, 1)
        # Trim to exact n dimension if needed
        ref_sfd_row_rcp_expanded = ref_sfd_row_rcp_expanded[:, :n_after_swiglu, :]

        # Apply scale to reference output: ref = ref * ref_sfd_row_rcp
        ref_after_row_quant = torch.einsum("mnl,mnl->mnl", ref_after_swiglu, ref_sfd_row_rcp_expanded)

        # Convert ref_after_row_quant : f32 -> f8 -> f32
        ref_ = torch.empty(*(1, valid_m, n_after_swiglu), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
        ref_ = from_dlpack(ref_, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        ref_.element_type = _convert_to_cutlass_data_type(d_dtype)
        ref_device = ref_after_row_quant.cuda()
        ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        cute.testing.convert(ref_tensor, ref_)
        cute.testing.convert(ref_, ref_tensor)

        ref_tensors["d_col_ref"] = ref_device.clone().permute(1, 0, 2)

    return ref_tensors


# =============================================================================
# Reference Checking
# =============================================================================


def check_ref_grouped_gemm_swiglu(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    cfg: Dict[str, Any],
    atol: float = 1e-1,
    rtol: float = 1e-2,
    skip_ref: bool = False,
) -> None:
    """Check grouped GEMM SwiGLU result against reference.

    :param inputs: Dictionary of input tensors (from allocate_grouped_gemm_input_tensors)
    :param outputs: Dictionary of output tensors (from allocate_grouped_gemm_output_tensors)
    :param cfg: Configuration dictionary (from grouped_gemm_swiglu_init)
    :param atol: Absolute tolerance
    :param rtol: Relative tolerance
    :param skip_ref: Skip reference check if True
    """
    if skip_ref:
        print("Skipping reference check")
        return

    # Run reference
    ref_tensors = run_grouped_gemm_swiglu_ref(
        a_ref=inputs["a_ref"].to(torch.float32),
        b_ref=inputs["b_ref"].to(torch.float32),
        sfa_ref=inputs["sfa_ref"].to(torch.float32),
        sfb_ref=inputs["sfb_ref"].to(torch.float32),
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        aligned_group_m_list=inputs["aligned_group_m_list"],
        valid_m=inputs["valid_m"],
        generate_amax=(outputs.get("amax_tensor") is not None),
        generate_sfd=(outputs.get("sfd_row_tensor") is not None),
        norm_const_tensor=inputs.get("norm_const_tensor"),
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        sf_dtype=cfg["sf_dtype"],
    )

    torch.cuda.synchronize()

    c_gpu = outputs["c_tensor"][: inputs["valid_m"]]
    c_ref = ref_tensors["c_ref"]
    torch.testing.assert_close(
        c_gpu.cpu().float(),
        c_ref.cpu().to(cfg["c_dtype"]).to(torch.float32),
        atol=atol,
        rtol=rtol,
    )

    if cfg["d_dtype"] in [torch.float32, torch.float16, torch.bfloat16]:
        if ref_tensors.get("amax_ref") is not None:
            amax_gpu = outputs["amax_tensor"]
            amax_ref = ref_tensors["amax_ref"]
            torch.testing.assert_close(
                amax_gpu.cpu().squeeze(),
                amax_ref.cpu(),
                atol=atol,
                rtol=rtol,
            )

        d_gpu = outputs["d_tensor"][: inputs["valid_m"]]
        d_ref = ref_tensors["d_ref"]
        torch.testing.assert_close(
            d_gpu.cpu().float(),
            d_ref.cpu().to(cfg["d_dtype"]).to(torch.float32),
            atol=atol,
            rtol=rtol,
        )
    elif cfg["d_dtype"] in [torch.float8_e4m3fn, torch.float8_e5m2]:
        if ref_tensors.get("sfd_row_ref") is not None:  # generate_sfd
            # sfd_row_ref
            sfd_row_gpu = outputs["sfd_row_tensor"]
            sfd_row_ref = ref_tensors["sfd_row_ref"]
            torch.testing.assert_close(
                sfd_row_gpu.cpu().float(),
                sfd_row_ref.cpu().to(torch.float32),
                atol=atol,
                rtol=rtol,
            )

            # d_ref (row)
            d_gpu = outputs["d_tensor"]
            d_ref = ref_tensors["d_ref"]
            torch.testing.assert_close(
                d_gpu.cpu().float(),
                d_ref.to(cfg["d_dtype"]).to(torch.float32).cpu(),
                atol=atol,
                rtol=rtol,
            )

            # sfd_col
            if cfg["discrete_col_sfd"]:
                # discrete col sfd
                group_m_list = inputs["aligned_group_m_list"]
                group_n_tile_list = [group // 128 for group in group_m_list]
                m_tile = ref_tensors["sfd_col_ref"].shape[2]

                sfd_col_torch_gpu_f8 = outputs["sfd_col_tensor"].cpu().to(torch.float32)
                sfd_col_ref_f32 = ref_tensors["sfd_col_ref"].cpu().to(torch.float32)

                res_real_idx = 0
                cumsum_n = 0
                total_n = sum(group_n_tile_list)
                for n_tile in group_n_tile_list:
                    for m_idx in range(m_tile):
                        for n_idx in range(n_tile):
                            res_real_m_idx = res_real_idx // total_n
                            res_real_n_idx = res_real_idx % total_n

                            ref_real_n_idx = n_idx + cumsum_n
                            ref_slice = sfd_col_ref_f32[:, :, m_idx, :, ref_real_n_idx, :]
                            res_slice = sfd_col_torch_gpu_f8[:, :, res_real_m_idx, :, res_real_n_idx, :]
                            torch.testing.assert_close(
                                ref_slice,
                                res_slice,
                                atol=atol,
                                rtol=rtol,
                            )
                            res_real_idx += 1
                    cumsum_n += n_tile
            else:
                # contiguous col sfd
                sfd_col_gpu = outputs["sfd_col_tensor"]
                sfd_col_ref = ref_tensors["sfd_col_ref"]
                torch.testing.assert_close(
                    sfd_col_gpu.cpu().float(),
                    sfd_col_ref.cpu().to(torch.float32),
                    atol=atol,
                    rtol=rtol,
                )

            # d_col_ref
            d_col_gpu = outputs["d_col_tensor"]
            d_col_ref = ref_tensors["d_col_ref"]
            torch.testing.assert_close(
                d_col_gpu.cpu().float(),
                d_col_ref.to(cfg["d_dtype"]).to(torch.float32).cpu(),
                atol=atol,
                rtol=rtol,
            )
        else:
            # Note: This is outside support surface
            d_gpu = outputs["d_tensor"][: inputs["valid_m"]]
            d_ref = ref_tensors["d_ref"][: inputs["valid_m"]]
            torch.testing.assert_close(
                d_gpu.cpu().float(),
                d_ref.cpu().to(cfg["d_dtype"]).to(torch.float32),
                atol=atol,
                rtol=rtol,
            )

    else:
        raise NotImplementedError(f"Unsupported dtype: {cfg['d_dtype']}")
