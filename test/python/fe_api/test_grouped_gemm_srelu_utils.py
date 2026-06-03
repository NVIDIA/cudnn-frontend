"""
Utilities and parameterization for Grouped GEMM SReLU tests.
Contains test configuration fixtures, tensor creation, and reference implementations.

Reference: continugous_blockscaled_grouped_gemm_srelu_quant_fusion.py (lines 3518-4825)
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

# =============================================================================
# Parameterization Marks
# =============================================================================

GROUPED_GEMM_SWIGLU_COMMON_MARKS = [
    pytest.mark.parametrize("cd_major", ["n"]),
    pytest.mark.parametrize("acc_dtype", [torch.float32]),
    pytest.mark.parametrize("cluster_shape_mn", [(2, 1), (1, 1)]),
    pytest.mark.parametrize("vector_f32", [True, False]),
]

GROUPED_GEMM_SWIGLU_FP8_TYPE_MARKS = [
    pytest.mark.parametrize(
        "ab_dtype",
        [
            torch.float8_e4m3fn,
        ],
    ),
    pytest.mark.parametrize("c_dtype", [torch.bfloat16]),
    pytest.mark.parametrize(
        "d_dtype",
        [
            torch.float8_e4m3fn,
        ],
    ),
]

GROUPED_GEMM_SWIGLU_FP4_TYPE_MARKS = [
    pytest.mark.parametrize(
        "ab_dtype",
        [
            torch.uint8,
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
            # torch.float16,
            torch.bfloat16,
            torch.float32,
        ],
    ),
]

GROUPED_GEMM_SWIGLU_PARAM_MARKS_FP8 = GROUPED_GEMM_SWIGLU_FP8_TYPE_MARKS + [
    pytest.mark.parametrize("cd_major", ["n"]),
    pytest.mark.parametrize("acc_dtype", [torch.float32]),
    pytest.mark.parametrize("mma_tiler_mn", [(256, 256)]),
    pytest.mark.parametrize("cluster_shape_mn", [(2, 1)]),
    pytest.mark.parametrize("vector_f32", [False]),
    pytest.mark.parametrize("sf_vec_size,sf_dtype", [(32, torch.float8_e8m0fnu)]),
    pytest.mark.parametrize("discrete_col_sfd", [True]),
]

GROUPED_GEMM_SWIGLU_PARAM_MARKS_FP4 = (
    GROUPED_GEMM_SWIGLU_FP4_TYPE_MARKS
    + GROUPED_GEMM_SWIGLU_COMMON_MARKS
    + [
        pytest.mark.parametrize("mma_tiler_mn", [(256, 256), (128, 256)]),
        pytest.mark.parametrize(
            "sf_vec_size,sf_dtype",
            [
                (16, torch.float8_e8m0fnu),
                (16, torch.float8_e4m3fn),
                (32, torch.float8_e8m0fnu),
                (32, torch.float8_e4m3fn),
            ],
        ),
        pytest.mark.parametrize("discrete_col_sfd", [False]),
    ]
)

GROUPED_GEMM_SWIGLU_PARAM_MARKS_BIAS_FP4 = (
    GROUPED_GEMM_SWIGLU_FP4_TYPE_MARKS
    + GROUPED_GEMM_SWIGLU_COMMON_MARKS
    + [
        pytest.mark.parametrize("mma_tiler_mn", [(128, 256), (256, 256)]),
        pytest.mark.parametrize(
            "sf_vec_size,sf_dtype",
            [
                (16, torch.float8_e8m0fnu),
                (16, torch.float8_e4m3fn),
                (32, torch.float8_e8m0fnu),
            ],
        ),
        pytest.mark.parametrize("discrete_col_sfd", [False]),
    ]
)


def with_grouped_gemm_srelu_params_fp4(func):
    """Decorator to apply grouped GEMM SReLU FP4 test parameters."""
    for mark in reversed(GROUPED_GEMM_SWIGLU_PARAM_MARKS_FP4):
        func = mark(func)
    return func


def with_grouped_gemm_srelu_params_fp8(func):
    """Decorator to apply grouped GEMM SReLU FP8 test parameters."""
    for mark in reversed(GROUPED_GEMM_SWIGLU_PARAM_MARKS_FP8):
        func = mark(func)
    return func


def with_grouped_gemm_srelu_params_bias_fp4(func):
    """Decorator to apply grouped GEMM SReLU dense bias FP4 test parameters."""
    for mark in reversed(GROUPED_GEMM_SWIGLU_PARAM_MARKS_BIAS_FP4):
        func = mark(func)
    return func


# =============================================================================
# Configuration Initialization
# =============================================================================


def grouped_gemm_srelu_init(
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
    b_major: str = "k",
    enable_bias: bool = False,
) -> Dict[str, Any]:
    """Initialize configuration for Grouped GEMM SReLU tests.

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
    :param b_major: Major dimension for B tensor.
    :param enable_bias: Allocate dense bias tensor for fused bias tests
    :return: Configuration dictionary
    """
    major, minor = torch.cuda.get_device_capability()
    compute_capability = major * 10 + minor
    if compute_capability < 100:
        pytest.skip(f"Environment not supported: requires compute capability >= 10, found {major}")

    # Parse CLI options
    nkl_str = request.config.getoption("--grouped-gemm-nkl", default=None)
    group_m_str = request.config.getoption("--grouped-gemm-group-m", default=None)
    skip_ref = request.config.getoption("--skip-ref", default=False)

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

    config = {
        "n": n,
        "k": k,
        "l": l,
        "group_m_list": group_m_list,
        "m_aligned": 256,
        "mma_tiler_mn": mma_tiler_mn,
        "cluster_shape_mn": cluster_shape_mn,
        "ab_dtype": ab_dtype,
        "c_dtype": c_dtype,
        "d_dtype": d_dtype,
        "b_major": b_major,
        "cd_major": cd_major,
        "acc_dtype": acc_dtype,
        "sf_vec_size": sf_vec_size,
        "sf_dtype": sf_dtype,
        "vector_f32": vector_f32,
        "skip_ref": skip_ref,
        "discrete_col_sfd": discrete_col_sfd,
        "enable_bias": enable_bias,
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
    m_aligned: int = 256,
    permuted_m: Optional[int] = None,
) -> Tuple[int, List[int], torch.Tensor]:
    """Create padded_offsets tensor from group_m_list.

    :param group_m_list: List of M values for each group (will be aligned to m_aligned)
    :param m_aligned: Alignment requirement for group M dimension. MUST equal
                      the grouped GEMM kernel FIX_PAD_SIZE (256)
    :param permuted_m: Optional padded M dimension for CUDA graph support. If provided,
                     padded_offsets will be padded to include this size.
                     The kernel determines valid tiles from padded_offsets[-1].

    :return: Tuple of (valid_m, aligned_group_m_list, padded_offsets_tensor)
    """
    valid_m = 0
    aligned_group_m_list = []
    padded_offsets = []

    for group_m in group_m_list:
        aligned_group_m = ((group_m + m_aligned - 1) // m_aligned) * m_aligned
        valid_m += aligned_group_m
        aligned_group_m_list.append(aligned_group_m)

        # padded_offsets[i] = cumulative sum up to and including expert i
        padded_offsets.append(valid_m)

    # Apply padding if requested (for cuda_graph support)
    if permuted_m is not None:
        if permuted_m < valid_m:
            raise ValueError(f"permuted_m ({permuted_m}) must be >= valid_m ({valid_m}). " f"Cannot pad to a smaller size.")
        # Note: permuted_m padding is handled by the caller creating A/D tensors with larger M
        # padded_offsets[-1] still equals valid_m (not permuted_m)

    # Convert to tensor
    padded_offsets_tensor = torch.tensor(padded_offsets, dtype=torch.int32).cuda()

    return (
        valid_m,
        aligned_group_m_list,
        padded_offsets_tensor,
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
    permuted_m: Optional[int] = None,
    norm_const: float = 0.01,
    b_major: str = "k",
    enable_bias: bool = False,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Allocate input tensors for grouped GEMM SReLU.

    :param permuted_m: Optional padded M dimension for cuda_graph support. If provided,
                     A matrix, D matrix, and scale factor A will be padded to this size.
                     The kernel calculates valid tiles from padded_offsets[-1].

    :return: Dictionary containing all input tensors and metadata
    """

    valid_m, aligned_group_m_list, padded_offsets_tensor = create_mask(group_m_list, m_aligned, permuted_m)

    tensor_m = permuted_m if permuted_m is not None else valid_m

    # Standalone grouped kernels use raw-byte tensors for FP4 payloads with the
    # full logical K still present in the visible tensor shape.
    if ab_dtype == torch.uint8:
        try:
            import cutlass
            import cutlass.torch as cutlass_torch
        except ImportError:
            pytest.skip("CUTLASS is not installed; skipping grouped uint8 raw-FP4 tests.")

        a_ref = cutlass_torch.matrix(1, tensor_m, k, False, cutlass.Float32).cuda()
        b_ref = cutlass_torch.matrix(l, n, k, b_major == "n", cutlass.Float32).cuda()
        _, a_tensor = cutlass_torch.cute_tensor_like(
            a_ref,
            cutlass.Float4E2M1FN,
            is_dynamic_layout=True,
            assumed_align=16,
        )
        _, b_tensor = cutlass_torch.cute_tensor_like(
            b_ref,
            cutlass.Float4E2M1FN,
            is_dynamic_layout=True,
            assumed_align=16,
        )
        a_tensor = a_tensor.view(torch.uint8)
        b_tensor = b_tensor.view(torch.uint8)
    else:
        # Note: b tensor can be n-major for mxfp8 dSrelu; otherwise, a and b tensors are always k-major
        a_ref, a_tensor = create_and_permute_tensor(1, tensor_m, k, False, ab_dtype)
        b_ref, b_tensor = create_and_permute_tensor(l, n, k, b_major == "n", ab_dtype)

    sfa_ref, sfa_tensor = create_scale_factor_tensor(1, tensor_m, k, sf_vec_size, sf_dtype)
    sfb_ref, sfb_tensor = create_scale_factor_tensor(l, n, k, sf_vec_size, sf_dtype)

    alpha_tensor = torch.randint(-2, 2, (l,), dtype=torch.float32, device=device).float()
    beta_tensor = torch.randint(-2, 2, (l,), dtype=torch.float32, device=device).float()  # dSrelu only

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
        "beta_tensor": beta_tensor,
        "prob_tensor": prob_tensor,
        "bias_tensor": None,
        "padded_offsets_tensor": padded_offsets_tensor,
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

    if enable_bias:
        result["bias_tensor"] = torch.empty((l, n), dtype=torch.bfloat16, device=device).uniform_(-2.0, 2.0).transpose(0, 1)

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
    """Allocate output tensors for grouped GEMM SReLU.

    :return: Dictionary containing all output tensors
    """
    n_out = n  # After SReLU

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


def run_grouped_gemm_srelu_ref(
    a_ref: torch.Tensor,
    b_ref: torch.Tensor,
    sfa_ref: torch.Tensor,
    sfb_ref: torch.Tensor,
    alpha_tensor: torch.Tensor,
    prob_tensor: torch.Tensor,
    aligned_group_m_list: List[int],
    valid_m: int,
    bias_tensor: Optional[torch.Tensor] = None,
    generate_amax: bool = False,
    generate_sfd: bool = False,
    norm_const_tensor: Optional[torch.Tensor] = None,
    c_dtype: torch.dtype = torch.bfloat16,
    d_dtype: torch.dtype = torch.float32,
    sf_vec_size: int = 16,
    sf_dtype: torch.dtype = torch.float8_e8m0fnu,
) -> torch.Tensor:
    """Run reference implementation for grouped GEMM SReLU.

    Matches the reference checking in continugous_blockscaled_grouped_gemm_srelu_quant_fusion.py
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
    n_out = n
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

    if bias_tensor is not None:
        start = 0
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            ref[start:end, :, 0] = ref[start:end, :, 0] + bias_tensor[:, i].unsqueeze(0).to(torch.float32)
            start = end

    ref_tensors["c_ref"] = ref.clone()

    # Step 3: Apply squared-ReLU and probability gating elementwise
    ref_after_srelu = torch.relu(ref) ** 2
    ref_after_srelu = ref_after_srelu * prob_tensor.expand(-1, n_out, -1)
    ref_tensors["d_ref"] = ref_after_srelu.clone()

    if generate_amax:
        amax_ref = torch.empty((l, 1), dtype=torch.float32, device=a_ref.device)
        start = 0
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            amax_ref[i, 0] = compute_reference_amax(ref_after_srelu[start:end, :, 0].clone())
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

        n_out_aligned = ceil_div(n_out, 128) * 128
        if n_out_aligned != n_out:
            zeros = torch.zeros(
                ref_after_srelu.shape[0],
                n_out_aligned - n_out,
                ref_after_srelu.shape[2],
                dtype=ref_after_srelu.dtype,
                device=ref_after_srelu.device,
            )
            ref_after_srelu_sf = torch.cat([ref_after_srelu, zeros], dim=1)
        else:
            ref_after_srelu_sf = ref_after_srelu

        # 1. Compute reference SFDRow (m, sfn, l) in fp32
        sfn = ceil_div(n_out_aligned, sf_vec_size)
        # Resahpe ref to (l, m, sfn, sf_vec_size)
        ref_for_sf = ref_after_srelu_sf.permute(2, 0, 1).contiguous()  # (l, m, n)
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

        # convert ref_after_srelu f32 tensor to cute f32 tensor
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
        ref_sfd_row_rcp_expanded = ref_sfd_row_rcp[:valid_m, :, :].unsqueeze(2).expand(valid_m, sfn, sf_vec_size, 1)
        ref_sfd_row_rcp_expanded = ref_sfd_row_rcp_expanded.reshape(valid_m, sfn * sf_vec_size, 1)
        # Trim to exact n dimension if needed
        ref_sfd_row_rcp_expanded = ref_sfd_row_rcp_expanded[:, :n_out, :]

        # Apply scale to reference output: ref = ref * ref_sfd_row_rcp
        ref_after_row_quant = torch.einsum("mnl,mnl->mnl", ref_after_srelu, ref_sfd_row_rcp_expanded)
        ref_tensors["d_ref"] = ref_after_row_quant.cuda().to(d_dtype).to(torch.float32).clone()

        ref_d_col = ref_after_srelu.permute(2, 1, 0).contiguous().permute(1, 2, 0)
        ref_col_sf = ref_after_srelu_sf.permute(2, 1, 0).contiguous().permute(1, 2, 0)
        n_col = ref_d_col.shape[1]
        sfn_col = ceil_div(n_col, sf_vec_size)
        valid_m_col = ref_d_col.shape[0]
        valid_m_col_aligned = ceil_div(valid_m_col, 128) * 128
        ref_for_sf_col = ref_col_sf.permute(2, 0, 1).contiguous()
        ref_for_sf_col = ref_for_sf_col.view(1, valid_m_col_aligned, sfn_col, sf_vec_size)
        ref_for_sf_col, _ = torch.abs(ref_for_sf_col).max(dim=3)
        ref_sfd_col_f32 = ref_for_sf_col * norm_const * get_dtype_rcp_limits(d_dtype)
        ref_sfd_col_f32 = ref_sfd_col_f32.permute(1, 2, 0)

        ref_sfd_col_f8_torch = torch.empty(*(1, valid_m_col_aligned, sfn_col), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
        ref_sfd_col_f8 = from_dlpack(ref_sfd_col_f8_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        ref_sfd_col_f8.element_type = _convert_to_cutlass_data_type(sf_dtype)
        ref_sfd_col_f32_device = ref_sfd_col_f32.cuda()
        ref_sfd_col_f32_tensor = from_dlpack(ref_sfd_col_f32_device, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        cute.testing.convert(ref_sfd_col_f32_tensor, ref_sfd_col_f8)
        cute.testing.convert(ref_sfd_col_f8, ref_sfd_col_f32_tensor)
        ref_sfd_col_f32 = ref_sfd_col_f32_device.cpu()

        ref_sfd_col_f32_cute_torch_tensor_cpu, _ = create_sf_layout_tensor(1, valid_m_col_aligned, n_col, sf_vec_size)
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
            from_dlpack(ref_sfd_col_f32),
            from_dlpack(ref_sfd_col_f32_cute_torch_tensor_cpu),
        )
        ref_sfd_col_f32 = ref_sfd_col_f32.cuda()
        ref_tensors["sfd_col_ref"] = ref_sfd_col_f32_cute_torch_tensor_cpu.clone()

        ref_sfd_col_rcp = norm_const * ref_sfd_col_f32.reciprocal()
        ref_sfd_col_rcp = torch.clamp(ref_sfd_col_rcp, max=3.40282346638528859812e38)
        ref_sfd_col_rcp_expanded = ref_sfd_col_rcp[:valid_m_col, :, :].unsqueeze(2).expand(valid_m_col, sfn_col, sf_vec_size, 1)
        ref_sfd_col_rcp_expanded = ref_sfd_col_rcp_expanded.reshape(valid_m_col, sfn_col * sf_vec_size, 1)
        ref_sfd_col_rcp_expanded = ref_sfd_col_rcp_expanded[:, :n_col, :]

        ref_after_col_quant = torch.einsum("mnl,mnl->mnl", ref_d_col, ref_sfd_col_rcp_expanded)

        ref_col_f8_torch = torch.empty(*(1, valid_m_col, n_col), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
        ref_col_f8 = from_dlpack(ref_col_f8_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        ref_col_f8.element_type = _convert_to_cutlass_data_type(d_dtype)
        ref_col_device = ref_after_col_quant.cuda()
        ref_col_tensor = from_dlpack(ref_col_device, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        cute.testing.convert(ref_col_tensor, ref_col_f8)
        cute.testing.convert(ref_col_f8, ref_col_tensor)

        ref_tensors["d_col_ref"] = ref_col_device.clone().permute(1, 0, 2)

    return ref_tensors


# =============================================================================
# Reference Checking
# =============================================================================


def check_ref_grouped_gemm_srelu(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    cfg: Dict[str, Any],
    atol: float = 1e-1,
    rtol: float = 1e-2,
    skip_ref: bool = False,
) -> None:
    if skip_ref:
        return

    torch.cuda.synchronize()
    ref_tensors = run_grouped_gemm_srelu_ref(
        a_ref=inputs["a_ref"],
        b_ref=inputs["b_ref"],
        sfa_ref=inputs["sfa_ref"],
        sfb_ref=inputs["sfb_ref"],
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        aligned_group_m_list=inputs["aligned_group_m_list"],
        valid_m=inputs["valid_m"],
        bias_tensor=inputs.get("bias_tensor"),
        generate_amax=outputs.get("amax_tensor") is not None,
        generate_sfd=outputs.get("sfd_row_tensor") is not None and outputs.get("sfd_col_tensor") is not None,
        norm_const_tensor=inputs.get("norm_const_tensor"),
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        sf_dtype=cfg["sf_dtype"],
    )

    torch.testing.assert_close(outputs["c_tensor"].float(), ref_tensors["c_ref"].float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(outputs["d_tensor"].float(), ref_tensors["d_ref"].float(), atol=atol, rtol=rtol)

    if "d_col_ref" in ref_tensors:
        torch.testing.assert_close(outputs["d_col_tensor"].float(), ref_tensors["d_col_ref"].float(), atol=atol, rtol=rtol)

    if outputs.get("amax_tensor") is not None and "amax_ref" in ref_tensors:
        torch.testing.assert_close(outputs["amax_tensor"].float(), ref_tensors["amax_ref"].float(), atol=atol, rtol=rtol)

    if outputs.get("sfd_row_tensor") is not None and "sfd_row_ref" in ref_tensors:
        torch.testing.assert_close(
            outputs["sfd_row_tensor"].float(),
            ref_tensors["sfd_row_ref"].to(outputs["sfd_row_tensor"].device).float(),
            atol=atol,
            rtol=rtol,
        )

    if outputs.get("sfd_col_tensor") is not None and "sfd_col_ref" in ref_tensors:
        sfd_col_tensor = outputs["sfd_col_tensor"].float()
        sfd_col_ref = ref_tensors["sfd_col_ref"].to(outputs["sfd_col_tensor"].device).float()
        if cfg.get("discrete_col_sfd", False):
            # Mirror the original standalone discrete-col verification, which
            # remaps packed tiles rather than comparing the whole buffer directly.
            group_n_tile_list = [group // 128 for group in inputs["aligned_group_m_list"]]
            m_tile = sfd_col_ref.shape[2]
            res_real_idx = 0
            cumsum_n = 0
            total_n = sum(group_n_tile_list)

            for n_tile in group_n_tile_list:
                for m_idx in range(m_tile):
                    for n_idx in range(n_tile):
                        res_real_m_idx = res_real_idx // total_n
                        res_real_n_idx = res_real_idx % total_n
                        ref_real_n_idx = n_idx + cumsum_n

                        ref_slice = sfd_col_ref[:, :, m_idx, :, ref_real_n_idx, :]
                        res_slice = sfd_col_tensor[:, :, res_real_m_idx, :, res_real_n_idx, :]
                        torch.testing.assert_close(
                            res_slice,
                            ref_slice,
                            atol=atol,
                            rtol=rtol,
                        )
                        res_real_idx += 1
                cumsum_n += n_tile
        else:
            torch.testing.assert_close(
                sfd_col_tensor,
                sfd_col_ref,
                atol=atol,
                rtol=rtol,
            )
