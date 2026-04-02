"""
Utilities and parameterization for Grouped GEMM Quant tests.
Contains test configuration fixtures, tensor creation, and reference implementations.

Reference: continugous_blockscaled_grouped_gemm_quant.py
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

GROUPED_GEMM_QUANT_PARAM_MARKS_FP8 = [
    pytest.mark.parametrize(
        "ab_dtype",
        [
            torch.float8_e4m3fn,
        ],
    ),
    pytest.mark.parametrize(
        "c_dtype",
        [
            torch.bfloat16,
        ],
    ),
    pytest.mark.parametrize(
        "d_dtype",
        [
            torch.float8_e4m3fn,
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

GROUPED_GEMM_QUANT_PARAM_MARKS_FP4 = [
    pytest.mark.parametrize(
        "ab_dtype",
        [
            torch.float4_e2m1fn_x2,
        ],
    ),
    pytest.mark.parametrize(
        "c_dtype",
        [
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
            (128, 256),
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


def with_grouped_gemm_quant_params_fp4(func):
    """Decorator to apply grouped GEMM Quant FP4 test parameters."""
    for mark in reversed(GROUPED_GEMM_QUANT_PARAM_MARKS_FP4):
        func = mark(func)
    return func


def with_grouped_gemm_quant_params_fp8(func):
    """Decorator to apply grouped GEMM Quant FP8 test parameters."""
    for mark in reversed(GROUPED_GEMM_QUANT_PARAM_MARKS_FP8):
        func = mark(func)
    return func


# =============================================================================
# Configuration Initialization
# =============================================================================


def grouped_gemm_quant_init(
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
    """Initialize configuration for Grouped GEMM Quant tests.

    :return: Configuration dictionary
    """
    major, minor = torch.cuda.get_device_capability()
    compute_capability = major * 10 + minor
    if compute_capability < 100:
        pytest.skip(f"Environment not supported: requires compute capability >= 10, found {major}")

    nkl_str = request.config.getoption("--grouped-gemm-nkl", default=None)
    group_m_str = request.config.getoption("--grouped-gemm-group-m", default=None)
    skip_ref = request.config.getoption("--skip-ref", default=False)

    if nkl_str is not None:
        n, k, num_groups = [int(x.strip()) for x in nkl_str.split(",")]
    else:
        n, k, num_groups = 512, 512, 4

    if group_m_str is not None:
        group_m_list = [int(x.strip()) for x in group_m_str.split(",")]
    else:
        group_m_list = [256] * num_groups

    config = {
        "n": n,
        "k": k,
        "l": num_groups,
        "group_m_list": group_m_list,
        "m_aligned": 256,
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


# =============================================================================
# Tensor Allocation
# =============================================================================


def allocate_grouped_gemm_quant_output_tensors(
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
    """Allocate output tensors for grouped GEMM Quant.

    Unlike SwiGLU, D uses the full N dimension (no halving).

    :return: Dictionary containing all output tensors
    """
    # C tensor is internal placeholder (generate_c=False), but needed for kernel compilation
    _, c_tensor = create_and_permute_tensor(1, tensor_m, n, cd_major == "m", c_dtype)
    _, d_tensor = create_and_permute_tensor(1, tensor_m, n, cd_major == "m", d_dtype)
    _, d_col_tensor = create_and_permute_tensor(1, tensor_m, n, cd_major == "m", d_dtype)

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
    ]:
        # SFD uses full N (not n/2 like SwiGLU)
        sfd_row_ref, sfd_row_tensor = create_scale_factor_tensor(1, tensor_m, n, sf_vec_size, sf_dtype)
        result["sfd_row_tensor"] = sfd_row_tensor
        result["sfd_row_ref"] = sfd_row_ref

        sfd_col_ref, sfd_col_tensor = create_scale_factor_tensor(1, n, tensor_m, sf_vec_size, sf_dtype)
        result["sfd_col_tensor"] = sfd_col_tensor
        result["sfd_col_ref"] = sfd_col_ref

    return result


# =============================================================================
# Reference Implementation
# =============================================================================


def run_grouped_gemm_quant_ref(
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
    c_dtype: torch.dtype = torch.bfloat16,  # noqa: ARG001 kept for API consistency
    d_dtype: torch.dtype = torch.float32,
    sf_vec_size: int = 16,
    sf_dtype: torch.dtype = torch.float8_e8m0fnu,
    bias_ref: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Run reference implementation for grouped GEMM Quant.

    Plain GEMM: D = quant(alpha * A * SFA * B * SFB)
    No SwiGLU activation, output uses full N dimension.
    If ``bias_ref`` is set (shape ``(n, l)``), the pre-quant epilogue matches the
    fused kernel: ``alpha * GEMM + bias * prob`` per row (no extra ``prob`` multiply on GEMM).

    :return: Dictionary of reference tensors
    """
    n, k, l = b_ref.shape
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

    # Step 3: prob gating — without bias: multiply full result by prob. With bias: kernel
    # fuses (acc * alpha + bias * prob), so add bias * prob per row/expert and do not
    # multiply the GEMM part by prob again.
    if bias_ref is None:
        ref = ref * prob_tensor.expand(-1, n, -1)
    else:
        bias_f32 = bias_ref.to(torch.float32)
        start = 0
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            prob_slice = prob_tensor[start:end, 0, 0].to(torch.float32).unsqueeze(1)
            b_slice = bias_f32[:, i].unsqueeze(0)
            ref[start:end, :, 0] = ref[start:end, :, 0] + prob_slice * b_slice
            start = end
    ref_tensors["d_ref"] = ref.clone()

    if generate_amax:
        amax_ref = torch.empty((l,), dtype=torch.float32, device=a_ref.device)
        start = 0
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            amax_ref[i] = compute_reference_amax(ref[start:end, :, 0].clone())
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

        # Row SFD: compute per-row scale factors
        n_aligned = ceil_div(n, 128) * 128
        valid_m_aligned = ceil_div(valid_m, 128) * 128
        if n_aligned != n:
            zeros = torch.zeros(
                ref.shape[0],
                n_aligned - n,
                ref.shape[2],
                dtype=ref.dtype,
                device=ref.device,
            )
            ref_sf = torch.cat([ref, zeros], dim=1)
        else:
            ref_sf = ref
        sfn = ceil_div(n_aligned, sf_vec_size)
        ref_for_sf = ref_sf.permute(2, 0, 1).contiguous()  # (l, m, n)
        ref_for_sf = ref_for_sf.view(1, valid_m_aligned, sfn, sf_vec_size)
        ref_for_sf, _ = torch.abs(ref_for_sf).max(dim=3)  # (l, m, sfn)
        ref_sfd_row_f32 = ref_for_sf * norm_const * get_dtype_rcp_limits(d_dtype)
        ref_sfd_row_f32 = ref_sfd_row_f32.permute(1, 2, 0)

        # Convert fp32 -> f8 -> fp32 roundtrip
        ref_sfd_row_f8_torch = torch.empty(*(1, valid_m_aligned, sfn), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
        ref_sfd_row_f8 = from_dlpack(ref_sfd_row_f8_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        ref_sfd_row_f8.element_type = _convert_to_cutlass_data_type(sf_dtype)
        ref_sfd_row_f32_device = ref_sfd_row_f32.cuda()
        ref_sfd_row_f32_tensor = from_dlpack(ref_sfd_row_f32_device, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        cute.testing.convert(ref_sfd_row_f32_tensor, ref_sfd_row_f8)
        cute.testing.convert(ref_sfd_row_f8, ref_sfd_row_f32_tensor)
        ref_sfd_row_f32 = ref_sfd_row_f32_device.cpu()

        ref_sfd_row_f32_cute_torch_tensor_cpu, _ = create_sf_layout_tensor(1, valid_m_aligned, n, sf_vec_size)
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
            from_dlpack(ref_sfd_row_f32),
            from_dlpack(ref_sfd_row_f32_cute_torch_tensor_cpu),
        )
        ref_sfd_row_f32 = ref_sfd_row_f32.cuda()
        ref_tensors["sfd_row_ref"] = ref_sfd_row_f32_cute_torch_tensor_cpu.clone()

        # Quantized output with row scale factor
        ref_sfd_row_rcp = norm_const * ref_sfd_row_f32.reciprocal()
        ref_sfd_row_rcp = torch.clamp(ref_sfd_row_rcp, max=3.40282346638528859812e38)
        ref_sfd_row_rcp_expanded = ref_sfd_row_rcp[:valid_m, :, :].unsqueeze(2).expand(valid_m, sfn, sf_vec_size, 1)
        ref_sfd_row_rcp_expanded = ref_sfd_row_rcp_expanded.reshape(valid_m, sfn * sf_vec_size, 1)
        ref_sfd_row_rcp_expanded = ref_sfd_row_rcp_expanded[:, :n, :]

        ref_after_row_quant = torch.einsum("mnl,mnl->mnl", ref, ref_sfd_row_rcp_expanded)
        ref_tensors["d_ref"] = ref_after_row_quant.clone()

        # Col SFD
        ref_col = ref.permute(2, 1, 0).contiguous().permute(1, 2, 0)
        ref_col_sf = ref_sf.permute(2, 1, 0).contiguous().permute(1, 2, 0)
        n_col = ref_col.shape[1]
        sfn_col = ceil_div(n_col, sf_vec_size)
        valid_m_col = ref_col.shape[0]
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

        # Quantized col output
        ref_sfd_col_rcp = norm_const * ref_sfd_col_f32.reciprocal()
        ref_sfd_col_rcp = torch.clamp(ref_sfd_col_rcp, max=3.40282346638528859812e38)
        ref_sfd_col_rcp_expanded = ref_sfd_col_rcp[:valid_m_col, :, :].unsqueeze(2).expand(valid_m_col, sfn_col, sf_vec_size, 1)
        ref_sfd_col_rcp_expanded = ref_sfd_col_rcp_expanded.reshape(valid_m_col, sfn_col * sf_vec_size, 1)
        ref_sfd_col_rcp_expanded = ref_sfd_col_rcp_expanded[:, :n_col, :]

        ref_after_col_quant = torch.einsum("mnl,mnl->mnl", ref_col, ref_sfd_col_rcp_expanded)

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


def check_ref_grouped_gemm_quant(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    cfg: Dict[str, Any],
    atol: float = 1e-1,
    rtol: float = 1e-2,
    skip_ref: bool = False,
) -> None:
    """Check grouped GEMM Quant result against reference.

    :param inputs: Dictionary of input tensors
    :param outputs: Dictionary of output tensors
    :param cfg: Configuration dictionary
    :param atol: Absolute tolerance
    :param rtol: Relative tolerance
    :param skip_ref: Skip reference check if True
    """
    if skip_ref:
        print("Skipping reference check")
        return

    ref_tensors = run_grouped_gemm_quant_ref(
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
        bias_ref=inputs.get("bias_ref"),
    )

    torch.cuda.synchronize()

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
        if ref_tensors.get("sfd_row_ref") is not None:
            sfd_row_gpu = outputs["sfd_row_tensor"]
            sfd_row_ref = ref_tensors["sfd_row_ref"]
            torch.testing.assert_close(
                sfd_row_gpu.cpu().float(),
                sfd_row_ref.cpu().to(torch.float32),
                atol=atol,
                rtol=rtol,
            )

            d_gpu = outputs["d_tensor"]
            d_ref = ref_tensors["d_ref"]
            torch.testing.assert_close(
                d_gpu.cpu().float(),
                d_ref.to(cfg["d_dtype"]).to(torch.float32).cpu(),
                atol=atol,
                rtol=rtol,
            )

            if cfg["discrete_col_sfd"]:
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
                            res_real_n_idx = res_real_idx % total_n

                            ref_real_n_idx = n_idx + cumsum_n
                            ref_slice = sfd_col_ref_f32[:, :, m_idx, :, ref_real_n_idx, :]
                            res_slice = sfd_col_torch_gpu_f8[:, :, res_real_idx // total_n, :, res_real_n_idx, :]
                            torch.testing.assert_close(
                                ref_slice,
                                res_slice,
                                atol=atol,
                                rtol=rtol,
                            )
                            res_real_idx += 1
                    cumsum_n += n_tile
            else:
                sfd_col_gpu = outputs["sfd_col_tensor"]
                sfd_col_ref = ref_tensors["sfd_col_ref"]
                torch.testing.assert_close(
                    sfd_col_gpu.cpu().float(),
                    sfd_col_ref.cpu().to(torch.float32),
                    atol=atol,
                    rtol=rtol,
                )

            d_col_gpu = outputs["d_col_tensor"]
            d_col_ref = ref_tensors["d_col_ref"]
            torch.testing.assert_close(
                d_col_gpu.cpu().float(),
                d_col_ref.to(cfg["d_dtype"]).to(torch.float32).cpu(),
                atol=atol,
                rtol=rtol,
            )
        else:
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
