"""
Utilities and parameterization for Grouped GEMM dSwiGLU backward tests.
Contains test configuration fixtures, tensor creation, and reference implementations.
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
from test_grouped_gemm_swiglu_utils import (
    get_dtype_rcp_limits,
)

# =============================================================================
# Parameterization Marks
# =============================================================================

GROUPED_GEMM_DSWIGLU_PARAM_MARKS_FP8 = [
    pytest.mark.parametrize(
        "ab_dtype",
        [
            torch.float8_e4m3fn,
            # torch.float8_e5m2,
        ],
    ),
    pytest.mark.parametrize(
        "c_dtype",
        [
            torch.float8_e4m3fn,
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
        ],
    ),
    pytest.mark.parametrize("b_major", ["k", "n"]),
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

GROUPED_GEMM_DSWIGLU_PARAM_MARKS_FP4 = [
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
            # torch.float32,
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


def with_grouped_gemm_dswiglu_params_fp4(func):
    """Decorator to apply grouped GEMM dSwiGLU FP4 test parameters."""
    for mark in reversed(GROUPED_GEMM_DSWIGLU_PARAM_MARKS_FP4):
        func = mark(func)
    return func


def with_grouped_gemm_dswiglu_params_fp8(func):
    """Decorator to apply grouped GEMM dSwiGLU FP8 test parameters."""
    for mark in reversed(GROUPED_GEMM_DSWIGLU_PARAM_MARKS_FP8):
        func = mark(func)
    return func


# =============================================================================
# Tensor Allocation
# =============================================================================
def allocate_grouped_gemm_dswiglu_tensors(
    tensor_m: int,
    n: int,
    l: int,
    ab_dtype: torch.dtype,
    c_dtype: torch.dtype,
    d_dtype: torch.dtype,
    cd_major: str,
    sf_dtype: torch.dtype,
    sf_vec_size: int = 16,
    input_tensors: Optional[Dict] = None,
    output_tensors: Optional[Dict] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Allocate backward tensors for grouped GEMM dSwiGLU backward.

    :return: Newly allocated input and output tensors. Modifies input_tensors and output_tensors dictionaries in place if provided.
    """

    # D has same shape as C - contains interleaved ab and dswiglu in 32-column blocks
    _, c_tensor = create_and_permute_tensor(
        1, tensor_m, n * 2, cd_major == "m", c_dtype
    )  # Note: c_tensor is an input tensor rather than an output tensor but is being kept as an output tensor to eventually merge with the forward allocation
    _, d_row_tensor = create_and_permute_tensor(1, tensor_m, n * 2, cd_major == "m", d_dtype)
    _, d_col_tensor = create_and_permute_tensor(1, tensor_m, n * 2, cd_major == "m", d_dtype)
    dprob_tensor = torch.zeros((tensor_m, 1, 1), dtype=torch.float32).cuda()

    _input_tensors = {
        "c_tensor": c_tensor,
    }
    _output_tensors = {
        "d_row_tensor": d_row_tensor,
        "d_col_tensor": d_col_tensor,
        "dprob_tensor": dprob_tensor,
        "sfd_row_tensor": None,
        "sfd_col_tensor": None,
        "amax_tensor": None,
    }

    if d_dtype in [torch.bfloat16, torch.float16]:
        # amax shape for dswiglu is (l, 2, 1) - two amax values per expert
        _output_tensors["amax_tensor"] = torch.full((l, 2, 1), float("-inf"), dtype=torch.float32).cuda()

    if ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and sf_dtype in [
        torch.float8_e8m0fnu,
        torch.float8_e4m3fn,
    ]:  # generate_sfd
        sfd_row_ref, sfd_row_tensor = create_scale_factor_tensor(1, tensor_m, n * 2, sf_vec_size, sf_dtype)
        _output_tensors["sfd_row_tensor"] = sfd_row_tensor
        _output_tensors["sfd_row_ref"] = sfd_row_ref

        sfd_col_ref, sfd_col_tensor = create_scale_factor_tensor(1, n * 2, tensor_m, sf_vec_size, sf_dtype)
        _output_tensors["sfd_col_tensor"] = sfd_col_tensor
        _output_tensors["sfd_col_ref"] = sfd_col_ref

    if input_tensors is not None:
        input_tensors.update(_input_tensors)
        _input_tensors = input_tensors
    if output_tensors is not None:
        output_tensors.update(_output_tensors)
        _output_tensors = output_tensors
    return _input_tensors, _output_tensors


# =============================================================================
# Reference Implementations
# =============================================================================


def compute_reference_row_quant(src, d_dtype, sf_dtype, vec_size, norm_const) -> (torch.Tensor, torch.Tensor):
    """
    Compute reference quantized value on CPU.

    Args:
        src: torch.Tensor, source tensor
        dst_type: Type[cutlass.Numeric], destination type
        vec_size: int, vector size

    Returns:
        torch.Tensor: quantized reference tensor
        torch.Tensor: scale factor tensor
    """

    try:
        from cutlass.cute.runtime import from_dlpack
        import cutlass.cute as cute
        from cudnn.datatypes import _convert_to_cutlass_data_type
    except ImportError:
        pytest.skip("CUTLASS not available for scale factor conversion")

    m = src.shape[0]
    n = src.shape[1]
    l = src.shape[2]

    # 1. Compute reference SFD (m, sfn, l) in fp32
    sfn = ceil_div(n, vec_size)
    sfm = ceil_div(m, 128) * 128
    # Reshape ref to (l, m, sfn, vec_size)
    src_reshaped = src.permute(2, 0, 1).contiguous()  # (l, m, n)
    # l is involved in valid_m
    src_reshaped = src_reshaped.view(l, sfm, sfn, vec_size)
    # Take abs max over vec_size dimension
    src_reshaped, _ = torch.abs(src_reshaped).max(dim=3)  # (l, m, sfn)
    # Multiply by norm_const and rcp_limits
    src_sfd_f32 = src_reshaped * norm_const * get_dtype_rcp_limits(d_dtype)
    # Permute to (m, sfn, l)
    src_sfd_f32 = src_sfd_f32.permute(1, 2, 0)
    # Convert fp32 -> f8 -> fp32 for src_sfd_f32
    src_sfd_f8_torch = torch.empty(*(l, sfm, sfn), dtype=torch.uint8, device=src.device).permute(1, 2, 0)
    src_sfd_f8 = from_dlpack(src_sfd_f8_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    src_sfd_f8.element_type = _convert_to_cutlass_data_type(sf_dtype)
    src_sfd_f32_device = src_sfd_f32.to(src.device)
    ref_sfd_f32_tensor = from_dlpack(src_sfd_f32_device, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    # 2. Convert sfd from fp32 to scale factor
    cute.testing.convert(ref_sfd_f32_tensor, src_sfd_f8)
    cute.testing.convert(src_sfd_f8, ref_sfd_f32_tensor)
    src_sfd_f32 = src_sfd_f32_device.cpu()
    # ref_sfd_f32 for fp32 reference check
    ref_sfd_f32, _ = create_sf_layout_tensor(l, sfm, n, vec_size)
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(src_sfd_f32),
        from_dlpack(ref_sfd_f32),
    )

    # 3. Quantized output with scale factor
    # Compute reciprocal of src_sfd_f32 and multiply by norm_const
    src_sfd_f32_rcp = norm_const * src_sfd_f32.to(src.device).reciprocal()
    # Expand the sfn dimension by repeating each value sf_vec_size times
    # src_sfd_f32_rcp: (m, sfn, l) -> (m, sfn, sf_vec_size, l) -> (m, n, l)
    src_sfd_f32_rcp_expanded = src_sfd_f32_rcp.unsqueeze(2).expand(sfm, sfn, vec_size, l)
    src_sfd_f32_rcp_expanded = src_sfd_f32_rcp_expanded.reshape(sfm, sfn * vec_size, l)
    # Trim to exact n dimension if needed
    src_sfd_f32_rcp_expanded = src_sfd_f32_rcp_expanded[:, :n, :]
    # Apply scale to reference output: ref = ref * src_sfd_f32_rcp
    src_d_f32_torch = torch.einsum("mnl,mnl->mnl", src, src_sfd_f32_rcp_expanded)
    # Convert to d_dtype, then convert back to fp32 for reference check
    src_d_f8_torch = torch.empty(*(l, m, n), dtype=torch.uint8, device=src.device).permute(1, 2, 0)
    src_d_f8 = from_dlpack(src_d_f8_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    src_d_f8.element_type = _convert_to_cutlass_data_type(d_dtype)
    src_d_f32_torch = src_d_f32_torch.to(src.device)
    src_d_f32 = from_dlpack(src_d_f32_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    cute.testing.convert(src_d_f32, src_d_f8)
    cute.testing.convert(src_d_f8, src_d_f32)

    return (ref_sfd_f32, src_d_f32_torch)


def run_grouped_gemm_dswiglu_ref(
    a_ref: torch.Tensor,
    b_ref: torch.Tensor,
    c_ref: torch.Tensor,  # C tensor (intermediate from forward pass) in float32
    sfa_ref: torch.Tensor,
    sfb_ref: torch.Tensor,
    alpha_tensor: torch.Tensor,
    beta_tensor: torch.Tensor,
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
) -> Dict[str, torch.Tensor]:
    """Run reference implementation for grouped GEMM dSwiGLU backward.

    Based on the reference in continugous_blockscaled_grouped_gemm_dswiglu_quant_fusion.py

    The dSwiGLU backward pass computes:
    1. GEMM: ref = alpha^2 * (SFA * A) @ (SFB * B)^T per group
    2. Deinterleave C tensor with 32-column swizzling into gate/input
       (gate in even blocks, input in odd blocks)
    3. Compute sigmoid and swish derivatives
    4. dprob = sum over chunks of (swish * c_input * ref)
    5. d2 = ref * prob * swish
    6. d1 = ref * prob * c_input * sig * (1 + c_gate * (1 - sig))
    7. Interleave d1 and d2 back into swizzled [M, 2N, L] layout

    :param a_ref: A tensor (tensor_m, k, 1) in float32
    :param b_ref: B tensor (n, k, l) in float32
    :param c_ref: C tensor (tensor_m, 2*n_out, 1) from forward pass in float32
    :param sfa_ref: Scale factor A tensor (tensor_m, k, 1) in float32
    :param sfb_ref: Scale factor B tensor (n, k, l) in float32
    :param alpha_tensor: Per-group alpha scaling (l,)
    :param beta_tensor: Per-group beta scaling (l,)
    :param prob_tensor: Per-row probability scaling (tensor_m, 1, 1)
    :param aligned_group_m_list: Aligned M values per group
    :param valid_m: Total valid M dimension
    :param generate_amax: Generate AMAX tensor
    :param generate_sfd: Generate SFD tensor
    :param norm_const_tensor: Normalization constant tensor (1,)
    :param c_dtype: Intermediate C tensor dtype
    :param d_dtype: Output D tensor dtype
    :param sf_vec_size: Scale factor vector size
    :param sf_dtype: Scale factor dtype
    :return: Dictionary of reference tensors
    """
    n, k, l = b_ref.shape
    ref_tensors = {}

    # Step 1: Compute GEMM per group with scale factors
    ref = torch.empty((1, valid_m, n), dtype=torch.float32, device=a_ref.device)
    start = 0
    for i, group_m in enumerate(aligned_group_m_list):
        end = start + group_m
        res_a = torch.einsum(
            "mk,mk->mk",
            a_ref[start:end, :, 0],
            sfa_ref[start:end, :, 0],
        )
        res_b = torch.einsum("nk,nk->nk", b_ref[:, :, i], sfb_ref[:, :, i])
        ref[0, start:end, :] = torch.einsum("mk,nk->mn", res_a * alpha_tensor[i].item(), res_b * alpha_tensor[i].item())
        start = end
    ref = ref.permute((1, 2, 0))  # shape [M, N, 1]

    # Step 2: Handle C tensor with 32-column swizzling
    # C is [M, N, 1] with blocks of 32 alternating gate/input
    c_full = torch.empty((valid_m, n * 2, 1), dtype=torch.float32, device=a_ref.device)
    start = 0
    for i, group_m in enumerate(aligned_group_m_list):
        end = start + group_m
        c_full[start:end, :, 0] = c_ref[start:end, :, 0] * beta_tensor[i].item()
        start = end

    group = 32
    assert n % 2 == 0, "N must be divisible by 2 (contains up and gate)"
    num_blocks_n = n // group

    # Build block indices for 2N and N
    cols_2n = torch.arange(n * 2, dtype=torch.long, device=a_ref.device)
    block_cols_2n = cols_2n.view((n * 2) // group, group)
    dest_idx_glu = block_cols_2n[0::2].reshape(-1)  # even blocks in 2N hold glu
    dest_idx_ab = block_cols_2n[1::2].reshape(-1)  # odd blocks in 2N hold ab

    cols_n = torch.arange(n, dtype=torch.long, device=a_ref.device)
    block_cols_n = cols_n.view(num_blocks_n, group)
    src_idx_n = block_cols_n.reshape(-1)

    # Deinterleave C into input/gate halves by 32-wide blocks
    c_input = c_full.index_select(dim=1, index=dest_idx_ab)  # shape [M, N, L]
    c_gate = c_full.index_select(dim=1, index=dest_idx_glu)  # shape [M, N, L]
    sig = torch.sigmoid(c_gate)
    swish = c_gate * sig

    # Step 3: Compute dprob reference
    ref_dprob = swish * c_input * ref
    chunk_sums = [torch.sum(chunk, dim=1, keepdim=True) for chunk in torch.split(ref_dprob, 32, dim=1)]
    ref_dprob = torch.sum(torch.cat(chunk_sums, dim=1), dim=1, keepdim=True)  # (m, 1, l)
    ref_tensors["dprob_ref"] = ref_dprob

    # Step 4: Compute dSwiGLU formulas
    prob = prob_tensor.expand(-1, n, -1)
    ab = ref * prob * swish
    dswiglu = ref * prob * c_input * sig * (1 + c_gate * (1 - sig))

    # Step 5: Interleave [dswiglu, ab] back into swizzled [M, N, 1] by 32-wide blocks
    ref_d = torch.empty_like(c_full)
    # Place AB blocks at even positions
    ref_d.index_copy_(dim=1, index=dest_idx_ab, source=ab.index_select(dim=1, index=src_idx_n))
    # Place dswiglu blocks at odd positions
    ref_d.index_copy_(
        dim=1,
        index=dest_idx_glu,
        source=dswiglu.index_select(dim=1, index=src_idx_n),
    )

    ref_tensors["d_ref"] = ref_d.clone()

    # Step 6: Generate amax for FP4/BF16 output
    if generate_amax:
        ref_amax = torch.empty((l, 2, 1), dtype=torch.float32, device=a_ref.device)
        start = 0
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            ref_amax[i, 0] = torch.tensor(compute_reference_amax(dswiglu[start:end, :, 0].clone()))
            ref_amax[i, 1] = torch.tensor(compute_reference_amax(ab[start:end, :, 0].clone()))
            start = end
        ref_tensors["amax_ref"] = ref_amax

    # Step 7: Generate SFD for FP8 output
    if generate_sfd:
        norm_const = norm_const_tensor[0].item()
        sfd_row_ref_f32, d_ref_f32 = compute_reference_row_quant(ref_d, d_dtype, sf_dtype, sf_vec_size, norm_const)
        ref_tensors["sfd_row_ref"] = sfd_row_ref_f32.clone()
        ref_tensors["d_ref"] = d_ref_f32.clone()

        ref_d_col = ref_d.permute(2, 1, 0).contiguous().permute(1, 2, 0)
        sfd_col_ref_f32, d_col_ref_f32 = compute_reference_row_quant(ref_d_col, d_dtype, sf_dtype, sf_vec_size, norm_const)
        ref_tensors["sfd_col_ref"] = sfd_col_ref_f32.clone()
        ref_tensors["d_col_ref"] = d_col_ref_f32.clone()

    return ref_tensors


# =============================================================================
# Reference Checking
# =============================================================================


def check_ref_grouped_gemm_dswiglu(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    cfg: Dict[str, Any],
    atol: float = 1e-1,
    rtol: float = 1e-2,
    skip_ref: bool = False,
) -> None:
    """Check grouped GEMM dSwiGLU result against reference.

    :param inputs: Dictionary of input tensors
    :param outputs: Dictionary of output tensors
    :param cfg: Configuration dictionary
    :param atol: Absolute tolerance
    :param rtol: Relative tolerance
    :param skip_ref: Skip reference check if True
    """
    if skip_ref:
        print("Skipping reference check for config: ", cfg)
        return

    # Run reference
    ref_tensors = run_grouped_gemm_dswiglu_ref(
        a_ref=inputs["a_ref"].to(torch.float32),
        b_ref=inputs["b_ref"].to(torch.float32),
        c_ref=inputs["c_tensor"].to(torch.float32),
        sfa_ref=inputs["sfa_ref"].to(torch.float32),
        sfb_ref=inputs["sfb_ref"].to(torch.float32),
        alpha_tensor=inputs["alpha_tensor"],
        beta_tensor=inputs["beta_tensor"],
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

    # Check dprob output
    if ref_tensors.get("dprob_ref") is not None and outputs.get("dprob_tensor") is not None:
        dprob_gpu = outputs["dprob_tensor"][: inputs["valid_m"]]
        dprob_ref = ref_tensors["dprob_ref"]
        torch.testing.assert_close(
            dprob_gpu.cpu().float(),
            dprob_ref.cpu().float(),
            atol=atol,
            rtol=rtol,
        )

    if cfg["d_dtype"] in [torch.float32, torch.float16, torch.bfloat16]:
        # Check amax
        if ref_tensors.get("amax_ref") is not None and outputs.get("amax_tensor") is not None:
            amax_gpu = outputs["amax_tensor"]
            amax_ref = ref_tensors["amax_ref"]
            torch.testing.assert_close(
                amax_gpu.cpu(),
                amax_ref.cpu(),
                atol=atol,
                rtol=rtol,
            )

        # Check D tensor
        if ref_tensors.get("d_ref") is not None:
            d_gpu = outputs["d_row_tensor"][: inputs["valid_m"]]
            d_ref = ref_tensors["d_ref"]
            torch.testing.assert_close(
                d_gpu.cpu().float(),
                d_ref.cpu().to(cfg["d_dtype"]).to(torch.float32),
                atol=atol,
                rtol=rtol,
            )

    elif cfg["d_dtype"] in [torch.float8_e4m3fn, torch.float8_e5m2]:
        if ref_tensors.get("sfd_row_ref") is not None:  # generate_sfd
            # Check sfd_row
            sfd_row_gpu = outputs["sfd_row_tensor"]
            sfd_row_ref = ref_tensors["sfd_row_ref"]
            torch.testing.assert_close(
                sfd_row_gpu.cpu().float(),
                sfd_row_ref.cpu().to(torch.float32),
                atol=atol,
                rtol=rtol,
            )

            # Check D (row)
            d_gpu = outputs["d_row_tensor"]
            d_ref = ref_tensors["d_ref"]
            torch.testing.assert_close(
                d_gpu.cpu().float(),
                d_ref.cpu().to(cfg["d_dtype"]).to(torch.float32),
                atol=atol,
                rtol=rtol,
            )

            # Check sfd_col
            if cfg["discrete_col_sfd"]:
                group_n_tile_list = [group // 128 for group in inputs["aligned_group_m_list"]]
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
                sfd_col_gpu = outputs["sfd_col_tensor"]
                sfd_col_ref = ref_tensors["sfd_col_ref"]
                torch.testing.assert_close(
                    sfd_col_gpu.cpu().to(torch.float32),
                    sfd_col_ref.cpu().to(torch.float32),
                    atol=atol,
                    rtol=rtol,
                )

            # Check d_col
            if ref_tensors.get("d_col_ref") is not None:
                d_col_gpu = outputs["d_col_tensor"]
                d_col_ref = ref_tensors["d_col_ref"]
                torch.testing.assert_close(
                    d_col_gpu.permute(1, 0, 2).cpu().float(),
                    d_col_ref.to(cfg["d_dtype"]).to(torch.float32).cpu(),
                    atol=atol,
                    rtol=rtol,
                )
        else:
            # Non-generate_sfd path
            if ref_tensors.get("d_ref") is not None:
                d_gpu = outputs["d_row_tensor"][: inputs["valid_m"]]
                d_ref = ref_tensors["d_ref"][: inputs["valid_m"]]
                torch.testing.assert_close(
                    d_gpu.cpu().float(),
                    d_ref.cpu().to(cfg["d_dtype"]).to(torch.float32),
                    atol=atol,
                    rtol=rtol,
                )
    else:
        raise NotImplementedError(f"Unsupported dtype: {cfg['d_dtype']}")
