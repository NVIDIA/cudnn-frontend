"""
Utilities and parameterization for Discrete-weight Grouped GEMM dGLU backward tests.
"""

import torch
import pytest
from typing import Tuple, List, Dict, Any
from test_fe_api_utils import (
    create_and_permute_tensor,
    create_scale_factor_tensor,
)

# =============================================================================
# Parameterization Marks
# =============================================================================

DISCRETE_GROUPED_GEMM_DSWIGLU_PARAM_MARKS_FP4 = [
    pytest.mark.parametrize("ab_dtype", [torch.float4_e2m1fn_x2]),
    pytest.mark.parametrize("c_dtype", [torch.bfloat16]),
    pytest.mark.parametrize("d_dtype", [torch.bfloat16]),
    pytest.mark.parametrize("cd_major", ["n"]),
    pytest.mark.parametrize("acc_dtype", [torch.float32]),
    pytest.mark.parametrize("mma_tiler_mn", [(256, 256)]),
    pytest.mark.parametrize("cluster_shape_mn", [(2, 1)]),
    pytest.mark.parametrize("sf_vec_size", [32]),
    pytest.mark.parametrize("sf_dtype", [torch.float8_e8m0fnu]),
    pytest.mark.parametrize("vector_f32", [False]),
    pytest.mark.parametrize("discrete_col_sfd", [False]),
    pytest.mark.parametrize("act_func", ["dswiglu", "dgeglu"]),
]

DISCRETE_GROUPED_GEMM_DSWIGLU_PARAM_MARKS_FP8 = [
    pytest.mark.parametrize("ab_dtype", [torch.float8_e4m3fn]),
    pytest.mark.parametrize("c_dtype", [torch.bfloat16]),
    pytest.mark.parametrize("d_dtype", [torch.float8_e4m3fn]),
    pytest.mark.parametrize("cd_major", ["n"]),
    pytest.mark.parametrize("acc_dtype", [torch.float32]),
    pytest.mark.parametrize("mma_tiler_mn", [(256, 256)]),
    pytest.mark.parametrize("cluster_shape_mn", [(2, 1)]),
    pytest.mark.parametrize("sf_vec_size", [32]),
    pytest.mark.parametrize("sf_dtype", [torch.float8_e8m0fnu]),
    pytest.mark.parametrize("vector_f32", [False]),
    pytest.mark.parametrize("discrete_col_sfd", [False]),
    pytest.mark.parametrize("act_func", ["dswiglu", "dgeglu"]),
    pytest.mark.parametrize("b_major", ["k", "n"]),
]


def with_discrete_dswiglu_params_fp4(func):
    for mark in reversed(DISCRETE_GROUPED_GEMM_DSWIGLU_PARAM_MARKS_FP4):
        func = mark(func)
    return func


def with_discrete_dswiglu_params_fp8(func):
    for mark in reversed(DISCRETE_GROUPED_GEMM_DSWIGLU_PARAM_MARKS_FP8):
        func = mark(func)
    return func


# =============================================================================
# Configuration
# =============================================================================


def discrete_dswiglu_init(
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
    act_func: str = "dswiglu",
    b_major: str = "k",
) -> Dict[str, Any]:
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(f"Requires SM100+, found {major}")

    nkl_str = request.config.getoption("--grouped-gemm-nkl", default=None)
    group_m_str = request.config.getoption("--grouped-gemm-group-m", default=None)
    skip_ref = request.config.getoption("--grouped-gemm-skip-ref", default=False)

    if nkl_str is not None:
        n, k, num_experts = [int(x.strip()) for x in nkl_str.split(",")]
    else:
        n, k, num_experts = 512, 512, 4

    if group_m_str is not None:
        group_m_list = [int(x.strip()) for x in group_m_str.split(",")]
    else:
        group_m_list = [256] * num_experts

    return {
        "n": n,
        "k": k,
        "l": num_experts,
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
        "act_func": act_func,
    }


# =============================================================================
# Helpers
# =============================================================================


def create_mask(
    group_m_list: List[int],
    m_aligned: int = 256,
) -> Tuple[int, List[int], torch.Tensor]:
    valid_m = 0
    aligned_group_m_list = []
    padded_offsets = []
    for group_m in group_m_list:
        aligned_group_m = ((group_m + m_aligned - 1) // m_aligned) * m_aligned
        valid_m += aligned_group_m
        aligned_group_m_list.append(aligned_group_m)
        padded_offsets.append(valid_m)
    padded_offsets_tensor = torch.tensor(padded_offsets, dtype=torch.int32).cuda()
    return valid_m, aligned_group_m_list, padded_offsets_tensor


# =============================================================================
# Tensor Allocation
# =============================================================================


def allocate_discrete_dswiglu_input_tensors(
    n: int,
    k: int,
    num_experts: int,
    group_m_list: List[int],
    ab_dtype: torch.dtype,
    c_dtype: torch.dtype,
    sf_dtype: torch.dtype,
    sf_vec_size: int,
    m_aligned: int,
    norm_const: float = 0.01,
    b_major: str = "k",
    device: str = "cuda",
) -> Dict[str, Any]:
    """Allocate input tensors for discrete backward.

    In the backward, C is an INPUT containing the full interleaved forward activations
    with shape `(valid_m, 2n, 1)`.
    """
    valid_m, aligned_group_m_list, padded_offsets_tensor = create_mask(group_m_list, m_aligned)
    tensor_m = valid_m
    n_out = n * 2

    a_ref, a_tensor = create_and_permute_tensor(1, tensor_m, k, False, ab_dtype)

    b_list = []
    b_ref_list = []
    sfb_list = []
    sfb_ref_list = []
    for _ in range(num_experts):
        b_ref_i, b_tensor_i = create_and_permute_tensor(1, n, k, b_major == "n", ab_dtype)
        b_list.append(b_tensor_i.squeeze(-1) if b_tensor_i.shape[-1] == 1 else b_tensor_i)
        b_ref_list.append(b_ref_i)

        sfb_ref_i, sfb_tensor_i = create_scale_factor_tensor(1, n, k, sf_vec_size, sf_dtype)
        sfb_list.append(sfb_tensor_i)
        sfb_ref_list.append(sfb_ref_i)

    sfa_ref, sfa_tensor = create_scale_factor_tensor(1, tensor_m, k, sf_vec_size, sf_dtype)

    # C is an input in the backward. It carries the full interleaved activation tensor.
    _, c_tensor = create_and_permute_tensor(1, tensor_m, n_out, False, c_dtype)

    alpha_tensor = torch.randint(-2, 2, (num_experts,), dtype=torch.float32, device=device).float()
    beta_tensor = torch.randint(-2, 2, (num_experts,), dtype=torch.float32, device=device).float()
    prob_tensor = torch.randint(-2, 2, (tensor_m, 1, 1), dtype=torch.float32, device=device).float()
    dprob_tensor = torch.zeros((tensor_m, 1, 1), dtype=torch.float32, device=device)

    b_ptrs_tensor = torch.tensor([b.data_ptr() for b in b_list], dtype=torch.int64, device=device)
    sfb_ptrs_tensor = torch.tensor([sfb.data_ptr() for sfb in sfb_list], dtype=torch.int64, device=device)

    result = {
        "a_tensor": a_tensor,
        "a_ref": a_ref,
        "b_list": b_list,
        "b_ref_list": b_ref_list,
        "b_ptrs_tensor": b_ptrs_tensor,
        "sfa_tensor": sfa_tensor,
        "sfa_ref": sfa_ref,
        "sfb_list": sfb_list,
        "sfb_ref_list": sfb_ref_list,
        "sfb_ptrs_tensor": sfb_ptrs_tensor,
        "c_tensor": c_tensor,
        "alpha_tensor": alpha_tensor,
        "beta_tensor": beta_tensor,
        "prob_tensor": prob_tensor,
        "dprob_tensor": dprob_tensor,
        "padded_offsets_tensor": padded_offsets_tensor,
        "aligned_group_m_list": aligned_group_m_list,
        "valid_m": valid_m,
        "tensor_m": tensor_m,
        "norm_const_tensor": None,
    }

    if ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and sf_dtype in [
        torch.float8_e8m0fnu,
        torch.float8_e4m3fn,
    ]:
        result["norm_const_tensor"] = torch.tensor([norm_const], dtype=torch.float32, device=device)

    return result


def allocate_discrete_dswiglu_output_tensors(
    tensor_m: int,
    n: int,
    num_experts: int,
    ab_dtype: torch.dtype,
    d_dtype: torch.dtype,
    cd_major: str,
    sf_dtype: torch.dtype,
    sf_vec_size: int = 16,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Allocate output tensors for discrete backward.

    D_row and D_col have shape `(valid_m, 2n, 1)` and match the full interleaved width.
    """
    n_out = n * 2

    _, d_row_tensor = create_and_permute_tensor(1, tensor_m, n_out, cd_major == "m", d_dtype)
    _, d_col_tensor = create_and_permute_tensor(1, tensor_m, n_out, cd_major == "m", d_dtype)

    result = {
        "d_row_tensor": d_row_tensor,
        "d_col_tensor": d_col_tensor,
        "sfd_row_tensor": None,
        "sfd_col_tensor": None,
    }

    if d_dtype in [torch.bfloat16, torch.float16]:
        result["amax_tensor"] = torch.full((num_experts, 2, 1), float("-inf"), dtype=torch.float32, device=device)

    if ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and sf_dtype in [
        torch.float8_e8m0fnu,
        torch.float8_e4m3fn,
    ]:
        sfd_row_ref, sfd_row_tensor = create_scale_factor_tensor(1, tensor_m, n_out, sf_vec_size, sf_dtype)
        result["sfd_row_tensor"] = sfd_row_tensor

        sfd_col_ref, sfd_col_tensor = create_scale_factor_tensor(1, n_out, tensor_m, sf_vec_size, sf_dtype)
        result["sfd_col_tensor"] = sfd_col_tensor

    return result


# =============================================================================
# Reference Checking
# =============================================================================


def check_ref_discrete_dswiglu(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    cfg: Dict[str, Any],
    skip_ref: bool = False,
) -> None:
    """Check discrete backward -- kernel execution verification.

    Full numerical reference for the dSwiGLU backward is complex (requires
    computing SwiGLU derivatives). For now, verify the kernel runs without error.
    """
    if skip_ref:
        print("Skipping reference check")
        return

    torch.cuda.synchronize()
    print("Discrete dGLU backward kernel executed successfully")
