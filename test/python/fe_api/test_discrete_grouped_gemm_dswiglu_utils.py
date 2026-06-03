"""
Utilities and parameterization for Discrete-weight Grouped GEMM dGLU backward tests.
"""

import torch
import pytest
from typing import Tuple, List, Dict, Any, Optional
from test_fe_api_utils import (
    compute_reference_amax,
    create_and_permute_tensor,
    create_scale_factor_tensor,
)
from fe_api.test_grouped_gemm_dswiglu_utils import compute_reference_row_quant

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
    generate_dbias: bool = False,
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
        "dbias_tensor": None,
        "sfd_row_tensor": None,
        "sfd_col_tensor": None,
    }

    if d_dtype in [torch.bfloat16, torch.float16]:
        result["amax_tensor"] = torch.full((num_experts, 2, 1), float("-inf"), dtype=torch.float32, device=device)

    if generate_dbias:
        result["dbias_tensor"] = torch.zeros((num_experts, n_out, 1), dtype=torch.bfloat16, device=device)

    if ab_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and sf_dtype in [
        torch.float8_e8m0fnu,
        torch.float8_e4m3fn,
    ]:
        sfd_row_ref, sfd_row_tensor = create_scale_factor_tensor(1, tensor_m, n_out, sf_vec_size, sf_dtype)
        result["sfd_row_tensor"] = sfd_row_tensor

        sfd_col_ref, sfd_col_tensor = create_scale_factor_tensor(1, n_out, tensor_m, sf_vec_size, sf_dtype)
        result["sfd_col_tensor"] = sfd_col_tensor

    return result


def run_discrete_dswiglu_ref(
    a_ref: torch.Tensor,
    b_ref_list: List[torch.Tensor],
    c_ref: torch.Tensor,
    sfa_ref: torch.Tensor,
    sfb_ref_list: List[torch.Tensor],
    alpha_tensor: torch.Tensor,
    beta_tensor: torch.Tensor,
    prob_tensor: torch.Tensor,
    aligned_group_m_list: List[int],
    valid_m: int,
    generate_dbias: bool = False,
    generate_amax: bool = False,
    generate_sfd: bool = False,
    norm_const_tensor: Optional[torch.Tensor] = None,
    d_dtype: torch.dtype = torch.float32,
    sf_vec_size: int = 16,
    sf_dtype: torch.dtype = torch.float8_e8m0fnu,
) -> Dict[str, torch.Tensor]:
    num_experts = len(b_ref_list)
    n = b_ref_list[0].shape[0]
    n_out = n * 2
    ref_tensors = {}

    ref = torch.empty((1, valid_m, n), dtype=torch.float32, device=a_ref.device)
    start = 0
    for i, group_m in enumerate(aligned_group_m_list):
        end = start + group_m
        res_a = torch.einsum("mk,mk->mk", a_ref[start:end, :, 0], sfa_ref[start:end, :, 0])
        res_b = torch.einsum("nk,nk->nk", b_ref_list[i][:, :, 0], sfb_ref_list[i][:, :, 0])
        ref[0, start:end, :] = torch.einsum("mk,nk->mn", res_a * alpha_tensor[i].item(), res_b * alpha_tensor[i].item())
        start = end
    ref = ref.permute((1, 2, 0))

    c_full = torch.empty((valid_m, n_out, 1), dtype=torch.float32, device=a_ref.device)
    start = 0
    for i, group_m in enumerate(aligned_group_m_list):
        end = start + group_m
        c_full[start:end, :, 0] = c_ref[start:end, :, 0] * beta_tensor[i].item()
        start = end

    group = 32
    cols_2n = torch.arange(n_out, dtype=torch.long, device=a_ref.device)
    block_cols_2n = cols_2n.view(n_out // group, group)
    dest_idx_glu = block_cols_2n[0::2].reshape(-1)
    dest_idx_ab = block_cols_2n[1::2].reshape(-1)

    cols_n = torch.arange(n, dtype=torch.long, device=a_ref.device)
    block_cols_n = cols_n.view(n // group, group)
    src_idx_n = block_cols_n.reshape(-1)

    c_input = c_full.index_select(dim=1, index=dest_idx_ab)
    c_gate = c_full.index_select(dim=1, index=dest_idx_glu)
    sig = torch.sigmoid(c_gate)
    swish = c_gate * sig

    ref_dprob = swish * c_input * ref
    chunk_sums = [torch.sum(chunk, dim=1, keepdim=True) for chunk in torch.split(ref_dprob, 32, dim=1)]
    ref_tensors["dprob_ref"] = torch.sum(torch.cat(chunk_sums, dim=1), dim=1, keepdim=True)

    prob = prob_tensor.expand(-1, n, -1)
    ab = ref * prob * swish
    dswiglu = ref * prob * c_input * sig * (1 + c_gate * (1 - sig))

    ref_d = torch.empty_like(c_full)
    ref_d.index_copy_(dim=1, index=dest_idx_ab, source=ab.index_select(dim=1, index=src_idx_n))
    ref_d.index_copy_(dim=1, index=dest_idx_glu, source=dswiglu.index_select(dim=1, index=src_idx_n))
    ref_tensors["d_ref"] = ref_d.clone()

    if generate_dbias:
        ref_dbias = torch.zeros((num_experts, n_out, 1), dtype=torch.bfloat16, device=a_ref.device)
        start = 0
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            ref_dbias[i, :, 0] = ref_d[start:end, :, 0].sum(dim=0).to(torch.bfloat16)
            start = end
        ref_tensors["dbias_ref"] = ref_dbias

    if generate_amax:
        ref_amax = torch.empty((num_experts, 2, 1), dtype=torch.float32, device=a_ref.device)
        start = 0
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            ref_amax[i, 0] = torch.tensor(compute_reference_amax(dswiglu[start:end, :, 0].clone()))
            ref_amax[i, 1] = torch.tensor(compute_reference_amax(ab[start:end, :, 0].clone()))
            start = end
        ref_tensors["amax_ref"] = ref_amax

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


def check_ref_discrete_dswiglu(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    cfg: Dict[str, Any],
    atol: float = 1e-1,
    rtol: float = 1e-2,
    skip_ref: bool = False,
) -> None:
    """Check discrete backward against a CPU reference for dSwiGLU."""
    if skip_ref:
        print("Skipping reference check")
        return

    is_dgeglu = cfg.get("act_func") == "dgeglu"
    if is_dgeglu:
        torch.cuda.synchronize()
        print("dGeGLU activation: execution checked, numerical reference skipped")
        return

    ref_tensors = run_discrete_dswiglu_ref(
        a_ref=inputs["a_ref"].to(torch.float32),
        b_ref_list=[b.to(torch.float32) for b in inputs["b_ref_list"]],
        c_ref=inputs["c_tensor"].to(torch.float32),
        sfa_ref=inputs["sfa_ref"].to(torch.float32),
        sfb_ref_list=[sfb.to(torch.float32) for sfb in inputs["sfb_ref_list"]],
        alpha_tensor=inputs["alpha_tensor"],
        beta_tensor=inputs["beta_tensor"],
        prob_tensor=inputs["prob_tensor"],
        aligned_group_m_list=inputs["aligned_group_m_list"],
        valid_m=inputs["valid_m"],
        generate_dbias=(outputs.get("dbias_tensor") is not None),
        generate_amax=(outputs.get("amax_tensor") is not None),
        generate_sfd=(outputs.get("sfd_row_tensor") is not None),
        norm_const_tensor=inputs.get("norm_const_tensor"),
        d_dtype=cfg["d_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        sf_dtype=cfg["sf_dtype"],
    )

    torch.cuda.synchronize()

    if ref_tensors.get("dprob_ref") is not None:
        dprob_tensor = outputs.get("dprob_tensor", inputs.get("dprob_tensor"))
        torch.testing.assert_close(
            dprob_tensor[: inputs["valid_m"]].cpu().float(),
            ref_tensors["dprob_ref"].cpu().float(),
            atol=atol,
            rtol=rtol,
        )

    if ref_tensors.get("dbias_ref") is not None and outputs.get("dbias_tensor") is not None:
        max_m = max(inputs["aligned_group_m_list"])
        use_2cta_instrs = cfg["mma_tiler_mn"][0] == 256
        num_tiles_per_expert = max_m // (cfg["mma_tiler_mn"][0] // (2 if use_2cta_instrs else 1))
        dbias_ref = ref_tensors["dbias_ref"].cpu().float()
        dbias_atol = max(dbias_ref.abs().max().item() * 0.008 * (num_tiles_per_expert**0.5), atol)
        torch.testing.assert_close(
            outputs["dbias_tensor"].cpu().float(),
            dbias_ref,
            atol=dbias_atol,
            rtol=rtol,
        )

    if cfg["d_dtype"] in [torch.float32, torch.float16, torch.bfloat16]:
        if ref_tensors.get("amax_ref") is not None and outputs.get("amax_tensor") is not None:
            torch.testing.assert_close(
                outputs["amax_tensor"].cpu(),
                ref_tensors["amax_ref"].cpu(),
                atol=atol,
                rtol=rtol,
            )

        torch.testing.assert_close(
            outputs["d_row_tensor"][: inputs["valid_m"]].cpu().float(),
            ref_tensors["d_ref"].cpu().to(cfg["d_dtype"]).to(torch.float32),
            atol=atol,
            rtol=rtol,
        )
    elif cfg["d_dtype"] in [torch.float8_e4m3fn, torch.float8_e5m2]:
        fp8_d_atol = max(atol, 0.125 if cfg["d_dtype"] == torch.float8_e4m3fn else 0.25)
        fp8_d_rtol = max(rtol, 0.125 if cfg["d_dtype"] == torch.float8_e4m3fn else 0.25)

        if ref_tensors.get("sfd_row_ref") is not None:
            torch.testing.assert_close(
                outputs["sfd_row_tensor"].cpu().float(),
                ref_tensors["sfd_row_ref"].cpu().to(torch.float32),
                atol=atol,
                rtol=rtol,
            )
            torch.testing.assert_close(
                outputs["d_row_tensor"].cpu().float(),
                ref_tensors["d_ref"].cpu().to(cfg["d_dtype"]).to(torch.float32),
                atol=fp8_d_atol,
                rtol=fp8_d_rtol,
            )
        else:
            torch.testing.assert_close(
                outputs["d_row_tensor"][: inputs["valid_m"]].cpu().float(),
                ref_tensors["d_ref"][: inputs["valid_m"]].cpu().to(cfg["d_dtype"]).to(torch.float32),
                atol=fp8_d_atol,
                rtol=fp8_d_rtol,
            )
