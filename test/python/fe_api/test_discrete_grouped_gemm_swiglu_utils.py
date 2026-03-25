"""
Utilities and parameterization for Discrete-weight Grouped GEMM GLU tests.
Contains test configuration fixtures, tensor creation, and reference implementations.
"""

import torch
import pytest
from typing import Tuple, List, Dict, Any
from test_fe_api_utils import (
    compute_reference_amax,
    create_and_permute_tensor,
    create_scale_factor_tensor,
)

# =============================================================================
# Parameterization Marks
# =============================================================================

DISCRETE_GROUPED_GEMM_PARAM_MARKS_FP8 = [
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
        ],
    ),
    pytest.mark.parametrize("sf_vec_size", [32]),
    pytest.mark.parametrize(
        "sf_dtype",
        [
            torch.float8_e8m0fnu,
        ],
    ),
    pytest.mark.parametrize("vector_f32", [False]),
    pytest.mark.parametrize("discrete_col_sfd", [False]),
    pytest.mark.parametrize("act_func", ["swiglu", "geglu"]),
    pytest.mark.parametrize("b_major", ["k", "n"]),
]

DISCRETE_GROUPED_GEMM_PARAM_MARKS_FP4 = [
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
        ],
    ),
    pytest.mark.parametrize("sf_vec_size", [32]),
    pytest.mark.parametrize(
        "sf_dtype",
        [
            torch.float8_e8m0fnu,
        ],
    ),
    pytest.mark.parametrize("vector_f32", [False]),
    pytest.mark.parametrize("discrete_col_sfd", [False]),
    pytest.mark.parametrize("act_func", ["swiglu", "geglu"]),
]


def with_discrete_grouped_gemm_params_fp4(func):
    """Decorator to apply discrete grouped GEMM FP4 test parameters."""
    for mark in reversed(DISCRETE_GROUPED_GEMM_PARAM_MARKS_FP4):
        func = mark(func)
    return func


def with_discrete_grouped_gemm_params_fp8(func):
    """Decorator to apply discrete grouped GEMM FP8 test parameters."""
    for mark in reversed(DISCRETE_GROUPED_GEMM_PARAM_MARKS_FP8):
        func = mark(func)
    return func


# =============================================================================
# Configuration Initialization
# =============================================================================


def discrete_grouped_gemm_init(
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
    act_func: str = "swiglu",
    b_major: str = "k",
) -> Dict[str, Any]:
    """Initialize configuration for discrete grouped GEMM GLU tests."""
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(f"Environment not supported: requires compute capability >= 10, found {major}")

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

    config = {
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

    return config


# =============================================================================
# Helper Functions
# =============================================================================


def create_mask(
    group_m_list: List[int],
    m_aligned: int = 256,
) -> Tuple[int, List[int], torch.Tensor]:
    """Create padded_offsets tensor from group_m_list."""
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


def allocate_discrete_input_tensors(
    n: int,
    k: int,
    num_experts: int,
    group_m_list: List[int],
    ab_dtype: torch.dtype,
    sf_dtype: torch.dtype,
    sf_vec_size: int,
    m_aligned: int,
    norm_const: float = 0.01,
    b_major: str = "k",
    device: str = "cuda",
) -> Dict[str, Any]:
    """Allocate input tensors for discrete grouped GEMM GLU.

    Creates per-expert B and SFB tensors as separate allocations (lists).
    """
    valid_m, aligned_group_m_list, padded_offsets_tensor = create_mask(group_m_list, m_aligned)
    tensor_m = valid_m

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

    alpha_tensor = torch.randint(-2, 2, (num_experts,), dtype=torch.float32, device=device).float()
    prob_tensor = torch.randint(-2, 2, (tensor_m, 1, 1), dtype=torch.float32, device=device).float()

    # Pre-built device pointer tensors (int64) -- the production-path input format
    b_ptrs_tensor = torch.tensor(
        [b.data_ptr() for b in b_list],
        dtype=torch.int64,
        device=device,
    )
    sfb_ptrs_tensor = torch.tensor(
        [sfb.data_ptr() for sfb in sfb_list],
        dtype=torch.int64,
        device=device,
    )

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
        "alpha_tensor": alpha_tensor,
        "prob_tensor": prob_tensor,
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


def allocate_discrete_output_tensors(
    tensor_m: int,
    n: int,
    num_experts: int,
    ab_dtype: torch.dtype,
    c_dtype: torch.dtype,
    d_dtype: torch.dtype,
    cd_major: str,
    sf_dtype: torch.dtype,
    sf_vec_size: int = 16,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Allocate output tensors for discrete grouped GEMM GLU."""
    n_out = n // 2

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
        result["amax_tensor"] = torch.full((num_experts, 1), float("-inf"), dtype=torch.float32, device=device)

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
# Reference Implementation
# =============================================================================


def run_discrete_grouped_gemm_ref(
    a_ref: torch.Tensor,
    b_ref_list: List[torch.Tensor],
    sfa_ref: torch.Tensor,
    sfb_ref_list: List[torch.Tensor],
    alpha_tensor: torch.Tensor,
    prob_tensor: torch.Tensor,
    aligned_group_m_list: List[int],
    valid_m: int,
    generate_amax: bool = False,
    c_dtype: torch.dtype = torch.bfloat16,
    d_dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Run reference implementation for discrete grouped GEMM GLU (SwiGLU).

    The math is identical to the contiguous kernel reference, but B/SFB
    are provided as per-expert lists instead of a stacked (n, k, l) tensor.

    :param a_ref: A tensor (tensor_m, k, 1) in float32
    :param b_ref_list: List of per-expert B tensors, each (n, k, 1) in float32
    :param sfa_ref: Scale factor A tensor (tensor_m, k, 1) in float32
    :param sfb_ref_list: List of per-expert SFB tensors, each (n, k, 1) in float32
    :param alpha_tensor: Per-group alpha scaling (l,)
    :param prob_tensor: Per-row probability scaling (tensor_m, 1, 1)
    :param aligned_group_m_list: Aligned M values per group
    :param valid_m: Total valid M dimension
    :param generate_amax: Generate AMAX tensor
    :param c_dtype: Intermediate C tensor dtype
    :param d_dtype: Output D tensor dtype
    :return: Dict with c_ref, d_ref, and optionally amax_ref
    """
    num_experts = len(b_ref_list)
    n = b_ref_list[0].shape[0]
    n_out = n // 2
    ref_tensors = {}

    # Step 1: Compute GEMM per group with scale factors
    ref = torch.empty((1, valid_m, n), dtype=torch.float32, device=a_ref.device)
    start = 0
    for i, group_m in enumerate(aligned_group_m_list):
        end = start + group_m
        # a_ref is (tensor_m, k, 1), b_ref_list[i] is (n, k, 1)
        res_a = torch.einsum("mk,mk->mk", a_ref[start:end, :, 0], sfa_ref[start:end, :, 0])
        res_b = torch.einsum("nk,nk->nk", b_ref_list[i][:, :, 0], sfb_ref_list[i][:, :, 0])
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
    assert num_blocks % 2 == 0, "Number of 32-col blocks must be even (pairs of gate/up)"

    cols = torch.arange(n, device=ref.device, dtype=torch.long)
    block_cols = cols.view(num_blocks, group)
    gate_idx = block_cols[0::2].reshape(-1)
    up_idx = block_cols[1::2].reshape(-1)
    ref_gate = ref.index_select(1, gate_idx)
    ref_up = ref.index_select(1, up_idx)

    # SwiGLU: up * (gate * sigmoid(gate))
    ref_gate = ref_gate * torch.sigmoid(ref_gate)
    ref_after_swiglu = ref_up * ref_gate

    # Step 4: Apply prob
    ref_after_swiglu = ref_after_swiglu * prob_tensor.expand(-1, n_out, -1)
    ref_tensors["d_ref"] = ref_after_swiglu.clone()

    if generate_amax:
        amax_ref = torch.empty((num_experts,), dtype=torch.float32, device=a_ref.device)
        start = 0
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            amax_ref[i] = compute_reference_amax(ref_after_swiglu[start:end, :, 0].clone())
            start = end
        ref_tensors["amax_ref"] = amax_ref

    return ref_tensors


# =============================================================================
# Reference Checking
# =============================================================================


def check_ref_discrete_grouped_gemm(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    cfg: Dict[str, Any],
    atol: float = 1e-1,
    rtol: float = 1e-2,
    skip_ref: bool = False,
) -> None:
    """Check discrete grouped GEMM GLU result against CPU reference.

    :param inputs: Dictionary of input tensors (from allocate_discrete_input_tensors)
    :param outputs: Dictionary of output tensors (from allocate_discrete_output_tensors or wrapper)
    :param cfg: Configuration dictionary (from discrete_grouped_gemm_init)
    :param atol: Absolute tolerance
    :param rtol: Relative tolerance
    :param skip_ref: Skip reference check if True
    """
    if skip_ref:
        print("Skipping reference check")
        return

    # The CPU reference only implements SwiGLU. For GeGLU, we can still check
    # the C tensor (GEMM result before activation, identical for both) but must
    # skip the D/AMAX check since the activation math differs.
    is_geglu = cfg.get("act_func") in ["geglu", "dgeglu"]

    ref_tensors = run_discrete_grouped_gemm_ref(
        a_ref=inputs["a_ref"].to(torch.float32),
        b_ref_list=[b.to(torch.float32) for b in inputs["b_ref_list"]],
        sfa_ref=inputs["sfa_ref"].to(torch.float32),
        sfb_ref_list=[sfb.to(torch.float32) for sfb in inputs["sfb_ref_list"]],
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        aligned_group_m_list=inputs["aligned_group_m_list"],
        valid_m=inputs["valid_m"],
        generate_amax=(outputs.get("amax_tensor") is not None),
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
    )

    torch.cuda.synchronize()

    # Check C tensor (intermediate GEMM result)
    c_gpu = outputs["c_tensor"][: inputs["valid_m"]]
    c_ref = ref_tensors["c_ref"]
    torch.testing.assert_close(
        c_gpu.cpu().float(),
        c_ref.cpu().to(cfg["c_dtype"]).to(torch.float32),
        atol=atol,
        rtol=rtol,
    )

    if is_geglu:
        print("GeGLU activation: C tensor checked, D/AMAX skipped (no GeGLU CPU reference)")
        return

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
        pass
