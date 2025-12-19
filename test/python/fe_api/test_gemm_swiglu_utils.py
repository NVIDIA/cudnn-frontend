"""
Utilities and parameterization for GEMM SwiGLU tests.
Contains test configuration fixtures, tensor creation, and reference implementations.
"""

import torch
import pytest
from typing import Optional, Tuple
from test_fe_api_utils import (
    create_and_permute_tensor,
    create_scale_factor_tensor,
    create_sf_layout_tensor,
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L,
)
from test_low_precision_matmul import (
    _bfloat16_to_float4_e2m1fn_x2,
    float4_e2m1fn_x2_to_float32,
)

GEMM_SWIGLU_PARAM_MARKS = [
    pytest.mark.parametrize("a_major", ["k", "m"]),
    pytest.mark.parametrize("b_major", ["k", "n"]),
    pytest.mark.parametrize("c_major", ["m", "n"]),
    pytest.mark.parametrize(
        "ab_dtype",
        [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ],
    ),
    pytest.mark.parametrize(
        "ab12_dtype", [torch.float16, torch.bfloat16, torch.float32]
    ),
    pytest.mark.parametrize(
        "acc_dtype", [torch.float32]
    ),  # Note: float16 accumulator is supported but disabled in testing
    pytest.mark.parametrize("c_dtype", [torch.float16, torch.bfloat16]),
    pytest.mark.parametrize(
        "mma_tiler_mn", [(128, 128), (128, 64), (256, 256), (256, 128)]
    ),
    pytest.mark.parametrize("cluster_shape_mn", [(1, 1), (2, 2), (4, 4)]),
]


def with_gemm_swiglu_params(func):
    for mark in reversed(GEMM_SWIGLU_PARAM_MARKS):
        func = mark(func)
    return func


GEMM_SWIGLU_QUANT_PARAM_MARKS = [
    pytest.mark.parametrize("a_major", ["k", "m"]),
    pytest.mark.parametrize("b_major", ["k", "n"]),
    pytest.mark.parametrize("c_major", ["m", "n"]),
    pytest.mark.parametrize(
        "ab_dtype",
        [
            torch.float4_e2m1fn_x2,
            # torch.uint8,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ],
    ),
    pytest.mark.parametrize(
        "ab12_dtype",
        [
            # torch.float32
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ],
    ),
    pytest.mark.parametrize(
        "c_dtype",
        [
            # torch.float32
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ],
    ),
    pytest.mark.parametrize("acc_dtype", [torch.float32]),
    pytest.mark.parametrize(
        "mma_tiler_mn",
        [
            (128, 128),
            (256, 256),
            # (128, 64),
            # (256, 128),
        ],
    ),
    pytest.mark.parametrize(
        "cluster_shape_mn",
        [
            (1, 1),
            (2, 2),
            (4, 4),
        ],
    ),
    pytest.mark.parametrize("sf_vec_size", [16, 32]),
    pytest.mark.parametrize("sf_dtype", [torch.float8_e8m0fnu, torch.float8_e4m3fn]),
    pytest.mark.parametrize("vector_f32", [True, False]),
]


def with_gemm_swiglu_quant_params(func):
    for mark in reversed(GEMM_SWIGLU_QUANT_PARAM_MARKS):
        func = mark(func)
    return func


def gemm_swiglu_init(
    request,
    a_major,
    b_major,
    c_major,
    ab_dtype,
    ab12_dtype,
    acc_dtype,
    c_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    ### Quantize only arguments
    sf_vec_size: Optional[int] = None,
    sf_dtype: Optional[torch.dtype] = None,
    vector_f32: Optional[bool] = None,
):
    """Initialize configuration for GEMM SwiGLU tests."""
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(
            f"Environment not supported: requires compute capability >= 10, found {major}"
        )

    mnkl_str = request.config.getoption("--gemm-swiglu-mnkl", default=None)
    mma_tiler_str = request.config.getoption("--gemm-swiglu-mma-tiler", default=None)
    cluster_shape_str = request.config.getoption(
        "--gemm-swiglu-cluster-shape", default=None
    )
    alpha_opt = request.config.getoption("--gemm-swiglu-alpha", default=None)
    skip_ref = request.config.getoption("--gemm-swiglu-skip-ref", default=False)

    if mnkl_str is not None:
        m, n, k, l = [int(x.strip()) for x in mnkl_str.split(",")]
    else:
        m, n, k, l = 256, 256, 512, 2

    if mma_tiler_str is not None:
        mma_tiler_mn = tuple(int(x.strip()) for x in mma_tiler_str.split(","))
    if cluster_shape_str is not None:
        cluster_shape_mn = tuple(int(x.strip()) for x in cluster_shape_str.split(","))

    alpha = float(alpha_opt) if alpha_opt is not None else 1.0

    config = {
        "m": m,
        "n": n,
        "k": k,
        "l": l,
        "mma_tiler_mn": mma_tiler_mn,
        "cluster_shape_mn": cluster_shape_mn,
        "alpha": alpha,
        "skip_ref": skip_ref,
        "a_major": a_major,
        "b_major": b_major,
        "c_major": c_major,
        "ab_dtype": ab_dtype,
        "ab12_dtype": ab12_dtype,
        "acc_dtype": acc_dtype,
        "c_dtype": c_dtype,
    }

    # Add quantization parameters if provided
    if sf_vec_size is not None:
        config["sf_vec_size"] = sf_vec_size
    if sf_dtype is not None:
        config["sf_dtype"] = sf_dtype
    if vector_f32 is not None:
        config["vector_f32"] = vector_f32

    return config


def get_dtype_rcp_limits(dtype: torch.dtype) -> float:
    if dtype == torch.float8_e5m2:
        return 1 / 128.0
    elif dtype == torch.float8_e4m3fn:
        return 1 / 448.0
    elif dtype in {torch.float4_e2m1fn_x2, torch.uint8}:
        return 1 / 6.0
    return 1.0


def run_gemm_swiglu_quant_ref(
    a_ref,
    b_ref,
    sfa_ref,
    sfb_ref,
    norm_const_ref,
    alpha,
    ab_dtype,
    ab12_dtype,
    c_dtype,
    sfc_dtype,
    sf_vec_size,
):
    res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
    res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
    ab12_ref = alpha * torch.einsum("mkl,nkl->mnl", res_a, res_b)
    ab12_ref_ret = ab12_ref.to(ab12_dtype).to(torch.float32)

    group = 32
    n = b_ref.shape[0]
    assert n % group == 0, "N must be divisible by 32 for GLU block grouping"
    num_blocks = n // group
    assert (
        num_blocks % 2 == 0
    ), "Number of 32-col blocks must be even (pairs of input/gate)"

    cols = torch.arange(n, device=ab12_ref.device, dtype=torch.long)
    block_cols = cols.view(num_blocks, group)
    input_idx = block_cols[0::2].reshape(-1)
    gate_idx = block_cols[1::2].reshape(-1)
    ref_input = ab12_ref.index_select(1, input_idx)
    ref_gate = ab12_ref.index_select(1, gate_idx)
    c_ref = ref_input * (ref_gate * torch.sigmoid(ref_gate))

    amax_ref = None
    sfc_ref = None
    if c_dtype == torch.bfloat16 and ab_dtype in {torch.float4_e2m1fn_x2, torch.uint8}:
        amax_ref = torch.tensor(compute_reference_amax(c_ref.clone()))
    elif c_dtype in {torch.float8_e5m2, torch.float8_e4m3fn}:
        try:
            from cutlass.cute.runtime import from_dlpack
            import cutlass.cute as cute
            from cudnn.datatypes import _convert_to_cutlass_data_type
        except ImportError:
            pytest.skip("CUTLASS not available for scale factor conversion")

        sfn = (n // 2 + sf_vec_size - 1) // sf_vec_size
        m = a_ref.shape[0]
        sfm = (m + 128 - 1) // 128 * 128
        l = c_ref.shape[2]
        ref_for_sf = c_ref.permute(2, 0, 1).contiguous()
        ref_for_sf = ref_for_sf.view(l, sfm, sfn, sf_vec_size)
        ref_for_sf, _ = torch.abs(ref_for_sf).max(dim=3)
        ref_sfc_f32 = ref_for_sf * norm_const_ref * get_dtype_rcp_limits(c_dtype)
        ref_sfc_f32 = ref_sfc_f32.permute(1, 2, 0)

        # For some reason, using `ref_sfc_32_torch = ref_sfc_f32.to(sfc_dtype).to(torch.float32)` leads to different/incorrect results
        ref_sfc_f8_torch = torch.empty(
            (l, sfm, sfn), dtype=torch.uint8, device="cuda"
        ).permute(1, 2, 0)
        ref_sfc_f8 = from_dlpack(
            ref_sfc_f8_torch, assumed_align=16
        ).mark_layout_dynamic(leading_dim=1)
        ref_sfc_f8.element_type = _convert_to_cutlass_data_type(sfc_dtype)
        ref_sfc_f32_device = ref_sfc_f32.cuda()
        ref_sfc_f32_tensor = from_dlpack(
            ref_sfc_f32_device, assumed_align=16
        ).mark_layout_dynamic(leading_dim=1)
        cute.testing.convert(ref_sfc_f32_tensor, ref_sfc_f8)
        cute.testing.convert(ref_sfc_f8, ref_sfc_f32_tensor)
        ref_sfc_32 = ref_sfc_f32_device.cpu()

        ref_sfc_f32_cute_torch_tensor_cpu, _ = create_sf_layout_tensor(
            l, sfm, n // 2, sf_vec_size
        )
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
            from_dlpack(ref_sfc_32),
            from_dlpack(ref_sfc_f32_cute_torch_tensor_cpu),
        )
        sfc_ref = ref_sfc_f32_cute_torch_tensor_cpu.clone()

        ref_sfc_rcp = norm_const_ref * ref_sfc_32.reciprocal()
        ref_sfc_rcp_expanded = ref_sfc_rcp.unsqueeze(2).expand(sfm, sfn, sf_vec_size, l)
        ref_sfc_rcp_expanded = ref_sfc_rcp_expanded.reshape(sfm, sfn * sf_vec_size, l)
        ref_sfc_rcp_expanded = ref_sfc_rcp_expanded[:, : n // 2, :]
        c_ref = torch.einsum("mnl,mnl->mnl", c_ref, ref_sfc_rcp_expanded)
        c_ref = c_ref.to(c_dtype).to(torch.float32)

    return ab12_ref_ret, c_ref, sfc_ref, amax_ref


def compute_reference_amax(output_tensor: torch.Tensor) -> float:
    if output_tensor.dtype != torch.float32:
        output_fp32 = output_tensor.float()
    else:
        output_fp32 = output_tensor
    return torch.amax(torch.abs(output_fp32)).item()


def run_gemm_swiglu_ref(a_ref, b_ref, alpha):
    ab12_ref, c_ref = None, None
    if a_ref.dtype in {torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2}:
        ab12_ref = alpha * torch.einsum("mkl,nkl->mnl", (a_ref).cpu(), (b_ref).cpu())
    else:
        ab12_ref = (alpha * torch.einsum("mkl,nkl->mnl", (a_ref), (b_ref))).cpu()

    group = 32
    n = b_ref.shape[0]
    assert n % group == 0, "N must be divisible by 32 for GLU block grouping"
    num_blocks = n // group
    assert (
        num_blocks % 2 == 0
    ), "Number of 32-col blocks must be even (pairs of input/gate)"

    cols = torch.arange(n, device=ab12_ref.device, dtype=torch.long)
    block_cols = cols.view(num_blocks, group)
    input_idx = block_cols[0::2].reshape(-1)
    gate_idx = block_cols[1::2].reshape(-1)
    c_ref = ab12_ref.index_select(1, input_idx) * (
        ab12_ref.index_select(1, gate_idx)
        * torch.sigmoid(ab12_ref.index_select(1, gate_idx))
    )
    c_ref = c_ref.to(torch.float32)

    return ab12_ref, c_ref


def check_ref_gemm_swiglu(
    a: torch.Tensor,
    b: torch.Tensor,
    ab12: torch.Tensor,
    c: torch.Tensor,
    alpha: float = 1.0,
    skip_ref: bool = False,
):
    if not skip_ref:
        a_ref = a.clone().to(torch.float32)
        b_ref = b.clone().to(torch.float32)
        ab12_ref, c_ref = run_gemm_swiglu_ref(a_ref, b_ref, alpha)

        is_ab12_fp8 = ab12.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}
        if is_ab12_fp8:
            torch.testing.assert_close(
                ab12.cpu().to(torch.float32),
                ab12_ref.to(torch.float32),
                atol=0.1,
                rtol=0.1,
            )
        else:
            torch.testing.assert_close(
                ab12.cpu(), ab12_ref.to(ab12.dtype), atol=0.01, rtol=9e-03
            )

        is_c_fp8 = c.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}
        if is_c_fp8:
            torch.testing.assert_close(
                c.cpu().to(torch.float32),
                c_ref.to(torch.float32),
                atol=0.1,
                rtol=0.1,
            )
        else:
            torch.testing.assert_close(
                c.cpu(), c_ref.to(c.dtype), atol=0.01, rtol=9e-03
            )
    else:
        print("Skipping reference check")


def check_ref_gemm_swiglu_quant(
    a: torch.Tensor,
    a_ref: torch.Tensor,
    b: torch.Tensor,
    b_ref: torch.Tensor,
    sfa_ref: torch.Tensor,
    sfb_ref: torch.Tensor,
    ab12: torch.Tensor,
    c: torch.Tensor,
    sfc: Optional[torch.Tensor],
    amax: Optional[torch.Tensor],
    norm_const_ref: Optional[torch.Tensor],
    sf_vec_size: int = 16,
    alpha: float = 1.0,
    skip_ref: bool = False,
):
    if skip_ref:
        print("Skipping reference check")
        return

    ab_dtype = a.dtype
    c_dtype = c.dtype
    ab12_dtype = ab12.dtype
    a_ref = a_ref.clone().to(torch.float32).cpu()
    b_ref = b_ref.clone().to(torch.float32).cpu()
    sfa_ref = sfa_ref.float().cpu()
    sfb_ref = sfb_ref.float().cpu()
    norm_const_ref = (
        norm_const_ref.float().cpu() if norm_const_ref is not None else None
    )
    sfc_dtype = sfc.dtype if sfc is not None else None
    ab12_ref, c_ref, sfc_ref, amax_ref = run_gemm_swiglu_quant_ref(
        a_ref,
        b_ref,
        sfa_ref,
        sfb_ref,
        norm_const_ref,
        alpha,
        ab_dtype,
        ab12_dtype,
        c_dtype,
        sfc_dtype,
        sf_vec_size,
    )

    if ab12.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
        torch.testing.assert_close(
            ab12.cpu().to(torch.float32),
            ab12_ref.to(torch.float32),
            atol=0.01,
            rtol=0.01,
        )
    else:
        torch.testing.assert_close(
            ab12.cpu(), ab12_ref.to(ab12.dtype), atol=0.01, rtol=0.01
        )

    if c_dtype in {torch.float32, torch.float16, torch.bfloat16}:
        torch.testing.assert_close(c.cpu(), c_ref.to(c.dtype), atol=0.01, rtol=0.01)
        if c_dtype == torch.bfloat16 and ab_dtype in {
            torch.float4_e2m1fn_x2,
            torch.uint8,
        }:
            reference_amax = torch.tensor(compute_reference_amax(c_ref.clone()))
            torch.testing.assert_close(
                amax.cpu().squeeze(), reference_amax, atol=0.01, rtol=0.01
            )
    elif c_dtype in {torch.float8_e5m2, torch.float8_e4m3fn}:
        torch.testing.assert_close(
            sfc.cpu().to(torch.float32),
            sfc_ref.to(torch.float32),
            atol=0.01,
            rtol=0.01,
        )
        torch.testing.assert_close(
            c.cpu().to(torch.float32), c_ref.to(torch.float32), atol=0.01, rtol=0.01
        )


def allocate_input_tensors(
    m: int,
    n: int,
    k: int,
    l: int,
    ab_dtype: torch.dtype,
    a_major: str,
    b_major: str,
    is_block_scaled: bool = False,
    ### block scaled only params
    sf_vec_size: Optional[int] = None,
    sf_dtype: Optional[torch.dtype] = None,
    c_dtype: Optional[torch.dtype] = None,
    norm_const: float = 1.0,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    a_ref, a_tensor = create_and_permute_tensor(l, m, k, a_major == "m", ab_dtype)
    b_ref, b_tensor = create_and_permute_tensor(l, n, k, b_major == "n", ab_dtype)
    ### Block scaled-only params
    sfa_ref, sfa_tensor, sfb_ref, sfb_tensor, norm_const_tensor = (
        None,
        None,
        None,
        None,
        None,
    )
    if is_block_scaled:
        sfa_ref, sfa_tensor = create_scale_factor_tensor(l, m, k, sf_vec_size, sf_dtype)
        sfb_ref, sfb_tensor = create_scale_factor_tensor(l, n, k, sf_vec_size, sf_dtype)
        if c_dtype in {torch.float8_e5m2, torch.float8_e4m3fn}:
            norm_const_tensor = torch.tensor([norm_const], dtype=torch.float32).cuda()

    return (
        a_tensor,
        a_ref,
        b_tensor,
        b_ref,
        sfa_tensor,
        sfa_ref,
        sfb_tensor,
        sfb_ref,
        norm_const_tensor,
    )


def allocate_output_tensors(
    m: int,
    n: int,
    l: int,
    ab12_dtype: torch.dtype,
    c_dtype: torch.dtype,
    c_major: str,
    is_block_scaled: bool = False,
    ### block scaled only params
    sf_vec_size: Optional[int] = None,
    sf_dtype: Optional[torch.dtype] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    _, ab12_tensor = create_and_permute_tensor(l, m, n, c_major == "m", ab12_dtype)
    _, c_tensor = create_and_permute_tensor(l, m, n // 2, c_major == "m", c_dtype)

    ### Block scaled-only params
    sfc_ref, sfc_tensor, amax_tensor = None, None, None
    if is_block_scaled:
        if c_dtype in {torch.float8_e5m2, torch.float8_e4m3fn}:
            sfc_ref, sfc_tensor = create_scale_factor_tensor(
                l, m, n // 2, sf_vec_size, sf_dtype
            )
        if c_dtype == torch.bfloat16:
            amax_tensor = torch.full(
                (1, 1, 1), -float("inf"), device="cuda", dtype=torch.float32
            )

    return ab12_tensor, c_tensor, sfc_tensor, sfc_ref, amax_tensor
