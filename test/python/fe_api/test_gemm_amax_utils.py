"""
Utilities and parameterization for GEMM Amax tests.
Contains test configuration fixtures, tensor creation, and reference implementations.
"""

import torch
from cudnn.datatypes import _convert_to_cutlass_data_type
import pytest
from test_low_precision_matmul import (
    _bfloat16_to_float4_e2m1fn_x2,
    float4_e2m1fn_x2_to_float32,
)
from test_fe_api_utils import create_and_permute_tensor, create_scale_factor_tensor


# Parameterization marks for GEMM Amax
GEMM_AMAX_PARAM_MARKS = [
    pytest.mark.parametrize("a_major", ["k", "m"]),
    pytest.mark.parametrize("b_major", ["k", "n"]),
    pytest.mark.parametrize("c_major", ["m", "n"]),
    pytest.mark.parametrize(
        "ab_dtype",
        [torch.float8_e5m2, torch.float8_e4m3fn, torch.uint8, torch.float4_e2m1fn_x2],
    ),
    pytest.mark.parametrize(
        "sf_dtype", [torch.float8_e8m0fnu, torch.int8, torch.float8_e4m3fn]
    ),
    pytest.mark.parametrize(
        "c_dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
            torch.float4_e2m1fn_x2,
            torch.uint8,
        ],
    ),
    pytest.mark.parametrize("acc_dtype", [torch.float32]),
    pytest.mark.parametrize("sf_vec_size", [16, 32]),
    pytest.mark.parametrize("mma_tiler_mn", [(128, 128), (128, 256)]),
    pytest.mark.parametrize("cluster_shape_mn", [(1, 1), (1, 2), (2, 2)]),
]


def with_gemm_amax_params(func):
    """Apply all GEMM Amax parameterization marks to a test function."""
    for mark in reversed(GEMM_AMAX_PARAM_MARKS):
        func = mark(func)
    return func


def gemm_amax_init(
    request,
    a_major,
    b_major,
    c_major,
    ab_dtype,
    sf_dtype,
    c_dtype,
    acc_dtype,
    sf_vec_size,
    mma_tiler_mn,
    cluster_shape_mn,
):
    """Build test config, allowing CLI overrides for problem size/tiling/cluster/skip-ref."""
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(
            f"Environment not supported: requires compute capability >= 10, found {major}"
        )

    mnkl_str = request.config.getoption("--gemm-amax-mnkl", default=None)
    mma_tiler_str = request.config.getoption("--gemm-amax-mma-tiler", default=None)
    cluster_shape_str = request.config.getoption(
        "--gemm-amax-cluster-shape", default=None
    )
    skip_ref = request.config.getoption("--gemm-amax-skip-ref", default=False)

    if mnkl_str is not None:
        m, n, k, l = [int(x.strip()) for x in mnkl_str.split(",")]
    else:
        m, n, k, l = 512, 256, 256, 1

    if mma_tiler_str is not None:
        mma_tiler_mn = tuple(int(x.strip()) for x in mma_tiler_str.split(","))
    if cluster_shape_str is not None:
        cluster_shape_mn = tuple(int(x.strip()) for x in cluster_shape_str.split(","))

    return {
        "m": m,
        "n": n,
        "k": k,
        "l": l,
        "ab_dtype": ab_dtype,
        "sf_dtype": sf_dtype,
        "sf_vec_size": sf_vec_size,
        "c_dtype": c_dtype,
        "acc_dtype": acc_dtype,
        "a_major": a_major,
        "b_major": b_major,
        "c_major": c_major,
        "mma_tiler_mn": mma_tiler_mn,
        "cluster_shape_mn": cluster_shape_mn,
        "skip_ref": skip_ref,
    }


def allocate_input_tensors(
    m, n, k, l, ab_dtype, sf_dtype, sf_vec_size, a_major, b_major
):
    """Allocate and initialize input tensors for GEMM Amax tests."""
    a_ref, a_tensor = create_and_permute_tensor(l, m, k, a_major == "m", ab_dtype)
    b_ref, b_tensor = create_and_permute_tensor(l, n, k, b_major == "n", ab_dtype)
    sfa_ref, sfa_tensor = create_scale_factor_tensor(l, m, k, sf_vec_size, sf_dtype)
    sfb_ref, sfb_tensor = create_scale_factor_tensor(l, n, k, sf_vec_size, sf_dtype)

    return a_tensor, a_ref, b_tensor, b_ref, sfa_tensor, sfa_ref, sfb_tensor, sfb_ref


def allocate_output_tensors(m, n, l, c_dtype, c_major):
    """Allocate and initialize output tensors for GEMM Amax tests."""
    _, c_tensor = create_and_permute_tensor(l, m, n, c_major == "m", c_dtype)
    amax_tensor = torch.full(
        (1, 1, 1), -float("inf"), device="cuda", dtype=torch.float32
    )
    return c_tensor, amax_tensor


def check_ref_gemm_amax(a, b, sfa_ref, sfb_ref, c, amax, skip_ref=False):
    """Check GEMM Amax output against reference implementation."""
    if skip_ref:
        print("Skipping reference check")
        return

    a_ref = a.float().cpu()
    b_ref = b.float().cpu()
    sfa_ref = sfa_ref.float().cpu()
    sfb_ref = sfb_ref.float().cpu()

    res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
    res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
    c_ref = torch.einsum("mkl,nkl->mnl", res_a, res_b)
    amax_ref = torch.amax(torch.abs(c_ref)).to(torch.float32).reshape(1, 1, 1)

    # For FP8 outputs, use cute.testing.convert() to match kernel's conversion behavior
    is_c_fp8 = c.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}
    is_c_fp4 = c.dtype in {torch.float4_e2m1fn_x2, torch.uint8}

    if is_c_fp8:
        from cutlass.cute.runtime import from_dlpack
        from cudnn.datatypes import _convert_to_cutlass_data_type
        import cutlass.cute as cute

        m, n, l = c_ref.shape
        # Convert ref: f32 -> f8 -> f32 using CUTE's conversion
        ref_f8_ = torch.empty(l, m, n, dtype=torch.uint8, device="cuda").permute(
            1, 2, 0
        )
        ref_f8 = from_dlpack(ref_f8_, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        )
        ref_f8.element_type = _convert_to_cutlass_data_type(c.dtype)
        ref_device = c_ref.permute(2, 0, 1).contiguous().permute(1, 2, 0).cuda()
        ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        )
        cute.testing.convert(ref_tensor, ref_f8)  # f32 -> f8
        cute.testing.convert(ref_f8, ref_tensor)  # f8 -> f32
        c_ref = ref_device.cpu()

        torch.testing.assert_close(
            c_ref.to(torch.float32), c.cpu().to(torch.float32), atol=0.1, rtol=0.1
        )
    elif is_c_fp4:
        fp4_c_ref = _bfloat16_to_float4_e2m1fn_x2(
            c_ref.permute(2, 0, 1).to(torch.bfloat16)
        )
        c_ref = (
            float4_e2m1fn_x2_to_float32(fp4_c_ref).to(torch.float32).permute(1, 2, 0)
        )

        c_f32 = (
            float4_e2m1fn_x2_to_float32(
                c.cpu().permute(2, 0, 1).view(torch.float4_e2m1fn_x2)
            )
            .to(torch.float32)
            .permute(1, 2, 0)
        )

        torch.testing.assert_close(
            c_ref.to(torch.float32), c_f32.to(torch.float32), atol=0.1, rtol=0.1
        )
    else:
        c_ref = c_ref.to(c.dtype)
        torch.testing.assert_close(c_ref, c.cpu(), atol=0.01, rtol=0.01)

    torch.testing.assert_close(amax_ref, amax.cpu(), atol=1e-01, rtol=1e-01)
