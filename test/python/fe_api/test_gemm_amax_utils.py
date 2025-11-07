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


try:
    import cutlass.cute as cute

    @cute.jit
    def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        sf_ref_tensor: cute.Tensor,
        sf_mma_tensor: cute.Tensor,
    ):
        """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
        # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
        # group to ((32, 4, rest_m), (4, rest_k), l)
        import cutlass

        sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
        sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
        for i in cutlass.range(cute.size(sf_ref_tensor)):
            mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
            sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]

except Exception:
    cute = None
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L = None


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
    a_ref, a_tensor = _create_and_permute_tensor(l, m, k, a_major == "m", ab_dtype)
    b_ref, b_tensor = _create_and_permute_tensor(l, n, k, b_major == "n", ab_dtype)
    sfa_ref, sfa_tensor = create_scale_factor_tensor(l, m, k, sf_vec_size, sf_dtype)
    sfb_ref, sfb_tensor = create_scale_factor_tensor(l, n, k, sf_vec_size, sf_dtype)

    return a_tensor, a_ref, b_tensor, b_ref, sfa_tensor, sfa_ref, sfb_tensor, sfb_ref


def allocate_output_tensors(m, n, l, c_dtype, c_major):
    """Allocate and initialize output tensors for GEMM Amax tests."""
    _, c_tensor = _create_and_permute_tensor(l, m, n, c_major == "m", c_dtype)
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


def _create_and_permute_tensor(l_val, mode0, mode1, is_mode0_major, dtype):
    # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
    # else: (l, mode0, mode1) -> (mode0, mode1, l)
    shape = (l_val, mode1, mode0) if is_mode0_major else (l_val, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
    f32_tensor = torch.rand(shape, dtype=torch.float32)

    dtype_tensor = None
    ref_tensor = None
    if dtype not in {torch.float4_e2m1fn_x2, torch.uint8}:
        dtype_tensor = f32_tensor.to(dtype).permute(permute_order).cuda()
        ref_tensor = dtype_tensor.to(torch.float32)
    else:
        dtype_tensor = _bfloat16_to_float4_e2m1fn_x2(f32_tensor.to(torch.bfloat16))
        ref_tensor = (
            float4_e2m1fn_x2_to_float32(dtype_tensor)
            .to(torch.float32)
            .permute(permute_order)
            .cuda()
        )
        dtype_tensor = dtype_tensor.permute(permute_order).cuda().view(dtype)

    return ref_tensor, dtype_tensor


# Create scale factor tensor SFA/SFB
def create_scale_factor_tensor(l, mn, k, sf_vec_size, dtype):
    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(k, sf_vec_size)
    ref_shape = (l, mn, sf_k)

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

    ref_permute_order = (1, 2, 0)
    mma_permute_order = (3, 4, 1, 5, 2, 0)

    # Create f32 ref torch tensor (cpu)
    ref_f32_torch_tensor_cpu = (
        torch.empty(ref_shape, dtype=torch.float32)
        .uniform_(1, 3)
        .permute(ref_permute_order)
        .to(torch.int8)
        .to(torch.float32)
    )

    # Create f32 cute torch tensor (cpu)
    cute_f32_torch_tensor_cpu = torch.zeros(mma_shape, dtype=torch.float32).permute(
        mma_permute_order
    )

    # convert ref f32 tensor to cute f32 tensor
    try:
        from cutlass.cute.runtime import from_dlpack

        cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
            from_dlpack(ref_f32_torch_tensor_cpu),
            from_dlpack(cute_f32_torch_tensor_cpu),
        )
    except Exception:
        pytest.skip(
            "CUTLASS is not installed; skipping GEMM Amax tests requiring CUTLASS."
        )

    # reshape makes memory contiguous
    ref_f32_torch_tensor_cpu = (
        ref_f32_torch_tensor_cpu.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, mn, sf_k, sf_vec_size)
        .reshape(l, mn, sf_k * sf_vec_size)
        .permute(*ref_permute_order)
    )
    ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]

    if dtype != torch.int8:
        cute_torch_tensor = cute_f32_torch_tensor_cpu.to(dtype).cuda()
    else:
        cute_torch_tensor = (
            cute_f32_torch_tensor_cpu.to(torch.float8_e8m0fnu).cuda().view(dtype)
        )

    return ref_f32_torch_tensor_cpu.cuda(), cute_torch_tensor
