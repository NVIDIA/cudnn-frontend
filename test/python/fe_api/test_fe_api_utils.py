import torch
import pytest
from test_low_precision_matmul import (
    _bfloat16_to_float4_e2m1fn_x2,
    float4_e2m1fn_x2_to_float32,
)

try:
    import cutlass.cute as cute
    import cutlass

    @cute.jit
    def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        sf_ref_tensor: cute.Tensor,
        sf_mma_tensor: cute.Tensor,
    ):
        """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
        # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
        # group to ((32, 4, rest_m), (4, rest_k), l)

        sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
        sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
        for i in cutlass.range(cute.size(sf_ref_tensor)):
            mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
            sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]

except Exception:
    cute = None
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L = None


def create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype):
    # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
    # else: (l, mode0, mode1) -> (mode0, mode1, l)
    shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
    is_unsigned = dtype in {
        torch.uint16,
        torch.uint32,
        torch.uint64,
    }  # uint8 is interpreted as fp4x2
    min_val = 0 if is_unsigned else -2
    max_val = 4 if is_unsigned else 2

    dtype_tensor = None
    ref_tensor = None

    # Generate random values according to dtype support
    if dtype in {torch.int8, torch.int16, torch.int32, torch.int64}:
        ref_tensor = torch.randint(
            int(min_val), int(max_val), shape, dtype=torch.int32, device="cuda"
        ).permute(permute_order)
        dtype_tensor = ref_tensor.to(dtype)
    if dtype not in {torch.float4_e2m1fn_x2, torch.uint8}:
        dtype_tensor = (
            torch.empty(shape, dtype=torch.float32, device="cuda")
            .uniform_(float(min_val), float(max_val))
            .permute(permute_order)
            .to(dtype)
        )
        ref_tensor = dtype_tensor.to(torch.float32)
    else:
        dtype_tensor = _bfloat16_to_float4_e2m1fn_x2(
            torch.empty(shape, dtype=torch.float32, device="cuda")
            .uniform_(float(min_val), float(max_val))
            .to(torch.bfloat16)
        )
        ref_tensor = (
            float4_e2m1fn_x2_to_float32(dtype_tensor)
            .to(torch.float32)
            .permute(permute_order)
        )
        dtype_tensor = dtype_tensor.permute(permute_order).view(dtype)

    return ref_tensor, dtype_tensor


def create_sf_layout_tensor(l, mn, nk, sf_vec_size):
    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(nk, sf_vec_size)

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

    mma_permute_order = (3, 4, 1, 5, 2, 0)

    # Create f32 cute torch tensor (cpu)
    cute_f32_torch_tensor_cpu = torch.zeros(mma_shape, dtype=torch.float32).permute(
        mma_permute_order
    )

    return cute_f32_torch_tensor_cpu, sf_k


# Create scale factor tensor SFA/SFB
def create_scale_factor_tensor(l, mn, k, sf_vec_size, dtype):
    cute_f32_torch_tensor_cpu, sf_k = create_sf_layout_tensor(l, mn, k, sf_vec_size)
    ref_shape = (l, mn, sf_k)
    ref_permute_order = (1, 2, 0)

    # Create f32 ref torch tensor (cpu)
    ref_f32_torch_tensor_cpu = (
        torch.empty(ref_shape, dtype=torch.float32)
        .uniform_(1, 3)
        .permute(ref_permute_order)
        .to(torch.int8)
        .to(torch.float32)
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
            "CUTLASS is not installed; skipping tests due to scale factor tensor creation requiring CUTLASS."
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
