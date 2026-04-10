"""Utilities for grouped GEMM wgrad FE API tests."""

from typing import Any, Dict, Tuple

import pytest
import torch
from fe_api.test_fe_api_utils import ceil_div

GROUPED_GEMM_WGRAD_PARAM_MARKS_FP4 = [
    pytest.mark.parametrize("ab_dtype", [torch.float4_e2m1fn_x2]),
    pytest.mark.parametrize("wgrad_dtype", [torch.bfloat16]),
    pytest.mark.parametrize("acc_dtype", [torch.float32]),
    pytest.mark.parametrize("mma_tiler_mn", [(128, 128), (256, 256)]),
    pytest.mark.parametrize("cluster_shape_mn", [(1, 1), (2, 1)]),
    pytest.mark.parametrize("sf_vec_size", [16]),
    pytest.mark.parametrize("sf_dtype", [torch.float8_e4m3fn]),
]

GROUPED_GEMM_WGRAD_PARAM_MARKS_FP8 = [
    pytest.mark.parametrize("ab_dtype", [torch.float8_e4m3fn]),
    pytest.mark.parametrize("wgrad_dtype", [torch.bfloat16]),
    pytest.mark.parametrize("acc_dtype", [torch.float32]),
    pytest.mark.parametrize("mma_tiler_mn", [(128, 128), (256, 256)]),
    pytest.mark.parametrize("cluster_shape_mn", [(1, 1), (2, 1)]),
    pytest.mark.parametrize("sf_vec_size", [32]),
    pytest.mark.parametrize("sf_dtype", [torch.float8_e8m0fnu]),
]


def with_grouped_gemm_wgrad_params_fp4(func):
    for mark in reversed(GROUPED_GEMM_WGRAD_PARAM_MARKS_FP4):
        func = mark(func)
    return func


def with_grouped_gemm_wgrad_params_fp8(func):
    for mark in reversed(GROUPED_GEMM_WGRAD_PARAM_MARKS_FP8):
        func = mark(func)
    return func


def _wgrad_round_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


def _wgrad_create_fp8_tensor(shape: tuple, data_dtype: torch.dtype) -> torch.Tensor:
    elem_cnt = 1
    for s in shape:
        elem_cnt *= s
    return torch.randint(-1, 2, (elem_cnt,), dtype=torch.bfloat16, device="cuda").to(data_dtype).reshape(shape)


def _wgrad_create_fp4_tensor(logical_shape: tuple, packed_dim: int = -1) -> torch.Tensor:
    fp4_nibble_map = {0: 0x0, 1: 0x2, -1: 0xA}
    ndim = len(logical_shape)
    packed_dim = packed_dim % ndim
    assert logical_shape[packed_dim] % 2 == 0

    idx_tensor = torch.randint(-1, 2, logical_shape, dtype=torch.int8, device="cuda")
    nibbles = torch.zeros_like(idx_tensor, dtype=torch.uint8, device="cuda")
    for val, nib in fp4_nibble_map.items():
        nibbles[idx_tensor == val] = nib

    need_perm = packed_dim != ndim - 1
    if need_perm:
        perm_to_last = list(range(ndim))
        perm_to_last[packed_dim], perm_to_last[-1] = perm_to_last[-1], perm_to_last[packed_dim]
        nibbles = nibbles.permute(perm_to_last).contiguous()

    packed_uint8 = (nibbles[..., 1::2] << 4) | nibbles[..., ::2]
    tensor = packed_uint8.view(torch.float4_e2m1fn_x2)

    if need_perm:
        inv_perm = list(range(ndim))
        inv_perm[packed_dim], inv_perm[-1] = inv_perm[-1], inv_perm[packed_dim]
        tensor = tensor.permute(inv_perm)
    return tensor


def _wgrad_create_scale_tensor(shape: tuple, scale_dtype: torch.dtype) -> torch.Tensor:
    elem_cnt = 1
    for s in shape:
        elem_cnt *= s
    return torch.randint(1, 3, (elem_cnt,), dtype=torch.float32, device="cuda").to(scale_dtype).reshape(shape)


def _wgrad_to_blocked(scale_2d: torch.Tensor) -> torch.Tensor:
    rows, cols = scale_2d.shape
    row_blocks = ceil_div(rows, 128)
    col_blocks = ceil_div(cols, 4)
    padded_rows = row_blocks * 128
    padded_cols = col_blocks * 4

    if (rows, cols) == (padded_rows, padded_cols):
        padded = scale_2d
    else:
        padded = torch.zeros((padded_rows, padded_cols), dtype=scale_2d.dtype, device=scale_2d.device)
        padded[:rows, :cols] = scale_2d

    blocks = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def _wgrad_cat_byte_reinterpretable(tensors: list, dim: int = 0) -> torch.Tensor:
    first = tensors[0]
    if first.is_floating_point() and first.element_size() == 1:
        return torch.cat([t.view(torch.uint8) for t in tensors], dim=dim).view(first.dtype)
    return torch.cat(tensors, dim=dim)


def _wgrad_assemble_scales_2d2d(raw_scales: list, non_k_size: int) -> torch.Tensor:
    flat_parts = [_wgrad_to_blocked(scale) for scale in raw_scales]
    all_flat = _wgrad_cat_byte_reinterpretable(flat_parts, dim=0)
    return all_flat.reshape(_wgrad_round_up(non_k_size, 128), -1)


def grouped_gemm_wgrad_init(
    ab_dtype: torch.dtype,
    wgrad_dtype: torch.dtype,
    acc_dtype: torch.dtype,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    sf_vec_size: int,
    sf_dtype: torch.dtype,
) -> Dict[str, Any]:
    major, minor = torch.cuda.get_device_capability()
    compute_capability = major * 10 + minor
    if compute_capability < 100:
        pytest.skip(f"Environment not supported: requires compute capability >= 100, found {compute_capability}")

    return {
        "m": 384,
        "n": 640,
        "l": 2,
        "group_k_list": [256, 384],
        "ab_dtype": ab_dtype,
        "wgrad_dtype": wgrad_dtype,
        "acc_dtype": acc_dtype,
        "mma_tiler_mn": mma_tiler_mn,
        "cluster_shape_mn": cluster_shape_mn,
        "sf_vec_size": sf_vec_size,
        "sf_dtype": sf_dtype,
        "tolerance": 0.1 if ab_dtype != torch.float4_e2m1fn_x2 else 0.2,
    }


def allocate_grouped_gemm_wgrad_tensors(cfg: Dict[str, Any]) -> Dict[str, Any]:
    m, n, l = cfg["m"], cfg["n"], cfg["l"]
    group_k_list = cfg["group_k_list"]
    tokens_sum = sum(group_k_list)
    ab_dtype = cfg["ab_dtype"]
    sf_dtype = cfg["sf_dtype"]
    sf_vec_size = cfg["sf_vec_size"]

    is_fp4 = ab_dtype == torch.float4_e2m1fn_x2
    if is_fp4:
        a_tensor = _wgrad_create_fp4_tensor((m, tokens_sum), packed_dim=-1)
        b_tensor = _wgrad_create_fp4_tensor((tokens_sum, n), packed_dim=0)
        has_global_scale = sf_vec_size == 16
    else:
        a_tensor = _wgrad_create_fp8_tensor((m, tokens_sum), ab_dtype)
        b_tensor = _wgrad_create_fp8_tensor((tokens_sum, n), ab_dtype).T.contiguous().T
        has_global_scale = False

    offsets_tensor = torch.tensor([sum(group_k_list[: i + 1]) for i in range(l)], dtype=torch.int32, device="cuda")
    raw_scale_a = [_wgrad_create_scale_tensor((m, ceil_div(k_val, sf_vec_size)), sf_dtype) for k_val in group_k_list]
    raw_scale_b = [_wgrad_create_scale_tensor((n, ceil_div(k_val, sf_vec_size)), sf_dtype) for k_val in group_k_list]
    sfa_tensor = _wgrad_assemble_scales_2d2d(raw_scale_a, m)
    sfb_tensor = _wgrad_assemble_scales_2d2d(raw_scale_b, n)

    global_scale_a = torch.randint(1, 3, (l,), dtype=torch.float32, device="cuda") if has_global_scale else None
    global_scale_b = torch.randint(1, 3, (l,), dtype=torch.float32, device="cuda") if has_global_scale else None

    ref_result = None
    try:
        from torch.nn.functional import scaled_grouped_mm, ScalingType, SwizzleType

        scale_a_arg = sfa_tensor
        scale_b_arg = sfb_tensor
        recipe_a = ScalingType.BlockWise1x32
        recipe_b = ScalingType.BlockWise1x32
        if has_global_scale:
            scale_a_arg = [sfa_tensor, global_scale_a]
            scale_b_arg = [sfb_tensor, global_scale_b]
            recipe_a = [ScalingType.BlockWise1x16, ScalingType.TensorWise]
            recipe_b = [ScalingType.BlockWise1x16, ScalingType.TensorWise]

        ref_result = scaled_grouped_mm(
            a_tensor,
            b_tensor,
            scale_a=scale_a_arg,
            scale_recipe_a=recipe_a,
            scale_b=scale_b_arg,
            scale_recipe_b=recipe_b,
            swizzle_a=SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=SwizzleType.SWIZZLE_32_4_4,
            offs=offsets_tensor,
            output_dtype=cfg["wgrad_dtype"],
        )
    except (ImportError, ValueError, RuntimeError) as exc:
        if not isinstance(exc, ImportError) and "No gemm implementation was found" not in str(exc):
            raise
        ref_result = None

    return {
        "a_tensor": a_tensor,
        "b_tensor": b_tensor,
        "sfa_tensor": sfa_tensor,
        "sfb_tensor": sfb_tensor,
        "offsets_tensor": offsets_tensor,
        "global_scale_a": global_scale_a,
        "global_scale_b": global_scale_b,
        "ref_result": ref_result,
    }


def allocate_grouped_gemm_wgrad_output(cfg: Dict[str, Any], accumulate_on_output: bool = False) -> torch.Tensor:
    shape = (cfg["l"], cfg["m"], cfg["n"])
    if accumulate_on_output:
        return torch.zeros(shape, dtype=cfg["wgrad_dtype"], device="cuda")
    return torch.empty(shape, dtype=cfg["wgrad_dtype"], device="cuda")


def check_ref_grouped_gemm_wgrad(wgrad_tensor: torch.Tensor, ref_result: torch.Tensor, tolerance: float = 0.1) -> None:
    if ref_result is None:
        return
    torch.testing.assert_close(
        wgrad_tensor.to(torch.float32).cpu(),
        ref_result.to(torch.float32).cpu(),
        atol=tolerance,
        rtol=tolerance,
    )
