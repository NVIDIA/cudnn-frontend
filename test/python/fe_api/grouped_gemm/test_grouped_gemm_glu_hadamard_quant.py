"""Tests for grouped GEMM GLU + Hadamard + Quant forward fusion (SM100+)."""

from typing import Dict, Optional

import pytest
import torch

from cudnn.gemm.cutedsl.grouped.glu_hadamard_quant.rht_utils import HADAMARD_SIZE
from test_low_precision_matmul import float4_e2m1fn_x2_to_float32
from test_utils import torch_fork_set_rng
from fe_api.grouped_gemm.test_discrete_grouped_gemm_swiglu_utils import allocate_discrete_input_tensors
from fe_api.grouped_gemm.test_grouped_gemm_swiglu_utils import allocate_grouped_gemm_input_tensors, grouped_gemm_swiglu_init
from fe_api.grouped_gemm.test_grouped_gemm_wgrad_utils import _skip_unless_e5m3_supported
from fe_api.test_fe_api_utils import (
    DYNAMIC_SHAPES_M_VALUES,
    reencode_sf_tensor_as_ue5m3,
    ue5m3_bytes_to_fp32,
)

import cutlass

# The glu_hadamard_quant kernel references cutlass.FloatNV8E5M3FNU
# unconditionally at compile time (moe_blockscaled_grouped_gemm_glu_hadamard_quant.py),
# and that dtype only exists in cutlass-dsl >= 4.8. Skip the whole file on
# older builds instead of failing every test with an AttributeError.
pytestmark = pytest.mark.skipif(
    not hasattr(cutlass, "FloatNV8E5M3FNU"),
    reason="glu_hadamard_quant kernels require cutlass-dsl >= 4.8 (cutlass.FloatNV8E5M3FNU)",
)

FP4_EXECUTION_CASES = [
    (torch.float4_e2m1fn_x2, torch.float8_e4m3fn, 16),
    (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu, 16),
]


def _make_cfg(request, *, ab_dtype, sf_dtype, sf_vec_size, enable_bias=False) -> Dict:
    return grouped_gemm_swiglu_init(
        request,
        ab_dtype=ab_dtype,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        vector_f32=False,
        discrete_col_sfd=False,
        enable_bias=enable_bias,
    )


# =============================================================================
# Reference implementations
# =============================================================================


def _run_grouped_gemm_glu_ref(
    inputs: Dict,
    act_func: str,
    glu_alpha: Optional[float] = None,
    glu_limit: Optional[float] = None,
) -> Dict:
    n, _, l = inputs["b_ref"].shape
    n_out = n // 2
    valid_m = inputs["valid_m"]
    aligned_group_m_list = inputs["aligned_group_m_list"]

    ref = torch.empty((1, valid_m, n), dtype=torch.float32, device=inputs["a_ref"].device)
    start = 0
    for i, group_m in enumerate(aligned_group_m_list):
        end = start + group_m
        res_a = torch.einsum("mk,mk->mk", inputs["a_ref"][start:end, :, 0].to(torch.float32), inputs["sfa_ref"][start:end, :, 0].to(torch.float32))
        res_b = torch.einsum("nk,nk->nk", inputs["b_ref"][:, :, i].to(torch.float32), inputs["sfb_ref"][:, :, i].to(torch.float32))
        ref[0, start:end, :] = torch.einsum("mk,nk->mn", res_a, res_b)
        start = end
    ref = ref.permute((1, 2, 0))

    start = 0
    for i, group_m in enumerate(aligned_group_m_list):
        end = start + group_m
        ref[start:end, :, 0] = ref[start:end, :, 0] * inputs["alpha_tensor"][i].item()
        start = end

    if inputs.get("bias_tensor") is not None:
        start = 0
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            ref[start:end, :, 0] = ref[start:end, :, 0] + inputs["bias_tensor"][:, i].unsqueeze(0).to(torch.float32)
            start = end

    group = 32
    assert n % group == 0, "N must be divisible by 32 for GLU block grouping"
    num_blocks = n // group
    assert num_blocks % 2 == 0, "Number of 32-col blocks must be even"

    cols = torch.arange(n, device=ref.device, dtype=torch.long)
    block_cols = cols.view(num_blocks, group)
    gate_idx = block_cols[0::2].reshape(-1)
    up_idx = block_cols[1::2].reshape(-1)
    ref_gate = ref.index_select(1, gate_idx)
    ref_up = ref.index_select(1, up_idx)

    # The kernel clamps BOTH gate and up to [-glu_limit, glu_limit] when set
    # (after the C store, before the activation).
    if glu_limit is not None:
        ref_gate = torch.clamp(ref_gate, min=-glu_limit, max=glu_limit)
        ref_up = torch.clamp(ref_up, min=-glu_limit, max=glu_limit)

    if act_func == "swiglu":
        ref_after_glu = ref_up * (ref_gate * torch.sigmoid(ref_gate))
    elif act_func == "geglu":
        ref_after_glu = (ref_up + 1.0) * ref_gate * torch.sigmoid(1.702 * ref_gate)
    else:
        raise ValueError(f"Unsupported act_func {act_func}")

    ref_after_glu = ref_after_glu * inputs["prob_tensor"].expand(-1, n_out, -1)
    # glu_alpha is a final output scale in this kernel.
    if glu_alpha is not None and glu_alpha != 1.0:
        ref_after_glu = ref_after_glu * glu_alpha
    return {"c_ref": ref.clone(), "d_ref": ref_after_glu}


def _hadamard16(device: torch.device) -> torch.Tensor:
    """Sylvester (natural FWHT order) 16x16 Hadamard matrix."""
    h = torch.ones((1, 1), dtype=torch.float32, device=device)
    for _ in range(4):
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


def _rht_ref(d_ref: torch.Tensor, rowwise: bool) -> torch.Tensor:
    """Orthonormal H16 transform of the bf16-rounded D reference, at D's (m, f)
    orientation. rowwise transforms 16-feature blocks per token; colwise
    transforms 16-token blocks per feature. Returns (m, f) bf16."""
    x = d_ref[:, :, 0].to(torch.bfloat16).to(torch.float32)
    h = _hadamard16(x.device)
    m, f = x.shape
    if rowwise:
        blocks = x.view(m, f // HADAMARD_SIZE, HADAMARD_SIZE)
        out = torch.einsum("ij,mbj->mbi", h, blocks).reshape(m, f)
    else:
        blocks = x.view(m // HADAMARD_SIZE, HADAMARD_SIZE, f)
        out = torch.einsum("ij,bjf->bif", h, blocks).reshape(m, f)
    return (out * 0.25).to(torch.bfloat16)


def _nvfp4_sf_ref(x: torch.Tensor, norm_const: float) -> torch.Tensor:
    """e4m3 block scales of x (rows, cols) with (1, 16) blocks along the last dim,
    replicating the kernel's op order: amax * (1/6), then * norm_const."""
    xf = x.to(torch.bfloat16).to(torch.float32)
    rows, cols = xf.shape
    amax = xf.view(rows, cols // HADAMARD_SIZE, HADAMARD_SIZE).abs().amax(dim=-1)
    return ((amax * (1.0 / 6.0)) * norm_const).to(torch.float8_e4m3fn)


def _swizzled_sf_to_flat(sf_tensor: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Gather M32x4xrm_K4xrk_L scale storage into logical (rows, cols/16)."""
    sf_cols = (cols + HADAMARD_SIZE - 1) // HADAMARD_SIZE
    row_idx = torch.arange(rows, device=sf_tensor.device, dtype=torch.long).view(rows, 1)
    col_idx = torch.arange(sf_cols, device=sf_tensor.device, dtype=torch.long).view(1, sf_cols)
    return sf_tensor[
        row_idx % 32,
        (row_idx // 32) % 4,
        row_idx // 128,
        col_idx % 4,
        col_idx // 4,
        0,
    ]


def _check_nvfp4_output(
    values: torch.Tensor,
    sf_tensor: torch.Tensor,
    ref: torch.Tensor,
    norm_const: float,
    name: str,
) -> None:
    """Check unpacked e2m1 values (rows, cols) + e4m3 scales (rows, cols/16),
    with (1, 16) quantization blocks along the last dim, against the f32
    reference (rows, cols): scales against the reference scale computation (one
    e4m3 step of slack), data via a per-block dequantization error bound (half
    the widest e2m1 grid gap, plus saturation headroom)."""
    ref_bf16 = ref.to(torch.bfloat16).to(torch.float32)
    sf_ref = _nvfp4_sf_ref(ref, norm_const)
    sf_flat = _swizzled_sf_to_flat(sf_tensor, ref.shape[0], ref.shape[1]) if sf_tensor.dim() == 6 else sf_tensor
    torch.testing.assert_close(
        sf_flat.float().cpu(),
        sf_ref.float().cpu(),
        atol=1e-2,
        rtol=0.14,
        msg=lambda m: f"{name} block scales mismatch\n{m}",
    )

    decode_scale = sf_flat.float().repeat_interleave(HADAMARD_SIZE, dim=1) / norm_const
    dequant = values * decode_scale
    err = (dequant - ref_bf16).abs()
    bound = 1.5 * decode_scale + 5e-2
    bad = err > bound
    assert not bad.any(), f"{name}: {int(bad.sum())} dequantized elements exceed the quantization error bound (max err {err[bad].max().item():.4f})"


# =============================================================================
# Output checking
# =============================================================================


def _check_outputs(
    inputs: Dict,
    outputs: Dict,
    cfg: Dict,
    *,
    act_func: str,
    rht_output: bool,
    rht_rowwise: bool,
    glu_alpha: Optional[float] = None,
    glu_limit: Optional[float] = None,
    norm_const: float = 1.0,
    rht_norm_const: float = 1.0,
    ref_tensors: Optional[Dict] = None,
    sf_fp8_dtype_override: Optional[str] = None,
) -> None:
    if ref_tensors is None:
        ref_tensors = _run_grouped_gemm_glu_ref(inputs, act_func, glu_alpha=glu_alpha, glu_limit=glu_limit)
    valid_m = inputs["valid_m"]
    c_ref = ref_tensors["c_ref"]
    d_ref = ref_tensors["d_ref"]

    torch.testing.assert_close(
        outputs["c_tensor"][:valid_m].cpu().float(),
        c_ref.cpu().to(cfg["c_dtype"]).to(torch.float32),
        atol=1e-1,
        rtol=1e-2,
    )

    if outputs["sfd_tensor"] is None:
        torch.testing.assert_close(
            outputs["d_tensor"][:valid_m].cpu().float(),
            d_ref.cpu().to(torch.bfloat16).to(torch.float32),
            atol=1e-1,
            rtol=1e-2,
        )
    else:
        sfd_tensor = outputs["sfd_tensor"]
        if sf_fp8_dtype_override == "e5m3":
            sfd_tensor = ue5m3_bytes_to_fp32(sfd_tensor)
        _check_nvfp4_output(
            float4_e2m1fn_x2_to_float32(outputs["d_tensor"][:valid_m, :, 0].view(torch.uint8).cpu()),
            sfd_tensor.cpu(),
            d_ref[:, :, 0].cpu(),
            norm_const,
            "D",
        )

    if not rht_output:
        assert outputs["rht_tensor"] is None
        assert outputs["sfrht_tensor"] is None
        return

    rht_ref = _rht_ref(d_ref.cpu(), rht_rowwise)
    if outputs["sfrht_tensor"] is None:
        torch.testing.assert_close(
            outputs["rht_tensor"][:valid_m, :, 0].cpu().float(),
            rht_ref.float(),
            atol=1e-1,
            rtol=1e-2,
        )
    elif rht_rowwise:
        sfrht_tensor = outputs["sfrht_tensor"]
        if sf_fp8_dtype_override == "e5m3":
            sfrht_tensor = ue5m3_bytes_to_fp32(sfrht_tensor)
        _check_nvfp4_output(
            float4_e2m1fn_x2_to_float32(outputs["rht_tensor"][:valid_m, :, 0].view(torch.uint8).cpu()),
            sfrht_tensor.cpu(),
            rht_ref.float(),
            rht_norm_const,
            "RHT",
        )
    else:
        # Colwise: the packed data stays at D's (m, f) orientation (nibbles pair
        # adjacent features), but quantization blocks are (16, 1) token blocks,
        # so check through the transposed unpacked values and the swizzled
        # SF(N_out, valid_m) scale domain.
        sfrht_tensor = outputs["sfrht_tensor"]
        if sf_fp8_dtype_override == "e5m3":
            sfrht_tensor = ue5m3_bytes_to_fp32(sfrht_tensor)
        _check_nvfp4_output(
            float4_e2m1fn_x2_to_float32(outputs["rht_tensor"][:valid_m, :, 0].view(torch.uint8).cpu()).t(),
            sfrht_tensor.cpu(),
            rht_ref.float().t(),
            rht_norm_const,
            "RHT",
        )


# =============================================================================
# Runners
# =============================================================================


def _run_wrapper(
    request,
    *,
    ab_dtype,
    sf_dtype,
    sf_vec_size,
    act_func="swiglu",
    enable_bias=False,
    d_dtype=torch.bfloat16,
    rht_output=True,
    rht_dtype=torch.bfloat16,
    rht_rowwise=False,
    glu_alpha=None,
    glu_limit=None,
    sf_fp8_dtype_override=None,
):
    cfg = _make_cfg(request, ab_dtype=ab_dtype, sf_dtype=sf_dtype, sf_vec_size=sf_vec_size, enable_bias=enable_bias)
    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
        enable_bias=enable_bias,
    )

    # NVFP4 global encode scales, derived from the reference like production
    # does from calibration (2688 = 448 * 6): without them the e4m3 block
    # scales saturate on this test data's magnitude.
    ref_tensors = _run_grouped_gemm_glu_ref(inputs, act_func, glu_alpha=glu_alpha, glu_limit=glu_limit)
    norm_const = 1.0
    rht_norm_const = 1.0
    if d_dtype == torch.float4_e2m1fn_x2:
        norm_const = 2688.0 / ref_tensors["d_ref"].to(torch.bfloat16).float().abs().max().item()
    if rht_output and rht_dtype == torch.float4_e2m1fn_x2:
        rht_ref = _rht_ref(ref_tensors["d_ref"].cpu(), rht_rowwise)
        rht_norm_const = 2688.0 / rht_ref.float().abs().max().item()

    if sf_fp8_dtype_override == "e5m3":
        # Rewrite the scale bytes as UE5M3 in place; values are exact in both
        # formats so the fp32 reference stays valid.
        reencode_sf_tensor_as_ue5m3(inputs["sfa_tensor"])
        reencode_sf_tensor_as_ue5m3(inputs["sfb_tensor"])

    from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100

    outputs = grouped_gemm_glu_hadamard_quant_wrapper_sm100(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        bias_tensor=inputs["bias_tensor"],
        acc_dtype=cfg["acc_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=d_dtype,
        cd_major=cfg["cd_major"],
        rht_output=rht_output,
        rht_dtype=rht_dtype,
        rht_rowwise=rht_rowwise,
        glu_alpha=glu_alpha,
        glu_limit=glu_limit,
        norm_const=norm_const,
        rht_norm_const=rht_norm_const,
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        sf_fp8_dtype_override=sf_fp8_dtype_override,
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        act_func=act_func,
    )

    _check_outputs(
        inputs,
        outputs,
        cfg,
        act_func=act_func,
        rht_output=rht_output,
        rht_rowwise=rht_rowwise,
        glu_alpha=glu_alpha,
        glu_limit=glu_limit,
        norm_const=norm_const,
        rht_norm_const=rht_norm_const,
        ref_tensors=ref_tensors,
        sf_fp8_dtype_override=sf_fp8_dtype_override,
    )


def _run_compile_execute(request, *, ab_dtype, sf_dtype, sf_vec_size, act_func="swiglu", sf_fp8_dtype_override=None):
    cfg = _make_cfg(request, ab_dtype=ab_dtype, sf_dtype=sf_dtype, sf_vec_size=sf_vec_size)
    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
        enable_bias=False,
    )

    if sf_fp8_dtype_override == "e5m3":
        # Rewrite the scale bytes as UE5M3 in place; values are exact in both
        # formats so the fp32 reference stays valid.
        reencode_sf_tensor_as_ue5m3(inputs["sfa_tensor"])
        reencode_sf_tensor_as_ue5m3(inputs["sfb_tensor"])

    valid_m = inputs["valid_m"]
    n = cfg["n"]
    n_out = n // 2
    device = inputs["a_tensor"].device

    def alloc_n_major(rows, cols, dtype):
        return torch.empty_strided((rows, cols, 1), (cols, 1, rows * cols), dtype=dtype, device=device)

    outputs = {
        "c_tensor": alloc_n_major(valid_m, n, cfg["c_dtype"]),
        "d_tensor": alloc_n_major(valid_m, n_out, torch.bfloat16),
        "sfd_tensor": None,
        "rht_tensor": alloc_n_major(valid_m, n_out, torch.bfloat16),
        "sfrht_tensor": None,
    }

    from cudnn import GroupedGemmGluHadamardQuantSm100

    api = GroupedGemmGluHadamardQuantSm100(
        sample_a=inputs["a_tensor"],
        sample_b=inputs["b_tensor"],
        sample_c=outputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_sfb=inputs["sfb_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_prob=inputs["prob_tensor"],
        sample_rht=outputs["rht_tensor"],
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        sf_fp8_dtype_override=sf_fp8_dtype_override,
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        act_func=act_func,
    )
    api.check_support()
    api.compile()
    api.execute(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        c_tensor=outputs["c_tensor"],
        d_tensor=outputs["d_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        rht_tensor=outputs["rht_tensor"],
    )

    _check_outputs(
        inputs,
        outputs,
        cfg,
        act_func=act_func,
        rht_output=True,
        rht_rowwise=False,
        sf_fp8_dtype_override=sf_fp8_dtype_override,
    )


def _run_discrete_wrapper(request, *, ab_dtype, sf_dtype, sf_vec_size, act_func="swiglu", sf_fp8_dtype_override=None):
    cfg = _make_cfg(request, ab_dtype=ab_dtype, sf_dtype=sf_dtype, sf_vec_size=sf_vec_size)
    inputs = allocate_discrete_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        num_experts=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
    )
    inputs["b_ref"] = torch.cat(inputs["b_ref_list"], dim=2)
    inputs["sfb_ref"] = torch.cat(inputs["sfb_ref_list"], dim=2)

    from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100

    outputs = grouped_gemm_glu_hadamard_quant_wrapper_sm100(
        a_tensor=inputs["a_tensor"],
        b_ptrs=inputs["b_ptrs_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_ptrs=inputs["sfb_ptrs_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        n=cfg["n"],
        b_dtype=inputs["b_list"][0].dtype,
        b_major=cfg["b_major"],
        bias_tensor=inputs["bias_tensor"],
        acc_dtype=cfg["acc_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=torch.bfloat16,
        cd_major=cfg["cd_major"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        sf_fp8_dtype_override=sf_fp8_dtype_override,
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        act_func=act_func,
    )

    _check_outputs(
        inputs,
        outputs,
        cfg,
        act_func=act_func,
        rht_output=True,
        rht_rowwise=False,
        sf_fp8_dtype_override=sf_fp8_dtype_override,
    )


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("ab_dtype,sf_dtype,sf_vec_size", FP4_EXECUTION_CASES)
def test_grouped_gemm_glu_hadamard_quant_compile_execute_fp4(request, ab_dtype, sf_dtype, sf_vec_size):
    _run_compile_execute(
        request,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("act_func", ["swiglu", "geglu"])
def test_grouped_gemm_glu_hadamard_quant_wrapper_fp4(request, act_func):
    _run_wrapper(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e4m3fn,
        sf_vec_size=16,
        act_func=act_func,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_quant_wrapper_rowwise(request):
    _run_wrapper(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e4m3fn,
        sf_vec_size=16,
        rht_rowwise=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_quant_wrapper_no_rht(request):
    _run_wrapper(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e4m3fn,
        sf_vec_size=16,
        rht_output=False,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_quant_wrapper_with_bias(request):
    _run_wrapper(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e8m0fnu,
        sf_vec_size=16,
        enable_bias=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_quant_wrapper_glu_alpha_limit(request):
    _run_wrapper(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e4m3fn,
        sf_vec_size=16,
        glu_alpha=1.702,
        glu_limit=7.0,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_quant_wrapper_quant_d(request):
    _run_wrapper(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e4m3fn,
        sf_vec_size=16,
        d_dtype=torch.float4_e2m1fn_x2,
        rht_output=False,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("rht_rowwise", [False, True])
def test_grouped_gemm_glu_hadamard_quant_wrapper_quant_rht(request, rht_rowwise):
    _run_wrapper(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e4m3fn,
        sf_vec_size=16,
        rht_dtype=torch.float4_e2m1fn_x2,
        rht_rowwise=rht_rowwise,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_quant_wrapper_quant_full(request):
    _run_wrapper(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e4m3fn,
        sf_vec_size=16,
        d_dtype=torch.float4_e2m1fn_x2,
        rht_dtype=torch.float4_e2m1fn_x2,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("rht_rowwise", [False, True], ids=["colwise", "rowwise"])
def test_grouped_gemm_glu_hadamard_quant_wrapper_quant_rht_e5m3(request, rht_rowwise):
    """quant_rht with the input block scales carried as UE5M3 bytes in e4m3 storage."""
    _skip_unless_e5m3_supported()
    _run_wrapper(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e4m3fn,
        sf_vec_size=16,
        rht_dtype=torch.float4_e2m1fn_x2,
        rht_rowwise=rht_rowwise,
        sf_fp8_dtype_override="e5m3",
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_quant_wrapper_quant_full_e5m3(request):
    """quant_full with the input block scales carried as UE5M3 bytes in e4m3 storage."""
    _skip_unless_e5m3_supported()
    _run_wrapper(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e4m3fn,
        sf_vec_size=16,
        d_dtype=torch.float4_e2m1fn_x2,
        rht_dtype=torch.float4_e2m1fn_x2,
        sf_fp8_dtype_override="e5m3",
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_quant_compile_execute_e5m3(request):
    """Class-API compile/execute path with e5m3-reinterpreted input scales."""
    _skip_unless_e5m3_supported()
    _run_compile_execute(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e4m3fn,
        sf_vec_size=16,
        sf_fp8_dtype_override="e5m3",
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_quant_e5m3_is_not_cached_as_e4m3(request):
    """sf_fp8_dtype_override must take part in the compile cache key.

    Identical scale-factor bytes decode to different values under E4M3 and
    UE5M3, so if sf_fp8_dtype_override were omitted from the key the second
    call would reuse the first kernel and silently return E4M3 results.
    """
    _skip_unless_e5m3_supported()

    # One problem, one set of scale-factor bytes, two interpretations. Any
    # difference in the output can only come from sf_fp8_dtype_override.
    cfg = _make_cfg(request, ab_dtype=torch.float4_e2m1fn_x2, sf_dtype=torch.float8_e4m3fn, sf_vec_size=16)
    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
        enable_bias=False,
    )

    from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100

    def run(sf_fp8_dtype_override):
        outputs = grouped_gemm_glu_hadamard_quant_wrapper_sm100(
            a_tensor=inputs["a_tensor"],
            b_tensor=inputs["b_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            sfb_tensor=inputs["sfb_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            prob_tensor=inputs["prob_tensor"],
            bias_tensor=inputs["bias_tensor"],
            acc_dtype=cfg["acc_dtype"],
            c_dtype=cfg["c_dtype"],
            d_dtype=torch.bfloat16,
            cd_major=cfg["cd_major"],
            rht_output=True,
            rht_dtype=torch.bfloat16,
            rht_rowwise=False,
            mma_tiler_mn=cfg["mma_tiler_mn"],
            cluster_shape_mn=cfg["cluster_shape_mn"],
            sf_vec_size=cfg["sf_vec_size"],
            sf_fp8_dtype_override=sf_fp8_dtype_override,
            vector_f32=cfg["vector_f32"],
            m_aligned=cfg["m_aligned"],
            act_func="swiglu",
        )
        return outputs["d_tensor"].float().clone()

    d_e4m3 = run(None)
    d_e5m3 = run("e5m3")

    assert not torch.equal(
        d_e5m3, d_e4m3
    ), "e5m3 and e4m3 produced identical output from identical scale-factor bytes; sf_fp8_dtype_override is likely missing from the compile cache key"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    "sf_fp8_dtype_override,overrides,expected",
    [
        pytest.param(
            "e5m3",
            dict(sf_dtype=torch.float8_e8m0fnu),
            "requires torch.float8_e4m3fn scale-factor storage",
            id="e8m0_carrier",
        ),
        pytest.param("e4m3", {}, "sf_fp8_dtype_override must be", id="e4m3_is_not_an_override"),
        pytest.param("e5m2", {}, "sf_fp8_dtype_override must be", id="unknown_format"),
    ],
)
def test_grouped_gemm_glu_hadamard_quant_rejects_unsupported_sf_fp8_dtype(request, sf_fp8_dtype_override, overrides, expected):
    """e5m3 is only reachable through the FP4xFP4 atom with e4m3-carried scales."""
    if sf_fp8_dtype_override == "e5m3":
        _skip_unless_e5m3_supported()

    # Construct kernel config
    cfg_kwargs = dict(
        ab_dtype=torch.float4_e2m1fn_x2,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=16,
        sf_dtype=torch.float8_e4m3fn,
        vector_f32=False,
        discrete_col_sfd=False,
        enable_bias=False,
    )
    cfg_kwargs.update(overrides)
    cfg = grouped_gemm_swiglu_init(request, **cfg_kwargs)

    # Allocate input tensors
    inputs = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
        enable_bias=False,
    )

    from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100

    # Check that calling kernel API triggers exception
    with pytest.raises(ValueError, match=expected):
        grouped_gemm_glu_hadamard_quant_wrapper_sm100(
            a_tensor=inputs["a_tensor"],
            b_tensor=inputs["b_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            sfb_tensor=inputs["sfb_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            prob_tensor=inputs["prob_tensor"],
            bias_tensor=inputs["bias_tensor"],
            acc_dtype=cfg["acc_dtype"],
            c_dtype=cfg["c_dtype"],
            d_dtype=cfg["d_dtype"],
            cd_major=cfg["cd_major"],
            mma_tiler_mn=cfg["mma_tiler_mn"],
            cluster_shape_mn=cfg["cluster_shape_mn"],
            sf_vec_size=cfg["sf_vec_size"],
            sf_fp8_dtype_override=sf_fp8_dtype_override,
            vector_f32=cfg["vector_f32"],
            m_aligned=cfg["m_aligned"],
            act_func="swiglu",
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_quant_wrapper_discrete_fp4(request):
    _run_discrete_wrapper(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e8m0fnu,
        sf_vec_size=16,
        act_func="swiglu",
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("group_m_list", [[256, 256, 256, 256], DYNAMIC_SHAPES_M_VALUES])
def test_grouped_gemm_glu_hadamard_quant_wrapper_cache_dynamic_m_smoke(request, monkeypatch, group_m_list):
    from cudnn import grouped_gemm_glu_hadamard_quant_wrapper_sm100
    import cudnn.gemm.cutedsl.grouped.glu_hadamard_quant.api as grouped_gemm_glu_hadamard_quant_api

    grouped_gemm_glu_hadamard_quant_api._cache_of_GroupedGemmGluHadamardQuantSm100Objects.clear()

    compile_count = {"value": 0}

    def counted_compile(self):
        compile_count["value"] += 1
        return None

    monkeypatch.setattr(grouped_gemm_glu_hadamard_quant_api.GroupedGemmGluHadamardQuantSm100, "compile", counted_compile)
    monkeypatch.setattr(grouped_gemm_glu_hadamard_quant_api.GroupedGemmGluHadamardQuantSm100, "check_support", lambda self: True)
    monkeypatch.setattr(grouped_gemm_glu_hadamard_quant_api.GroupedGemmGluHadamardQuantSm100, "execute", lambda self, **kwargs: None)

    cfg = _make_cfg(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e8m0fnu,
        sf_vec_size=16,
    )
    cfg["group_m_list"] = list(group_m_list)
    cfg["l"] = len(group_m_list)

    for _ in range(2):
        inputs = allocate_grouped_gemm_input_tensors(
            n=cfg["n"],
            k=cfg["k"],
            l=cfg["l"],
            group_m_list=cfg["group_m_list"],
            ab_dtype=cfg["ab_dtype"],
            sf_dtype=cfg["sf_dtype"],
            sf_vec_size=cfg["sf_vec_size"],
            m_aligned=cfg["m_aligned"],
            b_major=cfg["b_major"],
            enable_bias=False,
        )
        grouped_gemm_glu_hadamard_quant_wrapper_sm100(
            a_tensor=inputs["a_tensor"],
            b_tensor=inputs["b_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            sfb_tensor=inputs["sfb_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            prob_tensor=inputs["prob_tensor"],
            acc_dtype=cfg["acc_dtype"],
            c_dtype=cfg["c_dtype"],
            d_dtype=cfg["d_dtype"],
            cd_major=cfg["cd_major"],
            mma_tiler_mn=cfg["mma_tiler_mn"],
            cluster_shape_mn=cfg["cluster_shape_mn"],
            sf_vec_size=cfg["sf_vec_size"],
            vector_f32=cfg["vector_f32"],
            m_aligned=cfg["m_aligned"],
            act_func="swiglu",
        )

    assert compile_count["value"] == 1
    assert len(grouped_gemm_glu_hadamard_quant_api._cache_of_GroupedGemmGluHadamardQuantSm100Objects) == 1
    grouped_gemm_glu_hadamard_quant_api._cache_of_GroupedGemmGluHadamardQuantSm100Objects.clear()
