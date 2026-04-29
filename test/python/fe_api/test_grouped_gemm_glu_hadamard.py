"""Tests for grouped GEMM GLU + Hadamard forward fusion (SM100+)."""

from typing import Dict

import pytest
import torch

from test_utils import torch_fork_set_rng
from fe_api.test_fe_api_utils import DYNAMIC_SHAPES_M_VALUES, compute_reference_amax
from fe_api.test_grouped_gemm_swiglu_utils import allocate_grouped_gemm_input_tensors, grouped_gemm_swiglu_init

FP4_EXECUTION_CASES = [
    (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu, 16),
    (torch.float4_e2m1fn_x2, torch.float8_e4m3fn, 16),
    (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu, 32),
    (torch.uint8, torch.float8_e8m0fnu, 16),
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


def _apply_hadamard(d_ref: torch.Tensor) -> torch.Tensor:
    from cudnn.grouped_gemm.grouped_gemm_glu_hadamard.hadamard_utils import HADAMARD_SIZE, hadamard_matrix

    valid_m, n_out, _ = d_ref.shape
    hadamard = hadamard_matrix(HADAMARD_SIZE, dtype=torch.float32, device=d_ref.device)
    ref_view = d_ref.squeeze(-1).to(torch.bfloat16).to(torch.float32).view(valid_m, n_out // HADAMARD_SIZE, HADAMARD_SIZE)
    return (ref_view @ hadamard).view(valid_m, n_out, 1)


def _run_grouped_gemm_glu_ref(inputs: Dict, act_func: str) -> Dict:
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

    if act_func == "swiglu":
        ref_after_glu = ref_up * (ref_gate * torch.sigmoid(ref_gate))
    elif act_func == "geglu":
        ref_gate = torch.clamp(ref_gate, max=7.0)
        ref_up = torch.clamp(ref_up, min=-7.0, max=7.0)
        ref_after_glu = (ref_up + 1.0) * ref_gate * torch.sigmoid(1.702 * ref_gate)
    else:
        raise ValueError(f"Unsupported act_func {act_func}")

    ref_after_glu = ref_after_glu * inputs["prob_tensor"].expand(-1, n_out, -1)
    return {"c_ref": ref.clone(), "d_ref": ref_after_glu}


def _check_reference(inputs: Dict, outputs: Dict, cfg: Dict, *, act_func: str) -> None:
    ref_tensors = _run_grouped_gemm_glu_ref(inputs, act_func)

    torch.testing.assert_close(
        outputs["c_tensor"][: inputs["valid_m"]].cpu().float(),
        ref_tensors["c_ref"].cpu().to(cfg["c_dtype"]).to(torch.float32),
        atol=1e-1,
        rtol=1e-2,
    )

    d_hadamard_ref = _apply_hadamard(ref_tensors["d_ref"])

    torch.testing.assert_close(
        outputs["d_tensor"][: inputs["valid_m"]].cpu().float(),
        d_hadamard_ref.cpu().to(cfg["d_dtype"]).to(torch.float32),
        atol=1e-1,
        rtol=1e-2,
    )

    if outputs["amax_tensor"] is not None:
        amax_ref = torch.empty((cfg["l"],), dtype=torch.float32)
        start = 0
        for i, group_m in enumerate(inputs["aligned_group_m_list"]):
            end = start + group_m
            amax_ref[i] = compute_reference_amax(d_hadamard_ref[start:end, :, 0].clone())
            start = end
        torch.testing.assert_close(
            outputs["amax_tensor"].cpu().reshape(-1),
            amax_ref,
            atol=1e-1,
            rtol=1e-2,
        )


def _allocate_outputs(inputs: Dict, cfg: Dict) -> Dict:
    valid_m = inputs["valid_m"]
    n = cfg["n"]
    n_out = n // 2
    l = cfg["l"]
    device = inputs["a_tensor"].device

    return {
        "c_tensor": torch.empty_strided((valid_m, n, 1), (n, 1, valid_m * n), dtype=cfg["c_dtype"], device=device),
        "d_tensor": torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=cfg["d_dtype"], device=device),
        "amax_tensor": torch.full((l, 1), float("-inf"), dtype=torch.float32, device=device),
    }


def _run_compile_execute(request, *, ab_dtype, sf_dtype, sf_vec_size, act_func="swiglu", enable_bias=False):
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
    outputs = _allocate_outputs(inputs, cfg)

    from cudnn import GroupedGemmGluHadamardSm100

    api = GroupedGemmGluHadamardSm100(
        sample_a=inputs["a_tensor"],
        sample_b=inputs["b_tensor"],
        sample_c=outputs["c_tensor"],
        sample_d=outputs["d_tensor"],
        sample_sfa=inputs["sfa_tensor"],
        sample_sfb=inputs["sfb_tensor"],
        sample_padded_offsets=inputs["padded_offsets_tensor"],
        sample_alpha=inputs["alpha_tensor"],
        sample_prob=inputs["prob_tensor"],
        sample_amax=outputs["amax_tensor"],
        sample_bias=inputs["bias_tensor"],
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
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
        amax_tensor=outputs["amax_tensor"],
        bias_tensor=inputs["bias_tensor"],
    )

    _check_reference(inputs, outputs, cfg, act_func=act_func)


def _run_wrapper(request, *, ab_dtype, sf_dtype, sf_vec_size, act_func="swiglu", enable_bias=False):
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

    from cudnn import grouped_gemm_glu_hadamard_wrapper_sm100

    outputs = grouped_gemm_glu_hadamard_wrapper_sm100(
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
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        act_func=act_func,
    )

    _check_reference(inputs, outputs, cfg, act_func=act_func)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("ab_dtype,sf_dtype,sf_vec_size", FP4_EXECUTION_CASES)
def test_grouped_gemm_glu_hadamard_compile_execute_fp4(request, ab_dtype, sf_dtype, sf_vec_size):
    _run_compile_execute(
        request,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("ab_dtype,sf_dtype,sf_vec_size", FP4_EXECUTION_CASES)
@pytest.mark.parametrize("act_func", ["swiglu", "geglu"])
def test_grouped_gemm_glu_hadamard_wrapper_fp4(request, ab_dtype, sf_dtype, sf_vec_size, act_func):
    _run_wrapper(
        request,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        act_func=act_func,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_wrapper_with_bias(request):
    _run_wrapper(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e8m0fnu,
        sf_vec_size=16,
        act_func="swiglu",
        enable_bias=True,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("group_m_list", [[256, 256, 256, 256], DYNAMIC_SHAPES_M_VALUES])
def test_grouped_gemm_glu_hadamard_wrapper_cache_dynamic_m_smoke(request, monkeypatch, group_m_list):
    from cudnn import grouped_gemm_glu_hadamard_wrapper_sm100
    from cudnn.grouped_gemm.grouped_gemm_glu_hadamard import api as grouped_gemm_glu_hadamard_api

    grouped_gemm_glu_hadamard_api._cache_of_GroupedGemmGluHadamardSm100Objects.clear()

    compile_count = {"value": 0}

    def counted_compile(self):
        compile_count["value"] += 1
        return None

    monkeypatch.setattr(grouped_gemm_glu_hadamard_api.GroupedGemmGluHadamardSm100, "compile", counted_compile)
    monkeypatch.setattr(grouped_gemm_glu_hadamard_api.GroupedGemmGluHadamardSm100, "check_support", lambda self: True)
    monkeypatch.setattr(grouped_gemm_glu_hadamard_api.GroupedGemmGluHadamardSm100, "execute", lambda self, **kwargs: None)

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
        grouped_gemm_glu_hadamard_wrapper_sm100(
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
    assert len(grouped_gemm_glu_hadamard_api._cache_of_GroupedGemmGluHadamardSm100Objects) == 1
    grouped_gemm_glu_hadamard_api._cache_of_GroupedGemmGluHadamardSm100Objects.clear()
