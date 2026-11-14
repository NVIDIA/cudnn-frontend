# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for grouped GEMM GLU + Hadamard forward fusion (SM100+)."""

from typing import Dict

import pytest
import torch

from determinism_utils import assert_replays
from test_utils import bitwise_bits, torch_fork_set_rng
from gemm.cutedsl.test_discrete_grouped_gemm_swiglu_utils import allocate_discrete_input_tensors
from core.cutedsl.test_fe_api_utils import DYNAMIC_SHAPES_M_VALUES, compute_reference_amax
from gemm.cutedsl.test_grouped_gemm_swiglu_utils import allocate_grouped_gemm_input_tensors, grouped_gemm_swiglu_init

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


def _apply_transpose_hadamard(d_ref: torch.Tensor, aligned_group_m_list) -> torch.Tensor:
    from cudnn.gemm.cutedsl.grouped.glu_hadamard.hadamard_utils import HADAMARD_SIZE, hadamard_matrix

    valid_m, n_out, _ = d_ref.shape
    hadamard = hadamard_matrix(HADAMARD_SIZE, dtype=torch.float32, device=d_ref.device)
    out = torch.empty((valid_m, n_out), dtype=torch.float32, device=d_ref.device)
    start = 0
    for group_m in aligned_group_m_list:
        end = start + group_m
        group_t = d_ref[start:end, :, 0].to(torch.bfloat16).to(torch.float32).T.contiguous()
        group_t = group_t.view(n_out, group_m // HADAMARD_SIZE, HADAMARD_SIZE)
        out[start:end, :] = ((group_t @ hadamard) * (HADAMARD_SIZE**-0.5)).view(n_out, group_m).T
        start = end
    return out.unsqueeze(-1)


def _run_grouped_gemm_glu_ref(inputs: Dict, act_func: str, situ_beta1: float = 4.0, situ_beta2: float = 25.0) -> Dict:
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
    elif act_func == "situglu":
        ref_after_glu = situ_beta1 * torch.tanh(ref_gate / situ_beta1) * torch.sigmoid(ref_gate) * situ_beta2 * torch.tanh(ref_up / situ_beta2)
    else:
        raise ValueError(f"Unsupported act_func {act_func}")

    ref_after_glu = ref_after_glu * inputs["prob_tensor"].expand(-1, n_out, -1)
    return {"c_ref": ref.clone(), "d_ref": ref_after_glu}


def _check_reference(
    inputs: Dict,
    outputs: Dict,
    cfg: Dict,
    *,
    act_func: str,
    situ_beta1: float = 4.0,
    situ_beta2: float = 25.0,
) -> None:
    ref_tensors = _run_grouped_gemm_glu_ref(inputs, act_func, situ_beta1, situ_beta2)

    torch.testing.assert_close(
        outputs["c_tensor"][: inputs["valid_m"]].cpu().float(),
        ref_tensors["c_ref"].cpu().to(cfg["c_dtype"]).to(torch.float32),
        atol=1e-1,
        rtol=1e-2,
    )

    torch.testing.assert_close(
        outputs["d_tensor"][: inputs["valid_m"]].cpu().float(),
        ref_tensors["d_ref"].cpu().to(cfg["d_dtype"]).to(torch.float32),
        atol=1e-1,
        rtol=1e-2,
    )

    if outputs["amax_tensor"] is not None:
        amax_ref = torch.empty((cfg["l"],), dtype=torch.float32)
        start = 0
        for i, group_m in enumerate(inputs["aligned_group_m_list"]):
            end = start + group_m
            amax_ref[i] = compute_reference_amax(ref_tensors["d_ref"][start:end, :, 0].clone())
            start = end
        torch.testing.assert_close(
            outputs["amax_tensor"].cpu().reshape(-1),
            amax_ref,
            atol=1e-1,
            rtol=1e-2,
        )

    if outputs["post_rht_amax_tensor"] is not None:
        d_hadamard_ref = _apply_transpose_hadamard(ref_tensors["d_ref"], inputs["aligned_group_m_list"])
        post_rht_amax_ref = torch.empty((cfg["l"],), dtype=torch.float32)
        start = 0
        for i, group_m in enumerate(inputs["aligned_group_m_list"]):
            end = start + group_m
            post_rht_amax_ref[i] = compute_reference_amax(d_hadamard_ref[start:end, :, 0].clone())
            start = end
        torch.testing.assert_close(
            outputs["post_rht_amax_tensor"].cpu().reshape(-1),
            post_rht_amax_ref,
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
        "post_rht_amax_tensor": torch.full((l, 1), float("-inf"), dtype=torch.float32, device=device),
    }


def _run_compile_execute(
    request,
    *,
    ab_dtype,
    sf_dtype,
    sf_vec_size,
    act_func="swiglu",
    enable_bias=False,
    situ_beta1=4.0,
    situ_beta2=25.0,
    execute_empty_input=False,
    execute_situ_beta1=None,
    execute_situ_beta2=None,
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
        sample_post_rht_amax=outputs["post_rht_amax_tensor"],
        sample_bias=inputs["bias_tensor"],
        acc_dtype=cfg["acc_dtype"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        cluster_shape_mn=cfg["cluster_shape_mn"],
        sf_vec_size=cfg["sf_vec_size"],
        vector_f32=cfg["vector_f32"],
        m_aligned=cfg["m_aligned"],
        act_func=act_func,
        situ_beta1=situ_beta1,
        situ_beta2=situ_beta2,
    )
    api.check_support()
    api.compile()
    api.execute(
        a_tensor=inputs["a_tensor"][:0] if execute_empty_input else inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        c_tensor=outputs["c_tensor"],
        d_tensor=outputs["d_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        amax_tensor=outputs["amax_tensor"],
        post_rht_amax_tensor=outputs["post_rht_amax_tensor"],
        bias_tensor=inputs["bias_tensor"],
        situ_beta1=execute_situ_beta1,
        situ_beta2=execute_situ_beta2,
    )

    if execute_empty_input:
        return
    reference_situ_beta1 = situ_beta1 if execute_situ_beta1 is None else execute_situ_beta1
    reference_situ_beta2 = situ_beta2 if execute_situ_beta2 is None else execute_situ_beta2
    _check_reference(
        inputs,
        outputs,
        cfg,
        act_func=act_func,
        situ_beta1=reference_situ_beta1,
        situ_beta2=reference_situ_beta2,
    )


def _run_wrapper(
    request,
    *,
    ab_dtype,
    sf_dtype,
    sf_vec_size,
    act_func="swiglu",
    enable_bias=False,
    situ_beta1=4.0,
    situ_beta2=25.0,
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
        situ_beta1=situ_beta1,
        situ_beta2=situ_beta2,
    )

    _check_reference(inputs, outputs, cfg, act_func=act_func, situ_beta1=situ_beta1, situ_beta2=situ_beta2)


def _run_discrete_wrapper(request, *, ab_dtype, sf_dtype, sf_vec_size, act_func="swiglu"):
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

    from cudnn import grouped_gemm_glu_hadamard_wrapper_sm100

    outputs = grouped_gemm_glu_hadamard_wrapper_sm100(
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
@pytest.mark.parametrize(("situ_beta1", "situ_beta2"), [(4.0, 25.0), (2.0, 8.0)])
def test_grouped_gemm_glu_hadamard_wrapper_situglu_fp4(request, situ_beta1, situ_beta2):
    _run_wrapper(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e8m0fnu,
        sf_vec_size=16,
        act_func="situglu",
        situ_beta1=situ_beta1,
        situ_beta2=situ_beta2,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_execute_uses_compiled_situglu_betas(request):
    # Construct and compile for non-default beta values, then intentionally omit
    # them from execute(). The launch must inherit the values stored by the API
    # object so both the compile-time beta1 specialization and runtime beta2
    # scalar match the numerical reference.
    _run_compile_execute(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e8m0fnu,
        sf_vec_size=16,
        act_func="situglu",
        situ_beta1=2.0,
        situ_beta2=8.0,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_execute_uses_runtime_situglu_beta2(request):
    # beta2 is a runtime scalar, so execute() may override the value stored by
    # the compiled API object. Verify both the launch and numerical reference
    # use the explicit execution-time value.
    _run_compile_execute(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e8m0fnu,
        sf_vec_size=16,
        act_func="situglu",
        execute_situ_beta2=8.0,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    ("execute_situ_beta1", "execute_situ_beta2", "match"),
    [(0.0, None, "situ_beta1"), (None, float("nan"), "situ_beta2")],
)
def test_grouped_gemm_glu_hadamard_execute_empty_input_validates_situglu_betas(
    request,
    execute_situ_beta1,
    execute_situ_beta2,
    match,
):
    with pytest.raises(ValueError, match=match):
        _run_compile_execute(
            request,
            ab_dtype=torch.float4_e2m1fn_x2,
            sf_dtype=torch.float8_e8m0fnu,
            sf_vec_size=16,
            act_func="situglu",
            execute_empty_input=True,
            execute_situ_beta1=execute_situ_beta1,
            execute_situ_beta2=execute_situ_beta2,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    ("situ_beta1", "situ_beta2", "match"),
    [(0.0, 25.0, "situ_beta1"), (4.0, float("nan"), "situ_beta2")],
)
def test_grouped_gemm_glu_hadamard_empty_input_validates_situglu_betas(
    request,
    situ_beta1,
    situ_beta2,
    match,
):
    from cudnn import grouped_gemm_glu_hadamard_wrapper_sm100

    cfg = _make_cfg(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e8m0fnu,
        sf_vec_size=16,
    )
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

    with pytest.raises(ValueError, match=match):
        grouped_gemm_glu_hadamard_wrapper_sm100(
            a_tensor=inputs["a_tensor"][:0],
            b_tensor=inputs["b_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            sfb_tensor=inputs["sfb_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            prob_tensor=inputs["prob_tensor"][:0],
            act_func="situglu",
            situ_beta1=situ_beta1,
            situ_beta2=situ_beta2,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_wrapper_discrete_fp4(request):
    _run_discrete_wrapper(
        request,
        ab_dtype=torch.float4_e2m1fn_x2,
        sf_dtype=torch.float8_e8m0fnu,
        sf_vec_size=16,
        act_func="swiglu",
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
    from cudnn.gemm.cutedsl.grouped.glu_hadamard import api as grouped_gemm_glu_hadamard_api

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


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_wrapper_cache_specializes_situ_beta1_only(request, monkeypatch):
    from cudnn import grouped_gemm_glu_hadamard_wrapper_sm100
    from cudnn.gemm.cutedsl.grouped.glu_hadamard import api as grouped_gemm_glu_hadamard_api

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

    for situ_beta1, situ_beta2 in ((4.0, 25.0), (4.0, 8.0), (2.0, 25.0), (4.0, 25.0)):
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
            act_func="situglu",
            situ_beta1=situ_beta1,
            situ_beta2=situ_beta2,
        )

    # beta1 changes the generated activation code and therefore creates a new
    # compiled API. beta2 is a runtime FP32 scalar and reuses the beta1=4 API.
    assert compile_count["value"] == 2
    assert len(grouped_gemm_glu_hadamard_api._cache_of_GroupedGemmGluHadamardSm100Objects) == 2
    grouped_gemm_glu_hadamard_api._cache_of_GroupedGemmGluHadamardSm100Objects.clear()


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_wrapper_cache_separates_devices(request, monkeypatch):
    from cudnn import grouped_gemm_glu_hadamard_wrapper_sm100
    from cudnn.gemm.cutedsl.grouped.glu_hadamard import api as grouped_gemm_glu_hadamard_api
    from gemm.cutedsl.test_grouped_gemm_canonical_layouts import assert_same

    first = torch.cuda.current_device()
    other = next((i for i in range(torch.cuda.device_count()) if i != first and torch.cuda.get_device_capability(i)[0] >= 10), None)
    if other is None:
        pytest.skip("two SM100+ CUDA devices are required")
    monkeypatch.setattr(grouped_gemm_glu_hadamard_api, "_cache_of_GroupedGemmGluHadamardSm100Objects", {})

    cfg = _make_cfg(request, ab_dtype=torch.float4_e2m1fn_x2, sf_dtype=torch.float8_e8m0fnu, sf_vec_size=16)
    source = allocate_grouped_gemm_input_tensors(
        n=cfg["n"],
        k=cfg["k"],
        l=cfg["l"],
        group_m_list=cfg["group_m_list"],
        ab_dtype=cfg["ab_dtype"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        m_aligned=cfg["m_aligned"],
        b_major=cfg["b_major"],
    )
    host = {name: value.cpu() if isinstance(value, torch.Tensor) else value for name, value in source.items()}

    def call(inputs):
        return grouped_gemm_glu_hadamard_wrapper_sm100(
            a_tensor=inputs["a_tensor"],
            b_tensor=inputs["b_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            sfb_tensor=inputs["sfb_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            prob_tensor=inputs["prob_tensor"],
            c_dtype=cfg["c_dtype"],
            d_dtype=cfg["d_dtype"],
            mma_tiler_mn=cfg["mma_tiler_mn"],
            cluster_shape_mn=cfg["cluster_shape_mn"],
            sf_vec_size=cfg["sf_vec_size"],
        )

    retained = {}
    graphs = []
    try:
        for device in (first, other):
            with torch.cuda.device(device):
                inputs = {name: value.to(device=device) if isinstance(value, torch.Tensor) else value for name, value in host.items()}
                call(inputs)  # Compile once on each device, without clearing the shared cache.
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                graph = torch.cuda.CUDAGraph()
                graphs.append((device, graph))
                with torch.cuda.graph(graph, stream=stream):
                    captured = call(inputs)
                retained[device] = inputs, graph, captured

        for device in (first, other, first):
            with torch.cuda.device(device):
                inputs, graph, captured = retained[device]
                inputs["alpha_tensor"].mul_(0.5)
                expected = call(inputs)
                graph.replay()
                torch.cuda.synchronize()
                for name in ("c_tensor", "d_tensor", "amax_tensor", "post_rht_amax_tensor"):
                    assert_same(name, captured[name], expected[name])
    finally:
        for device, graph in graphs:
            with torch.cuda.device(device):
                graph.reset()


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("ab_dtype,sf_dtype,sf_vec_size", FP4_EXECUTION_CASES[:3])
@pytest.mark.parametrize("flat_sf", [False, True], ids=["physical_sf", "flat_sf"])
def test_grouped_gemm_glu_hadamard_wrapper_canonical_matches_legacy(request, ab_dtype, sf_dtype, sf_vec_size, flat_sf):
    from cudnn import grouped_gemm_glu_hadamard_wrapper_sm100
    from gemm.cutedsl.test_grouped_gemm_canonical_layouts import SF_PHYSICAL_PERMUTE, assert_same

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
    )

    def call(canonical):
        a, b, prob = inputs["a_tensor"], inputs["b_tensor"], inputs["prob_tensor"]
        sfa, sfb = inputs["sfa_tensor"], inputs["sfb_tensor"]
        if canonical:
            a, b, prob = a.squeeze(-1), b.permute(2, 0, 1), prob.view(-1)
            sfa, sfb = sfa.permute(*SF_PHYSICAL_PERMUTE), sfb.permute(*SF_PHYSICAL_PERMUTE)
            assert b.is_contiguous() and sfa.is_contiguous() and sfb.is_contiguous()
            if flat_sf:
                sfa, sfb = sfa.reshape(-1), sfb.reshape(-1)
        return grouped_gemm_glu_hadamard_wrapper_sm100(
            a_tensor=a,
            b_tensor=b,
            sfa_tensor=sfa,
            sfb_tensor=sfb,
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            prob_tensor=prob,
            c_dtype=cfg["c_dtype"],
            d_dtype=cfg["d_dtype"],
            mma_tiler_mn=cfg["mma_tiler_mn"],
            cluster_shape_mn=cfg["cluster_shape_mn"],
            sf_vec_size=cfg["sf_vec_size"],
        )

    legacy = call(canonical=False)
    for _ in range(2):
        canonical = call(canonical=True)
        assert canonical["c_tensor"].shape == (inputs["tensor_m"], cfg["n"])
        assert canonical["d_tensor"].ndim == 2
        for name in ("c_tensor", "d_tensor", "amax_tensor", "post_rht_amax_tensor"):
            assert_same(name, canonical[name], legacy[name])


# =============================================================================
# SReLU epilogue options: saved-C regeneration, tanh_clamp_scale, rht_sign_mask
# =============================================================================
#
# These pin four contracts that a tolerance comparison against an FP32 reference cannot see:
#
# 1. D is evaluated from C *as stored in c_dtype*, so a consumer that recomputes the activation
#    from the saved C (the dSReLU backward, activation recomputation) reproduces D bit for bit.
# 2. Amax is the amax of D as stored in d_dtype, and PostRhtAmax is the amax of the transformed
#    values after rounding to BF16 -- the values the downstream NVFP4 quantizer actually sees.
# 3. tanh_clamp_scale and rht_sign_mask are trace-time constants: both must reach the kernel
#    and both must be part of the wrapper's cache key.
# 4. All of the above hold on both post-RHT paths (shared-memory FWHT and TMEM).
#
# The references below are therefore computed from the kernel's own stored C / D rather than
# from the dequantized inputs; test_grouped_gemm_glu_hadamard_wrapper_fp4 and friends already
# cover the GEMM itself.

# Unequal expert sizes and one empty expert, so per-expert reductions cannot pass by symmetry.
SRELU_OPTION_GROUP_M_LIST = [256, 0, 512, 256]

# Arbitrary non-trivial pattern, the all-negated mask (must equal no mask: H(-x) = -H(x)) and a
# single flipped element.
RHT_SIGN_MASKS = [
    pytest.param(0xA5C3, id="mask_a5c3"),
    pytest.param(0xFFFF, id="mask_all"),
    pytest.param(0x0001, id="mask_bit0"),
]

POST_RHT_AMAX_PATHS = [
    pytest.param(False, id="smem_post_rht_amax"),
    pytest.param(True, id="tmem_post_rht_amax"),
]

# Parameter order of grouped_gemm_glu_hadamard_wrapper_sm100 up to and including
# current_stream, frozen. Public signatures are append-only (root AGENTS.md): a caller that
# passes its stream positionally must keep binding it to current_stream.
_WRAPPER_LEGACY_POSITIONAL_ORDER = [
    "a_tensor",
    "sfa_tensor",
    "padded_offsets",
    "alpha_tensor",
    "prob_tensor",
    "b_tensor",
    "sfb_tensor",
    "b_ptrs",
    "sfb_ptrs",
    "n",
    "b_dtype",
    "b_major",
    "bias_tensor",
    "acc_dtype",
    "c_dtype",
    "d_dtype",
    "cd_major",
    "mma_tiler_mn",
    "cluster_shape_mn",
    "sf_vec_size",
    "vector_f32",
    "m_aligned",
    "act_func",
    "situ_beta1",
    "situ_beta2",
    "use_dynamic_sched",
    "use_tmem_post_rht_amax",
    "current_stream",
]


def _build_srelu_option_case(request, *, c_dtype=torch.bfloat16, d_dtype=torch.bfloat16, vector_f32=False, small_alpha=False):
    cfg = _make_cfg(request, ab_dtype=torch.float4_e2m1fn_x2, sf_dtype=torch.float8_e8m0fnu, sf_vec_size=16)
    cfg["c_dtype"], cfg["d_dtype"], cfg["vector_f32"] = c_dtype, d_dtype, vector_f32
    if request.config.getoption("--grouped-gemm-group-m", default=None) is None and cfg["l"] == len(SRELU_OPTION_GROUP_M_LIST):
        cfg["group_m_list"] = list(SRELU_OPTION_GROUP_M_LIST)
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
    )
    # The allocator draws alpha from {-2, ..., 1}: an expert can land on 0 (C == 0 everywhere)
    # and relu(C)^2 overflows FP16 at |alpha| >= 1. Pin alpha so every expert is live, and for
    # the unclamped cases keep D finite in both output dtypes (small_alpha) while C still needs
    # rounding to c_dtype (a power-of-two scale does not make the accumulator representable).
    magnitudes = (0.0625, 0.125, 0.03125) if small_alpha else (1.0, 2.0, 0.5)
    alpha = [magnitudes[i % 3] * (-1.0 if i % 2 else 1.0) for i in range(cfg["l"])]
    inputs["alpha_tensor"].copy_(torch.tensor(alpha, dtype=torch.float32, device=inputs["alpha_tensor"].device))
    return inputs, cfg


def _run_srelu_option_case(case, **wrapper_kwargs):
    """Run the wrapper over an already-built case, so repeated calls share identical inputs."""
    from cudnn import grouped_gemm_glu_hadamard_wrapper_sm100

    inputs, cfg = case
    wrapper_kwargs.setdefault("act_func", "srelu")
    outputs = grouped_gemm_glu_hadamard_wrapper_sm100(
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
        **wrapper_kwargs,
    )
    torch.cuda.synchronize()
    return outputs


def _stored_cd_prob(inputs, outputs):
    """The stored C and D (valid rows, on CPU, in their own dtypes) and the FP32 prob column."""
    valid_m = inputs["valid_m"]
    c = outputs["c_tensor"][:valid_m, :, 0].cpu()
    d = outputs["d_tensor"][:valid_m, :, 0].cpu()
    prob = inputs["prob_tensor"][:valid_m, :, 0].float().cpu()
    return c, d, prob


def _assert_same_bytes(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, f"{label}: shape or dtype differs"
    a, b = bitwise_bits(actual.reshape(-1)), bitwise_bits(expected.reshape(-1))
    count = torch.count_nonzero(a != b).item()
    assert count == 0, f"{label}: {count}/{a.numel()} bytes differ"


def _expert_slices(inputs):
    start = 0
    for expert, group_m in enumerate(inputs["aligned_group_m_list"]):
        yield expert, slice(start, start + group_m), group_m
        start += group_m


def _assert_amax_is_stored_d_amax(inputs, outputs, d: torch.Tensor) -> None:
    amax = outputs["amax_tensor"].reshape(-1).cpu()
    for expert, rows, group_m in _expert_slices(inputs):
        if group_m == 0:
            continue
        want = d[rows].float().abs().max().item()
        assert amax[expert].item() == want, f"expert {expert}: amax {amax[expert].item()} != amax of stored D {want}"


def _post_rht_amax_ref(inputs, d: torch.Tensor, rht_sign_mask: int = 0) -> Dict[int, float]:
    """Per-expert amax of the randomized Hadamard transform of the stored D, rounded to BF16.

    Bit ``i`` of ``rht_sign_mask`` negates element ``i`` of each 16-token block before the
    normalized 16-point transform. Computed in FP64, where the transform of BF16/FP16 inputs
    is exact, then rounded to BF16 as the consumer's BF16 transform output is.
    """
    from cudnn.gemm.cutedsl.grouped.glu_hadamard.hadamard_utils import HADAMARD_SIZE, hadamard_matrix

    hadamard = hadamard_matrix(HADAMARD_SIZE, dtype=torch.float64, device=d.device)
    signs = torch.tensor([-1.0 if (rht_sign_mask >> i) & 1 else 1.0 for i in range(HADAMARD_SIZE)], dtype=torch.float64)
    ref = {}
    for expert, rows, group_m in _expert_slices(inputs):
        if group_m == 0:
            continue
        blocks = d[rows].double().T.reshape(d.shape[1], group_m // HADAMARD_SIZE, HADAMARD_SIZE) * signs
        transformed = (blocks @ hadamard) * (HADAMARD_SIZE**-0.5)
        ref[expert] = transformed.to(torch.bfloat16).float().abs().max().item()
    return ref


def _assert_post_rht_amax(inputs, outputs, d: torch.Tensor, rht_sign_mask: int = 0) -> None:
    post_rht_amax = outputs["post_rht_amax_tensor"].reshape(-1).cpu()
    for expert, want in _post_rht_amax_ref(inputs, d, rht_sign_mask).items():
        got = post_rht_amax[expert].item()
        assert got == want, f"expert {expert}: post-RHT amax {got} != reference {want} (rht_sign_mask={rht_sign_mask:#06x})"


@pytest.mark.L0
def test_grouped_gemm_glu_hadamard_wrapper_keeps_legacy_positional_order():
    """New wrapper options are appended after current_stream, never inserted before it."""
    import inspect

    from cudnn import grouped_gemm_glu_hadamard_wrapper_sm100

    params = list(inspect.signature(grouped_gemm_glu_hadamard_wrapper_sm100).parameters)
    assert params[: len(_WRAPPER_LEGACY_POSITIONAL_ORDER)] == _WRAPPER_LEGACY_POSITIONAL_ORDER
    assert {"tanh_clamp_scale", "rht_sign_mask"} <= set(params[len(_WRAPPER_LEGACY_POSITIONAL_ORDER) :])


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("vector_f32", [False, True], ids=["scalar_f32", "vector_f32"])
@pytest.mark.parametrize("use_tmem_post_rht_amax", POST_RHT_AMAX_PATHS)
@pytest.mark.parametrize("cd_dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_grouped_gemm_glu_hadamard_srelu_saved_c_regeneration(request, cd_dtype, use_tmem_post_rht_amax, vector_f32):
    """D is bit-identical to the activation recomputed from the saved C, in FP32 arithmetic."""
    case = _build_srelu_option_case(request, c_dtype=cd_dtype, d_dtype=cd_dtype, vector_f32=vector_f32, small_alpha=True)
    inputs, _ = case
    outputs = _run_srelu_option_case(case, use_tmem_post_rht_amax=use_tmem_post_rht_amax)
    c, d, prob = _stored_cd_prob(inputs, outputs)

    assert bool(torch.isfinite(d.float()).all()), "D overflowed; the regeneration check would compare infinities"
    assert torch.count_nonzero(d).item() > d.numel() // 8, "D is mostly zero; the regeneration check would be vacuous"

    c_relu = torch.relu(c.float())
    regenerated = ((c_relu * c_relu) * prob).to(cd_dtype)
    _assert_same_bytes(d, regenerated, "D vs activation regenerated from saved C")

    _assert_amax_is_stored_d_amax(inputs, outputs, d)
    _assert_post_rht_amax(inputs, outputs, d)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("vector_f32", [False, True], ids=["scalar_f32", "vector_f32"])
@pytest.mark.parametrize("use_tmem_post_rht_amax", POST_RHT_AMAX_PATHS)
@pytest.mark.parametrize("tanh_clamp_scale", [16.0, 32.0], ids=["clamp16", "clamp32"])
def test_grouped_gemm_glu_hadamard_srelu_tanh_clamp_scale(request, tanh_clamp_scale, use_tmem_post_rht_amax, vector_f32):
    """The clamped epilogue computes (s * tanh(relu(C) / s))^2 * prob from the saved C."""
    case = _build_srelu_option_case(request, vector_f32=vector_f32)
    inputs, cfg = case
    outputs = _run_srelu_option_case(case, tanh_clamp_scale=tanh_clamp_scale, use_tmem_post_rht_amax=use_tmem_post_rht_amax)
    c, d, prob = _stored_cd_prob(inputs, outputs)

    # The inputs must reach both the near-linear region and deep saturation of the clamp.
    c_relu = torch.relu(c.double())
    assert bool((c_relu > 4 * tanh_clamp_scale).any()) and bool(((c_relu > 0) & (c_relu < tanh_clamp_scale / 4)).any())

    # The kernel uses the approximate tanh, so this is a tolerance check, not a bitwise one.
    d_ref = (tanh_clamp_scale * torch.tanh(c_relu / tanh_clamp_scale)) ** 2 * prob.double()
    torch.testing.assert_close(d.float(), d_ref.to(cfg["d_dtype"]).float(), atol=1e-1, rtol=1e-2)

    # t is capped at 1 before use, so the output bound holds exactly, not within a tolerance.
    bound = (tanh_clamp_scale * tanh_clamp_scale) * prob.abs()
    assert bool((d.float().abs() <= bound).all()), "D exceeds s^2 * |prob|"

    _assert_amax_is_stored_d_amax(inputs, outputs, d)
    _assert_post_rht_amax(inputs, outputs, d)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_srelu_tanh_clamp_scale_cache_key(request):
    """tanh_clamp_scale is a trace-time constant, so it must select its own compiled kernel.

    A cache key that omitted it would silently serve the first scale's kernel to every later
    one. Interleave three settings over identical inputs and require each to keep its own
    behaviour, and None to stay bit-identical to omitting the argument.
    """
    case = _build_srelu_option_case(request)
    inputs, _ = case

    def stored_d(**kwargs):
        return _stored_cd_prob(inputs, _run_srelu_option_case(case, **kwargs))[1]

    prob_abs = inputs["prob_tensor"][: inputs["valid_m"], :, 0].float().abs().cpu()
    omitted = stored_d()
    d16 = stored_d(tanh_clamp_scale=16.0)
    d32 = stored_d(tanh_clamp_scale=32.0)
    explicit_none = stored_d(tanh_clamp_scale=None)
    d16_again = stored_d(tanh_clamp_scale=16.0)

    assert bool((d16.float().abs() <= 16.0 * 16.0 * prob_abs).all())
    assert bool((d32.float().abs() <= 32.0 * 32.0 * prob_abs).all())
    assert bool((d32.float().abs() > 16.0 * 16.0 * prob_abs).any()), "the 32.0 call was served the 16.0 kernel"
    assert bool((omitted.float().abs() > 32.0 * 32.0 * prob_abs).any()), "the unclamped call was served a clamped kernel"
    _assert_same_bytes(explicit_none, omitted, "tanh_clamp_scale=None vs argument omitted")
    _assert_same_bytes(d16_again, d16, "tanh_clamp_scale=16.0 repeated after other scales")


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("use_tmem_post_rht_amax", POST_RHT_AMAX_PATHS)
@pytest.mark.parametrize("rht_sign_mask", RHT_SIGN_MASKS)
def test_grouped_gemm_glu_hadamard_srelu_rht_sign_mask(request, rht_sign_mask, use_tmem_post_rht_amax):
    """rht_sign_mask changes only PostRhtAmax, to the amax of the sign-randomized transform.

    Runs with the soft clamp: unclamped relu(C)^2 on this data has isolated outliers that
    dominate their 16-token block, and a block dominated by one element has the same
    transform amax under every sign pattern, which would make the mask unobservable.
    """
    case = _build_srelu_option_case(request)
    inputs, _ = case
    common = dict(tanh_clamp_scale=16.0, use_tmem_post_rht_amax=use_tmem_post_rht_amax)
    unmasked = _run_srelu_option_case(case, **common)
    masked = _run_srelu_option_case(case, rht_sign_mask=rht_sign_mask, **common)
    _, d, _ = _stored_cd_prob(inputs, masked)

    for name in ("c_tensor", "d_tensor", "amax_tensor"):
        _assert_same_bytes(masked[name], unmasked[name], f"{name} with vs without rht_sign_mask")

    _assert_post_rht_amax(inputs, unmasked, d)
    _assert_post_rht_amax(inputs, masked, d, rht_sign_mask)

    live = [expert for expert, _, group_m in _expert_slices(inputs) if group_m]
    masked_amax = masked["post_rht_amax_tensor"].reshape(-1).cpu()[live]
    unmasked_amax = unmasked["post_rht_amax_tensor"].reshape(-1).cpu()[live]
    if rht_sign_mask == 0xFFFF:
        # Negating every input negates every output: the amax cannot move.
        assert torch.equal(masked_amax, unmasked_amax)
    else:
        assert not torch.equal(masked_amax, unmasked_amax), "rht_sign_mask did not reach the kernel (or was dropped from the cache key)"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("use_tmem_post_rht_amax", POST_RHT_AMAX_PATHS)
def test_grouped_gemm_glu_hadamard_srelu_options_replay(request, use_tmem_post_rht_amax):
    """Every returned output keeps identical bytes across repeated calls with fixed inputs.

    This is run-to-run repeatability under scheduling pressure only. It does not establish
    batch invariance: Amax and PostRhtAmax are whole-expert reductions.
    """
    case = _build_srelu_option_case(request)

    def run():
        outputs = _run_srelu_option_case(case, tanh_clamp_scale=16.0, rht_sign_mask=0xA5C3, use_tmem_post_rht_amax=use_tmem_post_rht_amax)
        return {name: outputs[name] for name in ("c_tensor", "d_tensor", "amax_tensor", "post_rht_amax_tensor")}

    assert_replays(run)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"act_func": "swiglu", "tanh_clamp_scale": 16.0}, "tanh_clamp_scale is only supported with act_func='srelu'", id="clamp_with_glu"),
        pytest.param({"tanh_clamp_scale": 0.0}, "tanh_clamp_scale must be finite and positive", id="clamp_zero"),
        pytest.param({"tanh_clamp_scale": -16.0}, "tanh_clamp_scale must be finite and positive", id="clamp_negative"),
        pytest.param({"tanh_clamp_scale": float("inf")}, "tanh_clamp_scale must be finite and positive", id="clamp_inf"),
        pytest.param({"tanh_clamp_scale": float("nan")}, "tanh_clamp_scale must be finite and positive", id="clamp_nan"),
        pytest.param({"rht_sign_mask": -1}, "rht_sign_mask must be a 16-bit integer", id="mask_negative"),
        pytest.param({"rht_sign_mask": 1 << 16}, "rht_sign_mask must be a 16-bit integer", id="mask_too_wide"),
        pytest.param({"rht_sign_mask": 1.5}, "rht_sign_mask must be a 16-bit integer", id="mask_not_int"),
    ],
)
def test_grouped_gemm_glu_hadamard_srelu_options_reject_invalid(request, kwargs, match):
    case = _build_srelu_option_case(request)
    with pytest.raises(ValueError, match=match):
        _run_srelu_option_case(case, **kwargs)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("invalid_mask", [1.5, 1.0, "1"], ids=["mask_1.5", "mask_1.0", "mask_str"])
def test_grouped_gemm_glu_hadamard_srelu_rejects_invalid_rht_sign_mask_on_warm_cache(request, invalid_mask):
    """An invalid mask is rejected even when a valid call has already cached the kernel it would map to.

    The wrapper normalizes the mask with int() for its cache key, so 1.5, 1.0 and "1" all land on
    the entry a valid rht_sign_mask=1 call creates. Rejection must not depend on prior calls.
    """
    case = _build_srelu_option_case(request)
    common = dict(tanh_clamp_scale=16.0)
    _run_srelu_option_case(case, rht_sign_mask=1, **common)
    with pytest.raises(ValueError, match="rht_sign_mask must be a 16-bit integer"):
        _run_srelu_option_case(case, rht_sign_mask=invalid_mask, **common)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_glu_hadamard_rejects_rht_sign_mask_with_sample_hadamard(request):
    """A caller-supplied Hadamard tensor already carries its signs; a mask on top is ambiguous."""
    from cudnn import GroupedGemmGluHadamardSm100
    from cudnn.gemm.cutedsl.grouped.glu_hadamard.hadamard_utils import HADAMARD_SIZE, hadamard_matrix

    cfg = _make_cfg(request, ab_dtype=torch.float4_e2m1fn_x2, sf_dtype=torch.float8_e8m0fnu, sf_vec_size=16)
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
    )
    outputs = _allocate_outputs(inputs, cfg)
    with pytest.raises(ValueError, match="rht_sign_mask cannot be combined with sample_hadamard"):
        GroupedGemmGluHadamardSm100(
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
            sample_post_rht_amax=outputs["post_rht_amax_tensor"],
            sample_hadamard=hadamard_matrix(HADAMARD_SIZE, dtype=torch.bfloat16, device=inputs["a_tensor"].device),
            rht_sign_mask=0xA5C3,
        )
