# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical layouts for the grouped GEMM backward wrappers: dGLU, dgrad quant, and wgrad."""

import pytest
import torch
from test_utils import torch_fork_set_rng

from gemm.cutedsl.test_grouped_gemm_canonical_layouts import SF_PHYSICAL_PERMUTE, assert_same, dswiglu_case

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")


def run_dglu(cfg, inputs, canonical, b_major="k", flat_sf=False, deterministic=False, sfa_override=None):
    from cudnn import grouped_gemm_dglu_wrapper_sm100

    m = inputs["tensor_m"]
    b = inputs["b_tensor"]
    a, c, prob = inputs["a_tensor"], inputs["c_tensor"], inputs["prob_tensor"]
    sfa, sfb = inputs["sfa_tensor"], inputs["sfb_tensor"]
    if b_major == "n":
        b = b.permute(2, 1, 0).contiguous().permute(2, 1, 0)
    dprob = torch.zeros((m, 1, 1), dtype=torch.float32, device="cuda")
    if canonical:
        a, c, prob, dprob = a.squeeze(-1), c.squeeze(-1), prob.view(-1), dprob.view(-1)
        b = b.permute(2, 1, 0) if b_major == "n" else b.permute(2, 0, 1)
        sfa, sfb = sfa.permute(*SF_PHYSICAL_PERMUTE), sfb.permute(*SF_PHYSICAL_PERMUTE)
        assert b.is_contiguous() and sfa.is_contiguous() and sfb.is_contiguous()
        if flat_sf:
            sfa, sfb = sfa.reshape(-1), sfb.reshape(-1)
    if sfa_override is not None:
        sfa = sfa_override
    return grouped_gemm_dglu_wrapper_sm100(
        a_tensor=a,
        c_tensor=c,
        sfa_tensor=sfa,
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        beta_tensor=inputs["beta_tensor"],
        prob_tensor=prob,
        dprob_tensor=dprob,
        b_tensor=b,
        sfb_tensor=sfb,
        b_major=b_major,
        norm_const_tensor=inputs.get("norm_const_tensor"),
        d_dtype=cfg["d_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        discrete_col_sfd=True,
        use_dynamic_sched=True,
        deterministic=deterministic,
    )


def assert_dglu_outputs_equal(canonical, legacy, m, n2, exact_dprob):
    assert canonical["d_row_tensor"].shape == (m, n2)
    assert canonical["dprob_tensor"].shape == (m,)
    assert canonical["sfd_row_tensor"].is_contiguous()
    assert_same("d_row", canonical["d_row_tensor"], legacy["d_row_tensor"])
    assert_same("d_col", canonical["d_col_tensor"], legacy["d_col_tensor"])
    assert_same("sfd_row", canonical["sfd_row_tensor"], legacy["sfd_row_tensor"].permute(*SF_PHYSICAL_PERMUTE))
    assert_same("sfd_col", canonical["sfd_col_tensor"], legacy["sfd_col_tensor"].permute(*SF_PHYSICAL_PERMUTE))
    if exact_dprob:
        assert_same("dprob", canonical["dprob_tensor"], legacy["dprob_tensor"])
    else:
        # dprob accumulates with atomic float adds; ordering differs between launches.
        torch.testing.assert_close(canonical["dprob_tensor"].reshape(-1), legacy["dprob_tensor"].reshape(-1), rtol=1e-4, atol=1e-4)


@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("b_major", ["k", "n"])
@pytest.mark.parametrize("flat_sf", [False, True], ids=["physical_sf", "flat_sf"])
def test_dglu_canonical_matches_legacy(request, b_major, flat_sf):
    cfg, inputs = dswiglu_case(request)
    legacy = run_dglu(cfg, inputs, canonical=False, b_major=b_major)
    cold = run_dglu(cfg, inputs, canonical=True, b_major=b_major, flat_sf=flat_sf)
    warm = run_dglu(cfg, inputs, canonical=True, b_major=b_major, flat_sf=flat_sf)
    for result in (cold, warm):
        assert_dglu_outputs_equal(result, legacy, inputs["tensor_m"], cfg["n"] * 2, exact_dprob=False)


@torch_fork_set_rng(seed=0)
def test_dglu_canonical_deterministic_dprob(request):
    cfg, inputs = dswiglu_case(request)
    legacy = run_dglu(cfg, inputs, canonical=False, b_major="n", deterministic=True)
    canonical = run_dglu(cfg, inputs, canonical=True, b_major="n", flat_sf=True, deterministic=True)
    assert_dglu_outputs_equal(canonical, legacy, inputs["tensor_m"], cfg["n"] * 2, exact_dprob=True)


def test_dglu_canonical_rejects_short_scale_buffer(request):
    cfg, inputs = dswiglu_case(request)
    run_dglu(cfg, inputs, canonical=True, flat_sf=True)
    short = inputs["sfa_tensor"].permute(*SF_PHYSICAL_PERMUTE).reshape(-1)[:-512]
    with pytest.raises(ValueError, match="complete MMA-packed"):
        run_dglu(cfg, inputs, canonical=True, flat_sf=True, sfa_override=short)


@torch_fork_set_rng(seed=0)
def test_quant_canonical_n_major_b():
    from cudnn import grouped_gemm_quant_wrapper_sm100
    from gemm.cutedsl.test_grouped_gemm_glu_canonical import natural_inputs
    from gemm.cutedsl.test_grouped_gemm_wrapper_memo import mxfp8_inputs

    inputs = mxfp8_inputs([512] * 4)
    natural = natural_inputs(inputs)
    column_wise = inputs["b_tensor"].permute(2, 1, 0).contiguous()

    def call(a, sfa, b, b_major):
        return grouped_gemm_quant_wrapper_sm100(
            a_tensor=a,
            sfa_tensor=sfa,
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            b_tensor=b,
            sfb_tensor=natural["sfb_tensor"],
            b_major=b_major,
            d_dtype=torch.bfloat16,
            sf_vec_size=32,
            use_dynamic_sched=True,
        )

    legacy = call(natural["a_tensor"], natural["sfa_tensor"], column_wise.permute(2, 1, 0), "k")
    canonical = call(natural["a_tensor"], natural["sfa_tensor"], column_wise, "n")
    assert canonical["d_tensor"].shape == legacy["d_tensor"].shape
    assert_same("d", canonical["d_tensor"], legacy["d_tensor"])
    assert_same("amax", canonical["amax_tensor"], legacy["amax_tensor"])


def wgrad_call(inputs, sfa, sfb):
    import cudnn

    return cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        offsets_tensor=inputs["offsets_tensor"],
        output_mode="dense",
        acc_dtype=torch.float32,
        wgrad_dtype=torch.bfloat16,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        sf_vec_size=32,
    )


def wgrad_inputs(group_k_list):
    from gemm.cutedsl.test_grouped_gemm_wgrad_utils import allocate_grouped_gemm_wgrad_tensors, grouped_gemm_wgrad_init

    cfg = grouped_gemm_wgrad_init(torch.float8_e4m3fn, torch.bfloat16, torch.float32, (128, 128), (1, 1), 32, torch.float8_e8m0fnu)
    cfg["group_k_list"] = group_k_list
    return allocate_grouped_gemm_wgrad_tensors(cfg)


@torch_fork_set_rng(seed=0)
def test_wgrad_flat_scale_factors(monkeypatch):
    from cudnn.gemm.cutedsl.grouped.wgrad import api

    monkeypatch.setattr(api, "_cache_of_GroupedGemmWgradSm100Objects", {})
    monkeypatch.setattr(api, "_wgrad_wrapper_memo", {})
    inputs = wgrad_inputs([256, 384])
    legacy = wgrad_call(inputs, inputs["sfa_tensor"], inputs["sfb_tensor"])
    flat = wgrad_call(inputs, inputs["sfa_tensor"].reshape(-1), inputs["sfb_tensor"].reshape(-1))
    assert_same("wgrad", flat["wgrad_tensor"], legacy["wgrad_tensor"])

    entries = len(api._cache_of_GroupedGemmWgradSm100Objects)
    other = wgrad_inputs([512, 384])
    legacy = wgrad_call(other, other["sfa_tensor"], other["sfb_tensor"])
    flat = wgrad_call(other, other["sfa_tensor"].reshape(-1), other["sfb_tensor"].reshape(-1))
    assert_same("wgrad", flat["wgrad_tensor"], legacy["wgrad_tensor"])
    assert len(api._cache_of_GroupedGemmWgradSm100Objects) == entries
