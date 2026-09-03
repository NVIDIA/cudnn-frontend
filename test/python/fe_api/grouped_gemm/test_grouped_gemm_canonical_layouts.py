# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Canonical (natural row-major) layout acceptance for the contiguous grouped GEMM
SwiGLU forward and dSwiGLU backward wrappers.

Each test runs the same problem twice on identical device buffers -- once through
the pre-permuted kernel-facing views (legacy) and once through the canonical
forms: A (sum_m, k) row-major, B (l, n, k) C-contiguous, SFA/SFB as dense
C-contiguous buffers (physical atom shape or flat 1-D), prob (sum_m,) -- and
requires identical results. Canonical calls return natural-shaped outputs.
"""

import pytest
import torch
from test_utils import torch_fork_set_rng

from fe_api.grouped_gemm.test_grouped_gemm_swiglu_utils import (
    grouped_gemm_swiglu_init,
    allocate_grouped_gemm_input_tensors,
)
from fe_api.grouped_gemm.test_grouped_gemm_dswiglu_utils import (
    allocate_grouped_gemm_dswiglu_tensors,
)

# Inverse of the (3, 4, 1, 5, 2, 0) MMA permute: recovers the physical C-contiguous
# (l, mn//128, rest_k, 32, 4, 4) allocation underneath a kernel-facing SF view.
SF_PHYSICAL_PERMUTE = (5, 2, 4, 0, 1, 3)


def swiglu_case(request, ab_dtype, d_dtype, sf_vec_size, sf_dtype):
    cfg = grouped_gemm_swiglu_init(
        request=request,
        ab_dtype=ab_dtype,
        c_dtype=torch.bfloat16,
        d_dtype=d_dtype,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
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
    )
    return cfg, inputs


def run_swiglu(cfg, inputs, canonical, flat_sf=False, prob_dtype=None, alpha=...):
    from cudnn import grouped_gemm_swiglu_wrapper_sm100

    prob = inputs["prob_tensor"]
    if alpha is ...:
        alpha = inputs["alpha_tensor"]
    if canonical:
        a = inputs["a_tensor"].squeeze(-1)
        b = inputs["b_tensor"].permute(2, 0, 1)
        assert b.is_contiguous()
        sfa = inputs["sfa_tensor"].permute(*SF_PHYSICAL_PERMUTE)
        sfb = inputs["sfb_tensor"].permute(*SF_PHYSICAL_PERMUTE)
        assert sfa.is_contiguous() and sfb.is_contiguous()
        if flat_sf:
            sfa = sfa.reshape(-1)
            sfb = sfb.reshape(-1)
        prob = prob.view(-1)
    else:
        a, b = inputs["a_tensor"], inputs["b_tensor"]
        sfa, sfb = inputs["sfa_tensor"], inputs["sfb_tensor"]
    if prob_dtype is not None:
        prob = prob.to(prob_dtype)
    return grouped_gemm_swiglu_wrapper_sm100(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=alpha,
        norm_const_tensor=inputs.get("norm_const_tensor"),
        prob_tensor=prob,
        d_dtype=cfg["d_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
    )


def assert_same(name, canonical_t, legacy_t):
    if legacy_t is None and canonical_t is None:
        return
    legacy_flat = legacy_t.reshape(-1)
    canonical_flat = canonical_t.reshape(-1)
    if canonical_flat.dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2):
        legacy_flat = legacy_flat.view(torch.uint8)
        canonical_flat = canonical_flat.view(torch.uint8)
    torch.testing.assert_close(canonical_flat, legacy_flat, rtol=0, atol=0, msg=lambda m: f"{name}: {m}")


SWIGLU_CASES = [
    pytest.param(torch.float8_e4m3fn, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu, id="fp8"),
    pytest.param(torch.float4_e2m1fn_x2, torch.bfloat16, 16, torch.float8_e8m0fnu, id="fp4"),
]


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("flat_sf", [False, True], ids=["physical_sf", "flat_sf"])
@pytest.mark.parametrize("ab_dtype,d_dtype,sf_vec_size,sf_dtype", SWIGLU_CASES)
def test_grouped_gemm_swiglu_canonical_matches_legacy(request, ab_dtype, d_dtype, sf_vec_size, sf_dtype, flat_sf):
    try:
        cfg, inputs = swiglu_case(request, ab_dtype, d_dtype, sf_vec_size, sf_dtype)
        legacy = run_swiglu(cfg, inputs, canonical=False)
        canonical = run_swiglu(cfg, inputs, canonical=True, flat_sf=flat_sf)
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    m = inputs["tensor_m"]
    n, n_out = cfg["n"], cfg["n"] // 2
    assert canonical["c_tensor"].shape == (m, n)
    assert canonical["d_tensor"].shape == (m, n_out)
    assert canonical["d_col_tensor"].shape == (m, n_out)
    assert_same("c", canonical["c_tensor"], legacy["c_tensor"])
    assert_same("d", canonical["d_tensor"], legacy["d_tensor"])
    if legacy["sfd_row_tensor"] is not None:
        # d_col is only written on the quantized (generate_sfd) path
        assert_same("d_col", canonical["d_col_tensor"], legacy["d_col_tensor"])
        assert canonical["sfd_row_tensor"].is_contiguous()
        assert canonical["sfd_col_tensor"].is_contiguous()
        assert_same("sfd_row", canonical["sfd_row_tensor"], legacy["sfd_row_tensor"].permute(*SF_PHYSICAL_PERMUTE))
        assert_same("sfd_col", canonical["sfd_col_tensor"], legacy["sfd_col_tensor"].permute(*SF_PHYSICAL_PERMUTE))
    if legacy["amax_tensor"] is not None:
        assert_same("amax", canonical["amax_tensor"], legacy["amax_tensor"])


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_swiglu_canonical_bf16_prob(request):
    try:
        cfg, inputs = swiglu_case(request, torch.float8_e4m3fn, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu)
        # prob values are small integers, exactly representable in bf16, so the
        # bf16-prob run must match the fp32-prob run bitwise.
        legacy = run_swiglu(cfg, inputs, canonical=False)
        canonical = run_swiglu(cfg, inputs, canonical=True, prob_dtype=torch.bfloat16)
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    assert_same("d", canonical["d_tensor"], legacy["d_tensor"])
    assert_same("c", canonical["c_tensor"], legacy["c_tensor"])


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_swiglu_alpha_defaults_to_ones(request):
    try:
        cfg, inputs = swiglu_case(request, torch.float8_e4m3fn, torch.float8_e4m3fn, 32, torch.float8_e8m0fnu)
        ones = torch.ones_like(inputs["alpha_tensor"])
        explicit = run_swiglu(cfg, inputs, canonical=True, alpha=ones)
        defaulted = run_swiglu(cfg, inputs, canonical=True, alpha=None)
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    assert_same("d", defaulted["d_tensor"], explicit["d_tensor"])
    assert_same("c", defaulted["c_tensor"], explicit["c_tensor"])


def dswiglu_case(request):
    cfg = grouped_gemm_swiglu_init(
        request=request,
        ab_dtype=torch.float8_e4m3fn,
        c_dtype=torch.bfloat16,
        d_dtype=torch.float8_e4m3fn,
        cd_major="n",
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
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
    )
    inputs, _ = allocate_grouped_gemm_dswiglu_tensors(
        tensor_m=inputs["tensor_m"],
        n=cfg["n"],
        l=cfg["l"],
        ab_dtype=cfg["ab_dtype"],
        c_dtype=cfg["c_dtype"],
        d_dtype=cfg["d_dtype"],
        cd_major=cfg["cd_major"],
        sf_dtype=cfg["sf_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        input_tensors=inputs,
    )
    return cfg, inputs


def run_dswiglu(cfg, inputs, canonical, flat_sf=False, prob_dtype=None, alpha=...):
    from cudnn import grouped_gemm_dswiglu_wrapper_sm100

    prob = inputs["prob_tensor"]
    if alpha is ...:
        alpha = inputs["alpha_tensor"]
    if canonical:
        a = inputs["a_tensor"].squeeze(-1)
        b = inputs["b_tensor"].permute(2, 0, 1)
        c = inputs["c_tensor"].squeeze(-1)
        sfa = inputs["sfa_tensor"].permute(*SF_PHYSICAL_PERMUTE)
        sfb = inputs["sfb_tensor"].permute(*SF_PHYSICAL_PERMUTE)
        assert b.is_contiguous() and sfa.is_contiguous() and sfb.is_contiguous()
        if flat_sf:
            sfa = sfa.reshape(-1)
            sfb = sfb.reshape(-1)
        prob = prob.view(-1)
    else:
        a, b, c = inputs["a_tensor"], inputs["b_tensor"], inputs["c_tensor"]
        sfa, sfb = inputs["sfa_tensor"], inputs["sfb_tensor"]
    if prob_dtype is not None:
        prob = prob.to(prob_dtype)
    return grouped_gemm_dswiglu_wrapper_sm100(
        a_tensor=a,
        b_tensor=b,
        c_tensor=c,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=alpha,
        beta_tensor=inputs["beta_tensor"],
        prob_tensor=prob,
        norm_const_tensor=inputs.get("norm_const_tensor"),
        d_dtype=cfg["d_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("flat_sf", [False, True], ids=["physical_sf", "flat_sf"])
def test_grouped_gemm_dswiglu_canonical_matches_legacy(request, flat_sf):
    try:
        cfg, inputs = dswiglu_case(request)
        legacy = run_dswiglu(cfg, inputs, canonical=False)
        canonical = run_dswiglu(cfg, inputs, canonical=True, flat_sf=flat_sf)
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    m = inputs["tensor_m"]
    n2 = cfg["n"] * 2
    assert canonical["d_row_tensor"].shape == (m, n2)
    assert canonical["d_col_tensor"].shape == (m, n2)
    assert canonical["dprob_tensor"].shape == (m,)
    assert_same("d_row", canonical["d_row_tensor"], legacy["d_row_tensor"])
    assert_same("d_col", canonical["d_col_tensor"], legacy["d_col_tensor"])
    # dprob accumulates with atomic float adds; ordering differs between launches.
    torch.testing.assert_close(canonical["dprob_tensor"].reshape(-1), legacy["dprob_tensor"].reshape(-1), rtol=1e-4, atol=1e-4)
    if legacy["sfd_row_tensor"] is not None:
        assert canonical["sfd_row_tensor"].is_contiguous()
        assert_same("sfd_row", canonical["sfd_row_tensor"], legacy["sfd_row_tensor"].permute(*SF_PHYSICAL_PERMUTE))
        assert_same("sfd_col", canonical["sfd_col_tensor"], legacy["sfd_col_tensor"].permute(*SF_PHYSICAL_PERMUTE))
    if legacy["amax_tensor"] is not None:
        assert_same("amax", canonical["amax_tensor"], legacy["amax_tensor"])


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_dswiglu_canonical_bf16_prob(request):
    try:
        cfg, inputs = dswiglu_case(request)
        legacy = run_dswiglu(cfg, inputs, canonical=False)
        canonical = run_dswiglu(cfg, inputs, canonical=True, prob_dtype=torch.bfloat16)
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    assert_same("d_row", canonical["d_row_tensor"], legacy["d_row_tensor"])
    torch.testing.assert_close(canonical["dprob_tensor"].reshape(-1), legacy["dprob_tensor"].reshape(-1), rtol=1e-4, atol=1e-4)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_grouped_gemm_dswiglu_alpha_defaults_to_ones(request):
    try:
        cfg, inputs = dswiglu_case(request)
        ones = torch.ones_like(inputs["alpha_tensor"])
        explicit = run_dswiglu(cfg, inputs, canonical=True, alpha=ones)
        defaulted = run_dswiglu(cfg, inputs, canonical=True, alpha=None)
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")
    assert_same("d_row", defaulted["d_row_tensor"], explicit["d_row_tensor"])
    torch.testing.assert_close(defaulted["dprob_tensor"], explicit["dprob_tensor"], rtol=1e-4, atol=1e-4)
