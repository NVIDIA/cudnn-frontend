# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical layouts for the BF16 grouped GEMM backends."""

import pytest
import torch

from gemm.cutedsl.test_grouped_gemm_bf16_utils import make_grouped_gemm_bf16_problem
from gemm.cutedsl.test_grouped_gemm_dglu_bf16_utils import make_grouped_gemm_dglu_bf16_problem
from gemm.cutedsl.test_grouped_gemm_glu_bf16_utils import make_grouped_gemm_glu_bf16_problem
from gemm.cutedsl.test_grouped_gemm_wgrad_bf16_utils import make_grouped_gemm_wgrad_bf16_problem

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")


def assert_same(name, actual, expected):
    assert torch.equal(actual.reshape(-1).view(torch.int16), expected.reshape(-1).view(torch.int16)), name


def glu_call(problem, canonical):
    import cudnn

    a, b, prob = problem["a"], problem["b"], problem["prob"]
    if canonical:
        a, b, prob = a.squeeze(-1), problem["b_reference"], prob.view(-1)
    return cudnn.grouped_gemm_glu_wrapper_sm100(
        a_tensor=a,
        sfa_tensor=None,
        padded_offsets=problem["offsets"],
        alpha_tensor=problem["alpha"],
        b_tensor=b,
        prob_tensor=prob,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        generate_c=True,
    )


def test_glu_canonical_matches_legacy():
    problem = make_grouped_gemm_glu_bf16_problem()
    m, n = problem["a"].shape[0], problem["n"]
    legacy = glu_call(problem, canonical=False)
    for _ in range(2):
        canonical = glu_call(problem, canonical=True)
        assert canonical["c_tensor"].shape == (m, n) and canonical["d_tensor"].shape == (m, n // 2)
        assert_same("c", canonical["c_tensor"], legacy["c_tensor"])
        assert_same("d", canonical["d_tensor"], legacy["d_tensor"])


def gemm_call(problem, layout, d_tensor=None):
    import cudnn

    a, b, prob = problem["a"], problem["b"], problem["prob"]
    b_major = "k"
    if layout != "legacy":
        a, b, prob = a.squeeze(-1), problem["b_storage"], prob.view(-1)
    if layout == "canonical_n":
        b, b_major = b.transpose(1, 2).contiguous(), "n"
    return cudnn.grouped_gemm_wrapper_sm100(
        a_tensor=a,
        padded_offsets=problem["offsets"],
        alpha_tensor=problem["alpha"],
        b_tensor=b,
        b_major=b_major,
        prob_tensor=prob,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        d_tensor=d_tensor,
        generate_c=True,
    )


@pytest.mark.parametrize("layout", ["canonical", "canonical_n"])
def test_gemm_canonical_matches_legacy(layout):
    problem = make_grouped_gemm_bf16_problem()
    m, n = problem["a"].shape[0], problem["n"]
    legacy = gemm_call(problem, "legacy")
    for _ in range(2):
        canonical = gemm_call(problem, layout)
        assert canonical["c_tensor"].shape == (m, n) and canonical["d_tensor"].shape == (m, n)
        assert_same("c", canonical["c_tensor"], legacy["c_tensor"])
        assert_same("d", canonical["d_tensor"], legacy["d_tensor"])
    d_out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    assert gemm_call(problem, layout, d_tensor=d_out)["d_tensor"] is d_out
    assert_same("caller d", d_out, legacy["d_tensor"])


def dglu_call(problem, layout):
    import cudnn

    a, b, c, prob = problem["a"], problem["b"], problem["c"], problem["prob"]
    dprob = torch.zeros_like(problem["dprob"])
    b_major = "k"
    if layout != "legacy":
        a, b, c, prob, dprob = a.squeeze(-1), problem["b_reference"], c.squeeze(-1), prob.view(-1), dprob.view(-1)
    if layout == "canonical_n":
        b, b_major = b.transpose(1, 2).contiguous(), "n"
    return cudnn.grouped_gemm_dglu_wrapper_sm100(
        a_tensor=a,
        c_tensor=c,
        sfa_tensor=None,
        padded_offsets=problem["offsets"],
        alpha_tensor=problem["alpha"],
        beta_tensor=problem["beta"],
        prob_tensor=prob,
        dprob_tensor=dprob,
        b_tensor=b,
        b_major=b_major,
        d_dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("layout", ["canonical", "canonical_n"])
def test_dglu_canonical_matches_legacy(layout):
    problem = make_grouped_gemm_dglu_bf16_problem()
    m = problem["a"].shape[0]
    legacy = dglu_call(problem, "legacy")
    for _ in range(2):
        canonical = dglu_call(problem, layout)
        assert canonical["d_row_tensor"].shape == (m, 2 * problem["n"]) and canonical["dprob_tensor"].shape == (m,)
        assert_same("d_row", canonical["d_row_tensor"], legacy["d_row_tensor"])
        # dprob accumulates with atomic float adds; ordering differs between launches.
        torch.testing.assert_close(canonical["dprob_tensor"].reshape(-1), legacy["dprob_tensor"].reshape(-1), rtol=1e-4, atol=1e-4)


def wgrad_call(problem, layout):
    import cudnn

    return cudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=problem["a"],
        b_tensor=problem["b"],
        sfa_tensor=None,
        sfb_tensor=None,
        offsets_tensor=problem["offsets"],
        output_mode="dense",
        acc_dtype=torch.float32,
        wgrad_dtype=problem["output_dtype"],
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )


@pytest.mark.parametrize(
    "make_problem,call",
    [
        (make_grouped_gemm_glu_bf16_problem, lambda problem, layout: glu_call(problem, layout == "canonical")),
        (make_grouped_gemm_bf16_problem, gemm_call),
        (make_grouped_gemm_dglu_bf16_problem, dglu_call),
        (make_grouped_gemm_wgrad_bf16_problem, wgrad_call),
    ],
    ids=["glu", "gemm", "dglu", "wgrad"],
)
@pytest.mark.parametrize("layout", ["legacy", "canonical"])
def test_execute_does_not_read_offsets_to_host(make_problem, call, layout):
    problem = make_problem()
    call(problem, layout)
    # A new offsets tensor of the same shape reuses the compiled kernel and must not be copied to the host.
    problem["offsets"] = problem["offsets"].clone()
    torch.cuda.set_sync_debug_mode("error")
    try:
        call(problem, layout)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_class_api_rejects_layout_mismatch():
    import cudnn

    problem = make_grouped_gemm_bf16_problem()
    m, n = problem["a"].shape[0], problem["n"]
    a, prob = problem["a"].squeeze(-1), problem["prob"].view(-1)
    c, d = (torch.empty((m, n), dtype=torch.bfloat16, device="cuda") for _ in range(2))
    api = cudnn.GroupedGemmSm100(
        sample_a=a,
        sample_c=c,
        sample_d=d,
        sample_padded_offsets=problem["offsets"],
        sample_alpha=problem["alpha"],
        sample_b=problem["b_storage"],
        sample_prob=prob,
    )
    assert api.check_support()
    api.compile()
    kwargs = dict(c_tensor=c, d_tensor=d, padded_offsets=problem["offsets"], alpha_tensor=problem["alpha"], b_tensor=problem["b_storage"], prob_tensor=prob)
    api.execute(a_tensor=a, **kwargs)
    with pytest.raises(ValueError, match="layout mismatch"):
        api.execute(a_tensor=problem["a"], **kwargs)
