# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the Unfused Subchannel-Scaled Grouped GEMM Kernel (SM100+)

Second-level-scaled unfused block-scaled grouped GEMM for MoE workloads:
NVFP4 A/B with per-(1, 16) e4m3 first-level scales plus f32 second-level
(subchannel) scales, BF16 D output. The kernel output is compared BYTE-EXACT
(``torch.equal``) against the pure-torch reference -- do not loosen this to a
tolerance-based comparison; find the real gap instead.
"""

import pytest
import torch

from test_utils import torch_fork_set_rng
from fe_api.grouped_gemm.test_grouped_gemm_unfused_subchannel_scaled_utils import (
    build_discrete_pointers,
    make_unfused_subchannel_scaled_problem,
)


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("SM100 is required")


def _wrapper():
    from cudnn import grouped_gemm_unfused_subchannel_scaled_wrapper_sm100

    return grouped_gemm_unfused_subchannel_scaled_wrapper_sm100


def _run_dense(problem, **overrides):
    kwargs = dict(
        a_tensor=problem["a_tensor"],
        sfa_tensor=problem["sfa_tensor"],
        sfa2_tensor=problem["sfa2_tensor"],
        padded_offsets=problem["padded_offsets"],
        alpha_tensor=problem["alpha_tensor"],
        prob_tensor=problem["prob_tensor"],
        b_tensor=problem["b_tensor"],
        sfb_tensor=problem["sfb_tensor"],
        sfb2_tensor=problem["sfb2_tensor"],
        bias_tensor=problem["bias_tensor"],
        block2_shape=problem["block2_shape"],
    )
    kwargs.update(overrides)
    return _wrapper()(**kwargs)


def _assert_byte_exact(out_d, ref_d):
    got = out_d.squeeze(-1)
    torch.cuda.synchronize()
    assert got.dtype == ref_d.dtype == torch.bfloat16
    assert torch.equal(got, ref_d), (
        f"kernel output differs from reference: "
        f"max abs diff {(got.float() - ref_d.float()).abs().max().item()}, "
        f"mismatches {(got != ref_d).sum().item()}/{ref_d.numel()}"
    )


@pytest.mark.L0
@pytest.mark.parametrize(
    "mma_tiler_mn,cluster_shape_mn",
    [((256, 128), (2, 1)), ((256, 256), (2, 1)), ((256, 128), (2, 2))],
    ids=["t256x128-c2x1", "t256x256-c2x1", "t256x128-c2x2"],
)
@torch_fork_set_rng(seed=0)
def test_unfused_subchannel_scaled_wrapper_dense_fp4(mma_tiler_mn, cluster_shape_mn):
    problem = make_unfused_subchannel_scaled_problem(m_per_expert=256, n=256, k=512, l=2)
    out = _run_dense(problem, mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn)
    _assert_byte_exact(out["d_tensor"], problem["ref_d"])


@pytest.mark.L0
@torch_fork_set_rng(seed=1)
def test_unfused_subchannel_scaled_wrapper_dense_bias():
    problem = make_unfused_subchannel_scaled_problem(m_per_expert=256, n=256, k=512, l=2, with_bias=True, seed=1)
    out = _run_dense(problem)
    _assert_byte_exact(out["d_tensor"], problem["ref_d"])


@pytest.mark.L0
@torch_fork_set_rng(seed=2)
def test_unfused_subchannel_scaled_class_api_dense():
    from cudnn import GroupedGemmUnfusedSubchannelScaledSm100

    problem = make_unfused_subchannel_scaled_problem(m_per_expert=256, n=256, k=512, l=2, seed=2)
    valid_m, n = problem["valid_m"], problem["n"]
    d_tensor = torch.empty_strided((valid_m, n, 1), (n, 1, valid_m * n), dtype=torch.bfloat16, device="cuda")

    api = GroupedGemmUnfusedSubchannelScaledSm100(
        sample_a=problem["a_tensor"],
        sample_sfa=problem["sfa_tensor"],
        sample_sfa2=problem["sfa2_tensor"],
        sample_padded_offsets=problem["padded_offsets"],
        sample_alpha=problem["alpha_tensor"],
        sample_prob=problem["prob_tensor"],
        sample_d=d_tensor,
        sample_b=problem["b_tensor"],
        sample_sfb=problem["sfb_tensor"],
        sample_sfb2=problem["sfb2_tensor"],
        block2_shape=problem["block2_shape"],
    )
    assert api.check_support()
    api.compile()
    api.execute(
        a_tensor=problem["a_tensor"],
        sfa_tensor=problem["sfa_tensor"],
        sfa2_tensor=problem["sfa2_tensor"],
        padded_offsets=problem["padded_offsets"],
        alpha_tensor=problem["alpha_tensor"],
        prob_tensor=problem["prob_tensor"],
        d_tensor=d_tensor,
        b_tensor=problem["b_tensor"],
        sfb_tensor=problem["sfb_tensor"],
        sfb2_tensor=problem["sfb2_tensor"],
    )
    _assert_byte_exact(d_tensor, problem["ref_d"])


@pytest.mark.L0
@torch_fork_set_rng(seed=3)
def test_unfused_subchannel_scaled_wrapper_discrete_fp4():
    # n=512, k=1024 with sgn=sgk=256 -> per-expert SFB2 block is 2x4 f32 =
    # 32 bytes, satisfying the 16B discrete alignment gate with l=2.
    problem = make_unfused_subchannel_scaled_problem(m_per_expert=256, n=512, k=1024, l=2, seed=3)
    ptrs = build_discrete_pointers(problem)
    out = _wrapper()(
        a_tensor=problem["a_tensor"],
        sfa_tensor=problem["sfa_tensor"],
        sfa2_tensor=problem["sfa2_tensor"],
        padded_offsets=problem["padded_offsets"],
        alpha_tensor=problem["alpha_tensor"],
        prob_tensor=problem["prob_tensor"],
        b_ptrs=ptrs["b_ptrs"],
        sfb_ptrs=ptrs["sfb_ptrs"],
        sfb2_ptrs=ptrs["sfb2_ptrs"],
        n=problem["n"],
        b_dtype=torch.float4_e2m1fn_x2,
        block2_shape=problem["block2_shape"],
    )
    _assert_byte_exact(out["d_tensor"], problem["ref_d"])


@pytest.mark.L1
@torch_fork_set_rng(seed=4)
def test_unfused_subchannel_scaled_block2_512():
    problem = make_unfused_subchannel_scaled_problem(m_per_expert=256, n=512, k=1024, l=2, block2_shape=(1, 512, 512), seed=4)
    out = _run_dense(problem)
    _assert_byte_exact(out["d_tensor"], problem["ref_d"])


@pytest.mark.L1
@torch_fork_set_rng(seed=5)
def test_unfused_subchannel_scaled_larger_shape():
    problem = make_unfused_subchannel_scaled_problem(m_per_expert=512, n=512, k=1024, l=4, seed=5)
    out = _run_dense(problem)
    _assert_byte_exact(out["d_tensor"], problem["ref_d"])


@pytest.mark.L0
@torch_fork_set_rng(seed=6)
def test_unfused_subchannel_scaled_wrapper_cache_and_preallocated_d():
    from cudnn.gemm.cutedsl.grouped.unfused_subchannel_scaled.api import (
        _cache_of_GroupedGemmUnfusedSubchannelScaledSm100Objects as cache,
    )

    problem = make_unfused_subchannel_scaled_problem(m_per_expert=256, n=256, k=512, l=2, seed=6)
    out1 = _run_dense(problem)
    n_cached = len(cache)
    # Second call with the same configuration must reuse the cached object.
    out2 = _run_dense(problem)
    assert len(cache) == n_cached
    assert torch.equal(out1["d_tensor"], out2["d_tensor"])
    # Preallocated output path
    valid_m, n = problem["valid_m"], problem["n"]
    d_pre = torch.empty_strided((valid_m, n, 1), (n, 1, valid_m * n), dtype=torch.bfloat16, device="cuda")
    out3 = _run_dense(problem, d_tensor=d_pre)
    assert out3["d_tensor"] is d_pre
    _assert_byte_exact(d_pre, problem["ref_d"])


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
def test_unfused_subchannel_scaled_check_support_negatives():
    problem = make_unfused_subchannel_scaled_problem(m_per_expert=256, n=256, k=512, l=2, seed=7)

    # Unsupported MMA tiler
    with pytest.raises((ValueError, AssertionError)):
        _run_dense(problem, mma_tiler_mn=(128, 256))

    # Missing prob (the kernel always fuses the prob multiply)
    with pytest.raises(ValueError):
        _run_dense(problem, prob_tensor=None)

    # SFA2 with the wrong (row-major) stride: shape passes, stride must not
    rows, cols = problem["sfa2_tensor"].shape[0], problem["sfa2_tensor"].shape[1]
    sfa2_bad = torch.zeros((rows, cols, 1), dtype=torch.float32, device="cuda")  # stride (cols, 1, 1)
    sfa2_bad[:, :, 0] = problem["sfa2_tensor"][:, :, 0]
    with pytest.raises((ValueError, AssertionError)):
        _run_dense(problem, sfa2_tensor=sfa2_bad)

    # e5m3 override requires Rubin
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (10, 7):
        with pytest.raises((ValueError, AssertionError)):
            _run_dense(problem, sf_fp8_dtype_override="e5m3")


@pytest.mark.L0
@torch_fork_set_rng(seed=8)
def test_unfused_subchannel_scaled_discrete_sfb2_alignment_gate():
    # n=256, k=256, sgn=sgk=256 -> per-expert SFB2 block is 1x1 f32 = 4 bytes:
    # with l=2 the second expert's base cannot be 16B-aligned -> must be rejected.
    problem = make_unfused_subchannel_scaled_problem(m_per_expert=256, n=256, k=256, l=2, seed=8)
    ptrs = build_discrete_pointers(problem)
    with pytest.raises((ValueError, AssertionError)):
        _wrapper()(
            a_tensor=problem["a_tensor"],
            sfa_tensor=problem["sfa_tensor"],
            sfa2_tensor=problem["sfa2_tensor"],
            padded_offsets=problem["padded_offsets"],
            alpha_tensor=problem["alpha_tensor"],
            prob_tensor=problem["prob_tensor"],
            b_ptrs=ptrs["b_ptrs"],
            sfb_ptrs=ptrs["sfb_ptrs"],
            sfb2_ptrs=ptrs["sfb2_ptrs"],
            n=problem["n"],
            b_dtype=torch.float4_e2m1fn_x2,
            block2_shape=problem["block2_shape"],
        )
