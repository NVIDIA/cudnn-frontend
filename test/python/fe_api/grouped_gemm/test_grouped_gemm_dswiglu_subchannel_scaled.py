# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the Subchannel-Scaled dSwiGLU Grouped GEMM Kernel (SM100+, Rubin dispatch)

Second-level-scaled dSwiGLU-backward block-scaled grouped GEMM: NVFP4 A/B with
per-(1, 16) e4m3 first-level scales plus f32 second-level (subchannel) scales,
fused dSwiGLU epilogue with dprob, optional dbias, second-level output
descales (sfd2), and optional NVFP4 D quantization (interleaved or
deinterleaved gate/up layout). d / d_quant / d_quant_sf / sfd2 are compared
BYTE-EXACT (``torch.equal``) against the ported harness references — do not
loosen these to tolerance-based comparisons; find the real gap instead.
dprob/dbias use tolerances only because their cross-tile atomic accumulation
order is nondeterministic beyond two tiles.

On Rubin the API dispatches to the sm107 kernel module transparently, so the
whole suite doubles as its e2e coverage; the ``sf_e5m3`` ids additionally feed
UE5M3-encoded first-level scales through ``sf_fp8_dtype_override="e5m3"`` (the
e5m3 path is Rubin-only and skips elsewhere).
"""

import pytest
import torch

from test_utils import torch_fork_set_rng
from fe_api.grouped_gemm.test_grouped_gemm_wgrad_utils import _skip_unless_e5m3_supported
from fe_api.grouped_gemm.test_grouped_gemm_dswiglu_subchannel_scaled_utils import (
    _sf_atom_unpack,
    _sf_to_mma,
    build_discrete_pointers_dswiglu,
    make_dswiglu_subchannel_problem,
)


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("SM100 is required")


def _wrapper():
    from cudnn import grouped_gemm_dswiglu_subchannel_scaled_wrapper_sm100

    return grouped_gemm_dswiglu_subchannel_scaled_wrapper_sm100


def _run(problem, discrete=False, **overrides):
    kwargs = dict(
        a_tensor=problem["a_tensor"],
        sfa_tensor=problem["sfa_tensor"],
        sfa2_tensor=problem["sfa2_tensor"],
        c_tensor=problem["c_tensor"],
        padded_offsets=problem["padded_offsets"],
        alpha_tensor=problem["alpha_tensor"],
        beta_tensor=problem["beta_tensor"],
        prob_tensor=problem["prob_tensor"],
        d_dtype=(torch.float4_e2m1fn_x2 if problem["d_quant"] else torch.bfloat16),
        norm_const_tensor=problem["norm_const_tensor"],
        dbias=problem["with_dbias"],
        dbias_dtype=problem["dbias_dtype"],
        block2_shape=problem["block2_shape"],
        d_deinterleaved=problem["d_deinterleaved"],
        sf_fp8_dtype_override=problem["sf_fp8_dtype_override"],
    )
    if discrete:
        ptrs = build_discrete_pointers_dswiglu(problem)
        kwargs.update(
            b_ptrs=ptrs["b_ptrs"],
            sfb_ptrs=ptrs["sfb_ptrs"],
            sfb2_ptrs=ptrs["sfb2_ptrs"],
            n=problem["n"],
            b_dtype=torch.float4_e2m1fn_x2,
        )
    else:
        kwargs.update(
            b_tensor=problem["b_tensor"],
            sfb_tensor=problem["sfb_tensor"],
            sfb2_tensor=problem["sfb2_tensor"],
        )
    kwargs.update(overrides)
    return _wrapper()(**kwargs)


# First-level scale format axis: e4m3 everywhere, e5m3 only on Rubin (skips
# elsewhere). The e5m3 leg re-encodes the same scale values as UE5M3 bytes, so
# the references are unchanged while the kernel must decode them as e5m3 and
# emit e5m3-encoded d_quant_sf.
SF_FORMATS = pytest.mark.parametrize("sf_fmt", ["e4m3", "e5m3"], ids=["sf_e4m3", "sf_e5m3"])


def _check_outputs(out, problem):
    torch.cuda.synchronize()
    mt, n, nd = problem["valid_m"], problem["n"], problem["nd"]
    n2 = 2 * n
    deint = problem["d_deinterleaved"]

    # ---- d (byte-exact) ----
    if not problem["d_quant"]:
        got = out["d_tensor"].squeeze(-1)
        ref = problem["ref_d"]
        assert got.dtype == ref.dtype == torch.bfloat16
        assert torch.equal(got, ref), (
            f"bf16 d differs from reference: max abs diff "
            f"{(got.float() - ref.float()).abs().max().item()}, "
            f"mismatches {(got != ref).sum().item()}/{ref.numel()}"
        )
        assert out["d_quant_tensor"] is None and out["d_quant_sf_tensor"] is None
    else:
        assert out["d_tensor"] is None
        got_q = out["d_quant_tensor"].view(torch.uint8).squeeze(-1)
        ref_q = problem["ref_d_quant"]
        assert got_q.shape == ref_q.shape == (mt, n)
        assert torch.equal(got_q, ref_q), f"d_quant bytes differ: mismatches " f"{(got_q != ref_q).sum().item()}/{ref_q.numel()}"
        got_sf = _sf_atom_unpack(out["d_quant_sf_tensor"].view(torch.uint8), mt, n2)
        ref_sf = problem["ref_d_quant_sf"]
        assert got_sf.shape == ref_sf.shape == (mt, n2 // 16)
        assert torch.equal(got_sf, ref_sf), f"d_quant_sf bytes differ: mismatches " f"{(got_sf != ref_sf).sum().item()}/{ref_sf.numel()}"

    # ---- sfd2 (byte-exact; atomic max commutes) ----
    sfd2 = out["sfd2_tensor"]
    if deint:
        assert torch.equal(sfd2[:, :nd, 0], problem["ref_sfd2_gate"])
        assert torch.equal(sfd2[:, nd:, 0], problem["ref_sfd2_up"])
    else:
        assert torch.equal(sfd2[:, 0::2, 0], problem["ref_sfd2_gate"])
        assert torch.equal(sfd2[:, 1::2, 0], problem["ref_sfd2_up"])
    assert torch.equal(out["sfd2_gate_tensor"][:, :, 0], problem["ref_sfd2_gate"])
    assert torch.equal(out["sfd2_up_tensor"][:, :, 0], problem["ref_sfd2_up"])

    # ---- dprob (atomic f32 adds across n-tiles) ----
    torch.testing.assert_close(out["dprob_tensor"].reshape(-1, 1), problem["ref_dprob"], atol=1e-6, rtol=0)

    # ---- dbias ----
    if problem["with_dbias"]:
        atol = 5e-2 if problem["dbias_dtype"] == torch.bfloat16 else 1e-6
        torch.testing.assert_close(out["dbias_tensor"].squeeze(-1), problem["ref_dbias"], atol=atol, rtol=0)
    else:
        assert out["dbias_tensor"] is None


@pytest.mark.L0
def test_sf_atom_unpack_roundtrip():
    """Validate _sf_atom_unpack independently: scatter flat scales into the
    MMA-tiled view with the established forward converter (_sf_to_mma, the
    same one the kernel's SFA/SFB inputs are built with) and check the unpack
    inverts it exactly."""
    torch.manual_seed(0)
    mt, ksf = 256, 16  # n2 = 256
    flat = torch.rand((1, mt, ksf), dtype=torch.float32) * 4.0 + 0.5
    mma = _sf_to_mma(flat)  # e4m3, MMA-tiled (32, 4, mt//128, 4, rest, 1) view
    got = _sf_atom_unpack(mma.view(torch.uint8), mt, ksf * 16)
    expected = flat[0].to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    assert torch.equal(got, expected)


@pytest.mark.L0
@SF_FORMATS
@pytest.mark.parametrize("with_dbias", [False, True], ids=["nodbias", "dbias"])
@torch_fork_set_rng(seed=0)
def test_dswiglu_subchannel_wrapper_dense_bf16_interleaved(with_dbias, sf_fmt):
    problem = make_dswiglu_subchannel_problem(m_per_expert=256, n=128, k=512, l=2, with_dbias=with_dbias, d_deinterleaved=False, seed=0, sf_fmt=sf_fmt)
    out = _run(problem)
    _check_outputs(out, problem)


@pytest.mark.L0
@SF_FORMATS
@torch_fork_set_rng(seed=1)
def test_dswiglu_subchannel_wrapper_dense_quant_deint(sf_fmt):
    problem = make_dswiglu_subchannel_problem(m_per_expert=256, n=128, k=512, l=2, d_quant=True, d_deinterleaved=True, with_dbias=True, seed=1, sf_fmt=sf_fmt)
    out = _run(problem)
    _check_outputs(out, problem)


@pytest.mark.L0
@torch_fork_set_rng(seed=2)
def test_dswiglu_subchannel_wrapper_dense_quant_interleaved():
    problem = make_dswiglu_subchannel_problem(m_per_expert=256, n=128, k=512, l=2, d_quant=True, d_deinterleaved=False, with_dbias=True, seed=2)
    out = _run(problem)
    _check_outputs(out, problem)


@pytest.mark.L0
@SF_FORMATS
@torch_fork_set_rng(seed=3)
def test_dswiglu_subchannel_wrapper_discrete_quant_deint(sf_fmt):
    # n=256, k=1024 with sgn=sgk=256 -> per-expert SFB2 block is 1x4 f32 =
    # 16 bytes, satisfying the 16B discrete alignment gate with l=2.
    problem = make_dswiglu_subchannel_problem(m_per_expert=256, n=256, k=1024, l=2, d_quant=True, d_deinterleaved=True, seed=3, sf_fmt=sf_fmt)
    out = _run(problem, discrete=True)
    _check_outputs(out, problem)


@pytest.mark.L0
@torch_fork_set_rng(seed=4)
def test_dswiglu_subchannel_check_support_negatives():
    quant_problem = make_dswiglu_subchannel_problem(m_per_expert=256, n=128, k=512, l=2, d_quant=True, d_deinterleaved=True, seed=4)
    bf16_problem = make_dswiglu_subchannel_problem(m_per_expert=256, n=128, k=512, l=2, d_deinterleaved=False, seed=4)

    # Deinterleaved layout requires the quantized (fp4) D output
    with pytest.raises((ValueError, AssertionError)):
        _run(bf16_problem, d_deinterleaved=True)

    # Quantized D output requires norm_const
    with pytest.raises((ValueError, AssertionError)):
        _run(quant_problem, norm_const_tensor=None)

    # Unsupported MMA tiler (only (256, 128))
    with pytest.raises((ValueError, AssertionError)):
        _run(quant_problem, mma_tiler_mn=(256, 256))

    # Unsupported cluster shape (only (2, 1))
    with pytest.raises((ValueError, AssertionError)):
        _run(quant_problem, cluster_shape_mn=(1, 1))

    # sgn must be a multiple of the MMA tile N (128)
    with pytest.raises((ValueError, AssertionError)):
        _run(quant_problem, block2_shape=(1, 192, 256))

    # Generator gate: n % 128 != 0 (output width 2n % 256 != 0)
    with pytest.raises(AssertionError):
        make_dswiglu_subchannel_problem(m_per_expert=256, n=64, k=512, l=2, seed=4)


@pytest.mark.L1
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@torch_fork_set_rng(seed=5)
def test_dswiglu_subchannel_block2_512_quant_deint(discrete):
    # sgn=512 spans 4 N work tiles -> exercises the sfd2 cross-tile arrival
    # protocol (sfd2_n_contrib=4). k=2048 (not 1024) so the discrete per-expert
    # SFB2 block (1x4 f32 = 16 B) passes the 16B discrete alignment gate.
    problem = make_dswiglu_subchannel_problem(
        m_per_expert=256,
        n=512,
        k=2048,
        l=2,
        block2_shape=(1, 512, 512),
        d_quant=True,
        d_deinterleaved=True,
        seed=5,
    )
    out = _run(problem, discrete=discrete)
    _check_outputs(out, problem)


@pytest.mark.L1
@torch_fork_set_rng(seed=6)
def test_dswiglu_subchannel_partial_sfd2_block():
    # n=128 with sgn=256: nd=1, the single sfd2 block covers only half its
    # nominal width (partial block), and the quant fold map broadcast is
    # clamped to the real n/16 width.
    problem = make_dswiglu_subchannel_problem(
        m_per_expert=256,
        n=128,
        k=512,
        l=2,
        block2_shape=(1, 256, 256),
        d_quant=True,
        d_deinterleaved=True,
        seed=6,
    )
    assert problem["nd"] == 1
    out = _run(problem)
    _check_outputs(out, problem)


@pytest.mark.L1
@torch_fork_set_rng(seed=7)
def test_dswiglu_subchannel_dbias_f32():
    # float32 dbias: the testing-only plain-atomic path, tight tolerance.
    problem = make_dswiglu_subchannel_problem(
        m_per_expert=256,
        n=128,
        k=512,
        l=2,
        d_quant=True,
        d_deinterleaved=True,
        with_dbias=True,
        dbias_dtype=torch.float32,
        seed=7,
    )
    out = _run(problem)
    _check_outputs(out, problem)


@pytest.mark.L1
@torch_fork_set_rng(seed=8)
def test_dswiglu_subchannel_larger():
    problem = make_dswiglu_subchannel_problem(m_per_expert=512, n=256, k=1024, l=4, d_quant=True, d_deinterleaved=True, seed=8)
    out = _run(problem)
    _check_outputs(out, problem)


@pytest.mark.L0
@torch_fork_set_rng(seed=9)
def test_dswiglu_subchannel_wrapper_cache():
    from cudnn.gemm.cutedsl.grouped.dswiglu_subchannel_scaled.api import (
        _cache_of_GroupedGemmDswigluSubchannelScaledSm100Objects as cache,
    )

    problem = make_dswiglu_subchannel_problem(m_per_expert=256, n=128, k=512, l=2, d_quant=True, d_deinterleaved=True, seed=9)
    out1 = _run(problem)
    n_cached = len(cache)
    # Second call with the same configuration must reuse the cached object.
    out2 = _run(problem)
    assert len(cache) == n_cached
    torch.cuda.synchronize()
    assert torch.equal(out1["d_quant_tensor"].view(torch.uint8), out2["d_quant_tensor"].view(torch.uint8))
    assert torch.equal(out1["sfd2_tensor"], out2["sfd2_tensor"])
    _check_outputs(out2, problem)


@pytest.mark.L0
@SF_FORMATS
@torch_fork_set_rng(seed=10)
def test_dswiglu_subchannel_dsmem_rowwise_cluster22(sf_fmt):
    """Rowwise-sfd2 DSMEM reduction (sgn=256, cluster (2,2): sgn/cta_n ==
    cluster_n) versus the gmem atomic + counter protocol at the same cluster
    shape: both must match the reference byte-exact AND each other bitwise
    (identical bit-pattern u32 max, order-independent)."""
    problem = make_dswiglu_subchannel_problem(m_per_expert=256, n=256, k=512, l=2, with_dbias=True, d_quant=True, d_deinterleaved=True, seed=10, sf_fmt=sf_fmt)
    out_dsmem = _run(problem, cluster_shape_mn=(2, 2), dsmem_rowwise=True)
    _check_outputs(out_dsmem, problem)
    out_gmem = _run(problem, cluster_shape_mn=(2, 2), dsmem_rowwise=False)
    _check_outputs(out_gmem, problem)
    torch.cuda.synchronize()
    assert torch.equal(out_dsmem["d_quant_tensor"].view(torch.uint8), out_gmem["d_quant_tensor"].view(torch.uint8))
    assert torch.equal(out_dsmem["d_quant_sf_tensor"].view(torch.uint8), out_gmem["d_quant_sf_tensor"].view(torch.uint8))
    assert torch.equal(out_dsmem["sfd2_tensor"], out_gmem["sfd2_tensor"])


@pytest.mark.L0
@torch_fork_set_rng(seed=11)
def test_dswiglu_subchannel_cluster22_bf16_interleaved():
    """cluster_n=2 on the bf16-D path: row_dsmem is inert (generate_sfd off),
    validating the plain cluster_n>1 enablement (scheduler/multicast)."""
    problem = make_dswiglu_subchannel_problem(m_per_expert=256, n=256, k=512, l=2, d_quant=False, d_deinterleaved=False, seed=11)
    out = _run(problem, cluster_shape_mn=(2, 2))
    _check_outputs(out, problem)


@pytest.mark.L1
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@torch_fork_set_rng(seed=12)
def test_dswiglu_subchannel_dsmem_rowwise_sgn512_cluster24(discrete):
    """sgn=512 with cluster (2,4): four N work tiles per sfd2 block, all
    cluster peers — the widest rowwise DSMEM geometry."""
    problem = make_dswiglu_subchannel_problem(
        m_per_expert=256, n=512, k=2048, l=2, with_dbias=True, d_quant=True, d_deinterleaved=True, block2_shape=(1, 512, 512), seed=12
    )
    out = _run(problem, discrete=discrete, cluster_shape_mn=(2, 4), dsmem_rowwise=True)
    _check_outputs(out, problem)


@pytest.mark.L0
@torch_fork_set_rng(seed=13)
def test_dswiglu_subchannel_cluster_negatives():
    problem = make_dswiglu_subchannel_problem(m_per_expert=256, n=384, k=512, l=2, d_quant=True, d_deinterleaved=True, seed=13)
    # ceil(384/128) = 3 N work tiles is not a cluster_n=2 multiple.
    with pytest.raises((ValueError, AssertionError)):
        _run(problem, cluster_shape_mn=(2, 2))
    problem2 = make_dswiglu_subchannel_problem(m_per_expert=256, n=256, k=512, l=2, d_quant=True, d_deinterleaved=True, seed=13)
    # cluster_m=4 is not supported by the FE API (per-expert-M gate not
    # verifiable host-side on ragged inputs).
    with pytest.raises((ValueError, AssertionError)):
        _run(problem2, cluster_shape_mn=(4, 2))


@pytest.mark.L0
@torch_fork_set_rng(seed=14)
def test_dswiglu_subchannel_e5m3_rejected_off_rubin(monkeypatch):
    """e5m3 scale factors decode only in the sm107 MMA op; pretend to be Blackwell."""
    import cudnn.api_base as api_base
    from cudnn import GroupedGemmDswigluSubchannelScaledSm100

    monkeypatch.setattr(api_base, "get_device_type", lambda: "blackwell")
    problem = make_dswiglu_subchannel_problem(m_per_expert=256, n=128, k=512, l=2, d_quant=True, d_deinterleaved=True, seed=14)
    mt, n2 = problem["valid_m"], 2 * problem["n"]
    d_quant = torch.zeros((1, mt, n2 // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2).permute(1, 2, 0)
    d_quant_sf = torch.zeros((1, mt // 128, (n2 // 16) // 4, 32, 4, 4), dtype=torch.float8_e4m3fn, device="cuda").permute(3, 4, 1, 5, 2, 0)
    dprob = torch.zeros((mt, 1, 1), dtype=torch.float32, device="cuda")
    sfd2 = torch.zeros((1, 2 * problem["nd"], problem["sfd2_rows"]), dtype=torch.float32, device="cuda").permute(2, 1, 0)
    api = GroupedGemmDswigluSubchannelScaledSm100(
        sample_a=problem["a_tensor"],
        sample_sfa=problem["sfa_tensor"],
        sample_sfa2=problem["sfa2_tensor"],
        sample_c=problem["c_tensor"],
        sample_padded_offsets=problem["padded_offsets"],
        sample_alpha=problem["alpha_tensor"],
        sample_beta=problem["beta_tensor"],
        sample_prob=problem["prob_tensor"],
        sample_d=d_quant,
        sample_dprob=dprob,
        sample_sfd2=sfd2,
        sample_d_quant_sf=d_quant_sf,
        sample_norm_const=problem["norm_const_tensor"],
        sample_b=problem["b_tensor"],
        sample_sfb=problem["sfb_tensor"],
        sample_sfb2=problem["sfb2_tensor"],
        sf_fp8_dtype_override="e5m3",
    )
    with pytest.raises(ValueError, match="requires Rubin"):
        api.check_support()

    # Unknown override formats are rejected everywhere.
    with pytest.raises(ValueError, match="sf_fp8_dtype_override must be"):
        _run(problem, sf_fp8_dtype_override="e4m3")


@pytest.mark.L0
@torch_fork_set_rng(seed=15)
def test_dswiglu_subchannel_e5m3_is_not_cached_as_e4m3():
    """sf_fp8_dtype_override must take part in the compile cache key: identical
    scale bytes decode differently under E4M3 and UE5M3."""
    _skip_unless_e5m3_supported()
    problem = make_dswiglu_subchannel_problem(m_per_expert=256, n=128, k=512, l=2, d_quant=False, d_deinterleaved=False, seed=15)
    d_e4m3 = _run(problem)["d_tensor"].float().clone()
    d_e5m3 = _run(problem, sf_fp8_dtype_override="e5m3")["d_tensor"].float().clone()
    torch.cuda.synchronize()
    assert not torch.equal(d_e4m3, d_e5m3), "e5m3 and e4m3 produced identical output from identical scale-factor bytes"
