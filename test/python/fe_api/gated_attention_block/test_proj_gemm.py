# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stages (1) and (6) — the two dense projections, on the shipped FROST GEMM.

These write no kernel, so what is under test is the WIRING and the block's
claims about it: that the weight is consumed in its checkpoint ``[N, K]``
layout with no repack, that the ``frost_gemm`` plan is actually the one that
ran, and that the result matches a reference.

The engine is an sm100-family template (sm_100..sm_119), so on any other device
these skip — and the skip is deliberate rather than a silent pass: a projection
measured on a backend plan would not be measuring FROST at all.
"""

import dataclasses
import os

import pytest
import torch

pytestmark = pytest.mark.L0

# The FROST GEMM is opt-in and the flag is read at import; set it before the
# first `import cudnn` in this process. conftest's autouse fixture covers the
# SDPA engines, not this one.

from cudnn.gated_attention_block import GatedAttentionBlockGeometry  # noqa: E402
from cudnn.gated_attention_block.api import _FusedQkvProjection, _out_projection, _qkv_gate_projection  # noqa: E402
from cudnn.gated_attention_block.kernels.proj_gemm import (  # noqa: E402
    FusedProjGemmPlan,
    NormRopeFusionParams,
    build_fused_proj_gemm,
    build_proj_gemm,
    run_fused_proj_gemm,
    run_fused_proj_gemm_fp8,
    run_proj_gemm,
    validate_norm_rope_params,
)


def _frost_gemm_unavailable() -> str | None:
    if not torch.cuda.is_available():
        return "no CUDA device"
    from cudnn.gemm.frost.kernel_registry import PIPELINE_ARCH_RANGES

    cc = torch.cuda.get_device_capability()
    arch = cc[0] * 10 + cc[1]
    spans = PIPELINE_ARCH_RANGES.get("sm100", ())
    if not any(lo <= arch < hi for lo, hi in spans):
        return f"the sm100 GEMM template family does not run on sm_{arch}"
    return None


requires_frost_gemm = pytest.mark.skipif(_frost_gemm_unavailable() is not None, reason=_frost_gemm_unavailable() or "")

GEOM = GatedAttentionBlockGeometry(d_model=4096, h_q=32, h_kv=2, d_head=256, rope_dim=64)


def _run(plan, a, w, out):
    ws = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    run_proj_gemm(plan, a, w, out, ws)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Shape algebra — no GPU
# ---------------------------------------------------------------------------


def test_the_two_projections_have_the_shapes_the_block_claims():
    """397B full-attention layer: 17408x4096 in, 4096x8192 out."""
    p1 = _qkv_gate_projection(GEOM, batch=1, seq_len=8192, dtype=torch.bfloat16)
    p6 = _out_projection(GEOM, batch=1, seq_len=8192, dtype=torch.bfloat16)
    assert (p1.m, p1.k, p1.n) == (8192, 4096, 17408)
    assert (p6.m, p6.k, p6.n) == (8192, 8192, 4096)
    assert p1.flops() == 2 * 8192 * 17408 * 4096
    assert p1.n == GEOM.n_qkvg and p6.k == GEOM.h_q * GEOM.d_head


@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test_declines_unsupported_dtypes(dtype):
    p = _qkv_gate_projection(GEOM, batch=1, seq_len=128, dtype=dtype)
    with pytest.raises(NotImplementedError, match="bf16/f16"):
        p.check_support()


def test_missing_frost_plan_raises_rather_than_falling_back():
    """The failure mode this guards is silent: an unpinned graph runs a cuDNN
    backend plan and every number measured off it is of that kernel. The
    message must name WHY -- arch or the opt-in flag -- because the two look
    identical from the call site."""
    from cudnn.gated_attention_block.kernels.proj_gemm import _why_no_frost_plan

    why = _why_no_frost_plan()
    assert "Refusing to fall back" in why
    assert ("sm100 GEMM template family" in why) or ("CUDNN_FRONTEND_ENABLE_FROST_ENGINES" in why) or ("declined this shape" in why)


# ---------------------------------------------------------------------------
# Numerics — needs an sm100-family device
# ---------------------------------------------------------------------------


@requires_frost_gemm
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m, k, n", [(256, 256, 256), (1024, 4096, 2176), (512, 2048, 512)])
def test_matches_torch_linear(dtype, m, k, n):
    """The weight goes in as a checkpoint holds it -- ``[N, K]`` row-major, read
    transposed by the graph's declared stride. If this ever needed a repack, the
    block would be paying a hidden copy per execute (Rule 2)."""
    torch.manual_seed(0)
    a = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.1).to(dtype)
    w = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05).to(dtype)
    out = torch.empty(m, n, device="cuda", dtype=dtype)

    plan = build_proj_gemm(m=m, k=k, n=n, dtype=dtype, label="test")
    _run(plan, a, w, out)

    ref = torch.nn.functional.linear(a.float(), w.float())
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
    assert cos > 0.999, f"cos {cos}"
    assert torch.isfinite(out.float()).all()


@requires_frost_gemm
def test_rank2_and_rank3_binds_agree():
    """The graph declares rank-3 operands; the block holds rank-2 [M, K]
    matrices. The adapter must bind a VIEW, so both spellings give bit-identical
    output and neither allocates."""
    m, k, n, dtype = 512, 1024, 512, torch.bfloat16
    torch.manual_seed(0)
    a = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.1).to(dtype)
    w = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05).to(dtype)
    plan = build_proj_gemm(m=m, k=k, n=n, dtype=dtype, label="rank")
    o2 = torch.empty(m, n, device="cuda", dtype=dtype)
    o3 = torch.empty(1, m, n, device="cuda", dtype=dtype)
    _run(plan, a, w, o2)
    _run(plan, a.unsqueeze(0), w.unsqueeze(0), o3)
    torch.testing.assert_close(o2, o3[0], rtol=0, atol=0)


@requires_frost_gemm
def test_output_is_not_silently_zero():
    """Guards the >256 KiB tcgen05 SMEM-descriptor hazard specifically.

    On Rubin the tile catalog sizes its pipeline from the OVERSIZED 327 KiB
    carveout, and the GEMM templates carry no ``desc_version`` handling. A
    version-0 descriptor's ``start_address`` is 14 bits, so an MMA-operand
    buffer at or past 262144 wraps to the bottom of SMEM and the accumulator
    comes out EXACTLY ZERO -- no crash, no error, and a cosine check against a
    zero tensor is NaN rather than a clean failure. Assert non-zero explicitly.
    """
    m, k, n = 1024, 4096, 4096
    a = torch.full((m, k), 0.05, device="cuda", dtype=torch.bfloat16)
    w = torch.full((n, k), 0.05, device="cuda", dtype=torch.bfloat16)
    out = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    plan = build_proj_gemm(m=m, k=k, n=n, dtype=torch.bfloat16, label="nonzero")
    _run(plan, a, w, out)
    nonzero = (out != 0).float().mean().item()
    assert nonzero > 0.99, f"only {100 * nonzero:.1f}% of the output is non-zero -- suspect the >256 KiB SMEM descriptor version"
    torch.testing.assert_close(out.float().mean().item(), 0.05 * 0.05 * k, rtol=2e-2, atol=0)


@requires_frost_gemm
def test_block_shapes_run_end_to_end():
    """The real 397B geometry at a small M, through the stage objects."""
    m_tokens = 256
    for factory, k_in in ((_qkv_gate_projection, GEOM.d_model), (_out_projection, GEOM.h_q * GEOM.d_head)):
        st = factory(GEOM, batch=1, seq_len=m_tokens, dtype=torch.bfloat16)
        st.check_support()
        st.compile()
        a = (torch.randn(st.m, st.k, device="cuda", dtype=torch.float32) * 0.05).to(torch.bfloat16)
        w = (torch.randn(st.n, st.k, device="cuda", dtype=torch.float32) * 0.02).to(torch.bfloat16)
        out = torch.empty(st.m, st.n, device="cuda", dtype=torch.bfloat16)
        ws = torch.empty(st.workspace_bytes(), dtype=torch.uint8, device="cuda")
        st.execute(a, w, out, ws)
        torch.cuda.synchronize()
        assert st.k == k_in
        ref = torch.nn.functional.linear(a.float(), w.float())
        cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
        assert cos > 0.999, f"{st.name}: cos {cos}"


# ---------------------------------------------------------------------------
# The FORK: norm+RoPE fused into the epilogue (Rubin bf16 only)
# ---------------------------------------------------------------------------

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference import apply_partial_rope, build_rope_tables, qk_norm_rope_reference  # noqa: E402

_SM107 = (10, 7)
requires_rubin = pytest.mark.skipif(
    not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != _SM107, reason="the fused stage-(1) fork is rendered for sm_107a"
)
# Small geometry: the fusion's constraints are on d_head (== the 256-wide tile)
# and rope_dim (whole subtile pairs), not on head counts, so h_q=4 keeps the
# N=3072 slab cheap while every Q | GATE | K | V arm is still exercised.
_FUSED_GEOM = GatedAttentionBlockGeometry(d_model=1024, h_q=4, h_kv=2, d_head=256, rope_dim=64)


def _fused_stage(geom=_FUSED_GEOM, m=1000, want_rstd=True, dtype=torch.bfloat16):
    return _FusedQkvProjection(geom, batch=1, seq_len=m, dtype=dtype, want_rstd=want_rstd)


@pytest.mark.parametrize(
    "bad, match", [(dict(d_head=128, rope_dim=64), "d_head == 256"), (dict(d_head=256, rope_dim=32), "rope_dim"), (dict(d_head=256, rope_dim=256), "rope_dim")]
)
def test_fused_stage_declines_geometry_the_epilogue_cannot_tile(bad, match):
    """The tile constraint is checked BEFORE the device, so the decline reads the
    same on every box and names what it is about -- not a JIT error later."""
    geom = GatedAttentionBlockGeometry(d_model=512, h_q=4, h_kv=2, **bad)
    with pytest.raises(NotImplementedError, match=match):
        _fused_stage(geom).check_support()


def test_fused_stage_declines_f16():
    with pytest.raises(NotImplementedError, match="bf16"):
        _fused_stage(dtype=torch.float16).check_support()


def test_fusion_params_offsets_follow_the_layout_contract():
    p = NormRopeFusionParams(d_head=256, rope_dim=64, h_q=32, h_kv=2)
    assert p.offsets == _FUSED_GEOM.__class__(d_model=4096, h_q=32, h_kv=2, d_head=256, rope_dim=64).qkvg_offsets
    assert p.n_qkvg == 17408
    validate_norm_rope_params(p)
    with pytest.raises(ValueError, match="norm_source"):
        validate_norm_rope_params(NormRopeFusionParams(norm_source="tma"))


def test_qk_norm_knob_is_appended_and_validated():
    """``qk_norm`` (then ``quant_mxfp8``) are APPENDED with defaults that spell every existing
    key identically -- a norm-on render's template-loader cache key is unchanged.  Under
    ``qk_norm=False`` there is no rstd to emit and no weight load to delete (``const_w`` has
    no meaning; ``const_cs`` == ``const``); ``off`` and the FP8 fork stay legal."""
    assert NormRopeFusionParams() == NormRopeFusionParams(qk_norm=True)
    assert NormRopeFusionParams() == NormRopeFusionParams(qk_norm=True, quant_mxfp8=False)
    names = [f.name for f in dataclasses.fields(NormRopeFusionParams)]
    assert names[-3:] == ["quant_fp8", "qk_norm", "quant_mxfp8"], names  # the FROZEN order (PR-B section 1.5)
    with pytest.raises(ValueError, match="want_rstd"):
        validate_norm_rope_params(NormRopeFusionParams(qk_norm=False, want_rstd=True))
    with pytest.raises(ValueError, match="const_w"):
        validate_norm_rope_params(NormRopeFusionParams(qk_norm=False, norm_source="const_w", quant_fp8=True))
    validate_norm_rope_params(NormRopeFusionParams(qk_norm=False, norm_source="const_cs", quant_fp8=True))  # == const under norm-off
    validate_norm_rope_params(NormRopeFusionParams(qk_norm=False, norm_source="off"))  # bf16 control: legal
    validate_norm_rope_params(NormRopeFusionParams(qk_norm=False, quant_fp8=True))
    validate_norm_rope_params(NormRopeFusionParams(qk_norm=False, norm_source="const"))


def test_quant_mxfp8_key_loads_the_twin_and_stays_exclusive():
    """The MXFP8 twin (kernels/proj_gemm_norm_rope_mxfp8.py) is selected by ``quant_mxfp8`` and by
    nothing else: its geometry rules hold, the two quant flags are exclusive, and an MXFP8 key
    loads the TWIN's template -- never the bf16 rendering.  Import-time only (no cute compile):
    the launch-level proofs live in test_proj_gemm_mxfp8.py."""
    from cudnn.frost.template_loader import load_template
    from cudnn.gated_attention_block.kernels import proj_gemm as _pg

    with pytest.raises(ValueError, match="at most one"):
        validate_norm_rope_params(NormRopeFusionParams(quant_fp8=True, quant_mxfp8=True))
    with pytest.raises(ValueError, match="want_rstd"):
        validate_norm_rope_params(NormRopeFusionParams(quant_mxfp8=True, want_rstd=True))
    with pytest.raises(ValueError, match="off"):
        validate_norm_rope_params(NormRopeFusionParams(quant_mxfp8=True, norm_source="off"))
    validate_norm_rope_params(NormRopeFusionParams(quant_mxfp8=True))
    validate_norm_rope_params(NormRopeFusionParams(quant_mxfp8=True, qk_norm=False))
    assert os.path.basename(_pg._FUSED_TEMPLATE_MXFP8) == "proj_gemm_norm_rope_mxfp8.py" and os.path.exists(_pg._FUSED_TEMPLATE_MXFP8)
    mod = load_template(_pg._FUSED_TEMPLATE_MXFP8, NormRopeFusionParams(quant_mxfp8=True), tag="proj_gemm_norm_rope_mxfp8_keycheck")
    assert mod.PARAMS.quant_mxfp8 is True and mod.PARAMS.quant_fp8 is False


def test_fused_runners_check_the_weights_against_the_artifact_before_launch():
    """Norm weights present iff the params say ``qk_norm`` -- both directions, on the host, before
    the (None) launch handle is touched: a mismatch would otherwise be a tvm-ffi ABI error."""
    m, k, dt = 64, 128, torch.bfloat16
    p_on = NormRopeFusionParams(d_head=256, rope_dim=64, h_q=4, h_kv=2)
    p_off = NormRopeFusionParams(d_head=256, rope_dim=64, h_q=4, h_kv=2, qk_norm=False)
    a, w, out = torch.zeros(m, k, dtype=dt), torch.zeros(p_on.n_qkvg, k, dtype=dt), torch.zeros(m, p_on.n_qkvg, dtype=dt)
    cs = torch.zeros(m, 64, dtype=dt)
    wd = torch.ones(256, dtype=dt)
    plan_on = FusedProjGemmPlan(params=p_on, module=None, launch=None)
    plan_off = FusedProjGemmPlan(params=p_off, module=None, launch=None)
    with pytest.raises(ValueError, match="qk_norm=True"):
        run_fused_proj_gemm(plan_on, a, w, out, None, None, cs, cs, stream=0)
    with pytest.raises(ValueError, match="qk_norm=False"):
        run_fused_proj_gemm(plan_off, a, w, out, wd, wd, cs, cs, stream=0)
    with pytest.raises(ValueError, match="qk_norm=False"):
        run_fused_proj_gemm(plan_off, a, w, out, None, wd, cs, cs, stream=0)  # one of the two: still a mismatch
    # the FP8 runner shares the check (its tables are bf16 regardless of the e4m3 operands)
    if _FP8 is not None:
        p8_off = NormRopeFusionParams(**{**p_off.__dict__, "quant_fp8": True})
        plan8 = FusedProjGemmPlan(params=p8_off, module=None, launch=None, fp8=True)
        a8, w8 = torch.zeros(m, k, dtype=_FP8), torch.zeros(p_on.n_qkvg, k, dtype=_FP8)
        outs = (
            torch.zeros(m, p_on.n_q, dtype=_FP8),
            torch.zeros(m, p_on.n_kv, dtype=_FP8),
            torch.zeros(m, p_on.n_kv, dtype=_FP8),
            torch.zeros(m, p_on.n_gate, dtype=dt),
        )
        with pytest.raises(ValueError, match="qk_norm=False"):
            run_fused_proj_gemm_fp8(plan8, a8, w8, *outs, wd, wd, cs, cs, torch.zeros(4), stream=0)


def _bf16_fork_params(geom=_FUSED_GEOM, **kw) -> NormRopeFusionParams:
    return NormRopeFusionParams(d_head=geom.d_head, rope_dim=geom.rope_dim, h_q=geom.h_q, h_kv=geom.h_kv, eps=geom.qk_norm_eps, **kw)


@requires_rubin
@pytest.mark.parametrize("mode", ["norm", "rope_only", "rope_only_stage"])
@pytest.mark.parametrize("m", [1000, 256])  # 1000: a tail tile (not a multiple of 256) + rows past M
def test_fused_stage_matches_the_fp32_oracle(m, mode):
    """Q/K normed + rotated on the fp32 ACCUMULATOR with one rounding; GATE/V untouched; rstd exact.

    ``rope_only`` (``qk_norm=False``): the Q/K tiles are rotated only -- no pass A, no rsqrt,
    no weight loads, no rstd; the weights are None at the ABI.  Its oracle is the fp32 GEMM +
    partial RoPE with one rounding; its PROVABLE bitwise control is the same params at
    ``norm_source='off'`` (the plain rendered GEMM): the passthrough dims ``[rope_dim, d)`` of
    every Q/K head and every GATE / V cell are the raw accumulator rounded once in both, so
    they must be ``torch.equal`` (NOT vs ``proj32.to(bf16)``: TF32 + another accumulation order).

    ``rope_only`` drives the fork DIRECTLY (``build_fused_proj_gemm`` / ``run_fused_proj_gemm``:
    the bitwise claim is about the kernel); ``rope_only_stage`` drives the SAME artifact through
    the block's stage object on a ``qk_norm=False`` geometry -- ``_FusedQkvProjection.params()``
    spelling ``qk_norm=False``, ``check_support`` admitting it, ``execute`` taking ``None``
    weights -- i.e. the plumbing the block itself uses under ``fuse_norm_rope=True``.
    """
    qk_norm = mode == "norm"
    g = _FUSED_GEOM if qk_norm else dataclasses.replace(_FUSED_GEOM, qk_norm=False)
    torch.manual_seed(0)
    dt = torch.bfloat16
    stream = torch.cuda.current_stream().cuda_stream
    if mode != "rope_only":
        st = _FusedQkvProjection(g, batch=1, seq_len=m, dtype=dt, want_rstd=qk_norm)
        st.check_support()
        st.compile()
        launch = lambda out, rq, rk, wq, wk: st.execute(
            h, w, out, wq, wk, cos, sin, rq if qk_norm else None, rk if qk_norm else None, stream=stream
        )  # noqa: E731
    else:
        plan = build_fused_proj_gemm(_bf16_fork_params(g, qk_norm=False, want_rstd=False))
        launch = lambda out, rq, rk, wq, wk: run_fused_proj_gemm(plan, h, w, out, wq, wk, cos, sin, None, None, stream=stream)  # noqa: E731
    h = (torch.randn(m, g.d_model, device="cuda") * 0.5).to(dt)
    w = (torch.randn(g.n_qkvg, g.d_model, device="cuda") * 0.02).to(dt)
    wq = (1.0 + 0.1 * torch.randn(g.d_head, device="cuda")).to(dt) if qk_norm else None
    wk = (1.0 + 0.1 * torch.randn(g.d_head, device="cuda")).to(dt) if qk_norm else None
    cos, sin = build_rope_tables(m, g.rope_dim, batch=1, device="cuda", dtype=dt)
    out = torch.full((m, g.n_qkvg), 1.5e30, device="cuda", dtype=dt)  # sentinel: a survivor is a never-written cell
    rq = torch.full((m, g.h_q), -1.0, device="cuda")
    rk = torch.full((m, g.h_kv), -1.0, device="cuda")
    launch(out, rq, rk, wq, wk)
    torch.cuda.synchronize()
    assert not (out.float() == 1.5e30).any()

    proj32 = torch.nn.functional.linear(h.float(), w.float())
    ref = proj32.to(dt).clone()
    o_q, o_g, o_k, o_v = g.qkvg_offsets
    refr = {}
    for off, hh, wn in ((o_q, g.h_q, wq), (o_k, g.h_kv, wk)):
        x4 = proj32[:, off : off + hh * g.d_head].view(1, m, hh, g.d_head)
        if qk_norm:
            y, r = qk_norm_rope_reference(x4, wn, cos, sin, g.rope_dim, g.qk_norm_eps)
            refr[off] = r.view(m, hh)
        else:
            y = apply_partial_rope(x4.float(), cos, sin, g.rope_dim).to(dt)
        ref[:, off : off + hh * g.d_head] = y.view(m, hh * g.d_head)
    for off, hh in ((o_q, g.h_q), (o_g, g.h_q), (o_k, g.h_kv), (o_v, g.h_kv)):
        got, exp = out[:, off : off + hh * g.d_head].float(), ref[:, off : off + hh * g.d_head].float()
        cos_sim = torch.nn.functional.cosine_similarity(got.flatten(), exp.flatten(), dim=0).item()
        assert cos_sim > 0.9999, f"block at col {off}: cos {cos_sim}"
        assert torch.isfinite(got).all()
    if qk_norm:
        torch.testing.assert_close(rq, refr[o_q], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(rk, refr[o_k], rtol=1e-4, atol=1e-5)
    # two-launch trick: a first-launch race would show here
    out2 = torch.empty_like(out)
    launch(out2, rq, rk, wq, wk)
    torch.cuda.synchronize()
    torch.testing.assert_close(out2, out, rtol=0, atol=0)
    if not qk_norm:
        # the bitwise control: the plain rendered GEMM (norm_source='off') at the same params
        ctrl = build_fused_proj_gemm(_bf16_fork_params(g, qk_norm=False, want_rstd=False, norm_source="off"))
        out_off = torch.empty_like(out)
        run_fused_proj_gemm(ctrl, h, w, out_off, None, None, cos, sin, None, None, stream=stream)
        torch.cuda.synchronize()
        assert torch.equal(out[:, o_g:o_k], out_off[:, o_g:o_k]), "GATE block differs from the plain GEMM under rope_only"
        assert torch.equal(out[:, o_v:], out_off[:, o_v:]), "V block differs from the plain GEMM under rope_only"
        for off, hh in ((o_q, g.h_q), (o_k, g.h_kv)):
            got4 = out[:, off : off + hh * g.d_head].view(m, hh, g.d_head)
            ctl4 = out_off[:, off : off + hh * g.d_head].view(m, hh, g.d_head)
            assert torch.equal(got4[..., g.rope_dim :], ctl4[..., g.rope_dim :]), f"Q/K passthrough dims at col {off} differ from the plain GEMM"
            assert not torch.equal(got4[..., : g.rope_dim], ctl4[..., : g.rope_dim]), "the rotated dims must NOT equal the un-rotated GEMM"


# ---------------------------------------------------------------------------
# FP8 E4M3 inputs + the per-tensor descale (alpha) epilogue
# ---------------------------------------------------------------------------

_FP8 = getattr(torch, "float8_e4m3fn", None)
_FP8_MAX = 448.0
requires_fp8 = pytest.mark.skipif(_FP8 is None, reason="this torch has no float8_e4m3fn")


def _quant_e4m3(x: torch.Tensor):
    """Per-tensor amax/448 quantization; returns ``(x_fp8, descale)`` like the SDPA FP8 suite's ``_quant``."""
    dq = (x.abs().amax().clamp_min(1e-8) / _FP8_MAX).item()
    return (x / dq).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8), dq


def _fp8_case(m, k, n, alpha_val, *, use_alpha=True, seed=0):
    torch.manual_seed(seed)
    a32 = torch.randn(m, k, device="cuda") * 0.5
    w32 = torch.randn(n, k, device="cuda") * 0.02
    a8, dqa = _quant_e4m3(a32)
    w8, dqw = _quant_e4m3(w32)
    alpha = torch.tensor([alpha_val], dtype=torch.float32, device="cuda")
    plan = build_proj_gemm(m=m, k=k, n=n, dtype=_FP8, label="fp8", alpha=use_alpha)
    out = torch.full((m, n), 1.5e30, device="cuda", dtype=torch.bfloat16)  # sentinel
    ws = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    run_proj_gemm(plan, a8, w8, out, ws, alpha=alpha if use_alpha else None)
    torch.cuda.synchronize()
    ref32 = torch.nn.functional.linear(a8.float(), w8.float()) * (alpha_val if use_alpha else 1.0)
    return plan, out, ref32


def _assert_bf16_of_fp32(out, ref32, label):
    """The kernel accumulates the SAME fp8 products in fp32 and rounds once to
    bf16, so it must match the fp32 reference to bf16 rounding: one ulp of the
    output's magnitude, plus fp32 reassociation noise."""
    assert not (out.float() == 1.5e30).any(), f"{label}: sentinel survivors"
    assert torch.isfinite(out.float()).all()
    err = (out.float() - ref32).abs()
    scale = ref32.abs().max().item()
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref32.flatten(), dim=0).item()
    assert cos > 0.9999, f"{label}: cos {cos}"
    # bf16 has 8 significand bits: 1 ulp at the max magnitude is scale * 2^-8; allow 2 ulps + reassociation.
    assert err.max().item() <= 2.0 * scale * 2.0**-8 + 1e-3 * scale, f"{label}: max|err| {err.max().item():.3e} vs scale {scale:.3e}"


@requires_frost_gemm
@requires_fp8
@pytest.mark.parametrize("m, k, n", [(256, 256, 256), (1000, 4096, 17408), (512, 8192, 4096)])
def test_fp8_matmul_with_alpha_matches_fp32_reference(m, k, n):
    """E4M3 x E4M3 -> fp32 accumulate -> * alpha -> bf16, vs the exact fp32
    reference of the SAME quantized operands. M=1000 is a tail tile; the other
    two are the block's qkv_gate_proj (K=4096, N=17408) and out_proj (K=8192,
    N=4096) geometries."""
    plan, out, ref32 = _fp8_case(m, k, n, 0.37)
    assert plan.has_alpha and plan.dtype == _FP8 and plan.out_dtype == torch.bfloat16
    _assert_bf16_of_fp32(out, ref32, f"fp8 {m}x{k}x{n}")


@requires_frost_gemm
@requires_fp8
def test_fp8_alpha_is_actually_applied():
    """alpha=0.5 must halve the output of alpha=1.0 -- guards a silently-dropped aux binding."""
    _, out_half, _ = _fp8_case(256, 512, 512, 0.5)
    _, out_one, _ = _fp8_case(256, 512, 512, 1.0)
    torch.testing.assert_close(out_half.float(), 0.5 * out_one.float(), rtol=1e-2, atol=1e-3)


@requires_frost_gemm
@requires_fp8
def test_fp8_without_alpha_still_runs():
    plan, out, ref32 = _fp8_case(256, 512, 768, 1.0, use_alpha=False)
    assert not plan.has_alpha
    _assert_bf16_of_fp32(out, ref32, "fp8 no-alpha")
    with pytest.raises(ValueError, match="no alpha"):
        run_proj_gemm(
            plan,
            torch.empty(256, 512, dtype=_FP8, device="cuda"),
            torch.empty(768, 512, dtype=_FP8, device="cuda"),
            out,
            torch.empty(1, dtype=torch.uint8, device="cuda"),
            alpha=torch.ones(1, device="cuda"),
        )


@requires_frost_gemm
@requires_fp8
def test_fp8_alpha_plan_refuses_a_missing_alpha():
    plan = build_proj_gemm(m=256, k=512, n=512, dtype=_FP8, label="fp8", alpha=True)
    a = torch.zeros(256, 512, dtype=_FP8, device="cuda")
    w = torch.zeros(512, 512, dtype=_FP8, device="cuda")
    out = torch.empty(256, 512, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="alpha="):
        run_proj_gemm(plan, a, w, out, torch.empty(1, dtype=torch.uint8, device="cuda"))


def test_fp8_needs_k_multiple_of_16():
    if _FP8 is None:
        pytest.skip("no float8_e4m3fn")
    with pytest.raises(ValueError, match="K % 16"):
        build_proj_gemm(m=256, k=200, n=256, dtype=_FP8, label="fp8", pin_frost=False)


def test_unsupported_dtype_message_names_fp8():
    with pytest.raises(ValueError, match="fp8-e4m3"):
        build_proj_gemm(m=16, k=16, n=16, dtype=torch.float32, label="x", pin_frost=False)


# ---------------------------------------------------------------------------
# The FP8 FORK: norm+RoPE AND the e4m3 quantization of Q/K/V fused into the FP8
# GEMM epilogue (Rubin).  Oracle = the fp32 chain of the FP8 fusion design
# contract, quantized ONCE with torch's saturating e4m3 cast.
# ---------------------------------------------------------------------------

_E4M3_NAN_BYTE = 0x7F  # e4m3fn NaN; `cvt.rn.satfinite` never produces it, so a surviving byte is a never-written cell


def _e4m3_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of e4m3 at magnitude |x|: 2^(e-3) for normals (|x| >= 2^-6), 2^-9 for subnormals.

    Built from the fp32 exponent bits (exact) -- NOT ``torch.exp2`` / ``torch.log2``,
    which torch JIT-compiles through NVRTC and which fails on cc 10.7
    ("invalid value for --gpu-architecture").
    """
    a = x.float().abs()
    _, e = torch.frexp(a.clamp_min(2.0**-6))  # a = m * 2^e with m in [0.5, 1)  ->  floor(log2 a) = e - 1
    k = e.to(torch.int32) - 1 - 3  # exponent of the ulp for a normal e4m3 (3 mantissa bits)
    ulp_normal = ((k + 127) << 23).to(torch.int32).view(torch.float32)  # exactly 2^k
    return torch.where(a >= 2.0**-6, ulp_normal, torch.full_like(a, 2.0**-9))


def compare_e4m3(got8: torch.Tensor, ref8: torch.Tensor) -> dict:
    """Bit-equal fraction, max |diff| and max |diff| in e4m3 ulps (of the larger magnitude) between two e4m3 tensors."""
    g, r = got8.float(), ref8.float()
    d = (g - r).abs()
    ulps = d / _e4m3_ulp(torch.maximum(g.abs(), r.abs()))
    return dict(
        bit_equal=torch.equal(got8.view(torch.uint8), ref8.view(torch.uint8)),
        bit_equal_frac=(got8.view(torch.uint8) == ref8.view(torch.uint8)).float().mean().item(),
        max_abs_diff=d.max().item(),
        max_ulps=ulps.max().item(),
        n_off=(d > 0).sum().item(),
    )


def fp8_fused_oracle(a8, w8, wq, wk, cos, sin, p: NormRopeFusionParams, alpha: float, scale_q: float, scale_k: float, scale_v: float) -> dict:
    """The design contract's fp32 chain, one rounding per output.

    ``y = alpha * acc`` (exact fp32 products of the SAME e4m3 operands); Q/K:
    ``rsqrt(mean(y^2) + eps)`` (== ``rsqrt(sum(acc^2) * alpha^2 / d + eps)``, the
    form the kernel folds), ``* w``, RoPE, ``e4m3_satfinite(x * scale)``; V:
    ``e4m3_satfinite(y * scale_v)``; GATE: ``y`` in fp32 (the test rounds to bf16).
    ``p.qk_norm=False``: Q/K skip the norm (``rsqrt`` / ``w``) and are rotated only.
    Returns the e4m3 references in the fork's OUTPUT layout -- the COMPACT
    per-tensor ``q8 [m, h_q*d]`` / ``k8 [m, h_kv*d]`` / ``v8 [m, h_kv*d]`` (round 2:
    there is no ``[m, n_qkv]`` slab) -- plus the fp32 chain per class.
    """
    m = a8.shape[-2]
    d, r = p.d_head, p.rope_dim
    o_q, o_g, o_k, o_v = p.offsets
    y = torch.nn.functional.linear(a8.float(), w8.float()) * alpha
    cos3, sin3 = cos.reshape(1, m, r), sin.reshape(1, m, r)
    q4, k4 = y[:, o_q : o_q + p.h_q * d].view(1, m, p.h_q, d), y[:, o_k : o_k + p.h_kv * d].view(1, m, p.h_kv, d)
    if p.qk_norm:
        qn, _ = qk_norm_rope_reference(q4, wq, cos3, sin3, r, p.eps)
        kn, _ = qk_norm_rope_reference(k4, wk, cos3, sin3, r, p.eps)
    else:  # RoPE only: the descaled accumulator rotated, no norm, no weights
        qn, kn = apply_partial_rope(q4, cos3, sin3, r), apply_partial_rope(k4, cos3, sin3, r)
    qn, kn = qn.view(m, p.h_q * d), kn.view(m, p.h_kv * d)
    v32 = y[:, o_v : o_v + p.h_kv * d]
    q8 = (qn * scale_q).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
    k8 = (kn * scale_k).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
    v8 = (v32 * scale_v).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
    assert q8.shape == (m, p.n_q) and k8.shape == v8.shape == (m, p.n_kv)
    return dict(q8=q8, k8=k8, v8=v8, gate32=y[:, o_g : o_g + p.h_q * d], qn32=qn, kn32=kn, v32=v32)


def fp8_fused_inputs(m: int, k: int, p: NormRopeFusionParams, seed: int = 0) -> dict:
    """e4m3 h / W (amax/448 per tensor), bf16 norm weights + RoPE tables, and the scales
    calibrated on the oracle's own fp32 activations (the block's offline-calibration stand-in)."""
    torch.manual_seed(seed)
    dt = torch.bfloat16
    a32 = torch.randn(m, k, device="cuda") * 0.5
    w32 = torch.randn(p.n_qkvg, k, device="cuda") * 0.02
    a8, dqa = _quant_e4m3(a32)
    w8, dqw = _quant_e4m3(w32)
    alpha = dqa * dqw
    # qk_norm=False: no norm weights exist (the runner requires both None)
    wq = (1.0 + 0.1 * torch.randn(p.d_head, device="cuda")).to(dt) if p.qk_norm else None
    wk = (1.0 + 0.1 * torch.randn(p.d_head, device="cuda")).to(dt) if p.qk_norm else None
    cos, sin = build_rope_tables(m, p.rope_dim, batch=1, device="cuda", dtype=dt)
    cos2, sin2 = cos.reshape(m, p.rope_dim).contiguous(), sin.reshape(m, p.rope_dim).contiguous()
    pre = fp8_fused_oracle(a8, w8, wq, wk, cos2, sin2, p, alpha, 1.0, 1.0, 1.0)
    amax_scale = lambda t: float(_FP8_MAX / t.float().abs().amax().clamp_min(1e-8))  # noqa: E731
    # Three DIFFERENT scales (and alpha != 1) so a swapped class / dropped fold is caught, not masked.
    scale_q, scale_k, scale_v = amax_scale(pre["qn32"]), amax_scale(pre["kn32"]) * 0.5, amax_scale(pre["v32"]) * 2.0
    ref = fp8_fused_oracle(a8, w8, wq, wk, cos2, sin2, p, alpha, scale_q, scale_k, scale_v)
    qscal = torch.tensor([alpha, scale_q, scale_k, scale_v], dtype=torch.float32, device="cuda")
    return dict(a8=a8, w8=w8, wq=wq, wk=wk, cos=cos2, sin=sin2, qscal=qscal, alpha=alpha, scales=(scale_q, scale_k, scale_v), ref=ref)


def fp8_fused_sentinels(m: int, p: NormRopeFusionParams):
    """Sentinel-filled outputs in the fork's compact layout: ``(q8, k8, v8, gate16)``.

    e4m3 outputs get the NaN byte (``cvt.rn.satfinite`` never produces it), the
    bf16 gate ``1.5e30``; a survivor after the launch is a never-written cell.
    """

    def _e4m3(width):
        return torch.full((m, width), _E4M3_NAN_BYTE, dtype=torch.uint8, device="cuda").view(_FP8)

    gate16 = torch.full((m, p.n_gate), 1.5e30, dtype=torch.bfloat16, device="cuda")
    return _e4m3(p.n_q), _e4m3(p.n_kv), _e4m3(p.n_kv), gate16


def fp8_fused_sentinel_survivors(q8, k8, v8, gate16) -> dict:
    """Never-written cells per output (all four must be 0 after a launch)."""
    return dict(
        q8=int((q8.view(torch.uint8) == _E4M3_NAN_BYTE).sum().item()),
        k8=int((k8.view(torch.uint8) == _E4M3_NAN_BYTE).sum().item()),
        v8=int((v8.view(torch.uint8) == _E4M3_NAN_BYTE).sum().item()),
        gate16=int((gate16.float() == 1.5e30).sum().item()),
    )


def _fused_fp8_params(geom=_FUSED_GEOM, **kw) -> NormRopeFusionParams:
    return NormRopeFusionParams(d_head=geom.d_head, rope_dim=geom.rope_dim, h_q=geom.h_q, h_kv=geom.h_kv, eps=geom.qk_norm_eps, quant_fp8=True, **kw)


def test_fp8_fusion_params_are_inference_only_and_keep_the_bf16_keys():
    """``quant_fp8`` is the LAST field with a False default, so every bf16 params key
    (the template-loader cache key) is unchanged; under FP8 there is no rstd and no
    ``off`` arm, and the slab widths follow the layout contract."""
    assert NormRopeFusionParams() == NormRopeFusionParams(quant_fp8=False)
    p = NormRopeFusionParams(d_head=256, rope_dim=64, h_q=32, h_kv=2, quant_fp8=True)
    validate_norm_rope_params(p)
    assert (p.n_gate, p.n_qkv, p.n_qkvg) == (8192, 9216, 17408)
    assert (p.n_q, p.n_kv) == (8192, 512)  # the compact q8 / k8 / v8 widths (round 2: no slab)
    with pytest.raises(ValueError, match="want_rstd"):
        validate_norm_rope_params(NormRopeFusionParams(quant_fp8=True, want_rstd=True))
    with pytest.raises(ValueError, match="off"):
        validate_norm_rope_params(NormRopeFusionParams(quant_fp8=True, norm_source="off"))
    validate_norm_rope_params(NormRopeFusionParams(want_rstd=True, norm_source="off"))  # bf16: both still legal


def test_fused_runners_refuse_the_other_fork_before_touching_a_tensor():
    """The bf16 and FP8 plans have different ABIs; each runner refuses the other's plan
    on the ``fp8`` flag alone (Rule 1: no silent misbind)."""
    p8 = _fused_fp8_params()
    fake8 = FusedProjGemmPlan(params=p8, module=None, launch=None, fp8=True)
    fake16 = FusedProjGemmPlan(params=NormRopeFusionParams(**{**p8.__dict__, "quant_fp8": False}), module=None, launch=None)
    with pytest.raises(ValueError, match="run_fused_proj_gemm_fp8"):
        run_fused_proj_gemm(fake8, None, None, None, None, None, None, None, stream=0)
    with pytest.raises(ValueError, match="bf16 fork"):
        run_fused_proj_gemm_fp8(fake16, None, None, None, None, None, None, None, None, None, None, None, stream=0)


@requires_rubin
@requires_fp8
@pytest.mark.parametrize("qk_norm", [True, False], ids=["norm", "rope_only"])
@pytest.mark.parametrize("m", [1000, 256])  # 1000: a tail tile (not a multiple of 256) + rows past M, clipped by TMA on all FOUR outputs
def test_fused_fp8_stage_matches_the_e4m3_oracle(m, qk_norm):
    """Q/K normed + rotated on the fp32 accumulator and quantized ONCE; V descaled +
    quantized; GATE bf16(alpha * acc) -- each against the contract's fp32 chain,
    each landing in its OWN compact tensor (q8 / k8 / v8 / gate16, round 2).

    Q/K/V are compared as e4m3 to the oracle's e4m3: the vast majority of cells
    bit-equal, the rest within ONE e4m3 ulp (fp32 reassociation flips midpoints).
    The NaN-byte sentinel proves every cell of all four outputs was written --
    which is also the check that each descriptor's own-width OOB clip is right.

    ``rope_only`` (``qk_norm=False``): the Q/K per-row multiplier is ``alpha * scale``
    (the V arm's form), no pass A / rsqrt / weight loads; the weights are None.
    """
    g = _FUSED_GEOM
    p = _fused_fp8_params(g, qk_norm=qk_norm)
    plan = build_fused_proj_gemm(p)
    assert plan.fp8
    inp = fp8_fused_inputs(m, g.d_model, p)
    q8, k8, v8, gate16 = fp8_fused_sentinels(m, p)
    stream = torch.cuda.current_stream().cuda_stream
    run_fused_proj_gemm_fp8(plan, inp["a8"], inp["w8"], q8, k8, v8, gate16, inp["wq"], inp["wk"], inp["cos"], inp["sin"], inp["qscal"], stream=stream)
    torch.cuda.synchronize()
    surv = fp8_fused_sentinel_survivors(q8, k8, v8, gate16)
    assert all(v == 0 for v in surv.values()), f"sentinel survivors (never-written cells): {surv}"

    ref = inp["ref"]
    for name, got, exp, x32, sc in (
        ("Q", q8, ref["q8"], ref["qn32"], inp["scales"][0]),
        ("K", k8, ref["k8"], ref["kn32"], inp["scales"][1]),
        ("V", v8, ref["v8"], ref["v32"], inp["scales"][2]),
    ):
        c = compare_e4m3(got, exp)
        assert c["bit_equal_frac"] > 0.99, f"{name}: only {100 * c['bit_equal_frac']:.2f}% of cells bit-equal to the e4m3 oracle ({c})"
        assert c["max_ulps"] <= 1.0, f"{name}: max |diff| {c['max_abs_diff']:.3e} = {c['max_ulps']:.2f} e4m3 ulps ({c})"
        # Against the UNquantized fp32 chain the fused output can only be as close as e4m3
        # itself allows (3 mantissa bits + the saturation these scales provoke: ~0.998 for V),
        # so the bar is the ORACLE's own e4m3, not a fixed constant.
        cos_sim = torch.nn.functional.cosine_similarity(got.float().flatten() / sc, x32.flatten(), dim=0).item()
        cos_oracle = torch.nn.functional.cosine_similarity(exp.float().flatten() / sc, x32.flatten(), dim=0).item()
        assert cos_sim >= cos_oracle - 1e-4, f"{name}: cos vs the fp32 chain {cos_sim} (the oracle's own e4m3: {cos_oracle})"
    _assert_bf16_of_fp32(gate16, ref["gate32"], "GATE")

    # two-launch trick on all FOUR outputs: a first-launch race shows as a non-zero delta
    q8_2, k8_2, v8_2, gate16_2 = fp8_fused_sentinels(m, p)
    run_fused_proj_gemm_fp8(plan, inp["a8"], inp["w8"], q8_2, k8_2, v8_2, gate16_2, inp["wq"], inp["wk"], inp["cos"], inp["sin"], inp["qscal"], stream=stream)
    torch.cuda.synchronize()
    for a_, b_ in ((q8_2, q8), (k8_2, k8), (v8_2, v8)):
        assert torch.equal(a_.view(torch.uint8), b_.view(torch.uint8))
    assert torch.equal(gate16_2, gate16)


@requires_rubin
@requires_fp8
def test_fused_fp8_scales_and_alpha_are_actually_applied():
    """Halving scale_q halves q8 (dequantized Q agrees) and leaves k8 / v8 untouched;
    halving alpha halves GATE and leaves the normed q8 unchanged (alpha cancels in the
    norm) but changes v8 -- guards a dropped / misplaced qscal read (it is read once per
    tile BEFORE the accumulator wait) or a class -> descriptor / scale swap."""
    g = _FUSED_GEOM
    p = _fused_fp8_params(g)
    plan = build_fused_proj_gemm(p)
    m = 256
    inp = fp8_fused_inputs(m, g.d_model, p)
    stream = torch.cuda.current_stream().cuda_stream

    def run(qscal):
        q8, k8, v8, gate16 = fp8_fused_sentinels(m, p)
        run_fused_proj_gemm_fp8(plan, inp["a8"], inp["w8"], q8, k8, v8, gate16, inp["wq"], inp["wk"], inp["cos"], inp["sin"], qscal, stream=stream)
        torch.cuda.synchronize()
        return q8, k8, v8, gate16

    _u8 = lambda t: t.view(torch.uint8)  # noqa: E731
    alpha, (sq, sk, sv) = inp["alpha"], inp["scales"]
    base_q, base_k, base_v, base_g = run(torch.tensor([alpha, sq, sk, sv], device="cuda"))
    dbl_q, dbl_k, dbl_v, _ = run(torch.tensor([alpha, 0.5 * sq, sk, sv], device="cuda"))
    # half the scale -> half the e4m3 magnitude (exact in e4m3 unless the value was saturated / subnormal)
    torch.testing.assert_close(dbl_q.float(), 0.5 * base_q.float(), rtol=0.13, atol=0.02)
    assert torch.equal(_u8(dbl_k), _u8(base_k)) and torch.equal(_u8(dbl_v), _u8(base_v))  # K / V untouched
    half_q, half_k, half_v, half_g = run(torch.tensor([0.5 * alpha, sq, sk, sv], device="cuda"))
    torch.testing.assert_close(half_g.float(), 0.5 * base_g.float(), rtol=1e-2, atol=1e-3)
    # alpha cancels in Q/K's norm (up to eps and rounding) but NOT in V
    assert (_u8(half_q) == _u8(base_q)).float().mean().item() > 0.99
    assert (_u8(half_k) == _u8(base_k)).float().mean().item() > 0.99
    assert not torch.equal(_u8(half_v), _u8(base_v))


@requires_rubin
@requires_fp8
def test_fused_fp8_runner_rejects_wrong_bindings():
    g = _FUSED_GEOM
    p = _fused_fp8_params(g)
    plan = build_fused_proj_gemm(p)
    m = 256
    inp = fp8_fused_inputs(m, g.d_model, p)
    q8, k8, v8, gate16 = fp8_fused_sentinels(m, p)
    ok = dict(a=inp["a8"], w=inp["w8"], q=q8, k=k8, v=v8, g=gate16, wq=inp["wq"], wk=inp["wk"], cos=inp["cos"], sin=inp["sin"], qs=inp["qscal"])

    def launch(**over):
        kw = {**ok, **over}
        run_fused_proj_gemm_fp8(
            plan,
            kw["a"],
            kw["w"],
            kw["q"],
            kw["k"],
            kw["v"],
            kw["g"],
            kw["wq"],
            kw["wk"],
            kw["cos"],
            kw["sin"],
            kw["qs"],
            stream=torch.cuda.current_stream().cuda_stream,
        )

    with pytest.raises(ValueError, match="out_q8"):
        launch(q=torch.empty(m, p.n_qkv, dtype=_FP8, device="cuda"))  # the round-1 slab width, not h_q*d
    with pytest.raises(ValueError, match="out_k8"):
        launch(k=torch.empty(m, p.n_q, dtype=_FP8, device="cuda"))  # q8's width handed as k8
    with pytest.raises(ValueError, match="out_v8"):
        launch(v=torch.empty(m, p.n_kv, dtype=torch.bfloat16, device="cuda"))  # right shape, wrong dtype
    with pytest.raises(ValueError, match="out_v8"):
        launch(v=torch.empty(m, 2 * p.n_kv, dtype=_FP8, device="cuda")[:, p.n_kv :])  # a non-contiguous column slice
    with pytest.raises(ValueError, match="out_gate16"):
        launch(g=torch.empty(m, p.n_gate, dtype=torch.float16, device="cuda"))
    with pytest.raises(ValueError, match="qscal"):
        launch(qs=inp["qscal"].to(torch.bfloat16))
    with pytest.raises(ValueError, match="qscal"):
        launch(qs=inp["qscal"][:3])
    with pytest.raises(ValueError, match="bf16 table"):
        launch(cos=inp["cos"].to(_FP8))  # the table dtype is the ACTIVATION dtype, not a.dtype
    with pytest.raises(ValueError, match="e4m3 operands"):
        launch(a=inp["a8"].to(torch.bfloat16))
