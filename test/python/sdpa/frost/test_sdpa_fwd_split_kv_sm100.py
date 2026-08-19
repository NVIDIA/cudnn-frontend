# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""KV-split tests for the SM100 d128 f16/bf16 prefill kernel.

The split must change NOTHING numerically: the recombined output has to match
both a torch fp32 reference and the unsplit kernel to fp16 rounding.

Coverage: split counts that divide the KV tile count evenly and ones that do not
(including more splits than tiles, which leaves some splits empty — the case
that exercises the empty-mainloop / LSE = -inf identity path), dense and causal
masks, MHA and GQA.
"""

import math
import os

import pytest
import torch

from frost_test_utils import requires_blackwell, requires_dsl

pytestmark = [requires_blackwell, requires_dsl]

D = 128
TILE_N = 128
# One cga2 cluster covers TILES_Q * TILE_M * CTA_MMA Q rows.
ROWS_PER_CLUSTER = 2 * 128 * 2


def _ref_sdpa(q, k, v, scale, is_causal, kh):
    """fp32 reference over BSHD inputs; returns BSHD."""
    if kh != q.shape[2]:
        rep = q.shape[2] // kh
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
    qb, kb, vb = (t.float().permute(0, 2, 1, 3) for t in (q, k, v))
    s = torch.matmul(qb, kb.transpose(-1, -2)) * scale
    if is_causal:
        s_q, s_kv = qb.shape[2], kb.shape[2]
        i = torch.arange(s_q, device=q.device).view(s_q, 1)
        j = torch.arange(s_kv, device=q.device).view(1, s_kv)
        s = s.masked_fill(j > i, float("-inf"))  # top-left causal
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, vb).permute(0, 2, 1, 3)


def _kernel_module(splits, dtype_qkv, causal, cta_mma=2):
    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    path = os.path.join(os.path.dirname(os.path.abspath(api_dsl.__file__)), "kernels", "prefill_d128_f16_sm100.py")
    kw = {"dtype_qkv": dtype_qkv, "split_kv": splits, "cta_mma": cta_mma}
    if causal:
        kw["window_right"] = 0
    return load_template(path, TemplateParams(**kw), tag=f"splitkv{splits}_d{dtype_qkv}_{'caus' if causal else 'dense'}_cga{cta_mma}")


def _run(splits, B, H, KH, SQ, SKV, dtype, causal, cta_mma=2):
    """Launch the split kernel + DSL combine; returns the recombined O (BSHD fp32)."""
    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.sdpa.fwd.kernels import split_combine_sm100 as comb

    dev = "cuda"
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(0)
    q = torch.randn(B, SQ, H, D, device=dev, dtype=dtype)
    k = torch.randn(B, SKV, KH, D, device=dev, dtype=dtype)
    v = torch.randn(B, SKV, KH, D, device=dev, dtype=dtype)

    mod = _kernel_module(splits, 3 if dtype == torch.float16 else 2, causal, cta_mma=cta_mma)
    fn = mod.compile(b=B, qh=H, kh=KH, sq=SQ, skv=SKV, d_qk=D, d_v=D, has_lse=True)

    o_p = torch.zeros(splits * B, SQ, H, D, device=dev, dtype=dtype)
    lse_p = torch.zeros(splits * B, H, SQ, device=dev, dtype=torch.float32)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        q,
        k,
        v,
        o_p,
        lse_p,
        torch.zeros(H, dtype=torch.float32, device=dev),  # sinks (ABI slot)
        torch.zeros(B, dtype=torch.int32, device=dev),  # seq_kv_lens (unused)
        torch.zeros(1, dtype=torch.int64, device=dev),  # o_desc (THD only)
        (B, H, KH, SQ, SKV, 0),
        cutlass.Float32(scale * math.log2(math.e)),
        cutlass.Int32(0),
        None,
        stream=stream,
    )
    if splits == 1:
        torch.cuda.synchronize()
        return o_p.float(), (q, k, v, scale)

    o_out = torch.zeros(B, SQ, H, D, device=dev, dtype=dtype)
    lse_out = torch.zeros(B, H, SQ, device=dev, dtype=torch.float32)
    cfn = comb.compile(b=B, h=H, sq=SQ, d_v=D, splits=splits, dtype_o="f16" if dtype == torch.float16 else "bf16", has_lse=True)
    cfn(o_p, lse_p, o_out, lse_out, None, (B, H, SQ, D), cutlass.Int32(splits), stream=stream)
    torch.cuda.synchronize()
    assert not torch.isnan(o_out).any(), "NaN in combined O"
    return o_out.float(), (q, k, v, scale)


@pytest.mark.L0
@pytest.mark.parametrize("splits", [2, 4, 8], ids=lambda s: f"split{s}")
def test_split_kv_matches_reference_dense(splits):
    """The target shape class: tiny S_q against a long KV run."""
    B, H, SQ, SKV = 1, 4, 128, 2048
    got, (q, k, v, scale) = _run(splits, B, H, H, SQ, SKV, torch.float16, causal=False)
    ref = _ref_sdpa(q, k, v, scale, is_causal=False, kh=H)
    assert (got - ref).abs().max().item() <= 2e-2


@pytest.mark.L0
@pytest.mark.parametrize("splits", [3, 4, 8], ids=lambda s: f"split{s}")
def test_split_kv_uneven_and_empty_splits(splits):
    """5 KV tiles over 3/4/8 splits: unequal chunks, and at 8 three EMPTY splits.

    An empty split ends with total_sum == 0, so its epilogue writes O := 0 /
    LSE := -inf — the identity of the combine's log-sum-exp.  Getting this wrong
    shows up as NaN or as a wrong normalization, not as a small drift.
    """
    B, H, SQ, SKV = 1, 4, 128, 5 * TILE_N
    assert SKV // TILE_N == 5
    got, (q, k, v, scale) = _run(splits, B, H, H, SQ, SKV, torch.float16, causal=False)
    ref = _ref_sdpa(q, k, v, scale, is_causal=False, kh=H)
    assert (got - ref).abs().max().item() <= 2e-2


@pytest.mark.L0
@pytest.mark.parametrize("splits", [2, 4], ids=lambda s: f"split{s}")
def test_split_kv_causal(splits):
    """Causal: the split must cut the MASKED range, so each split gets real work."""
    B, H, SQ, SKV = 1, 4, 1024, 1024
    got, (q, k, v, scale) = _run(splits, B, H, H, SQ, SKV, torch.float16, causal=True)
    ref = _ref_sdpa(q, k, v, scale, is_causal=True, kh=H)
    assert (got - ref).abs().max().item() <= 2e-2


@pytest.mark.L0
@pytest.mark.parametrize("splits", [4], ids=lambda s: f"split{s}")
def test_split_kv_gqa(splits):
    B, H, KH, SQ, SKV = 2, 8, 2, 128, 2048
    got, (q, k, v, scale) = _run(splits, B, H, KH, SQ, SKV, torch.float16, causal=False)
    ref = _ref_sdpa(q, k, v, scale, is_causal=False, kh=KH)
    assert (got - ref).abs().max().item() <= 2e-2


@pytest.mark.L0
def test_split_kv_matches_unsplit():
    """Split vs unsplit on identical inputs — the split is occupancy-only."""
    B, H, SQ, SKV = 1, 4, 128, 2048
    base, _ = _run(1, B, H, H, SQ, SKV, torch.float16, causal=False)
    for splits in (2, 4, 8):
        got, _ = _run(splits, B, H, H, SQ, SKV, torch.float16, causal=False)
        # Both round to fp16; the split reassociates the sum, so allow one ulp
        # of drift at fp16 magnitudes rather than demanding bit-equality.
        assert (got - base).abs().max().item() <= 2e-2, f"split{splits} diverged from unsplit"


@pytest.mark.L0
def test_split_kv_bf16():
    B, H, SQ, SKV = 1, 4, 128, 2048
    got, (q, k, v, scale) = _run(4, B, H, H, SQ, SKV, torch.bfloat16, causal=False)
    ref = _ref_sdpa(q, k, v, scale, is_causal=False, kh=H)
    assert (got - ref).abs().max().item() <= 1e-1  # bf16 has ~8 mantissa bits


@pytest.mark.L0
def test_split_kv_rejects_unsupported_combos():
    """The config validator is the backstop for what the combine cannot express."""
    from cudnn.frost.tile_dsl.scheduler import SCHED_LPT
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d128

    with pytest.raises(ValueError, match="split_kv"):
        make_cfg_d128(TemplateParams(split_kv=0))
    # Both scheduler policies are supported: NATURAL carries the split on the
    # batch axis, LPT on the flattened x axis (make_split_helpers._lpt_split_of).
    assert make_cfg_d128(TemplateParams(split_kv=4, sched_policy=SCHED_LPT))[0].SPLIT_KV == 4
    with pytest.raises(ValueError, match="sink"):
        make_cfg_d128(TemplateParams(split_kv=4, has_sink=True))
    with pytest.raises(ValueError, match="dense-only"):
        make_cfg_d128(TemplateParams(split_kv=4, thd_varlen=True, seq_kv_lens_present=True))


@pytest.mark.L0
def test_split_kv_requires_lse():
    """has_lse=False + split is rejected: the per-split LSE IS the combine weight."""
    mod = _kernel_module(4, 3, causal=False)
    with pytest.raises(ValueError, match="has_lse"):
        mod.compile(b=1, qh=4, kh=4, sq=128, skv=2048, d_qk=D, d_v=D, has_lse=False)


# --- cga1 (CTA_MMA=1) ---------------------------------------------------
#
# cga1 drops the collective 2-CTA MMA: one independent CTA per tile, covering
# TILES_Q*TILE_M = 256 Q rows instead of 512.  It halves both the wasted MMA
# work at small S_q and the CTAs per tile, so with KV split twice as many splits
# fit in one wave.  It is SMEM-neutral only because make_cfg_d128 turns QO_ALIAS
# on for cga1 (no collective MMA to halve per-CTA K/V), which is the part most
# likely to break -- hence the direct cga1-vs-cga2 output comparison below.


@pytest.mark.L0
@pytest.mark.parametrize("splits", [1, 4, 8], ids=lambda s: f"split{s}")
def test_cga1_matches_reference(splits):
    B, H, SQ, SKV = 1, 4, 128, 2048
    got, (q, k, v, scale) = _run(splits, B, H, H, SQ, SKV, torch.float16, causal=False, cta_mma=1)
    ref = _ref_sdpa(q, k, v, scale, is_causal=False, kh=H)
    assert (got - ref).abs().max().item() <= 2e-2


@pytest.mark.L0
def test_cga1_matches_cga2():
    """Same inputs through both cluster widths — cga1 is an occupancy change only."""
    B, H, SQ, SKV = 1, 4, 128, 2048
    for splits in (1, 4):
        a, _ = _run(splits, B, H, H, SQ, SKV, torch.float16, causal=False, cta_mma=2)
        b, _ = _run(splits, B, H, H, SQ, SKV, torch.float16, causal=False, cta_mma=1)
        assert (a - b).abs().max().item() <= 2e-2, f"cga1 diverged from cga2 at split{splits}"


@pytest.mark.L0
def test_cga1_causal_and_gqa():
    B, H, KH, SQ, SKV = 1, 8, 2, 1024, 1024
    got, (q, k, v, scale) = _run(4, B, H, KH, SQ, SKV, torch.float16, causal=True, cta_mma=1)
    ref = _ref_sdpa(q, k, v, scale, is_causal=True, kh=KH)
    assert (got - ref).abs().max().item() <= 2e-2


@pytest.mark.L0
def test_cga1_requires_qo_alias_and_smem_fits():
    """cga1 must enable QO_ALIAS, and both widths must stay inside the SMEM cap."""
    from dataclasses import replace

    from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d128, _validate_cfg_d128, _d128_smem_bytes, _SM100_MAX_DYN_SMEM

    cfg1, _ = make_cfg_d128(TemplateParams(cta_mma=1))
    cfg2, _ = make_cfg_d128(TemplateParams(cta_mma=2))
    assert cfg1.QO_ALIAS == 1 and cfg2.QO_ALIAS == 0
    assert cfg1.CGA_M == cfg1.CTA_MMA == 1 and cfg2.CGA_M == cfg2.CTA_MMA == 2
    # Both fit, and cga1 only fits *because* of the alias.
    assert _d128_smem_bytes(cfg1) <= _SM100_MAX_DYN_SMEM
    assert _d128_smem_bytes(cfg2) <= _SM100_MAX_DYN_SMEM
    assert _d128_smem_bytes(replace(cfg1, QO_ALIAS=0)) > _SM100_MAX_DYN_SMEM
    with pytest.raises(ValueError, match="QO_ALIAS is mandatory"):
        _validate_cfg_d128(replace(cfg1, QO_ALIAS=0))
    with pytest.raises(ValueError, match="cta_mma"):
        make_cfg_d128(TemplateParams(cta_mma=3))


@pytest.mark.L0
def test_split_kv_and_cta_mma_flavor_gating():
    """Flavors must REJECT knobs they do not honour, not ignore them.

    ``split_kv`` / ``cta_mma`` sit on the TemplateParams shared by every SM100
    flavor, but a flavor only honours them once its make_cfg_* threads them into
    a Cfg AND its kernel reads them.  Silently ignoring split_kv is a
    wrong-answer bug, not a no-op: the caller sizes an (S*B)-batch partial
    workspace and runs the combine while the kernel writes only slots [0, B),
    leaving the rest at lse_partial = 0 instead of -inf — weight exp(0 - M) != 0,
    so they corrupt the reduction rather than dropping out.
    """
    from cudnn.sdpa.fwd.config_sm100 import (
        _CTA_MMA_FLAVORS,
        _SPLIT_KV_FLAVORS,
        TemplateParams,
        make_cfg_d128,
        make_cfg_d192,
        make_cfg_d256,
        make_cfg_d512,
    )

    mk = {"d128": make_cfg_d128, "d192": make_cfg_d192, "d256": make_cfg_d256, "d512": make_cfg_d512}
    for name, f in mk.items():
        f(TemplateParams())  # defaults must always build
        if name in _SPLIT_KV_FLAVORS:
            assert f(TemplateParams(split_kv=4))[0].SPLIT_KV == 4
        else:
            with pytest.raises(ValueError, match="split_kv is not implemented"):
                f(TemplateParams(split_kv=4))
        if name in _CTA_MMA_FLAVORS:
            assert f(TemplateParams(cta_mma=1))[0].CTA_MMA == 1
        else:
            with pytest.raises(ValueError, match="cta_mma is not selectable"):
                f(TemplateParams(cta_mma=1))


# --- empty-split coverage across every f16 flavor -------------------------
#
# More splits than KV tiles leaves some splits with an EMPTY range. That path is
# what deadlocked d128 during bring-up: the empty-tile handshake
# (mb_empty_mainloop) was compile-time gated on MASK_FLAGS != 0, so at MASK_NONE
# the producers ran a full prologue while correction took the empty path and the
# kernel hung. d192 additionally has its own CAN_HAVE_EMPTY_KV predicate gating
# nine more sites. Every flavor therefore needs this exercised, not just d128 --
# a regression here shows up as a HANG, not a wrong number.

_F16_FLAVORS = {
    "d128": ("prefill_d128_f16_sm100.py", 128, 128),
    "d192": ("prefill_d192_d128_f16_sm100.py", 192, 128),
    "d256": ("prefill_d256_f16_sm100.py", 256, 256),
    "d512": ("prefill_d512_f16_sm100.py", 512, 512),
}


@pytest.mark.L0
@pytest.mark.parametrize("flavor", sorted(_F16_FLAVORS))
def test_empty_splits_every_flavor(flavor):
    """5 KV tiles over 8 splits -> 3 empty splits, on every f16 flavor."""
    import math as _math
    import os as _os

    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.kernels import split_combine_sm100 as comb

    kmod, d_qk, d_v = _F16_FLAVORS[flavor]
    B, H, SQ, SKV, S = 1, 4, 128, 5 * TILE_N, 8
    assert SKV // TILE_N < S, "this test must leave some splits empty"
    dev = "cuda"
    scale = 1.0 / _math.sqrt(d_qk)
    torch.manual_seed(0)

    q = torch.randn(B, SQ, H, d_qk, device=dev, dtype=torch.float16)
    k = torch.randn(B, SKV, H, d_qk, device=dev, dtype=torch.float16)
    v = torch.randn(B, SKV, H, d_v, device=dev, dtype=torch.float16)

    path = _os.path.join(_os.path.dirname(_os.path.abspath(api_dsl.__file__)), "kernels", kmod)
    mod = load_template(path, TemplateParams(dtype_qkv=3, split_kv=S), tag=f"empty_{flavor}_{S}")
    fn = mod.compile(b=B, qh=H, kh=H, sq=SQ, skv=SKV, d_qk=d_qk, d_v=d_v, has_lse=True)

    o_p = torch.zeros(S * B, SQ, H, d_v, device=dev, dtype=torch.float16)
    lse_p = torch.zeros(S * B, H, SQ, device=dev, dtype=torch.float32)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        q,
        k,
        v,
        o_p,
        lse_p,
        torch.zeros(H, dtype=torch.float32, device=dev),
        torch.zeros(B, dtype=torch.int32, device=dev),
        torch.zeros(1, dtype=torch.int64, device=dev),
        (B, H, H, SQ, SKV, 0),
        cutlass.Float32(scale * _math.log2(_math.e)),
        cutlass.Int32(0),
        None,
        stream=stream,
    )
    o_out = torch.zeros(B, SQ, H, d_v, device=dev, dtype=torch.float16)
    cfn = comb.compile(b=B, h=H, sq=SQ, d_v=d_v, splits=S, dtype_o="f16", has_lse=False)
    cfn(o_p, lse_p, o_out, None, None, (B, H, SQ, d_v), cutlass.Int32(S), stream=stream)
    torch.cuda.synchronize()

    assert not torch.isnan(o_out).any(), f"{flavor}: NaN from an empty split"
    ref = _ref_sdpa(q, k, v, scale, is_causal=False, kh=H)
    assert (o_out.float() - ref).abs().max().item() <= 2e-2


# --- cga1 dtype coverage --------------------------------------------------
#
# The d128 flavor's config gate admits cta_mma=1 for every dtype, because
# f16/bf16, fp8 and mxfp8 all share make_cfg_d128. That is correct for
# f16/bf16/fp8 but NOT for mxfp8: its per-32-block E8M0 scale factors are staged
# in SMEM by the mxfp8 kernel alone, which config_sm100._d128_smem_bytes cannot
# see, and they push a cga1 CTA to 237024 B against the 232448 B cap. Without
# the kernel-local guard the launch is rejected by the driver at runtime.


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_cga1_half_dtypes(dtype):
    B, H, SQ, SKV = 1, 4, 128, 2048
    got, (q, k, v, scale) = _run(4, B, H, H, SQ, SKV, dtype, causal=False, cta_mma=1)
    ref = _ref_sdpa(q, k, v, scale, is_causal=False, kh=H)
    tol = 2e-2 if dtype == torch.float16 else 1e-1
    assert (got - ref).abs().max().item() <= tol


@pytest.mark.L0
def test_cga1_stage_depth_scales_with_cluster_width():
    """cga1 halves the KV stage depth for the fp8 family, as cuDNN's kernels do.

    cuDNN scales the stage count with the cluster width (stages_kv = N * CTA_MMA)
    so that stages x per-CTA-buffer -- and hence SMEM -- stays constant. FROST
    needs the same for fp8/mxfp8: at STAGES_KV=4 a cga1 CTA asked for 237024 B
    against the 232448 B sm_100a cap, because mxfp8 also stages E8M0 scale
    factors that _d128_smem_bytes cannot see. f16/bf16 already fit at cga1 via
    the Q/O alias and keep STAGES_KV=2.
    """
    from cudnn.frost.tile_dsl.constants import DTYPE_E4M3, DTYPE_FP16
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d128

    for cta_mma, want_fp8_stages in ((2, 4), (1, 2)):
        cfg_fp8, _ = make_cfg_d128(TemplateParams(dtype_qkv=DTYPE_E4M3, dtype_o=DTYPE_FP16, cta_mma=cta_mma))
        assert cfg_fp8.STAGES_KV == want_fp8_stages, (cta_mma, cfg_fp8.STAGES_KV)
        cfg_f16, _ = make_cfg_d128(TemplateParams(dtype_qkv=DTYPE_FP16, cta_mma=cta_mma))
        assert cfg_f16.STAGES_KV == 2, (cta_mma, cfg_f16.STAGES_KV)
    # cga1 must also turn the Q/O alias on, which is what pays for the doubled
    # per-CTA K/V in the first place.
    assert make_cfg_d128(TemplateParams(dtype_qkv=DTYPE_FP16, cta_mma=1))[0].QO_ALIAS == 1


# --- fp8 / mxfp8 split coverage ------------------------------------------
#
# These two flavors share make_cfg_d128 with f16/bf16 but have their own ABIs
# (amax outputs for fp8; per-32-block E8M0 scale factors for mxfp8), so they are
# not reachable through _run above. Both cluster widths are exercised: mxfp8 at
# cga1 in particular only fits once STAGES_KV halves with the cluster width.


def _fp8_family_split(kfile, dtype_qkv, splits, cta_mma, mx):
    import math as _math
    import os as _os

    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_FP16
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.kernels import split_combine_sm100 as comb

    B, H, SQ, SKV, D = 1, 4, 128, 2048, 128
    dev = "cuda"
    scale = 1.0 / _math.sqrt(D)
    torch.manual_seed(0)
    kdir = _os.path.join(_os.path.dirname(_os.path.abspath(api_dsl.__file__)), "kernels")
    params = TemplateParams(dtype_qkv=dtype_qkv, dtype_o=DTYPE_FP16, split_kv=splits, cta_mma=cta_mma)
    mod = load_template(_os.path.join(kdir, kfile), params, tag=f"t_{kfile[:16]}_{splits}_{cta_mma}")
    fn = mod.compile(b=B, qh=H, kh=H, sq=SQ, skv=SKV, has_lse=True)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    o_p = torch.zeros(splits * B, SQ, H, D, device=dev, dtype=torch.float16)
    lse_p = torch.zeros(splits * B, H, SQ, device=dev, dtype=torch.float32)
    amax_o = torch.zeros(1, dtype=torch.float32, device=dev)
    zH = torch.zeros(H, dtype=torch.float32, device=dev)
    zB = torch.zeros(B, dtype=torch.int32, device=dev)
    ps = (B, H, H, SQ, SKV, 0)
    log2e = cutlass.Float32(scale * _math.log2(_math.e))

    if not mx:
        mk = lambda *sh: (torch.randn(*sh, device=dev) * 0.5).clamp(-448, 448).to(torch.float8_e4m3fn)
        q, k, v = mk(B, SQ, H, D), mk(B, SKV, H, D), mk(B, SKV, H, D)
        # The FP8 entry takes four 1-element fp32 DEVICE scale tensors
        # (descale_q/k/v, scale_o) — the scales fold in-kernel — and no Amax_S.
        one = lambda: torch.ones(1, dtype=torch.float32, device=dev)
        # o_desc dummy + n_thd_units=0: THD-only ABI slots (dense fold), like the f16 call above.
        o_desc = torch.zeros(1, dtype=torch.int64, device=dev)
        fn(q, k, v, o_p, lse_p, zH, zB, o_desc, ps, log2e, cutlass.Float32(1.0), cutlass.Int32(0), one(), one(), one(), one(), amax_o, stream=stream)
        qf, kf, vf = (t.float().permute(0, 2, 1, 3) for t in (q, k, v))
    else:
        from sdpa.mxfp8_quant import quantize_to_mxfp8

        qr = torch.randn(B, H, SQ, D, device=dev) * 0.5
        kr = torch.randn(B, H, SKV, D, device=dev) * 0.5
        vr = torch.randn(B, H, SKV, D, device=dev) * 0.5
        a, adq, aswz, b_, bdq, bswz = quantize_to_mxfp8(qr, B, H, SQ, D)
        q8, sfq, qf = a, aswz, adq.reshape(B, H, SQ, D).float()
        a, adq, aswz, b_, bdq, bswz = quantize_to_mxfp8(kr, B, H, SKV, D)
        k8, sfk, kf = a, aswz, adq.reshape(B, H, SKV, D).float()
        a, adq, aswz, b_, bdq, bswz = quantize_to_mxfp8(vr, B, H, SKV, D)
        v8, sfv, vf = b_, bswz, bdq.reshape(B, H, SKV, D).float()
        sfq, sfk, sfv = (t.reshape(B, H, -1, 512).view(torch.int8).contiguous() for t in (sfq, sfk, sfv))
        q8 = q8.reshape(B, H, SQ, D).permute(0, 2, 1, 3).contiguous()
        k8 = k8.reshape(B, H, SKV, D).permute(0, 2, 1, 3).contiguous()
        v8 = v8.reshape(B, H, SKV, D).permute(0, 2, 1, 3).contiguous()
        # o_desc dummy + n_thd_units=0: THD-only ABI slots (dense fold), like the f16 call above.
        o_desc = torch.zeros(1, dtype=torch.int64, device=dev)
        fn(q8, k8, v8, o_p, sfq, sfk, sfv, lse_p, amax_o, zH, zB, o_desc, ps, log2e, cutlass.Int32(0), stream=stream)

    ref = torch.matmul(torch.softmax(torch.matmul(qf, kf.transpose(-1, -2)) * scale, -1), vf).permute(0, 2, 1, 3)
    if splits == 1:
        torch.cuda.synchronize()
        return o_p.float(), ref, amax_o
    o_out = torch.zeros(B, SQ, H, D, device=dev, dtype=torch.float16)
    # has_amax: at splits > 1 the per-split epilogues skip their amax write, so
    # the combine is what reports it -- over the RECOMBINED O.
    cfn = comb.compile(b=B, h=H, sq=SQ, d_v=D, splits=splits, dtype_o="f16", has_lse=False, has_amax=True)
    cfn(o_p, lse_p, o_out, None, amax_o, (B, H, SQ, D), cutlass.Int32(splits), stream=stream)
    torch.cuda.synchronize()
    assert not torch.isnan(o_out).any(), "NaN in combined fp8-family O"
    return o_out.float(), ref, amax_o


def _assert_amax_is_of_the_output(amax_o, got):
    """amax_o must describe the OUTPUT the caller receives, at any split count.

    The per-split epilogues each see only their own partial, and the recombined
    O is a convex combination of those, so |O| <= max_s |O_s|: a max taken over
    partials silently over-reports (measured 1.5x at 2 splits, 2.9x at 8) and
    would hand the caller a far-too-loose quantization scale.  Comparing against
    the recombined tensor is what catches that; comparing against the partials
    would pass either way.
    """
    reported = amax_o.item()
    true_amax = got.abs().max().item()
    # The kernel takes its amax on the fp32 value before the half store, so the
    # readback can differ by one rounding step in that direction only.
    assert reported >= true_amax * 0.99, f"amax_o {reported} under-reports |O| {true_amax}"
    assert reported <= true_amax * 1.01, f"amax_o {reported} over-reports |O| {true_amax} — computed over partials?"


@pytest.mark.L0
@pytest.mark.parametrize("cta_mma", [2, 1], ids=["cga2", "cga1"])
@pytest.mark.parametrize("splits", [1, 4], ids=lambda s: f"split{s}")
def test_split_kv_fp8(splits, cta_mma):
    from cudnn.frost.tile_dsl.constants import DTYPE_E4M3

    got, ref, amax_o = _fp8_family_split("prefill_d128_fp8_sm100.py", DTYPE_E4M3, splits, cta_mma, mx=False)
    assert (got - ref).abs().max().item() <= 5e-2
    _assert_amax_is_of_the_output(amax_o, got)


@pytest.mark.L0
@pytest.mark.parametrize("cta_mma", [2, 1], ids=["cga2", "cga1"])
@pytest.mark.parametrize("splits", [1, 4], ids=lambda s: f"split{s}")
def test_split_kv_mxfp8(splits, cta_mma):
    from cudnn.frost.tile_dsl.constants import DTYPE_E4M3

    got, ref, amax_o = _fp8_family_split("prefill_d128_mxfp8_sm100.py", DTYPE_E4M3, splits, cta_mma, mx=True)
    assert (got - ref).abs().max().item() <= 1.5e-1
    _assert_amax_is_of_the_output(amax_o, got)


# --- gaps found by auditing the suite against what the config permits -----


@pytest.mark.L0
@pytest.mark.parametrize("splits", [1, 4], ids=lambda s: f"split{s}")
def test_combine_lse_matches_reference(splits):
    """The RECOMBINED LSE is an output too, and nothing else here checks it.

    split_combine_sm100 computes lse = M + log(sum_s exp(lse_s - M)); every other
    test only compares O, so a wrong LSE would pass all of them.
    """
    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.sdpa.fwd.kernels import split_combine_sm100 as comb

    B, H, SQ, SKV = 2, 4, 128, 2048
    dev = "cuda"
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(0)
    q = torch.randn(B, SQ, H, D, device=dev, dtype=torch.float16)
    k = torch.randn(B, SKV, H, D, device=dev, dtype=torch.float16)
    v = torch.randn(B, SKV, H, D, device=dev, dtype=torch.float16)

    mod = _kernel_module(splits, 3, causal=False)
    fn = mod.compile(b=B, qh=H, kh=H, sq=SQ, skv=SKV, d_qk=D, d_v=D, has_lse=True)
    o_p = torch.zeros(splits * B, SQ, H, D, device=dev, dtype=torch.float16)
    lse_p = torch.zeros(splits * B, H, SQ, device=dev, dtype=torch.float32)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        q,
        k,
        v,
        o_p,
        lse_p,
        torch.zeros(H, dtype=torch.float32, device=dev),
        torch.zeros(B, dtype=torch.int32, device=dev),
        torch.zeros(1, dtype=torch.int64, device=dev),
        (B, H, H, SQ, SKV, 0),
        cutlass.Float32(scale * math.log2(math.e)),
        cutlass.Int32(0),
        None,
        stream=stream,
    )

    # Reference LSE = logsumexp of the scaled scores, in natural log.
    qb, kb = (t.float().permute(0, 2, 1, 3) for t in (q, k))
    ref_lse = torch.logsumexp(torch.matmul(qb, kb.transpose(-1, -2)) * scale, dim=-1)  # [B,H,SQ]

    if splits == 1:
        torch.cuda.synchronize()
        got_lse = lse_p.view(B, H, SQ)
    else:
        o_out = torch.zeros(B, SQ, H, D, device=dev, dtype=torch.float16)
        lse_out = torch.zeros(B, H, SQ, device=dev, dtype=torch.float32)
        cfn = comb.compile(b=B, h=H, sq=SQ, d_v=D, splits=splits, dtype_o="f16", has_lse=True)
        cfn(o_p, lse_p, o_out, lse_out, None, (B, H, SQ, D), cutlass.Int32(splits), stream=stream)
        torch.cuda.synchronize()
        got_lse = lse_out
    assert not torch.isnan(got_lse).any(), "NaN in recombined LSE"
    assert (got_lse - ref_lse).abs().max().item() <= 5e-3


@pytest.mark.L0
@pytest.mark.parametrize("flavor", sorted(_F16_FLAVORS))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_even_splits_every_flavor_batched(flavor, dtype):
    """Even splits, B > 1, both half dtypes, on every flavor.

    B > 1 matters specifically: the split-major slot is batch + split*B and the
    grid.z decode is b % n_batch / b // n_batch, all of which are trivially
    correct at B == 1 and so were barely exercised.
    """
    import os as _os

    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.kernels import split_combine_sm100 as comb

    kmod, d_qk, d_v = _F16_FLAVORS[flavor]
    B, H, SQ, SKV, S = 3, 4, 128, 2048, 4
    dev = "cuda"
    scale = 1.0 / math.sqrt(d_qk)
    torch.manual_seed(0)
    q = torch.randn(B, SQ, H, d_qk, device=dev, dtype=dtype)
    k = torch.randn(B, SKV, H, d_qk, device=dev, dtype=dtype)
    v = torch.randn(B, SKV, H, d_v, device=dev, dtype=dtype)

    path = _os.path.join(_os.path.dirname(_os.path.abspath(api_dsl.__file__)), "kernels", kmod)
    params = TemplateParams(dtype_qkv=3 if dtype == torch.float16 else 2, split_kv=S)
    mod = load_template(path, params, tag=f"even_{flavor}_{dtype}_{S}")
    fn = mod.compile(b=B, qh=H, kh=H, sq=SQ, skv=SKV, d_qk=d_qk, d_v=d_v, has_lse=True)
    o_p = torch.zeros(S * B, SQ, H, d_v, device=dev, dtype=dtype)
    lse_p = torch.zeros(S * B, H, SQ, device=dev, dtype=torch.float32)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        q,
        k,
        v,
        o_p,
        lse_p,
        torch.zeros(H, dtype=torch.float32, device=dev),
        torch.zeros(B, dtype=torch.int32, device=dev),
        torch.zeros(1, dtype=torch.int64, device=dev),
        (B, H, H, SQ, SKV, 0),
        cutlass.Float32(scale * math.log2(math.e)),
        cutlass.Int32(0),
        None,
        stream=stream,
    )
    o_out = torch.zeros(B, SQ, H, d_v, device=dev, dtype=dtype)
    cfn = comb.compile(b=B, h=H, sq=SQ, d_v=d_v, splits=S, dtype_o="f16" if dtype == torch.float16 else "bf16", has_lse=False)
    cfn(o_p, lse_p, o_out, None, None, (B, H, SQ, d_v), cutlass.Int32(S), stream=stream)
    torch.cuda.synchronize()
    assert not torch.isnan(o_out).any(), f"{flavor}: NaN"
    ref = _ref_sdpa(q, k, v, scale, is_causal=False, kh=H)
    assert (o_out.float() - ref).abs().max().item() <= (2e-2 if dtype == torch.float16 else 1e-1)


# --- masks the config permits with split, previously untested -------------
#
# split_kv only rejects sinks and THD, so SWA / bottom-right / padded are all
# reachable. They are also where _bounds_for_tile_split does its real work:
# it slices the ALREADY-masked [left, right) and clamps unmasked_lo/hi into the
# slice, and none of that was exercised. The reference is the one the sibling
# suite uses to model this kernel's exact mask semantics.


def _run_masked(kfile, d_qk, d_v, splits, *, B, H, KH, SQ, SKV, tp_kwargs, seq_kv_lens=None, seq_q_lens=None, dtype=torch.float16, cta_mma=2):
    """Launch split kernel + combine under an arbitrary mask; returns (got, q, k, v, scale)."""
    import os as _os

    import cutlass
    import cuda.bindings.driver as cuda_driver

    from cudnn.frost.template_loader import load_template
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.kernels import split_combine_sm100 as comb

    dev = "cuda"
    scale = 1.0 / math.sqrt(d_qk)
    torch.manual_seed(0)
    q = torch.randn(B, SQ, H, d_qk, device=dev, dtype=dtype)
    k = torch.randn(B, SKV, KH, d_qk, device=dev, dtype=dtype)
    v = torch.randn(B, SKV, KH, d_v, device=dev, dtype=dtype)

    path = _os.path.join(_os.path.dirname(_os.path.abspath(api_dsl.__file__)), "kernels", kfile)
    params = TemplateParams(dtype_qkv=3 if dtype == torch.float16 else 2, split_kv=splits, cta_mma=cta_mma, **tp_kwargs)
    mod = load_template(path, params, tag=f"mask_{kfile[:16]}_{splits}_{cta_mma}_{sorted(tp_kwargs.items())}")
    fn = mod.compile(b=B, qh=H, kh=KH, sq=SQ, skv=SKV, d_qk=d_qk, d_v=d_v, has_lse=True)

    o_p = torch.zeros(splits * B, SQ, H, d_v, device=dev, dtype=dtype)
    lse_p = torch.zeros(splits * B, H, SQ, device=dev, dtype=torch.float32)
    skv_t = seq_kv_lens if seq_kv_lens is not None else torch.zeros(B, dtype=torch.int32, device=dev)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        q,
        k,
        v,
        o_p,
        lse_p,
        torch.zeros(H, dtype=torch.float32, device=dev),
        skv_t,
        torch.zeros(1, dtype=torch.int64, device=dev),
        (B, H, KH, SQ, SKV, 0),
        cutlass.Float32(scale * math.log2(math.e)),
        cutlass.Int32(0),
        seq_q_lens,
        stream=stream,
    )
    if splits == 1:
        torch.cuda.synchronize()
        return o_p.float(), q, k, v, scale
    o_out = torch.zeros(B, SQ, H, d_v, device=dev, dtype=dtype)
    cfn = comb.compile(b=B, h=H, sq=SQ, d_v=d_v, splits=splits, dtype_o="f16" if dtype == torch.float16 else "bf16", has_lse=False)
    cfn(o_p, lse_p, o_out, None, None, (B, H, SQ, d_v), cutlass.Int32(splits), stream=stream)
    torch.cuda.synchronize()
    assert not torch.isnan(o_out).any(), "NaN under mask+split"
    return o_out.float(), q, k, v, scale


def _bhsd(t):
    return t.permute(0, 2, 1, 3)


@pytest.mark.L0
@pytest.mark.parametrize("splits", [1, 4], ids=lambda s: f"split{s}")
def test_split_kv_swa_causal(splits):
    """Causal + sliding window: both mask bits, so unmasked_lo AND hi are live."""
    from test_sdpa_fwd_dsl_sm100 import _ref_sdpa_full

    W = 256
    B, H, SQ, SKV = 1, 4, 1024, 1024
    got, q, k, v, scale = _run_masked(
        "prefill_d128_f16_sm100.py", 128, 128, splits, B=B, H=H, KH=H, SQ=SQ, SKV=SKV, tp_kwargs=dict(window_right=0, window_left=W)
    )
    ref = _ref_sdpa_full(_bhsd(q), _bhsd(k), _bhsd(v), scale=scale, is_causal=True, swa_window=W)
    assert (got - ref.float().permute(0, 2, 1, 3)).abs().max().item() <= 2e-2


@pytest.mark.L0
@pytest.mark.parametrize("splits", [1, 4], ids=lambda s: f"split{s}")
def test_split_kv_bottom_right_causal(splits):
    """Bottom-right anchored causal — the diagonal sits at (S_q, S_kv)."""
    from test_sdpa_fwd_dsl_sm100 import _ref_sdpa_full

    B, H, SQ, SKV = 1, 4, 128, 2048
    got, q, k, v, scale = _run_masked(
        "prefill_d128_f16_sm100.py", 128, 128, splits, B=B, H=H, KH=H, SQ=SQ, SKV=SKV, tp_kwargs=dict(window_right=0, bottom_right=True)
    )
    ref = _ref_sdpa_full(_bhsd(q), _bhsd(k), _bhsd(v), scale=scale, is_causal=True, bottom_right=True)
    assert (got - ref.float().permute(0, 2, 1, 3)).abs().max().item() <= 2e-2


@pytest.mark.L0
@pytest.mark.parametrize("splits", [1, 4], ids=lambda s: f"split{s}")
def test_split_kv_padded_kv(splits):
    """Per-batch KV padding: the split must slice each batch's OWN live range."""
    from test_sdpa_fwd_dsl_sm100 import _ref_sdpa_full

    B, H, SQ, SKV = 2, 4, 128, 2048
    lens = torch.tensor([2048, 1531], dtype=torch.int32, device="cuda")  # 2nd ends mid-tile
    got, q, k, v, scale = _run_masked(
        "prefill_d128_f16_sm100.py", 128, 128, splits, B=B, H=H, KH=H, SQ=SQ, SKV=SKV, tp_kwargs=dict(seq_kv_lens_present=True), seq_kv_lens=lens
    )
    ref = _ref_sdpa_full(_bhsd(q), _bhsd(k), _bhsd(v), scale=scale, seq_kv_lens=lens)
    assert (got - ref.float().permute(0, 2, 1, 3)).abs().max().item() <= 2e-2


@pytest.mark.L0
@pytest.mark.parametrize("flavor", ["d192", "d256", "d512"])
def test_split_kv_causal_other_flavors(flavor):
    """Causal + split beyond d128 — _split_chunk slices the masked range here too."""
    from test_sdpa_fwd_dsl_sm100 import _ref_sdpa_full

    kmod, d_qk, d_v = _F16_FLAVORS[flavor]
    B, H, SQ, SKV = 1, 4, 1024, 1024
    got, q, k, v, scale = _run_masked(kmod, d_qk, d_v, 4, B=B, H=H, KH=H, SQ=SQ, SKV=SKV, tp_kwargs=dict(window_right=0))
    ref = _ref_sdpa_full(_bhsd(q), _bhsd(k), _bhsd(v), scale=scale, is_causal=True)
    assert (got - ref.float().permute(0, 2, 1, 3)).abs().max().item() <= 2e-2


@pytest.mark.L0
@pytest.mark.parametrize("splits", [1, 4], ids=lambda s: f"split{s}")
@pytest.mark.parametrize("flavor", ["d128", "d512"])
def test_split_kv_padded_q_trim(flavor, splits):
    """Dense padded-Q trim (seq_q_lens) + split.

    Q rows at or past the batch's actual length must come back O := 0 (cuDNN
    >= 9.14 convention). Under split that has to hold for EVERY split's partial,
    or the combine mixes live and dead rows -- the trim is applied per split in
    the epilogue, after which the dead row's lse = -inf makes it drop out.

    """
    from test_sdpa_fwd_dsl_sm100 import _ref_sdpa_full

    kmod, d_qk, d_v = _F16_FLAVORS[flavor]
    # SQ spans several CGA tiles (ROWS_PER_CLUSTER = 512 on d128) so the short
    # batch has both a mid-tile trim AND fully collapsed tiles past its length.
    B, H, SQ, SKV = 2, 4, 1024, 2048
    kv_lens = torch.tensor([2048, 2048], dtype=torch.int32, device="cuda")
    q_lens = torch.tensor([1024, 137], dtype=torch.int32, device="cuda")  # 2nd trims mid-tile
    got, q, k, v, scale = _run_masked(
        kmod,
        d_qk,
        d_v,
        splits,
        B=B,
        H=H,
        KH=H,
        SQ=SQ,
        SKV=SKV,
        tp_kwargs=dict(seq_kv_lens_present=True, seq_q_lens_present=True),
        seq_kv_lens=kv_lens,
        seq_q_lens=q_lens,
    )
    ref = _ref_sdpa_full(_bhsd(q), _bhsd(k), _bhsd(v), scale=scale, seq_kv_lens=kv_lens)
    ref = ref.float().permute(0, 2, 1, 3)  # -> BSHD
    # Rows past the per-batch Q length must be exactly zero.
    rows = torch.arange(SQ, device=got.device).view(1, SQ, 1, 1)
    dead = rows >= q_lens.view(B, 1, 1, 1)
    assert got[dead.expand_as(got)].abs().max().item() == 0.0, "trimmed Q rows are not zero"
    live = ~dead
    assert (got - ref)[live.expand_as(got)].abs().max().item() <= 2e-2
