# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MXFP8 block backward's GEMM STAGES on their own -- ``_GemmStage(block_scale=True)`` over the K-major block-scale
drivers of ``kernels/proj_gemm.py`` -- before the block assembles them.

* The block-scale ``_GemmStage`` (``api_bwd.py``): B7 / B8 declared the MXFP8 way (``dtype=e4m3, block_scale=True,
  out_dtype=torch.bfloat16``; no alpha -- the E8M0 dequant is exact in the MMA; the 64-byte MMA K resolved from an UNSET
  ``mma_tile_k_bytes`` at declaration) compile to the forced tile's 64-byte-MMA-K twin
  (``..._128x256x64_cluster2x1_2ctamma``, named through ``tile_config.as_mma_tile_k`` and pinned against BOTH the GEMM
  suite's spelling and the block's own ``_forced_tile_name``), carry no alpha and a bf16 output, declare both operands
  K-major (``("k", "k")``), and -- fed two K-major transposed e4m3 artifacts with their padded F8_128x4 scale-factor
  blobs -- match the fp64 reference of the operands dequantized THROUGH THE BLOBS under the GEMM suite's bound
  (``test_proj_gemm_bwd._assert_close_vs_fp64``: rtol 2^-7, atol rtol * max|ref|), two launches bitwise, sentinel-free,
  and bitwise the bare driver's result.  Every other declaration is a typed decline naming the field (host): a ragged
  token axis (``K % 32``) names the fixes (a multiple of 32, ``need_dw_qkvg=False``, the per-tensor fp8 backward),
  ``alpha``, a non-bf16 ``out_dtype``, a 32-byte MMA K, a non-e4m3 ``dtype``, a ``w_dtype`` outside the stage's OWN served
  set (``_GemmStage.BLOCK_SCALE_W_DTYPES``: e4m3 today -- the e2m1 cell follows that tuple, so the fp4 weight modes'
  backward lifts the guard without an edit here), an unserved ``(sf_dtype, block_size)`` pair; the ``K`` multiple a decline
  names is the scale block ``block_scale_pairing`` resolves, never a number of this module's; the blobs are required iff the
  plan is block-scale and refused otherwise; the drivers' operand checks (a ``.t()`` view, a slab slice, a wrong-sized blob)
  surface through ``execute`` before any launch; a stage at the default kwargs is the per-tensor stage, byte-identical.

Tolerances are the GEMM suite's own (``_check_fp8_cell`` -> ``_assert_close_vs_fp64``), never a new one.  Accept tests
need the Rubin device the block binds (``requires_rubin``); the declaration cells run on any host, the driver-surfacing
cells on any CUDA device.
"""

import functools
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block.api_bwd import GatedAttentionBlockBwd, _GemmStage, _OutProjDgrad, _OutProjWgrad, _QkvGateDgrad, _QkvGateWgrad  # noqa: E402
from cudnn.gated_attention_block.kernels.proj_gemm import (  # noqa: E402
    ProjGemmPlan,
    block_scale_pairing,
    run_dgrad_gemm_block_scale,
    run_wgrad_gemm_block_scale,
    sf_blob_bytes,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_proj_gemm_bwd import _FORCED_TILE_K64, _GEOMS, _SENTINEL, _check_fp8_cell, _fp4_operand, _mx_operand, _stage_mkn  # noqa: E402

# The REGISTERED marker of cutedsl/conftest.py (the skip is applied at collection).
requires_rubin = pytest.mark.requires_rubin
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_FP8 = getattr(torch, "float8_e4m3fn", None)
_E2M1 = getattr(torch, "float4_e2m1fn_x2", None)
needs_fp8 = pytest.mark.skipif(_FP8 is None, reason="this torch has no float8_e4m3fn")
needs_fp4 = pytest.mark.skipif(_FP8 is None or _E2M1 is None, reason="this torch has no float8_e4m3fn / float4_e2m1fn_x2")

# The two block-scale GEMMs of the MXFP8 backward and the two that stay per-tensor; `_stage_mkn` (the GEMM suite) spells
# their (m, k, n) at the block's two geometries.
_BLOCK_SCALE_STAGES = {"B7_dw_qkvg": _QkvGateWgrad, "B8_dh": _QkvGateDgrad}
_PER_TENSOR_STAGES = {"B1_dw_o": _OutProjWgrad, "B2_do_gated": _OutProjDgrad}


def _mx_stage(stage: str, m: int, k: int, n: int, **over):
    """A backward GEMM stage declared the MXFP8 way -- no ``mma_tile_k_bytes`` (the stage resolves it), no alpha; ``over``
    perturbs one field for the decline cells."""
    kw = dict(m=m, k=k, n=n, dtype=_FP8, label=f"mxfp8_{stage}", block_scale=True, out_dtype=torch.bfloat16)
    kw.update(over)
    return {**_BLOCK_SCALE_STAGES, **_PER_TENSOR_STAGES}[stage](**kw)


@functools.lru_cache(maxsize=None)
def _compiled_stage(stage: str, geom_id: str, t: int):
    st = _mx_stage(stage, *_stage_mkn(stage, geom_id, t))
    st.check_support()
    st.compile()
    return st


class _SpyJit:
    """Stands in for a plan's JIT: records the variant pack of every launch it is handed, launches nothing."""

    def __init__(self):
        """No launches recorded yet."""
        self.calls = []

    def __call__(self, vp, **kw):
        """Record the variant pack; the launch kwargs (stream, workspace) are accepted and ignored."""
        self.calls.append(vp)


def _hand_built_block_scale_plan(m: int, k: int, n: int, label: str) -> ProjGemmPlan:
    """A block-scale plan with no graph and a spy JIT: the drivers' operand checks run, nothing launches."""
    plan = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=m, k=k, n=n, label=label, dtype=_FP8, out_dtype=torch.bfloat16, block_scale=True)
    plan.jit = _SpyJit()
    return plan


# ---------------------------------------------------------------------------
# Declaration (host)
# ---------------------------------------------------------------------------


@needs_fp8
def test_mxfp8_gemm_stage_declines_are_typed():
    """The block-scale stage is served in ONE form; every other declaration is typed and names its field.  The served form:
    ``block_scale=True`` on e4m3 codes, no alpha, a bf16 output, the 64-byte MMA K resolved from an unset ``mma_tile_k_bytes``
    (an explicit 64 is the same declaration), ``("k", "k")`` majors, and the forced tile's K64 twin named by BOTH
    ``expected_tile_config_name()`` and the block's ``_forced_tile_name`` before any compile.  Declines: a ragged token axis
    (``K % 32``: the message names ``need_dw_qkvg``'s fix, the multiple of 32 and the per-tensor backward; a dgrad's ragged K
    takes the feature-width wording), ``alpha=True``, an unset / e4m3 / half / fp32 ``out_dtype``, a 32- or 48-byte MMA K, a
    bf16 / fp16 ``dtype`` with ``block_scale``, a ``w_dtype`` outside ``_GemmStage.BLOCK_SCALE_W_DTYPES`` (bf16 always; e2m1
    while that tuple names e4m3 alone -- the cell follows the tuple, so the fp4 weight modes' backward serves it here without
    an edit), an unserved ``(sf_dtype, block_size)`` pair (the driver's ``block_scale_pairing`` message) and a non-catalog
    block size; the four block-scale fields on a per-tensor stage are refused by name (e4m3 and bf16 alike), never dropped;
    ``execute(sf_a=, sf_b=)`` both directions and ``alpha`` on a block-scale plan, typed before any driver (stand-in plans)."""
    import cudnn

    m, k, n = _stage_mkn("B7_dw_qkvg", "test", 2048)
    served = _mx_stage("B7_dw_qkvg", m, k, n)
    served.check_support()
    assert served.block_scale and served.is_e4m3 and served.alpha is False and served.out_dtype == torch.bfloat16
    assert served.mma_tile_k_bytes == 64, "an unset mma_tile_k_bytes resolves to the 64-byte MMA K at declaration"
    assert served.w_dtype is None and served.block_size == 32 and served.sf_dtype is None
    assert served.majors == ("k", "k")
    assert served.expected_tile_config_name() == _FORCED_TILE_K64 == GatedAttentionBlockBwd._forced_tile_name(served)
    _mx_stage("B7_dw_qkvg", m, k, n, mma_tile_k_bytes=64).check_support()  # the explicit 64 is the same declaration
    m8, k8, n8 = _stage_mkn("B8_dh", "test", 2048)
    dgrad = _mx_stage("B8_dh", m8, k8, n8)
    dgrad.check_support()
    assert dgrad.majors == ("k", "k") and dgrad.mma_tile_k_bytes == 64
    assert dgrad.expected_tile_config_name() == _FORCED_TILE_K64 == GatedAttentionBlockBwd._forced_tile_name(dgrad)
    for geom_id in _GEOMS:
        for stage in _BLOCK_SCALE_STAGES:
            st = _mx_stage(stage, *_stage_mkn(stage, geom_id, 2048))
            st.check_support()
            assert st.expected_tile_config_name() == _FORCED_TILE_K64 == GatedAttentionBlockBwd._forced_tile_name(st), (stage, geom_id)
    # K % 32 on the token-axis wgrad: T = 1000 and T = 2000 are refused with the fixes named; T = 2016 = 63 x 32 is served
    with pytest.raises(ValueError, match="need_dw_qkvg") as ei:
        _mx_stage("B7_dw_qkvg", m, 1000, n).check_support()
    msg = str(ei.value)
    assert "K=1000" in msg and "multiple of 32" in msg and "quant=QuantSpec" in msg and "T = B*S" in msg, msg
    # the multiple the decline names is the scale block the driver's pairing resolves for the served pair, not a number of the stage's
    _, k_block = block_scale_pairing(dtype=_FP8, w_dtype=_FP8, sf_dtype=served.sf_dtype, block_size=served.block_size, label="pin")
    assert k_block == 32 and f"per {k_block}-element K block, so K must be a multiple of {k_block} " in msg, msg
    with pytest.raises(ValueError, match="need_dw_qkvg"):
        _mx_stage("B7_dw_qkvg", m, 2000, n).check_support()
    _mx_stage("B7_dw_qkvg", m, 2016, n).check_support()
    # a dgrad's ragged K (n_qkvg = 5128): the feature-width wording, no token-axis fix
    with pytest.raises(ValueError, match="multiple of 32") as ei:
        _mx_stage("B8_dh", 2048, 5128, 512).check_support()
    assert "need_dw_qkvg" not in str(ei.value) and "K=5128" in str(ei.value), str(ei.value)
    with pytest.raises(ValueError, match="alpha"):
        _mx_stage("B7_dw_qkvg", m, k, n, alpha=True).check_support()
    for bad in (None, _FP8, torch.float16, torch.float32):
        with pytest.raises(ValueError, match="out_dtype"):
            _mx_stage("B7_dw_qkvg", m, k, n, out_dtype=bad).check_support()
    for bad in (32, 48):
        with pytest.raises(ValueError, match="mma_tile_k_bytes"):
            _mx_stage("B7_dw_qkvg", m, k, n, mma_tile_k_bytes=bad).check_support()
    for dt in (torch.bfloat16, torch.float16):
        with pytest.raises(NotImplementedError, match="block_scale"):
            _mx_stage("B7_dw_qkvg", m, k, n, dtype=dt).check_support()
    # w_dtype: refused by name outside the stage's OWN served set.  `_GemmStage.BLOCK_SCALE_W_DTYPES` names e4m3 alone today (bf16
    # is never a block-scale code dtype); the e2m1 cell follows the tuple -- refused while it is absent, served at the default
    # (sf_dtype, block_size) once the fp4 weight modes' backward adds it -- so that lift needs no edit in this module.
    served_w = _GemmStage.BLOCK_SCALE_W_DTYPES
    assert _FP8 in served_w and torch.bfloat16 not in served_w, served_w
    for wdt in (torch.bfloat16,) + ((_E2M1,) if _E2M1 is not None else ()):
        if wdt in served_w:
            _mx_stage("B7_dw_qkvg", m, k, n, w_dtype=wdt).check_support()
        else:
            with pytest.raises(NotImplementedError, match="w_dtype"):
                _mx_stage("B7_dw_qkvg", m, k, n, w_dtype=wdt).check_support()
    _mx_stage("B7_dw_qkvg", m, k, n, w_dtype=_FP8).check_support()  # the same dtype spelled explicitly is the served row
    with pytest.raises(ValueError, match="no block-scale GEMM row"):
        _mx_stage("B7_dw_qkvg", m, k, n, block_size=16).check_support()  # E8M0 per 16 is no catalog row for e4m3 x e4m3
    with pytest.raises(ValueError, match="block_size"):
        _mx_stage("B7_dw_qkvg", m, k, n, block_size=64).check_support()
    with pytest.raises(ValueError, match="no block-scale GEMM row"):
        _mx_stage("B7_dw_qkvg", m, k, n, sf_dtype=cudnn.data_type.FP8_E4M3).check_support()
    _mx_stage("B7_dw_qkvg", m, k, n, sf_dtype=cudnn.data_type.FP8_E8M0).check_support()  # the default spelled explicitly
    # the block-scale fields on a per-tensor stage -- e4m3 and bf16 alike -- are refused by name, never dropped
    pt = dict(m=m, k=k, n=n, dtype=_FP8, label="b7", mma_tile_k_bytes=64, out_dtype=torch.bfloat16, alpha=True)
    _QkvGateWgrad(**pt).check_support()
    for over in (dict(w_dtype=_FP8), dict(block_size=16), dict(sf_dtype=cudnn.data_type.FP8_E8M0)):
        (field,) = over
        with pytest.raises(NotImplementedError, match=field):
            _QkvGateWgrad(**pt, **over).check_support()
        with pytest.raises(NotImplementedError, match=field):
            _QkvGateWgrad(m=m, k=k, n=n, dtype=torch.bfloat16, label="b7", **over).check_support()
    # a per-tensor stage's majors and unset MMA K are untouched by the append
    bf = _QkvGateWgrad(m=m, k=k, n=n, dtype=torch.bfloat16, label="b7")
    bf.check_support()
    assert bf.majors == ("m", "n") and bf.mma_tile_k_bytes is None and not bf.block_scale
    assert _QkvGateDgrad(m=m8, k=k8, n=n8, dtype=torch.bfloat16, label="b8").majors == ("k", "n")
    assert _QkvGateWgrad(**pt).majors == ("m", "n") and _QkvGateWgrad(**pt).mma_tile_k_bytes == 64
    # execute(sf_a=, sf_b=): required iff the plan is block-scale, refused otherwise; alpha refused on a block-scale plan --
    # typed before any driver call (stand-in plans, no launch)
    st = _mx_stage("B7_dw_qkvg", m, k, n)
    with pytest.raises(RuntimeError, match="compile"):
        st.execute(None, None, None, None, stream=0)
    st.plan = SimpleNamespace(has_alpha=False, block_scale=True)
    one = torch.zeros(1)
    with pytest.raises(ValueError, match="sf_a"):
        st.execute(None, None, None, None, stream=0, sf_b=one)
    with pytest.raises(ValueError, match="sf_b"):
        st.execute(None, None, None, None, stream=0, sf_a=one)
    with pytest.raises(ValueError, match="sf_a"):
        st.execute(None, None, None, None, stream=0)
    with pytest.raises(ValueError, match="alpha"):
        st.execute(None, None, None, None, stream=0, alpha=one, sf_a=one, sf_b=one)
    pt_st = _QkvGateWgrad(**pt)
    pt_st.plan = SimpleNamespace(has_alpha=True, block_scale=False)
    with pytest.raises(ValueError, match="sf_a"):
        pt_st.execute(None, None, None, None, stream=0, alpha=one, sf_a=one)
    with pytest.raises(ValueError, match="sf_b"):
        pt_st.execute(None, None, None, None, stream=0, alpha=one, sf_b=one)


def _fp4_stage_kw(row: str) -> dict:
    """The block-scale declaration of one fp4 row: the MIXED row (e4m3 A x e2m1 W; the pairing resolves E8M0 per 32) or the NVFP4 row
    (e2m1 x e2m1 at block 16 with e4m3 scales spelled -- ``sf_dtype=None`` resolves to the same)."""
    import cudnn

    if row == "mixed":
        return dict(w_dtype=_E2M1)
    return dict(dtype=_E2M1, w_dtype=_E2M1, block_size=16, sf_dtype=cudnn.data_type.FP8_E4M3)


# The fp4 weight modes' two dgrads as block-scale stages: B8 on the mixed row (an MXFP4 W_qkvg), B2 on the mixed row (an MXFP4 W_o with an
# MX-rowwise e4m3 dY) and B2 on the NVFP4 row (an NVFP4 W_o with dY cast to NVFP4) -- the GEMM suite's `_fp4_operand` builds the e2m1 sides.
_FP4_STAGE_CASES = [("B8_dh", "mixed"), ("B2_do_gated", "mixed"), ("B2_do_gated", "nvfp4")]


@needs_fp4
def test_fp4_gemm_stage_rows_are_typed():
    """The block-scale stage serves the fp4 weight modes' two dgrad rows next to the MXFP8 one -- :attr:`_GemmStage.BLOCK_SCALE_ROWS`:
    the MIXED row (e4m3 A, ``w_dtype`` e2m1; the pairing resolves E8M0 per 32) and the NVFP4 x NVFP4 row (``dtype = w_dtype`` e2m1,
    ``block_size=16``, e4m3 scales -- spelled or resolved from None) pass ``check_support``, resolve the 64-byte MMA K and name the forced
    tile's K64 twin through BOTH ``expected_tile_config_name()`` and the block's ``_forced_tile_name`` (the MXFP8 row's string);
    ``compile()`` forwards the declaration as given.  Declines, typed by name: the mixed row the other way round (e2m1 A x e4m3 W) and
    MXFP4 x MXFP4 (e2m1 x e2m1 at E8M0 per 32, also an e2m1 A with ``w_dtype`` None) -- rows no stage of this backward declares; the
    mixed pair at block 16 and with e4m3 scales (the pairing's refusals); a bf16 A with an e2m1 W; alpha on the NVFP4 row; ``K % 16``
    on the NVFP4 row; the e2m1 ``w_dtype`` on a per-tensor stage."""
    import cudnn

    m8, k8, n8 = _stage_mkn("B8_dh", "test", 2048)
    m2, k2, n2 = _stage_mkn("B2_do_gated", "test", 2048)
    for stage, row in _FP4_STAGE_CASES:
        m, k, n = (m8, k8, n8) if stage == "B8_dh" else (m2, k2, n2)
        st = _mx_stage(stage, m, k, n, **_fp4_stage_kw(row))
        st.check_support()
        assert st.block_scale and st.alpha is False and st.out_dtype == torch.bfloat16 and st.mma_tile_k_bytes == 64 and st.majors == ("k", "k")
        assert (st.is_e4m3, st.is_e2m1) == ((True, False) if row == "mixed" else (False, True)), (stage, row)
        assert st.expected_tile_config_name() == _FORCED_TILE_K64 == GatedAttentionBlockBwd._forced_tile_name(st), (stage, row)
    for geom_id in _GEOMS:
        for stage, row in _FP4_STAGE_CASES:
            st = _mx_stage(stage, *_stage_mkn(stage, geom_id, 2048), **_fp4_stage_kw(row))
            st.check_support()
            assert st.expected_tile_config_name() == _FORCED_TILE_K64 == GatedAttentionBlockBwd._forced_tile_name(st), (stage, row, geom_id)
    assert (_FP8, _E2M1, 32) in _GemmStage.BLOCK_SCALE_ROWS and (_E2M1, _E2M1, 16) in _GemmStage.BLOCK_SCALE_ROWS and _E2M1 in _GemmStage.BLOCK_SCALE_W_DTYPES
    _mx_stage("B2_do_gated", m2, k2, n2, dtype=_E2M1, w_dtype=_E2M1, block_size=16).check_support()  # sf_dtype None resolves to e4m3 for two e2m1 sides at 16
    _mx_stage("B2_do_gated", m2, k2, n2, w_dtype=_E2M1, sf_dtype=cudnn.data_type.FP8_E8M0).check_support()  # the mixed row's default spelled
    # rows the pairing serves but no stage of this backward declares
    with pytest.raises(NotImplementedError, match="A e2m1 x W e4m3") as ei:
        _mx_stage("B2_do_gated", m2, k2, n2, dtype=_E2M1, w_dtype=_FP8).check_support()
    assert "not a rendering this backward declares" in str(ei.value) and "A e4m3 x W e2m1 at one scale per 32" in str(ei.value), str(ei.value)
    with pytest.raises(NotImplementedError, match="A e2m1 x W e2m1 at one scale per 32"):
        _mx_stage("B2_do_gated", m2, k2, n2, dtype=_E2M1, w_dtype=_E2M1, block_size=32).check_support()  # MXFP4 x MXFP4
    with pytest.raises(NotImplementedError, match="A e2m1 x W e2m1 at one scale per 32"):
        _mx_stage("B2_do_gated", m2, k2, n2, dtype=_E2M1).check_support()  # w_dtype None = e2m1 at the default block 32: MXFP4 x MXFP4
    # the pairing's own refusals through the stage
    with pytest.raises(ValueError, match="no block-scale GEMM row"):
        _mx_stage("B8_dh", m8, k8, n8, w_dtype=_E2M1, block_size=16).check_support()
    with pytest.raises(ValueError, match="no block-scale GEMM row"):
        _mx_stage("B8_dh", m8, k8, n8, w_dtype=_E2M1, sf_dtype=cudnn.data_type.FP8_E4M3).check_support()
    with pytest.raises(NotImplementedError, match="block_scale"):
        _mx_stage("B8_dh", m8, k8, n8, dtype=torch.bfloat16, w_dtype=_E2M1).check_support()
    with pytest.raises(ValueError, match="alpha"):
        _mx_stage("B2_do_gated", m2, k2, n2, alpha=True, **_fp4_stage_kw("nvfp4")).check_support()
    with pytest.raises(ValueError, match="multiple of 16") as ei:
        _mx_stage("B2_do_gated", m2, 520, n2, **_fp4_stage_kw("nvfp4")).check_support()
    assert "float8_e4m3fn scale per 16-element K block" in str(ei.value) and "K=520" in str(ei.value), str(ei.value)
    with pytest.raises(NotImplementedError, match="w_dtype"):
        _QkvGateDgrad(m=m8, k=k8, n=n8, dtype=_FP8, label="b8", mma_tile_k_bytes=64, out_dtype=torch.bfloat16, alpha=True, w_dtype=_E2M1).check_support()


@needs_fp4
def test_fp4_gemm_stage_compile_forwards_the_fp4_declaration(monkeypatch):
    """``compile()`` hands ``build_proj_gemm`` the fp4 rows' declarations as given -- the mixed row's ``w_dtype=e2m1`` at the defaults
    ``block_size=32, sf_dtype=None``, the NVFP4 row's ``dtype=w_dtype=e2m1, block_size=16, sf_dtype=FP8_E4M3`` -- with the resolved
    ``mma_tile_k_bytes=64``, ``alpha=False``, a bf16 ``out_dtype`` and the K-major majors (a spy stands in for the driver)."""
    import cudnn

    import cudnn.gated_attention_block.kernels.proj_gemm as pg

    seen = {}

    def spy(**kw):
        """Stands in for ``build_proj_gemm``: records the kwargs the stage hands it, returns a placeholder plan."""
        seen.clear()
        seen.update(kw)
        return "plan"

    monkeypatch.setattr(pg, "build_proj_gemm", spy)
    for stage, row in _FP4_STAGE_CASES:
        m, k, n = _stage_mkn(stage, "test", 2048)
        st = _mx_stage(stage, m, k, n, **_fp4_stage_kw(row))
        st.check_support()
        st.compile()
        assert st.plan == "plan"
        want = (True, _E2M1, 32, None) if row == "mixed" else (True, _E2M1, 16, cudnn.data_type.FP8_E4M3)
        assert (seen["block_scale"], seen["w_dtype"], seen["block_size"], seen["sf_dtype"]) == want, (stage, row, seen)
        assert seen["dtype"] == (_FP8 if row == "mixed" else _E2M1) and (seen["m"], seen["k"], seen["n"]) == (m, k, n), seen
        assert (seen["a_major"], seen["b_major"]) == ("k", "k") and seen["mma_tile_k_bytes"] == 64 and seen["alpha"] is False, seen
        assert seen["out_dtype"] == torch.bfloat16, seen


@requires_rubin
@needs_fp4
@pytest.mark.parametrize("stage,row", _FP4_STAGE_CASES, ids=[f"{c[0]}-{c[1]}" for c in _FP4_STAGE_CASES])
def test_fp4_gemm_stage_execute_matches_fp64_through_the_blobs(stage, row):
    """The fp4 rows' stages at the test geometry (T = 2048): each plan IS the forced tile's K64 twin on the FROST JIT with the row's scale
    dtype and block; ``execute(sf_a=, sf_b=)`` over the GEMM suite's operands -- the mixed row's e4m3 gradient x PACKED e2m1 weight, the
    NVFP4 row's packed e2m1 gradient x packed e2m1 weight -- vs the fp64 product of the operands dequantized THROUGH THE BLOBS under the
    bf16-output bound; two launches bitwise; bitwise the bare driver's result; a missing blob is a typed refusal before any launch."""
    import cudnn

    m, k, n = _stage_mkn(stage, "test", 2048)
    st = _mx_stage(stage, m, k, n, **_fp4_stage_kw(row))
    st.check_support()
    st.compile()
    plan = st.plan
    assert plan.jit is not None, f"{stage} {row}: no JIT artifact -- the forced compile fell back to the graph heuristic (route {plan.route!r})"
    assert plan.tile_config_name == _FORCED_TILE_K64 == st.expected_tile_config_name() == GatedAttentionBlockBwd._forced_tile_name(st), (
        plan.tile_config_name,
        plan.route,
    )
    assert plan.mma_tile_k_bytes == 64 == plan.jit.config.mma_tile_k_bytes and (plan.a_major, plan.b_major) == ("k", "k") and not plan.has_alpha
    if row == "mixed":
        assert (plan.dtype, plan.w_dtype, plan.block_size, plan.sf_dtype) == (_FP8, _E2M1, 32, cudnn.data_type.FP8_E8M0)
        a, sf_a, a64 = _mx_operand(m, k, seed=1)
        w, sf_w, w64 = _fp4_operand(n, k, "mxfp4", seed=2)
    else:
        assert (plan.dtype, plan.w_dtype, plan.block_size, plan.sf_dtype) == (_E2M1, _E2M1, 16, cudnn.data_type.FP8_E4M3)
        a, sf_a, a64 = _fp4_operand(m, k, "nvfp4", seed=1)
        w, sf_w, w64 = _fp4_operand(n, k, "nvfp4", seed=2)
    ws = torch.empty(st.workspace_bytes(), dtype=torch.uint8, device="cuda")
    out1 = torch.full((m, n), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    out2, out3 = out1.clone(), out1.clone()
    stream = torch.cuda.current_stream().cuda_stream
    st.execute(a, w, out1, ws, stream=stream, sf_a=sf_a, sf_b=sf_w)
    st.execute(a, w, out2, ws, stream=stream, sf_a=sf_a, sf_b=sf_w)
    run_dgrad_gemm_block_scale(plan, a, w, out3, ws, sf_dy=sf_a, sf_w_t=sf_w, stream=stream)
    torch.cuda.synchronize()
    _check_fp8_cell(out1, out2, a64 @ w64.T, f"fp4 {row} {stage} @ test, T=2048, {plan.tile_config_name} ({plan.route})")
    assert torch.equal(out1, out3), f"{stage} {row}: the stage's output differs from the bare driver's"
    with pytest.raises(ValueError, match="sf_b"):
        st.execute(a, w, out1, ws, stream=stream, sf_a=sf_a)
    print(f"\n{stage} {row}: {plan.tile_config_name}, route {plan.route}, workspace {plan.workspace_bytes} B")


@needs_fp8
def test_mxfp8_gemm_stage_compile_forwards_the_block_scale_declaration(monkeypatch):
    """``compile()`` hands ``build_proj_gemm`` the block-scale declaration as given -- ``block_scale=True``, ``w_dtype=None``,
    ``block_size=32``, ``sf_dtype=None`` -- with the resolved ``mma_tile_k_bytes=64``, ``alpha=False``, a bf16 ``out_dtype`` and the
    K-major majors on both stages (a spy stands in for the driver; nothing is built)."""
    import cudnn.gated_attention_block.kernels.proj_gemm as pg

    seen = {}

    def spy(**kw):
        """Stands in for ``build_proj_gemm``: records the kwargs the stage hands it, returns a placeholder plan."""
        seen.clear()
        seen.update(kw)
        return "plan"

    monkeypatch.setattr(pg, "build_proj_gemm", spy)
    for stage in _BLOCK_SCALE_STAGES:
        m, k, n = _stage_mkn(stage, "test", 2048)
        st = _mx_stage(stage, m, k, n)
        st.check_support()
        st.compile()
        assert st.plan == "plan"
        assert (seen["m"], seen["k"], seen["n"], seen["dtype"], seen["label"]) == (m, k, n, _FP8, f"mxfp8_{stage}"), seen
        assert (seen["a_major"], seen["b_major"]) == ("k", "k") and seen["mma_tile_k_bytes"] == 64, seen
        assert (seen["block_scale"], seen["w_dtype"], seen["block_size"], seen["sf_dtype"]) == (True, None, 32, None), seen
        assert seen["alpha"] is False and seen["out_dtype"] == torch.bfloat16, seen


@requires_cuda
@needs_fp8
def test_mxfp8_gemm_stage_execute_surfaces_the_drivers_operand_checks():
    """``execute`` on a block-scale plan dispatches to ``run_wgrad_gemm_block_scale`` / ``run_dgrad_gemm_block_scale``, whose
    checks run BEFORE any launch and surface through the stage with the DRIVER's words: an operand handed as the ``.t()`` view of
    the un-transposed tensor (right shape, wrong stride-1 axis), a column slice of a wider slab (row stride != K), a wrong-sized
    blob (named by the driver's keyword -- ``sf_x_t`` / ``sf_dy_t`` on the wgrad, ``sf_w_t`` / ``sf_dy`` on the dgrad -- with the
    F8_128x4 arithmetic).  The declared operands reach the (spy) launch exactly once per call."""
    dm, n_qkvg, t = 512, 5120, 2048
    dev = "cuda"
    stream = torch.cuda.current_stream().cuda_stream
    ws = torch.empty(1, dtype=torch.uint8, device=dev)
    # the wgrad: dQKVG^T [N, T] against h^T [dm, T]
    st = _mx_stage("B7_dw_qkvg", n_qkvg, t, dm)
    st.check_support()
    st.plan = _hand_built_block_scale_plan(n_qkvg, t, dm, "bs_wgrad")
    a8 = torch.zeros(n_qkvg, t, dtype=_FP8, device=dev)
    w8 = torch.zeros(dm, t, dtype=_FP8, device=dev)
    dw = torch.zeros(n_qkvg, dm, dtype=torch.bfloat16, device=dev)
    sf_a = torch.zeros(sf_blob_bytes(n_qkvg, t), dtype=torch.uint8, device=dev)
    sf_w = torch.zeros(sf_blob_bytes(dm, t), dtype=torch.uint8, device=dev)
    with pytest.raises(
        ValueError, match=r"dy_t \(dy_like\^T, \[rows, T\]\) has strides \(1, 5120\) but this plan declared a contiguous row-major \[5120, 2048\]"
    ):
        st.execute(torch.zeros(t, n_qkvg, dtype=_FP8, device=dev).t(), w8, dw, ws, stream=stream, sf_a=sf_a, sf_b=sf_w)
    with pytest.raises(ValueError, match=r"x_t \(x\^T, \[cols, T\]\) has strides \(2112, 1\)"):
        st.execute(a8, torch.zeros(dm, t + 64, dtype=_FP8, device=dev)[:, :t], dw, ws, stream=stream, sf_a=sf_a, sf_b=sf_w)
    with pytest.raises(ValueError, match=r"sf_x_t has \d+ bytes; the F8_128x4 blob over 512 rows x K=2048"):
        st.execute(a8, w8, dw, ws, stream=stream, sf_a=sf_a, sf_b=sf_w[:-512])
    with pytest.raises(ValueError, match=r"sf_dy_t has \d+ bytes; the F8_128x4 blob over 5120 rows x K=2048"):
        st.execute(a8, w8, dw, ws, stream=stream, sf_a=sf_a[:-512], sf_b=sf_w)
    with pytest.raises(ValueError, match=r"dw view has shape \(1, 5120, 256\)"):
        st.execute(a8, w8, torch.zeros(n_qkvg, dm // 2, dtype=torch.bfloat16, device=dev), ws, stream=stream, sf_a=sf_a, sf_b=sf_w)
    assert not st.plan.jit.calls, "a refused operand reached the launch"
    st.execute(a8, w8, dw, ws, stream=stream, sf_a=sf_a, sf_b=sf_w)
    assert len(st.plan.jit.calls) == 1
    # the dgrad: dQKVG [T, N] against W_qkvg^T [dm, N]
    dg = _mx_stage("B8_dh", t, n_qkvg, dm)
    dg.check_support()
    dg.plan = _hand_built_block_scale_plan(t, n_qkvg, dm, "bs_dgrad")
    dy8 = torch.zeros(t, n_qkvg, dtype=_FP8, device=dev)
    wt8 = torch.zeros(dm, n_qkvg, dtype=_FP8, device=dev)
    dx = torch.zeros(t, dm, dtype=torch.bfloat16, device=dev)
    sf_dy = torch.zeros(sf_blob_bytes(t, n_qkvg), dtype=torch.uint8, device=dev)
    sf_wt = torch.zeros(sf_blob_bytes(dm, n_qkvg), dtype=torch.uint8, device=dev)
    with pytest.raises(ValueError, match=r"w_t \(w\^T, \[N, K\]\) has strides \(1, 512\)"):
        dg.execute(dy8, torch.zeros(n_qkvg, dm, dtype=_FP8, device=dev).t(), dx, ws, stream=stream, sf_a=sf_dy, sf_b=sf_wt)
    with pytest.raises(ValueError, match=r"sf_w_t has \d+ bytes; the F8_128x4 blob over 512 rows x K=5120"):
        dg.execute(dy8, wt8, dx, ws, stream=stream, sf_a=sf_dy, sf_b=sf_wt[:-512])
    with pytest.raises(ValueError, match=r"sf_dy has \d+ bytes; the F8_128x4 blob over 2048 rows x K=5120"):
        dg.execute(dy8, wt8, dx, ws, stream=stream, sf_a=sf_dy[:-512], sf_b=sf_wt)
    assert not dg.plan.jit.calls, "a refused operand reached the launch"
    dg.execute(dy8, wt8, dx, ws, stream=stream, sf_a=sf_dy, sf_b=sf_wt)
    assert len(dg.plan.jit.calls) == 1


# ---------------------------------------------------------------------------
# The compiled stages and their numerics (Rubin)
# ---------------------------------------------------------------------------


@requires_rubin
@needs_fp8
@pytest.mark.parametrize("geom_id", list(_GEOMS))
def test_mxfp8_gemm_stages_compile_to_the_forced_k64_tile(geom_id):
    """B7 / B8 at the test and the 397B column shapes (T = 2048): the plan IS the forced tile's K64 twin on the FROST JIT (a
    fallback to the heuristic is a FAILURE), block-scale with E8M0 scales per 32, no alpha, a bf16 output, both operands K-major,
    the 64-byte MMA K on the plan AND on the JIT config; the tile name is ONE string across the GEMM suite's spelling, the stage's
    ``expected_tile_config_name()`` and the block's ``_forced_tile_name``; the route (``graph+jit`` or ``jit-only``) is recorded."""
    import cudnn

    for stage in _BLOCK_SCALE_STAGES:
        st = _compiled_stage(stage, geom_id, 2048)
        plan = st.plan
        assert plan.jit is not None, f"{stage} @ {geom_id}: no JIT artifact -- the forced compile fell back to the graph heuristic (route {plan.route!r})"
        assert plan.block_scale and plan.sf_dtype == cudnn.data_type.FP8_E8M0 and plan.block_size == 32, (stage, geom_id, plan.sf_dtype, plan.block_size)
        assert plan.tile_config_name == _FORCED_TILE_K64 == st.expected_tile_config_name() == GatedAttentionBlockBwd._forced_tile_name(st), (
            stage,
            geom_id,
            plan.tile_config_name,
            plan.route,
        )
        assert plan.mma_tile_k_bytes == 64 == st.mma_tile_k_bytes and plan.jit.config.mma_tile_k_bytes == 64, (stage, geom_id, plan.mma_tile_k_bytes)
        assert not plan.has_alpha and plan.alpha is None and plan.out_dtype == torch.bfloat16 and plan.dtype == _FP8, (stage, geom_id)
        assert (plan.a_major, plan.b_major) == ("k", "k") == st.majors, (stage, geom_id, plan.a_major, plan.b_major)
        assert plan.route in ("graph+jit", "jit-only"), (stage, geom_id, plan.route)
        print(f"\n{stage} @ {geom_id}: {plan.tile_config_name}, route {plan.route}, workspace {plan.workspace_bytes} B")


# (stage, geom, T): both block-scale stages at the test geometry (T = 2048 and the ragged 32-multiple 2016 = 63 x 32: a partial
# CTA K tile for the wgrad, a partial M tile for the dgrad) and at the 397B column shapes at T = 2048 -- the driver suite's cases.
_NUMERICS_CASES = [
    ("B7_dw_qkvg", "test", 2048),
    ("B7_dw_qkvg", "test", 2016),
    ("B8_dh", "test", 2048),
    ("B8_dh", "test", 2016),
    ("B7_dw_qkvg", "397B", 2048),
    ("B8_dh", "397B", 2048),
]


@requires_rubin
@needs_fp8
@pytest.mark.parametrize("stage,geom_id,t", _NUMERICS_CASES, ids=[f"{c[0]}-{c[1]}-T{c[2]}" for c in _NUMERICS_CASES])
def test_mxfp8_gemm_stage_execute_matches_fp64_through_the_blobs(stage, geom_id, t):
    """Each block-scale stage over the GEMM suite's MXFP8 operands (e4m3 codes with DISTINCTIVE per-32-block E8M0 exponents, their
    padded F8_128x4 blobs: ``_mx_operand``): ``execute(sf_a=, sf_b=)`` vs the fp64 product of the operands dequantized THROUGH THE
    BLOBS under the GEMM suite's bf16-output bound; two launches bitwise; no sentinel survivor; bitwise the bare driver's result;
    a missing blob and an ``alpha`` are typed refusals before any launch."""
    st = _compiled_stage(stage, geom_id, t)
    m, k, n = st.m, st.k, st.n
    ws = torch.empty(st.workspace_bytes(), dtype=torch.uint8, device="cuda")
    a8, sf_a, a64 = _mx_operand(m, k, seed=1)  # A [m, k] K-major: dQKVG^T [N, T] (wgrad) or dQKVG [T, N] (dgrad)
    w8, sf_w, w64 = _mx_operand(n, k, seed=2)  # B [n, k] K-major: h^T [dm, T] (wgrad) or W_qkvg^T [dm, N] (dgrad)
    out1 = torch.full((m, n), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    out2 = out1.clone()
    out3 = out1.clone()
    stream = torch.cuda.current_stream().cuda_stream
    st.execute(a8, w8, out1, ws, stream=stream, sf_a=sf_a, sf_b=sf_w)
    st.execute(a8, w8, out2, ws, stream=stream, sf_a=sf_a, sf_b=sf_w)
    # the bare driver on the stage's own plan: the stage is a dispatcher and adds nothing numerically
    if st.kind == "wgrad":
        run_wgrad_gemm_block_scale(st.plan, a8, w8, out3, ws, sf_dy_t=sf_a, sf_x_t=sf_w, stream=stream)
    else:
        run_dgrad_gemm_block_scale(st.plan, a8, w8, out3, ws, sf_dy=sf_a, sf_w_t=sf_w, stream=stream)
    torch.cuda.synchronize()
    _check_fp8_cell(out1, out2, a64 @ w64.T, f"mxfp8 {stage} @ {geom_id}, T={t}, {st.plan.tile_config_name} ({st.plan.route})")
    assert torch.equal(out1, out3), f"{stage} @ {geom_id}, T={t}: the stage's output differs from the bare driver's"
    with pytest.raises(ValueError, match="sf_b"):
        st.execute(a8, w8, out1, ws, stream=stream, sf_a=sf_a)
    with pytest.raises(ValueError, match="alpha"):
        st.execute(a8, w8, out1, ws, stream=stream, alpha=torch.ones(1, device="cuda"), sf_a=sf_a, sf_b=sf_w)
