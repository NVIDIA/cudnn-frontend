# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The quantized block backward's STAGES, each on its own: the e4m3 projection GEMM stage, the fp8 SDPA stage and the
fp8 backward oracle -- before the block assembles them.

* ``_GemmStage`` over e4m3 (``api_bwd.py``): the four backward GEMM stages declared the quantized way (``dtype=e4m3,
  mma_tile_k_bytes=64, out_dtype=bf16, alpha=True``) compile to the forced tile's 64-byte-MMA-K twin
  (``..._128x256x64_cluster2x1_2ctamma``, named through ``tile_config.as_mma_tile_k`` and pinned against the GEMM suite's
  own spelling), carry the alpha epilogue and a bf16 output on the ``graph+jit`` route, and -- with the descale product
  written into a 4-byte-stride slot of a scalar block, the way the block's quantize launch publishes it -- match the fp64
  reference of the dequantized products under the GEMM suite's bound (``test_proj_gemm_bwd._assert_close_vs_fp64``:
  rtol 2^-7, atol rtol * max|ref|), two launches bitwise, sentinel-free.  Every other declaration is a typed decline
  naming the field (host).
* ``_SdpaBwdFp8``: the stage over ``SdpaBwdDslSm107Fp8(external_delta=True, amax_requested=("amax_dP",))``.  Synthetic
  e4m3 q / k / v / dO built the way the fp8 SDPA backward suite builds them (unit-normal draws, per-tensor amax scales), the
  exact fp64 LSE, the block's ``delta = rowsum(bf16 dO * bf16 O)`` (zeros on the pad rows), the twelve scalars as
  4-byte-stride slot views -- the stage's bf16 dQ / dK / dV against the row's own reference
  (``test/python/sdpa/fp8_ref.compute_ref_backward`` fed the SAME delta, LSE, scales and band) under the fp8 suite's recipe
  (atol 0.08 / rtol 0.2 with ``assert_close_fp8_grad``'s midpoint-flip budget), ``amax_dP`` within its tolerance of
  ``max |dS|``; causal and dense, GQA 8/2 and MHA, S in {256, 512, 992} (a padded q / kv tile), B = 2 once.  The stage's
  scratch carries no ``delta`` region.  The row's dead ``o`` is bound to an EXISTING e4m3 operand (``do8``, or ``og8``
  where the block carves it): the gradients are bitwise whatever is bound there, a NaN-filled buffer included.  The
  scalar dict is checked by name, dtype, count, device and alignment BEFORE the adapter (host).
* the oracle ``gated_block_reference.gated_attention_block_fp8_bwd_reference``: its SDPA node, fed the row suite's data,
  returns BITWISE the row's own reference; the whole oracle's dQ / dK / dV are bitwise that reference over the oracle's own
  e4m3 payloads / LSE / delta; the seeded mode reproduces the downstream bitwise; the unmodelled mode runs.

Tolerances are the two suites' own (never a new one): the GEMM bound above and the fp8 SDPA backward suite's
``_FP8_GRAD_TOL`` / ``_AMAX_DS_TOL`` (``test/python/sdpa/frost/test_sdpa_bwd_fp8_sm107.py``), restated here by value
with their origin named.  ``scale_s = 2 ** 8`` is the block's P scale (P <= 1); the row's own suite runs 2 ** 5 -- both
served regimes of one kernel.  Accept tests need the Rubin device the block binds (``requires_rubin``); the declaration,
scalar-dict and oracle cells run on any CUDA device.
"""

import functools
import math
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

from cudnn.gated_attention_block import GatedAttentionBlockGeometry, QuantSpec  # noqa: E402
from cudnn.gated_attention_block.api_bwd import _OutProjDgrad, _OutProjWgrad, _QkvGateDgrad, _QkvGateWgrad, _SdpaBwdFp8  # noqa: E402
from cudnn.sdpa.bwd.prepared_sm107 import FP8_SCALARS  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import (  # noqa: E402
    FP8_E4M3,
    RefGeometry,
    _Fp8RowCfg,
    _Fp8SdpaRow,
    _key_padding_and_causal_mask,
    fp64_attention,
    fp8_row_mask_args,
    gated_attention_block_fp8_bwd_reference,
    make_inputs,
    quant_e4m3,
    quantize_block_inputs,
)
from test_proj_gemm_bwd import _FORCED_TILE, _FORCED_TILE_K64, _GEOMS, _check_fp8_cell, _fp8_dgrad_operands, _fp8_wgrad_operands, _stage_mkn  # noqa: E402

# The REGISTERED marker of cutedsl/conftest.py (the skip is applied at collection).
requires_rubin = pytest.mark.requires_rubin
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_COMMON = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)
_D = 256
_SCALE_S_LOG2 = 8  # the block's P scale: P <= 1 -> P * 2**8 <= 256 < 448 (the row's suite runs 2**5; one kernel, two served regimes)
# The fp8 SDPA backward suite's recipe (test/python/sdpa/frost/test_sdpa_bwd_fp8_sm107.py, "tolerances: ONE place"), by value:
# dequantized gradients within atol 0.08 / rtol 0.2 of the fp8-modelled reference under assert_close_fp8_grad's midpoint-flip
# budget; amax_dP fp32 on both sides.  Reused, never widened.
_FP8_GRAD_TOL = dict(atol=0.08, rtol=0.2)
_AMAX_DS_TOL = dict(atol=1e-4, rtol=1e-2)

# (stage, class): the four backward GEMMs; `_stage_mkn` (the GEMM suite) spells their (m, k, n) at the block's two geometries.
_STAGES = {"B1_dw_o": _OutProjWgrad, "B7_dw_qkvg": _QkvGateWgrad, "B2_do_gated": _OutProjDgrad, "B8_dh": _QkvGateDgrad}


@pytest.fixture(autouse=True)
def _no_tf32():
    """An fp32 torch reference is a TF32 reference on Blackwell+ unless pinned; the row's reference is fp32, so pin and print."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"\nallow_tf32={torch.backends.cuda.matmul.allow_tf32}")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _e4m3_stage(stage: str, m: int, k: int, n: int, **over):
    """A backward GEMM stage declared the quantized way; ``over`` perturbs one field for the decline cells."""
    kw = dict(m=m, k=k, n=n, dtype=FP8_E4M3, label=f"fp8_{stage}", mma_tile_k_bytes=64, out_dtype=torch.bfloat16, alpha=True)
    kw.update(over)
    return _STAGES[stage](**kw)


@functools.lru_cache(maxsize=None)
def _compiled_stage(stage: str, geom_id: str, t: int):
    st = _e4m3_stage(stage, *_stage_mkn(stage, geom_id, t))
    st.check_support()
    st.compile()
    return st


def _cuda_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


# ---------------------------------------------------------------------------
# The e4m3 GEMM stage
# ---------------------------------------------------------------------------


def test_fp8_gemm_stage_declines_are_typed():
    """The e4m3 stage is served in ONE form; every other declaration is typed and names its field: e4m3 without ``alpha``,
    an e4m3 / half / unset ``out_dtype``, ``mma_tile_k_bytes`` unset or not a tcgen05 K width, ``K % 16 != 0`` (the wgrad's
    message names the ``B*S`` rule and its three fixes; the dgrad's does not); ``alpha`` / ``out_dtype`` on a bf16 stage;
    ``execute(alpha=)`` both directions, before the driver.  The default kwargs stay the bf16 stage's declaration."""
    m, k, n = _stage_mkn("B1_dw_o", "test", 2048)
    _e4m3_stage("B1_dw_o", m, k, n).check_support()  # the served form
    with pytest.raises(ValueError, match="alpha"):
        _e4m3_stage("B1_dw_o", m, k, n, alpha=False).check_support()
    for bad in (None, FP8_E4M3, torch.float16):
        with pytest.raises(ValueError, match="out_dtype"):
            _e4m3_stage("B1_dw_o", m, k, n, out_dtype=bad).check_support()
    with pytest.raises(ValueError, match="mma_tile_k_bytes"):
        _e4m3_stage("B1_dw_o", m, k, n, mma_tile_k_bytes=None).check_support()
    with pytest.raises(ValueError, match="mma_tile_k_bytes"):
        _e4m3_stage("B1_dw_o", m, k, n, mma_tile_k_bytes=48).check_support()
    with pytest.raises(ValueError, match=r"K % 16 == 0") as exc:
        _e4m3_stage("B1_dw_o", m, 1000, n).check_support()  # a weight gradient over T = 1000 tokens
    assert "B*S % 16 == 0" in str(exc.value) and "without weight gradients" in str(exc.value)
    with pytest.raises(ValueError, match=r"K % 16 == 0") as exc:
        _e4m3_stage("B8_dh", 2048, 5128, 512).check_support()  # a dgrad over K = n_qkvg = 5128
    assert "B*S" not in str(exc.value)
    for over in (dict(alpha=True), dict(out_dtype=torch.bfloat16)):
        with pytest.raises(NotImplementedError, match="alpha" if "alpha" in over else "out_dtype"):
            _OutProjWgrad(m=m, k=k, n=n, dtype=torch.bfloat16, label="b1", **over).check_support()
    with pytest.raises(NotImplementedError, match=r"mma_tile_k_bytes=64 is a knob of an 8-bit \(e4m3\) GEMM stage"):
        _OutProjWgrad(m=m, k=k, n=n, dtype=torch.bfloat16, label="b1", mma_tile_k_bytes=64).check_support()
    with pytest.raises(NotImplementedError, match="e4m3"):
        _OutProjWgrad(m=m, k=k, n=n, dtype=torch.float32, label="b1").check_support()
    st = _OutProjWgrad(m=m, k=k, n=n, dtype=torch.bfloat16, label="b1")
    st.check_support()
    assert st.out_dtype is None and st.alpha is False and st.mma_tile_k_bytes is None and not st.is_e4m3
    # execute(alpha=): required iff the plan carries it, typed before any driver call (a stand-in plan, no launch)
    st8 = _e4m3_stage("B1_dw_o", m, k, n)
    with pytest.raises(RuntimeError, match="compile"):
        st8.execute(None, None, None, None, stream=0)
    st8.plan = SimpleNamespace(has_alpha=True)
    with pytest.raises(ValueError, match="alpha"):
        st8.execute(None, None, None, None, stream=0)
    st.plan = SimpleNamespace(has_alpha=False)
    with pytest.raises(ValueError, match="alpha"):
        st.execute(None, None, None, None, stream=0, alpha=torch.zeros(1))


def test_fp8_gemm_stage_expected_tile_is_derived_from_the_forced_config():
    """``expected_tile_config_name()``: the forced K32 config's name for a bf16 stage and for an e4m3 stage at 32, its
    64-byte-MMA-K twin at 64 -- the GEMM suite's own spelling of both -- and ``None`` where the forced tile does not apply;
    every backward GEMM of both block geometries takes the forced tile."""
    m, k, n = _stage_mkn("B2_do_gated", "test", 2048)
    assert _e4m3_stage("B2_do_gated", m, k, n).expected_tile_config_name() == _FORCED_TILE_K64
    assert _e4m3_stage("B2_do_gated", m, k, n, mma_tile_k_bytes=32).expected_tile_config_name() == _FORCED_TILE
    assert _OutProjDgrad(m=m, k=k, n=n, dtype=torch.bfloat16, label="b2").expected_tile_config_name() == _FORCED_TILE
    assert _OutProjDgrad(m=m, k=k, n=224, dtype=torch.bfloat16, label="b2").expected_tile_config_name() is None
    for geom_id in _GEOMS:
        for stage in _STAGES:
            assert _e4m3_stage(stage, *_stage_mkn(stage, geom_id, 2048)).expected_tile_config_name() == _FORCED_TILE_K64, (stage, geom_id)


@requires_rubin
@pytest.mark.parametrize("geom_id", list(_GEOMS))
def test_fp8_gemm_stages_compile_to_the_forced_k64_tile(geom_id):
    """The four e4m3 stages at the test and the 397B column shapes (T = 2048): the plan IS the forced tile's K64 twin on
    the FROST JIT (a fallback to the heuristic is a FAILURE), carries the alpha epilogue and a bf16 output, records the
    64-byte MMA K, takes the ``graph+jit`` route with the stage's majors."""
    for stage in _STAGES:
        st = _compiled_stage(stage, geom_id, 2048)
        plan = st.plan
        assert plan.jit is not None, f"{stage} @ {geom_id}: no JIT artifact -- the forced compile fell back to the graph heuristic (route {plan.route!r})"
        assert plan.tile_config_name == _FORCED_TILE_K64 == st.expected_tile_config_name(), (stage, geom_id, plan.tile_config_name, plan.route)
        assert plan.has_alpha and plan.out_dtype == torch.bfloat16 and plan.dtype == FP8_E4M3 and not plan.block_scale, (stage, geom_id)
        assert plan.mma_tile_k_bytes == 64 and plan.jit.config.mma_tile_k_bytes == 64, (stage, geom_id, plan.mma_tile_k_bytes)
        assert plan.route == "graph+jit" and (plan.a_major, plan.b_major) == st.majors, (stage, geom_id, plan.route, plan.a_major, plan.b_major)


@requires_rubin
@pytest.mark.parametrize("stage", list(_STAGES))
def test_fp8_gemm_stage_execute_matches_fp64_with_a_slot_alpha(stage):
    """Each e4m3 stage at the test geometry (T = 2048) over the GEMM suite's fp8 operands, the descale product written
    into a 4-byte-stride slot of a scalar block (a torch write outside the stage, standing in for the block's quantize
    launch) and bound as ``execute(alpha=)``: vs ``(A^T @ X) * alpha`` / ``(A @ W) * alpha`` in fp64 under the GEMM
    suite's bf16-output bound; two launches bitwise; no sentinel survivor; ``alpha`` omitted is a typed refusal."""
    st = _compiled_stage(stage, "test", 2048)
    m, k, n = st.m, st.k, st.n
    ws = torch.empty(st.workspace_bytes(), dtype=torch.uint8, device="cuda")
    block = torch.zeros(64, dtype=torch.float32, device="cuda")  # a 256-B scalar block; slot 10 is 4-B aligned, not 16-B
    alpha = block[10:11]
    assert alpha.data_ptr() % 4 == 0 and alpha.data_ptr() % 16 == 8
    if st.kind == "wgrad":
        a8, d_a, b8, d_b, out1 = _fp8_wgrad_operands(m, k, n)
        prod64 = a8.double().T @ b8.double()
    else:
        a8, d_a, b8, d_b, out1 = _fp8_dgrad_operands(m, k, n)
        prod64 = a8.double() @ b8.double()
    torch.mul(d_a.reshape(1), d_b.reshape(1), out=alpha)  # ONE fp32 RN product into the slot, as the quantize launch publishes it
    out2 = out1.clone()
    stream = torch.cuda.current_stream().cuda_stream
    st.execute(a8, b8, out1, ws, stream=stream, alpha=alpha)
    st.execute(a8, b8, out2, ws, stream=stream, alpha=alpha)
    torch.cuda.synchronize()
    _check_fp8_cell(out1, out2, prod64 * alpha.double(), f"fp8 {stage} @ test, T=2048, K64, slot alpha ({st.plan.tile_config_name})")
    with pytest.raises(ValueError, match="alpha"):
        st.execute(a8, b8, out1, ws, stream=stream)


# ---------------------------------------------------------------------------
# The fp8 SDPA stage
# ---------------------------------------------------------------------------


@requires_cuda
def test_sdpa_bwd_fp8_stage_declares_the_row_contract():
    """The stage's declaration (no compile, any CUDA device): ``SdpaBwdDslSm107Fp8`` with ``external_delta=True``,
    ``amax_requested == {"amax_dP"}``, e4m3 payloads, bf16 gradients, no per-batch lengths, ``deterministic=False``, the
    geometry's mask; ``delta_shape`` is the adapter's ``(B, H_q, ceil128(S))``; the scratch carves NO ``delta`` region; the
    twelve scalar names are the row's; the typed declines (a non-bf16 ``grad_dtype``, ``d_head != 256``, a non-Rubin
    device) fire before any adapter work; ``execute`` before ``compile`` is a ``RuntimeError``."""
    geom = GatedAttentionBlockGeometry(**_COMMON)
    dev = _cuda_device()
    st = _SdpaBwdFp8(geom, batch=2, seq_len=500, grad_dtype=torch.bfloat16, device=dev)
    impl = st._build_impl()
    assert type(impl).__name__ == "SdpaBwdDslSm107Fp8"
    assert impl.external_delta is True and set(impl.amax_requested) == {"amax_dP"}
    assert impl.dtype == FP8_E4M3 and impl.grad_dtype == torch.bfloat16 and impl.seq_kv_lens_present is False and impl.deterministic is False
    assert impl.is_causal == geom.is_causal and impl.scale_softmax == pytest.approx(geom.scale) and impl.thd is False
    assert st.delta_shape == (2, 8, 512) == tuple(impl.external_delta_shape)
    regions = [name for name, _shape, _dtype in st._impl._scratch_plan()]
    assert "delta" not in regions and regions and st.scratch_workspace_bytes() == impl.scratch_workspace_bytes() > 0
    assert st.scalar_names() == tuple(FP8_SCALARS) and len(FP8_SCALARS) == 12
    with pytest.raises(NotImplementedError, match="grad_dtype"):
        _SdpaBwdFp8(geom, batch=1, seq_len=256, grad_dtype=torch.float16, device=dev).check_support()
    with pytest.raises(NotImplementedError, match="256"):
        _SdpaBwdFp8(
            GatedAttentionBlockGeometry(d_model=512, h_q=4, h_kv=4, d_head=128, rope_dim=64), batch=1, seq_len=256, grad_dtype=torch.bfloat16, device=dev
        ).check_support()
    if tuple(torch.cuda.get_device_capability()) != (10, 7):
        with pytest.raises(NotImplementedError, match="Rubin"):
            st.check_support()
    fresh = _SdpaBwdFp8(geom, batch=1, seq_len=256, grad_dtype=torch.bfloat16, device=dev)
    with pytest.raises(RuntimeError, match="compile"):
        fresh.execute(*([None] * 9), workspace=None, stream=0, delta=None, scalars={}, amax_dp=None)


@requires_cuda
def test_sdpa_bwd_fp8_stage_scalar_dict_is_checked_before_the_adapter():
    """``execute(scalars=)`` over a stand-in adapter: a missing name, an unexpected name, a CPU / fp64 / 2-element scalar
    and a bad ``amax_dp`` are typed ``ValueError``s naming the operand and the adapter is never reached; the good call hands
    every slot view through by identity, the amax / delta / workspace / stream as given, and the nine io tensors as
    ``(B, H, S, D)`` transposes of the compact buffers (BSHD-physical strides, no copy)."""
    geom = GatedAttentionBlockGeometry(**_COMMON)
    st = _SdpaBwdFp8(geom, batch=1, seq_len=256, grad_dtype=torch.bfloat16, device=_cuda_device())
    calls = []
    st._impl = SimpleNamespace(execute=lambda *args, **kw: calls.append((args, kw)))
    b, s, hq, hkv = 1, 256, geom.h_q, geom.h_kv
    e8 = lambda h: torch.zeros(b, s, h, _D, dtype=FP8_E4M3, device="cuda")  # noqa: E731
    bf = lambda h: torch.zeros(b, s, h, _D, dtype=torch.bfloat16, device="cuda")  # noqa: E731
    io = (e8(hq), e8(hkv), e8(hkv), e8(hq), e8(hq), torch.zeros(b, hq, s, device="cuda"), bf(hq), bf(hkv), bf(hkv))
    block = torch.zeros(64, dtype=torch.float32, device="cuda")  # the scalar block: 4-B-stride slot views
    scalars = {n: block[i : i + 1] for i, n in enumerate(FP8_SCALARS)}
    amax = block[14:15]
    kw = dict(
        workspace=torch.empty(16, dtype=torch.uint8, device="cuda"),
        stream=torch.cuda.current_stream().cuda_stream,
        delta=torch.zeros(b, hq, 256, device="cuda"),
    )
    bad = dict(scalars)
    bad.pop("descale_dP")
    with pytest.raises(ValueError, match="descale_dP"):
        st.execute(*io, scalars=bad, amax_dp=amax, **kw)
    with pytest.raises(ValueError, match="extra_scale"):
        st.execute(*io, scalars=dict(scalars, extra_scale=block[20:21]), amax_dp=amax, **kw)
    for wrong in (torch.ones(1), torch.ones(1, dtype=torch.float64, device="cuda"), torch.ones(2, device="cuda"), 1.0):
        with pytest.raises(ValueError, match="scale_dP"):
            st.execute(*io, scalars=dict(scalars, scale_dP=wrong), amax_dp=amax, **kw)
    with pytest.raises(ValueError, match="amax_dp"):
        st.execute(*io, scalars=scalars, amax_dp=torch.ones(1), **kw)
    with pytest.raises(ValueError, match="scalars"):
        st.execute(*io, scalars=list(scalars.values()), amax_dp=amax, **kw)
    assert not calls, "a refused scalar set must never reach the adapter"
    st.execute(*io, scalars=scalars, amax_dp=amax, **kw)
    assert len(calls) == 1
    args, got = calls[0]
    assert all(got[n] is scalars[n] for n in FP8_SCALARS)
    assert got["amax_dP"] is amax and got["delta_tensor"] is kw["delta"] and got["workspace"] is kw["workspace"]
    assert int(got["current_stream"]) == int(kw["stream"])
    assert len(args) == 9 and args[5] is io[5]
    for a, x in zip(args[:5] + args[6:], io[:5] + io[6:]):
        h = x.shape[2]
        assert a.shape == (b, h, s, _D) and a.data_ptr() == x.data_ptr() and a.stride() == (s * h * _D, _D, h * _D, 1), (a.shape, a.stride())


def _row_cell(b: int, s: int, hq: int, hkv: int, causal: bool, *, seed: int = 0):
    """One SDPA-stage cell built the fp8 SDPA backward suite's way: unit-normal q / k / v / dO quantized per tensor at
    their amax scales (dO rounded to bf16 FIRST -- the gate backward's buffer -- then e4m3 at ``scale_do``), the exact fp64
    forward over the dequantized codes -> the record's bf16 ``O`` and the exact fp32 natural-log LSE, the block's
    ``delta = rowsum(bf16 dO * bf16 O)`` in fp32 with zeros on the pad rows, and the row's reference (``compute_ref_backward``
    fed the SAME delta / LSE / band / scales; ``scale_dP`` from the reference's own dP amax -- the suite's delayed recipe;
    bf16 gradients at unit scale)."""
    from sdpa.fp8_ref import compute_ref_backward
    from sdpa.helpers import get_fp8_scale_factor

    geom = GatedAttentionBlockGeometry(d_model=hq * _D, h_q=hq, h_kv=hkv, d_head=_D, rope_dim=64, is_causal=causal)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(h):
        return torch.randn(b, s, h, _D, generator=gen).to("cuda")

    def quant(x):
        scale = get_fp8_scale_factor(x.abs().max().item(), FP8_E4M3)
        return quant_e4m3(x, scale), scale

    q8, sq = quant(draw(hq))
    k8, sk = quant(draw(hkv))
    v8, sv = quant(draw(hkv))
    do_bf16 = draw(hq).to(torch.bfloat16)
    scale_do = get_fp8_scale_factor(do_bf16.float().abs().max().item(), FP8_E4M3)
    do8 = quant_e4m3(do_bf16, scale_do)
    allowed = _key_padding_and_causal_mask(s, s, is_causal=causal, seq_lens=None, batch_index=0, q_lo=0, device="cuda", s_q_total=s)
    o64, lse64 = fp64_attention(q8.double() / sq, k8.double() / sk, v8.double() / sv, allowed, geom.scale)
    o_bf16 = o64.to(torch.bfloat16)
    lse = lse64.float().contiguous()  # [B, H_q, S], natural log -- the forward's exact Stats
    s_pad = -(-s // 128) * 128
    delta = torch.zeros(b, hq, s_pad, dtype=torch.float32, device="cuda")
    delta[:, :, :s] = (do_bf16.float() * o_bf16.float()).sum(-1).permute(0, 2, 1)
    scale_s = 2.0**_SCALE_S_LOG2
    left, right, align = fp8_row_mask_args(causal, False, -1)

    def ref_bwd(return_intermediates=False, dp_scale=None):
        return compute_ref_backward(
            q8, k8, v8, do8, do8, geom.scale, 1.0 / sq, 1.0 / sk, 1.0 / sv, scale_s, 1.0 / scale_s, FP8_E4M3, 1.0, 1.0 / scale_do, torch.bfloat16,
            left_bound=left, right_bound=right, diag_align=align, stats=lse.reshape(b, hq, s, 1), return_intermediates=return_intermediates,
            quantize_ds=True, dP_scale=dp_scale, quantize_grads=True, delta=delta[:, :, :s],
        )  # fmt: skip

    dq_ref, dk_ref, dv_ref, _dsink, dp_amax, _dq_amax, _dk_amax, _dv_amax, inter = ref_bwd(return_intermediates=True)
    dp_scale = get_fp8_scale_factor(dp_amax, FP8_E4M3)  # what the reference derived for itself at dP_scale=None
    ds_amax = inter["ds_scaled"].abs().max().item() / dp_scale  # the fp32 dS the row's amax_dP reduces
    return SimpleNamespace(
        geom=geom,
        b=b,
        s=s,
        hq=hq,
        hkv=hkv,
        causal=causal,
        q8=q8,
        k8=k8,
        v8=v8,
        do8=do8,
        do_bf16=do_bf16,
        lse=lse,
        delta=delta,
        allowed=allowed,
        scales=dict(q=sq, k=sk, v=sv, do=scale_do, s=scale_s, dp=dp_scale),
        refs=dict(dQ=dq_ref, dK=dk_ref, dV=dv_ref),
        ds_amax=ds_amax,
        dp_amax=dp_amax,
        ref_bwd=lambda sel: ref_bwd(return_intermediates=sel, dp_scale=dp_scale),
        deq=dict(q=q8.float() / sq, k=k8.float() / sk, dO=do8.float() / scale_do),
    )


@functools.lru_cache(maxsize=None)
def _compiled_sdpa_stage(b: int, s: int, hq: int, hkv: int, causal: bool):
    geom = GatedAttentionBlockGeometry(d_model=hq * _D, h_q=hq, h_kv=hkv, d_head=_D, rope_dim=64, is_causal=causal)
    st = _SdpaBwdFp8(geom, batch=b, seq_len=s, grad_dtype=torch.bfloat16, device=_cuda_device())
    st.check_support()
    st.compile()
    return st


def _run_stage(cell, st, *, o_dead=None, poison=float("nan")):
    """One execute of the stage over ``cell``'s operands: the twelve scalars as 4-byte-stride slot views of a scalar block
    (the block's layout), ``amax_dp`` a zeroed slot, the outputs poison-filled first; ``o_dead`` replaces ``do8`` as the
    row's dead ``o``.  Returns the outputs and the amax after a sync."""
    b, s, hq, hkv, sc = cell.b, cell.s, cell.hq, cell.hkv, cell.scales
    ws = torch.empty(max(st.scratch_workspace_bytes(), 1), dtype=torch.uint8, device="cuda")
    vals = dict(
        descale_q=1.0 / sc["q"],
        descale_k=1.0 / sc["k"],
        descale_v=1.0 / sc["v"],
        descale_s=1.0 / sc["s"],
        scale_s=sc["s"],
        descale_o=1.0,  # dead: read by nothing under the external delta
        descale_dO=1.0 / sc["do"],
        descale_dP=1.0 / sc["dp"],
        scale_dQ=1.0,
        scale_dK=1.0,
        scale_dV=1.0,
        scale_dP=sc["dp"],
    )
    block = torch.zeros(64, dtype=torch.float32, device="cuda")
    scalars = {}
    for i, name in enumerate(FP8_SCALARS):
        block[i] = vals[name]
        scalars[name] = block[i : i + 1]
    amax = block[14:15]
    dq = torch.full((b, s, hq, _D), poison, dtype=torch.bfloat16, device="cuda")
    dk = torch.full((b, s, hkv, _D), poison, dtype=torch.bfloat16, device="cuda")
    dv = torch.full((b, s, hkv, _D), poison, dtype=torch.bfloat16, device="cuda")
    st.execute(
        cell.q8,
        cell.k8,
        cell.v8,
        cell.do8 if o_dead is None else o_dead,
        cell.do8,
        cell.lse,
        dq,
        dk,
        dv,
        workspace=ws,
        stream=torch.cuda.current_stream().cuda_stream,
        delta=cell.delta,
        scalars=scalars,
        amax_dp=amax,
    )
    torch.cuda.synchronize()
    return SimpleNamespace(dq=dq, dk=dk, dv=dv, amax_dp=amax.clone())


# (B, S, H_q, H_kv, causal): the three S of the matrix at GQA 8/2 under both masks (992: a padded q AND kv tile), the MHA arm under
# each mask (no dK fold; dK quantized straight into the caller's buffer), and B = 2 at the bitwise cell's geometry.
_ROW_CELLS = [(1, s, 8, 2, c) for s in (256, 512, 992) for c in (True, False)] + [(1, 512, 8, 8, True), (1, 992, 8, 8, False), (2, 512, 8, 2, True)]


def _cell_id(c):
    return f"B{c[0]}-S{c[1]}-H{c[2]}x{c[3]}-{'causal' if c[4] else 'dense'}"


@requires_rubin
@pytest.mark.parametrize("b,s,hq,hkv,causal", _ROW_CELLS, ids=[_cell_id(c) for c in _ROW_CELLS])
def test_sdpa_bwd_fp8_stage_matches_the_row_reference(b, s, hq, hkv, causal):
    """The stage's bf16 dQ / dK / dV vs the row's reference fed the SAME delta, LSE, band and scales under the fp8 suite's
    recipe (atol 0.08 / rtol 0.2 with the midpoint-flip budget, flips proved from the reference's own intermediates);
    ``amax_dP`` within its tolerance of ``max |dS|``; every output finite (the poison never survives); the compiled stage
    carries ``external_delta`` and no ``delta`` scratch region."""
    from sdpa.fp8 import assert_close_fp8_grad

    cell = _row_cell(b, s, hq, hkv, causal)
    st = _compiled_sdpa_stage(b, s, hq, hkv, causal)
    assert st._impl.external_delta is True and "delta" not in [n for n, _s, _d in st._impl._scratch_plan()]
    run = _run_stage(cell, st)
    keys = dict(dQ=s, dK=s, dV=s)
    operands = dict(dQ=cell.deq["k"], dK=cell.deq["q"], dV=cell.deq["dO"])
    flip = dict(dQ=1.0 / cell.scales["dp"], dK=1.0 / cell.scales["dp"], dV=1.0 / cell.scales["s"])
    for name, got in (("dQ", run.dq), ("dK", run.dk), ("dV", run.dv)):
        assert torch.isfinite(got.float()).all(), f"{name}: non-finite output (poison survived, or NaN)"
        assert_close_fp8_grad(
            got.float(),
            cell.refs[name].float(),
            _FP8_GRAD_TOL["atol"],
            _FP8_GRAD_TOL["rtol"],
            tag=name,
            keys=keys[name],
            operand=operands[name],
            flip_unit=flip[name],
            intermediates=lambda selection: cell.ref_bwd(selection)[8],
            fp8_dtype=FP8_E4M3,
            out_dtype=torch.bfloat16,
        )
    a = run.amax_dp.item()
    assert math.isfinite(a) and a > 0.0, f"amax_dP was not written ({a})"
    msg = f"amax_dP {a:.6f} vs max|dS| {cell.ds_amax:.6f} (max|dP| {cell.dp_amax:.4f}; scale_dP {cell.scales['dp']})"
    print(msg)
    assert abs(a - cell.ds_amax) <= _AMAX_DS_TOL["atol"] + _AMAX_DS_TOL["rtol"] * cell.ds_amax, msg


@requires_rubin
def test_sdpa_bwd_fp8_stage_binds_the_dead_o_without_a_slot():
    """The row's ``o`` is REQUIRED by its ABI and read by nothing under the external delta, so the stage binds an EXISTING
    e4m3 operand there instead of carving a slot: ``do8`` (the block without an out-projection weight gradient), an
    ``og8``-like buffer (with one), or a NaN-filled buffer -- the gradients and ``amax_dP`` are BITWISE the same for all
    three."""
    cell = _row_cell(1, 512, 8, 2, True)
    st = _compiled_sdpa_stage(1, 512, 8, 2, True)
    base = _run_stage(cell, st)
    og8 = quant_e4m3(torch.randn(1, 512, 8, _D, device="cuda"), 1.0)
    alt = _run_stage(cell, st, o_dead=og8)
    nan8 = torch.full((1, 512, 8, _D), float("nan"), device="cuda").to(FP8_E4M3)
    assert torch.isnan(nan8.float()).all()
    poisoned = _run_stage(cell, st, o_dead=nan8)
    assert torch.isfinite(base.dq.float()).all() and torch.isfinite(base.dk.float()).all() and torch.isfinite(base.dv.float()).all()
    for other, what in ((alt, "an og8-like operand"), (poisoned, "a NaN-filled operand")):
        for name in ("dq", "dk", "dv"):
            x, y = getattr(base, name), getattr(other, name)
            n_diff = (x.view(torch.int16) != y.view(torch.int16)).sum().item()
            assert n_diff == 0, f"{name}: binding {what} as the dead o changed {n_diff} elements -- o is NOT dead on this chain"
        assert other.amax_dp.item() == base.amax_dp.item(), (what, other.amax_dp.item(), base.amax_dp.item())


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


@requires_cuda
def test_fp8_oracle_agrees_with_the_row_suite_at_one_cell():
    """Two bitwise self-checks of the oracle's SDPA node and one of its modes.  (a) The node (``_Fp8SdpaRow``) fed the row
    suite's cell (the same e4m3 payloads, bf16 dO, delta, LSE, band and scales) returns BITWISE the row's own reference for
    dQ / dK / dV, rebuilds the same ``do8`` and reports the same ``max |dS|``.  (b) The whole oracle on the block's test
    geometry: ``compute_ref_backward`` over the oracle's OWN ``q8 / k8 / v8 / do8``, LSE and delta returns bitwise its
    ``dq / dk / dv``.  (c) Seeded with those, the downstream ``dh / dw_*`` are bitwise the modelled run's; the unmodelled
    mode runs finite (its cosine against the modelled run is printed, not asserted)."""
    from sdpa.fp8_ref import compute_ref_backward
    from sdpa.helpers import get_fp8_scale_factor

    cell = _row_cell(1, 512, 8, 2, True)
    sc = cell.scales
    q = (cell.q8.double() / sc["q"]).requires_grad_(True)
    k = (cell.k8.double() / sc["k"]).requires_grad_(True)
    v = (cell.v8.double() / sc["v"]).requires_grad_(True)
    cfg = _Fp8RowCfg(
        scale=cell.geom.scale, is_causal=True, causal_bottom_right=False, window_left=-1, window_right=-1,
        scale_q=sc["q"], scale_k=sc["k"], scale_v=sc["v"], scale_s=sc["s"], scale_do=sc["do"], scale_dp=sc["dp"], modelled=True,
    )  # fmt: skip
    holder = {}
    o = _Fp8SdpaRow.apply(q, k, v, cell.q8, cell.k8, cell.v8, cell.allowed, cfg, cell.lse, cell.delta[:, :, : cell.s], None, holder)
    assert torch.allclose(
        o.to(torch.bfloat16).float(), fp64_attention(q.detach(), k.detach(), v.detach(), cell.allowed, cell.geom.scale)[0].to(torch.bfloat16).float()
    )
    dq, dk, dv = torch.autograd.grad(o, (q, k, v), cell.do_bf16.double())
    for name, got in (("dQ", dq), ("dK", dk), ("dV", dv)):
        ref = cell.refs[name].double().contiguous()
        assert torch.equal(
            got.contiguous(), ref
        ), f"{name}: the oracle's row node differs from the row suite's reference (max|diff| {(got - ref).abs().max().item():.3e})"
    assert torch.equal(holder["do8"].view(torch.uint8), cell.do8.view(torch.uint8)) and holder["amax_dp"] == cell.ds_amax
    # (b) the whole oracle on the block's test geometry, then the row's reference over its own payloads
    rg = RefGeometry(**_COMMON, is_causal=True)
    b, s = 1, 256
    inp16 = make_inputs(rg, batch=b, seq_len=s, dtype=torch.bfloat16)
    inp_q, desc = quantize_block_inputs(inp16)
    spec = QuantSpec(**desc, scale_q=1.0, scale_k=1.0, scale_v=4.0, scale_o=8.0)
    dy = (torch.randn(b, s, rg.d_model, generator=torch.Generator(device="cuda").manual_seed(1), device="cuda") * 0.05).to(torch.bfloat16)
    scale_s = 2.0**_SCALE_S_LOG2
    kw = dict(scale_dy=get_fp8_scale_factor(dy.float().abs().max().item(), FP8_E4M3), scale_do=1.0, scale_dqkvg=1.0, scale_s=scale_s)
    warm = gated_attention_block_fp8_bwd_reference(inp_q, rg, spec, dy, scale_dp=1.0, **kw)
    scale_dp = get_fp8_scale_factor(warm["amax_dp"], FP8_E4M3)  # calibrate dS the way a caller would (one warm-up at 1.0)
    kw.update(scale_do=get_fp8_scale_factor(warm["do8"].float().abs().max().item(), FP8_E4M3), scale_dp=scale_dp)
    res = gated_attention_block_fp8_bwd_reference(inp_q, rg, spec, dy, **kw)
    left, right, align = fp8_row_mask_args(rg.is_causal, rg.causal_bottom_right, rg.window_left)
    out = compute_ref_backward(
        res["q8"], res["k8"], res["v8"], res["do8"], res["do8"], rg.scale, 1.0 / spec.scale_q, 1.0 / spec.scale_k, 1.0 / spec.scale_v, scale_s, 1.0 / scale_s,
        FP8_E4M3, 1.0, 1.0 / kw["scale_do"], torch.bfloat16, left_bound=left, right_bound=right, diag_align=align, stats=res["lse"].float().reshape(b, rg.h_q, s, 1),
        quantize_ds=True, dP_scale=scale_dp, quantize_grads=False, delta=res["delta"],
    )  # fmt: skip
    for name, ref32 in (("dq", out[0]), ("dk", out[1]), ("dv", out[2])):
        assert torch.equal(
            res[name], ref32.to(torch.bfloat16).double().contiguous()
        ), f"{name}: the oracle's SDPA output differs from the row reference over its own payloads"
    assert torch.equal(res["dv"].reshape(-1), res["dv_band"].reshape(-1)) and res["amax_dp"] > 0 and torch.isfinite(res["dh"]).all()
    # (c) seeded with the modelled run's own bf16 dq / dk / dv: the downstream is bitwise; the unmodelled mode runs finite
    seeds = {n: res[n].to(torch.bfloat16) for n in ("dq", "dk", "dv")}
    seeded = gated_attention_block_fp8_bwd_reference(inp_q, rg, spec, dy, seeded=seeds, **kw)
    for name in ("dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm", "dq_pre", "dk_pre", "dg"):
        assert torch.equal(seeded[name], res[name]), f"{name}: the seeded oracle differs from the modelled one it was seeded from"
    assert seeded["amax_dp"] is None and torch.equal(seeded["dq"], res["dq"])
    plain = gated_attention_block_fp8_bwd_reference(inp_q, rg, spec, dy, modelled=False, **kw)

    def cos(a, c):
        a, c = a.double().flatten(), c.double().flatten()
        return (a @ c / (a.norm() * c.norm())).item()

    assert all(torch.isfinite(plain[n]).all() for n in ("dh", "dw_qkvg", "dw_o", "dq", "dk", "dv")) and plain["amax_dp"] > 0
    print(
        "unmodelled vs modelled cos: "
        + ", ".join(f"{n} {cos(plain[n], res[n]):.5f}" for n in ("dh", "dw_qkvg", "dw_o", "dq", "dk", "dv"))
        + f"; amax_dp {plain['amax_dp']:.5f} vs {res['amax_dp']:.5f} (scale_dP {scale_dp})"
    )
