# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MXFP8 block backward's SDPA STAGE and its oracle, each on its own, before the block assembles them.

* ``_SdpaBwdMxfp8`` (``api_bwd.py``): the stage over ``SdpaBwdDslSm107Mxfp8(external_delta=True)``.  Synthetic bf16-rounded q / k / v /
  dO quantized the MXFP8 SDPA backward suite's way (its ``_Quant``: rowwise + columnwise 1x32 e4m3 payloads with the F8_128x4 scale
  factors in the kernel's own layouts -- the block's quantizer kernel writes the same bytes), the exact fp64 forward over the
  dequantized rowwise operands -> the record's bf16 ``O`` and the exact fp32 natural-log LSE, the block's ``delta = rowsum(bf16 dO *
  bf16 O)`` in fp32 (zeros on the pad rows) -- the stage's bf16 dQ / dK / dV against the row's own reference
  (``sdpa.mxfp8_ref.compute_ref_backward`` fed the SAME delta, LSE, scales and band, dS quantized per 1x32 block both ways) under the
  row's recipe for its block-scaled chain (atol 0.08 / rtol 0.2 with ``assert_close_fp8_grad``'s midpoint-flip budget on all three
  gradients), causal and dense, GQA 8/2 and MHA, S in {256, 512, 992} (a padded q AND kv tile), B = 2 once.  The stage's scratch
  carries no ``delta`` region.  On this row the external delta IS the row's own pre-pass (the fp32 dot of the bf16 ports in
  ``dot_do_o``'s order): fed the delta a standalone ``SdpaBwdDslSm107Mxfp8(external_delta=False)`` plan wrote (read back from its
  carve), the stage's gradients are BITWISE that plan's -- a torch ``.sum(-1)`` of the same products is another fp32 order and moves a
  handful of elements by one bf16 ulp, counted and printed.  The row's dead ``o_f16`` /
  ``dO_f16`` ports are bound to EXISTING bf16 buffers: the gradients are bitwise whatever bf16 tensors stand there, a NaN-filled one
  included.  The scale-factor dict is checked by name, dtype, device, byte count, contiguity and alignment BEFORE the adapter (host).
* the stage inputs: the block's own ``quantize_mxfp8`` kernel (``axis="row"`` and ``"col"``) on the cell's bf16 source writes BITWISE the payloads and blobs
  this module feeds the stage (the row suite's ``_Quant``), so the stage tests exercise exactly the bytes the block hands the stage.
* the oracle ``gated_block_reference.gated_attention_block_mxfp8_bwd_reference``: its SDPA node, fed the row suite's data, returns
  BITWISE the row's own reference under ``fold="once"`` (dQ / dK under ``fold="kernel"`` too, dV then being the modelled fold of the
  per-Q-head partials); the whole oracle's once-rounded dQ / dK / dV are bitwise that reference over the oracle's own payloads; the
  seeded mode reproduces the downstream bitwise; the unmodelled mode runs.  The fold model (``mx_fold_dv_kernel_order``: per-Q-head
  bf16 partials summed in fp32 in head order, rounded once) is pinned against the KERNEL's own fold on Rubin -- the stage's dV is
  bitwise the model applied to the kernel's own bf16 ``dv_part`` region, and the stage's dK bitwise the once-rounded fp32 sum of its
  fp32 ``dk_part`` region.  ``mx_sf_ref_of`` inverts both SDPA scale-factor layouts bitwise against ``quantize_to_mxfp8``.
* ``sdpa.mxfp8_ref.compute_ref_backward(delta=)``: the appended hook's default path is bitwise the pre-hook behaviour, a given delta
  replaces the internal row-sum.

Tolerances are the row suite's own (never a new one): ``_GRAD_TOL`` of ``test/python/sdpa/frost/test_sdpa_bwd_mxfp8_sm107.py``, IMPORTED
(``_row_tol``) so a change of the row's recipe reaches this module instead of drifting past a copy.  Accept tests need the Rubin device
the block binds (``requires_rubin``); the declaration, dict-check and oracle cells run on any CUDA device.
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

from cudnn.gated_attention_block import GatedAttentionBlockGeometry, MxQuantSpec  # noqa: E402
from cudnn.gated_attention_block.api_bwd import _SdpaBwdMxfp8  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import (  # noqa: E402
    FP8_E4M3,
    RefGeometry,
    _key_padding_and_causal_mask,
    _MxRowCfg,
    _MxSdpaRow,
    fp64_attention,
    fp8_row_mask_args,
    gated_attention_block_mxfp8_bwd_reference,
    make_inputs,
    mx_fold_dv_kernel_order,
    mx_sf_ref_of,
    quantize_block_inputs_mxfp8,
)

# The REGISTERED marker of cutedsl/conftest.py (the skip is applied at collection).
requires_rubin = pytest.mark.requires_rubin
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_COMMON = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)
_D = 256


def _row_suite():
    """The MXFP8 SDPA backward suite (``test/python/sdpa/frost/test_sdpa_bwd_mxfp8_sm107.py``): its ``_Quant`` (the producer's rowwise +
    columnwise F8_128x4 quantization of a ``[B, S, H, D]`` tensor -- the kernel's SDPA layouts), its ``_GRAD_TOL`` (atol 0.08 / rtol 0.2,
    "tolerances, ONE table"), ``_to_bshd`` and ``_report`` -- imported, never restated, so the row's recipe cannot drift past this module."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # test/python: the sdpa.* modules
    for p in (root, os.path.join(root, "sdpa", "frost")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import test_sdpa_bwd_mxfp8_sm107 as row

    return row


def _row_tol() -> dict:
    return dict(_row_suite()._GRAD_TOL)


@pytest.fixture(autouse=True)
def _no_tf32():
    """The fp64 / fp32 reference arithmetic must not be silently TF32 (a hand-rolled fp32 reference is a TF32 reference otherwise)."""
    prev = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev


def _cuda_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


# ---------------------------------------------------------------------------
# The stage: declaration + the scale-factor dict (host)
# ---------------------------------------------------------------------------


@requires_cuda
def test_sdpa_bwd_mxfp8_stage_declares_the_row_contract():
    """The stage's declaration (no compile, any CUDA device): ``SdpaBwdDslSm107Mxfp8`` with ``external_delta=True``, e4m3 payloads, bf16
    gradients, no per-batch lengths, ``deterministic=False``, no amax, the geometry's mask; ``delta_shape`` is the adapter's ``(B, H_q,
    ceil128(S))``; the scratch carves NO ``delta`` region but does carve the block-scaled chain's payload / atom regions and, under GQA,
    an fp32 ``dk_part`` beside a bf16 ``dv_part``; the seven scale-factor names are the row's and their declared dims hold exactly the
    adapter's byte counts; the launch-census terms read off the adapter (one head chunk, one dQ launch per group member); the typed
    declines (a non-bf16 ``grad_dtype``, ``d_head != 256``, a non-Rubin device) fire before any adapter work; ``execute`` before
    ``compile`` is a ``RuntimeError``."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import _MXFP8_SF_ROLES

    geom = GatedAttentionBlockGeometry(**_COMMON)
    dev = _cuda_device()
    st = _SdpaBwdMxfp8(geom, batch=2, seq_len=500, grad_dtype=torch.bfloat16, device=dev)
    impl = st._build_impl()
    assert type(impl).__name__ == "SdpaBwdDslSm107Mxfp8"
    assert impl.external_delta is True and tuple(getattr(impl, "amax_requested", ())) == ()
    assert impl.dtype == FP8_E4M3 and impl.out_dtype == torch.bfloat16 and impl.seq_kv_lens_present is False and impl.deterministic is False
    assert impl.is_causal == geom.is_causal and impl.scale_softmax == pytest.approx(geom.scale) and impl.thd is False
    assert st.delta_shape == (2, 8, 512) == tuple(impl.external_delta_shape)
    plan = {name: (tuple(int(x) for x in shape), dt) for name, shape, dt in st._impl._scratch_shapes()}
    assert "delta" not in plan and st.scratch_workspace_bytes() == impl.scratch_workspace_bytes() > 0
    assert {"ds_ws", "ds_dq", "sf_ds_dk", "sf_ds_dq"} <= set(plan), sorted(plan)
    assert plan["dk_part"][1] == torch.float32 and plan["dv_part"][1] == torch.bfloat16, "GQA: fp32 dK partials (rounded once by the fold), bf16 dV partials"
    assert st.sf_roles() == tuple(_MXFP8_SF_ROLES) and len(st.sf_roles()) == 7
    shapes = st.sf_shapes()
    assert set(shapes) == set(st.sf_roles())
    for n in st.sf_roles():
        assert math.prod(shapes[n]) == impl._sf_expected_bytes(n), (n, shapes[n])
    assert shapes["sf_v"] == (2, 2, 512, 8) == shapes["sf_k"] and shapes["sf_k_T"] == (2, 2, 8, 512), "sf_v is the ROWWISE form the adapter asserts"
    assert st.head_chunks() == 1 and st.dq_launches_per_chunk() == geom.h_q // geom.h_kv == 4
    with pytest.raises(NotImplementedError, match="grad_dtype"):
        _SdpaBwdMxfp8(geom, batch=1, seq_len=256, grad_dtype=torch.float16, device=dev).check_support()
    with pytest.raises(NotImplementedError, match="256"):
        _SdpaBwdMxfp8(
            GatedAttentionBlockGeometry(d_model=512, h_q=4, h_kv=4, d_head=128, rope_dim=64), batch=1, seq_len=256, grad_dtype=torch.bfloat16, device=dev
        ).check_support()
    if tuple(torch.cuda.get_device_capability()) != (10, 7):
        with pytest.raises(NotImplementedError, match="Rubin"):
            st.check_support()
    fresh = _SdpaBwdMxfp8(geom, batch=1, seq_len=256, grad_dtype=torch.bfloat16, device=dev)
    with pytest.raises(RuntimeError, match="compile"):
        fresh.execute(*([None] * 9), workspace=None, stream=0, delta=None, q_T8=None, k_T8=None, do_T8=None, do16=None, sf={})


@requires_cuda
def test_sdpa_bwd_mxfp8_stage_sf_dict_is_checked_before_the_adapter():
    """``execute(sf=)`` over the declared adapter with a recording ``execute``: a missing name, an unexpected name, an int8 / CPU /
    short / non-contiguous blob and a list instead of a dict are typed ``ValueError``s naming the blob, and the adapter is never reached;
    the good call hands every blob through by identity, the delta / workspace / stream as given, and the thirteen io tensors as
    ``(B, H, S, D)`` transposes of the compact buffers (BSHD-physical strides, no copy)."""
    geom = GatedAttentionBlockGeometry(**_COMMON)
    st = _SdpaBwdMxfp8(geom, batch=1, seq_len=256, grad_dtype=torch.bfloat16, device=_cuda_device())
    st._impl = st._build_impl()
    calls = []
    st._impl.execute = lambda *args, **kw: calls.append((args, kw))
    b, s, hq, hkv = 1, 256, geom.h_q, geom.h_kv
    e8 = lambda h: torch.zeros(b, s, h, _D, dtype=FP8_E4M3, device="cuda")  # noqa: E731
    bf = lambda h: torch.zeros(b, s, h, _D, dtype=torch.bfloat16, device="cuda")  # noqa: E731
    io = (e8(hq), e8(hkv), e8(hkv), bf(hq), e8(hq), torch.zeros(b, hq, s, device="cuda"), bf(hq), bf(hkv), bf(hkv))
    extra = dict(q_T8=e8(hq), k_T8=e8(hkv), do_T8=e8(hq), do16=bf(hq))
    sf = {n: torch.zeros(shape, dtype=torch.uint8, device="cuda") for n, shape in st.sf_shapes().items()}
    kw = dict(
        workspace=torch.empty(16, dtype=torch.uint8, device="cuda"),
        stream=torch.cuda.current_stream().cuda_stream,
        delta=torch.zeros(b, hq, 256, device="cuda"),
    )
    bad = dict(sf)
    bad.pop("sf_do_T")
    with pytest.raises(ValueError, match="sf_do_T"):
        st.execute(*io, **extra, sf=bad, **kw)
    with pytest.raises(ValueError, match="sf_extra"):
        st.execute(*io, **extra, sf=dict(sf, sf_extra=sf["sf_q"]), **kw)
    for wrong, why in (
        (sf["sf_k"].view(torch.int8), "int8"),
        (torch.zeros(st.sf_shapes()["sf_k"], dtype=torch.uint8), "cpu"),
        (torch.zeros(16, dtype=torch.uint8, device="cuda"), "short"),
        (torch.zeros(1, 2, 512, 16, dtype=torch.uint8, device="cuda")[..., ::2], "strided"),
    ):
        with pytest.raises(ValueError, match="sf_k"):
            st.execute(*io, **extra, sf=dict(sf, sf_k=wrong), **kw)
    with pytest.raises(ValueError, match="sf must be a dict"):
        st.execute(*io, **extra, sf=list(sf.values()), **kw)
    assert not calls, "a refused scale-factor set must never reach the adapter"
    st.execute(*io, **extra, sf=sf, **kw)
    assert len(calls) == 1
    args, got = calls[0]
    assert all(got[n] is sf[n] for n in st.sf_roles())
    assert got["delta_tensor"] is kw["delta"] and got["workspace"] is kw["workspace"] and int(got["current_stream"]) == int(kw["stream"])
    assert len(args) == 9 and args[5] is io[5]
    for a, x in list(zip(args[:5] + args[6:], io[:5] + io[6:])) + [
        (got[k], v)
        for k, v in (("q_T_tensor", extra["q_T8"]), ("k_T_tensor", extra["k_T8"]), ("do_T_tensor", extra["do_T8"]), ("do_f16_tensor", extra["do16"]))
    ]:
        h = x.shape[2]
        assert a.shape == (b, h, s, _D) and a.data_ptr() == x.data_ptr() and a.stride() == (s * h * _D, _D, h * _D, 1), (a.shape, a.stride())


# ---------------------------------------------------------------------------
# The reference's delta hook and the scale-factor inverse (host)
# ---------------------------------------------------------------------------


def _mx_cell_tensors(b, hq, hkv, s, seed=0):
    """bf16-rounded unit-normal q / k / v / dO ``[B, S, H, D]`` on the CUDA device, the row suite's draw."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(h):
        return torch.randn(b, s, h, _D, generator=gen).to(torch.bfloat16).float().to("cuda")

    return draw(hq), draw(hkv), draw(hkv), draw(hq)


@requires_cuda
def test_mxfp8_ref_delta_hook_defaults_to_the_internal_dot():
    """``compute_ref_backward(delta=None)`` (the default, every existing caller) returns dQ / dK / dV BITWISE the pre-hook behaviour --
    spelled out as the explicit fp32 row-sum of the bf16 ports handed through the hook --, a delta of the wrong element count is refused,
    and a different delta moves dQ (the hook is live)."""
    import cudnn
    from sdpa.mxfp8_ref import compute_ref_backward

    row = _row_suite()
    b, hq, hkv, s = 1, 4, 2, 256
    q, k, v, do = _mx_cell_tensors(b, hq, hkv, s)
    qQ, qK, qV, qdO = row._Quant(q, s), row._Quant(k, s), row._Quant(v, s), row._Quant(do, s)
    o16 = torch.randn(b, hq, s, _D, device="cuda").to(torch.bfloat16)
    do16 = do.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    lse = (torch.randn(b, hq, s, device="cuda").abs() + 5.0).contiguous()
    args = (
        qQ.ref_d,
        qQ.ref_s,
        qK.ref_d,
        qK.ref_s,
        qV.ref_d,
        o16,
        do16,
        qdO.ref_d,
        qdO.ref_s,
        1.0 / math.sqrt(_D),
        qQ.sfref_d,
        qQ.sfref_s,
        qK.sfref_d,
        qK.sfref_s,
        qV.sfref_d,
        qdO.sfref_d,
        qdO.sfref_s,
    )
    kw = dict(torch_itype=FP8_E4M3, torch_otype=torch.bfloat16, right_bound=0, diag_align=cudnn.diagonal_alignment.TOP_LEFT, stats=lse, quantize_ds=True)
    default = compute_ref_backward(*args, **kw)
    delta = (o16.float() * do16.float()).sum(-1)
    explicit = compute_ref_backward(*args, **kw, delta=delta)
    assert all(torch.equal(x, y) for x, y in zip(default[:3], explicit[:3])), "the default path is not the explicit row-sum of the bf16 ports"
    assert all(
        torch.equal(x, y) for x, y in zip(default[:3], compute_ref_backward(*args, **kw, delta=delta.unsqueeze(-1))[:3])
    ), "[b, h_q, s_q, 1] is the same delta"
    with pytest.raises(ValueError, match="delta must hold"):
        compute_ref_backward(*args, **kw, delta=delta[:, :, : s // 2])
    moved = compute_ref_backward(*args, **kw, delta=delta + 1.0)
    assert not torch.equal(default[0], moved[0]), "a different delta must move dQ: the hook is dead"


@requires_cuda
def test_mx_sf_ref_of_inverts_both_sdpa_layouts():
    """``mx_sf_ref_of`` (the oracle's reading of a kernel scale-factor blob) returns BITWISE ``quantize_to_mxfp8``'s per-element
    ``sf_d_ref`` / ``sf_s_ref`` from the rowwise / columnwise F8_128x4 blobs of the same quantization, at an S that pads to a tile and at
    one that is a tile multiple, with B * H > 1; the row suite's ``_Quant`` blobs read back to its own ``sfref_d / sfref_s``."""
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    row = _row_suite()
    for b, h, s in ((1, 2, 256), (2, 3, 500)):
        x = torch.randn(b, h, s, _D, device="cuda")
        _fd, sd_ref, blob_d, _fs, ss_ref, blob_s = quantize_to_mxfp8(x, b, h, s, _D)
        assert torch.equal(mx_sf_ref_of(blob_d, "rowwise", b, h, s, _D), sd_ref)
        assert torch.equal(mx_sf_ref_of(blob_s, "columnwise", b, h, s, _D), ss_ref)
        q = row._Quant(x.permute(0, 2, 1, 3).to(torch.bfloat16).float().contiguous(), s)
        assert torch.equal(mx_sf_ref_of(q.sf_d, "rowwise", b, h, s, _D), q.sfref_d) and torch.equal(mx_sf_ref_of(q.sf_s, "columnwise", b, h, s, _D), q.sfref_s)
    with pytest.raises(ValueError, match="layout"):
        mx_sf_ref_of(blob_d, "diagonal", 2, 3, 500, _D)
    with pytest.raises(ValueError, match="bytes"):
        mx_sf_ref_of(blob_d[:-1], "rowwise", 2, 3, 500, _D)


# ---------------------------------------------------------------------------
# The stage on Rubin: the row's reference, the row's own pre-pass, the dead ports, the fold
# ---------------------------------------------------------------------------


def _row_cell(b: int, s: int, hq: int, hkv: int, causal: bool, *, seed: int = 0):
    """One SDPA-stage cell built the MXFP8 SDPA backward suite's way: bf16-rounded unit-normal q / k / v / dO quantized rowwise AND
    columnwise (``_Quant``: the kernel's SDPA-layout payloads and scale factors), the exact fp64 forward over the dequantized rowwise
    operands -> the record's bf16 ``O`` and the exact fp32 natural-log LSE, the block's ``delta = rowsum(bf16 dO * bf16 O)`` in fp32 with
    zeros on the pad rows, and the row's reference (``compute_ref_backward`` fed the SAME delta / LSE / band, dS 1x32 both ways)."""
    from sdpa.mxfp8_ref import compute_ref_backward

    row = _row_suite()
    geom = GatedAttentionBlockGeometry(d_model=hq * _D, h_q=hq, h_kv=hkv, d_head=_D, rope_dim=64, is_causal=causal)
    q, k, v, do = _mx_cell_tensors(b, hq, hkv, s, seed)
    do_bf16 = do.to(torch.bfloat16)  # [B, S, H, D]: the gate backward's buffer, the tensor the dO quantize reads
    qQ, qK, qV, qdO = row._Quant(q, s), row._Quant(k, s), row._Quant(v, s), row._Quant(do_bf16.float(), s)
    allowed = _key_padding_and_causal_mask(s, s, is_causal=causal, seq_lens=None, batch_index=0, q_lo=0, device="cuda", s_q_total=s)
    o64, lse64 = fp64_attention(
        qQ.deq_d().permute(0, 2, 1, 3).double(), qK.deq_d().permute(0, 2, 1, 3).double(), qV.deq_d().permute(0, 2, 1, 3).double(), allowed, geom.scale
    )
    o_bf16 = o64.to(torch.bfloat16).contiguous()  # [B, S, H, D]: the record's pre-gate O
    lse = lse64.float().contiguous()  # [B, H_q, S], natural log -- the forward's exact Stats
    s_pad = -(-s // 128) * 128
    delta = torch.zeros(b, hq, s_pad, dtype=torch.float32, device="cuda")
    delta[:, :, :s] = (do_bf16.float() * o_bf16.float()).sum(-1).permute(0, 2, 1)
    left, right, align = fp8_row_mask_args(causal, False, -1)
    o_ref, do_ref = o_bf16.permute(0, 2, 1, 3).contiguous(), do_bf16.permute(0, 2, 1, 3).contiguous()
    dq_ref, dk_ref, dv_ref, _dsink = compute_ref_backward(
        qQ.ref_d, qQ.ref_s, qK.ref_d, qK.ref_s, qV.ref_d, o_ref, do_ref, qdO.ref_d, qdO.ref_s, geom.scale,
        qQ.sfref_d, qQ.sfref_s, qK.sfref_d, qK.sfref_s, qV.sfref_d, qdO.sfref_d, qdO.sfref_s,
        torch_itype=FP8_E4M3, torch_otype=torch.bfloat16, left_bound=left, right_bound=right, diag_align=align, stats=lse, quantize_ds=True, delta=delta[:, :, :s],
    )  # fmt: skip
    return SimpleNamespace(
        geom=geom, b=b, s=s, hq=hq, hkv=hkv, causal=causal, quant=dict(q=qQ, k=qK, v=qV, dO=qdO), o_bf16=o_bf16, do_bf16=do_bf16, lse=lse, delta=delta,
        allowed=allowed, refs=dict(dQ=dq_ref, dK=dk_ref, dV=dv_ref), o_ref=o_ref, do_ref=do_ref, scale=geom.scale,
    )  # fmt: skip


def _sf_dict(cell) -> dict:
    """The seven blobs in the kernel's layouts: ``sf_v`` ROWWISE (the backward's dP operand), the ``_T`` ones columnwise."""
    q, k, v, do = cell.quant["q"], cell.quant["k"], cell.quant["v"], cell.quant["dO"]
    return dict(sf_q=q.sf_d, sf_q_T=q.sf_s, sf_k=k.sf_d, sf_k_T=k.sf_s, sf_v=v.sf_d, sf_do=do.sf_d, sf_do_T=do.sf_s)


@functools.lru_cache(maxsize=None)
def _compiled_sdpa_stage(b: int, s: int, hq: int, hkv: int, causal: bool):
    geom = GatedAttentionBlockGeometry(d_model=hq * _D, h_q=hq, h_kv=hkv, d_head=_D, rope_dim=64, is_causal=causal)
    st = _SdpaBwdMxfp8(geom, batch=b, seq_len=s, grad_dtype=torch.bfloat16, device=_cuda_device())
    st.check_support()
    st.compile()
    return st


def _run_stage(cell, st, *, o16=None, do16=None, poison=float("nan"), ws=None, delta=None):
    """One execute of the stage over ``cell``'s operands, the outputs poison-filled first; ``o16`` / ``do16`` replace the record's O and
    the bf16 dO as the row's dead ports; ``delta`` replaces the cell's torch row-sum.  Returns the outputs (and the workspace, for the
    region read-backs) after a sync."""
    b, s, hq, hkv = cell.b, cell.s, cell.hq, cell.hkv
    q, k, v, do = cell.quant["q"], cell.quant["k"], cell.quant["v"], cell.quant["dO"]
    ws = torch.empty(max(st.scratch_workspace_bytes(), 1), dtype=torch.uint8, device="cuda") if ws is None else ws
    dq = torch.full((b, s, hq, _D), poison, dtype=torch.bfloat16, device="cuda")
    dk = torch.full((b, s, hkv, _D), poison, dtype=torch.bfloat16, device="cuda")
    dv = torch.full((b, s, hkv, _D), poison, dtype=torch.bfloat16, device="cuda")
    st.execute(
        q.pay_d,
        k.pay_d,
        v.pay_d,
        cell.o_bf16 if o16 is None else o16,
        do.pay_d,
        cell.lse,
        dq,
        dk,
        dv,
        workspace=ws,
        stream=torch.cuda.current_stream().cuda_stream,
        delta=cell.delta if delta is None else delta,
        q_T8=q.pay_s,
        k_T8=k.pay_s,
        do_T8=do.pay_s,
        do16=cell.do_bf16 if do16 is None else do16,
        sf=_sf_dict(cell),
    )
    torch.cuda.synchronize()
    return SimpleNamespace(dq=dq, dk=dk, dv=dv, ws=ws)


# (B, S, H_q, H_kv, causal): the three S of the matrix at GQA 8/2 under both masks (992: a padded q AND kv tile), the MHA arm under each
# mask (no dK / dV fold: the GEMM and the kernel write the caller's buffers), and B = 2 at the bitwise cell's geometry.
_ROW_CELLS = [(1, s, 8, 2, c) for s in (256, 512, 992) for c in (True, False)] + [(1, 512, 8, 8, True), (1, 992, 8, 8, False), (2, 512, 8, 2, True)]


def _cell_id(c):
    return f"B{c[0]}-S{c[1]}-H{c[2]}x{c[3]}-{'causal' if c[4] else 'dense'}"


@requires_rubin
@pytest.mark.parametrize("b,s,hq,hkv,causal", _ROW_CELLS, ids=[_cell_id(c) for c in _ROW_CELLS])
def test_sdpa_bwd_mxfp8_stage_matches_the_row_reference(b, s, hq, hkv, causal):
    """The stage's bf16 dQ / dK / dV vs the row's reference fed the SAME delta, LSE, band and scale factors under the row's recipe for the
    block-scaled chain (atol 0.08 / rtol 0.2 with the midpoint-flip budget on all three: the kernel and the oracle each round dS to e4m3
    per 32-block from fp32 values ~1e-6 apart, the row's dS-flip class); every output finite (the poison never survives); the compiled
    stage carries ``external_delta`` and no ``delta`` scratch region; under GQA the dK partials are carved fp32 and the dV partials bf16."""
    from sdpa.fp8 import assert_close_fp8_grad

    row = _row_suite()
    tol = _row_tol()
    cell = _row_cell(b, s, hq, hkv, causal)
    st = _compiled_sdpa_stage(b, s, hq, hkv, causal)
    plan = {name: dt for name, _shape, dt in st._impl._scratch_plan()}
    assert st._impl.external_delta is True and "delta" not in plan
    if hq != hkv:
        assert plan["dk_part"] == torch.float32 and plan["dv_part"] == torch.bfloat16
    run = _run_stage(cell, st)
    for name, got in (("dQ", run.dq), ("dK", run.dk), ("dV", run.dv)):
        g = got.permute(0, 2, 1, 3).float()  # [B, H, S, D]
        assert torch.isfinite(g).all(), f"{name}: non-finite output (poison survived, or NaN)"
        row._report(
            f"{name} vs the row's e4m3-dS (1x32 both ways) reference, fp8 recipe, {_cell_id((b, s, hq, hkv, causal))}",
            g,
            cell.refs[name],
            tol["atol"],
            tol["rtol"],
        )
        assert_close_fp8_grad(g, cell.refs[name].float(), tol["atol"], tol["rtol"], tag=name, keys=s, budget=1e-5)


@requires_rubin
def test_sdpa_bwd_mxfp8_stage_external_delta_is_bitwise_the_rows_own_pre_pass():
    """On this row the external delta IS the row's own pre-pass (``dot_do_o`` over the bf16 ``o_f16`` / ``dO_f16`` ports): fed the delta a
    standalone ``SdpaBwdDslSm107Mxfp8(external_delta=False)`` plan WROTE (read back out of its own carve, as the row suite's pin reads it),
    the stage's dQ / dK / dV are ``torch.equal`` that plan's over the same payloads, scale factors, LSE and bf16 ports -- the row's bitwise
    pin, now at the block; the standalone plan carves the delta region the stage's plan lacks.  A torch ``.sum(-1)`` of the same bf16
    products is NOT ``dot_do_o``'s fp32 reduction order: its delta differs from the chain's in the last fp32 bits of some rows and moves a
    handful of dQ elements by one bf16 ulp -- counted and printed, never asserted away (the block's gate backward forms its delta in
    ``dot_do_o``'s order, which is the contract the bitwise claim rests on)."""
    from cudnn.sdpa.bwd import prepared_sm107
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Mxfp8
    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import R_DELTA
    from cudnn.sdpa.fwd.api_dsl import ws_align

    row = _row_suite()
    b, s, hq, hkv, causal = 1, 512, 8, 2, True
    cell = _row_cell(b, s, hq, hkv, causal)
    st = _compiled_sdpa_stage(b, s, hq, hkv, causal)
    q, k, v, do = cell.quant["q"], cell.quant["k"], cell.quant["v"], cell.quant["dO"]
    view = lambda t: t.permute(0, 2, 1, 3)  # noqa: E731
    outs = {n: torch.full((b, s, h, _D), float("nan"), dtype=torch.bfloat16, device="cuda") for n, h in (("dQ", hq), ("dK", hkv), ("dV", hkv))}
    stats = cell.lse.unsqueeze(-1).contiguous()
    own = SdpaBwdDslSm107Mxfp8(
        view(q.pay_d), view(k.pay_d), view(v.pay_d), row._to_bshd(cell.o_ref), view(do.pay_d), stats, view(outs["dQ"]), view(outs["dK"]), view(outs["dV"]),
        sample_q_T=view(q.pay_s), sample_k_T=view(k.pay_s), sample_do_T=view(do.pay_s), sample_do_f16=row._to_bshd(cell.do_ref),
        sample_sf_q=q.sf_d, sample_sf_q_T=q.sf_s, sample_sf_k=k.sf_d, sample_sf_k_T=k.sf_s, sample_sf_v=v.sf_d, sample_sf_do=do.sf_d, sample_sf_do_T=do.sf_s,
        scale_softmax=cell.scale, is_causal=causal, external_delta=False,
    )  # fmt: skip
    own.check_support()
    own.compile()
    ws = torch.empty(max(own.scratch_workspace_bytes(), 1), dtype=torch.uint8, device="cuda")
    own.execute(
        q_tensor=view(q.pay_d), k_tensor=view(k.pay_d), v_tensor=view(v.pay_d), o_tensor=row._to_bshd(cell.o_ref), do_tensor=view(do.pay_d), stats_tensor=stats,
        dq_tensor=view(outs["dQ"]), dk_tensor=view(outs["dK"]), dv_tensor=view(outs["dV"]),
        q_T_tensor=view(q.pay_s), k_T_tensor=view(k.pay_s), do_T_tensor=view(do.pay_s), do_f16_tensor=row._to_bshd(cell.do_ref),
        sf_q=q.sf_d, sf_q_T=q.sf_s, sf_k=k.sf_d, sf_k_T=k.sf_s, sf_v=v.sf_d, sf_do=do.sf_d, sf_do_T=do.sf_s, workspace=ws,
    )  # fmt: skip
    torch.cuda.synchronize()
    assert own.scratch_workspace_bytes() - st.scratch_workspace_bytes() == ws_align(b * hq * 512 * 4), "the standalone plan carries exactly the delta region"
    # the chain's own delta: region R_DELTA of the standalone plan's carve, read back after its execute (the row suite's `_mxfp8_chain_delta`)
    regions, _offset = prepared_sm107._regions(own, prepared_sm107._REGION_SLOTS_MXFP8)
    off, shape, _strides = regions[R_DELTA]
    assert shape == tuple(own.external_delta_shape) == (b, hq, 512) == st.delta_shape
    chain_delta = ws[off : off + 4 * math.prod(shape)].view(torch.float32).view(*shape).clone()
    assert torch.isfinite(chain_delta).all()
    ext = _run_stage(cell, st, delta=chain_delta)
    for name, mine, theirs in (("dQ", ext.dq, outs["dQ"]), ("dK", ext.dk, outs["dK"]), ("dV", ext.dv, outs["dV"])):
        n_diff = (mine.view(torch.int16) != theirs.view(torch.int16)).sum().item()
        assert n_diff == 0, f"{name}: the stage fed the row's own delta differs from the row's own pre-pass in {n_diff} of {mine.numel()} elements"
    # informational: the torch row-sum of the same bf16 products is another fp32 summation order
    d_delta = (chain_delta - cell.delta).abs()
    rows_off = int((d_delta.reshape(-1) > 0).sum())
    tor = _run_stage(cell, st)
    moved = {
        n: int((getattr(tor, k).view(torch.int16) != theirs.view(torch.int16)).sum())
        for n, k, theirs in (("dQ", "dq", outs["dQ"]), ("dK", "dk", outs["dK"]), ("dV", "dv", outs["dV"]))
    }
    print(
        f"\ntorch .sum(-1) delta vs the chain's dot_do_o: {rows_off} of {d_delta.numel()} rows differ (max |diff| {d_delta.max().item():.3e} at max |delta| "
        f"{chain_delta.abs().max().item():.3e}); gradients moved by that order: {moved} elements of {ext.dq.numel()} / {ext.dk.numel()} / {ext.dv.numel()}"
    )


@requires_rubin
def test_sdpa_bwd_mxfp8_stage_binds_the_dead_ports_without_a_slot():
    """The row's ``o_f16`` / ``dO_f16`` are REQUIRED by its ABI and read by nothing under the external delta, so the stage binds EXISTING
    bf16 buffers there instead of carving slots: the record's O and the gate backward's dO, random bf16 tensors, or NaN-filled ones -- the
    gradients are BITWISE the same for all three."""
    b, s, hq, hkv, causal = 1, 512, 8, 2, True
    cell = _row_cell(b, s, hq, hkv, causal)
    st = _compiled_sdpa_stage(b, s, hq, hkv, causal)
    base = _run_stage(cell, st)
    rnd = torch.randn(b, s, hq, _D, device="cuda").to(torch.bfloat16)
    alt = _run_stage(cell, st, o16=rnd, do16=rnd.clone())
    nan16 = torch.full((b, s, hq, _D), float("nan"), device="cuda", dtype=torch.bfloat16)
    poisoned = _run_stage(cell, st, o16=nan16, do16=nan16)
    assert all(torch.isfinite(getattr(base, n).float()).all() for n in ("dq", "dk", "dv"))
    for other, what in ((alt, "random bf16 tensors"), (poisoned, "NaN-filled tensors")):
        for name in ("dq", "dk", "dv"):
            x, y = getattr(base, name), getattr(other, name)
            n_diff = (x.view(torch.int16) != y.view(torch.int16)).sum().item()
            assert n_diff == 0, f"{name}: binding {what} as the dead o_f16 / dO_f16 changed {n_diff} elements -- the ports are NOT dead on this chain"


@requires_rubin
def test_sdpa_bwd_mxfp8_stage_inputs_are_the_block_quantizers_bytes():
    """The operands this module feeds the stage (the row suite's ``_Quant``: ``quantize_to_mxfp8`` re-laid into the kernel's two SDPA
    layouts) are BITWISE what the block's own quantizer kernel writes from the same bf16 source -- ``quantize_mxfp8`` with ``axis="row"``
    (the rowwise payload + the per-(b, h, 128-row tile) 1 KiB blob) and ``axis="col"`` (the columnwise payload + the D-plane-major blob) --
    so the stage tests exercise exactly the bytes the block hands the stage, and a layout drift in either producer shows here."""
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import compile_quantize_mxfp8, run_quantize_mxfp8, sf_bytes

    b, s, hq, hkv, causal = 1, 992, 8, 2, False  # a padded q / kv tile: the pad rows' SF bytes are written too (0x00, as the torch pad)
    cell = _row_cell(b, s, hq, hkv, causal)
    src = cell.do_bf16.reshape(b * s, hq, _D)  # the gate backward's bf16 dO, [T, H, D]
    stream = torch.cuda.current_stream().cuda_stream
    for axis, pay, blob in (("row", cell.quant["dO"].pay_d, cell.quant["dO"].sf_d), ("col", cell.quant["dO"].pay_s, cell.quant["dO"].sf_s)):
        r = compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=hq, d=_D, axis=axis)
        dst = torch.empty(b * s, hq, _D, dtype=FP8_E4M3, device="cuda")
        dst.view(torch.uint8).fill_(0xFF)
        sf = torch.full((sf_bytes(b, hq, s, _D),), 0xFF, dtype=torch.uint8, device="cuda")
        run_quantize_mxfp8(r, src, dst, sf, batch=b, seq_len=s, stream=stream)
        torch.cuda.synchronize()
        assert torch.equal(
            dst.view(torch.uint8).reshape(-1), pay.reshape(-1).view(torch.uint8)
        ), f"axis={axis}: the block quantizer's e4m3 codes differ from the stage test's"
        assert sf.numel() == blob.numel() and torch.equal(
            sf, blob.reshape(-1)
        ), f"axis={axis}: the block quantizer's scale-factor bytes differ from the stage test's (layout drift)"


@requires_rubin
def test_mxfp8_oracle_fold_model_is_the_kernels_order():
    """The oracle's fold model against the KERNEL's own fold, at a GQA cell of the stage: (dV) ``mx_fold_dv_kernel_order`` applied to the
    kernel's own bf16 ``dv_part`` region (per-Q-head partials) is BITWISE the stage's dV -- so the model sums in ``dkv_reduce``'s order
    (the group's q heads ascending, fp32 from zero) and rounds where the kernel rounds; (dK) the fixed-order fp32 sum of the kernel's own
    fp32 ``dk_part`` region rounded once is bitwise the stage's dK -- the once-rounded fold the oracle's dK is.  The partial regions are
    read back through the plan's own carve."""
    from cudnn.sdpa.bwd import prepared_sm107
    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import R_MX_DK_PART, R_MX_DV_PART

    b, s, hq, hkv, causal = 1, 512, 8, 2, True
    group = hq // hkv
    cell = _row_cell(b, s, hq, hkv, causal)
    st = _compiled_sdpa_stage(b, s, hq, hkv, causal)
    run = _run_stage(cell, st)
    regions, _offset = prepared_sm107._regions(st._impl, prepared_sm107._REGION_SLOTS_MXFP8)
    plan = {name: dt for name, _shape, dt in st._impl._scratch_plan()}

    def region(slot, name):
        off, shape, _strides = regions[slot]
        dt = plan[name]
        return run.ws[off : off + math.prod(shape) * dt.itemsize].view(dt).view(*shape).clone()

    dv_parts, dk_parts = region(R_MX_DV_PART, "dv_part"), region(R_MX_DK_PART, "dk_part")
    assert dv_parts.dtype == torch.bfloat16 and dk_parts.dtype == torch.float32 and dv_parts.shape == dk_parts.shape == (b, s, hq, _D)
    assert torch.isfinite(dv_parts.float()).all() and torch.isfinite(dk_parts).all()
    dv_model = mx_fold_dv_kernel_order(dv_parts, group)
    n_dv = (dv_model.view(torch.int16) != run.dv.view(torch.int16)).sum().item()
    assert n_dv == 0, f"the dV fold model differs from the kernel's dkv_reduce over its own bf16 partials in {n_dv} of {run.dv.numel()} elements"
    acc = torch.zeros(b, s, hkv, _D, dtype=torch.float32, device="cuda")
    for g in range(group):
        acc = acc + dk_parts[:, :, g::group]
    n_dk = (acc.to(torch.bfloat16).view(torch.int16) != run.dk.view(torch.int16)).sum().item()
    assert n_dk == 0, f"dK is not the once-rounded fixed-order fp32 fold of the kernel's own fp32 partials ({n_dk} of {run.dk.numel()} differ)"
    # informational: the kernel's dV (one bf16 rounding per group member) vs the once-rounded reference.  The region's partials are ALREADY
    # bf16, so re-summing them in fp32 and rounding once is dkv_reduce's own arithmetic -- equal to the kernel's dV by construction (the bitwise
    # pin above) and NOT a once-rounded fold: the per-member roundings sit in the partials and cannot be undone here.  The second number is a
    # consistency check of the fold model's arithmetic; the once-rounded fold's cost is the oracle-side print of
    # test_mxfp8_oracle_agrees_with_the_row_suite_at_one_cell.
    ref = cell.refs["dV"].permute(0, 2, 1, 3).float()
    refold = torch.zeros_like(acc)
    for g in range(group):
        refold = refold + dv_parts[:, :, g::group].float()
    rel = lambda x: ((x.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()  # noqa: E731
    print(
        f"\nGQA 8/2 causal S=512 dV vs the once-rounded reference: kernel (bf16 partials, one rounding per member) relative RMS {rel(run.dv):.3e}; "
        f"consistency check -- the same bf16 partials re-summed in fp32 and rounded once (dkv_reduce's arithmetic, equal by construction) "
        f"{rel(refold.to(torch.bfloat16)):.3e}"
    )


# ---------------------------------------------------------------------------
# The oracle (any CUDA device)
# ---------------------------------------------------------------------------


@requires_cuda
def test_mxfp8_oracle_agrees_with_the_row_suite_at_one_cell():
    """Three bitwise self-checks of the oracle's SDPA node and its modes.  (a) The node (``_MxSdpaRow``) fed the row suite's cell (the
    same e4m3 payloads and scale factors, bf16 dO, delta, LSE and band) returns BITWISE the row's own reference for dQ / dK / dV under
    ``fold="once"``; under ``fold="kernel"`` dQ / dK are still bitwise and dV is the fold model of the per-Q-head partials
    (``mx_fold_dv_kernel_order`` of ``dv_parts`` reproduces it; its distance to the once-rounded dV is printed).  (b) The whole oracle on
    the block's test geometry: ``compute_ref_backward`` over the oracle's OWN payloads, scale factors, LSE and delta returns bitwise its
    once-rounded ``dq / dk / dv``, and the kernel-fold run shares dQ / dK with it.  (c) Seeded with those, the downstream ``dh / dw_*``
    are bitwise the modelled run's; the unmodelled mode runs finite (its cosine against the modelled run is printed, not asserted)."""
    import cudnn
    from sdpa.helpers import get_fp8_scale_factor
    from sdpa.mxfp8_ref import compute_ref_backward

    row = _row_suite()
    b, s, hq, hkv = 1, 512, 8, 2
    group = hq // hkv
    q, k, v, do = _mx_cell_tensors(b, hq, hkv, s)
    do_bf16 = do.to(torch.bfloat16)
    qQ, qK, qV, qdO = row._Quant(q, s), row._Quant(k, s), row._Quant(v, s), row._Quant(do_bf16.float(), s)
    scale = 1.0 / math.sqrt(_D)
    allowed = _key_padding_and_causal_mask(s, s, is_causal=True, seq_lens=None, batch_index=0, q_lo=0, device="cuda", s_q_total=s)
    leaves = [qq.deq_d().permute(0, 2, 1, 3).double().requires_grad_(True) for qq in (qQ, qK, qV)]
    o64, lse64 = fp64_attention(*(x.detach() for x in leaves), allowed, scale)
    o_bf16, lse32 = o64.to(torch.bfloat16), lse64.float().contiguous()
    delta = (do_bf16.float() * o_bf16.float()).sum(-1).permute(0, 2, 1).contiguous()
    o_ref, do_ref = o_bf16.permute(0, 2, 1, 3).contiguous(), do_bf16.permute(0, 2, 1, 3).contiguous()
    refs = compute_ref_backward(
        qQ.ref_d, qQ.ref_s, qK.ref_d, qK.ref_s, qV.ref_d, o_ref, do_ref, qdO.ref_d, qdO.ref_s, scale,
        qQ.sfref_d, qQ.sfref_s, qK.sfref_d, qK.sfref_s, qV.sfref_d, qdO.sfref_d, qdO.sfref_s,
        torch_itype=FP8_E4M3, torch_otype=torch.bfloat16, right_bound=0, diag_align=cudnn.diagonal_alignment.TOP_LEFT, stats=lse32, quantize_ds=True, delta=delta,
    )  # fmt: skip
    points = dict(
        q=(qQ.ref_d, qQ.sfref_d.view(b, hq, s, _D)), q_T=(qQ.ref_s, qQ.sfref_s.view(b, hq, s, _D)),
        k=(qK.ref_d, qK.sfref_d.view(b, hkv, s, _D)), k_T=(qK.ref_s, qK.sfref_s.view(b, hkv, s, _D)), v=(qV.ref_d, qV.sfref_d.view(b, hkv, s, _D)),
        do8=qdO.pay_d, sf_do=qdO.sf_d, do_T8=qdO.pay_s, sf_do_T=qdO.sf_s,
    )  # fmt: skip
    for fold in ("once", "kernel"):
        cfg = _MxRowCfg(scale=scale, is_causal=True, causal_bottom_right=False, window_left=-1, window_right=-1, modelled=True, fold=fold)
        holder = {}
        o = _MxSdpaRow.apply(*leaves, allowed, cfg, lse32, delta, None, holder, o_bf16, do_bf16, points)
        assert torch.equal(o.to(torch.bfloat16), o_bf16)
        dq, dk, dv = torch.autograd.grad(o, leaves, do_bf16.double())
        for name, got, ref in (("dQ", dq, refs[0]), ("dK", dk, refs[1]), ("dV", dv, refs[2])):
            same = torch.equal(got.contiguous(), ref.permute(0, 2, 1, 3).double().contiguous())
            if fold == "once" or name != "dV":
                assert (
                    same
                ), f"{name} (fold={fold}): the oracle's row node differs from the row suite's reference (max|diff| {(got - ref.permute(0, 2, 1, 3).double()).abs().max().item():.3e})"
        assert torch.equal(holder["do8"].view(torch.uint8), qdO.pay_d.view(torch.uint8)) and torch.equal(holder["sf_do"].reshape(-1), qdO.sfref_d.reshape(-1))
        assert torch.equal(holder["do_T8"].view(torch.uint8), qdO.pay_s.view(torch.uint8)) and torch.equal(
            holder["sf_do_T"].reshape(-1), qdO.sfref_s.reshape(-1)
        )
        if fold == "kernel":
            assert holder["dv_parts"].shape == (b, s, hq, _D) and torch.equal(mx_fold_dv_kernel_order(holder["dv_parts"], group).double(), dv)
            assert torch.equal(holder["dv_once"], refs[2].permute(0, 2, 1, 3).double())
            rel = ((dv - holder["dv_once"]).pow(2).mean().sqrt() / holder["dv_once"].pow(2).mean().sqrt()).item()
            print(f"\nfold model: dV (per-member bf16 roundings) vs the once-rounded dV, relative RMS {rel:.3e} at GQA 8/2 S=512 causal")
            assert not torch.equal(dv, holder["dv_once"]), "the fold model must differ from the once-rounded fold under GQA"
    # (b) the whole oracle on the block's test geometry, then the row's reference over its own payloads
    rg = RefGeometry(**_COMMON, is_causal=True)
    bb, ss = 1, 256
    inp16 = make_inputs(rg, batch=bb, seq_len=ss, dtype=torch.bfloat16)
    inp_mx, desc = quantize_block_inputs_mxfp8(inp16)
    spec = MxQuantSpec(**desc, scale_o=8.0)
    dy = (torch.randn(bb, ss, rg.d_model, generator=torch.Generator(device="cuda").manual_seed(1), device="cuda") * 0.05).to(torch.bfloat16)
    kw = dict(scale_dy=get_fp8_scale_factor(dy.float().abs().max().item(), FP8_E4M3))
    res = gated_attention_block_mxfp8_bwd_reference(inp_mx, rg, spec, dy, **kw)
    once = gated_attention_block_mxfp8_bwd_reference(inp_mx, rg, spec, dy, fold="once", **kw)
    left, right, align = fp8_row_mask_args(rg.is_causal, rg.causal_bottom_right, rg.window_left)
    view = lambda t: t.permute(0, 2, 1, 3).contiguous()
    out = compute_ref_backward(
        view(once["q8"]), view(once["q_T8"]), view(once["k8"]), view(once["k_T8"]), view(once["v8"]), None, None, view(once["do8"]), view(once["do_T8"]), rg.scale,
        once["sf_q"], once["sf_q_T"], once["sf_k"], once["sf_k_T"], once["sf_v"], once["sf_do"], once["sf_do_T"],
        torch_itype=FP8_E4M3, torch_otype=torch.bfloat16, left_bound=left, right_bound=right, diag_align=align,
        stats=once["lse"].float().reshape(bb, rg.h_q, ss, 1), quantize_ds=True, delta=once["delta"],
    )  # fmt: skip
    for i, name in enumerate(("dq", "dk", "dv")):
        assert torch.equal(
            once[name], out[i].permute(0, 2, 1, 3).double().contiguous()
        ), f"{name}: the once-rounded oracle differs from the row reference over its own payloads"
    assert (
        torch.equal(res["dq"], once["dq"])
        and torch.equal(res["dk"], once["dk"])
        and torch.equal(res["dv_once"], once["dv"])
        and not torch.equal(res["dv"], once["dv"])
    )
    assert torch.equal(res["dv"].reshape(-1), res["dv_band"].reshape(-1)) and res["amax_dp"] is None and torch.isfinite(res["dh"]).all()
    assert res["dqkvg8"].shape == (bb * ss, rg.n_qkvg) and res["dqkvg_t8"].shape == (rg.n_qkvg, bb * ss) and res["dqkvg8"].dtype == FP8_E4M3
    # (c) seeded with the modelled run's own bf16 dq / dk / dv: the downstream is bitwise; the unmodelled mode runs finite
    seeds = {n: res[n].to(torch.bfloat16) for n in ("dq", "dk", "dv")}
    seeded = gated_attention_block_mxfp8_bwd_reference(inp_mx, rg, spec, dy, seeded=seeds, **kw)
    for name in ("dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm", "dq_pre", "dk_pre", "dg"):
        assert torch.equal(seeded[name], res[name]), f"{name}: the seeded oracle differs from the modelled one it was seeded from"
    assert seeded["amax_dp"] is None and torch.equal(seeded["dq"], res["dq"])
    plain = gated_attention_block_mxfp8_bwd_reference(inp_mx, rg, spec, dy, modelled=False, **kw)

    def cos(a, c):
        a, c = a.double().flatten(), c.double().flatten()
        return (a @ c / (a.norm() * c.norm())).item()

    assert all(torch.isfinite(plain[n]).all() for n in ("dh", "dw_qkvg", "dw_o", "dq", "dk", "dv"))
    print("unmodelled vs modelled cos: " + ", ".join(f"{n} {cos(plain[n], res[n]):.5f}" for n in ("dh", "dw_qkvg", "dw_o", "dq", "dk", "dv")))
