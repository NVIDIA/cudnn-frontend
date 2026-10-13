# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The PACKED (THD / varlen) forward of the gated attention block under MXFP8: ``GatedAttentionBlockFwd(thd=True, quant=MxQuantSpec)``.

The UNFUSED MXFP8 pipeline runs over a packed token matrix exactly as the bf16 / per-tensor FP8 ones do (``test_block_thd.py``):
every token-wise stage is the dense ``B=1, S=T`` block's launch, the SDPA runs the Rubin d256 MXFP8 row's THD arm -- and the three
``quantize_mxfp8`` stages run their PACKED arm, writing the row's per-sequence-tile-padded scale-factor layout: per head the tiles
of every sequence in ``cu_seqlens`` order, V's two D-planes adjacent inside a tile, the slot sized at the capacity
``H * ((T + 127 * B) // 128) * 1024`` bytes (``_sf_slot_bytes_packed`` -- the ONLY carve difference to the dense block), every
slack tile and pad byte ``0x00``, each tile's sequence resolved on device from the lengths tensor (no cap on ``B``).  The fp4 modes
that ride MXFP8 (an e2m1 ``W_qkvg``, an fp4 gated ``O``) ride here too; the fully fused quantized pipelines stay typed declines.

The reference is the dense MXFP8 fake-quant oracle run PER SEQUENCE on the codes and blobs the block reads
(``gated_block_reference.gated_attention_block_mxfp8_reference_packed``: ``h`` dequantized once through its ``T``-row blob, each
sequence's causal diagonal its own), at the dense MXFP8 suite's cosine floor (``test_block_mxfp8.MX_COS_FLOOR``, imported) -- the
kernels' unit-scale e4m3 P is not replicated by any oracle, which is why ``out`` is a cosine and the LSE (independent of P) a tight
absolute bound.  Verdicts are COLLECTED per sequence and asserted at the end, so a cross-sequence leak names its sequence.

Dense-vs-packed pins (no tolerance anywhere): ``B=1`` packed ``(512,)`` against the dense ``B=1, S=512`` MXFP8 training block over
the same bytes is BITWISE on ``out``, every record tensor, the e4m3 payload and the Q / K scale-factor slots, and V's slot is the
dense atoms in the plane-adjacent order (an exact permutation); uniform ``B=4`` packed ``(256,)*4`` against the dense ``B=4, S=256``
block is bitwise on every token-wise stage (slab, rstd, ``q8`` / ``k8`` / ``v8``), the three scale-factor slots an exact PERMUTATION
of the dense ones, and the SDPA-derived ``O`` / ``LSE`` / ``out`` per sequence at the packed bf16 suite's bounds.  Degenerate
packings are first-class cells (an empty sequence at the front, in the middle and trailing, a 5-token sequence, lengths that are not
tile multiples), the scale-factor slots are checked byte for byte under an E8M0-NaN (``0xFF``) workspace poison, and the training
record is bitwise the packed inference block's.

Accept tests are ``requires_rubin`` (the block targets SM107 only); the static pins (the declaration, the carve, the stage contract)
run on any device; every reject is host-side under ``torch.cuda.set_sync_debug_mode("error")``.
"""

import dataclasses
import gc
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

from cudnn.frost.tile_dsl.constants import SCHED_NATURAL  # noqa: E402
from cudnn.gated_attention_block import Fp4Format, GatedAttentionBlockFwd, GatedAttentionBlockGeometry, MxQuantSpec, saved_slab_views  # noqa: E402
from cudnn.gated_attention_block.api import _QuantizeMxfp8, _sf_slot_bytes, _sf_slot_bytes_packed, _sf_slots, _view  # noqa: E402
from cudnn.gated_attention_block.kernels.quantize_mxfp8 import n_sf_tiles_packed_cap, sf_bytes_packed, sf_tile_bytes  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_block_mxfp8 as mx_suite  # noqa: E402  -- the dense MXFP8 suite's constants and helpers, by module name
import test_block_thd as thd_suite  # noqa: E402  -- the packed bf16 / FP8 suite's helpers, by module name
from gated_block_reference import (  # noqa: E402
    RefGeometry,
    compare_packed,
    gated_attention_block_mxfp8_reference_packed,
    make_inputs,
    make_packed_inputs,
    mx_fake_quant_rowwise,
    mx_fake_quant_v_columnwise,
    mxfp8_calibrated_scale_o,
    mxfp8_calibrated_scale_o_packed,
    quantize_block_inputs_mxfp8,
    sequence_slices,
    unpack_e2m1,
)

E4M3 = torch.float8_e4m3fn
_FP4 = getattr(torch, "float4_e2m1fn_x2", None)
requires_rubin = pytest.mark.requires_rubin  # the REGISTERED marker of cutedsl/conftest.py
requires_cuda = thd_suite.requires_cuda
requires_fp4 = pytest.mark.skipif(_FP4 is None, reason=f"torch {torch.__version__} has no float4_e2m1fn_x2")

_COMMON, _LENS, _SENTINEL = thd_suite._COMMON, thd_suite._LENS, thd_suite._SENTINEL
_thd_kw, _lens, _form, _rows, _cos, _no_device_sync = (
    thd_suite._thd_kw,
    thd_suite._lens,
    thd_suite._form,
    thd_suite._rows,
    thd_suite._cos,
    thd_suite._no_device_sync,
)
MX_COS_FLOOR = mx_suite.MX_COS_FLOOR  # the dense MXFP8 oracle cells' cosine floor -- imported, never re-spelled
_MX_STAGES = mx_suite._MX_STAGES
_FP4_O_STAGES = _MX_STAGES[:-2] + ["quantize_fp4_o", "out_proj"]  # test_block_fp4.py's unfused fp4-O stage list
# The packed bf16 suite's bounds on the SDPA-derived tensors against the DENSE block (test_block_thd.py A12 /
# _check_forward_per_sequence): pre-gate O and LSE at atol 2e-2, out at cos > 0.999.  Imported by value, unchanged.
_O_LSE_ATOL = 2e-2
_OUT_VS_DENSE_COS = 0.999
_LSE_FP64_ATOL = 1e-4  # test_block_training_forward.py::test_quantized_saved_set_matches_the_oracle: the exact-Stats bound


# ---------------------------------------------------------------------------
# Building a packed MXFP8 block
# ---------------------------------------------------------------------------


def _quantized_packed_inputs(geom_kw, lens, *, max_seq_len=None, seed=0, cu_base=0, w_qkvg_fp4=False, o_fp4=None):
    """``make_packed_inputs`` -> the MXFP8 codes + PADDED F8_128x4 blobs over the ``T`` packed rows (``quantize_block_inputs_mxfp8``,
    token-wise: exactly the dense ``B=1, S=T`` block's inputs).  Returns ``(inp16, kern, orac, desc, meta)``: ``kern`` is what the block
    gets, ``orac`` the same dict for the oracle -- under ``w_qkvg_fp4`` with the e2m1 weight's VALUES as e4m3 SHADOW codes
    (``test_block_fp4._fp4w_inputs``: every E2M1 value is an e4m3 value exactly, so the oracle dequantizes the bytes the GEMM reads)."""
    inp16, meta = make_packed_inputs(RefGeometry(**geom_kw), lens, max_seq_len=max_seq_len, seed=seed, cu_base=cu_base)
    kern, desc = quantize_block_inputs_mxfp8(inp16, o_fp4=o_fp4, w_qkvg_fp4=w_qkvg_fp4)
    orac = kern
    if w_qkvg_fp4:
        orac = dict(kern, w_qkvg=unpack_e2m1(kern["w_qkvg"].view(torch.uint8)).to(E4M3))
    return inp16, kern, orac, desc, meta


def _declare_mx_thd(
    geom_kw, lens, *, cu=False, max_seq_len=None, seed=0, cu_base=0, training=False, return_lse=False, w_qkvg_fp4=False, o_fp4=None, scale_o=None, sentinel=True
):
    """A DECLARED (not compiled) packed MXFP8 block with its inputs, spec, output buffer and packing metadata.  ``scale_o`` defaults to
    the per-sequence oracle calibration (``mxfp8_calibrated_scale_o_packed``; 1.0 under ``o_fp4``, the block-scaled O has none).  The
    constructor runs under the sync-debug guard: a declaration reads no device."""
    g, rg = GatedAttentionBlockGeometry(**geom_kw), RefGeometry(**geom_kw)
    inp16, kern, orac, desc, meta = _quantized_packed_inputs(
        geom_kw, lens, max_seq_len=max_seq_len, seed=seed, cu_base=cu_base, w_qkvg_fp4=w_qkvg_fp4, o_fp4=o_fp4
    )
    if scale_o is None:
        scale_o = 1.0 if o_fp4 is not None else mxfp8_calibrated_scale_o_packed(orac, rg, meta["lens"])
    spec_kw = dict(desc, scale_o=float(scale_o))
    if w_qkvg_fp4:
        spec_kw["w_qkvg_dtype"] = _FP4
    if o_fp4 is not None:
        spec_kw["o_fp4"] = o_fp4
    spec = MxQuantSpec(**spec_kw)
    t = meta["t"]
    out = (
        torch.full((1, t, g.d_model), _SENTINEL, device="cuda", dtype=torch.bfloat16)
        if sentinel
        else torch.empty(1, t, g.d_model, device="cuda", dtype=torch.bfloat16)
    )
    kw = dict(quant=spec, sample_h_sf=kern["h_sf"], sample_w_qkvg_sf=kern["w_qkvg_sf"], **_thd_kw(meta, cu=cu))
    if o_fp4 is not None:
        kw["sample_w_o_sf"] = kern["w_o_sf"]
    if training:
        kw["save_for_backward"] = True
    if return_lse:
        kw["return_lse"] = True
    with _no_device_sync():
        blk = GatedAttentionBlockFwd(kern["h"], kern["w_qkvg"], kern["w_q_norm"], kern["w_k_norm"], kern["cos"], kern["sin"], kern["w_o"], out, g, **kw)
    return SimpleNamespace(
        blk=blk, g=g, rg=rg, inp16=inp16, kern=kern, orac=orac, desc=desc, spec=spec, out=out, meta=meta, seq_lens=_lens(meta, cu), form=_form(cu),
        cu=cu, o_fp4=o_fp4, w_qkvg_fp4=w_qkvg_fp4, geom_kw=geom_kw, ws=None, saved=None, refs=None,
    )  # fmt: skip


def _execute_mx(r, ws, *, saved=None, lse=None, out=None, blk=None, seq_lens=None):
    kern = r.kern
    extra = dict(w_o_sf=kern["w_o_sf"]) if r.o_fp4 is not None else {}
    (r.blk if blk is None else blk).execute(
        kern["h"],
        kern["w_qkvg"],
        kern["w_q_norm"],
        kern["w_k_norm"],
        kern["cos"],
        kern["sin"],
        kern["w_o"],
        r.out if out is None else out,
        ws,
        seq_lens=r.seq_lens if seq_lens is None else seq_lens,
        lse=lse,
        saved=saved,
        h_sf=kern["h_sf"],
        w_qkvg_sf=kern["w_qkvg_sf"],
        **extra,
    )


def _refs_of(r):
    return gated_attention_block_mxfp8_reference_packed(r.orac, r.rg, r.meta["lens"], descale_w_o=r.spec.descale_w_o, scale_o=r.spec.scale_o, o_fp4=r.o_fp4)


def _run_mx_thd(geom_kw, lens, *, training=False, refs=True, ws_fill=None, **kw):
    """Declare, compile and run a packed MXFP8 forward (inference, or training into a sentinel-filled record); ``ws_fill`` poisons the
    whole workspace with that byte before the launch.  Returns the declaration namespace plus ``ws, saved, refs``."""
    r = _declare_mx_thd(geom_kw, lens, training=training, **kw)
    if training:
        r.saved = thd_suite._alloc_packed_saved(r.g, r.kern, r.meta, seq_lens=r.seq_lens, form=r.form, sentinel=_SENTINEL, act_dtype=torch.bfloat16)
    r.blk.check_support()
    r.blk.compile()
    r.ws = torch.empty(r.blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    if ws_fill is not None:
        r.ws.fill_(ws_fill)
    _execute_mx(r, r.ws, saved=r.saved)
    torch.cuda.synchronize()
    if refs:
        r.refs = _refs_of(r)
    return r


def _sf_slots_of(r, ws=None):
    """The block's three scale-factor slots (flat uint8 views of its workspace), as the block hands them to the SDPA."""
    return r.blk._sf_views(r.ws if ws is None else ws, r.blk._layout())


def _payload_of(r, ws=None):
    """The compact e4m3 ``q8`` / ``k8`` / ``v8`` the quantize stages wrote, as uint8 views."""
    g, t, lay = r.g, r.meta["t"], r.blk._layout()
    ws = r.ws if ws is None else ws
    return tuple(_view(ws, off, (t, h, g.d_head), E4M3).view(torch.uint8) for off, h in ((lay.q8, g.h_q), (lay.k8, g.h_kv), (lay.v8, g.h_kv)))


def _assert_fully_written(r):
    out = _rows(r.out)
    assert torch.isfinite(out.float()).all(), "non-finite output rows"
    assert not (out == _SENTINEL).any(), f"{int((out == _SENTINEL).sum())} output cells were never written"
    if r.saved is not None:
        assert not (r.saved.o == _SENTINEL).any() and not (r.saved.lse == _SENTINEL).any(), "record rows were never written"


def _check_mx_per_sequence(r, *, floor=MX_COS_FLOOR):
    """``out`` per sequence against the per-sequence MXFP8 fake-quant oracle at the dense suite's cosine floor; failures collected."""
    out = _rows(r.out)

    def check(i, lo, hi, ref):
        c = _cos(out[lo:hi], ref[0])
        rel = ((out[lo:hi].float() - ref[0].float()).abs().max() / ref[0].float().abs().max().clamp_min(1e-30)).item()
        print(f"seq {i} [{lo}:{hi}]: cos={c:.6f} max_rel={rel:.3e}")
        assert c > floor, f"mxfp8 out cos {c:.6f}"

    failures = compare_packed(r.refs, r.meta["lens"], check)
    assert not failures, "\n".join(failures)


def _dense_mx_training(geom_kw, batch, seq_len):
    """The dense MXFP8 TRAINING block of ``test_block_training_forward.py`` (its calibrated spec, its record)."""
    from test_block_training_forward import _run_training_quant

    return _run_training_quant(geom_kw, batch, seq_len, "mxfp8")


# ---------------------------------------------------------------------------
# Static pins -- the declaration, the carve, the stage contract (any device)
# ---------------------------------------------------------------------------


def test_sf_slot_bytes_packed_is_the_quantizers_capacity_count():
    """``_sf_slot_bytes_packed`` IS ``kernels/quantize_mxfp8.sf_bytes_packed`` (one source): ``H * n_cap * 1024`` at d=256 with
    ``n_cap = (T + 127 * B) // 128`` -- at ``(300, 128, 200)`` 7 tiles per head against the dense ``(1, T)`` count's 5 (short by
    ``B - 1``); at ``B = 1`` the two counts coincide; ``_sf_slots`` switches on the appended ``thd_num_sequences``."""
    g = GatedAttentionBlockGeometry(**_COMMON)
    t, b = sum(_LENS), len(_LENS)
    assert n_sf_tiles_packed_cap(t, b) == (t + 127 * b) // 128 == 7 and -(-t // 128) == 5
    for h in (g.h_q, g.h_kv):
        assert _sf_slot_bytes_packed(b, h, t, g.d_head) == sf_bytes_packed(b, h, t, g.d_head) == h * 7 * sf_tile_bytes(g.d_head) == h * 7 * 1024
        assert _sf_slot_bytes_packed(b, h, t, g.d_head) - _sf_slot_bytes(1, h, t, g.d_head) == h * (b - 1) * 1024
        assert _sf_slot_bytes_packed(1, h, 512, g.d_head) == _sf_slot_bytes(1, h, 512, g.d_head)
    assert _sf_slots(g, 1, t) == [
        ("sf_q", _sf_slot_bytes(1, g.h_q, t, 256)),
        ("sf_k", _sf_slot_bytes(1, g.h_kv, t, 256)),
        ("sf_v", _sf_slot_bytes(1, g.h_kv, t, 256)),
    ]
    assert _sf_slots(g, 1, t, b) == [("sf_q", g.h_q * 7 * 1024), ("sf_k", g.h_kv * 7 * 1024), ("sf_v", g.h_kv * 7 * 1024)]


def test_quantize_mxfp8_stage_packed_contract_is_typed_both_ways():
    """The stage's packed fields: ``packed=True`` needs ``num_sequences >= 1`` and the SDPA layouts only (no canonical blob, no
    transposed store); a dense stage refuses ``num_sequences`` / ``cu_seqlens``; ``sf_bytes()`` (and the dual arm's second blob) is
    the packed capacity count under ``packed`` and the dense count otherwise.  No device."""
    g = GatedAttentionBlockGeometry(**_COMMON)
    t, b = sum(_LENS), len(_LENS)
    mk = lambda **kw: _QuantizeMxfp8(g, batch=1, seq_len=t, dtype_in=torch.bfloat16, heads=g.h_q, axis="row", name="q", **kw)  # noqa: E731
    with pytest.raises(ValueError, match="num_sequences >= 1"):
        mk(packed=True).check_support()
    with pytest.raises(ValueError, match="packed=True"):
        mk(packed=False, num_sequences=b).check_support()
    with pytest.raises(ValueError, match="packed=True"):
        mk(packed=False, cu_seqlens=True).check_support()
    with pytest.raises(ValueError, match="sdpa"):
        mk(packed=True, num_sequences=b, sf_layout="gemm").check_support()
    st = mk(packed=True, num_sequences=b, cu_seqlens=True)
    st.check_support()
    assert st.packed and st.num_sequences == b and st.cu_seqlens and st.rows() == t
    assert st.sf_bytes() == _sf_slot_bytes_packed(b, g.h_q, t, g.d_head) == sf_bytes_packed(b, g.h_q, t, g.d_head)
    assert mk().sf_bytes() == _sf_slot_bytes(1, g.h_q, t, g.d_head) != st.sf_bytes()
    dual = mk(packed=True, num_sequences=b, dual=True)
    dual.check_support()
    assert dual.sf_bytes() == dual.sf_bytes_second() == st.sf_bytes()


@requires_cuda
@pytest.mark.parametrize("cu", [False, True], ids=["lengths", "prefix"])
def test_thd_mxfp8_declaration_is_served_and_builds_the_packed_stages(cu):
    """A packed MXFP8 block DECLARES: the frozen nine-stage list, the three quantize stages on the PACKED arm (the packing's B, the
    lengths' form, ``sf_bytes()`` the capacity count), the SDPA stage packed on the MXFP8 row (``thd``, both totals, NATURAL, the
    block-scale family), the dense knobs untouched.  Host-side, no launch."""
    r = _declare_mx_thd(_COMMON, _LENS, cu=cu, sentinel=False)
    blk, g, meta, t = r.blk, r.g, r.meta, r.meta["t"]
    assert (blk.batch, blk.seq_len) == (1, t) and blk.thd and blk.mxfp8 and blk.cu_seqlens is cu and not blk.fuse_gate and not blk.fuse_norm_rope
    assert [s.name for s in blk._stages] == _MX_STAGES
    for st, h in ((blk._quant_q, g.h_q), (blk._quant_k, g.h_kv), (blk._quant_v, g.h_kv)):
        assert st.packed and st.num_sequences == meta["b"] and st.cu_seqlens is cu and (st.batch, st.seq_len) == (1, t) and st.sf_layout == "sdpa"
        assert st.sf_bytes() == _sf_slot_bytes_packed(meta["b"], h, t, g.d_head) == h * n_sf_tiles_packed_cap(t, meta["b"]) * 1024
    sd = blk._sdpa
    assert sd.thd and sd.mxfp8 and sd.fp8 and not sd.pertensor and sd.token_stride == 0 and sd.num_sequences == meta["b"] and sd.cu_seqlens is cu
    impl = sd._build_impl()
    assert impl.thd and impl.max_total_seq_len_q == t and impl.max_total_seq_len_kv == t and int(impl.sched_policy) == SCHED_NATURAL
    assert impl.cu_seq_q_lens is cu and impl.cu_seq_kv_lens is cu
    lay = blk._layout()
    assert lay.sf_q >= 0 and lay.sf_k >= 0 and lay.sf_v >= 0 and lay.sf_k - lay.sf_q >= blk._quant_q.sf_bytes()


@requires_cuda
@pytest.mark.parametrize("training", [False, True], ids=["inference", "training"])
def test_thd_mxfp8_workspace_carve_is_the_dense_b1_carve_plus_the_packed_sf_slots(training):
    """The packed MXFP8 block's carve IS the dense ``B=1, S=T`` MXFP8 block's, field by field, EXCEPT the three scale-factor slots,
    which are the packed capacity count (``sf_k`` / ``sf_v`` / the end offsets shift by exactly the capacity surplus); the stage lists
    are identical.  Declared blocks, any CUDA device."""
    r = _declare_mx_thd(_COMMON, _LENS, training=training, sentinel=False)
    g, t, b = r.g, r.meta["t"], r.meta["b"]
    geom, args, hsf, wsf = mx_suite._decl_tensors(_COMMON, batch=1, seq_len=t)
    kw = dict(quant=r.spec, sample_h_sf=hsf, sample_w_qkvg_sf=wsf)
    if training:
        kw["save_for_backward"] = True
    with _no_device_sync():
        dense = GatedAttentionBlockFwd(*args, geom, **kw)
    lp, ld = dataclasses.asdict(r.blk._layout()), dataclasses.asdict(dense._layout())
    assert [st.name for st in r.blk._stages] == [st.name for st in dense._stages] == _MX_STAGES
    surplus = {h: _sf_slot_bytes_packed(b, h, t, g.d_head) - _sf_slot_bytes(1, h, t, g.d_head) for h in (g.h_q, g.h_kv)}
    assert surplus[g.h_q] == g.h_q * (b - 1) * 1024 > 0 and surplus[g.h_kv] == g.h_kv * (b - 1) * 1024 > 0
    shifted = {
        "sf_q": 0,
        "sf_k": surplus[g.h_q],
        "sf_v": surplus[g.h_q] + surplus[g.h_kv],
        "engine_scratch": surplus[g.h_q] + 2 * surplus[g.h_kv],
        "total_bytes": surplus[g.h_q] + 2 * surplus[g.h_kv],
    }
    for k, vd in ld.items():
        vp = lp[k]
        if k in shifted:
            assert vp == vd + shifted[k], f"{k}: packed {vp} vs dense {vd} (+{shifted[k]} expected)"
        else:
            assert vp == vd, f"{k}: packed {vp} != dense {vd} -- the SF slots are the only carve difference"
    assert ld["total_bytes"] - ld["sf_v"] == _sf_slot_bytes(1, g.h_kv, t, g.d_head) and lp["total_bytes"] - lp["sf_v"] == _sf_slot_bytes_packed(
        b, g.h_kv, t, g.d_head
    )


@requires_cuda
@requires_fp4
@pytest.mark.parametrize("mode", ["w_qkvg_fp4", "o_nvfp4"])
def test_thd_mxfp8_fp4_modes_declare_under_thd(mode):
    """The fp4 modes ride the packed MXFP8 pipeline at declaration: an e2m1 ``W_qkvg`` keeps the nine stages with stage (1) on the mixed
    row; an fp4 ``O`` swaps the per-tensor tail for ``quantize_fp4_o`` + the block-scale out projection.  Both keep the three packed
    quantize stages."""
    r = _declare_mx_thd(_COMMON, _LENS, sentinel=False, w_qkvg_fp4=(mode == "w_qkvg_fp4"), o_fp4=Fp4Format.NVFP4 if mode == "o_nvfp4" else None)
    blk = r.blk
    assert blk.thd and blk.mxfp8 and all(st.packed for st in (blk._quant_q, blk._quant_k, blk._quant_v))
    if mode == "w_qkvg_fp4":
        assert [s.name for s in blk._stages] == _MX_STAGES and blk._proj.w_dtype == _FP4 and blk.o_fp4 is None
    else:
        assert [s.name for s in blk._stages] == _FP4_O_STAGES and blk.o_fp4 is Fp4Format.NVFP4 and r.spec.scale_o == 1.0


@requires_cuda
def test_thd_mxfp8_fully_fused_pipeline_stays_declined_naming_the_fork():
    """The FULLY FUSED MXFP8 pipeline under THD is still a typed decline, and its text now names BOTH reasons: the gated SDPA
    specialization has no THD arm, and the fused projection fork decodes a packed tile's sequence once per 128-row GEMM tile."""
    args, kw = thd_suite._decl_fp8_thd_args(mxfp8=True)
    with _no_device_sync(), pytest.raises(NotImplementedError, match="fully fused") as ei:
        GatedAttentionBlockFwd(*args, fuse_norm_rope=True, fuse_gate=True, **kw)
    msg = str(ei.value)
    assert "dense-only" in msg and "UNFUSED" in msg and "GEMM tile" in msg and "straddle" in msg
    # fuse_gate alone under THD + MXFP8 is still the knob's own decline.
    with _no_device_sync(), pytest.raises(NotImplementedError, match="fully fused|fuse_gate"):
        GatedAttentionBlockFwd(*args, fuse_gate=True, **kw)


# ---------------------------------------------------------------------------
# Accept -- Rubin only
# ---------------------------------------------------------------------------


@requires_rubin
@pytest.mark.parametrize("cu", [False, True], ids=["lengths", "prefix"])
def test_thd_mxfp8_unfused_matches_the_fake_quant_oracle(cu):
    """The UNFUSED MXFP8 pipeline under THD on ``(300, 128, 200)`` (causal, QK-norm; a tail tile on every sequence, none a 128-multiple):
    nine stages, the packed quantize arm, the SDPA on the MXFP8 row's THD arm (NATURAL, cga1, block scales, amax folded out); the
    sentinel-filled output fully written and finite; per sequence against the per-sequence fake-quant oracle at the dense MXFP8 suite's
    cosine floor; the three scale-factor slots carry no E8M0 NaN; a second execute into a fresh workspace is bitwise."""
    r = _run_mx_thd(_COMMON, _LENS, cu=cu)
    blk = r.blk
    assert [s.name for s in blk._stages] == _MX_STAGES and blk._quant_q.packed and blk._sdpa.mxfp8 and blk._sdpa.thd
    tp = blk._sdpa._impl.template_params()
    assert tp.sched_policy == SCHED_NATURAL and tp.cta_mma == 1 and tp.epilogue_gate is False
    assert blk._sdpa._impl._pertensor is False and blk._sdpa._impl.has_amax_o is False and blk._sdpa._impl.thd
    _assert_fully_written(r)
    for name, sf in zip(("sf_q", "sf_k", "sf_v"), _sf_slots_of(r)):
        assert not (sf == 0xFF).any(), f"{name}: an E8M0 NaN byte in the packed scale-factor slot"
    _check_mx_per_sequence(r)
    out2 = torch.full_like(r.out, _SENTINEL)
    ws2 = torch.empty_like(r.ws)
    _execute_mx(r, ws2, out=out2)
    torch.cuda.synchronize()
    assert torch.equal(out2, r.out), "a second execute into a fresh workspace differs"
    for a, b in zip(_sf_slots_of(r), _sf_slots_of(r, ws2)):
        assert torch.equal(a, b)


@requires_rubin
@pytest.mark.parametrize(
    "lens, s_max",
    [((300, 0, 200), 300), ((0, 128, 200), 200), ((300, 200, 0), 300), ((5, 130), 130)],
    ids=["middle_empty", "leading_empty", "trailing_empty", "5tok_and_2tile"],
)
def test_thd_mxfp8_degenerate_packings_under_scale_factor_poison(lens, s_max):
    """The whole workspace is poisoned with ``0xFF`` (E8M0 NaN) BEFORE the launch: every byte of the three scale-factor slots is written
    by the packed quantize (live tiles' pad bytes AND the slack tiles past the live total ``0x00``) -- an empty sequence owns no tile,
    a trailing empty one puts the slack right after the last live tile, a 5-token sequence is 123 pad rows -- the output is finite with
    no sentinel survivor, per sequence on the oracle, and bitwise the same block run on a clean workspace (the SDPA's packed scratch
    is written before it is read).  Run under the suite's ``timeout``: an empty sequence is a zero-length KV entry for the d256 row."""
    r = _run_mx_thd(_COMMON, lens, max_seq_len=s_max, ws_fill=0xFF)
    _assert_fully_written(r)
    cap = n_sf_tiles_packed_cap(r.meta["t"], r.meta["b"])
    live = sum(-(-n // 128) for n in lens)
    print(f"\nlens={lens}: live SF tiles per head {live}, capacity {cap} (slack {cap - live})")
    for name, sf, h in zip(("sf_q", "sf_k", "sf_v"), _sf_slots_of(r), (r.g.h_q, r.g.h_kv, r.g.h_kv)):
        tiles = sf.view(h, cap, 1024)
        assert not (sf == 0xFF).any(), f"{name}: {int((sf == 0xFF).sum())} poisoned bytes survived in the packed slot"
        assert (tiles[:, live:] == 0).all(), f"{name}: a slack tile past the live total is not zero-filled"
    _check_mx_per_sequence(r)
    clean = torch.empty_like(r.ws)
    out2 = torch.full_like(r.out, _SENTINEL)
    _execute_mx(r, clean, out=out2)
    torch.cuda.synchronize()
    assert torch.equal(out2, r.out), "the poisoned-workspace run differs from the clean one: a scratch byte was read before it was written"


@requires_rubin
def test_thd_mxfp8_b1_is_bitwise_the_dense_block():
    """``B = 1`` packed ``(512,)`` against the dense ``B=1, S=512`` MXFP8 TRAINING block over the SAME bytes (the dense training suite's
    own block and calibrated spec; the packed oracle calibration equals the dense one): ``out``, every record tensor, the e4m3
    ``q8`` / ``k8`` / ``v8`` and the Q / K scale-factor slots BITWISE -- one sequence over the same tiles -- and V's slot is the dense
    atoms in the plane-adjacent order (the exact permutation ``(h, n, plane, 512) <-> (plane, h, n, 512)``; byte-equal only at
    ``h * n == 1``).  A difference here is a finding, never a tolerance."""
    t = 512
    d = _dense_mx_training(_COMMON, 1, t)
    r = _run_mx_thd(_COMMON, (t,), training=True, refs=False, scale_o=d.spec.scale_o)
    assert r.spec == d.spec
    assert mxfp8_calibrated_scale_o_packed(r.orac, r.rg, r.meta["lens"]) == mxfp8_calibrated_scale_o(d.inp, RefGeometry(**_COMMON)) == d.spec.scale_o
    for k in ("h", "w_qkvg", "w_o", "cos", "sin", "h_sf", "w_qkvg_sf", "w_q_norm", "w_k_norm"):
        assert torch.equal(d.inp[k].reshape(r.kern[k].shape), r.kern[k]), k
    g = r.g
    for name, a, b in (
        ("proj_slab", r.saved.proj_slab, d.saved.proj_slab),
        ("rstd_q", r.saved.rstd_q, d.saved.rstd_q),
        ("rstd_k", r.saved.rstd_k, d.saved.rstd_k),
        ("saved.o", r.saved.o, d.saved.o),
        ("saved.lse", r.saved.lse, d.saved.lse),
        ("out", r.out, d.out),
    ):
        diff = (a.float() - b.float()).abs().max().item()
        assert torch.equal(a, b), f"{name}: packed B=1 differs from the dense block (max|diff| {diff:.3e}) -- a finding, not a tolerance"
    for name, a, b in zip(("q8", "k8", "v8"), _payload_of(r), _payload_of(SimpleNamespace(g=d.geom, meta=dict(t=t), blk=d.blk, ws=d.ws))):
        assert torch.equal(a, b), f"{name}: the quantize payload differs"
    sfq_p, sfk_p, sfv_p = _sf_slots_of(r)
    sfq_d, sfk_d, sfv_d = d.blk._sf_views(d.ws, d.blk._layout())
    n = -(-t // 128)
    assert n_sf_tiles_packed_cap(t, 1) == n and sfq_p.numel() == sfq_d.numel()
    assert torch.equal(sfq_p, sfq_d) and torch.equal(sfk_p, sfk_d), "the Q / K scale-factor slots differ at B = 1"
    assert torch.equal(
        sfv_p.view(g.h_kv, n, 2, 512).permute(2, 0, 1, 3), sfv_d.view(2, g.h_kv, n, 512)
    ), "V's packed slot is not the plane permutation of the dense one"
    assert not torch.equal(sfv_p, sfv_d), "h_kv * n > 1 here: the two V layouts cannot be byte-equal"


@requires_rubin
def test_thd_mxfp8_uniform_b4_matches_the_dense_block_per_sequence():
    """Uniform ``B = 4`` packed ``(256,)*4`` against the dense ``B=4, S=256`` MXFP8 TRAINING block over the SAME bytes: every token-wise
    stage BITWISE (slab, rstd, the e4m3 ``q8`` / ``k8`` / ``v8``), the three scale-factor slots an exact PERMUTATION of the dense ones
    (Q / K slabs ``(h, b, n) <-> (b, h, n)``; V ``(h, b, n, plane) <-> (plane, b, h, n)``) with the capacity's three slack tiles per
    head zero, the SDPA-derived ``O`` / ``LSE`` / ``out`` per sequence at the packed bf16 suite's bounds against the dense block (the
    first differing stage printed), and ``out`` per sequence on the oracle."""
    s, b = 256, 4
    d = _dense_mx_training(_COMMON, b, s)
    r = _run_mx_thd(_COMMON, (s,) * b, training=True, scale_o=d.spec.scale_o)
    assert r.spec == d.spec and mxfp8_calibrated_scale_o_packed(r.orac, r.rg, r.meta["lens"]) == d.spec.scale_o
    g, t = r.g, b * s
    assert (
        torch.equal(d.inp["h"].reshape(1, t, -1), r.kern["h"])
        and torch.equal(d.inp["cos"].reshape(1, t, -1), r.kern["cos"])
        and torch.equal(d.inp["h_sf"], r.kern["h_sf"])
    )
    assert torch.equal(r.saved.proj_slab, d.saved.proj_slab), "proj_slab differs: the token-wise stage-(1) GEMM is not the same launch"
    assert torch.equal(r.saved.rstd_q, d.saved.rstd_q.reshape(1, t, -1)) and torch.equal(r.saved.rstd_k, d.saved.rstd_k.reshape(1, t, -1))
    for name, a, c in zip(("q8", "k8", "v8"), _payload_of(r), _payload_of(SimpleNamespace(g=d.geom, meta=dict(t=t), blk=d.blk, ws=d.ws))):
        assert torch.equal(a, c), f"{name}: the token-wise quantize payload differs between packed and dense"
    n, cap = s // 128, n_sf_tiles_packed_cap(t, b)
    assert cap == b * n + (b - 1) and cap == 11
    sf_p, sf_d = _sf_slots_of(r), d.blk._sf_views(d.ws, d.blk._layout())
    for name, p, dd, h in zip(("sf_q", "sf_k"), sf_p[:2], sf_d[:2], (g.h_q, g.h_kv)):
        tiles = p.view(h, cap, 1024)
        assert torch.equal(
            tiles[:, : b * n].reshape(h, b, n, 1024).permute(1, 0, 2, 3), dd.view(b, h, n, 1024)
        ), f"{name}: not the permutation of the dense slabs"
        assert (tiles[:, b * n :] == 0).all(), f"{name}: slack tiles not zero"
    tiles_v = sf_p[2].view(g.h_kv, cap, 2, 512)
    assert torch.equal(
        tiles_v[:, : b * n].reshape(g.h_kv, b, n, 2, 512).permute(3, 1, 0, 2, 4), sf_d[2].view(2, b, g.h_kv, n, 512)
    ), "sf_v: not the permutation of the dense atoms"
    assert (tiles_v[:, b * n :] == 0).all()
    first_diff = None
    for i, (lo, hi) in enumerate(sequence_slices(r.meta["lens"])):
        for name, a, dd in (
            ("saved.o", r.saved.o[0, lo:hi], d.saved.o[i]),
            ("saved.lse", r.saved.lse[0, :, lo:hi], d.saved.lse[i]),
            ("out", r.out[0, lo:hi], d.out[i]),
        ):
            md = (a.float() - dd.float()).abs().max().item()
            if md and first_diff is None:
                first_diff = (i, name, md)
            if name == "out":
                assert _cos(a, dd) > _OUT_VS_DENSE_COS, f"seq {i} out vs dense: cos {_cos(a, dd):.6f}"
            else:
                torch.testing.assert_close(a.float(), dd.float(), rtol=0, atol=_O_LSE_ATOL, msg=f"seq {i} {name} vs dense (max|diff| {md:.3e})")
    print(f"\nuniform B=4 vs dense: first SDPA-derived difference {first_diff} (None = bitwise)")
    _check_mx_per_sequence(r)


@requires_rubin
@pytest.mark.parametrize(
    "lens, geom_kw",
    [((130, 5), _COMMON), (_LENS, {**_COMMON, "h_q": 24, "h_kv": 2}), (_LENS, {**_COMMON, "h_q": 8, "h_kv": 8})],
    ids=["ragged_130_5", "gqa_24_2", "mha_8_8"],
)
def test_thd_mxfp8_ragged_envelope_and_gqa(lens, geom_kw):
    """A ragged envelope (a 2-tile sequence beside a 5-token one: S_max 130, T 135) and the head ratios 24/2 (the GQA ratio the block
    is built for, at the test ``d_model``) and 8/8 (MHA), each per sequence on the oracle with the sentinel-filled output fully
    written."""
    r = _run_mx_thd(geom_kw, lens)
    _assert_fully_written(r)
    _check_mx_per_sequence(r)


@requires_rubin
@requires_fp4
@pytest.mark.parametrize("mode", ["w_qkvg_fp4", "o_nvfp4"])
def test_thd_mxfp8_fp4_modes_ride(mode):
    """The fp4 modes under THD, one accept cell each: an MXFP4 ``W_qkvg`` (stage (1) on the mixed e4m3 x e2m1 row; the oracle reads the
    e2m1 values as e4m3 shadow codes through the same blob) and an NVFP4 gated ``O`` (``quantize_fp4_o`` + the fp4 x fp4 block-scale
    out projection, no per-tensor scale) -- both per sequence on the oracle, the output fully written."""
    r = _run_mx_thd(_COMMON, _LENS, w_qkvg_fp4=(mode == "w_qkvg_fp4"), o_fp4=Fp4Format.NVFP4 if mode == "o_nvfp4" else None)
    blk = r.blk
    if mode == "w_qkvg_fp4":
        assert [s.name for s in blk._stages] == _MX_STAGES and blk._proj.w_dtype == _FP4 and blk._proj._plan.w_dtype == _FP4
    else:
        assert [s.name for s in blk._stages] == _FP4_O_STAGES and blk._out_proj._plan.block_scale and blk._out_proj._plan.dtype == _FP4
    _assert_fully_written(r)
    _check_mx_per_sequence(r)


@requires_rubin
@pytest.mark.parametrize("cu_base", [0, 100], ids=["prefix", "prefix_nonzero_base"])
def test_thd_mxfp8_lengths_and_prefix_forms_are_bitwise(cu_base):
    """The ``[B]`` lengths form and the ``[B+1]`` prefix-sum form (at base 0 and at a non-zero base) produce bitwise the same output,
    payload and scale-factor slots: both the SDPA's setup launch and the packed quantize normalize a prefix tensor to its first entry."""
    a = _run_mx_thd(_COMMON, _LENS, refs=False)
    b = _run_mx_thd(_COMMON, _LENS, refs=False, cu=True, cu_base=cu_base)
    assert a.spec == b.spec
    assert torch.equal(a.out, b.out)
    for x, y in zip(_payload_of(a) + tuple(_sf_slots_of(a)), _payload_of(b) + tuple(_sf_slots_of(b))):
        assert torch.equal(x, y)


@requires_rubin
def test_thd_mxfp8_training_record_is_bitwise_the_packed_inference_block():
    """The UNFUSED MXFP8 pipeline TRAINS under THD (``save_for_backward=True`` on ``(300, 128, 200)``, causal, QK-norm): the record is
    the bf16 record at ``(1, T)`` with ``saved.h`` the e4m3 codes, ``saved.seq_lens`` the lengths tensor itself and its form named.
    Same kernels, different buffers -- ``out``, the pre-gate ``O``, the LSE, the e4m3 ``q8`` / ``k8`` / ``v8``, the THREE scale-factor
    slots and the slab's GATE / V bands are the packed MXFP8 INFERENCE block's bit for bit; the record's Q/K bands are PRE-norm (the
    forward's own norm+RoPE over them reproduces the inference slab and the saved ``rstd`` bitwise, and the packed quantize stages
    replayed over the normed bands -- through the stage's ``execute(seq_lens=)`` -- reproduce ``q8`` / ``k8`` and the Q / K slots
    bitwise).  Per sequence: the saved LSE within 1e-4 of the fp64 log-sum-exp over the operands the SDPA READ (the MX fake-quant of
    the normed Q/K rowwise and of the sequence's V columnwise -- the quantized rows' exact-Stats contract) and ``out`` on the oracle."""
    from test_block_training_forward import _attention_fp64, _replay_norm

    tr = _run_mx_thd(_COMMON, _LENS, training=True)
    assert tr.blk.save_for_backward and tr.blk.return_lse and not tr.blk.inplace_qkv and tr.blk.thd
    assert tr.saved.h is tr.kern["h"] and tr.saved.h.dtype == E4M3 and tr.saved.o.dtype == tr.saved.proj_slab.dtype == torch.bfloat16
    assert tr.saved.seq_lens is tr.seq_lens and tr.saved.seq_lens_form == "lengths"
    _assert_fully_written(tr)
    _check_mx_per_sequence(tr)
    # The packed INFERENCE twin on the same inputs and spec, its LSE requested, its pre-gate O caught before stage (5) gates in place.
    inf = _declare_mx_thd(_COMMON, _LENS, return_lse=True, scale_o=tr.spec.scale_o, sentinel=False)
    assert inf.spec == tr.spec and not inf.blk.save_for_backward and inf.blk.inplace_qkv and inf.blk.return_lse
    inf.blk.check_support()
    inf.blk.compile()
    inf.ws = torch.empty(inf.blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    g, t, d = tr.g, tr.meta["t"], tr.g.d_head
    lse = torch.empty(1, g.h_q, t, dtype=torch.float32, device="cuda")
    caught = {}
    real_gate = inf.blk._gate.execute

    def gate_catching_o(o, gate, dst, current_stream=None):
        caught["o_pre"] = o.clone()  # the block launches on torch's current stream here, so the clone is ordered after the SDPA
        return real_gate(o, gate, dst, current_stream=current_stream)

    inf.blk._gate.execute = gate_catching_o
    _execute_mx(inf, inf.ws, lse=lse)
    torch.cuda.synchronize()
    assert inf.out.abs().max().item() > 0 and torch.isfinite(inf.out.float()).all()
    assert torch.equal(tr.out, inf.out), (tr.out.float() - inf.out.float()).abs().max().item()
    assert torch.equal(tr.saved.o, caught["o_pre"].view(1, t, g.h_q, d)), (tr.saved.o.float() - caught["o_pre"].view(1, t, g.h_q, d).float()).abs().max().item()
    assert torch.equal(tr.saved.lse, lse), (tr.saved.lse - lse).abs().max().item()
    lay_t, lay_i = tr.blk._layout(), inf.blk._layout()
    assert (
        lay_t.q >= 0 and lay_t.k >= 0 and lay_t.proj == -1 and lay_i.q == -1
    ), "the packed training carve reserves the compact normed Q/K; the inference carve does not"
    for nm, a, b in zip(("q8", "k8", "v8"), _payload_of(tr), _payload_of(inf)):
        assert torch.equal(a, b), f"{nm}: the quantize stages read different values"
    for nm, a, b in zip(("sf_q", "sf_k", "sf_v"), _sf_slots_of(tr), _sf_slots_of(inf)):
        assert torch.equal(a, b), f"{nm}: the packed scale factors differ between the training and the inference block"
    slab_i = _view(inf.ws, lay_i.proj, (t, g.n_qkvg), torch.bfloat16)
    tq, tgate, tk, tv = saved_slab_views(tr.saved.proj_slab, g, 1, t)
    iq, igate, ik, iv = saved_slab_views(slab_i, g, 1, t)
    assert torch.equal(tgate, igate) and torch.equal(tv, iv)
    assert not torch.equal(tq, iq) and not torch.equal(tk, ik), "the record's Q/K bands are the inference slab's POST-norm bands"
    nq, nk, rq, rk = _replay_norm(SimpleNamespace(blk=tr.blk, geom=g, batch=1, seq_len=t, inp=tr.kern), tq, tk)
    assert torch.equal(nq.view(1, t, g.h_q, d), iq) and torch.equal(nk.view(1, t, g.h_kv, d), ik), "norm+RoPE over the record's bands is not the inference slab"
    assert torch.equal(rq.view(1, t, g.h_q), tr.saved.rstd_q) and torch.equal(rk.view(1, t, g.h_kv), tr.saved.rstd_k)
    # The packed quantize stages replayed over the normed bands, through the stage's own packed execute (seq_lens=, no batch / seq_len).
    q8, k8, _v8 = _payload_of(tr)
    sfq, sfk, _sfv = _sf_slots_of(tr)
    q8r, k8r = torch.empty(t, g.h_q, d, dtype=E4M3, device="cuda"), torch.empty(t, g.h_kv, d, dtype=E4M3, device="cuda")
    sfq_r, sfk_r = torch.full_like(sfq, 0xFF), torch.full_like(sfk, 0xFF)
    tr.blk._quant_q.execute(nq, q8r, sfq_r, seq_lens=tr.seq_lens)
    tr.blk._quant_k.execute(nk, k8r, sfk_r, seq_lens=tr.seq_lens)
    torch.cuda.synchronize()
    assert torch.equal(q8r.view(torch.uint8), q8) and torch.equal(
        k8r.view(torch.uint8), k8
    ), "norm + packed quantize over the record's bands is not the forward's q8 / k8"
    assert torch.equal(sfq_r, sfq) and torch.equal(sfk_r, sfk), "the replayed packed scale factors differ from the forward's"
    # Per sequence: the saved LSE is the exact fp64 log-sum-exp over the MX fake-quant operands the SDPA read.
    nq4, nk4 = nq.view(1, t, g.h_q, d), nk.view(1, t, g.h_kv, d)

    def check(i, lo, hi, _ref):
        q64 = mx_fake_quant_rowwise(nq4[:, lo:hi]).double()
        k64 = mx_fake_quant_rowwise(nk4[:, lo:hi]).double()
        v64 = mx_fake_quant_v_columnwise(tv[:, lo:hi]).double()
        _o64, lse64 = _attention_fp64(q64, k64, v64, g)
        d_lse = (tr.saved.lse[0, :, lo:hi].double() - lse64[0]).abs().max().item()
        print(f"seq {i} [{lo}:{hi}]: max|dLSE|={d_lse:.3e} vs the fp64 log-sum-exp over the MX operands the SDPA read")
        assert d_lse <= _LSE_FP64_ATOL, f"the saved LSE is not the exact log-sum-exp of the operands the SDPA read (max |dLSE| {d_lse:.3e})"

    failures = compare_packed(tr.refs, tr.meta["lens"], check)
    assert not failures, "\n".join(failures)


@requires_rubin
def test_thd_mxfp8_quantize_stage_execute_contract_is_typed_both_ways():
    """On COMPILED stages: a packed quantize stage refuses ``batch`` / ``seq_len`` and needs ``seq_lens``; a dense one refuses
    ``seq_lens``; the packed host refuses a scale-factor buffer that is not the capacity count and a lengths tensor of the wrong
    length.  Host-side (``set_sync_debug_mode("error")``), nothing launches."""
    r = _run_mx_thd(_COMMON, _LENS, refs=False)
    g, t, b = r.g, r.meta["t"], r.meta["b"]
    src = torch.zeros(t, g.h_q, g.d_head, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty(t, g.h_q, g.d_head, dtype=E4M3, device="cuda")
    sf = torch.empty(r.blk._quant_q.sf_bytes(), dtype=torch.uint8, device="cuda")
    st = r.blk._quant_q
    with _no_device_sync():
        with pytest.raises(ValueError, match="needs seq_lens"):
            st.execute(src, dst, sf)
        with pytest.raises(ValueError, match="batch / seq_len"):
            st.execute(src, dst, sf, batch=1, seq_len=t, seq_lens=r.seq_lens)
        with pytest.raises(ValueError, match="capacity bound"):
            st.execute(src, dst, sf[: _sf_slot_bytes(1, g.h_q, t, g.d_head)], seq_lens=r.seq_lens)
        with pytest.raises(ValueError, match=f"of {b} elements"):
            st.execute(src, dst, sf, seq_lens=r.meta["cu_seqlens"])
    out_d, _ref, blk_d, _mx, _spec = mx_suite._run_mx_block(_COMMON, batch=1, seq_len=512)
    src_d = torch.zeros(512, g.h_q, g.d_head, dtype=torch.bfloat16, device="cuda")
    dst_d = torch.empty(512, g.h_q, g.d_head, dtype=E4M3, device="cuda")
    sf_d = torch.empty(blk_d._quant_q.sf_bytes(), dtype=torch.uint8, device="cuda")
    with _no_device_sync(), pytest.raises(ValueError, match="packed=False"):
        blk_d._quant_q.execute(src_d, dst_d, sf_d, batch=1, seq_len=512, seq_lens=r.seq_lens)
    assert not blk_d._quant_q.packed and blk_d._quant_q.sf_bytes() == _sf_slot_bytes(1, g.h_q, 512, g.d_head)


@requires_rubin
def test_thd_mxfp8_workspace_size_is_honest():
    """``get_workspace_size()`` is exact and never exceeded on the packed MXFP8 TRAINING block: a buffer 4096 B larger keeps its tail
    untouched, two executes allocate nothing, the result over a sentinel-filled buffer is bitwise the first run's (the SF slots too),
    the SDPA's packed scratch is non-zero and reported, one byte less is a typed ``ValueError``."""
    r = _run_mx_thd(_COMMON, _LENS, training=True, refs=False)
    blk = r.blk
    size, lay, sdpa_scratch = blk.get_workspace_size(), blk._layout(), blk._sdpa.scratch_workspace_bytes()
    print(
        f"\nworkspace {size} B; carve {lay.total_bytes} B; SDPA packed scratch {sdpa_scratch} B; packed SF slots {lay.sf_k - lay.sf_q} / {lay.sf_v - lay.sf_k} B"
    )
    assert sdpa_scratch > 0 and size >= lay.total_bytes + sdpa_scratch and size % 256 == 0
    assert lay.sf_k - lay.sf_q >= blk._quant_q.sf_bytes() and lay.sf_v - lay.sf_k >= blk._quant_k.sf_bytes()
    ws = torch.full((size + 4096,), 0xAB, dtype=torch.uint8, device="cuda")
    out = torch.empty_like(r.out)
    saved = thd_suite._alloc_packed_saved(blk.geom, r.kern, r.meta, seq_lens=r.seq_lens, form="lengths", act_dtype=torch.bfloat16)
    _execute_mx(r, ws, saved=saved, out=out)
    torch.cuda.synchronize()
    gc.collect()
    live = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    n0 = torch.cuda.memory_stats()["allocation.all.allocated"]
    _execute_mx(r, ws, saved=saved, out=out)
    _execute_mx(r, ws, saved=saved, out=out)
    torch.cuda.synchronize()
    n1 = torch.cuda.memory_stats()["allocation.all.allocated"]
    peak = torch.cuda.max_memory_allocated()
    assert n1 == n0, f"the packed MXFP8 forward made {n1 - n0} CUDA allocation(s) on the execute path (allocation.all.allocated {n0} -> {n1})"
    assert peak <= live, f"a temporary on the execute path: the allocator peak rose from {live} to {peak} bytes"
    assert torch.equal(ws[size:], torch.full((4096,), 0xAB, dtype=torch.uint8, device="cuda")), "bytes past get_workspace_size() were written"
    assert torch.equal(out, r.out) and torch.equal(saved.o, r.saved.o) and torch.equal(saved.lse, r.saved.lse)
    for a, b in zip(_sf_slots_of(r), _sf_slots_of(r, ws)):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match="workspace"):
        _execute_mx(r, ws[: size - 1], saved=saved, out=out)


@requires_rubin
def test_thd_mxfp8_launch_count_is_dense_plus_one():
    """CUPTI kernel records of one packed MXFP8 execute == the dense ``B=1, S=T`` MXFP8 block's nine + 1: the packed SDPA issues its
    metadata / descriptor SETUP launch before the main kernel; the three packed quantize launches are the dense three.  No hidden
    memcpy.  MEASURED, the names printed; a typed skip when CUPTI records nothing on this node."""
    r = _run_mx_thd(_COMMON, _LENS, refs=False)
    t = r.meta["t"]
    out_d, _ref, blk_d, mx_d, _spec = mx_suite._run_mx_block(_COMMON, batch=1, seq_len=t)
    ws_d = torch.empty(blk_d.get_workspace_size(), dtype=torch.uint8, device="cuda")
    k_thd, m_thd = thd_suite._cupti_kernels(lambda: _execute_mx(r, r.ws))
    k_dense, m_dense = thd_suite._cupti_kernels(lambda: mx_suite._execute(blk_d, mx_d, out_d, ws_d))
    if k_thd is None or k_dense is None:
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the launch count is unverified here")
    print(f"\npacked mxfp8: {len(k_thd)} kernels + {len(m_thd)} memset/memcpy:\n  " + "\n  ".join(k_thd))
    print(f"dense mxfp8:  {len(k_dense)} kernels + {len(m_dense)} memset/memcpy:\n  " + "\n  ".join(k_dense))
    assert not [n for n in m_thd if "memcpy" in n.lower()], "a hidden copy on the packed execute path"
    assert len(k_dense) == len(_MX_STAGES) == 9, (len(k_dense), k_dense)
    assert len(k_thd) == len(k_dense) + 1, (len(k_thd), len(k_dense))
