# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The PACKED (THD / varlen) forward of the gated attention block: ``GatedAttentionBlockFwd(thd=True, ...)``.

Under THD the block takes ONE packed token matrix ``[T, d_model]`` (or ``[1, T, d_model]``) holding ``B`` sequences back to
back, per-token RoPE tables whose positions restart at every sequence, and ``execute(seq_lens=)`` -- ``[B]`` int32 lengths
or ``[B+1]`` int32 prefix sums (``cu_seqlens=True``) -- as the ONE tensor the SDPA reads for both its Q and its KV side.
Every token-wise stage (the two projections, norm + RoPE, the gate, the quantize passes) runs exactly as a dense ``B=1,
S=T`` block would; only the SDPA runs its packed specialization.  The caller contract on the lengths (never read on the
host): every length in ``[0, max_seq_len]``, prefix sums non-decreasing (any base), and ``sum(lengths) == T``.

The reference is the dense fp32 oracle run PER SEQUENCE (``gated_block_reference.gated_attention_block_reference_packed``:
each sequence's causal diagonal / window / bottom-right alignment is its own), compared sequence by sequence at the dense
suites' bounds -- ``cos >= 0.999`` on ``out``, the SDPA stage test's ``atol 2e-2`` on ``saved.o`` / ``saved.lse``, one bf16
rounding (``rtol 2**-7, atol 1e-3``) on the saved stage-(1) bands, the norm test's bounds on ``rstd`` -- never a new looser
one.  A THD bug that leaks across a sequence boundary shows up as one sequence's rows contaminated by its neighbour's,
which a whole-tensor cosine would average away.  Verdicts are COLLECTED and asserted at the end, so a failure names its
sequence.  Degenerate packings are first-class cells: zero-length sequences at the front, in the middle and trailing
(``B`` varies -> pad with empties), a 5-token sequence (never a 1-token one: the fp64 oracle's ``softmax @ v`` on a 1x1
dies before any FROST kernel runs), lengths that are not tile multiples, rows past the live total.

Two dense-vs-packed pins: ``B=1`` packed ``(T,)`` against the dense ``B=1, S=T`` block is BITWISE on ``out`` and on every
record tensor (one sequence over the same tiles -- a difference there is a finding, never a tolerance change); uniform
``B=4`` packed ``(256,)*4`` against the dense ``B=4, S=256`` block is bitwise on the token-wise stages (``proj_slab``,
``rstd``) and at the dense suite's tolerance on the SDPA-derived ``O`` / ``LSE`` / ``out`` (the packed SDPA walks the tiles
differently and runs the padded-mask arm).

Accept tests are ``requires_rubin`` (the block targets SM107 only); the reject tests build CUDA tensors for a DECLARED block
and run under ``torch.cuda.set_sync_debug_mode("error")`` (``requires_cuda``: every decline is host-side, no device read);
the static pins (the packed LSE descriptor, the scheduler policy, the workspace carve) run anywhere, CPU included.
"""

import contextlib
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
from cudnn.gated_attention_block import (
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    MxQuantSpec,
    QuantSpec,
    SavedForBackward,
    saved_slab_views,
)  # noqa: E402
from cudnn.gated_attention_block.api import _Sdpa, _thd_lse_desc, _thd_lse_head_stride, _view  # noqa: E402
from cudnn.gated_attention_block.kernels.proj_gemm import sf_blob_bytes  # noqa: E402
from cudnn.sdpa.graph_analyzer import thd_stats_packing  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import (  # noqa: E402
    RefGeometry,
    TAIL_SENTINEL,
    amax_scale,
    assert_packing_contract,
    compare_packed,
    cu_seqlens_of,
    gated_attention_block_fp8_reference,
    gated_attention_block_reference_packed,
    make_inputs,
    make_packed_inputs,
    packed_rope_tables,
    qk_norm_rope_reference,
    quantize_block_inputs,
    sequence_slices,
)

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


# The REGISTERED marker of cutedsl/conftest.py (the skip is applied at collection).
requires_rubin = pytest.mark.requires_rubin
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_COMMON = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)
_LENS = (300, 128, 200)  # B=3, T=628, S_max=300: a tail tile on every sequence, none a 128-multiple
_SENTINEL = 1.5e30  # a finite magnitude no correct output cell holds: a survivor is an unwritten cell (bf16 / fp32)
_SENTINEL_FP16 = 6.0e4  # fp16 max is 65504: the 1.5e30 fill overflows there; no correct cell reaches 6e4 either
_ROPE_BASE = RefGeometry(**_COMMON).rope_base  # the reference geometry's RoPE base (the block geometry carries none)


def _sentinel_for(dtype):
    return _SENTINEL_FP16 if dtype == torch.float16 else _SENTINEL


_QK_NORM = pytest.mark.parametrize("qk_norm", [True, False], ids=["norm", "rope_only"])


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


@contextlib.contextmanager
def _no_device_sync():
    """Every reject below is HOST-side: a device read inside it is itself a failure (``set_sync_debug_mode("error")``)."""
    if not torch.cuda.is_available():
        yield
        return
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(prev)


# ---------------------------------------------------------------------------
# Building a packed block
# ---------------------------------------------------------------------------


def _thd_kw(meta, *, cu=False, max_seq_len=None):
    """The four appended constructor kwargs of a packed block, from a packing's metadata."""
    return dict(thd=True, num_sequences=meta["b"], max_seq_len=meta["max_seq_len"] if max_seq_len is None else max_seq_len, cu_seqlens=bool(cu))


def _lens(meta, cu=False):
    return meta["cu_seqlens"] if cu else meta["seq_lens"]


def _form(cu=False):
    return "prefix" if cu else "lengths"


def _rank2(x):
    """``[1, T, .]`` -> ``[T, .]`` (the rank-2 spelling a packed caller holds)."""
    return x.view(x.shape[1], x.shape[2])


def _declare_thd(geom_kw, lens, *, dtype=torch.bfloat16, rank2=False, cu=False, max_seq_len=None, seed=0, out_sentinel=None, **blk_kw):
    """A DECLARED (not compiled) packed block plus its inputs, output buffer and packing metadata.

    ``rank2`` hands ``h`` / ``cos`` / ``sin`` / ``out`` over as ``[T, .]`` instead of ``[1, T, .]`` (both are served)."""
    block_geom = GatedAttentionBlockGeometry(**geom_kw)
    inp, meta = make_packed_inputs(RefGeometry(**geom_kw), lens, dtype=dtype, seed=seed, max_seq_len=max_seq_len)
    out = torch.empty(1, meta["t"], block_geom.d_model, device="cuda", dtype=dtype)
    if out_sentinel is not None:
        out.fill_(_sentinel_for(dtype) if out_sentinel == _SENTINEL else out_sentinel)
    if rank2:
        inp["h"], inp["cos"], inp["sin"], out = _rank2(inp["h"]), _rank2(inp["cos"]), _rank2(inp["sin"]), _rank2(out)
    blk = GatedAttentionBlockFwd(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, block_geom, **_thd_kw(meta, cu=cu), **blk_kw
    )
    return blk, inp, out, meta


def _alloc_packed_saved(geom, inp, meta, *, seq_lens, form, sentinel=None, with_views=True, act_dtype=None):
    """A caller-owned record for a packed TRAINING forward (the proj_slab save mode): every tensor is the dense record's at
    ``B=1, S=T`` -- ``proj_slab [T, n_qkvg]``, ``o [1, T, H_q, D]``, ``lse [1, H_q, T]`` (head-major, head stride T),
    ``rstd_* [1, T, H]`` -- plus the packed lengths tensor and its FORM (``"lengths"`` / ``"prefix"``), which the record
    must say (``seq_lens_form``) so a padded dense record cannot be mistaken for a packed one.  ``act_dtype`` (appended): the
    ACTIVATION dtype of the slab / O buffers -- ``inp["h"].dtype`` by default (bf16 / fp16); ``torch.bfloat16`` for a packed
    per-tensor FP8 training forward, whose ``h`` is e4m3 codes while every activation it writes is bf16."""
    g, t = geom, meta["t"]
    dtype, dev = (inp["h"].dtype if act_dtype is None else act_dtype), inp["h"].device

    def buf(*shape, dt=dtype):
        x = torch.empty(*shape, dtype=dt, device=dev)
        if sentinel is not None:
            x.fill_(_sentinel_for(dt) if sentinel == _SENTINEL else sentinel)
        return x

    proj_slab = buf(t, g.n_qkvg)
    q_pre, gate, k_pre, _v = saved_slab_views(proj_slab, g, 1, t) if with_views else (None, None, None, None)
    return SavedForBackward(
        h=inp["h"],
        gate=gate,
        o=buf(1, t, g.h_q, g.d_head),
        lse=buf(1, g.h_q, t, dt=torch.float32),
        rstd_q=buf(1, t, g.h_q, dt=torch.float32) if g.qk_norm else None,
        rstd_k=buf(1, t, g.h_kv, dt=torch.float32) if g.qk_norm else None,
        q_pre=q_pre,
        k_pre=k_pre,
        proj_slab=proj_slab,
        seq_lens=seq_lens,
        seq_lens_form=form,
    )


def _execute(blk, inp, out, ws, *, seq_lens, saved=None, lse=None, current_stream=None):
    blk.execute(
        inp["h"],
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        out,
        ws,
        seq_lens=seq_lens,
        lse=lse,
        saved=saved,
        current_stream=current_stream,
    )


def _refs(inp, geom_kw, meta):
    return gated_attention_block_reference_packed(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], RefGeometry(**geom_kw), meta["lens"]
    )


def _run_thd(geom_kw, lens, *, dtype=torch.bfloat16, training=False, cu=False, rank2=False, sentinel=None, max_seq_len=None, seed=0, refs=True, **blk_kw):
    """Declare, compile and run a packed forward (inference, or training with a sentinel-filled record); returns a namespace
    with ``blk, inp, out, meta, seq_lens, saved, refs``."""
    if training:
        blk_kw.setdefault("save_for_backward", True)
    blk, inp, out, meta = _declare_thd(geom_kw, lens, dtype=dtype, rank2=rank2, cu=cu, max_seq_len=max_seq_len, seed=seed, out_sentinel=sentinel, **blk_kw)
    seq_lens = _lens(meta, cu)
    saved = _alloc_packed_saved(blk.geom, inp, meta, seq_lens=seq_lens, form=_form(cu), sentinel=sentinel) if training else None
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _execute(blk, inp, out, ws, seq_lens=seq_lens, saved=saved)
    torch.cuda.synchronize()
    return SimpleNamespace(
        blk=blk, inp=inp, out=out, meta=meta, seq_lens=seq_lens, saved=saved, ws=ws, refs=_refs(inp, geom_kw, meta) if refs else None, geom_kw=geom_kw
    )


def _rows(x):
    """``[1, T, ...]`` or ``[T, ...]`` -> ``[T, ...]``."""
    return x[0] if x.dim() >= 3 and x.shape[0] == 1 else x


def _check_forward_per_sequence(res, *, check_record=True, out_cos=0.999):
    """The dense suites' bounds, per sequence: ``out`` cos; under a record the bands at one bf16 rounding, pre-gate ``O``
    (cos + ``atol 2e-2``), ``LSE`` (``atol 2e-2``), ``rstd`` (the norm test's bounds vs the oracle norm of the block's own
    pre-norm bands, plus the measured-envelope cross-check against the oracle's rstd).  Failures are collected per
    sequence and asserted at the end."""
    g, saved, inp = res.blk.geom, res.saved, res.inp
    out = _rows(res.out)
    tol = dict(rtol=2**-7, atol=1e-3)
    assert torch.isfinite(out.float()).all(), "non-finite output rows"

    def check(i, lo, hi, ref):
        c = _cos(out[lo:hi], ref.out[0])
        assert c > out_cos, f"out cos {c:.6f}"
        if not (check_record and saved is not None):
            return
        torch.testing.assert_close(saved.q_pre[0, lo:hi], ref.q_pre[0], **tol)
        torch.testing.assert_close(saved.k_pre[0, lo:hi], ref.k_pre[0], **tol)
        torch.testing.assert_close(saved.gate[0, lo:hi], ref.gate[0], **tol)
        v = saved_slab_views(saved.proj_slab, g, 1, res.meta["t"])[3]
        torch.testing.assert_close(v[0, lo:hi], ref.v[0], **tol)
        o = saved.o[0, lo:hi]
        assert torch.isfinite(o.float()).all(), "non-finite saved.o rows"
        co = _cos(o, ref.o[0])
        assert co > 0.999, f"pre-gate O cos {co:.6f}"
        torch.testing.assert_close(o.float(), ref.o[0].float(), rtol=0, atol=2e-2)
        torch.testing.assert_close(saved.lse[0, :, lo:hi], ref.lse[0], rtol=0, atol=2e-2)
        if g.qk_norm:
            cos3, sin3 = inp["cos"].view(1, -1, g.rope_dim)[:, lo:hi], inp["sin"].view(1, -1, g.rope_dim)[:, lo:hi]
            _, rq = qk_norm_rope_reference(saved.q_pre[:, lo:hi], inp["w_q_norm"], cos3, sin3, g.rope_dim, g.qk_norm_eps, qk_norm=True)
            _, rk = qk_norm_rope_reference(saved.k_pre[:, lo:hi], inp["w_k_norm"], cos3, sin3, g.rope_dim, g.qk_norm_eps, qk_norm=True)
            torch.testing.assert_close(saved.rstd_q[:, lo:hi], rq, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(saved.rstd_k[:, lo:hi], rk, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(saved.rstd_q[:, lo:hi], ref.rstd_q, rtol=1e-3, atol=1e-5)
            torch.testing.assert_close(saved.rstd_k[:, lo:hi], ref.rstd_k, rtol=1e-3, atol=1e-5)
        else:
            assert saved.rstd_q is None and saved.rstd_k is None

    failures = compare_packed(res.refs, res.meta["lens"], check)
    assert not failures, "\n".join(failures)


def _assert_empty_sequences_leave_no_rows(res):
    """An empty sequence owns no rows; its neighbours' rows are checked exactly by the per-sequence comparison, and the
    sentinel-filled record / output must carry NO survivor anywhere (every row belongs to some sequence under sum == T)."""
    out = _rows(res.out)
    sent = _sentinel_for(out.dtype)
    assert not (out == sent).any(), f"{int((out == sent).sum())} output cells were never written"
    if res.saved is not None:
        assert not (res.saved.o == sent).any() and not (res.saved.lse == _SENTINEL).any(), "record rows were never written"


# ---------------------------------------------------------------------------
# Static pins -- the packed LSE descriptor, the scheduler policy, the workspace carve (any device)
# ---------------------------------------------------------------------------


def test_thd_lse_desc_is_head_major_with_head_stride_T():
    """The ONE definition of the packed Stats layout: ``_thd_lse_head_stride(T) == T`` and ``_thd_lse_desc`` declares the
    envelope shape ``(B, H_q, S_max)`` with strides ``(H_q*T, T, 1)`` -- the head-major packing the forward adapter
    classifies and the backward binds as a contiguous ``[1, H_q, T]`` -- which IS the dense record's ``[B, H_q, S]`` at
    ``B = 1, S = T``.  Pure Python, CPU descriptors."""
    b, h, s_max, t = 3, 8, 300, 628
    assert _thd_lse_head_stride(t) == t
    d = _thd_lse_desc(b, h, s_max, t, torch.device("cpu"))
    assert tuple(d.shape) == (b, h, s_max) and tuple(d.stride) == (h * t, t, 1) and d.dtype == torch.float32
    assert thd_stats_packing(d.stride[1], d.stride[2], h) == "head_major" and d.stride[1] == _thd_lse_head_stride(t)
    lse = torch.empty(1, h, t)  # the record tensor: head stride T, token stride 1, exactly H_q * T elements
    assert lse.stride() == (h * t, t, 1) and lse.numel() == h * _thd_lse_head_stride(t)


@pytest.mark.parametrize("cu", [False, True], ids=["lengths", "prefix"])
def test_thd_sdpa_stage_declares_natural(cu):
    """The SDPA stage under THD declares ``thd=True`` with both packed totals ``T``, the lengths FORM on both sides, the
    head-major LSE descriptor (head stride exactly ``T``) and ``sched_policy == SCHED_NATURAL`` (the packed grid is a
    flat unit grid driven by the persistent claim-counter scheduler; an LPT walk under THD attributes units wrong).
    The adapter's constructor touches no device, so this runs on CPU descriptors anywhere."""
    g = GatedAttentionBlockGeometry(**_COMMON)
    b, s_max, t = 3, 300, 628
    stage = _Sdpa(
        g,
        batch=1,
        seq_len=t,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        want_lse=True,
        token_stride=g.n_qkvg,
        thd=True,
        num_sequences=b,
        max_seq_len=s_max,
        cu_seqlens=cu,
    )
    impl = stage._build_impl()
    assert impl.thd is True
    assert impl.max_total_seq_len_q == t and impl.max_total_seq_len_kv == t
    assert impl.cu_seq_q_lens is cu and impl.cu_seq_kv_lens is cu
    assert int(impl.sched_policy) == SCHED_NATURAL
    assert tuple(impl.q_desc.shape) == (b, g.h_q, s_max, g.d_head) and tuple(impl.k_desc.shape) == (b, g.h_kv, s_max, g.d_head)
    assert impl.q_desc.stride[2] == g.n_qkvg, "in-place Q/K/V read the slab at its token stride"
    assert tuple(impl.lse_desc.shape) == (b, g.h_q, s_max)
    assert thd_stats_packing(impl.lse_desc.stride[1], impl.lse_desc.stride[2], g.h_q) == "head_major" and impl.lse_desc.stride[1] == _thd_lse_head_stride(t)
    # The dense stage is untouched: no THD, no totals, the row's own policy request.
    dense = _Sdpa(g, batch=1, seq_len=t, dtype=torch.bfloat16, device=torch.device("cpu"), want_lse=True)._build_impl()
    assert dense.thd is False and dense.max_total_seq_len_q is None and tuple(dense.lse_desc.shape) == (1, g.h_q, t)


# ---------------------------------------------------------------------------
# Rejects -- declaration and execute contracts, host-side (any CUDA device, no compile, no device read)
# ---------------------------------------------------------------------------


def _decl_thd_args(geom_kw=_COMMON, lens=_LENS, dtype=torch.bfloat16, rank2=False, seed=0):
    """The positional arguments of a packed block declaration (CUDA tensors, no launch)."""
    block_geom = GatedAttentionBlockGeometry(**geom_kw)
    inp, meta = make_packed_inputs(RefGeometry(**geom_kw), lens, dtype=dtype, seed=seed)
    out = torch.empty(1, meta["t"], block_geom.d_model, device="cuda", dtype=dtype)
    if rank2:
        inp["h"], inp["cos"], inp["sin"], out = _rank2(inp["h"]), _rank2(inp["cos"]), _rank2(inp["sin"]), _rank2(out)
    return (inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, block_geom), inp, meta


def _decl_fp8_thd_args(geom_kw=_COMMON, lens=_LENS, mxfp8=False):
    """A packed FP8 (or MXFP8) declaration: e4m3 ``h`` / weights, bf16 tables, the MXFP8 blobs when asked."""
    g = GatedAttentionBlockGeometry(**geom_kw)
    t = sum(lens)
    dev = "cuda"
    bf = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)  # noqa: E731
    e4 = torch.float8_e4m3fn
    h8 = torch.zeros(1, t, g.d_model, dtype=e4, device=dev)
    w8 = torch.zeros(g.n_qkvg, g.d_model, dtype=e4, device=dev)
    wo8 = torch.zeros(g.d_model, g.h_q * g.d_head, dtype=e4, device=dev)
    args = (h8, w8, bf(g.d_head), bf(g.d_head), bf(1, t, g.rope_dim), bf(1, t, g.rope_dim), wo8, bf(1, t, g.d_model), g)
    kw = dict(thd=True, num_sequences=len(lens), max_seq_len=max(lens))
    if mxfp8:
        kw.update(
            quant=MxQuantSpec(descale_w_o=0.03),
            sample_h_sf=torch.zeros(sf_blob_bytes(t, g.d_model), dtype=torch.uint8, device=dev),
            sample_w_qkvg_sf=torch.zeros(sf_blob_bytes(g.n_qkvg, g.d_model), dtype=torch.uint8, device=dev),
        )
    else:
        kw["quant"] = QuantSpec(descale_h=0.01, descale_w_qkvg=0.02, descale_w_o=0.03, scale_q=1.0, scale_k=2.0, scale_v=3.0, scale_o=4.0)
    return args, kw


@requires_cuda
def test_thd_declaration_accepts_both_packed_ranks_and_records_the_knobs():
    """A packed block is declared from ``[T, .]`` or ``[1, T, .]`` samples alike: internally ``batch = 1, seq_len = T``, the
    four knobs recorded, the SDPA stage built packed (``thd``, both totals, NATURAL), the dense knobs untouched."""
    for rank2 in (False, True):
        args, inp, meta = _decl_thd_args(rank2=rank2)
        with _no_device_sync():
            blk = GatedAttentionBlockFwd(*args, **_thd_kw(meta))
        assert (blk.batch, blk.seq_len) == (1, meta["t"]) and blk.thd is True
        assert (blk.num_sequences, blk.max_seq_len, blk.cu_seqlens) == (meta["b"], meta["max_seq_len"], False)
        assert blk._sdpa.thd is True and blk._sdpa.num_sequences == meta["b"] and blk._sdpa.max_seq_len == meta["max_seq_len"]
        impl = blk._sdpa._build_impl()
        assert impl.thd and impl.max_total_seq_len_q == meta["t"] and int(impl.sched_policy) == SCHED_NATURAL
        assert blk.inplace_qkv and not blk.seq_lens_present
    with _no_device_sync():
        blk_cu = GatedAttentionBlockFwd(*args, **_thd_kw(meta, cu=True))
    assert blk_cu.cu_seqlens is True and blk_cu._sdpa._build_impl().cu_seq_q_lens is True
    # A dense block records none of it.
    dense_inp = make_inputs(RefGeometry(**_COMMON), batch=2, seq_len=256)
    dense = GatedAttentionBlockFwd(
        dense_inp["h"], dense_inp["w_qkvg"], dense_inp["w_q_norm"], dense_inp["w_k_norm"], dense_inp["cos"], dense_inp["sin"], dense_inp["w_o"],
        torch.empty(2, 256, 512, device="cuda", dtype=torch.bfloat16), GatedAttentionBlockGeometry(**_COMMON),
    )  # fmt: skip
    assert dense.thd is False and dense.num_sequences is None and dense.max_seq_len is None and dense.cu_seqlens is False and dense._sdpa.thd is False


@requires_cuda
def test_thd_and_seq_lens_present_are_mutually_exclusive():
    """``thd=True`` with ``seq_lens_present=True`` is a ``ValueError`` -- under THD ``execute(seq_lens=)`` carries the
    packed lengths for both sides and there is no per-batch KV padding mask."""
    args, _inp, meta = _decl_thd_args()
    with _no_device_sync(), pytest.raises(ValueError, match="mutually exclusive"):
        GatedAttentionBlockFwd(*args, seq_lens_present=True, **_thd_kw(meta))


@requires_cuda
@pytest.mark.parametrize("mxfp8", [False, True], ids=["fp8", "mxfp8"])
def test_thd_declines_the_fully_fused_quantized_pipeline(mxfp8):
    """A FULLY FUSED quantized block (``fuse_norm_rope + fuse_gate`` under a QuantSpec / MxQuantSpec) under THD hears
    about the PIPELINE -- its gated SDPA specialization has no THD arm -- not about one knob."""
    args, kw = _decl_fp8_thd_args(mxfp8=mxfp8)
    with _no_device_sync(), pytest.raises(NotImplementedError, match="fully fused") as ei:
        GatedAttentionBlockFwd(*args, fuse_norm_rope=True, fuse_gate=True, **kw)
    assert "dense-only" in str(ei.value) and "UNFUSED" in str(ei.value)


@requires_cuda
def test_thd_declines_fuse_gate():
    """``fuse_gate=True`` is dense-only (the Rubin d256 SDPA's epilogue gate has no THD gate descriptor); the message
    names the knob to flip and that stage (5) then runs as its own launch."""
    args, _inp, meta = _decl_thd_args()
    with _no_device_sync(), pytest.raises(NotImplementedError, match="fuse_gate=True is dense-only"):
        GatedAttentionBlockFwd(*args, fuse_gate=True, **_thd_kw(meta))


@requires_cuda
def test_thd_serves_the_unfused_quantized_pipelines():
    """Both UNFUSED quantized pipelines are DECLARED under THD: per-tensor FP8 (the accept cell is
    test_thd_fp8_unfused_matches_the_fake_quant_oracle) and MXFP8 -- its three quantize stages on the PACKED arm (the SDPA row's
    per-sequence-tile-padded scale-factor layout, the packing's B recorded), the SDPA stage on the MXFP8 row's THD arm; the accept
    cells are test_block_thd_mxfp8.py's.  (This pin used to assert the MXFP8 decline; the block's MXFP8 quantize now writes the
    packed layout, so the decline is retired.)"""
    args8, kw8 = _decl_fp8_thd_args(mxfp8=False)
    with _no_device_sync():
        blk = GatedAttentionBlockFwd(*args8, **kw8)
    assert blk.thd and blk._sdpa.fp8 and blk._sdpa.pertensor and blk._sdpa.token_stride == 0 and not blk.fuse_gate
    args, kw = _decl_fp8_thd_args(mxfp8=True)
    with _no_device_sync():
        mx = GatedAttentionBlockFwd(*args, **kw)
    assert mx.thd and mx.mxfp8 and mx._sdpa.thd and mx._sdpa.mxfp8 and not mx._sdpa.pertensor and mx._sdpa.token_stride == 0 and not mx.fuse_gate
    assert all(st.packed and st.num_sequences == len(_LENS) and not st.cu_seqlens for st in (mx._quant_q, mx._quant_k, mx._quant_v))


@requires_cuda
def test_thd_requires_num_sequences_and_max_seq_len():
    """Both are REQUIRED under THD (the SDPA's unit grid, metadata and the backward's kv-blocked workspace are sized
    from them at build time); each one alone is the same typed decline."""
    args, _inp, meta = _decl_thd_args()
    for kw in (dict(thd=True), dict(thd=True, num_sequences=meta["b"]), dict(thd=True, max_seq_len=meta["max_seq_len"])):
        with _no_device_sync(), pytest.raises(ValueError, match="needs num_sequences"):
            GatedAttentionBlockFwd(*args, **kw)


@requires_cuda
def test_thd_bounds_on_num_sequences_and_max_seq_len_are_typed():
    """``num_sequences >= 1``, ``2 <= max_seq_len <= T`` and ``num_sequences * max_seq_len >= T`` -- every violation a
    ``ValueError`` quoting the bounds and the values.  The product bound is THE guard against a silent miscompute: a
    smaller product caps the SDPA chain's packed capacity below T and the tokens past it are simply not processed (the
    backward sizes delta at ceil128(cap) and the dS rows from the kv cap; the forward's units past the plan envelope never
    run, their O / LSE rows stay unwritten) -- NO message anywhere downstream."""
    args, _inp, meta = _decl_thd_args()  # T = 628
    t = meta["t"]
    for kw in (
        dict(num_sequences=0, max_seq_len=300),  # no sequence
        dict(num_sequences=3, max_seq_len=1),  # S = 1 is decode
        dict(num_sequences=3, max_seq_len=t + 1),  # longer than the packed total
        dict(num_sequences=2, max_seq_len=300),  # 2 * 300 = 600 < 628: the silent capacity cap
        dict(num_sequences=1, max_seq_len=t - 1),  # 1 * 627 < 628
    ):
        with _no_device_sync(), pytest.raises(ValueError, match="2 <= max_seq_len <= T") as ei:
            GatedAttentionBlockFwd(*args, thd=True, cu_seqlens=False, **kw)
        assert f"T={t}" in str(ei.value) and "num_sequences * max_seq_len >= T" in str(ei.value)
    # The boundary itself passes: B * S_max == T, and S_max == T with B == 1.
    with _no_device_sync():
        GatedAttentionBlockFwd(*args, thd=True, num_sequences=2, max_seq_len=314)
        GatedAttentionBlockFwd(*args, thd=True, num_sequences=1, max_seq_len=t)


@requires_cuda
def test_thd_knobs_on_a_dense_block_are_refused():
    """``num_sequences`` / ``max_seq_len`` / ``cu_seqlens`` without ``thd=True`` is a ``ValueError`` (a dense block takes
    none of them) -- never a silently ignored knob."""
    inp = make_inputs(RefGeometry(**_COMMON), batch=2, seq_len=256)
    out = torch.empty(2, 256, 512, device="cuda", dtype=torch.bfloat16)
    args = (inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, GatedAttentionBlockGeometry(**_COMMON))
    for kw in (dict(num_sequences=2), dict(max_seq_len=256), dict(cu_seqlens=True)):
        with _no_device_sync(), pytest.raises(ValueError, match="THD-only"):
            GatedAttentionBlockFwd(*args, **kw)


@requires_cuda
def test_thd_sample_ranks_are_typed():
    """Under THD ``sample_h`` is ``[T, d_model]`` or ``[1, T, d_model]`` -- a rank-4 sample, or a rank-3 one whose
    leading extent is not 1 (the dense ``[B, S, d_model]`` of a caller that forgot to pack), is a ``ValueError`` quoting
    the packed spelling; the same wording for ``cos`` / ``sin`` / ``out`` at ``check_support`` (before any arch gate)."""
    args, inp, meta = _decl_thd_args()
    h, cos, sin, out = inp["h"], inp["cos"], inp["sin"], args[7]
    t = meta["t"]
    bad_h3 = h.view(2, t // 2, -1)  # [B, S, d_model]: dense, not packed
    with _no_device_sync(), pytest.raises(ValueError, match=r"packed token matrix \[T, d_model\]"):
        GatedAttentionBlockFwd(bad_h3, *args[1:], **_thd_kw(meta))
    with _no_device_sync(), pytest.raises(ValueError, match=r"packed token matrix \[T, d_model\]"):
        GatedAttentionBlockFwd(h.view(1, 1, t, -1), *args[1:], **_thd_kw(meta))
    # cos / sin / out: the dense [B, S, .] spelling of the same rows is refused naming the tensor -- at declaration, or at
    # the latest at check_support (both are before any launch; which of the two speaks is the implementation's choice).
    for name, idx, bad in (("cos", 4, cos.view(2, t // 2, -1)), ("sin", 5, sin.view(2, t // 2, -1)), ("out", 7, out.view(2, t // 2, -1))):
        a = list(args)
        a[idx] = bad
        with _no_device_sync(), pytest.raises(ValueError, match=name):
            GatedAttentionBlockFwd(*a, **_thd_kw(meta)).check_support()


@requires_cuda
def test_thd_rejects_zero_tokens():
    """``T == 0`` (a sample with no rows) is a ``ValueError``: the SDPA adapters refuse a zero packed capacity and a GEMM
    over M = 0 has nothing to launch -- an empty step is the caller's early-out."""
    g = GatedAttentionBlockGeometry(**_COMMON)
    inp = make_inputs(RefGeometry(**_COMMON), batch=1, seq_len=8)
    z = lambda *s: torch.empty(*s, device="cuda", dtype=torch.bfloat16)  # noqa: E731
    with _no_device_sync(), pytest.raises(ValueError, match="T >= 1 packed tokens"):
        GatedAttentionBlockFwd(
            z(0, g.d_model),
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            z(0, g.rope_dim),
            z(0, g.rope_dim),
            inp["w_o"],
            z(0, g.d_model),
            g,
            thd=True,
            num_sequences=1,
            max_seq_len=2,
        )


def _declared_for_execute(geom_kw=_COMMON, lens=_LENS, *, training=False, cu=False):
    """A packed block whose execute-time ARGUMENT contracts can run on any CUDA device: the workspace carve is a pure
    function of the declaration, so ``_ws`` is set by hand (the argument checks run before any workspace byte is read)."""
    blk, inp, out, meta = _declare_thd(geom_kw, lens, cu=cu, save_for_backward=training)
    blk._ws = blk._layout()
    ws16 = torch.empty(16, dtype=torch.uint8, device="cuda")
    return blk, inp, out, meta, ws16


@requires_cuda
def test_thd_execute_requires_the_lengths_tensor():
    """``execute(seq_lens=None)`` on a packed block is a ``ValueError`` naming ``execute(seq_lens=)`` and the two forms
    -- before any launch (the SDPA builds its packed metadata from it)."""
    blk, inp, out, _meta, ws16 = _declared_for_execute()
    with _no_device_sync(), pytest.raises(ValueError, match=r"execute\(seq_lens=\) is required"):
        _execute(blk, inp, out, ws16, seq_lens=None)


@requires_cuda
@pytest.mark.parametrize("cu", [False, True], ids=["lengths", "prefix"])
def test_thd_execute_validates_the_lengths_tensor(cu):
    """The lengths tensor must be a contiguous 1-D int32 CUDA tensor on ``h``'s device with EXACTLY ``B`` (lengths) or
    ``B+1`` (prefix sums) elements -- dtype, rank, contiguity, element count (the OTHER form's count included: a
    ``cu_seqlens=True`` block refuses a ``[B]`` tensor and vice versa) and device are each a ``ValueError`` quoting the
    expected count, host-side (no value is ever read)."""
    blk, inp, out, meta, ws16 = _declared_for_execute(cu=cu)
    b = meta["b"]
    n_ok = b + 1 if cu else b
    n_other = b if cu else b + 1
    good = _lens(meta, cu)
    bad = {
        "dtype": good.to(torch.int64),
        "rank": good.view(1, n_ok),
        "contiguous": torch.zeros(2 * n_ok, dtype=torch.int32, device="cuda")[::2],
        "count (the other form)": _lens(meta, not cu),
        "count": torch.zeros(n_ok + 3, dtype=torch.int32, device="cuda"),
        "device": good.cpu(),
    }
    for why, ten in bad.items():
        if why == "count (the other form)":
            assert ten.numel() == n_other
        with _no_device_sync(), pytest.raises(ValueError, match="contiguous 1-D int32") as ei:
            _execute(blk, inp, out, ws16, seq_lens=ten)
        assert f"{n_ok} elements" in str(ei.value), (why, str(ei.value))


@requires_cuda
def test_thd_saved_seq_lens_is_required_and_the_record_states_its_form():
    """The record's lengths identity and its packing fact, through ``_check_saved_set`` on a declared training block (no launch, no device
    read): ``saved.seq_lens`` is REQUIRED under THD and must be the very tensor ``execute`` runs with (``is``) -- a ``None``
    or a clone names ``saved.seq_lens`` and says it is required; ``saved.seq_lens_form`` must say the block's form
    (``"lengths"`` / ``"prefix"`` per ``cu_seqlens``) -- ``None`` (the padded dense record's value) or the other form is a
    ``ValueError`` naming the field, and a DENSE block refuses a record that claims a packed form."""
    for cu in (False, True):
        blk, inp, out, meta = _declare_thd(_COMMON, _LENS, cu=cu, save_for_backward=True)
        lens = _lens(meta, cu)
        good = _alloc_packed_saved(blk.geom, inp, meta, seq_lens=lens, form=_form(cu))
        with _no_device_sync():
            bound = blk._check_saved_set(inp["h"], lens, None, good)
            assert bound.lse is good.lse and bound.proj.data_ptr() == good.proj_slab.data_ptr()
            with pytest.raises(ValueError, match=r"saved\.seq_lens") as ei:
                blk._check_saved_set(inp["h"], lens, None, dataclasses.replace(good, seq_lens=None))
            assert "REQUIRED" in str(ei.value)
            with pytest.raises(ValueError, match=r"saved\.seq_lens"):
                blk._check_saved_set(inp["h"], lens, None, dataclasses.replace(good, seq_lens=lens.clone()))
            with pytest.raises(ValueError, match=r"seq_lens_form"):
                blk._check_saved_set(inp["h"], lens, None, dataclasses.replace(good, seq_lens_form=None))
            with pytest.raises(ValueError, match=r"seq_lens_form"):
                blk._check_saved_set(inp["h"], lens, None, dataclasses.replace(good, seq_lens_form=_form(not cu)))
    # A dense training block refuses a record that claims a packed form (its seq_lens, if any, is a padding mask: form None).
    dinp = make_inputs(RefGeometry(**_COMMON), batch=2, seq_len=256)
    dense = GatedAttentionBlockFwd(
        dinp["h"], dinp["w_qkvg"], dinp["w_q_norm"], dinp["w_k_norm"], dinp["cos"], dinp["sin"], dinp["w_o"],
        torch.empty(2, 256, 512, device="cuda", dtype=torch.bfloat16), GatedAttentionBlockGeometry(**_COMMON), save_for_backward=True, seq_lens_present=True,
    )  # fmt: skip
    pad = torch.tensor([256, 128], dtype=torch.int32, device="cuda")
    g = dense.geom
    proj_slab = torch.empty(2 * 256, g.n_qkvg, device="cuda", dtype=torch.bfloat16)
    q_pre, gate, k_pre, _v = saved_slab_views(proj_slab, g, 2, 256)
    rec = dict(
        h=dinp["h"], gate=gate, q_pre=q_pre, k_pre=k_pre, proj_slab=proj_slab,
        o=torch.empty(2, 256, g.h_q, g.d_head, device="cuda", dtype=torch.bfloat16), lse=torch.empty(2, g.h_q, 256, device="cuda", dtype=torch.float32),
        rstd_q=torch.empty(2, 256, g.h_q, device="cuda", dtype=torch.float32), rstd_k=torch.empty(2, 256, g.h_kv, device="cuda", dtype=torch.float32), seq_lens=pad,
    )  # fmt: skip
    with _no_device_sync():
        dense._check_saved_set(dinp["h"], pad, None, SavedForBackward(**rec, seq_lens_form=None))  # the padded dense record
        with pytest.raises(ValueError, match="seq_lens_form"):
            dense._check_saved_set(dinp["h"], pad, None, SavedForBackward(**rec, seq_lens_form="lengths"))


@requires_cuda
def test_thd_workspace_carve_equals_dense_b1_except_engine_scratch():
    """The packed block's workspace carve IS the dense ``B=1, S=T`` block's, field by field, in every save / in-place
    mode -- no new slot (only the SDPA's own scratch, added at ``get_workspace_size`` after compile, differs).  Declared
    blocks, any CUDA device."""
    for kw in (dict(), dict(inplace_qkv=False), dict(save_for_backward=True), dict(save_for_backward=True, saved_gate_copy=True)):
        blk, inp, _out, meta = _declare_thd(_COMMON, _LENS, **kw)
        dense_inp = make_inputs(RefGeometry(**_COMMON), batch=1, seq_len=meta["t"])
        dense = GatedAttentionBlockFwd(
            dense_inp["h"], dense_inp["w_qkvg"], dense_inp["w_q_norm"], dense_inp["w_k_norm"], dense_inp["cos"], dense_inp["sin"], dense_inp["w_o"],
            torch.empty(1, meta["t"], 512, device="cuda", dtype=torch.bfloat16), GatedAttentionBlockGeometry(**_COMMON), **kw,
        )  # fmt: skip
        assert dataclasses.asdict(blk._layout()) == dataclasses.asdict(dense._layout()), kw
        assert [st.name for st in blk._stages] == [st.name for st in dense._stages], kw


# ---------------------------------------------------------------------------
# Accept -- Rubin only
# ---------------------------------------------------------------------------


@requires_rubin
@pytest.mark.parametrize("arm", ["inplace", "out_of_place", "fp16"])
def test_thd_matches_the_per_sequence_oracle(arm):
    """``(300, 128, 200)``, causal, QK-norm: the packed block against the per-sequence fp32 oracle -- in-place bf16
    inference (the default), out-of-place bf16 TRAINING (the record every tensor checked), and the fp16 forward.  The
    record / output are sentinel-filled first: no survivor anywhere (every row belongs to a sequence under sum == T)."""
    kw = dict(inplace=dict(), out_of_place=dict(training=True), fp16=dict(dtype=torch.float16))[arm]
    res = _run_thd(_COMMON, _LENS, sentinel=_SENTINEL, **kw)
    assert res.blk.thd and res.blk.batch == 1 and res.blk.seq_len == res.meta["t"]
    if arm == "inplace":
        assert res.blk.inplace_qkv and res.blk._sdpa.token_stride == res.blk.geom.n_qkvg
    if arm == "fp16":
        assert res.out.dtype == torch.float16
    _assert_empty_sequences_leave_no_rows(res)
    _check_forward_per_sequence(res)


@requires_rubin
def test_thd_rank2_inputs_are_the_rank3_block_bitwise():
    """The same packing handed over as ``[T, .]`` and as ``[1, T, .]`` gives bitwise the same ``out`` and record."""
    a = _run_thd(_COMMON, _LENS, training=True, rank2=False, refs=False)
    b = _run_thd(_COMMON, _LENS, training=True, rank2=True, refs=False)
    assert b.out.dim() == 2 and b.saved.h.dim() == 2
    assert torch.equal(_rows(a.out), _rows(b.out))
    for f in ("o", "lse", "proj_slab", "rstd_q", "rstd_k"):
        assert torch.equal(getattr(a.saved, f), getattr(b.saved, f)), f


@requires_rubin
def test_thd_dense_mask_arm():
    """The dense (no-mask) arm under THD: sequences whose length is not a 128-multiple are served (the packed lengths bound
    every KV tail; the dense ``S % 128`` tail rule is a dense-declaration rule), checked per sequence."""
    res = _run_thd({**_COMMON, "is_causal": False}, _LENS, training=True, sentinel=_SENTINEL)
    _assert_empty_sequences_leave_no_rows(res)
    _check_forward_per_sequence(res)


@requires_rubin
def test_thd_rope_only():
    """``qk_norm=False`` under THD: RoPE-only Q/K with per-token tables, no rstd in the record."""
    res = _run_thd({**_COMMON, "qk_norm": False}, _LENS, training=True, sentinel=_SENTINEL)
    assert res.saved.rstd_q is None and res.blk._norm_rope.want_rstd is False
    _check_forward_per_sequence(res)


@requires_rubin
@pytest.mark.parametrize("arm", ["bottom_right", "swa"])
def test_thd_mask_arms(arm):
    """Bottom-right causal (the diagonal anchored per SEQUENCE: at equal Q / KV lengths the plain-causal band) and a
    sliding window of 640 at lengths ``(900, 700)`` with ``S_max = 1024`` -- correctness only, per sequence through the
    oracle's own mask arms."""
    if arm == "bottom_right":
        res = _run_thd({**_COMMON, "is_causal": True, "causal_bottom_right": True}, _LENS, training=True)
        assert res.blk._sdpa._impl.causal_bottom_right
    else:
        res = _run_thd({**_COMMON, "is_causal": True, "window_left": 640}, (900, 700), training=True, max_seq_len=1024)
    _check_forward_per_sequence(res)


@requires_rubin
@pytest.mark.parametrize("h_kv", [2, 1, 8], ids=["gqa_8_2", "mqa", "mha"])
def test_thd_gqa(h_kv):
    """The three GQA fold paths under THD -- 8/2, MQA (``h_kv = 1``, group 8) and MHA (``h_kv = 8``, no fold) -- the fold is
    the SDPA adapter's, from the ``(B, H_q, .)`` / ``(B, H_kv, .)`` envelope descriptors the block declares."""
    res = _run_thd({**_COMMON, "h_kv": h_kv}, _LENS, training=True, sentinel=_SENTINEL)
    _assert_empty_sequences_leave_no_rows(res)
    _check_forward_per_sequence(res)


@requires_rubin
@pytest.mark.parametrize(
    "lens, s_max",
    [((256, 0, 128), 256), ((0, 256, 128), 256), ((5, 0, 0), 5)],
    ids=["middle_empty", "first_empty", "trailing_empties_T5"],
)
def test_thd_zero_length_sequence(lens, s_max):
    """Zero-length sequences -- in the middle, first, and two trailing ones with a 5-token survivor (the "B varies, pad with
    empties" use case, T close to B): an empty sequence owns no rows, its neighbours' rows are exact, the sentinel-filled
    record and output carry no survivor.  Never a 1-token sequence (the oracle's 1x1 softmax dies first)."""
    res = _run_thd(_COMMON, lens, training=True, sentinel=_SENTINEL, max_seq_len=s_max)
    assert res.meta["lens"] == list(lens) and 0 in res.meta["lens"]
    _assert_empty_sequences_leave_no_rows(res)
    _check_forward_per_sequence(res)


@requires_rubin
def test_thd_ragged_envelope():
    """``(130, 5)`` with ``S_max = 130`` (not a 128-multiple, a 5-token tail tile): the plan envelope ``ceil(S_max / tile)``
    units per (sequence, head) with dead units past the live total, checked per sequence."""
    res = _run_thd(_COMMON, (130, 5), training=True, sentinel=_SENTINEL)
    _assert_empty_sequences_leave_no_rows(res)
    _check_forward_per_sequence(res)


@requires_rubin
def test_thd_rows_past_the_live_total_are_untouched_by_the_sdpa():
    """What the SDPA PROMISES when ``sum(lengths) < T`` (a contract violation on the whole block, documented as such): it
    writes nothing past the live total -- ``saved.o`` rows and ``saved.lse`` columns ``[500, 640)`` keep their sentinel,
    the live rows match the per-sequence oracle, ``out[:500]`` is exact per sequence.  ``out[500:]`` is whatever the gate
    and the out projection make of sentinel rows (the token-wise stages run over all T rows) -- not asserted, and the
    reason the backward's weight gradients would be contaminated: the contract is ``sum == T``.  Built without the
    packing helpers (which assert the contract); declared ``num_sequences=2, max_seq_len=320`` so ``B * S_max >= T``."""
    geom_kw, t_decl, lens = _COMMON, 640, (300, 200)
    live = sum(lens)
    g = GatedAttentionBlockGeometry(**geom_kw)
    inp = make_inputs(RefGeometry(**geom_kw), batch=1, seq_len=t_decl)
    cos, sin = packed_rope_tables(lens + (t_decl - live,), g.rope_dim, base=_ROPE_BASE)  # per-token tables; the slack rows get positions too
    inp["cos"], inp["sin"] = cos.view(1, t_decl, g.rope_dim), sin.view(1, t_decl, g.rope_dim)
    out = torch.full((1, t_decl, g.d_model), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    blk = GatedAttentionBlockFwd(
        inp["h"],
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        out,
        g,
        save_for_backward=True,
        thd=True,
        num_sequences=2,
        max_seq_len=320,
    )
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    meta = dict(t=t_decl, b=2, lens=list(lens))
    saved = _alloc_packed_saved(g, inp, meta, seq_lens=seq_lens, form="lengths", sentinel=TAIL_SENTINEL)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _execute(blk, inp, out, ws, seq_lens=seq_lens, saved=saved)
    torch.cuda.synchronize()
    assert (saved.o[0, live:] == TAIL_SENTINEL).all(), f"{int((saved.o[0, live:] != TAIL_SENTINEL).sum())} O cells past the live total were written"
    assert (saved.lse[0, :, live:] == TAIL_SENTINEL).all(), "LSE columns past the live total were written"
    assert not (saved.o[0, :live] == TAIL_SENTINEL).any() and not (saved.lse[0, :, :live] == TAIL_SENTINEL).any(), "live rows were not written"
    refs = gated_attention_block_reference_packed(
        inp["h"][:, :live],
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"][:, :live],
        inp["sin"][:, :live],
        inp["w_o"],
        RefGeometry(**geom_kw),
        lens,
    )
    res = SimpleNamespace(blk=blk, inp=inp, out=out[:, :live], meta=dict(meta, t=live), saved=saved, refs=refs)
    # The bands / rstd checks index the record at the live rows only; the slab view below is over the DECLARED T.
    g2 = res.blk.geom

    def check(i, lo, hi, ref):
        assert _cos(out[0, lo:hi], ref.out[0]) > 0.999, "out"
        torch.testing.assert_close(saved.o[0, lo:hi].float(), ref.o[0].float(), rtol=0, atol=2e-2)
        torch.testing.assert_close(saved.lse[0, :, lo:hi], ref.lse[0], rtol=0, atol=2e-2)
        torch.testing.assert_close(saved.q_pre[0, lo:hi], ref.q_pre[0], rtol=2**-7, atol=1e-3)

    failures = compare_packed(refs, lens, check)
    assert not failures, "\n".join(failures)
    assert g2.n_qkvg == saved.proj_slab.shape[1]


@requires_rubin
def test_thd_b1_is_bitwise_the_dense_block():
    """``B = 1`` packed ``(512,)`` against the dense ``B=1, S=512`` training block over the SAME bytes: ``out`` and every
    record tensor BITWISE -- one sequence over the same tiles; a difference here is a finding, never a tolerance."""
    t = 512
    res = _run_thd(_COMMON, (t,), training=True, refs=False)
    dinp = make_inputs(RefGeometry(**_COMMON), batch=1, seq_len=t)
    for k in ("h", "w_qkvg", "w_o", "cos", "sin"):
        assert torch.equal(dinp[k], res.inp[k]), k
    g = res.blk.geom
    out_d = torch.empty(1, t, g.d_model, device="cuda", dtype=torch.bfloat16)
    dense = GatedAttentionBlockFwd(
        dinp["h"], dinp["w_qkvg"], dinp["w_q_norm"], dinp["w_k_norm"], dinp["cos"], dinp["sin"], dinp["w_o"], out_d, g, save_for_backward=True
    )
    proj_slab = torch.empty(t, g.n_qkvg, device="cuda", dtype=torch.bfloat16)
    q_pre, gate, k_pre, _v = saved_slab_views(proj_slab, g, 1, t)
    saved_d = SavedForBackward(
        h=dinp["h"], gate=gate, q_pre=q_pre, k_pre=k_pre, proj_slab=proj_slab,
        o=torch.empty(1, t, g.h_q, g.d_head, device="cuda", dtype=torch.bfloat16), lse=torch.empty(1, g.h_q, t, device="cuda", dtype=torch.float32),
        rstd_q=torch.empty(1, t, g.h_q, device="cuda", dtype=torch.float32), rstd_k=torch.empty(1, t, g.h_kv, device="cuda", dtype=torch.float32),
    )  # fmt: skip
    dense.check_support()
    dense.compile()
    ws = torch.empty(dense.get_workspace_size(), dtype=torch.uint8, device="cuda")
    dense.execute(dinp["h"], dinp["w_qkvg"], dinp["w_q_norm"], dinp["w_k_norm"], dinp["cos"], dinp["sin"], dinp["w_o"], out_d, ws, saved=saved_d)
    torch.cuda.synchronize()
    for name, a, b in (
        ("proj_slab", res.saved.proj_slab, saved_d.proj_slab),
        ("rstd_q", res.saved.rstd_q, saved_d.rstd_q),
        ("rstd_k", res.saved.rstd_k, saved_d.rstd_k),
    ):
        assert torch.equal(a, b), f"{name}: the token-wise stages differ between packed and dense"
    for name, a, b in (("saved.o", res.saved.o, saved_d.o), ("saved.lse", res.saved.lse, saved_d.lse), ("out", res.out, out_d)):
        d = (a.float() - b.float()).abs().max().item()
        assert torch.equal(a, b), f"{name}: packed B=1 differs from the dense block (max|diff| {d:.3e}) -- a finding, not a tolerance"


@requires_rubin
def test_thd_uniform_b4_matches_the_dense_block_per_sequence():
    """Uniform ``B = 4`` packed ``(256,)*4`` against the dense ``B=4, S=256`` block over the SAME bytes: the token-wise
    stages (``proj_slab``, ``rstd``) BITWISE; the SDPA-derived ``O`` / ``LSE`` / ``out`` per sequence at the dense suite's
    tolerance (the packed SDPA walks the tiles differently and runs the padded-mask arm; the first differing stage is
    printed).  Both also against the oracle."""
    s, b = 256, 4
    res = _run_thd(_COMMON, (s,) * b, training=True)
    dinp = make_inputs(RefGeometry(**_COMMON), batch=b, seq_len=s)
    assert torch.equal(dinp["h"].reshape(1, b * s, -1), res.inp["h"]) and torch.equal(dinp["cos"].reshape(1, b * s, -1), res.inp["cos"])
    g = res.blk.geom
    out_d = torch.empty(b, s, g.d_model, device="cuda", dtype=torch.bfloat16)
    dense = GatedAttentionBlockFwd(
        dinp["h"], dinp["w_qkvg"], dinp["w_q_norm"], dinp["w_k_norm"], dinp["cos"], dinp["sin"], dinp["w_o"], out_d, g, save_for_backward=True
    )
    proj_slab = torch.empty(b * s, g.n_qkvg, device="cuda", dtype=torch.bfloat16)
    q_pre, gate, k_pre, _v = saved_slab_views(proj_slab, g, b, s)
    saved_d = SavedForBackward(
        h=dinp["h"], gate=gate, q_pre=q_pre, k_pre=k_pre, proj_slab=proj_slab,
        o=torch.empty(b, s, g.h_q, g.d_head, device="cuda", dtype=torch.bfloat16), lse=torch.empty(b, g.h_q, s, device="cuda", dtype=torch.float32),
        rstd_q=torch.empty(b, s, g.h_q, device="cuda", dtype=torch.float32), rstd_k=torch.empty(b, s, g.h_kv, device="cuda", dtype=torch.float32),
    )  # fmt: skip
    dense.check_support()
    dense.compile()
    ws = torch.empty(dense.get_workspace_size(), dtype=torch.uint8, device="cuda")
    dense.execute(dinp["h"], dinp["w_qkvg"], dinp["w_q_norm"], dinp["w_k_norm"], dinp["cos"], dinp["sin"], dinp["w_o"], out_d, ws, saved=saved_d)
    torch.cuda.synchronize()
    assert torch.equal(res.saved.proj_slab, saved_d.proj_slab), "proj_slab differs: the token-wise stage-(1) GEMM is not the same launch"
    assert torch.equal(res.saved.rstd_q, saved_d.rstd_q.reshape(1, b * s, -1)) and torch.equal(res.saved.rstd_k, saved_d.rstd_k.reshape(1, b * s, -1))
    first_diff = None
    for i, (lo, hi) in enumerate(sequence_slices(res.meta["lens"])):
        for name, a, d in (
            ("saved.o", res.saved.o[0, lo:hi], saved_d.o[i]),
            ("saved.lse", res.saved.lse[0, :, lo:hi], saved_d.lse[i]),
            ("out", res.out[0, lo:hi], out_d[i]),
        ):
            md = (a.float() - d.float()).abs().max().item()
            if md and first_diff is None:
                first_diff = (i, name, md)
            if name == "out":
                assert _cos(a, d) > 0.999, f"seq {i} out vs dense: cos {_cos(a, d):.6f}"
            else:
                torch.testing.assert_close(a.float(), d.float(), rtol=0, atol=2e-2, msg=f"seq {i} {name} vs dense (max|diff| {md:.3e})")
    print(f"\nuniform B=4 vs dense: first SDPA-derived difference {first_diff} (None = bitwise)")
    _check_forward_per_sequence(res)


@requires_rubin
@pytest.mark.parametrize("cu_base", [0, 100], ids=["prefix", "prefix_nonzero_base"])
def test_thd_lengths_and_prefix_forms_are_bitwise(cu_base):
    """The same packing run as ``[B]`` lengths and as ``[B+1]`` prefix sums (at base 0, and sliced from a larger prefix
    at base 100 -- the kernels normalize to the first entry) gives ``torch.equal`` ``out`` and every record tensor."""
    a = _run_thd(_COMMON, _LENS, training=True, refs=False)
    blk, inp, out, meta = _declare_thd(_COMMON, _LENS, cu=True, save_for_backward=True)
    cu = torch.tensor(cu_seqlens_of(meta["lens"], base=cu_base), dtype=torch.int32, device="cuda")
    assert_packing_contract(cu, meta["t"], meta["max_seq_len"], meta["b"], cu=True)
    saved = _alloc_packed_saved(blk.geom, inp, meta, seq_lens=cu, form="prefix")
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _execute(blk, inp, out, ws, seq_lens=cu, saved=saved)
    torch.cuda.synchronize()
    assert torch.equal(a.out, out)
    for f in ("o", "lse", "proj_slab", "rstd_q", "rstd_k"):
        assert torch.equal(getattr(a.saved, f), getattr(saved, f)), f


@requires_rubin
def test_thd_lse_is_head_major_with_head_stride_T():
    """On the device: the forward writes ``saved.lse`` as a contiguous ``[1, H_q, T]`` -- head-major, head stride T --
    into a ``+inf``-poisoned tensor: every cell finite afterwards, ``saved.lse[0, :, lo:hi]`` equal to the per-sequence
    oracle LSE at the SDPA stage test's ``atol 2e-2``, and the dead-row / empty cells of a zero-length sequence absent (no
    rows).  The declared descriptor is the head-major one."""
    res = _run_thd(_COMMON, (300, 0, 200, 128), training=True, max_seq_len=300)
    g, t = res.blk.geom, res.meta["t"]
    impl = res.blk._sdpa._impl
    assert thd_stats_packing(impl.lse_desc.stride[1], impl.lse_desc.stride[2], g.h_q) == "head_major" and impl.lse_desc.stride[1] == t
    res.saved.lse.fill_(float("inf"))
    _execute(res.blk, res.inp, res.out, res.ws, seq_lens=res.seq_lens, saved=res.saved)
    torch.cuda.synchronize()
    assert res.saved.lse.shape == (1, g.h_q, t) and res.saved.lse.is_contiguous()
    assert torch.isfinite(res.saved.lse).all(), f"{int((~torch.isfinite(res.saved.lse)).sum())} LSE cells were never written"

    def check(i, lo, hi, ref):
        torch.testing.assert_close(res.saved.lse[0, :, lo:hi], ref.lse[0], rtol=0, atol=2e-2)

    failures = compare_packed(res.refs, res.meta["lens"], check)
    assert not failures, "\n".join(failures)


@requires_rubin
def test_thd_training_record_round_trips_into_the_backward():
    """The record contract, forward half: the packed training record -- every tensor at ``(1, T)``, ``saved.seq_lens`` the lengths
    tensor itself with its form -- is accepted by the backward's declaration (``GatedAttentionBlockBwd(thd=True, ...)``
    passes ``check_support``); the gradients themselves are the backward module's cells.  Also the gate-copy save mode's
    forward half: the compact ``saved.gate`` equals the proj_slab run's GATE band bitwise under THD."""
    from cudnn.gated_attention_block import GatedAttentionBlockBwd

    res = _run_thd(_COMMON, _LENS, training=True)
    g, t, meta = res.blk.geom, res.meta["t"], res.meta
    assert res.saved.seq_lens is res.seq_lens and res.saved.seq_lens_form == "lengths"
    dy = torch.randn(1, t, g.d_model, device="cuda").to(torch.bfloat16)
    bwd = GatedAttentionBlockBwd(
        dy, res.saved, res.inp["w_qkvg"], res.inp["w_q_norm"], res.inp["w_k_norm"], res.inp["cos"], res.inp["sin"], res.inp["w_o"], g, **_thd_kw(meta)
    )
    assert bwd.check_support()
    assert bwd.thd and (bwd.batch, bwd.seq_len) == (1, t)
    # gate-copy save mode under THD: the GATE band copied out, bitwise the slab run's.
    gc = _declare_thd(_COMMON, _LENS, save_for_backward=True, saved_gate_copy=True)
    blk, inp, out, meta2 = gc
    saved_gc = SavedForBackward(
        h=inp["h"], gate=torch.empty(1, t, g.h_q, g.d_head, device="cuda", dtype=torch.bfloat16), q_pre=None, k_pre=None, proj_slab=None,
        o=torch.empty(1, t, g.h_q, g.d_head, device="cuda", dtype=torch.bfloat16), lse=torch.empty(1, g.h_q, t, device="cuda", dtype=torch.float32),
        rstd_q=torch.empty(1, t, g.h_q, device="cuda", dtype=torch.float32), rstd_k=torch.empty(1, t, g.h_kv, device="cuda", dtype=torch.float32),
        seq_lens=meta2["seq_lens"], seq_lens_form="lengths",
    )  # fmt: skip
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    _execute(blk, inp, out, ws, seq_lens=meta2["seq_lens"], saved=saved_gc)
    torch.cuda.synchronize()
    assert torch.equal(saved_gc.gate, res.saved.gate) and torch.equal(saved_gc.o, res.saved.o) and torch.equal(saved_gc.lse, res.saved.lse)
    assert torch.equal(out, res.out)


@requires_rubin
def test_thd_two_runs_are_bitwise():
    """Two executes with the workspace POISONED between (0xFF = bf16 NaN) and the record / output NaN-filled give bitwise
    equal results -- no region depends on prior workspace content (the SDPA's packed metadata / descriptor scratch
    included)."""
    res = _run_thd(_COMMON, _LENS, training=True, refs=False)
    res.ws.fill_(0xFF)
    out2 = torch.full_like(res.out, float("nan"))
    saved2 = _alloc_packed_saved(res.blk.geom, res.inp, res.meta, seq_lens=res.seq_lens, form="lengths", sentinel=float("nan"))
    _execute(res.blk, res.inp, out2, res.ws, seq_lens=res.seq_lens, saved=saved2)
    torch.cuda.synchronize()
    assert torch.isfinite(out2.float()).all() and torch.equal(out2, res.out)
    for f in ("o", "lse", "proj_slab", "rstd_q", "rstd_k"):
        assert torch.equal(getattr(saved2, f), getattr(res.saved, f)), f


@requires_rubin
def test_thd_cuda_graph_replay_with_new_lengths():
    """One ``execute`` captured into a CUDA graph replays bitwise the eager run, and -- the lengths being device data the
    SDPA reads at replay (the grid is the plan-time envelope) -- a replay over NEW lengths written through the captured
    pointers (same ``B``, ``sum == T``, every length ``<= max_seq_len``; the per-token RoPE tables rewritten for the new
    packing) equals a fresh eager run over them.  The capture itself launches nothing."""
    res = _run_thd(_COMMON, _LENS, training=True, refs=False)
    blk, inp, meta, g = res.blk, res.inp, res.meta, res.blk.geom
    seq_lens = res.seq_lens  # the captured pointer
    ws = torch.empty_like(res.ws)
    out = torch.full_like(res.out, float("nan"))
    saved = _alloc_packed_saved(g, inp, meta, seq_lens=seq_lens, form="lengths", sentinel=float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _execute(blk, inp, out, ws, seq_lens=seq_lens, saved=saved)  # warm-up on the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    out.fill_(float("nan"))
    saved.o.fill_(float("nan"))
    saved.lse.fill_(float("nan"))
    ws.fill_(0xFF)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            _execute(blk, inp, out, ws, seq_lens=seq_lens, saved=saved)
        torch.cuda.synchronize()
        assert torch.isnan(out).all() and torch.isnan(saved.o).all(), "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        assert (
            torch.equal(out, res.out) and torch.equal(saved.o, res.saved.o) and torch.equal(saved.lse, res.saved.lse)
        ), "the replay differs from the eager run"
        # NEW lengths through the captured pointers: a permutation of the packing (same B, sum == T, each <= S_max).
        new_lens = [200, 300, 128]
        assert_packing_contract(new_lens, meta["t"], meta["max_seq_len"], meta["b"])
        cos2, sin2 = packed_rope_tables(new_lens, g.rope_dim, base=_ROPE_BASE)
        inp["cos"].copy_(cos2.view_as(inp["cos"]))
        inp["sin"].copy_(sin2.view_as(inp["sin"]))
        seq_lens.copy_(torch.tensor(new_lens, dtype=torch.int32, device="cuda"))
        ws.fill_(0xFF)
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        ref_out = torch.full_like(out, float("nan"))
        ref_saved = _alloc_packed_saved(g, inp, meta, seq_lens=seq_lens, form="lengths", sentinel=float("nan"))
        ws_ref = torch.empty_like(ws).fill_(0xFF)
        _execute(blk, inp, ref_out, ws_ref, seq_lens=seq_lens, saved=ref_saved)
        torch.cuda.synchronize()
        assert torch.isfinite(out.float()).all() and torch.equal(out, ref_out), "the replay over new lengths differs from eager"
        assert torch.equal(saved.o, ref_saved.o) and torch.equal(saved.lse, ref_saved.lse)
        refs = gated_attention_block_reference_packed(
            inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], RefGeometry(**_COMMON), new_lens
        )
        failures = compare_packed(refs, new_lens, lambda i, lo, hi, ref: torch.testing.assert_close(saved.lse[0, :, lo:hi], ref.lse[0], rtol=0, atol=2e-2))
        assert not failures, "\n".join(failures)
    finally:
        graph.reset()


@requires_rubin
def test_thd_workspace_size_is_honest():
    """``get_workspace_size()`` is exact and never exceeded: a buffer 4096 B larger keeps its tail untouched, two executes
    allocate nothing, the result over a sentinel-filled buffer is bitwise the first run's; the SDPA's packed scratch is
    non-zero and reported by the stage; one byte less is a typed ``ValueError``."""
    res = _run_thd(_COMMON, _LENS, training=True, refs=False)
    blk = res.blk
    size = blk.get_workspace_size()
    lay = blk._layout()
    sdpa_scratch = blk._sdpa.scratch_workspace_bytes()
    print(f"\nworkspace {size} B; carve {lay.total_bytes} B; SDPA packed scratch {sdpa_scratch} B")
    assert sdpa_scratch > 0, "the packed SDPA carves metadata / descriptor scratch (the dense '0 bytes' claim does not hold under THD)"
    assert size >= lay.total_bytes + sdpa_scratch and size % 256 == 0
    ws = torch.full((size + 4096,), 0xAB, dtype=torch.uint8, device="cuda")
    out = torch.empty_like(res.out)
    saved = _alloc_packed_saved(blk.geom, res.inp, res.meta, seq_lens=res.seq_lens, form="lengths")
    _execute(blk, res.inp, out, ws, seq_lens=res.seq_lens, saved=saved)
    torch.cuda.synchronize()
    # The allocation pin in the caching allocator's COUNTER form (test_block_training_forward.py): the cumulative allocation
    # count cannot be lowered by an unrelated release and still rises for a temporary the execute frees before returning; the
    # allocator peak is the second witness for such a temporary's bytes.  Every object the execute reads stays alive across it.
    gc.collect()
    live = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    n0 = torch.cuda.memory_stats()["allocation.all.allocated"]
    _execute(blk, res.inp, out, ws, seq_lens=res.seq_lens, saved=saved)
    _execute(blk, res.inp, out, ws, seq_lens=res.seq_lens, saved=saved)
    torch.cuda.synchronize()
    n1 = torch.cuda.memory_stats()["allocation.all.allocated"]
    peak = torch.cuda.max_memory_allocated()
    assert n1 == n0, f"the packed forward made {n1 - n0} CUDA allocation(s) on the execute path (allocation.all.allocated {n0} -> {n1})"
    assert peak <= live, f"a temporary on the packed forward's execute path: the allocator peak rose from {live} to {peak} bytes"
    assert torch.equal(ws[size:], torch.full((4096,), 0xAB, dtype=torch.uint8, device="cuda")), "bytes past get_workspace_size() were written"
    assert torch.equal(out, res.out) and torch.equal(saved.o, res.saved.o) and torch.equal(saved.lse, res.saved.lse)
    with pytest.raises(ValueError, match="workspace"):
        _execute(blk, res.inp, out, ws[: size - 1], seq_lens=res.seq_lens, saved=saved)


def _cupti_kernels(fn):
    """The CUDA kernel names of ONE call of ``fn`` (warm), via torch.profiler; None when CUPTI records nothing."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if not names:
        return None, None
    kernels = [n for n in names if "memset" not in n.lower() and "memcpy" not in n.lower()]
    return kernels, [n for n in names if n not in kernels]


@requires_rubin
def test_thd_launch_count_is_dense_plus_one():
    """CUPTI kernel records of one packed execute == the dense ``B=1, S=T`` block's + 1: the packed SDPA issues its
    metadata / descriptor SETUP launch before the main kernel; every other stage is the same launch.  No hidden memcpy.
    MEASURED, the names printed; a typed skip when CUPTI records nothing on this node."""
    res = _run_thd(_COMMON, _LENS, training=True, refs=False)
    t, g = res.meta["t"], res.blk.geom
    dinp = make_inputs(RefGeometry(**_COMMON), batch=1, seq_len=t)
    out_d = torch.empty(1, t, g.d_model, device="cuda", dtype=torch.bfloat16)
    dense = GatedAttentionBlockFwd(
        dinp["h"], dinp["w_qkvg"], dinp["w_q_norm"], dinp["w_k_norm"], dinp["cos"], dinp["sin"], dinp["w_o"], out_d, g, save_for_backward=True
    )
    proj_slab = torch.empty(t, g.n_qkvg, device="cuda", dtype=torch.bfloat16)
    q_pre, gate, k_pre, _v = saved_slab_views(proj_slab, g, 1, t)
    saved_d = SavedForBackward(
        h=dinp["h"], gate=gate, q_pre=q_pre, k_pre=k_pre, proj_slab=proj_slab,
        o=torch.empty(1, t, g.h_q, g.d_head, device="cuda", dtype=torch.bfloat16), lse=torch.empty(1, g.h_q, t, device="cuda", dtype=torch.float32),
        rstd_q=torch.empty(1, t, g.h_q, device="cuda", dtype=torch.float32), rstd_k=torch.empty(1, t, g.h_kv, device="cuda", dtype=torch.float32),
    )  # fmt: skip
    dense.check_support()
    dense.compile()
    ws_d = torch.empty(dense.get_workspace_size(), dtype=torch.uint8, device="cuda")
    k_thd, m_thd = _cupti_kernels(lambda: _execute(res.blk, res.inp, res.out, res.ws, seq_lens=res.seq_lens, saved=res.saved))
    k_dense, m_dense = _cupti_kernels(
        lambda: dense.execute(dinp["h"], dinp["w_qkvg"], dinp["w_q_norm"], dinp["w_k_norm"], dinp["cos"], dinp["sin"], dinp["w_o"], out_d, ws_d, saved=saved_d)
    )
    if k_thd is None or k_dense is None:
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the launch count is unverified here")
    print(f"\npacked: {len(k_thd)} kernels + {len(m_thd)} memset/memcpy:\n  " + "\n  ".join(k_thd))
    print(f"dense:  {len(k_dense)} kernels + {len(m_dense)} memset/memcpy:\n  " + "\n  ".join(k_dense))
    assert not [n for n in m_thd if "memcpy" in n.lower()], "a hidden copy on the packed execute path"
    assert len(k_thd) == len(k_dense) + 1, (len(k_thd), len(k_dense))


@requires_rubin
def test_thd_fp8_unfused_matches_the_fake_quant_oracle():
    """The UNFUSED per-tensor FP8 pipeline under THD (``QuantSpec``, amax scales calibrated on the packed data as the FP8
    suite does): compact e4m3 Q/K/V into the packed FP8 SDPA, per sequence against the fake-quant oracle at the FP8
    suite's cosine floor (0.99; the FP8 SDPA's e4m3 P is not replicated)."""
    res = _run_fp8_thd(_LENS)
    _check_fp8_per_sequence(res)


@requires_rubin
def test_thd_fp8_unfused_with_a_zero_length_sequence():
    """FP8 unfused THD with an EMPTY sequence ``(300, 0, 200)``: a zero-length sequence is a ``seq_kv_len == 0`` entry for
    the per-tensor FP8 d256 kernel (its documented hang history); run under the suite's ``timeout``, the neighbours exact,
    the sentinel-filled output carrying no survivor.  If this ever flakes, the fresh-process rate protocol applies (count
    over >= 8 processes, never a single rerun)."""
    res = _run_fp8_thd((300, 0, 200), max_seq_len=300)
    out = _rows(res.out)
    assert not (out == _SENTINEL).any(), f"{int((out == _SENTINEL).sum())} output cells were never written"
    _check_fp8_per_sequence(res)


def _run_fp8_thd(lens, *, max_seq_len=None, training=False, geom_kw=_COMMON):
    """The UNFUSED per-tensor FP8 packed forward over ``lens`` (``QuantSpec`` calibrated on the packed data as the FP8 suite
    does), INFERENCE by default; ``training=True`` (appended) declares ``save_for_backward=True`` and writes the bf16 training
    record (``_alloc_packed_saved`` at ``act_dtype=torch.bfloat16``, sentinel-filled; ``saved.h`` IS the e4m3 ``h``).  The
    namespace carries the per-sequence fake-quant references, the QuantSpec, the e4m3 inputs, the workspace and the record.
    ``geom_kw`` (appended): the block geometry, ``_COMMON`` by default (the packed fp8 BACKWARD suite varies the mask and the
    GQA group through it)."""
    g = GatedAttentionBlockGeometry(**geom_kw)
    inp, meta = make_packed_inputs(RefGeometry(**geom_kw), lens, max_seq_len=max_seq_len)
    inp8, desc = quantize_block_inputs(inp)
    t = meta["t"]
    # Static activation scales calibrated on this data through the oracle's own pre-quant activations (the FP8 suite's recipe).
    h32 = inp8["h"].float().view(t, g.d_model) * desc["descale_h"]
    proj = (h32 @ (inp8["w_qkvg"].float() * desc["descale_w_qkvg"]).t()).to(torch.bfloat16)
    o_q, _o_g, o_k, o_v = g.qkvg_offsets
    q = proj[:, o_q : o_q + g.h_q * g.d_head].reshape(1, t, g.h_q, g.d_head)
    k = proj[:, o_k : o_k + g.h_kv * g.d_head].reshape(1, t, g.h_kv, g.d_head)
    v = proj[:, o_v : o_v + g.h_kv * g.d_head]
    qn, _ = qk_norm_rope_reference(q, inp8["w_q_norm"], inp8["cos"], inp8["sin"], g.rope_dim, g.qk_norm_eps, qk_norm=g.qk_norm)
    kn, _ = qk_norm_rope_reference(k, inp8["w_k_norm"], inp8["cos"], inp8["sin"], g.rope_dim, g.qk_norm_eps, qk_norm=g.qk_norm)
    spec = QuantSpec(**desc, scale_q=amax_scale(qn), scale_k=amax_scale(kn), scale_v=amax_scale(v), scale_o=amax_scale(v) * 0.5)
    out = torch.full((1, t, g.d_model), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    kw = dict(quant=spec, **_thd_kw(meta))
    if training:
        kw["save_for_backward"] = True
    blk = GatedAttentionBlockFwd(inp8["h"], inp8["w_qkvg"], inp8["w_q_norm"], inp8["w_k_norm"], inp8["cos"], inp8["sin"], inp8["w_o"], out, g, **kw)
    assert blk.thd and blk._sdpa.fp8 and blk._sdpa.pertensor and blk._sdpa.token_stride == 0
    saved = _alloc_packed_saved(g, inp8, meta, seq_lens=meta["seq_lens"], form="lengths", sentinel=_SENTINEL, act_dtype=torch.bfloat16) if training else None
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    blk.execute(
        inp8["h"], inp8["w_qkvg"], inp8["w_q_norm"], inp8["w_k_norm"], inp8["cos"], inp8["sin"], inp8["w_o"], out, ws, seq_lens=meta["seq_lens"], saved=saved
    )
    torch.cuda.synchronize()
    ref_geom = RefGeometry(**geom_kw)
    refs = []
    for lo, hi in meta["slices"]:
        if hi == lo:
            refs.append(None)
            continue
        inp_i = dict(inp8, h=inp8["h"][:, lo:hi], cos=inp8["cos"][:, lo:hi], sin=inp8["sin"][:, lo:hi])
        refs.append(
            gated_attention_block_fp8_reference(
                inp_i, ref_geom, descale_h=spec.descale_h, descale_w_qkvg=spec.descale_w_qkvg, descale_w_o=spec.descale_w_o,
                scale_q=spec.scale_q, scale_k=spec.scale_k, scale_v=spec.scale_v, scale_o=spec.scale_o, fused=False,
            )
        )  # fmt: skip
    return SimpleNamespace(
        blk=blk, out=out, meta=meta, refs=refs, spec=spec, inp=inp8, desc=desc, ws=ws, saved=saved, seq_lens=meta["seq_lens"], geom=g, geom_kw=geom_kw
    )


def _check_fp8_per_sequence(res):
    out = _rows(res.out)
    assert torch.isfinite(out.float()).all()

    def check(i, lo, hi, ref):
        c = _cos(out[lo:hi], ref[0])
        assert c > 0.99, f"fp8 out cos {c:.6f}"

    failures = compare_packed(res.refs, res.meta["lens"], check)
    assert not failures, "\n".join(failures)


def _fp8_thd_inference_twin(res):
    """The packed per-tensor FP8 INFERENCE block on ``res``'s inputs and spec (in place, the inference default), its LSE
    requested and its PRE-gate O caught before stage (5) gates it in place; returns the output, that O, the LSE and the
    workspace slab (normed IN place -- the twin of ``test_block_training_forward._run_inference_quant`` under THD)."""
    inp, g, t = res.inp, res.geom, res.meta["t"]
    out = torch.zeros_like(res.out)
    infer = GatedAttentionBlockFwd(
        inp["h"],
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        out,
        g,
        quant=res.spec,
        return_lse=True,
        **_thd_kw(res.meta),
    )
    assert not infer.save_for_backward and infer.inplace_qkv and infer.return_lse and infer.thd and infer._sdpa.pertensor
    infer.check_support()
    infer.compile()
    ws = torch.empty(infer.get_workspace_size(), dtype=torch.uint8, device="cuda")
    lse = torch.empty(1, g.h_q, t, dtype=torch.float32, device="cuda")
    caught = {}
    real_gate = infer._gate.execute

    def gate_catching_o(o, gate, dst, current_stream=None):
        """Stands in for stage (5)'s ``execute``: clones the PRE-gate ``O`` into ``caught`` before the real gate overwrites it in place."""
        caught["o_pre"] = o.clone()  # the block launches on torch's current stream here, so the clone is ordered after the SDPA
        return real_gate(o, gate, dst, current_stream=current_stream)

    infer._gate.execute = gate_catching_o
    infer.execute(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws, seq_lens=res.seq_lens, lse=lse)
    torch.cuda.synchronize()
    slab = _view(ws, infer._layout().proj, (t, g.n_qkvg), torch.bfloat16)
    return SimpleNamespace(out=out, o_pre=caught["o_pre"].view(1, t, g.h_q, g.d_head), lse=lse, slab=slab, blk=infer, ws=ws)


@requires_rubin
def test_thd_fp8_training_record_is_bitwise_the_packed_inference_block():
    """The UNFUSED per-tensor FP8 pipeline TRAINS under THD (``save_for_backward=True`` on the packed batch ``(300, 128, 200)``,
    causal, QK-norm): the record is the bf16 record at ``(1, T)`` with ``saved.h`` the e4m3 ``h`` itself.  Same kernels,
    different buffers -- ``out``, the pre-gate ``O``, the LSE, the e4m3 ``q8`` / ``k8`` / ``v8`` and the slab's GATE / V bands
    are the packed FP8 INFERENCE block's bit for bit.  The record's Q/K bands are PRE-norm: not the inference slab's
    in-place-normed bands (the forward's own norm+RoPE over them reproduces that slab and the saved ``rstd`` bitwise), but the
    stage-(1) product in fp64 from the dequantized operands rounded once to bf16, at the bf16 training forward's band bound;
    ``rstd`` at the strict bound against the oracle norm of the block's own bands.  Per sequence: the LSE within 1e-4 of the
    fp64 log-sum-exp over the operands the SDPA READ (``q8`` / ``k8`` / ``v8`` over the descales, the sequence's own causal
    mask -- the quantized rows' exact-Stats contract) and ``out`` at the FP8 suite's cosine floor against the fake-quant oracle.
    The sentinel-filled record and output carry no survivor.  The dense twins are ``test_block_training_forward.py``'s
    quantized section; the backward over this record is ``test_block_thd_backward.py``'s quantized-record cell."""
    from test_block_training_forward import _attention_fp64, _dequantized_fp64_proj, _replay_norm

    tr = _run_fp8_thd(_LENS, training=True)
    assert tr.blk.save_for_backward and tr.blk.return_lse and not tr.blk.inplace_qkv and tr.blk.thd
    assert tr.saved.h is tr.inp["h"] and tr.saved.h.dtype == torch.float8_e4m3fn and tr.saved.o.dtype == tr.saved.proj_slab.dtype == torch.bfloat16
    assert tr.saved.seq_lens is tr.seq_lens and tr.saved.seq_lens_form == "lengths"
    _assert_empty_sequences_leave_no_rows(tr)
    _check_fp8_per_sequence(tr)
    inf = _fp8_thd_inference_twin(tr)
    g, t, d = tr.geom, tr.meta["t"], tr.geom.d_head
    e4 = torch.float8_e4m3fn
    assert inf.out.abs().max().item() > 0 and torch.isfinite(inf.out.float()).all()
    assert torch.equal(tr.out, inf.out), (tr.out.float() - inf.out.float()).abs().max().item()
    assert torch.equal(tr.saved.o, inf.o_pre), (tr.saved.o.float() - inf.o_pre.float()).abs().max().item()
    assert torch.equal(tr.saved.lse, inf.lse), (tr.saved.lse - inf.lse).abs().max().item()
    lay_t, lay_i = tr.blk._layout(), inf.blk._layout()
    assert (
        lay_t.q >= 0 and lay_t.k >= 0 and lay_t.proj == -1 and lay_i.q == -1
    ), "the packed training carve reserves the compact normed Q/K; the inference carve does not"
    for nm, h in (("q8", g.h_q), ("k8", g.h_kv), ("v8", g.h_kv)):
        a = _view(tr.ws, getattr(lay_t, nm), (t, h, d), e4)
        b = _view(inf.ws, getattr(lay_i, nm), (t, h, d), e4)
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), f"{nm}: the quantize stages read different values"
    tq, tgate, tk, tv = saved_slab_views(tr.saved.proj_slab, g, 1, t)
    iq, igate, ik, iv = saved_slab_views(inf.slab, g, 1, t)
    assert torch.equal(tgate, igate) and torch.equal(tv, iv)
    assert not torch.equal(tq, iq) and not torch.equal(tk, ik), "the record's Q/K bands are the inference slab's POST-norm bands"
    nq, nk, rq, rk = _replay_norm(SimpleNamespace(blk=tr.blk, geom=g, batch=1, seq_len=t, inp=tr.inp), tq, tk)
    assert torch.equal(nq.view(1, t, g.h_q, d), iq) and torch.equal(nk.view(1, t, g.h_kv, d), ik), "norm+RoPE over the record's bands is not the inference slab"
    assert torch.equal(rq.view(1, t, g.h_q), tr.saved.rstd_q) and torch.equal(rk.view(1, t, g.h_kv), tr.saved.rstd_k)
    # The bands ARE the stage-(1) product (the dequantized fp64 GEMM rounded once to bf16), not the normed values.
    proj = _dequantized_fp64_proj(tr.inp, tr.spec, "fp8").to(torch.bfloat16)
    o_q, o_g, o_k, o_v = g.qkvg_offsets
    tol = dict(rtol=2**-7, atol=1e-3)
    for nm, got, col, h in (("q_pre", tq, o_q, g.h_q), ("gate", tgate, o_g, g.h_q), ("k_pre", tk, o_k, g.h_kv), ("v", tv, o_v, g.h_kv)):
        torch.testing.assert_close(got, proj[:, col : col + h * d].view(1, t, h, d), **tol, msg=nm)
    qn_ref, rstd_q_ref = qk_norm_rope_reference(tq, tr.inp["w_q_norm"], tr.inp["cos"], tr.inp["sin"], g.rope_dim, g.qk_norm_eps, qk_norm=True)
    _, rstd_k_ref = qk_norm_rope_reference(tk, tr.inp["w_k_norm"], tr.inp["cos"], tr.inp["sin"], g.rope_dim, g.qk_norm_eps, qk_norm=True)
    assert not torch.allclose(qn_ref.float(), tq.float(), **tol), "the saved Q band already IS the normed Q: the record is POST-norm"
    torch.testing.assert_close(tr.saved.rstd_q, rstd_q_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(tr.saved.rstd_k, rstd_k_ref, rtol=1e-5, atol=1e-6)
    # Per sequence: the saved LSE is the exact fp64 log-sum-exp over the operands the SDPA read (the codes bitwise, over the descales).
    q64 = (_view(tr.ws, lay_t.q8, (t, g.h_q, d), e4).float() / tr.spec.scale_q).double()
    k64 = (_view(tr.ws, lay_t.k8, (t, g.h_kv, d), e4).float() / tr.spec.scale_k).double()
    v64 = (_view(tr.ws, lay_t.v8, (t, g.h_kv, d), e4).float() / tr.spec.scale_v).double()

    def check(i, lo, hi, _ref):
        """``compare_packed``'s per-sequence check: the saved LSE rows ``[lo:hi]`` must be within 1e-4 of the fp64 log-sum-exp over
        the operands the SDPA read (``_ref``, the sequence's oracle, is not needed here)."""
        _o64, lse64 = _attention_fp64(q64[None, lo:hi], k64[None, lo:hi], v64[None, lo:hi], g)
        d_lse = (tr.saved.lse[0, :, lo:hi].double() - lse64[0]).abs().max().item()
        print(f"seq {i} [{lo}:{hi}]: max|dLSE|={d_lse:.3e} vs the fp64 log-sum-exp over the operands the SDPA read")
        assert d_lse <= 1e-4, f"the saved LSE is not the exact log-sum-exp of the operands the SDPA read (max |dLSE| {d_lse:.3e})"

    failures = compare_packed(tr.refs, tr.meta["lens"], check)
    assert not failures, "\n".join(failures)


@requires_rubin
def test_thd_inplace_fused_norm_rope_inference():
    """``fuse_norm_rope=True`` (bf16 inference, in place) under THD: the projection fork norms and rotates per TOKEN with
    the per-token tables (``cos`` / ``sin`` are ``[T, rope_dim]`` rows to it), so it is served as the unfused chain is --
    per sequence against the oracle, and close to the unfused packed block."""
    res = _run_thd(_COMMON, _LENS, fuse_norm_rope=True, sentinel=_SENTINEL)
    assert res.blk.fuse_norm_rope and res.blk.inplace_qkv and res.blk._proj.name == "qkv_gate_proj_norm_rope"
    _assert_empty_sequences_leave_no_rows(res)
    _check_forward_per_sequence(res)
    unfused = _run_thd(_COMMON, _LENS, refs=False)
    assert _cos(res.out, unfused.out) > 0.999


@requires_rubin
def test_thd_dead_entry_padding_is_not_the_packed_path():
    """The dense padded forward (``seq_lens_present``) is unchanged by the packed arm: a dense ``B=2`` block with
    ``seq_lens = [S, 0]`` still yields ``saved.o[1] == 0``, ``lse[1] == -inf``, ``out[1] == 0`` exactly, and records
    ``seq_lens_form=None``."""
    from test_block_training_forward import _run_training

    s = 512
    lens = torch.tensor([s, 0], device="cuda", dtype=torch.int32)
    out, _ref, blk, saved, _inp = _run_training(_COMMON, 2, s, save_mode="proj_slab", seq_lens=lens, sentinel=_SENTINEL)
    assert blk.thd is False and saved.seq_lens is lens and saved.seq_lens_form is None
    assert (saved.o[1] == 0).all() and torch.isneginf(saved.lse[1]).all() and (out[1] == 0).all()
