# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The PACKED (THD / varlen) block backward: ``GatedAttentionBlockBwd(thd=True, ...)`` over the packed training record.

The record is the packed training forward's (``test_block_thd.py``): every tensor at ``(1, T)`` -- ``proj_slab [T, n_qkvg]``,
``o [1, T, H_q, D]``, ``lse [1, H_q, T]`` (head-major, head stride exactly T: the ONE packed Stats layout the forward writes
and the backward hands to the SDPA backward), ``rstd_* [1, T, H]`` -- plus ``saved.seq_lens`` (REQUIRED under THD: the
``[B]`` lengths or ``[B+1]`` prefix sums the forward ran with, bound by the backward's SDPA for both sides) and the record's
own statement of its form (``saved.seq_lens_form``: ``"lengths"`` / ``"prefix"``; ``None`` is a padded DENSE record, which a
packed backward refuses).  Every token-wise stage (the four GEMMs at ``K = T``, the gate backward, the recompute, the norm /
RoPE backward and its reduce) runs as the dense ``B=1, S=T`` block's; only the SDPA backward runs its packed chain.

The oracle is ``test_block_backward._fp64_oracle`` PER SEQUENCE on fp64 slices of ``h`` / ``cos`` / ``sin`` / ``dy``:
``dh[0, lo:hi]`` against the sequence's own ``dh``; ``dW_qkvg`` / ``dW_o`` / ``dW_q_norm`` / ``dW_k_norm`` against the fp64
SUM over the sequences; the ``dW_norm`` noise bound's ``mass`` is ``sqrt(sum_i mass_i^2)`` (the terms are the union of the
sequences' rows).  Tolerances are the dense backward suite's (``_RTOL`` / ``_ATOL_FRAC`` / the ``dW_norm`` noise bound),
never widened.  Degenerate packings are first-class: zero-length sequences at every position, a 5-token sequence (never
a 1-token one), tail tiles, the three GQA fold paths.  ``fuse_gate_bwd`` is a typed decline under THD (the packed chain's
delta has no external producer yet); ``fuse_wgrad_overlap`` is served and pinned bitwise the in-order block.

Accept tests are ``requires_rubin``; the reject tests build CUDA tensors for a DECLARED backward (``requires_cuda``, no
compile) under ``set_sync_debug_mode("error")``; the static pin of the SDPA backward stage's packed declaration runs on any
CUDA device without a compile.
"""

import dataclasses
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

from cudnn.gated_attention_block import (  # noqa: E402
    GatedAttentionBlockBwd,
    GatedAttentionBlockGeometry,
    SavedForBackward,
    gated_attention_block_backward,
    saved_slab_views,
)
from cudnn.gated_attention_block.api import _thd_lse_head_stride  # noqa: E402
from cudnn.gated_attention_block.api_bwd import _BWD_CACHE, _SdpaBwd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import RefGeometry, assert_packing_contract, compare_packed, cu_seqlens_of, make_inputs, sequence_slices  # noqa: E402
from test_block_backward import _alloc_grads, _assert_dw_norm_close, _assert_grad_close, _fp64_oracle, _make_dy  # noqa: E402
from test_block_backward import _execute as _execute_bwd  # noqa: E402
from test_block_thd import _COMMON, _LENS, _alloc_packed_saved, _declare_thd, _form, _lens, _no_device_sync, _run_thd, _thd_kw  # noqa: E402

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.requires_rubin
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.fixture(autouse=True)
def _no_tf32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


# ---------------------------------------------------------------------------
# Building a packed backward
# ---------------------------------------------------------------------------


def _declare_bwd_thd(geom_kw=_COMMON, lens=_LENS, *, cu=False, dtype=torch.bfloat16, record_kw=None, **bwd_kw):
    """A DECLARED (not compiled) packed backward over a DECLARED packed training forward's record -- CUDA tensors, no
    launch, any CUDA device.  ``record_kw`` overrides record fields (``seq_lens`` / ``seq_lens_form``) for the rejects."""
    fwd, inp, out, meta = _declare_thd(geom_kw, lens, dtype=dtype, cu=cu, save_for_backward=True)
    seq_lens = _lens(meta, cu)
    saved = _alloc_packed_saved(fwd.geom, inp, meta, seq_lens=seq_lens, form=_form(cu))
    if record_kw:
        saved = dataclasses.replace(saved, **record_kw)
    dy = _make_dy(out)
    kw = dict(_thd_kw(meta, cu=cu))
    kw.update(bwd_kw)
    bwd = GatedAttentionBlockBwd(dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], fwd.geom, **kw)
    return SimpleNamespace(blk=bwd, fwd=fwd, inp=inp, saved=saved, dy=dy, out=out, meta=meta, geom=fwd.geom, geom_kw=geom_kw, seq_lens=seq_lens)


def _packed_oracle(inp, geom_kw, dy, lens) -> dict:
    """The fp64 oracle per sequence: ``dh`` packed back row by row, every weight gradient the SUM over the sequences, the
    ``dW_norm`` masses combined as ``sqrt(sum mass_i^2)``; ``per_seq`` keeps each sequence's dict for localisation."""
    per_seq = []
    total = None
    dh = torch.zeros(1, dy.shape[1], dy.shape[2], dtype=torch.float64, device=dy.device)
    for lo, hi in sequence_slices(lens):
        if hi == lo:
            per_seq.append(None)
            continue
        inp_i = dict(inp, h=inp["h"][:, lo:hi], cos=inp["cos"][:, lo:hi], sin=inp["sin"][:, lo:hi])
        o = _fp64_oracle(inp_i, geom_kw, dy[:, lo:hi])
        per_seq.append(o)
        dh[0, lo:hi] = o["dh"][0]
        if total is None:
            total = {k: (o[k].clone() if o[k] is not None else None) for k in ("dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm")}
            total.update({k: (o[k] ** 2 if o.get(k) is not None else None) for k in ("dw_q_norm_mass", "dw_k_norm_mass")})
        else:
            for k in ("dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm"):
                if total[k] is not None:
                    total[k] += o[k]
            for k in ("dw_q_norm_mass", "dw_k_norm_mass"):
                if total.get(k) is not None:
                    total[k] += o[k] ** 2
    assert total is not None, "every sequence is empty"
    for k in ("dw_q_norm_mass", "dw_k_norm_mass"):
        if total.get(k) is not None:
            total[k] = total[k].sqrt()
    total["dh"] = dh
    total["per_seq"] = per_seq
    return total


_MEMO: dict = {}


def _backward_thd(geom_kw, lens, *, dtype=torch.bfloat16, cu=False, cu_base=0, max_seq_len=None, memo=True, fwd_kw=None, rank2=False, **bwd_kw):
    """Run the packed training forward (the record), declare / compile / run the packed backward, differentiate the
    per-sequence fp64 oracle.  Memoised per declaration so the contract tests reuse one compiled block.  ``rank2`` hands
    ``h`` / ``cos`` / ``sin`` / ``out`` to the forward and ``dy`` / ``dh`` / ``cos`` / ``sin`` to the backward as ``[T, .]``
    (the rank-2 spelling a packed caller holds; ``res.grads["dh"]`` is then rank 2 too)."""
    key = (
        tuple(sorted(geom_kw.items())),
        tuple(lens),
        str(dtype),
        cu,
        cu_base,
        max_seq_len,
        rank2,
        tuple(sorted((fwd_kw or {}).items())),
        tuple(sorted(bwd_kw.items())),
    )
    if memo and key in _MEMO:
        return _MEMO[key]
    res_f = _run_thd(geom_kw, lens, dtype=dtype, training=True, cu=cu, rank2=rank2, max_seq_len=max_seq_len, refs=False, **(fwd_kw or {}))
    if cu and cu_base:
        # Re-run the forward with the prefix tensor at a non-zero base, the record carrying THAT tensor.
        cu_t = torch.tensor(cu_seqlens_of(res_f.meta["lens"], base=cu_base), dtype=torch.int32, device="cuda")
        assert_packing_contract(cu_t, res_f.meta["t"], res_f.meta["max_seq_len"], res_f.meta["b"], cu=True)
        res_f.saved = dataclasses.replace(res_f.saved, seq_lens=cu_t)
        res_f.seq_lens = cu_t
        res_f.blk.execute(
            res_f.inp["h"],
            res_f.inp["w_qkvg"],
            res_f.inp["w_q_norm"],
            res_f.inp["w_k_norm"],
            res_f.inp["cos"],
            res_f.inp["sin"],
            res_f.inp["w_o"],
            res_f.out,
            res_f.ws,
            seq_lens=cu_t,
            saved=res_f.saved,
        )
        torch.cuda.synchronize()
    inp, saved, meta, g = res_f.inp, res_f.saved, res_f.meta, res_f.blk.geom
    dy = _make_dy(res_f.out)
    blk = GatedAttentionBlockBwd(
        dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], g, **_thd_kw(meta, cu=cu), **bwd_kw
    )
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    if rank2:
        grads["dh"] = grads["dh"].view(meta["t"], -1)  # the [T, d_model] dh a packed caller holds
    _execute_bwd(blk, inp, saved, dy, grads, ws)
    torch.cuda.synchronize()
    # The oracle works on [1, T, .] slices: view the rank-2 arm's tensors back for it.
    dy3 = dy if dy.ndim == 3 else dy.view(1, meta["t"], -1)
    inp3 = (
        inp
        if inp["h"].ndim == 3
        else dict(inp, h=inp["h"].view(1, meta["t"], -1), cos=inp["cos"].view(1, meta["t"], -1), sin=inp["sin"].view(1, meta["t"], -1))
    )
    res = SimpleNamespace(
        blk=blk, fwd=res_f.blk, inp=inp, saved=saved, dy=dy, out=res_f.out, ws=ws, grads=grads, meta=meta, geom=g, geom_kw=geom_kw, seq_lens=res_f.seq_lens,
        oracle=_packed_oracle(inp3, geom_kw, dy3, meta["lens"]),
    )  # fmt: skip
    if memo:
        _MEMO[key] = res
    return res


def _check_all_grads_packed(res) -> dict:
    """Every produced gradient against the per-sequence fp64 oracle: ``dh`` per sequence (collected verdicts), the weight
    gradients against the sum over sequences, ``dW_norm`` under the noise bound with the combined mass."""
    worst = {}
    if res.grads["dh"] is not None:
        dh = res.grads["dh"].view(1, res.meta["t"], -1)

        def check(i, lo, hi, _ref):
            worst[f"dh seq {i}"] = _assert_grad_close(dh[0, lo:hi], res.oracle["dh"][0, lo:hi], f"dh seq {i} [{lo}:{hi}]")

        failures = compare_packed(res.oracle["per_seq"], res.meta["lens"], check)
        assert not failures, "\n".join(failures)
        worst["dh"] = _assert_grad_close(dh, res.oracle["dh"], "dh (whole packed matrix)")
    for name in ("dw_qkvg", "dw_o"):
        if res.grads[name] is not None:
            worst[name] = _assert_grad_close(res.grads[name], res.oracle[name], name)
    for name in ("dw_q_norm", "dw_k_norm"):
        if res.grads[name] is not None:
            worst[name] = _assert_dw_norm_close(res.grads[name], res.oracle[name], res.oracle[name + "_mass"], name)
    print(f"worst cells: {worst}")
    return worst


# ---------------------------------------------------------------------------
# Static pin -- the SDPA backward stage's packed declaration (any CUDA device, no compile)
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("cu", [False, True], ids=["lengths", "prefix"])
def test_thd_sdpa_bwd_stage_declares_the_packed_chain(cu):
    """The SDPA backward stage under THD declares ``thd=True`` with both packed totals ``T``, the envelope descriptors
    ``(B, H, S_max, D)`` with ``(B, H_q, S_max, 1)`` Stats, ``thd_stats_token_major=False`` with
    ``thd_stats_head_stride == _thd_lse_head_stride(T)`` (the forward's packing, imported -- never a literal T),
    ``external_delta=False`` (the packed chain's own ``dot_do_o``) and no ``seq_kv_lens_present``.  The adapter's
    constructor touches no device."""
    g = GatedAttentionBlockGeometry(**_COMMON)
    b, s_max, t = 3, 300, 628
    st = _SdpaBwd(g, batch=1, seq_len=t, dtype=torch.bfloat16, device=torch.device("cuda"), thd=True, num_sequences=b, max_seq_len=s_max, cu_seqlens=cu)
    impl = st._build_impl()
    assert impl.thd is True and impl.external_delta is False and impl.seq_kv_lens_present is False
    assert impl.max_total_seq_len_q == t and impl.max_total_seq_len_kv == t
    assert impl.thd_stats_token_major is False and impl.thd_stats_head_stride == _thd_lse_head_stride(t) == t
    assert tuple(impl.stats_desc.shape) == (b, g.h_q, s_max, 1)
    assert tuple(impl.q_desc.shape) == (b, g.h_q, s_max, g.d_head) and tuple(impl.dk_desc.shape) == (b, g.h_kv, s_max, g.d_head)
    assert st.scratch_workspace_bytes() > 0
    dense = _SdpaBwd(g, batch=1, seq_len=t, dtype=torch.bfloat16, device=torch.device("cuda"))._build_impl()
    assert dense.thd is False and dense.max_total_seq_len_q is None


# ---------------------------------------------------------------------------
# Rejects -- declaration / check_support contracts, host-side (any CUDA device, no compile, no device read)
# ---------------------------------------------------------------------------


@requires_cuda
def test_thd_backward_declaration_records_the_knobs():
    """A packed backward is declared from a ``[1, T, d_model]`` (or ``[T, d_model]``) ``dy`` with the forward's four knobs:
    ``batch = 1, seq_len = T``, the SDPA backward stage packed, the padding knob off; ``check_support`` passes on the
    healthy record up to the arch gate."""
    res = _declare_bwd_thd()
    blk = res.blk
    assert (
        blk.thd
        and (blk.batch, blk.seq_len) == (1, res.meta["t"])
        and (blk.num_sequences, blk.max_seq_len, blk.cu_seqlens) == (res.meta["b"], res.meta["max_seq_len"], False)
    )
    assert blk._sdpa.thd and not blk.seq_lens_present and not blk.fuse_gate_bwd
    dy2 = res.dy.view(res.meta["t"], -1)
    blk2 = GatedAttentionBlockBwd(
        dy2,
        res.saved,
        res.inp["w_qkvg"],
        res.inp["w_q_norm"],
        res.inp["w_k_norm"],
        res.inp["cos"],
        res.inp["sin"],
        res.inp["w_o"],
        res.geom,
        **_thd_kw(res.meta),
    )
    assert (blk2.batch, blk2.seq_len) == (1, res.meta["t"])
    if _cc() != _SM107:
        with _no_device_sync(), pytest.raises(NotImplementedError, match="Rubin"):
            blk.check_support()


@requires_cuda
def test_thd_declines_fuse_gate_bwd():
    """``fuse_gate_bwd=True`` under THD is a typed ``NotImplementedError`` at ``check_support`` -- the packed chain
    computes its delta in the head-major ``[1, H_q, ceil128(T_q)]`` layout and the gate-backward kernel has no packed
    delta arm -- naming the knob to pass instead."""
    res = _declare_bwd_thd(fuse_gate_bwd=True)
    with _no_device_sync(), pytest.raises(NotImplementedError, match="fuse_gate_bwd=True is dense-only"):
        res.blk.check_support()


@requires_cuda
def test_thd_backward_and_seq_lens_present_are_mutually_exclusive():
    """``thd=True`` with ``seq_lens_present=True`` on the backward is the forward's ``ValueError``."""
    res = _declare_bwd_thd(seq_lens_present=True)
    with _no_device_sync(), pytest.raises(ValueError, match="mutually exclusive"):
        res.blk.check_support()


@requires_cuda
def test_thd_backward_requires_the_record_lengths_and_their_form():
    """Under THD ``saved.seq_lens`` must be the int32 ``[B]`` lengths (or ``[B+1]`` prefixes under ``cu_seqlens``) tensor
    the forward ran with -- ``None``, a wrong dtype, rank, element count (the other form's count included), device or a
    non-contiguous tensor is a ``ValueError`` naming the field (host-only, no value read); ``saved.seq_lens_form`` must
    agree with the block's ``cu_seqlens`` (``None`` -- a padded DENSE record -- or the other form is declined)."""
    for cu in (False, True):
        res = _declare_bwd_thd(cu=cu)
        none = _declare_bwd_thd(cu=cu, record_kw=dict(seq_lens=None))  # the declarations allocate (H2D of the inputs): outside the sync guard
        with _no_device_sync(), pytest.raises(ValueError, match=r"SavedForBackward\.seq_lens must be"):
            none.blk.check_support()
        b = res.meta["b"]
        n_ok = b + 1 if cu else b
        good = res.saved.seq_lens
        bads = {
            "dtype": good.to(torch.int64),
            "rank": good.view(1, n_ok),
            "count": torch.zeros(n_ok + 2, dtype=torch.int32, device="cuda"),
            "count (the other form)": _lens(res.meta, not cu),
            "contiguous": torch.zeros(2 * n_ok, dtype=torch.int32, device="cuda")[::2],
            "device": good.cpu(),
        }
        decls = {why: _declare_bwd_thd(cu=cu, record_kw=dict(seq_lens=bad)) for why, bad in bads.items()}
        for why, d in decls.items():
            with _no_device_sync(), pytest.raises(ValueError, match="seq_lens") as ei:
                d.blk.check_support()
            assert "seq_lens" in str(ei.value), why
        form_none = _declare_bwd_thd(cu=cu, record_kw=dict(seq_lens_form=None))
        form_other = _declare_bwd_thd(cu=cu, record_kw=dict(seq_lens_form=_form(not cu)))
        with _no_device_sync():
            with pytest.raises(ValueError, match="seq_lens_form"):
                form_none.blk.check_support()
            with pytest.raises(ValueError, match="seq_lens_form"):
                form_other.blk.check_support()


@requires_cuda
def test_padded_dense_record_into_a_thd_backward_is_declined():
    """A PADDED dense record (``seq_lens`` a ``[B]`` padding mask, ``seq_lens_form=None``) handed to a packed backward is
    declined typed -- the two records look alike (both carry a lengths tensor) and only the form field tells them apart;
    and a packed record into a DENSE backward keeps the dense decline (its ``seq_lens`` tensor is the fact)."""
    from test_block_training_forward import _alloc_saved, _declare

    b, s = 3, 256
    fwd, inp, out = _declare(_COMMON, b, s, seq_lens_present=True)
    pad = torch.tensor([256, 128, 0], dtype=torch.int32, device="cuda")
    padded = _alloc_saved(fwd.geom, inp, b, s, save_mode="proj_slab", seq_lens=pad)  # a dense padded record at (B, S)
    assert padded.seq_lens_form is None
    # Re-spelled at (1, T) as a packed caller might: the record says form None -> declined, naming the field.
    t = b * s
    flat = SavedForBackward(
        h=inp["h"].view(1, t, -1), gate=None, q_pre=None, k_pre=None, proj_slab=padded.proj_slab, o=padded.o.view(1, t, fwd.geom.h_q, fwd.geom.d_head),
        lse=padded.lse.permute(1, 0, 2).reshape(1, fwd.geom.h_q, t).contiguous(), rstd_q=padded.rstd_q.view(1, t, -1), rstd_k=padded.rstd_k.view(1, t, -1),
        seq_lens=pad, seq_lens_form=None,
    )  # fmt: skip
    dy = _make_dy(out.view(1, t, -1))
    blk = GatedAttentionBlockBwd(
        dy,
        flat,
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"].reshape(1, t, -1),
        inp["sin"].reshape(1, t, -1),
        inp["w_o"],
        fwd.geom,
        thd=True,
        num_sequences=b,
        max_seq_len=s,
    )
    with _no_device_sync(), pytest.raises(ValueError, match="seq_lens_form"):
        blk.check_support()
    # The other direction: a packed record into a dense backward -> the dense padding decline (a tensor in seq_lens).
    res = _declare_bwd_thd()
    dense_bwd = GatedAttentionBlockBwd(
        res.dy, res.saved, res.inp["w_qkvg"], res.inp["w_q_norm"], res.inp["w_k_norm"], res.inp["cos"], res.inp["sin"], res.inp["w_o"], res.geom
    )
    with _no_device_sync(), pytest.raises((NotImplementedError, ValueError), match="seq_lens"):
        dense_bwd.check_support()


@requires_cuda
def test_thd_backward_bounds_and_knob_placement_are_typed():
    """The forward's declaration contracts hold on the backward with the same texts -- both knobs required, the
    ``2 <= max_seq_len <= T`` / product bound, the THD-only knobs refused on a dense backward."""
    res = _declare_bwd_thd()
    args = (res.dy, res.saved, res.inp["w_qkvg"], res.inp["w_q_norm"], res.inp["w_k_norm"], res.inp["cos"], res.inp["sin"], res.inp["w_o"], res.geom)
    t = res.meta["t"]

    def decl(**kw):
        blk = GatedAttentionBlockBwd(*args, **kw)
        blk.check_support()

    with _no_device_sync():
        with pytest.raises(ValueError, match="needs num_sequences"):
            decl(thd=True)
        with pytest.raises(ValueError, match="needs num_sequences"):
            decl(thd=True, num_sequences=3)
        # The bounds, on a record CONSISTENT with num_sequences (3 lengths): S = 1, longer than T, and 3 * 200 = 600 < 628.
        for kw in (dict(num_sequences=3, max_seq_len=1), dict(num_sequences=3, max_seq_len=t + 1), dict(num_sequences=3, max_seq_len=200)):
            with pytest.raises(ValueError, match="2 <= max_seq_len <= T"):
                decl(thd=True, **kw)
        # A declaration whose num_sequences disagrees with the record's length count is a typed decline naming seq_lens
        # (the record check and the bounds check may speak in either order; both are host-side).
        for kw in (dict(num_sequences=0, max_seq_len=300), dict(num_sequences=2, max_seq_len=314)):
            with pytest.raises(ValueError, match="seq_lens|2 <= max_seq_len <= T"):
                decl(thd=True, **kw)
        for kw in (dict(num_sequences=3), dict(max_seq_len=300), dict(cu_seqlens=True)):
            with pytest.raises(ValueError, match="THD-only"):
                decl(**kw)


@requires_cuda
def test_thd_wrapper_derives_the_packing_from_the_record():
    """The convenience wrapper takes ``thd`` and ``max_seq_len`` only and derives ``num_sequences`` and ``cu_seqlens`` from
    the record (``saved.seq_lens.numel()`` / ``saved.seq_lens_form``); a record without lengths under ``thd`` surfaces the
    class's typed decline, never a wrapper crash."""
    import inspect

    params = inspect.signature(gated_attention_block_backward).parameters
    assert "thd" in params and "max_seq_len" in params
    assert "num_sequences" not in params and "cu_seqlens" not in params, "the wrapper derives both from the record"
    assert list(params)[-2:] == ["thd", "max_seq_len"], "appended LAST"
    res = _declare_bwd_thd(record_kw=dict(seq_lens=None))
    for ten in (res.saved.h, res.inp["w_qkvg"], res.inp["w_o"]):
        ten.requires_grad_(True)
    with _no_device_sync(), pytest.raises(ValueError, match=r"SavedForBackward\.seq_lens must be"):
        gated_attention_block_backward(
            res.dy,
            res.saved,
            res.inp["w_qkvg"],
            res.inp["w_q_norm"],
            res.inp["w_k_norm"],
            res.inp["cos"],
            res.inp["sin"],
            res.inp["w_o"],
            res.geom,
            thd=True,
            max_seq_len=res.meta["max_seq_len"],
        )
    # thd=True without max_seq_len: the wrapper's own message names max_seq_len alone (it has no num_sequences to ask for;
    # the class's text would name both).
    res2 = _declare_bwd_thd()
    for ten in (res2.saved.h, res2.inp["w_qkvg"], res2.inp["w_o"]):
        ten.requires_grad_(True)
    with _no_device_sync(), pytest.raises(ValueError, match=r"^thd=True on gated_attention_block_backward needs max_seq_len") as ei:
        gated_attention_block_backward(
            res2.dy,
            res2.saved,
            res2.inp["w_qkvg"],
            res2.inp["w_q_norm"],
            res2.inp["w_k_norm"],
            res2.inp["cos"],
            res2.inp["sin"],
            res2.inp["w_o"],
            res2.geom,
            thd=True,
        )
    assert "needs num_sequences" not in str(ei.value)


@requires_cuda
def test_thd_backward_rejects_zero_tokens():
    """``T == 0`` on the backward -- a ``dy`` with no rows over a record with no tokens -- is the forward's ``ValueError``: the
    SDPA adapters refuse a zero packed capacity and a GEMM over M = 0 has nothing to launch; an empty step is the caller's
    early-out.  It fires ahead of the ``max_seq_len <= T`` bound, so the message names the cause, not the bound."""
    g = GatedAttentionBlockGeometry(**_COMMON)
    inp = make_inputs(RefGeometry(**_COMMON), batch=1, seq_len=8)
    z = lambda *s, dt=torch.bfloat16: torch.empty(*s, device="cuda", dtype=dt)  # noqa: E731
    proj_slab = z(0, g.n_qkvg)
    q_pre, gate, k_pre, _v = saved_slab_views(proj_slab, g, 1, 0)
    saved = SavedForBackward(
        h=z(0, g.d_model), gate=gate, q_pre=q_pre, k_pre=k_pre, proj_slab=proj_slab, o=z(1, 0, g.h_q, g.d_head), lse=z(1, g.h_q, 0, dt=torch.float32),
        rstd_q=z(1, 0, g.h_q, dt=torch.float32), rstd_k=z(1, 0, g.h_kv, dt=torch.float32),
        seq_lens=torch.zeros(1, dtype=torch.int32, device="cuda"), seq_lens_form="lengths",
    )  # fmt: skip
    blk = GatedAttentionBlockBwd(
        z(0, g.d_model),
        saved,
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        z(0, g.rope_dim),
        z(0, g.rope_dim),
        inp["w_o"],
        g,
        thd=True,
        num_sequences=1,
        max_seq_len=2,
    )
    with _no_device_sync(), pytest.raises(ValueError, match="T >= 1 packed tokens"):
        blk.check_support()


# ---------------------------------------------------------------------------
# Accept -- Rubin only
# ---------------------------------------------------------------------------


@requires_rubin
@pytest.mark.parametrize("arm", ["causal-norm", "dense-norm", "causal-rope_only"])
def test_thd_gradients_match_fp64_autograd(arm):
    """``(300, 128, 200)`` B=3 through the packed training record: ``dh`` per sequence, the weight gradients against the
    fp64 sum over the sequences, ``dW_norm`` under the noise bound with the combined mass -- causal and dense, norm and
    RoPE-only.  The bounds are the dense backward suite's; the magnitudes are printed."""
    causal, qk_norm = arm.startswith("causal"), arm.endswith("norm")
    res = _backward_thd({**_COMMON, "is_causal": causal, "qk_norm": qk_norm}, _LENS)
    assert res.blk.thd and res.blk._sdpa._impl.thd and res.blk._sdpa._impl.external_delta is False
    assert (res.grads["dw_q_norm"] is None) == (not qk_norm)
    _check_all_grads_packed(res)


@requires_rubin
@pytest.mark.parametrize("arm", ["bottom_right", "swa"])
def test_thd_mask_arms_bwd(arm):
    """Bottom-right causal and the 640-window at lengths ``(900, 700)`` (``S_max = 1024``) through the packed backward,
    per sequence."""
    if arm == "bottom_right":
        res = _backward_thd({**_COMMON, "is_causal": True, "causal_bottom_right": True}, _LENS)
        assert res.blk._sdpa._impl.causal_bottom_right
    else:
        res = _backward_thd({**_COMMON, "is_causal": True, "window_left": 640}, (900, 700), max_seq_len=1024)
    _check_all_grads_packed(res)


@requires_rubin
@pytest.mark.parametrize("h_kv", [2, 1, 8], ids=["gqa_8_2", "mqa", "mha"])
def test_thd_gqa_bwd(h_kv):
    """The three fold paths of the packed backward -- 8/2 (dK / dV partials per Q head folded on device over the packed
    kv axis, bounded by the live total), MQA (group 8) and MHA (no fold)."""
    res = _backward_thd({**_COMMON, "h_kv": h_kv}, _LENS)
    _check_all_grads_packed(res)


@requires_rubin
@pytest.mark.parametrize(
    "lens, s_max",
    [((256, 0, 128), 256), ((0, 256, 128), 256), ((5, 0, 0), 5)],
    ids=["middle_empty", "first_empty", "trailing_empties_T5"],
)
def test_thd_zero_length_sequence_bwd(lens, s_max):
    """Zero-length sequences in the packed backward: ``dh`` of the live sequences exact per sequence, every weight gradient
    the live sequences' sum (an empty sequence contributes nothing), the gradients finite everywhere."""
    res = _backward_thd(_COMMON, lens, max_seq_len=s_max)
    assert 0 in res.meta["lens"]
    for name, ten in res.grads.items():
        if ten is not None:
            assert torch.isfinite(ten.float()).all(), name
    _check_all_grads_packed(res)


@requires_rubin
def test_thd_ragged_envelope_bwd():
    """``(130, 5)`` with ``S_max = 130``: the kv-blocked dS workspace at a non-multiple envelope with a 5-token tail."""
    res = _backward_thd(_COMMON, (130, 5))
    _check_all_grads_packed(res)


@requires_rubin
def test_thd_b1_bwd_is_bitwise_the_dense_block():
    """``B = 1`` packed ``(512,)`` against the dense ``B=1, S=512`` backward over the same record bytes: every gradient
    BITWISE (one sequence over the same tiles)."""
    from test_block_backward import _backward

    t = 512
    res = _backward_thd(_COMMON, (t,))
    dense = _backward(dict(_COMMON), batch=1, seq_len=t)
    for k in ("h", "w_qkvg", "w_o", "cos", "sin"):
        assert torch.equal(dense.inp[k], res.inp[k]), k
    assert (
        torch.equal(dense.dy, res.dy)
        and torch.equal(dense.saved.proj_slab, res.saved.proj_slab)
        and torch.equal(dense.saved.o, res.saved.o)
        and torch.equal(dense.saved.lse, res.saved.lse)
    )
    for name, ten in res.grads.items():
        if ten is not None:
            d = (ten.float() - dense.grads[name].float()).abs().max().item()
            assert torch.equal(ten, dense.grads[name]), f"{name}: packed B=1 differs from the dense backward (max|diff| {d:.3e}) -- a finding, not a tolerance"


@requires_rubin
def test_thd_uniform_b4_bwd_matches_the_dense_block_per_sequence():
    """Uniform ``B = 4`` packed ``(256,)*4`` against the dense ``B=4, S=256`` backward: the token-wise stage outputs
    (``dh`` rows per sequence, ``dW``) at the dense suite's bounds against the SAME fp64 oracle both are held to; the
    first differing gradient against the dense block is printed (the packed SDPA backward walks its kv-blocked workspace
    differently)."""
    from test_block_backward import _backward

    s, b = 256, 4
    res = _backward_thd(_COMMON, (s,) * b)
    dense = _backward(dict(_COMMON), batch=b, seq_len=s)
    assert torch.equal(dense.inp["h"].reshape(1, b * s, -1), res.inp["h"]) and torch.equal(dense.dy.reshape(1, b * s, -1), res.dy)
    first_diff = None
    for name, ten in res.grads.items():
        if ten is None:
            continue
        d_t = dense.grads[name].reshape(ten.shape)
        md = (ten.float() - d_t.float()).abs().max().item()
        if md and first_diff is None:
            first_diff = (name, md)
        if name in ("dh", "dw_qkvg", "dw_o"):
            _assert_grad_close(ten, d_t.double(), f"{name} vs the dense B=4 block")
    print(f"\nuniform B=4 vs dense: first differing gradient {first_diff} (None = bitwise)")
    _check_all_grads_packed(res)


@requires_rubin
@pytest.mark.parametrize("cu_base", [0, 100], ids=["prefix", "prefix_nonzero_base"])
def test_thd_lengths_and_prefix_forms_are_bitwise_bwd(cu_base):
    """On the backward: the same packing through the ``[B]`` lengths and the ``[B+1]`` prefix record (base 0 and base
    100) gives ``torch.equal`` gradients."""
    a = _backward_thd(_COMMON, _LENS)
    b = _backward_thd(_COMMON, _LENS, cu=True, cu_base=cu_base)
    assert b.saved.seq_lens_form == "prefix" and b.saved.seq_lens.numel() == b.meta["b"] + 1 and int(b.saved.seq_lens[0]) == cu_base
    for name, ten in a.grads.items():
        if ten is not None:
            assert torch.equal(ten, b.grads[name]), f"{name} differs between the lengths and the prefix form"


@requires_rubin
def test_thd_lse_consumer_reads_the_head_major_record():
    """The record contract, consumer half: the backward binds ``saved.lse`` as the contiguous ``[1, H_q, T]`` head-major tensor
    (``thd_stats_head_stride == T``) the forward wrote -- the gradients over THAT tensor match the oracle, and a record
    whose LSE is re-spelled token-major ``(T, H_q)`` is refused (the record contract: ``saved.lse`` is ``[1, H_q, T]``)."""
    res = _backward_thd(_COMMON, _LENS)
    impl = res.blk._sdpa._impl
    assert (
        impl.thd_stats_head_stride == _thd_lse_head_stride(res.meta["t"])
        and res.saved.lse.shape == (1, res.geom.h_q, res.meta["t"])
        and res.saved.lse.is_contiguous()
    )
    _check_all_grads_packed(res)
    bad = dataclasses.replace(res.saved, lse=res.saved.lse[0].t().contiguous().view(res.meta["t"], res.geom.h_q))
    grads = _alloc_grads(res.blk)
    with pytest.raises(ValueError, match=r"saved\.lse"):
        _execute_bwd(res.blk, res.inp, bad, res.dy, grads, res.ws)


@requires_rubin
def test_thd_fuse_wgrad_overlap_is_bitwise_the_in_order_block():
    """``fuse_wgrad_overlap=True`` under THD: the same launches on the side stream, every gradient ``torch.equal`` the
    in-order packed block's (workspace poisoned, gradients NaN-filled first)."""
    res = _backward_thd(_COMMON, _LENS)
    on = GatedAttentionBlockBwd(
        res.dy,
        res.saved,
        res.inp["w_qkvg"],
        res.inp["w_q_norm"],
        res.inp["w_k_norm"],
        res.inp["cos"],
        res.inp["sin"],
        res.inp["w_o"],
        res.geom,
        **_thd_kw(res.meta),
        fuse_wgrad_overlap=True,
    )
    on.check_support()
    on.compile()
    ws = torch.empty(on.get_workspace_size(), dtype=torch.uint8, device="cuda").fill_(0xFF)
    grads = _alloc_grads(on, fill=float("nan"))
    _execute_bwd(on, res.inp, res.saved, res.dy, grads, ws)
    torch.cuda.synchronize()
    assert on.fuse_wgrad_overlap and on._side is not None
    for name, ten in grads.items():
        if ten is not None:
            assert torch.isfinite(ten).all(), name
            assert torch.equal(ten, res.grads[name]), f"{name}: the overlapped packed block differs from the in-order one"


@requires_rubin
def test_thd_two_runs_are_bitwise_bwd():
    """Two executes with the workspace poisoned between (0xFF) and NaN-filled gradients: bitwise equal (the packed
    chain's metadata / dS workspace included)."""
    res = _backward_thd(_COMMON, _LENS)
    res.ws.fill_(0xFF)
    grads2 = _alloc_grads(res.blk, fill=float("nan"))
    _execute_bwd(res.blk, res.inp, res.saved, res.dy, grads2, res.ws)
    torch.cuda.synchronize()
    for name, ten in grads2.items():
        if ten is not None:
            assert torch.isfinite(ten).all(), name
            assert torch.equal(ten, res.grads[name]), f"{name} differs across two runs"


@requires_rubin
def test_thd_rank2_dy_and_dh_are_the_rank3_backward_bitwise():
    """``dy`` / ``dh`` / ``cos`` / ``sin`` handed over as ``[T, .]`` -- the rank-2 spelling a packed caller holds, over a record
    whose ``h`` is ``[T, d_model]`` -- through declare / compile / execute: every gradient bitwise the ``[1, T, .]`` block's
    (the declaration pin alone, ``test_thd_backward_declaration_records_the_knobs``, never ran the rank-2 execute path)."""
    a = _backward_thd(_COMMON, _LENS)
    b = _backward_thd(_COMMON, _LENS, rank2=True, memo=False)
    assert b.dy.ndim == 2 and b.saved.h.ndim == 2 and b.inp["cos"].ndim == 2 and b.grads["dh"].ndim == 2
    for name, ten in a.grads.items():
        if ten is None:
            assert b.grads[name] is None, name
            continue
        assert torch.equal(ten, b.grads[name].reshape(ten.shape)), f"{name}: the rank-2 backward differs from the rank-3 one"


@requires_rubin
def test_thd_cuda_graph_replay_with_new_lengths_bwd():
    """One packed backward captured into a CUDA graph replays bitwise the eager run; a replay over a NEW packing written
    through the captured pointers (the forward re-run eagerly over the new lengths so the record matches; same ``B``,
    ``sum == T``, each ``<= max_seq_len``) equals a fresh eager backward over it -- the setup kernel rebuilds the packed
    metadata per execute from the device lengths."""
    from gated_block_reference import packed_rope_tables

    res = _backward_thd(_COMMON, _LENS, memo=False)
    blk, inp, saved, dy, g, meta = res.blk, res.inp, res.saved, res.dy, res.geom, res.meta
    ws = torch.empty_like(res.ws)
    grads = _alloc_grads(blk, fill=float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _execute_bwd(blk, inp, saved, dy, grads, ws)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for ten in grads.values():
        if ten is not None:
            ten.fill_(float("nan"))
    ws.fill_(0xFF)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            _execute_bwd(blk, inp, saved, dy, grads, ws)
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.isnan(ten).all(), f"{name}: the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.equal(ten, res.grads[name]), f"{name}: the replay differs from the eager run"
        # A new packing through the captured pointers: new tables + lengths, the forward re-run eagerly into the same record.
        new_lens = [200, 300, 128]
        assert_packing_contract(new_lens, meta["t"], meta["max_seq_len"], meta["b"])
        cos2, sin2 = packed_rope_tables(new_lens, g.rope_dim, base=RefGeometry(**_COMMON).rope_base)
        inp["cos"].copy_(cos2.view_as(inp["cos"]))
        inp["sin"].copy_(sin2.view_as(inp["sin"]))
        res.seq_lens.copy_(torch.tensor(new_lens, dtype=torch.int32, device="cuda"))
        res.fwd.execute(
            inp["h"],
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            inp["cos"],
            inp["sin"],
            inp["w_o"],
            res.out,
            torch.empty(res.fwd.get_workspace_size(), dtype=torch.uint8, device="cuda"),
            seq_lens=res.seq_lens,
            saved=saved,
        )
        ws.fill_(0xFF)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        ref = _alloc_grads(blk, fill=float("nan"))
        ws_ref = torch.empty_like(ws).fill_(0xFF)
        _execute_bwd(blk, inp, saved, dy, ref, ws_ref)
        torch.cuda.synchronize()
        for name, ten in grads.items():
            if ten is not None:
                assert torch.isfinite(ten).all() and torch.equal(ten, ref[name]), f"{name}: the replay over new lengths differs from eager"
        oracle = _packed_oracle(inp, _COMMON, dy, new_lens)
        _assert_grad_close(grads["dh"].view(1, meta["t"], -1), oracle["dh"], "dh over the new packing")
    finally:
        graph.reset()


@requires_rubin
def test_thd_workspace_size_is_honest_bwd():
    """``get_workspace_size()`` exact and never exceeded on the packed backward: the tail of a 4096-B-larger buffer
    untouched, no allocation on the hot path, bitwise the memoised gradients; the SDPA backward's packed scratch is the
    stage's own number; one byte less is a typed ``ValueError``."""
    res = _backward_thd(_COMMON, _LENS)
    blk = res.blk
    size = blk.get_workspace_size()
    lay = blk._layout()
    print(f"\nworkspace {size} B; sdpa packed scratch {lay.sdpa_bwd_bytes} B; gemm scratch {lay.gemm_scratch_bytes} B")
    assert size == lay.total_bytes and lay.sdpa_bwd_bytes == blk._sdpa.scratch_workspace_bytes() > 0 and lay.delta == -1
    ws = torch.full((size + 4096,), 0xAB, dtype=torch.uint8, device="cuda")
    grads = _alloc_grads(blk)
    _execute_bwd(blk, res.inp, res.saved, res.dy, grads, ws)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    _execute_bwd(blk, res.inp, res.saved, res.dy, grads, ws)
    _execute_bwd(blk, res.inp, res.saved, res.dy, grads, ws)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, "execute allocated on the hot path"
    assert torch.equal(ws[size:], torch.full((4096,), 0xAB, dtype=torch.uint8, device="cuda")), "bytes past get_workspace_size() were written"
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), name
    with pytest.raises(ValueError, match="workspace is"):
        _execute_bwd(blk, res.inp, res.saved, res.dy, grads, ws[: size - 1])


@requires_rubin
def test_thd_launch_count_is_honest_bwd():
    """CUPTI kernel records of one packed backward == the launch table recomputed from the adapter's own facts: the block's
    9 token-wise launches + the packed chain's ``setup + [zero-fill] + dot_do_o + c x (own setup + main + (patch + dK) +
    q x (patch + dQ)) + [dkv_reduce]`` -- the main kernel's THD host issues its own per-chunk setup launch, and every
    stage-3 GEMM is preceded by a per-sequence descriptor-patch launch; ``q`` is read off the dQ record.  MEASURED on the
    first run (18 at this geometry: 9 + 1 + 1 + (1 + 1 + 2 + 2) + 1) and recorded; the names are printed; a typed skip
    when CUPTI records nothing on this node."""
    from torch.profiler import ProfilerActivity, profile

    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import _dq_launches

    res = _backward_thd(_COMMON, _LENS)
    blk, g = res.blk, res.geom
    impl = blk._sdpa._impl
    grp = g.h_q // g.h_kv
    c = g.h_q // impl._qh_chunk
    dq = _dq_launches(grp, impl._dq_b_head_group)
    chain = 1 + (1 if impl._zero_ws else 0) + 1 + c * (1 + 1 + 2 + 2 * dq) + (1 if grp > 1 else 0)
    formula = 9 + chain
    grads = _alloc_grads(blk)
    _execute_bwd(blk, res.inp, res.saved, res.dy, grads, res.ws)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _execute_bwd(blk, res.inp, res.saved, res.dy, grads, res.ws)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if not names:
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the launch count is unverified here")
    memsets = [n for n in names if "memset" in n.lower()]
    memcpys = [n for n in names if "memcpy" in n.lower()]
    kernels = [n for n in names if n not in memsets and n not in memcpys]
    print(
        f"\n{len(kernels)} kernels (formula {formula}: 9 block + {chain} chain; c={c}, q={dq}, zero_ws={impl._zero_ws}, grp={grp}), {len(memsets)} memsets, {len(memcpys)} memcpys:\n  "
        + "\n  ".join(names)
    )
    assert not memcpys, f"a hidden copy on the execute path: {memcpys}"
    assert len(kernels) == formula, (len(kernels), formula, kernels)


@requires_rubin
def test_thd_execute_lengths_must_be_the_record_tensor():
    """``execute(seq_lens=)`` on a packed backward is optional -- ``None`` binds ``saved.seq_lens`` -- and a given tensor
    must be ``saved.seq_lens`` itself (a clone with the same values is refused: the record carries the packed lengths)."""
    res = _backward_thd(_COMMON, _LENS)
    grads = _alloc_grads(res.blk, fill=float("nan"))
    _execute_bwd(res.blk, res.inp, res.saved, res.dy, grads, res.ws, seq_lens=res.saved.seq_lens)
    torch.cuda.synchronize()
    for name, ten in grads.items():
        if ten is not None:
            assert torch.equal(ten, res.grads[name]), name
    with pytest.raises(ValueError, match=r"saved\.seq_lens itself"):
        _execute_bwd(res.blk, res.inp, res.saved, res.dy, grads, res.ws, seq_lens=res.saved.seq_lens.clone())


@requires_rubin
def test_thd_convenience_wrapper_caches_per_packing_declaration():
    """Two wrapper calls differing only in ``max_seq_len`` (or in the record's form) build two blocks -- the cache key
    carries the packing facts -- and the wrapper's gradients equal the class's bitwise."""
    res = _backward_thd(_COMMON, _LENS)
    inp, saved, g = res.inp, res.saved, res.geom
    leaves = dict(h=saved.h, w_qkvg=inp["w_qkvg"], w_o=inp["w_o"], w_q_norm=inp["w_q_norm"], w_k_norm=inp["w_k_norm"])
    for ten in leaves.values():
        ten.requires_grad_(True)
    try:
        n0 = len(_BWD_CACHE)
        out1 = gated_attention_block_backward(
            res.dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], g, thd=True, max_seq_len=res.meta["max_seq_len"]
        )
        torch.cuda.synchronize()
        assert len(_BWD_CACHE) == n0 + 1
        for name, ten in res.grads.items():
            if ten is not None:
                assert torch.equal(out1[name], ten), name
        gated_attention_block_backward(
            res.dy,
            saved,
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            inp["cos"],
            inp["sin"],
            inp["w_o"],
            g,
            thd=True,
            max_seq_len=res.meta["max_seq_len"] + 64,
        )
        torch.cuda.synchronize()
        assert len(_BWD_CACHE) == n0 + 2, "max_seq_len is not in the wrapper's cache key"
        gated_attention_block_backward(
            res.dy, saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], g, thd=True, max_seq_len=res.meta["max_seq_len"]
        )
        assert len(_BWD_CACHE) == n0 + 2, "a repeated declaration must hit the cache"
        with pytest.raises(NotImplementedError, match="fuse_gate_bwd=True is dense-only"):
            gated_attention_block_backward(
                res.dy,
                saved,
                inp["w_qkvg"],
                inp["w_q_norm"],
                inp["w_k_norm"],
                inp["cos"],
                inp["sin"],
                inp["w_o"],
                g,
                thd=True,
                max_seq_len=res.meta["max_seq_len"],
                fuse_gate_bwd=True,
            )
    finally:
        for ten in leaves.values():
            ten.requires_grad_(False)
