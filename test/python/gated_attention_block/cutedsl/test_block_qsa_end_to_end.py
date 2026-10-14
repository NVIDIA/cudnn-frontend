# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block END TO END under ``QsaSpec`` -- stage (4) is the index-list sparse core -- against the QSA oracle.

Every Rubin cell (``requires_rubin``) declares a block with ``GatedAttentionBlockGeometry(qsa=QsaSpec(...))``, checks, compiles
and runs it TWICE on sentinel-filled outputs, and compares ``out``, the gated O (the workspace slot stage (5) wrote in place)
and the LSE against ``gated_attention_block_qsa_reference`` run on the SAME inputs and the SAME block list: ``isfinite`` and
"no sentinel survived" BEFORE any cosine, ``cos > 0.999`` on ``out`` (the end-to-end suite's bar), ``atol 2e-2`` on the gated
O and the LSE (the SDPA stage test's bar), a dead row (``seq_lens[b] == 0``, or an empty list with no open tail) EXACTLY
``out = 0`` / ``O = 0`` / ``LSE = -inf``, and the two launches bitwise.  Tolerances are the dense suites', none added.

The matrix: S in {512, 2051 (the last identity length), 2052 (the first length with a dropped block), 4096, 32768} x B in
{1, 2 (a different list per entry; ``seq_lens`` ``[S, S - 37]``; the dead entry ``[S, 0]``)} x geometries {24/2 d_model 2560
(TP 1), 12/1 (TP 2), 6/1 (TP 4: the LDG norm-kernel geometry), 32/2 d_model 4096, 8/2 d_model 512 for speed} x list sources
{full (the identity below the bound), the indexer reference, a random subset in random order} x ``fuse_norm_rope`` x
``return_lse`` x dtype {bf16, f16} x ``top_k`` {512, 256, 4} x the K / V layout {the slab stride (``inplace_qkv``), compact}
x the indexer band; the degenerate rows (a ``-1`` INSIDE the count is a VALUE -- that block absent; entries BEYOND the count
are never read -- bitwise invariant to garbage there and to the ``[B, S, top_k]`` spelling; ``block_lens`` out of range is
clamped on device; an empty list attends to its tail only; one block total at S = 4 / 5 / 7 with ``B x H_kv > 1``); the
ADVERSARIAL S = 2052 cell (a dominant key planted through ``h`` inside the block the 512-list omits for row 2051 -- within
budget vs the oracle ON the list, outside it vs the oracle on the FULL set by the plant's margin; both magnitudes reported);
the write-through twin (a ``QsaSpec`` block with ``paged_kv_page_size`` is bitwise the one without); the workspace of a
``QsaSpec`` block equal to the dense block's.  The ``fuse_gate`` axis: the gated block (stage (5) inside the sparse core's
epilogue) against the oracle AND against the unfused block on the same inputs -- the gated O within ONE rounding of the
activation dtype per element, the LSE bitwise, the dead entry exactly 0, one launch fewer -- over the geometries, both dtypes,
the fully fused form (``fuse_norm_rope`` too), compact K / V, ``top_k = 4`` and a padded second entry.

PACKED SEQUENCES (``thd=True``, since 2026-10-08): the sparse core's packed arm through the block -- ONE token matrix
``[T, d_model]`` (or ``[1, T, d_model]``) holding ``B`` sequences, ``execute(seq_lens=)`` as ``[B]`` lengths or ``[B+1]``
prefix sums (``cu_seqlens``) for both length sides, ``block_ids`` ``[T, top_k]`` with every id RELATIVE TO ITS SEQUENCE.
Every cell runs the per-sequence QSA oracle (``gated_attention_block_qsa_reference_packed`` + ``compare_packed``: a
cross-sequence leak names its sequence) at the same bounds, on the packings of the THD suites -- ``[300, 128, 200]`` (bf16
AND f16), ``[2048, 4096, 6144, 8192]`` with random lists past the identity bound, a zero-length sequence in front / in the
middle / trailing, a 5-token sequence (never a 1-token one), lengths that are not multiples of 4 (a 1-3 token tail on every
sequence), one block total with ``B x H_kv > 1`` -- across the forms (the prefix sums, the ``[T, .]`` input rank,
``block_lens`` absent, no LSE) and the pipeline knobs (fused norm + RoPE, the compact K / V layout, the indexer band,
``top_k`` 4 / 256, the TP and 397B geometries); a one-sequence packing is BITWISE the dense ``B = 1, S = T`` sparse block on
``out``, the gated O and the LSE; the workspace carries the packed metadata; the lengths are REQUIRED at execute.  THD
together with the in-block indexer stays a typed decline (``test_block_qsa.py``); the paged write-through composes with the
packed sparse block (the packed write-through cell below).
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry, QsaSpec, index_k_raw_view  # noqa: E402
from cudnn.gated_attention_block.api import _cols, _SparseSdpa, _view  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_qsa_reference import (  # noqa: E402
    RefQsaGeometry,
    RefQsaSpec,
    block_ids_contract_violations,
    dense_geometry,
    full_block_ids,
    gated_attention_block_qsa_reference,
    gated_attention_block_qsa_reference_packed,
    make_qsa_inputs,
    random_block_ids,
)
from gated_block_reference import (  # noqa: E402
    RefGeometry,
    apply_partial_rope,
    compare_packed,
    gated_attention_block_reference,
    make_inputs,
    make_packed_inputs,
)

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107

COS_OUT = 0.999  # the end-to-end suite's bar on `out` (test_block_end_to_end.py)
ATOL = 2e-2  # the SDPA stage test's bar on O / LSE (test_sdpa_stage_sm107.py) -- never widened here
BS = 4

# The Qwen3.8-Flash-Next QSA layer at TP 1 / TP 2 / TP 4, the 397B sibling, and a shrunk d256 geometry for speed.
GEOM_FLASH_NEXT = dict(d_model=2560, h_q=24, h_kv=2, d_head=256, rope_dim=64)
GEOM_TP2 = dict(d_model=2560, h_q=12, h_kv=1, d_head=256, rope_dim=64)
GEOM_TP4 = dict(d_model=2560, h_q=6, h_kv=1, d_head=256, rope_dim=64)  # the LDG norm-kernel geometry (no TMA tile fits 6 rows)
GEOM_397B = dict(d_model=4096, h_q=32, h_kv=2, d_head=256, rope_dim=64)  # the GQA group AT the record's cap (16)
GEOM_SMALL = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)

BF16, F16 = torch.bfloat16, torch.float16


def _sentinel(dtype):
    return 1.5e30 if dtype == torch.bfloat16 else 6.0e4  # f16 cannot hold 1.5e30; no block output reaches 6e4


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16)


def _i32(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device="cuda")


@dataclass
class _Cell:
    """What one block run leaves behind: the per-launch (out, lse, gated O) triples, the oracle, the block, the inputs."""

    runs: list
    ref: object
    blk: object
    inp: dict
    ws: torch.Tensor
    geom: object
    ids: torch.Tensor
    lens: Optional[torch.Tensor]
    mags: dict = field(default_factory=dict)


def _run(
    geom_kw,
    batch,
    seq_len,
    *,
    dtype=BF16,
    top_k=512,
    index_band=False,
    index_source="full",
    seq_lens=None,
    block_lens=True,
    return_lse=True,
    fuse_norm_rope=False,
    inplace_qkv=None,
    seed=0,
    launches=2,
    lists=None,
    inp=None,
    ref=None,
    paged_kv_page_size=0,
    pools=None,
    fuse_gate=False,
) -> _Cell:
    """Declare, check, compile and run a QsaSpec block ``launches`` times on sentinel-filled outputs; the oracle on the same
    inputs and list.  ``lists`` overrides the generated ``(block_ids, block_lens)``; ``inp`` / ``ref`` reuse another cell's;
    ``fuse_gate`` folds stage (5) into the sparse core's epilogue (the gated O is then the SDPA's own output in the same slot)."""
    band = index_band
    qsa = QsaSpec(top_k=top_k, index_band=band)
    ref_geom = RefQsaGeometry(**geom_kw, qsa=RefQsaSpec(top_k=top_k, index_band=band))
    geom = GatedAttentionBlockGeometry(**geom_kw, qsa=qsa)
    if inp is None:
        inp = make_qsa_inputs(ref_geom, batch, seq_len, dtype=dtype, seed=seed, index_source=index_source, seq_lens=seq_lens)
    ids, lens = (inp["block_ids"], inp["block_lens"]) if lists is None else lists
    lens_arg = lens if block_lens else None
    if ref is None:
        ref = gated_attention_block_qsa_reference(
            inp["h"],
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            inp["cos"],
            inp["sin"],
            inp["w_o"],
            ref_geom,
            block_ids=ids,
            block_lens=lens_arg,
            seq_lens=seq_lens,
        )
    t = batch * seq_len
    out = torch.empty(batch, seq_len, geom.d_model, device="cuda", dtype=dtype)
    blk = GatedAttentionBlockFwd(
        inp["h"],
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        out,
        geom,
        return_lse=return_lse,
        seq_lens_present=seq_lens is not None,
        fuse_norm_rope=fuse_norm_rope,
        inplace_qkv=inplace_qkv,
        paged_kv_page_size=paged_kv_page_size,
        fuse_gate=fuse_gate,
    )
    assert isinstance(blk._sdpa, _SparseSdpa)
    assert (
        blk._sdpa.fuse_gate == fuse_gate and (blk._gate is None) == fuse_gate
    ), "fuse_gate binds the gate through the sparse stage and does not build stage (5)"
    assert blk.check_support()
    blk.compile()
    ws = torch.full((blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device="cuda")
    lse = torch.empty(batch, geom.h_q, seq_len, device="cuda", dtype=torch.float32) if return_lse else None
    sent = _sentinel(dtype)
    lay = blk._layout()
    runs = []
    for _ in range(launches):
        out.fill_(sent)
        if lse is not None:
            lse.fill_(sent)
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
            block_ids=ids,
            block_lens=lens_arg,
            **(pools or {}),
        )
        torch.cuda.synchronize()
        o_gated = _view(ws, lay.o, (t, geom.h_q, geom.d_head), dtype).view(batch, seq_len, geom.h_q, geom.d_head).clone()
        runs.append((out.clone(), None if lse is None else lse.clone(), o_gated))
    return _Cell(runs=runs, ref=ref, blk=blk, inp=inp, ws=ws, geom=geom, ids=ids, lens=lens_arg)


def _check(c: _Cell, label: str) -> dict:
    """The standard assertions of every cell -- finite, no sentinel, dead rows exact, within the budget, launches bitwise --
    and the measured magnitudes, printed."""
    out, lse, o = c.runs[0]
    ref = c.ref
    sent = _sentinel(out.dtype)
    assert torch.isfinite(out.float()).all(), f"{label}: non-finite out"
    assert not (out.float() == sent).any(), f"{label}: a sentinel survived in out"
    assert torch.isfinite(o.float()).all(), f"{label}: non-finite gated O (a row the SDPA never wrote keeps the workspace fill)"
    for out2, lse2, o2 in c.runs[1:]:
        assert torch.equal(_bits(out), _bits(out2)) and torch.equal(_bits(o), _bits(o2)), f"{label}: two launches are not bitwise on out / O"
        if lse is not None:
            assert torch.equal(lse, lse2), f"{label}: two launches are not bitwise on LSE"
    dead = ref.n_visible == 0  # [B, S]
    cos = _cos(out, ref.out)
    d_out = float((out.float() - ref.out.float()).abs().max())
    d_o = float((o.float() - ref.o_gated.float()).abs().max())
    d_lse = 0.0
    if lse is not None:
        assert not (lse == sent).any(), f"{label}: a sentinel survived in LSE"
        dead_bhs = dead[:, None, :].expand_as(lse)
        if dead.any():
            assert torch.isinf(lse[dead_bhs]).all() and (lse[dead_bhs] < 0).all(), f"{label}: a dead row's LSE must be -inf exactly"
        live = ~dead_bhs
        d_lse = float((lse[live] - ref.lse[live]).abs().max()) if live.any() else 0.0
        assert d_lse <= ATOL, f"{label}: LSE off the oracle by {d_lse} (budget {ATOL})"
    if dead.any():
        assert (o[dead] == 0).all() and (out[dead] == 0).all(), f"{label}: a dead row's O and out must be exactly 0"
    assert cos > COS_OUT, f"{label}: cos(out) {cos} <= {COS_OUT}"
    assert d_o <= ATOL, f"{label}: gated O off the oracle by {d_o} (budget {ATOL})"
    c.mags = dict(cos=cos, d_out=d_out, d_o=d_o, d_lse=d_lse, dead=int(dead.sum()))
    print(f"\n{label}: cos(out) {cos:.6f} max|d out| {d_out:.4e} max|d O_gated| {d_o:.4e} max|d LSE| {d_lse:.2e} dead rows {int(dead.sum())}")
    return c.mags


# ============================================================================ the matrix
# (id, geometry, B, S, dtype, list source, top_k, seq_lens, block_lens given, return_lse, fuse_norm_rope, inplace_qkv, index_band)
_MATRIX = [
    # the two acceptance cells: the block end to end at the Flash-Next geometry, S = 512, the full list, bf16 AND f16
    pytest.param(GEOM_FLASH_NEXT, 1, 512, BF16, "full", 512, None, True, True, False, None, False, id="fn24-2-s512-bf16-full-lse"),
    pytest.param(GEOM_FLASH_NEXT, 1, 512, F16, "full", 512, None, True, True, False, None, False, id="fn24-2-s512-f16-full-lse"),
    # the dead entry
    pytest.param(GEOM_FLASH_NEXT, 2, 512, BF16, "full", 512, (512, 0), True, True, False, None, False, id="fn24-2-s512-b2-dead-entry-bf16"),
    # S: the last identity length, top_k 256 past its bound, the indexer-reference lists, the long-S cell
    pytest.param(GEOM_FLASH_NEXT, 1, 2051, BF16, "full", 512, None, True, True, False, None, False, id="fn24-2-s2051-bf16-full-identity"),
    pytest.param(GEOM_FLASH_NEXT, 1, 2051, BF16, "synthetic", 256, None, True, True, False, None, False, id="fn24-2-s2051-bf16-synthetic-top256"),
    pytest.param(GEOM_FLASH_NEXT, 1, 4096, BF16, "indexer", 512, None, True, True, False, None, False, id="fn24-2-s4096-bf16-indexer"),
    pytest.param(GEOM_FLASH_NEXT, 1, 32768, BF16, "synthetic", 512, None, True, True, False, None, False, id="fn24-2-s32768-bf16-synthetic"),
    # B = 2: a different list per entry; a padded entry; block_lens absent (the kernel's derived count)
    pytest.param(GEOM_FLASH_NEXT, 2, 512, BF16, "synthetic", 512, (512, 475), True, True, False, None, False, id="fn24-2-s512-b2-synthetic-lens-475"),
    pytest.param(GEOM_FLASH_NEXT, 2, 512, BF16, "synthetic", 512, None, False, True, False, None, False, id="fn24-2-s512-b2-synthetic-no-block-lens"),
    # the pipeline knobs: fused norm + RoPE (no band), no LSE, the compact K / V layout, the indexer band
    pytest.param(GEOM_FLASH_NEXT, 1, 512, BF16, "synthetic", 512, None, True, True, True, None, False, id="fn24-2-s512-fuse-norm-rope"),
    pytest.param(GEOM_FLASH_NEXT, 1, 512, BF16, "full", 512, None, True, False, False, None, False, id="fn24-2-s512-no-lse"),
    pytest.param(GEOM_FLASH_NEXT, 1, 512, BF16, "full", 512, None, True, True, False, False, False, id="fn24-2-s512-compact-kv"),
    pytest.param(GEOM_FLASH_NEXT, 1, 512, BF16, "synthetic", 512, None, True, True, False, None, True, id="fn24-2-s512-index-band"),
    # the TP geometries (12/1, 6/1), the 397B sibling (GQA 16 = the record's cap), the shrunk geometry
    pytest.param(GEOM_TP2, 1, 512, BF16, "synthetic", 512, None, True, True, False, None, False, id="tp2-12-1-s512-synthetic"),
    pytest.param(GEOM_TP2, 1, 512, F16, "full", 512, None, True, True, False, None, False, id="tp2-12-1-s512-f16-full"),
    pytest.param(GEOM_TP4, 2, 512, BF16, "synthetic", 512, None, True, True, False, None, False, id="tp4-6-1-s512-b2-synthetic-ldg-norm"),
    pytest.param(GEOM_397B, 1, 512, BF16, "synthetic", 512, None, True, True, False, None, False, id="g397b-32-2-s512-synthetic-gqa16"),
    pytest.param(GEOM_SMALL, 2, 512, BF16, "synthetic", 512, None, True, True, False, False, False, id="small8-2-s512-b2-synthetic-compact-kv"),
    pytest.param(GEOM_SMALL, 1, 300, BF16, "synthetic", 4, None, True, True, False, None, False, id="small8-2-s300-top4-synthetic"),
    pytest.param(GEOM_SMALL, 2, 300, F16, "full", 512, (300, 0), True, True, False, None, False, id="small8-2-s300-b2-f16-dead-entry"),
    pytest.param(GEOM_SMALL, 1, 2052, BF16, "synthetic", 512, None, True, True, False, None, False, id="small8-2-s2052-synthetic-dropped-block"),
]


@requires_rubin
@pytest.mark.parametrize("geom_kw, batch, seq_len, dtype, source, top_k, seq_lens, block_lens, return_lse, fuse_norm_rope, inplace_qkv, index_band", _MATRIX)
def test_qsa_block_matches_the_oracle(
    request, geom_kw, batch, seq_len, dtype, source, top_k, seq_lens, block_lens, return_lse, fuse_norm_rope, inplace_qkv, index_band
):
    """One cell of the matrix: the block vs the oracle on the same list, every standard assertion, the magnitudes printed."""
    lens_t = None if seq_lens is None else _i32(list(seq_lens))
    c = _run(
        geom_kw,
        batch,
        seq_len,
        dtype=dtype,
        top_k=top_k,
        index_band=index_band,
        index_source=source,
        seq_lens=lens_t,
        block_lens=block_lens,
        return_lse=return_lse,
        fuse_norm_rope=fuse_norm_rope,
        inplace_qkv=inplace_qkv,
        seed=seq_len + batch,
    )
    # the list is contract-clean (the test-side detector; the library never reads a list on the host)
    pos = torch.arange(seq_len, device="cuda").expand(batch, seq_len).reshape(-1)
    kv = torch.full((batch * seq_len,), seq_len, dtype=torch.long, device="cuda") if lens_t is None else lens_t.long().repeat_interleave(seq_len)
    assert (
        block_ids_contract_violations(
            c.ids.reshape(batch * seq_len, -1), pos, kv, block_size=BS, block_lens=c.lens.reshape(-1) if c.lens is not None else None, top_k=top_k
        )
        == []
    )
    m = _check(c, request.node.callspec.id)
    if seq_lens is not None and 0 in seq_lens:
        b = list(seq_lens).index(0)
        out, lse, o = c.runs[0]
        assert (out[b] == 0).all() and (o[b] == 0).all() and torch.isinf(lse[b]).all() and (lse[b] < 0).all(), "the dead entry must be exactly 0 / -inf"
        assert m["dead"] >= seq_len
    if source == "full" and seq_len <= QsaSpec(top_k=top_k).identity_bound and seq_lens is None:
        # the identity: with the full list the sparse block is the DENSE function within the budget (not bitwise: another summation order)
        dense_ref = gated_attention_block_reference(
            **{k: c.inp[k] for k in ("h", "w_qkvg", "w_q_norm", "w_k_norm", "cos", "sin", "w_o")}, geom=RefGeometry(**geom_kw)
        )
        cos_d = _cos(c.runs[0][0], dense_ref.out)
        d_lse = float((c.runs[0][1] - dense_ref.lse).abs().max()) if return_lse else 0.0
        print(f"identity vs the dense oracle: cos(out) {cos_d:.6f} max|d LSE| {d_lse:.2e}")
        assert cos_d > COS_OUT and d_lse <= ATOL


# ============================================================================ the degenerate rows (the shrunk geometry)
@requires_rubin
def test_a_minus_one_inside_the_count_is_a_masked_block():
    """DESIGN 4.6: a ``-1`` at an index BELOW the count contributes no key -- a VALUE the oracle reproduces with that block
    absent (``qsa_visible_mask``); the distance from the full-list function is reported, never asserted."""
    geom_kw, B, S = GEOM_SMALL, 1, 600
    pos = torch.arange(S, device="cuda").expand(B, S)
    ids, lens = full_block_ids(pos, 512, BS)
    holed = ids.clone()
    rows = torch.arange(S, device="cuda") >= 100
    cols = torch.arange(512, device="cuda") % 3 == 0
    hole = rows[None, :, None] & cols[None, None, :] & (holed >= 0)
    holed[hole] = -1
    assert int(hole.sum()) > 0 and block_ids_contract_violations(holed[0], pos[0], torch.full((S,), S, device="cuda"), block_size=BS, block_lens=lens[0]) != []
    c = _run(geom_kw, B, S, seed=11, lists=(holed.contiguous(), lens.contiguous()))
    m = _check(c, "small8-2 S=600 -1 inside the count (every 3rd entry of rows >= 100)")
    full = _run(geom_kw, B, S, seed=11, inp=c.inp, lists=(ids.contiguous(), lens.contiguous()))
    _check(full, "small8-2 S=600 the full list")
    teeth = float((c.runs[0][0][0, 100:].float() - full.runs[0][0][0, 100:].float()).abs().max())
    print(f"holed vs full list on rows >= 100: max|d out| {teeth:.4f} (the masked blocks move the function; reported)")
    assert m["d_o"] <= ATOL


@requires_rubin
def test_entries_beyond_the_count_are_never_read_and_both_list_spellings_agree():
    """DESIGN 4.6: entries at an index at or beyond the count are never read -- garbage there (0, 7, 1e6, -5) leaves out / O /
    LSE BITWISE the canonical run, with ``block_lens`` given or absent (the derived default count = the full list's), and the
    ``[B, S, top_k]`` spelling is bitwise the flat ``[T, top_k]`` one (an exact view, no copy)."""
    geom_kw, B, S = GEOM_SMALL, 2, 300
    pos = torch.arange(S, device="cuda").expand(B, S)
    gen = torch.Generator(device="cuda").manual_seed(5)
    ids, lens = random_block_ids(pos, 512, BS, generator=gen)
    canon = _run(geom_kw, B, S, seed=13, lists=(ids.contiguous(), lens.contiguous()))
    _check(canon, "small8-2 B=2 S=300 shuffled canonical")
    garbage = ids.clone()
    beyond = torch.arange(512, device="cuda")[None, None, :] >= lens.long()[:, :, None]
    fill = torch.tensor([0, 7, 1_000_000, -5], dtype=torch.int32, device="cuda")[torch.arange(512, device="cuda") % 4].expand(B, S, 512)
    garbage[beyond] = fill[beyond]
    assert block_ids_contract_violations(garbage.reshape(B * S, 512), pos.reshape(-1), torch.full((B * S,), S, device="cuda"), block_size=BS) != []
    o_c, l_c, g_c = canon.runs[0]
    for label, lists_, bl in (
        ("garbage beyond the count, block_lens given", (garbage.reshape(B * S, 512).contiguous(), lens.reshape(B * S).contiguous()), True),
        ("garbage beyond the count, block_lens absent", (garbage.contiguous(), lens.contiguous()), False),
        ("the canonical list, block_lens absent", (ids.reshape(B * S, 512).contiguous(), None), False),
    ):
        c = _run(geom_kw, B, S, seed=13, inp=canon.inp, ref=canon.ref, lists=lists_, block_lens=bl)
        _check(c, f"small8-2 B=2 S=300 {label}")
        o, l, g = c.runs[0]
        assert torch.equal(_bits(o), _bits(o_c)) and torch.equal(l, l_c) and torch.equal(_bits(g), _bits(g_c)), label


@requires_rubin
def test_block_lens_out_of_range_is_clamped_on_device_and_an_empty_list_attends_to_its_tail():
    """DESIGN 4.6: ``block_lens`` negative (-7 on odd rows) is clamped to 0 -> tail-only attention, DEAD (``out = 0``, ``LSE =
    -inf``) where there is no open tail (``(pos + 1) % 4 == 0``); above the derived count (1e5 on even rows) it is clamped to
    the default; an all-``-1`` list with ``block_lens = 0`` attends to its tail only on every row (dead where no tail)."""
    geom_kw, B, S = GEOM_SMALL, 1, 400
    pos = torch.arange(S, device="cuda").expand(B, S)
    ids, lens = full_block_ids(pos, 512, BS)
    bad = torch.where(
        torch.arange(S, device="cuda") % 2 == 1,
        torch.full((S,), -7, dtype=torch.int32, device="cuda"),
        torch.full((S,), 100_000, dtype=torch.int32, device="cuda"),
    )[None]
    c = _run(geom_kw, B, S, seed=17, lists=(ids.contiguous(), bad.contiguous()))
    m = _check(c, "small8-2 S=400 block_lens -7 (odd rows) / 1e5 (even rows)")
    expect_dead = int(((torch.arange(S) % 2 == 1) & ((torch.arange(S) + 1) % 4 == 0)).sum())
    assert m["dead"] == expect_dead == 100
    empty = torch.full_like(ids, -1)
    zero = torch.zeros_like(lens)
    e = _run(geom_kw, B, S, seed=17, inp=c.inp, lists=(empty.contiguous(), zero.contiguous()))
    me = _check(e, "small8-2 S=400 empty lists (block_lens = 0): the tail only")
    assert me["dead"] == int(((torch.arange(S) + 1) % 4 == 0).sum()) == 100


@requires_rubin
@pytest.mark.parametrize("seq_len", [4, 5, 7])
def test_one_block_total_with_two_sequences(seq_len):
    """DESIGN 4.6: one block total (S = 4) and S = 5 / 7 (a 1- / 3-token tail), ``B x H_kv > 1`` with one block -- the kernel's
    ring at trip count 1 under the whole block (projection, norm + RoPE, the sparse SDPA, the gate, the out projection)."""
    c = _run(GEOM_SMALL, 2, seq_len, seed=seq_len)
    _check(c, f"small8-2 B=2 S={seq_len}")


# ============================================================================ the adversarial cell
def _inverse_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """Undo ``apply_partial_rope`` at one position: rotate by the negative angle (the same table with ``-sin``)."""
    return apply_partial_rope(x[None, None, None, :], cos[None, None, :], -sin[None, None, :], rope_dim)[0, 0, 0]


@requires_rubin
def test_adversarial_s2052_planted_key_in_the_omitted_block():
    """DESIGN 4.3 at the block level.  At S = 2052 query 2051 has 513 complete blocks and the 512-wide full list omits block
    512 (tokens 2048..2051).  A dominant key is planted THROUGH ``h``: for one query head of each KV head, ``h[2049]`` /
    ``h[2050]`` is solved (least squares on the KV head's W_k rows) so that its post-norm key, rotated at its position,
    equals the row's post-norm, post-RoPE query -- the score beats every other key by >= 8 nats and the planted key holds
    > 99.9 % of the full-set mass (checked on the oracle's own post-RoPE Q / K).  The block is then within the budget vs the
    oracle ON the list everywhere, and at row 2051 for the planted heads OUTSIDE the budget vs the oracle on the FULL set
    (cos < 0.5 on the head's gated O, the diff >= 10 budgets, the LSE apart by more than the budget); every row below 2051 is
    within the budget vs the full set.  A block that ignored the list could not pass both.  Both magnitudes are reported."""
    geom_kw, B, S, row = GEOM_FLASH_NEXT, 1, 2052, 2051
    geom = GatedAttentionBlockGeometry(**geom_kw, qsa=QsaSpec())
    ref_geom = RefQsaGeometry(**geom_kw, qsa=RefQsaSpec())
    G = geom.gqa_ratio
    inp = make_qsa_inputs(ref_geom, B, S, dtype=BF16, seed=7, index_source="full")
    ids, lens = inp["block_ids"], inp["block_lens"]
    assert int(lens[0, row]) == 512 and int(ids[0, row].max()) == 511, "row 2051 lists blocks 0..511 and omits block 512"
    ref0 = gated_attention_block_qsa_reference(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], ref_geom, block_ids=ids, block_lens=lens
    )
    plants = [(0, 2049), (G, 2050)]  # (query head, planted token inside block 512) -- one per KV head
    o_k = geom.qkvg_offsets[2]
    for hq, tok in plants:
        kv = hq // G
        q_rot = ref0.q[0, row, hq].float()  # post-norm, post-RoPE at position 2051 (unit RMS: the norm weight is 1)
        k_pre = _inverse_partial_rope(q_rot, inp["cos"][0, tok].float(), inp["sin"][0, tok].float(), geom.rope_dim)
        w_k = inp["w_qkvg"][o_k + kv * geom.d_head : o_k + (kv + 1) * geom.d_head].float()  # [D, d_model]: k_pre = W_k h
        h_new = torch.linalg.pinv(w_k) @ k_pre  # the least-norm h with W_k h = k_pre (RMSNorm is scale-free: the direction is what matters)
        inp["h"][0, tok] = h_new.to(BF16)
    ref = gated_attention_block_qsa_reference(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], ref_geom, block_ids=ids, block_lens=lens
    )
    # the plant's margin and mass on the full set, from the oracle's own post-RoPE Q / K
    for hq, tok in plants:
        kv = hq // G
        sc = (ref.q[0, row, hq].float() @ ref.k[0, : row + 1, kv].float().t()) * geom.scale
        others = torch.cat([sc[:tok], sc[tok + 1 :]])
        margin = float(sc[tok] - others.max())
        mass = float(1.0 / (1.0 + torch.exp(others - sc[tok]).sum()))
        print(f"\nplant head {hq} token {tok}: score {float(sc[tok]):.2f}, margin {margin:.2f} nats, full-set mass {mass:.5f}")
        assert margin >= 8.0 and mass > 0.999, f"head {hq}: plant margin {margin:.2f} nats, mass {mass:.5f}"
    c = _run(geom_kw, B, S, seed=7, inp=inp, ref=ref, lists=(ids, lens))
    m = _check(c, "fn24-2 S=2052 adversarial: vs the oracle ON the list")
    # the FULL set: every one of the 513 complete blocks (the oracle at width 513 -- RefQsaSpec.top_k carries the width)
    pos = torch.arange(S, device="cuda").expand(B, S)
    ids_full, lens_full = full_block_ids(pos, 513, BS)
    assert int(ids_full[0, row].max()) == 512
    ref_full = gated_attention_block_qsa_reference(
        inp["h"],
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        ref_geom,
        block_ids=ids_full.contiguous(),
        block_lens=lens_full.contiguous(),
        spec=RefQsaSpec(top_k=513),
    )
    out, lse, o = c.runs[0]
    below = slice(0, row)
    cos_below = _cos(out[0, below], ref_full.out[0, below])
    d_o_below = float((o[0, below].float() - ref_full.o_gated[0, below].float()).abs().max())
    d_lse_below = float((lse[0, :, below] - ref_full.lse[0, :, below]).abs().max())
    print(f"rows < 2051 vs the FULL-set oracle: cos(out) {cos_below:.6f} max|d O_gated| {d_o_below:.4e} max|d LSE| {d_lse_below:.2e}")
    assert cos_below > COS_OUT and d_o_below <= ATOL and d_lse_below <= ATOL
    report = []
    for hq, tok in plants:
        ok, full_h = o[0, row, hq].float(), ref_full.o_gated[0, row, hq].float()
        diff = float((ok - full_h).abs().max())
        cos_h = float(torch.nn.functional.cosine_similarity(ok, full_h, dim=0))
        d_lse = float(abs(lse[0, hq, row] - ref_full.lse[0, hq, row]))
        assert (
            cos_h < 0.5 and diff >= 10 * ATOL and d_lse > ATOL
        ), f"head {hq}: cos {cos_h:.3f} diff {diff:.4f} dLSE {d_lse:.3f} -- the omitted block was attended"
        report.append(f"head {hq}: gated-O diff {diff:.4f} cos {cos_h:.3f}, LSE apart {d_lse:.3f}")
    d_out_row = float((out[0, row].float() - ref_full.out[0, row].float()).abs().max())
    assert d_out_row > ATOL
    print(f"row 2051 vs the FULL-set oracle: {'; '.join(report)}; out row max|d| {d_out_row:.4f} (within the list's budget: {m['d_o']:.4e} / {m['d_lse']:.2e})")


# ============================================================================ serving twin + the workspace pin
def _nan_pool(n_pages, h, ps, d, dtype=BF16):
    return torch.full((n_pages, h, ps, d), float("nan"), device="cuda", dtype=dtype)


@requires_rubin
def test_a_qsa_block_with_write_through_is_bitwise_the_one_without():
    """The write-through acceptance on the SPARSE block (the dense half is test_block_cache_write.py's): a ``QsaSpec`` block
    with the indexer band declared with ``paged_kv_page_size=16`` writes the post-RoPE K / V and the raw indexer key through
    into the pools at ``slot_mapping`` and its ``out`` / LSE are BITWISE the same ``QsaSpec`` block without write-through; the
    pools are bitwise the slab bands attention consumed; padded slots write nothing; ``get_workspace_size`` is unchanged;
    the extra stage sits right before the sparse SDPA."""
    geom_kw, B, S, ps = GEOM_FLASH_NEXT, 1, 512, 16
    plain = _run(geom_kw, B, S, index_band=True, index_source="synthetic", seed=3)
    _check(plain, "fn24-2 band S=512 plain")
    t, geom = B * S, plain.geom
    n_pages = t // ps + 2
    gen = torch.Generator(device="cuda").manual_seed(0)
    slot = torch.randperm(n_pages * ps, generator=gen, device="cuda")[:t].to(torch.int32)
    slot[::13] = -1
    k_cache, v_cache = _nan_pool(n_pages, geom.h_kv, ps, geom.d_head), _nan_pool(n_pages, geom.h_kv, ps, geom.d_head)
    index_cache = torch.full((n_pages, ps, geom.qsa.index_head_dim), float("nan"), device="cuda", dtype=BF16)
    wt = _run(
        geom_kw,
        B,
        S,
        index_band=True,
        index_source="synthetic",
        seed=3,
        inp=plain.inp,
        ref=plain.ref,
        lists=(plain.ids, plain.lens),
        paged_kv_page_size=ps,
        pools=dict(k_cache=k_cache, v_cache=v_cache, slot_mapping=slot, index_k_raw=index_cache),
    )
    _check(wt, "fn24-2 band S=512 write-through")
    assert torch.equal(_bits(wt.runs[0][0]), _bits(plain.runs[0][0])) and torch.equal(
        wt.runs[0][1], plain.runs[0][1]
    ), "write-through changed the sparse block's output"
    assert torch.equal(_bits(wt.runs[0][2]), _bits(plain.runs[0][2]))
    assert wt.blk.get_workspace_size() == plain.blk.get_workspace_size()
    lay = wt.blk._layout()
    proj = _view(wt.ws, lay.proj, (t, geom.n_qkvg), BF16)
    k_rows, v_rows = _cols(proj, geom.qkvg_offsets[2], geom.h_kv, geom.d_head), _cols(proj, geom.qkvg_offsets[3], geom.h_kv, geom.d_head)
    live = slot >= 0
    s_ = slot[live].long()
    assert torch.equal(_bits(k_cache[s_ // ps, :, s_ % ps, :]), _bits(k_rows[live])) and torch.equal(
        _bits(v_cache[s_ // ps, :, s_ % ps, :]), _bits(v_rows[live])
    )
    raw = index_k_raw_view(proj, geom, B, S)
    assert torch.equal(_bits(index_cache[s_ // ps, s_ % ps, :]), _bits(raw[0, live, 0]))
    for pool in (k_cache, v_cache):
        assert (~torch.isnan(pool.float())).all(-1).sum().item() == int(live.sum()) * geom.h_kv
    assert wt.blk._stages.index(wt.blk._cache_write) == wt.blk._stages.index(wt.blk._sdpa) - 1


@requires_rubin
def test_a_qsa_block_workspace_equals_the_dense_blocks():
    """The sparse stage folds 0 bytes of scratch (the kernel has no GMEM workspace), so a ``QsaSpec`` block without the band
    needs exactly the dense block's workspace at the same geometry -- and its stage list has the same length."""
    c = _run(GEOM_SMALL, 2, 256, seed=1)
    _check(c, "small8-2 B=2 S=256 full")
    geom = GatedAttentionBlockGeometry(**GEOM_SMALL)
    dense_inp = make_inputs(RefGeometry(**GEOM_SMALL), batch=2, seq_len=256, dtype=BF16)
    out = torch.empty(2, 256, geom.d_model, device="cuda", dtype=BF16)
    dense = GatedAttentionBlockFwd(
        dense_inp["h"],
        dense_inp["w_qkvg"],
        dense_inp["w_q_norm"],
        dense_inp["w_k_norm"],
        dense_inp["cos"],
        dense_inp["sin"],
        dense_inp["w_o"],
        out,
        geom,
        return_lse=True,
    )
    dense.check_support()
    dense.compile()
    assert c.blk._sdpa.scratch_workspace_bytes() == 0
    assert c.blk.get_workspace_size() == dense.get_workspace_size()
    assert len(c.blk._stages) == len(dense._stages)


# ============================================================================ THD: packed sequences through the sparse stage
def _packed_lists(lens, top_k: int, source: str, seed: int = 0, device="cuda") -> tuple:
    """Per-sequence lists for a packing, concatenated along the token axis: ``[T, top_k]`` ids RELATIVE TO THEIR SEQUENCE (block
    ``j`` of sequence ``b`` = its tokens ``[4j, 4j + 4)``) and the ``[T]`` counts -- the oracle's own builders run per sequence
    over that sequence's positions, as the standalone adapter's packed cells build them.  ``source``: ``"full"`` (every complete
    block below the identity bound) or ``"synthetic"`` (a random subset in random order).  An empty sequence contributes no rows."""
    g = torch.Generator(device=device).manual_seed(int(seed) + 0x5EED)
    ids, cnt = [], []
    for n in lens:
        if not int(n):
            continue
        pos = torch.arange(int(n), device=device)[None]
        i, c = full_block_ids(pos, top_k, BS) if source == "full" else random_block_ids(pos, top_k, BS, generator=g)
        ids.append(i[0])
        cnt.append(c[0])
    return torch.cat(ids).contiguous(), torch.cat(cnt).contiguous()


@dataclass
class _ThdCell:
    """What one packed block run leaves behind: per launch ``(out [1, T, d_model], lse [1, H_q, T] | None, o_gated [T, H_q, D])``,
    the per-sequence oracle list, the block, the inputs, the packing metadata, the lists and the lengths tensor it ran with."""

    runs: list
    refs: Optional[list]
    blk: object
    inp: dict
    meta: dict
    ws: torch.Tensor
    geom: object
    ids: torch.Tensor
    lens: Optional[torch.Tensor]
    seq_lens: torch.Tensor
    mags: dict = field(default_factory=dict)


def _run_thd(
    geom_kw,
    lens,
    *,
    dtype=BF16,
    top_k=512,
    source="full",
    cu=False,
    rank2=False,
    block_lens=True,
    return_lse=True,
    fuse_norm_rope=False,
    inplace_qkv=None,
    index_band=False,
    max_seq_len=None,
    seed=0,
    launches=2,
    lists=None,
    inp=None,
    refs=True,
    fuse_gate=False,
    paged_kv_page_size=0,
    pools=None,
) -> _ThdCell:
    """Declare, check, compile and run a PACKED (``thd=True``) ``QsaSpec`` block ``launches`` times on sentinel-filled outputs
    and the per-sequence QSA oracle on the same inputs and lists.  ``lens`` is the packing (ints, zero-length sequences
    allowed; ``max_seq_len`` defaults to the longest); ``cu`` hands the ``[B+1]`` prefix sums over instead of the ``[B]``
    lengths, ``rank2`` the ``[T, .]`` input spelling; ``lists`` overrides the generated ``(block_ids, block_lens)``; ``inp``
    reuses another cell's ``(inputs, meta)``; ``refs=False`` skips the oracle (the bitwise pins); ``fuse_gate`` (appended) folds
    stage (5) into the sparse core's epilogue on the packed GATE slice (the gated O is then the SDPA's own output in the same slot);
    ``paged_kv_page_size`` / ``pools`` (appended) declare the write-through and hand ``execute`` the pools (``k_cache`` / ``v_cache`` /
    ``slot_mapping``)."""
    ref_geom = RefQsaGeometry(**geom_kw, qsa=RefQsaSpec(top_k=top_k, index_band=index_band))
    geom = GatedAttentionBlockGeometry(**geom_kw, qsa=QsaSpec(top_k=top_k, index_band=index_band))
    if inp is None:
        inp, meta = make_packed_inputs(dense_geometry(ref_geom), lens, dtype=dtype, seed=seed, max_seq_len=max_seq_len)
        if index_band:
            # the INDEX band rows of W_qkvg, drawn as make_qsa_inputs draws them (a second generator); the raw key the band
            # projects is unused under packed sequences (the in-block indexer is a typed decline there)
            g2 = torch.Generator(device="cuda").manual_seed(int(seed) + 0x9E37)
            w_i = (torch.randn(ref_geom.qsa.index_band_width, ref_geom.d_model, generator=g2, device="cuda", dtype=torch.float32) * 0.02).to(dtype)
            inp["w_qkvg"] = torch.cat([inp["w_qkvg"], w_i], dim=0)
    else:
        inp, meta = inp
    t = meta["t"]
    ids, cnt = _packed_lists(lens, top_k, source, seed=seed) if lists is None else lists
    assert ids.shape == (t, top_k) and cnt.shape == (t,)
    lens_arg = cnt if block_lens else None
    seq_lens = meta["cu_seqlens"] if cu else meta["seq_lens"]
    ref_list = (
        gated_attention_block_qsa_reference_packed(
            inp["h"],
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            inp["cos"],
            inp["sin"],
            inp["w_o"],
            ref_geom,
            meta["lens"],
            block_ids=ids,
            block_lens=lens_arg,
        )
        if refs
        else None
    )
    h, cos, sin = inp["h"], inp["cos"], inp["sin"]
    out = torch.empty(1, t, geom.d_model, device="cuda", dtype=dtype)
    if rank2:
        h, cos, sin, out = h[0], cos[0], sin[0], out[0]
    blk = GatedAttentionBlockFwd(
        h,
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        cos,
        sin,
        inp["w_o"],
        out,
        geom,
        return_lse=return_lse,
        fuse_norm_rope=fuse_norm_rope,
        inplace_qkv=inplace_qkv,
        thd=True,
        num_sequences=meta["b"],
        max_seq_len=meta["max_seq_len"],
        cu_seqlens=cu,
        fuse_gate=fuse_gate,
        paged_kv_page_size=paged_kv_page_size,
    )
    assert isinstance(blk._sdpa, _SparseSdpa) and blk._sdpa.thd and (blk.batch, blk.seq_len) == (1, t)
    assert (
        blk._sdpa.fuse_gate == fuse_gate and (blk._gate is None) == fuse_gate
    ), "fuse_gate binds the gate through the packed sparse stage and does not build stage (5)"
    assert blk.check_support()
    blk.compile()
    ws = torch.full((blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device="cuda")
    lse = torch.empty(1, geom.h_q, t, device="cuda", dtype=torch.float32) if return_lse else None
    sent = _sentinel(dtype)
    lay = blk._layout()
    runs = []
    for _ in range(launches):
        out.fill_(sent)
        if lse is not None:
            lse.fill_(sent)
        blk.execute(
            h,
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            cos,
            sin,
            inp["w_o"],
            out,
            ws,
            seq_lens=seq_lens,
            lse=lse,
            block_ids=ids,
            block_lens=lens_arg,
            **(pools or {}),
        )
        torch.cuda.synchronize()
        o_gated = _view(ws, lay.o, (t, geom.h_q, geom.d_head), dtype).clone()
        runs.append((out.reshape(1, t, geom.d_model).clone(), None if lse is None else lse.clone(), o_gated))
    return _ThdCell(runs=runs, refs=ref_list, blk=blk, inp=inp, meta=meta, ws=ws, geom=geom, ids=ids, lens=lens_arg, seq_lens=seq_lens)


def _check_thd(c: _ThdCell, label: str) -> dict:
    """The standard assertions PER SEQUENCE: finite + no sentinel over the WHOLE packed output first (every row belongs to a
    sequence under ``sum == T``), the launches bitwise, then every non-empty sequence against ITS OWN oracle through
    ``compare_packed`` (the verdicts collected, a cross-sequence leak names its sequence): ``cos(out) > 0.999``, the gated O
    and the LSE within ``atol 2e-2``, no row dead (self-attention: every token sees itself).  The worst magnitudes over the
    sequences are printed and returned."""
    out, lse, o = c.runs[0]
    sent = _sentinel(out.dtype)
    assert torch.isfinite(out.float()).all(), f"{label}: non-finite out"
    assert not (out.float() == sent).any(), f"{label}: a sentinel survived in out"
    assert torch.isfinite(o.float()).all(), f"{label}: non-finite gated O (a row the SDPA never wrote keeps the workspace fill)"
    for out2, lse2, o2 in c.runs[1:]:
        assert torch.equal(_bits(out), _bits(out2)) and torch.equal(_bits(o), _bits(o2)), f"{label}: two launches are not bitwise on out / O"
        if lse is not None:
            assert torch.equal(lse, lse2), f"{label}: two launches are not bitwise on LSE"
    if lse is not None:
        assert not (lse == sent).any(), f"{label}: a sentinel survived in LSE"
    mags = dict(cos=1.0, d_out=0.0, d_o=0.0, d_lse=0.0, seqs=0)

    def check(i, lo, hi, ref):
        assert (ref.n_visible[0] > 0).all(), "a packed sequence has no dead row under self-attention"
        cos = _cos(out[0, lo:hi], ref.out[0])
        d_out = float((out[0, lo:hi].float() - ref.out[0].float()).abs().max())
        d_o = float((o[lo:hi].float() - ref.o_gated[0].float()).abs().max())
        d_lse = 0.0 if lse is None else float((lse[0, :, lo:hi] - ref.lse[0]).abs().max())
        mags.update(
            cos=min(mags["cos"], cos), d_out=max(mags["d_out"], d_out), d_o=max(mags["d_o"], d_o), d_lse=max(mags["d_lse"], d_lse), seqs=mags["seqs"] + 1
        )
        assert cos > COS_OUT, f"cos(out) {cos:.6f} <= {COS_OUT}"
        assert d_o <= ATOL, f"gated O off the oracle by {d_o} (budget {ATOL})"
        assert d_lse <= ATOL, f"LSE off the oracle by {d_lse} (budget {ATOL})"

    failures = compare_packed(c.refs, c.meta["lens"], check)
    assert not failures, f"{label}:\n" + "\n".join(failures)
    c.mags = mags
    print(
        f"\n{label}: lens {c.meta['lens']} ({mags['seqs']} live sequences): min cos(out) {mags['cos']:.6f} max|d out| {mags['d_out']:.4e} "
        f"max|d O_gated| {mags['d_o']:.4e} max|d LSE| {mags['d_lse']:.2e}"
    )
    return mags


# (id, geometry, packing, dtype, list source, top_k, cu_seqlens, rank-2 inputs, block_lens given, return_lse, fuse_norm_rope, inplace_qkv, index_band)
_THD_MATRIX = [
    # the contract's packings, bf16 AND f16, at the block-sparse model's geometry; random lists past the identity bound
    pytest.param(GEOM_FLASH_NEXT, [300, 128, 200], BF16, "full", 512, False, False, True, True, False, None, False, id="fn24-2-three-seqs-bf16-full"),
    pytest.param(GEOM_FLASH_NEXT, [300, 128, 200], F16, "full", 512, False, False, True, True, False, None, False, id="fn24-2-three-seqs-f16-full"),
    pytest.param(
        GEOM_FLASH_NEXT, [2048, 4096, 6144, 8192], BF16, "synthetic", 512, False, False, True, True, False, None, False, id="fn24-2-long-four-bf16-synthetic"
    ),
    # a zero-length sequence in front / in the middle / trailing; a 5-token sequence (bf16 and f16); 1-3 token tails; one block total
    pytest.param(GEOM_SMALL, [0, 300, 200], BF16, "full", 512, False, False, True, True, False, None, False, id="small8-2-empty-front"),
    pytest.param(GEOM_SMALL, [300, 0, 200], BF16, "synthetic", 512, False, False, True, True, False, None, False, id="small8-2-empty-middle-synthetic"),
    pytest.param(GEOM_SMALL, [300, 200, 0], BF16, "full", 512, True, False, True, True, False, None, False, id="small8-2-empty-trailing-cu"),
    pytest.param(GEOM_SMALL, [5, 300, 128], BF16, "full", 512, False, False, True, True, False, None, False, id="small8-2-five-token-bf16"),
    pytest.param(GEOM_SMALL, [5, 300, 128], F16, "full", 512, False, False, True, True, False, None, False, id="small8-2-five-token-f16"),
    pytest.param(GEOM_SMALL, [4097, 13, 2051], BF16, "synthetic", 512, False, False, True, True, False, None, False, id="small8-2-tails-4097-13-2051"),
    pytest.param(GEOM_SMALL, [4, 5, 7], BF16, "full", 512, False, False, True, True, False, None, False, id="small8-2-one-block-4-5-7"),
    # the forms: the [B+1] prefix sums with the [T, .] input rank, block_lens absent (the kernel's derived count), no LSE
    pytest.param(GEOM_SMALL, [300, 128, 200], BF16, "synthetic", 512, True, True, True, True, False, None, False, id="small8-2-cu-seqlens-rank2"),
    pytest.param(GEOM_SMALL, [300, 128, 200], BF16, "full", 512, False, False, False, True, False, None, False, id="small8-2-no-block-lens"),
    pytest.param(GEOM_SMALL, [300, 128, 200], BF16, "synthetic", 512, False, False, True, False, False, None, False, id="small8-2-no-lse"),
    # the pipeline knobs: fused norm + RoPE (no band), the compact K / V layout, the indexer band, top_k 4 / 256
    pytest.param(GEOM_FLASH_NEXT, [300, 128, 200], BF16, "synthetic", 512, False, False, True, True, True, None, False, id="fn24-2-fuse-norm-rope"),
    pytest.param(GEOM_SMALL, [300, 128, 200], BF16, "synthetic", 512, False, False, True, True, False, False, False, id="small8-2-compact-kv"),
    pytest.param(GEOM_SMALL, [300, 128, 200], BF16, "synthetic", 512, False, False, True, True, False, None, True, id="small8-2-index-band"),
    pytest.param(GEOM_SMALL, [300, 128, 200], BF16, "synthetic", 4, False, False, True, True, False, None, False, id="small8-2-top4"),
    pytest.param(GEOM_SMALL, [2051, 13, 2052], BF16, "synthetic", 256, False, False, True, True, False, None, False, id="small8-2-top256-2051-2052"),
    # the TP geometries and the 397B sibling (the GQA group at the record's cap)
    pytest.param(GEOM_TP2, [300, 128, 200], F16, "full", 512, False, False, True, True, False, None, False, id="tp2-12-1-f16-full"),
    pytest.param(GEOM_TP4, [300, 128, 200], BF16, "synthetic", 512, False, False, True, True, False, None, False, id="tp4-6-1-synthetic-ldg-norm"),
    pytest.param(GEOM_397B, [300, 128, 200], BF16, "synthetic", 512, False, False, True, True, False, None, False, id="g397b-32-2-synthetic-gqa16"),
]


@requires_rubin
@pytest.mark.parametrize("geom_kw, lens, dtype, source, top_k, cu, rank2, block_lens, return_lse, fuse_norm_rope, inplace_qkv, index_band", _THD_MATRIX)
def test_thd_qsa_block_matches_the_per_sequence_oracle(
    request, geom_kw, lens, dtype, source, top_k, cu, rank2, block_lens, return_lse, fuse_norm_rope, inplace_qkv, index_band
):
    """Every sequence of a packed ``QsaSpec`` block against its own oracle (``_check_thd``): the contract's packings, the
    degenerate ones, every form and knob of the matrix above."""
    c = _run_thd(
        geom_kw,
        lens,
        dtype=dtype,
        top_k=top_k,
        source=source,
        cu=cu,
        rank2=rank2,
        block_lens=block_lens,
        return_lse=return_lse,
        fuse_norm_rope=fuse_norm_rope,
        inplace_qkv=inplace_qkv,
        index_band=index_band,
    )
    _check_thd(c, request.node.callspec.id)


@requires_rubin
def test_thd_qsa_length_forms_and_input_ranks_are_bitwise():
    """The ``[B]`` lengths and the ``[B+1]`` prefix sums (at base 0 and sliced from a larger prefix), the ``[1, T, .]`` and the
    ``[T, .]`` input spellings, and ``block_lens`` given vs absent (the kernel's derived count equals the given count on a
    valid-prefix list) all run the SAME function: ``out``, the gated O and the LSE BITWISE across the five runs."""
    lens, geom_kw = [300, 128, 200], GEOM_SMALL
    base = _run_thd(geom_kw, lens, source="synthetic", launches=1, refs=False)
    inp = (base.inp, base.meta)
    lists = (base.ids, base.lens)
    variants = {
        "cu": _run_thd(geom_kw, lens, source="synthetic", cu=True, inp=inp, lists=lists, launches=1, refs=False),
        "rank2": _run_thd(geom_kw, lens, source="synthetic", rank2=True, inp=inp, lists=lists, launches=1, refs=False),
        "no-block-lens": _run_thd(geom_kw, lens, source="synthetic", block_lens=False, inp=inp, lists=lists, launches=1, refs=False),
    }
    # a prefix tensor sliced from a larger one (the kernels normalize a prefix to its first entry)
    big = torch.tensor([0, 77] + [77 + x for x in base.meta["cu"]], dtype=torch.int32, device="cuda")
    assert big[2:].is_contiguous() and big[2:].numel() == len(lens) + 1
    cu_blk, cu_ws = variants["cu"].blk, variants["cu"].ws
    t, g = base.meta["t"], cu_blk.geom
    out_s = torch.full((1, t, g.d_model), _sentinel(BF16), device="cuda", dtype=BF16)
    lse_s = torch.full((1, g.h_q, t), _sentinel(BF16), device="cuda", dtype=torch.float32)
    i = base.inp
    cu_blk.execute(
        i["h"],
        i["w_qkvg"],
        i["w_q_norm"],
        i["w_k_norm"],
        i["cos"],
        i["sin"],
        i["w_o"],
        out_s,
        cu_ws,
        seq_lens=big[2:],
        lse=lse_s,
        block_ids=base.ids,
        block_lens=base.lens,
    )
    torch.cuda.synchronize()
    sliced = (out_s.clone(), lse_s.clone(), _view(cu_ws, cu_blk._layout().o, (t, g.h_q, g.d_head), BF16).clone())
    out0, lse0, o0 = base.runs[0]
    for name, (out1, lse1, o1) in [(k, v.runs[0]) for k, v in variants.items()] + [("cu-sliced", sliced)]:
        assert torch.equal(_bits(out0), _bits(out1)), f"{name}: out differs from the lengths-form rank-3 run"
        assert torch.equal(_bits(o0), _bits(o1)), f"{name}: the gated O differs from the lengths-form rank-3 run"
        assert torch.equal(lse0, lse1), f"{name}: the LSE differs from the lengths-form rank-3 run"
    print(f"\nTHD forms bitwise: lengths / prefix sums / a sliced prefix / rank-2 inputs / block_lens absent, lens {lens}")


@requires_rubin
@pytest.mark.parametrize("dtype", [BF16, F16], ids=["bf16", "f16"])
def test_thd_one_sequence_is_bitwise_the_dense_b1_qsa_block(dtype):
    """``B = 1`` packed ``(T,)`` against the dense ``B = 1, S = T`` ``QsaSpec`` block over the SAME bytes and the SAME (random)
    list: ``out``, the gated O and the LSE BITWISE -- the token-wise stages are the same launches at ``(1, T)``, and the sparse
    core's packed arm does the same gathers in the same order as its dense form (the standalone adapter's pin, through the
    block).  A difference here is a finding, never a tolerance."""
    t, top_k = 600, 512
    ref_geom = RefQsaGeometry(**GEOM_FLASH_NEXT, qsa=RefQsaSpec(top_k=top_k))
    dinp = make_qsa_inputs(ref_geom, 1, t, dtype=dtype, seed=3, index_source="synthetic")
    pinp, meta = make_packed_inputs(dense_geometry(ref_geom), [t], dtype=dtype, seed=3)
    for k in ("h", "w_qkvg", "w_o", "cos", "sin", "w_q_norm", "w_k_norm"):
        assert torch.equal(dinp[k], pinp[k]), k  # the packed draw IS the dense B = 1 draw, element for element
    dense = _run(GEOM_FLASH_NEXT, 1, t, dtype=dtype, top_k=top_k, index_source="synthetic", inp=dinp, launches=1, seed=3)
    packed = _run_thd(
        GEOM_FLASH_NEXT,
        [t],
        dtype=dtype,
        top_k=top_k,
        inp=(pinp, meta),
        lists=(dinp["block_ids"].view(t, top_k), dinp["block_lens"].view(t)),
        launches=1,
        refs=False,
    )
    (out_d, lse_d, o_d), (out_p, lse_p, o_p) = dense.runs[0], packed.runs[0]
    d = float((out_p.float() - out_d.float()).abs().max())
    assert torch.equal(
        _bits(out_p), _bits(out_d.reshape(1, t, -1))
    ), f"out: packed B = 1 differs from the dense QsaSpec block (max|diff| {d:.3e}) -- a finding, not a tolerance"
    assert torch.equal(_bits(o_p), _bits(o_d.reshape(t, -1, o_d.shape[-1]))), "gated O: packed B = 1 differs from the dense QsaSpec block"
    assert torch.equal(lse_p, lse_d), "LSE: packed B = 1 differs from the dense QsaSpec block"
    print(
        f"\nTHD one sequence {dtype}: packed (T = {t},) BITWISE the dense B = 1 QsaSpec block on out / O / LSE (dense cos(out) vs the oracle {_cos(out_d, dense.ref.out):.6f})"
    )


@requires_rubin
def test_thd_qsa_workspace_carries_the_packed_metadata_and_execute_requires_the_lengths():
    """``get_workspace_size()`` of a packed ``QsaSpec`` block is the dense ``B = 1, S = T`` carve plus the engine arm sized by the
    larger of the GEMMs' scratch and the sparse stage's packed metadata (``THD_META_WORDS(B) = 4 B + 4`` int32 words, 16-byte
    rounded -- the ONE scratch the arm needs, folded through ``scratch_workspace_bytes``); ``execute`` REQUIRES ``seq_lens``
    (the block's typed refusal before any launch) and refuses a lengths tensor of the wrong extent or dtype and a list of the
    wrong rank."""
    from cudnn.frost.tile_dsl.thd import THD_META_WORDS
    from cudnn.gated_attention_block.api import _align_up

    lens = [300, 128, 200]
    c = _run_thd(GEOM_SMALL, lens, launches=1, refs=False)
    blk, i = c.blk, c.inp
    meta_bytes = -(-(THD_META_WORDS(len(lens)) * 4) // 16) * 16
    assert meta_bytes == 64 and blk._sdpa.scratch_workspace_bytes() == meta_bytes
    lay = blk._layout()
    engine = max(blk._proj.workspace_bytes(), blk._out_proj.workspace_bytes(), meta_bytes, 1)
    assert blk.get_workspace_size() == lay.total_bytes + _align_up(engine) == c.ws.numel()
    # the dense QsaSpec block at (1, T) carves the same intermediates (only the engine arm may differ)
    dense = GatedAttentionBlockFwd(
        i["h"], i["w_qkvg"], i["w_q_norm"], i["w_k_norm"], i["cos"], i["sin"], i["w_o"], torch.empty_like(i["h"]), blk.geom, return_lse=True
    )
    assert dense.check_support()
    assert dense._layout() == lay and dense._sdpa.scratch_workspace_bytes() == 0
    out = torch.empty(1, c.meta["t"], blk.geom.d_model, device="cuda", dtype=BF16)
    lse = torch.empty(1, blk.geom.h_q, c.meta["t"], device="cuda", dtype=torch.float32)
    args = (i["h"], i["w_qkvg"], i["w_q_norm"], i["w_k_norm"], i["cos"], i["sin"], i["w_o"], out, c.ws)
    with pytest.raises(ValueError, match=r"thd=True: execute\(seq_lens=\) is required"):
        blk.execute(*args, lse=lse, block_ids=c.ids, block_lens=c.lens)
    with pytest.raises(ValueError, match="of 3 elements"):
        blk.execute(*args, seq_lens=c.seq_lens[:2], lse=lse, block_ids=c.ids, block_lens=c.lens)
    with pytest.raises(ValueError, match="of 3 elements"):
        blk.execute(*args, seq_lens=c.meta["cu_seqlens"], lse=lse, block_ids=c.ids, block_lens=c.lens)
    with pytest.raises(ValueError, match="int32"):
        blk.execute(*args, seq_lens=c.seq_lens.to(torch.int64), lse=lse, block_ids=c.ids, block_lens=c.lens)
    with pytest.raises(ValueError, match="block_ids"):
        blk.execute(*args, seq_lens=c.seq_lens, lse=lse, block_ids=c.ids[None, None], block_lens=c.lens)
    print(f"\nTHD QsaSpec workspace: {blk.get_workspace_size()} B = carve {lay.total_bytes} + engine arm {_align_up(engine)} (packed metadata {meta_bytes} B)")


# ============================================================================ fuse_gate under QsaSpec: the gated cells
def test_fuse_gate_under_qsa_declares_the_gated_sparse_stage():
    """Host: ``fuse_gate=True`` under a ``QsaSpec`` CONSTRUCTS (the sparse core's epilogue-gate arm is bound through the block):
    the sparse stage carries the gate, stage (5) is not built, and the stage's declared operands include the GATE at the slab's
    GATE columns (the token stride of the fused projection) -- the inversion of the former typed decline."""
    geom = GatedAttentionBlockGeometry(**GEOM_SMALL, qsa=QsaSpec())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = lambda *shape, dt=BF16: torch.zeros(*shape, dtype=dt, device=dev)  # noqa: E731
    blk = GatedAttentionBlockFwd(
        z(1, 8, geom.d_model),
        z(geom.n_qkvg, geom.d_model),
        z(geom.d_head),
        z(geom.d_head),
        z(1, 8, 64),
        z(1, 8, 64),
        z(geom.d_model, geom.h_q * geom.d_head),
        z(1, 8, geom.d_model),
        geom,
        fuse_gate=True,
    )
    assert blk.fuse_gate and isinstance(blk._sdpa, _SparseSdpa) and blk._sdpa.fuse_gate and blk._gate is None
    ops = blk._sdpa._operands()
    assert ops["gate"] is not None and ops["gate"].shape == (1, 8, geom.h_q, geom.d_head) and ops["gate"].strides[1] == geom.n_qkvg
    ungated = GatedAttentionBlockFwd(
        z(1, 8, geom.d_model),
        z(geom.n_qkvg, geom.d_model),
        z(geom.d_head),
        z(geom.d_head),
        z(1, 8, 64),
        z(1, 8, 64),
        z(geom.d_model, geom.h_q * geom.d_head),
        z(1, 8, geom.d_model),
        geom,
    )
    assert ungated._sdpa._operands()["gate"] is None and ungated._gate is not None
    assert len(blk._stages) == len(ungated._stages) - 1


def _ulp_of(x: torch.Tensor, dtype) -> torch.Tensor:
    """One unit in the last place of ``dtype`` at |x| (CPU fp32 math; ``torch.log2`` / ``exp2`` are avoided on the device),
    floored at the dtype's smallest subnormal step so a zero never yields a zero budget."""
    mant, sub = (7, 2.0**-133) if dtype == torch.bfloat16 else (10, 2.0**-24)
    ax = x.detach().float().cpu().abs().clamp_min(sub)
    _, e = torch.frexp(ax)  # ax = m * 2^e, m in [0.5, 1) -> floor(log2 ax) = e - 1
    return torch.ldexp(torch.ones_like(ax), e - 1 - mant).clamp_min(sub)


# (geometry, B, S, dtype, list source, top_k, seq_lens, fuse_norm_rope, inplace_qkv): the fuse_gate axis of the matrix
_GATED = [
    pytest.param(GEOM_FLASH_NEXT, 1, 512, BF16, "full", 512, None, False, None, id="fn24-2-s512-bf16-full-gated"),
    pytest.param(GEOM_FLASH_NEXT, 1, 512, F16, "synthetic", 512, None, False, None, id="fn24-2-s512-f16-synthetic-gated"),
    pytest.param(GEOM_FLASH_NEXT, 2, 512, BF16, "full", 512, (512, 0), False, None, id="fn24-2-s512-b2-dead-entry-gated"),
    pytest.param(GEOM_FLASH_NEXT, 1, 512, BF16, "synthetic", 512, None, True, None, id="fn24-2-s512-fully-fused-gated"),
    pytest.param(GEOM_FLASH_NEXT, 1, 512, BF16, "full", 512, None, False, False, id="fn24-2-s512-compact-kv-gated"),
    pytest.param(GEOM_TP4, 2, 512, BF16, "synthetic", 512, None, False, None, id="tp4-6-1-s512-b2-gated"),
    pytest.param(GEOM_397B, 1, 512, BF16, "synthetic", 512, None, False, None, id="g397b-32-2-s512-gqa16-gated"),
    pytest.param(GEOM_SMALL, 1, 300, BF16, "synthetic", 4, None, False, None, id="small8-2-s300-top4-gated"),
    pytest.param(GEOM_SMALL, 2, 2052, BF16, "synthetic", 512, (2052, 2015), False, None, id="small8-2-s2052-b2-lens-gated"),
]


@requires_rubin
@pytest.mark.parametrize("geom_kw, batch, seq_len, dtype, source, top_k, seq_lens, fuse_norm_rope, inplace_qkv", _GATED)
def test_fuse_gate_under_qsa_matches_the_oracle_and_the_unfused_block(geom_kw, batch, seq_len, dtype, source, top_k, seq_lens, fuse_norm_rope, inplace_qkv):
    """The gated ``QsaSpec`` block (stage (5) inside the sparse core's epilogue) against (a) the oracle on the same list --
    every standard assertion of the matrix (``isfinite``, no sentinel, dead rows exact, ``cos > 0.999`` on ``out``, ``atol
    2e-2`` on the gated O and the LSE, two launches bitwise) -- and (b) the UNFUSED block on the SAME inputs and list:
    ``gated == unfused-then-gated`` within ONE rounding of the activation dtype per element of the gated O (the unfused
    pipeline rounds O to bf16 / f16 before stage (5); both use the same approximate tanh), the LSE BITWISE the unfused block's
    (the gate cannot touch a softmax statistic), the dead entry exactly 0 on both, one launch fewer.  The exact max diff is
    recorded; the oracle's budget is the dense suites', nothing added."""
    lens_t = None if seq_lens is None else _i32(list(seq_lens))
    kw = dict(dtype=dtype, top_k=top_k, index_source=source, seq_lens=lens_t, fuse_norm_rope=fuse_norm_rope, inplace_qkv=inplace_qkv, seed=seq_len + batch + 7)
    c_f = _run(geom_kw, batch, seq_len, fuse_gate=True, **kw)
    label = f"gated {geom_kw['h_q']}/{geom_kw['h_kv']} B={batch} S={seq_len} {dtype} {source} top_k={top_k}"
    mags = _check(c_f, label)
    c_u = _run(geom_kw, batch, seq_len, fuse_gate=False, inp=c_f.inp, ref=c_f.ref, **kw)
    assert len(c_f.blk._stages) == len(c_u.blk._stages) - 1, "fuse_gate removes stage (5)'s launch"
    out_f, lse_f, o_f = c_f.runs[0]
    out_u, lse_u, o_u = c_u.runs[0]
    assert lse_f is not None and torch.equal(lse_f, lse_u), "the LSE must be bitwise the unfused block's"
    dead = c_f.ref.n_visible == 0
    if dead.any():
        assert (o_f[dead] == 0).all() and (o_u[dead] == 0).all() and (out_f[dead] == 0).all(), "the dead entry gates to exactly 0 on both pipelines"
    d = (o_f.float() - o_u.float()).abs()
    ulps = d.cpu() / _ulp_of(torch.maximum(o_f.float().abs(), o_u.float().abs()), dtype)
    max_d, max_ulp, n_off = float(d.max()), float(ulps.max()), int((d > 0).sum())
    assert (
        max_ulp <= 1.0
    ), f"{label}: gated vs unfused-then-gated O {max_ulp:.3f} ulp at the element's magnitude (max|d| {max_d:.3e}); the budget is one rounding"
    assert max_d <= ATOL
    cos_out = _cos(out_f, out_u)
    d_out = float((out_f.float() - out_u.float()).abs().max())
    assert cos_out > COS_OUT
    print(
        f"{label}: vs unfused block max|d O_gated| {max_d:.4e} = {max_ulp:.3f} ulp ({n_off} of {d.numel()} elements differ), "
        f"out cos {cos_out:.6f} max|d out| {d_out:.4e}; vs oracle cos {mags['cos']:.6f} |dO| {mags['d_o']:.4e} |dLSE| {mags['d_lse']:.2e}; dead rows {int(dead.sum())}"
    )


# ============================================================================ fuse_gate x THD under QsaSpec: the gated packed cells
def test_fuse_gate_with_thd_under_qsa_declares_the_gated_packed_sparse_stage():
    """Host: ``thd=True`` together with ``fuse_gate=True`` under a ``QsaSpec`` CONSTRUCTS -- the sparse core's epilogue gate composes
    with its packed arm (the GATE is the packed ``[1, T, H_q, D]`` slab slice like O): the sparse stage carries both arms, stage (5)
    is not built, the declared GATE operand is the slab slice at the fused projection's token stride beside the packed lengths.
    The DENSE packed block folds its gate too (the SDPA's THD gate descriptor), so neither pipeline builds stage (5) under thd."""
    geom = GatedAttentionBlockGeometry(**GEOM_SMALL, qsa=QsaSpec())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = lambda *shape, dt=BF16: torch.zeros(*shape, dtype=dt, device=dev)  # noqa: E731
    args = lambda g: (
        z(1, 8, g.d_model),
        z(g.n_qkvg, g.d_model),
        z(g.d_head),
        z(g.d_head),
        z(1, 8, 64),
        z(1, 8, 64),
        z(g.d_model, g.h_q * g.d_head),
        z(1, 8, g.d_model),
    )  # noqa: E731
    blk = GatedAttentionBlockFwd(*args(geom), geom, thd=True, num_sequences=2, max_seq_len=8, fuse_gate=True)
    assert blk.fuse_gate and blk.thd and isinstance(blk._sdpa, _SparseSdpa) and blk._sdpa.fuse_gate and blk._sdpa.thd and blk._gate is None
    ops = blk._sdpa._operands()
    assert ops["gate"] is not None and ops["gate"].shape == (1, 8, geom.h_q, geom.d_head) and ops["gate"].strides[1] == geom.n_qkvg
    assert ops["seq_q_lens"] is not None and ops["seq_q_lens"].shape == (2,)
    dense = GatedAttentionBlockGeometry(**GEOM_SMALL)
    dblk = GatedAttentionBlockFwd(*args(dense), dense, thd=True, num_sequences=2, max_seq_len=8, fuse_gate=True)
    assert dblk.fuse_gate and dblk.gate_fused and dblk.thd and dblk._gate is None and not isinstance(dblk._sdpa, _SparseSdpa)


def test_fuse_gate_with_bottom_right_under_qsa_is_the_decode_modes_typed_decline():
    """Host: the bottom-right diagonal on the block's OWN K / V (``kv_source="self"``) has no serving meaning -- the decode step's rows
    and its speculative verify rows sit at the end of a CACHED sequence, which is the block-sparse DECODE mode over the pools
    (``kv_source="paged"``, served since 2026-10-14: ``test_block_qsa_decode.py``, the gate fused in the core's epilogue unsplit and in
    the combine under a split); the decline names that mode, and ``fuse_gate=True`` leaves it and its wording unchanged."""
    geom = GatedAttentionBlockGeometry(**GEOM_SMALL, qsa=QsaSpec(), causal_bottom_right=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = lambda *shape, dt=BF16: torch.zeros(*shape, dtype=dt, device=dev)  # noqa: E731
    with pytest.raises(NotImplementedError, match="bottom-right diagonal") as e:
        GatedAttentionBlockFwd(
            z(1, 8, geom.d_model),
            z(geom.n_qkvg, geom.d_model),
            z(geom.d_head),
            z(geom.d_head),
            z(1, 8, 64),
            z(1, 8, 64),
            z(geom.d_model, geom.h_q * geom.d_head),
            z(1, 8, geom.d_model),
            geom,
            fuse_gate=True,
        )
    assert "kv_source='paged'" in str(e.value) and "sparse adapter's record declines" not in str(e.value)


# (geometry, packing, dtype, list source, top_k, cu_seqlens, fuse_norm_rope, inplace_qkv): the fuse_gate axis of the THD matrix
_GATED_THD = [
    pytest.param(GEOM_FLASH_NEXT, [300, 128, 200], BF16, "full", 512, False, False, None, id="fn24-2-three-seqs-bf16-full-gated"),
    pytest.param(GEOM_FLASH_NEXT, [300, 128, 200], F16, "synthetic", 512, False, False, None, id="fn24-2-three-seqs-f16-synthetic-gated"),
    pytest.param(GEOM_SMALL, [300, 0, 200], BF16, "synthetic", 512, True, False, None, id="small8-2-empty-middle-cu-gated"),
    pytest.param(GEOM_FLASH_NEXT, [300, 128, 200], BF16, "synthetic", 512, False, True, None, id="fn24-2-fully-fused-gated"),
    pytest.param(GEOM_SMALL, [5, 300, 128], BF16, "full", 512, False, False, False, id="small8-2-five-token-compact-kv-gated"),
    pytest.param(GEOM_TP4, [300, 128, 200], BF16, "synthetic", 4, False, False, None, id="tp4-6-1-top4-gated"),
    pytest.param(GEOM_397B, [300, 128, 200], BF16, "synthetic", 512, False, False, None, id="g397b-32-2-gqa16-gated"),
    pytest.param(GEOM_SMALL, [4097, 13, 2051], BF16, "synthetic", 512, False, False, None, id="small8-2-tails-4097-13-2051-gated"),
]


@requires_rubin
@pytest.mark.parametrize("geom_kw, lens, dtype, source, top_k, cu, fuse_norm_rope, inplace_qkv", _GATED_THD)
def test_fuse_gate_under_thd_qsa_matches_the_per_sequence_oracle_and_the_unfused_packed_block(
    request, geom_kw, lens, dtype, source, top_k, cu, fuse_norm_rope, inplace_qkv
):
    """The gated PACKED ``QsaSpec`` block (``thd=True, fuse_gate=True``: stage (5) inside the sparse core's epilogue on the packed
    GATE slice) against (a) every sequence's own oracle (``_check_thd``: ``cos(out) > 0.999``, the gated O and the LSE within
    ``atol 2e-2``, two launches bitwise, no sentinel) and (b) the UNFUSED packed block on the SAME inputs and lists:
    ``gated == unfused-then-gated`` within ONE rounding of the activation dtype per element of the gated O (the unfused pipeline
    rounds O to bf16 / f16 before stage (5); both use the same approximate tanh), the LSE BITWISE the unfused block's, an empty
    sequence owns no rows on both, one launch fewer.  The exact max diff is recorded; the budgets are the THD suite's, nothing
    added."""
    kw = dict(dtype=dtype, top_k=top_k, source=source, cu=cu, fuse_norm_rope=fuse_norm_rope, inplace_qkv=inplace_qkv, seed=sum(lens) + 11)
    label = request.node.callspec.id
    c_f = _run_thd(geom_kw, lens, fuse_gate=True, **kw)
    mags = _check_thd(c_f, label)
    c_u = _run_thd(geom_kw, lens, fuse_gate=False, inp=(c_f.inp, c_f.meta), lists=(c_f.ids, c_f.lens), refs=False, **kw)
    assert len(c_f.blk._stages) == len(c_u.blk._stages) - 1, "fuse_gate removes stage (5)'s launch"
    (out_f, lse_f, o_f), (out_u, lse_u, o_u) = c_f.runs[0], c_u.runs[0]
    assert lse_f is not None and torch.equal(lse_f, lse_u), "the LSE must be bitwise the unfused packed block's"
    d = (o_f.float() - o_u.float()).abs()
    ulps = d.cpu() / _ulp_of(torch.maximum(o_f.float().abs(), o_u.float().abs()), dtype)
    max_d, max_ulp, n_off = float(d.max()), float(ulps.max()), int((d > 0).sum())
    assert (
        max_ulp <= 1.0
    ), f"{label}: gated vs unfused-then-gated O {max_ulp:.3f} ulp at the element's magnitude (max|d| {max_d:.3e}); the budget is one rounding"
    assert max_d <= ATOL
    cos_out = _cos(out_f, out_u)
    d_out = float((out_f.float() - out_u.float()).abs().max())
    assert cos_out > COS_OUT
    print(
        f"{label}: vs unfused packed block max|d O_gated| {max_d:.4e} = {max_ulp:.3f} ulp ({n_off} of {d.numel()} elements differ), "
        f"out cos {cos_out:.6f} max|d out| {d_out:.4e}; vs oracle cos {mags['cos']:.6f} |dO| {mags['d_o']:.4e} |dLSE| {mags['d_lse']:.2e}; "
        f"{mags['seqs']} live sequences of {len(lens)}"
    )


@requires_rubin
def test_a_packed_qsa_block_with_write_through_is_bitwise_the_one_without():
    """The write-through acceptance on the PACKED sparse block (the typed decline ``paged_kv_page_size with thd=True``, inverted
    2026-10-14; the dense half is ``test_a_qsa_block_with_write_through_is_bitwise_the_one_without``): a ``QsaSpec`` block declared
    ``thd=True`` with ``paged_kv_page_size=16`` writes every packed token's post-RoPE K / V row through into the pools at
    ``slot_mapping[token]`` and its ``out`` / LSE / gated O are BITWISE the same packed block without write-through (the same
    inputs, lists and lengths -- an empty sequence among them); the pools are bitwise the slab bands the sparse core consumed, a
    padded slot writes nothing, ``get_workspace_size`` is unchanged and the stage sits right before the packed sparse SDPA."""
    geom_kw, lens, ps = GEOM_SMALL, [300, 0, 128, 84], 16
    plain = _run_thd(geom_kw, lens, source="synthetic", seed=11)
    _check_thd(plain, "small8-2 packed 300/0/128/84 plain")
    t, geom = plain.meta["t"], plain.geom
    n_pages = t // ps + 2
    gen = torch.Generator(device="cuda").manual_seed(0)
    slot = torch.randperm(n_pages * ps, generator=gen, device="cuda")[:t].to(torch.int32)
    slot[::13] = -1
    k_cache, v_cache = _nan_pool(n_pages, geom.h_kv, ps, geom.d_head), _nan_pool(n_pages, geom.h_kv, ps, geom.d_head)
    wt = _run_thd(
        geom_kw,
        lens,
        source="synthetic",
        seed=11,
        lists=(plain.ids, plain.lens),
        inp=(plain.inp, plain.meta),
        refs=False,
        paged_kv_page_size=ps,
        pools=dict(k_cache=k_cache, v_cache=v_cache, slot_mapping=slot),
    )
    assert torch.equal(_bits(wt.runs[0][0]), _bits(plain.runs[0][0])) and torch.equal(
        wt.runs[0][1], plain.runs[0][1]
    ), "write-through changed the packed sparse block's output"
    assert torch.equal(_bits(wt.runs[0][2]), _bits(plain.runs[0][2]))
    assert wt.blk.get_workspace_size() == plain.blk.get_workspace_size()
    lay = wt.blk._layout()
    proj = _view(wt.ws, lay.proj, (t, geom.n_qkvg), BF16)
    k_rows, v_rows = _cols(proj, geom.qkvg_offsets[2], geom.h_kv, geom.d_head), _cols(proj, geom.qkvg_offsets[3], geom.h_kv, geom.d_head)
    live = slot >= 0
    s_ = slot[live].long()
    assert torch.equal(_bits(k_cache[s_ // ps, :, s_ % ps, :]), _bits(k_rows[live])) and torch.equal(
        _bits(v_cache[s_ // ps, :, s_ % ps, :]), _bits(v_rows[live])
    )
    for pool in (k_cache, v_cache):
        assert (~torch.isnan(pool.float())).all(-1).sum().item() == int(live.sum()) * geom.h_kv
    assert wt.blk._stages.index(wt.blk._cache_write) == wt.blk._stages.index(wt.blk._sdpa) - 1
