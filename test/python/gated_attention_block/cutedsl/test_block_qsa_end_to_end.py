# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block END TO END under ``QsaSpec`` -- stage (4) is the index-list sparse core -- against the QSA oracle.

Every Rubin cell (``requires_rubin``) declares a block with ``GatedAttentionBlockGeometry(qsa=QsaSpec(...))``, checks, compiles
and runs it TWICE on sentinel-filled outputs, and compares ``out``, the gated O (the workspace slot stage (5) wrote in place)
and the LSE against ``gated_attention_block_qsa_reference`` run on the SAME inputs and the SAME block list: ``isfinite`` and
"no sentinel survived" BEFORE any cosine, ``cos > 0.999`` on ``out`` (the end-to-end suite's bar), the gated O within the
repo's bf16 / f16 OUTPUT bound (``gated_block_bounds.assert_output_close``: per element ``|d| <= 2^-7 max|ref| + 2^-6 |ref|``
and ``cos >= 0.999``, the block backward's calibrated convention; ``max|d|`` printed beside the fraction of the bound), the LSE
within ``atol 2e-2`` (the SDPA stage test's bar on the LSE), a dead row (``seq_lens[b] == 0``, or an empty list with no open tail)
EXACTLY ``out = 0`` / ``O = 0`` / ``LSE = -inf``, and the two launches bitwise.  Tolerances are the block suites' own, none added.

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

The IN-BLOCK INDEXER (``QsaSpec(index_source="indexer", index_band=True)``): the block derives the selection itself -- the
band's queries normed + rotated, the raw key compressed per block at the block start, the scorer's top-k -- and the cells run
the TWO-STAGE oracle: (1) the selection as a SET, (a) against the top-k of the scores recomputed in fp64 from the kernel's OWN
bf16 queries and compressed keys up to ties (the scorer's fp32 accumulate), (b) against the fp32-operand oracle under the
MARGIN RULE -- a block in exactly one of the two sets is accepted iff its oracle score is within ``tol`` of the row's k-th
oracle score, ``tol`` = the MEASURED perturbation of the scores by the bf16 rounding of the operands (never a tuned number)
-- and (c) the same against the HF-rounding oracle (the pooled key cast to bf16 before its norm); the kernel's queries and
compressed keys within one bf16 ulp of the once-rounded fp32 chain; the count rule on every row; (2) attention EXACT given
the kernel's selection (the block oracle on the kernel's list, the standard assertions).  Plus the identity below the bound
(no scorer launch, bitwise the caller-list block on the full list), the step-0 reuse pin (``block_ids_out`` / ``block_lens_out``
fed to a caller-list block reproduce the output bitwise; ``index_k_compressed`` rows written and the rest untouched), and the
write-through composition.

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

from cudnn.gated_attention_block import (  # noqa: E402
    INDEX_COMPRESS_ROUNDING_DEFAULT,
    INDEX_COMPRESS_ROUNDINGS,
    KV_SOURCE_PAGED,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    QsaSpec,
    index_k_raw_view,
)
from cudnn.gated_attention_block.api import _cols, _SparseSdpa, _view  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_bounds import assert_output_close  # noqa: E402
from gated_block_qsa_reference import (  # noqa: E402
    RefQsaGeometry,
    RefQsaSpec,
    block_ids_contract_violations,
    dense_geometry,
    full_block_ids,
    gated_attention_block_qsa_reference,
    gated_attention_block_qsa_reference_packed,
    make_qsa_inputs,
    qsa_indexer_reference,
    qsa_visible_mask,
    random_block_ids,
)
from gated_block_reference import (  # noqa: E402
    RefGeometry,
    apply_partial_rope,
    build_rope_tables,
    compare_packed,
    gated_attention_block_reference,
    make_inputs,
    make_packed_inputs,
)

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107

COS_OUT = 0.999  # the end-to-end suite's bar on `out` (test_block_end_to_end.py)
# The SDPA stage test's bar on the LSE (test_sdpa_stage_sm107.py asserts atol 2e-2 on the LSE; its O checks are cos > 0.999) -- never widened here.
# The gated O is held to the repo's bf16 / f16 OUTPUT bound instead (gated_block_bounds.py: per element |d| <= 2^-7 max|ref| + 2^-6 |ref| and
# cos >= 0.999, the block backward's calibrated convention): an absolute 2e-2 on bf16 O values of magnitude up to ~3 is 1.3 ulps at |2, 4) and
# refused a documented rounding tail of the (1 + w) norm-weight convention (qkvg_from_hf) by ONE element in 12.6 M at 0.877 of the bound.
ATOL = 2e-2
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


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """The bf16 spacing at each |x| (8 bits of mantissa), floored at the smallest normal's spacing."""
    e = torch.floor(torch.log2(x.float().abs().clamp_min(2.0**-126)))
    return torch.pow(2.0, e - 7)


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
    ix: Optional[dict] = None  # the in-block indexer's record: kernel_ids / kernel_lens, the qi / kbar slots, launches, the caller buffers


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
    indexer=False,
    indexer_outputs=False,
    fuse_gate=False,
    index_cache_dtype=None,
    index_compress_rounding=INDEX_COMPRESS_ROUNDING_DEFAULT,
) -> _Cell:
    """Declare, check, compile and run a QsaSpec block ``launches`` times on sentinel-filled outputs; the oracle on the same
    inputs and list.  ``lists`` overrides the generated ``(block_ids, block_lens)``; ``inp`` / ``ref`` reuse another cell's.
    ``indexer=True`` declares ``index_source="indexer"`` (the band implied): the block derives the selection and the oracle
    runs on the KERNEL's list (``ix`` records it); ``indexer_outputs`` hands the three optional caller buffers in;
    ``fuse_gate`` folds stage (5) into the sparse core's epilogue (the gated O is then the SDPA's own output in the same slot);
    ``index_cache_dtype`` (appended) declares the indexer cache's dtype (``torch.float8_e4m3fn`` = the fp8 indexer cache);
    ``index_compress_rounding`` (appended) the indexer compress chains' rounding points (``QsaSpec.index_compress_rounding``)."""
    band = index_band or indexer
    qsa = QsaSpec(
        top_k=top_k,
        index_band=band,
        index_source="indexer" if indexer else "caller",
        index_cache_dtype=index_cache_dtype,
        **({} if not indexer else dict(index_compress_rounding=index_compress_rounding)),  # refused under caller lists off the default
    )
    ref_geom = RefQsaGeometry(**geom_kw, qsa=RefQsaSpec(top_k=top_k, index_band=band))
    geom = GatedAttentionBlockGeometry(**geom_kw, qsa=qsa)
    if inp is None:
        inp = make_qsa_inputs(ref_geom, batch, seq_len, dtype=dtype, seed=seed, index_source="indexer" if indexer else index_source, seq_lens=seq_lens)
    ids, lens = (inp["block_ids"], inp["block_lens"]) if lists is None else lists
    lens_arg = lens if block_lens else None
    if ref is None and not indexer:
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
    ix_kw, ix = {}, None
    if indexer:
        ix_kw = dict(w_iq_norm=inp["w_iq_norm"], w_ik_norm=inp["w_ik_norm"])
        if indexer_outputs:
            ix = dict(
                ids_out=torch.full((t, top_k), -7, dtype=torch.int32, device="cuda"),
                lens_out=torch.full((t,), -7, dtype=torch.int32, device="cuda"),
                kbar_out=torch.full((batch, seq_len // BS + 3, geom.qsa.index_head_dim), float("nan"), dtype=index_cache_dtype or dtype, device="cuda"),
            )
            ix_kw.update(block_ids_out=ix["ids_out"], block_lens_out=ix["lens_out"], index_k_compressed=ix["kbar_out"])
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
            **({} if indexer else dict(block_ids=ids, block_lens=lens_arg)),
            **(pools or {}),
            **ix_kw,
        )
        torch.cuda.synchronize()
        o_gated = _view(ws, lay.o, (t, geom.h_q, geom.d_head), dtype).view(batch, seq_len, geom.h_q, geom.d_head).clone()
        runs.append((out.clone(), None if lse is None else lse.clone(), o_gated))
    if indexer:
        # The kernel's OWN list (the caller's buffer, the workspace slot, or the plan-time identity below the bound) and the
        # operands it scored: the oracle runs on THAT list (stage 2 of the two-stage oracle), never on a list of its own.
        st = blk._indexer
        v = st.views(ws, lay)
        k_ids = (
            (ix["ids_out"] if ix is not None else (v["ids"] if v is not None else st._const["identity_ids"])).view(batch, seq_len, top_k).clone()
        )  # canonical IN PLACE (2026-10-29)
        k_lens = (ix["lens_out"] if ix is not None else st._const["counts"]).view(batch, seq_len).clone()
        ix = dict(
            ix or {},
            launches=st.launches,
            selects=st.selects,
            kernel_ids=k_ids,
            kernel_lens=k_lens,
            qi=None if v is None else v["qi"].clone(),
            kbar=None if v is None else v["kbar"].clone(),
            qi8=None if (v is None or "qi8" not in v) else v["qi8"].clone(),  # the fp8 indexer cache's e4m3 operands
            kbar8=None if (v is None or "kbar8" not in v) else v["kbar8"].clone(),
        )
        ids, lens_arg = k_ids, k_lens
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
                block_ids=k_ids,
                block_lens=k_lens,
                seq_lens=seq_lens,
            )
    return _Cell(runs=runs, ref=ref, blk=blk, inp=inp, ws=ws, geom=geom, ids=ids, lens=lens_arg, ix=ix)


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
    ob = assert_output_close(o, ref.o_gated, f"{label}: gated O vs the oracle")
    c.mags = dict(cos=cos, d_out=d_out, d_o=d_o, d_lse=d_lse, dead=int(dead.sum()), o_bound_ratio=ob["bound_ratio"], o_cos=ob["cos"])
    print(
        f"\n{label}: cos(out) {cos:.6f} max|d out| {d_out:.4e} max|d O_gated| {d_o:.4e} ({ob['bound_ratio']:.3f} of the output bound) max|d LSE| {d_lse:.2e} dead rows {int(dead.sum())}"
    )
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
    assert m["o_bound_ratio"] <= 1.0  # the holed list stays within the output bound (asserted in _check; restated on the magnitudes)


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
    assert cos_below > COS_OUT and d_lse_below <= ATOL
    assert_output_close(o[0, below], ref_full.o_gated[0, below], "rows < 2051: gated O vs the FULL-set oracle")
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


# ============================================================================ the in-block indexer: the two-stage oracle
_TIE_TOL = 1e-4  # the scorer's fp32-accumulate budget at the k-th boundary (the standalone selector suite's)


def _set_vs_topk(in_set: torch.Tensor, scores: torch.Tensor, top_k: int, tol: float) -> tuple:
    """Per row: the SET ``in_set`` ``[R, N]`` against the top-k of ``scores`` ``[R, N]`` (fp64; ``-inf`` = not visible), every
    member of the symmetric difference accepted iff its score is within ``tol`` of the row's k-th score (the margin rule).
    Returns ``(flips, rows_with_flips, max_gap)``; asserts the rule."""
    rows, n = scores.shape
    k_eff = min(top_k, n)
    tk = torch.topk(scores, k_eff, dim=-1)
    exp_valid = torch.isfinite(tk.values)
    in_exp = torch.zeros(rows, n + 1, dtype=torch.bool, device=scores.device)
    in_exp.scatter_(1, torch.where(exp_valid, tk.indices, torch.full_like(tk.indices, n)), torch.ones_like(exp_valid))
    in_exp = in_exp[:, :n]
    kth = torch.where(exp_valid, tk.values, torch.full_like(tk.values, float("inf"))).min(dim=-1).values
    differs = in_set ^ in_exp
    flips = int(differs.sum())
    if flips == 0:
        return 0, 0, 0.0
    gap = (scores - kth[:, None]).abs()[differs]
    assert torch.isfinite(gap).all(), f"{int((~torch.isfinite(gap)).sum())} flipped ids lie outside the row's visible range"
    assert bool((gap <= tol).all()), f"{flips} ids differ from the top-k beyond the margin: max gap {float(gap.max()):.3e} > tol {tol:.3e}"
    return flips, int(differs.any(dim=-1).sum()), float(gap.max())


def _indexer_refs_from_slab(c: _Cell) -> tuple:
    """The fp32 indexer chain on the KERNEL'S OWN slab band (read back from the workspace), the operands the stage normed and
    compressed: ``(q_ref [B, S, H_i, D_i], kbar_ref [B, floor(S / bs), D_i])`` in fp32 -- the queries RMSNormed (``w_iq_norm``)
    and rotated at the token's position, the raw key mean-pooled per complete block, RMSNormed (``w_ik_norm``) and rotated at
    the block start.  The oracle's own GEMM rounds the band differently from the FROST GEMM, so the one-rounding contract of the
    norm / compress chain is checked against THIS band, never against the oracle's."""
    geom, inp = c.geom, c.inp
    q = geom.qsa
    b, s = inp["h"].shape[:2]
    t, bs, h_i, d_i = b * s, q.block_size, q.index_heads, q.index_head_dim
    lay = c.blk._layout()
    proj = _view(c.ws, lay.proj, (t, geom.n_qkvg), inp["h"].dtype)
    q_raw = torch.as_strided(proj, (t, h_i, d_i), (geom.n_qkvg, d_i, 1), proj.storage_offset() + geom.qkvg_offsets[4]).float().view(b, s, h_i, d_i)
    k_raw = index_k_raw_view(proj, geom, b, s)[:, :, 0].float()  # [B, S, D_i]
    eps = float(q.index_norm_eps)
    y = q_raw * torch.rsqrt(q_raw.pow(2).mean(-1, keepdim=True) + eps) * inp["w_iq_norm"].float()
    q_ref = apply_partial_rope(y, inp["cos"].float(), inp["sin"].float(), geom.rope_dim)
    nb = s // bs
    pooled = k_raw[:, : nb * bs].reshape(b, nb, bs, d_i).mean(dim=2)
    kb = pooled * torch.rsqrt(pooled.pow(2).mean(-1, keepdim=True) + eps) * inp["w_ik_norm"].float()
    kbar_ref = apply_partial_rope(kb[:, :, None, :], inp["cos"][:, 0::bs][:, :nb].float(), inp["sin"][:, 0::bs][:, :nb].float(), geom.rope_dim)[:, :, 0]
    return q_ref, kbar_ref


def _assert_one_rounding(got: torch.Tensor, ref32: torch.Tensor, what: str) -> dict:
    """``got`` (bf16) against the fp32 chain ``ref32`` on the same inputs: every element within ONE bf16 ulp of the rounded
    reference, the ulp floored at the fp32 chain's own noise -- ``2^-22 x max(1, row max)`` (a couple of fp32 ulps at the row's
    largest magnitude: two fp32 chains with another rsqrt and another FMA order land there, and a cancelled RoPE element is
    small only in the result, not in the error).  Returns the max in ulps (unfloored), the max absolute error and the bitwise
    fraction."""
    ref16 = ref32.to(got.dtype).float()
    d = (got.float() - ref32).abs()
    floor = 2.0**-22 * ref32.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    tol = torch.maximum(_bf16_ulp(ref16), floor)
    worst = float((d / tol).max())
    assert worst <= 1.0, f"{what}: {worst:.2f} x the one-rounding budget off the fp32 chain (max |d| {float(d.max()):.3e})"
    return dict(ulps=worst, max_abs=float(d.max()), bitwise=float((got.float() == ref16).float().mean()))


def _membership(ids: torch.Tensor, n: int) -> torch.Tensor:
    """``[R, top_k]`` int32 ids (``-1`` = none) -> ``[R, n]`` bool membership."""
    rows = ids.shape[0]
    m = torch.zeros(rows, n + 1, dtype=torch.bool, device=ids.device)
    m.scatter_(1, torch.where(ids >= 0, ids.long(), torch.full_like(ids, n).long()), torch.ones_like(ids, dtype=torch.bool))
    return m[:, :n]


def _indexer_two_stage(c: _Cell, label: str) -> dict:
    """Stage 1 of the two-stage oracle on an indexer cell (stage 2 -- attention exact given the kernel's list -- is ``_check``):
    the count rule and the ``-1`` prefix form on EVERY row; below the bound the identity; past it (a) the selection as a set
    against the top-k of the fp64 scores of the kernel's OWN bf16 operands up to ties, (b) against the fp32-operand oracle under
    the margin rule with ``tol`` = the MEASURED operand-rounding perturbation (+ the tie budget), (c) the same against the
    HF-rounding oracle, (d) the kernel's queries and compressed keys within one bf16 ulp of the once-rounded fp32 chain; padding
    rows (at or past the KV length) compared on their VISIBLE blocks, the ids they list past the range counted and reported."""
    geom, inp, ix = c.geom, c.inp, c.ix
    q = geom.qsa
    b, s = inp["h"].shape[:2]
    bs, top_k, d_i, h_i = q.block_size, q.top_k, q.index_head_dim, q.index_heads
    dev = inp["h"].device
    seq_lens = inp["seq_lens"]
    lengths = torch.full((b,), s, dtype=torch.long, device=dev) if seq_lens is None else seq_lens.to(dev).long()
    pos = torch.arange(s, device=dev)
    counts = torch.minimum(torch.div(pos + 1, bs, rounding_mode="floor"), torch.tensor(top_k, device=dev)).expand(b, s)
    k_ids, k_lens = ix["kernel_ids"], ix["kernel_lens"]
    valid = k_ids >= 0
    assert torch.equal(k_lens.long(), counts), f"{label}: the counts are not min(top_k, floor((pos + 1) / 4))"
    assert torch.equal(valid, torch.arange(top_k, device=dev)[None, None, :] < counts[..., None]), f"{label}: the valid ids are not a prefix of the count"
    assert int(k_ids.max()) < s // bs, f"{label}: an id past the last complete block"
    n_exec = len(c.runs)  # the stage's counter is cumulative over the cell's executes
    m = dict(identity=not ix["selects"], launches=ix["launches"], executes=n_exec)
    if not ix["selects"]:
        j = (
            counts[..., None].to(torch.int32) - 1 - torch.arange(top_k, device=dev, dtype=torch.int32)
        )  # the identity DESCENDING: the block's one canonical order
        assert torch.equal(torch.where(valid, k_ids, torch.full_like(k_ids, -1)), torch.where(valid, j, torch.full_like(j, -1))), f"{label}: not the identity"
        assert ix["launches"] == n_exec * c.blk._indexer.launches_per_execute(
            "kbar_out" in ix
        ), f"{label}: {ix['launches']} indexer launches over {n_exec} executes below the identity bound"
        print(f"\n{label}: the identity list (S <= {q.identity_bound}), {ix['launches']} indexer launch(es) over {n_exec} executes")
        return m
    assert ix["launches"] == n_exec * c.blk._indexer.launches_per_execute("kbar_out" in ix), f"{label}: {ix['launches']} launches over {n_exec} executes"
    # the canonical order: the selected ids strictly descending, the -1 padding last (the top-k's own slot order is unspecified)
    diffs = (k_ids[..., :-1] - k_ids[..., 1:]).masked_fill(~valid[..., 1:], 1)
    assert bool((diffs > 0).all()), f"{label}: a row's list is not in descending order"
    ref_geom = RefQsaGeometry(**{k: getattr(geom, k) for k in ("d_model", "h_q", "h_kv", "d_head", "rope_dim")}, qsa=RefQsaSpec(top_k=top_k, index_band=True))
    spec = ref_geom.qsa
    oracle = {}
    for hf in (False, True):
        oracle[hf] = qsa_indexer_reference(
            inp["h"], inp["w_i"], inp["w_iq_norm"], inp["w_ik_norm"], None, None, inp["cos"], inp["sin"], spec, ref_geom, seq_lens=seq_lens, hf_rounding=hf
        )
    scores_o, _, lens_o, _, k_c_o = oracle[False]
    n_blocks = k_c_o.shape[1]
    live = pos[None, :] < lengths[:, None]  # [B, S]: the rows below the KV length, where the oracle and the kernel see the same blocks
    assert torch.equal(lens_o[live].long(), counts[live]), f"{label}: the oracle's counts disagree on live rows"
    qi_k = ix["qi"].view(b, s, h_i, d_i)
    kbar_k = ix["kbar"]
    nb_live = torch.div(lengths, bs, rounding_mode="floor")  # complete blocks per sequence
    blk_live = torch.arange(n_blocks, device=dev)[None, :] < nb_live[:, None]  # [B, NB]
    # the kernel's own scores (fp64 from its bf16 operands) under the position rule the scorer applied
    s_k = torch.relu(torch.einsum("bshd,bjd->bshj", qi_k.double(), kbar_k.double())).sum(2) / math.sqrt(d_i)
    limit = torch.arange(n_blocks, device=dev)[None, None, :] < torch.div(pos + 1, bs, rounding_mode="floor")[None, :, None]
    s_k = s_k.masked_fill(~limit, float("-inf"))
    in_k = _membership(k_ids.reshape(b * s, top_k), n_blocks).view(b, s, n_blocks)
    # (a) the set == top-k of the kernel-operand scores up to ties (live rows: the sets the kernel and the oracle both define)
    fl_a, rows_a, gap_a = _set_vs_topk(in_k[live], s_k[live], top_k, _TIE_TOL)
    # (b) the margin rule vs the fp32-operand oracle: tol = the measured perturbation of the scores by the operand rounding
    vis_live = torch.isfinite(scores_o) & live[..., None]
    tol_o = float((scores_o.double() - s_k).abs()[vis_live].max()) + _TIE_TOL
    fl_b, rows_b, gap_b = _set_vs_topk(in_k[live], scores_o[live].double(), top_k, tol_o)
    # (c) the HF-rounding arm (the pooled key cast to bf16 before its norm, both norms returning bf16)
    scores_hf = oracle[True][0]
    tol_hf = float((scores_hf.double() - s_k).abs()[vis_live].max()) + _TIE_TOL
    fl_c, rows_c, gap_c = _set_vs_topk(in_k[live], scores_hf[live].double(), top_k, tol_hf)
    # padding rows: the kernel's VISIBLE subset against the oracle's set; the ids it lists past the visible range are masked by the core
    pad = ~live
    extra = 0
    if bool(pad.any()):
        vis_rows = torch.arange(n_blocks, device=dev)[None, None, :] < nb_live[:, None, None]
        extra = int((in_k & ~vis_rows)[pad].sum())
        _set_vs_topk((in_k & vis_rows)[pad], scores_o[pad].double(), top_k, tol_o)
    n_live = int(live.sum())
    m.update(
        rows_live=n_live,
        flips_vs_own=fl_a,
        rows_vs_own=rows_a,
        gap_vs_own=gap_a,
        tol_oracle=tol_o,
        flips_vs_oracle=fl_b,
        rows_vs_oracle=rows_b,
        gap_vs_oracle=gap_b,
        tol_hf=tol_hf,
        flips_vs_hf=fl_c,
        rows_vs_hf=rows_c,
        gap_vs_hf=gap_c,
        padding_rows=int(pad.sum()),
        padding_extra_ids=extra,
        agree_oracle=100.0 * (1.0 - rows_b / max(1, n_live)),
    )
    print(
        f"\n{label}: {n_live} live rows; vs the kernel's own operands {fl_a} flips ({rows_a} rows, gap {gap_a:.2e} <= {_TIE_TOL}); "
        f"vs the fp32-operand oracle {fl_b} flips ({rows_b} rows, {m['agree_oracle']:.3f} % rows exact, gap {gap_b:.3e} <= tol {tol_o:.3e}); "
        f"vs the HF-rounding oracle {fl_c} flips ({rows_c} rows, gap {gap_c:.3e} <= tol {tol_hf:.3e}); "
        f"{int(pad.sum())} padding rows, {extra} ids past their range"
    )
    # (d) the operands: the kernel's qi / kbar against the fp32 chain on ITS OWN slab band (the FROST GEMM's rounding of the band
    # differs from the oracle's GEMM), one bf16 rounding; the distance to the ORACLE's compressed keys is reported as a magnitude.
    q_ref, kbar_ref = _indexer_refs_from_slab(c)
    mq = _assert_one_rounding(qi_k, q_ref, f"{label}: the indexer queries")
    mk = _assert_one_rounding(kbar_k[blk_live], kbar_ref[blk_live], f"{label}: the compressed keys")
    d_oracle = float((kbar_k.float() - k_c_o.float()).abs()[blk_live].max())
    m.update(q_ulps=mq["ulps"], q_bitwise=mq["bitwise"], k_ulps=mk["ulps"], k_bitwise=mk["bitwise"], k_vs_oracle_gemm=d_oracle)
    print(
        f"{label}: operands vs the fp32 chain on the kernel's band -- qi max {mq['ulps']:.2f} x the one-rounding budget ({100 * mq['bitwise']:.3f} % bitwise, max |d| {mq['max_abs']:.2e}), "
        f"kbar max {mk['ulps']:.2f} x ({100 * mk['bitwise']:.3f} % bitwise, max |d| {mk['max_abs']:.2e}); kbar vs the oracle's GEMM + chain max |d| {d_oracle:.2e}"
    )
    return m


# (id, geometry, B, S, seq_lens, caller outputs)
_INDEXER_MATRIX = [
    pytest.param(GEOM_SMALL, 1, 2560, None, False, id="small8-2-s2560-indexer"),
    pytest.param(GEOM_SMALL, 1, 2560, None, True, id="small8-2-s2560-indexer-outputs"),
    pytest.param(GEOM_SMALL, 2, 2300, (2300, 2100), False, id="small8-2-s2300-b2-indexer-lens-2100"),
    pytest.param(GEOM_SMALL, 1, 2052, None, False, id="small8-2-s2052-indexer-first-dropped-block"),
    pytest.param(GEOM_FLASH_NEXT, 1, 4096, None, True, id="fn24-2-s4096-indexer-outputs"),
    pytest.param(GEOM_SMALL, 2, 2051, None, False, id="small8-2-s2051-b2-indexer-identity"),
    pytest.param(GEOM_SMALL, 1, 300, None, True, id="small8-2-s300-indexer-identity-outputs"),
]


@requires_rubin
@pytest.mark.parametrize("geom_kw, batch, seq_len, seq_lens, outputs", _INDEXER_MATRIX)
def test_indexer_block_matches_the_two_stage_oracle(request, geom_kw, batch, seq_len, seq_lens, outputs):
    """One indexer cell: stage 1 (the selection, ``_indexer_two_stage``) and stage 2 (attention exact given the kernel's list,
    ``_check``); below the identity bound the block is BITWISE the caller-list block on the full list (the same core, the same
    list, the same operands) and launches no scorer."""
    lens_t = None if seq_lens is None else _i32(list(seq_lens))
    c = _run(geom_kw, batch, seq_len, indexer=True, indexer_outputs=outputs, seq_lens=lens_t, seed=seq_len + 7 * batch)
    label = request.node.callspec.id
    m1 = _indexer_two_stage(c, label)
    pos = torch.arange(seq_len, device="cuda").expand(batch, seq_len).reshape(-1)
    kv = torch.full((batch * seq_len,), seq_len, dtype=torch.long, device="cuda") if lens_t is None else lens_t.long().repeat_interleave(seq_len)
    # The list contract on every LIVE row, padded cells included (a padding row -- at or past its sequence's KV length -- may list
    # blocks past its range: the core masks them, and _indexer_two_stage counts those ids and compares the row's visible blocks).
    live = pos < kv
    assert int(live.sum()) == (batch * seq_len if lens_t is None else int(lens_t.long().clamp(max=seq_len).sum()))
    bad = block_ids_contract_violations(
        c.ids.reshape(batch * seq_len, -1)[live], pos[live], kv[live], block_size=BS, block_lens=c.lens.reshape(-1)[live], top_k=c.geom.qsa.top_k
    )
    assert bad == [], bad  # the kernel's list is contract-clean on every live row
    m2 = _check(c, label)
    if m1["identity"]:
        caller = _run(
            geom_kw,
            batch,
            seq_len,
            index_band=True,
            seq_lens=lens_t,
            inp=c.inp,
            ref=c.ref,
            lists=(c.ix["kernel_ids"].contiguous(), c.ix["kernel_lens"].contiguous()),
        )
        assert (
            torch.equal(_bits(c.runs[0][0]), _bits(caller.runs[0][0]))
            and torch.equal(c.runs[0][1], caller.runs[0][1])
            and torch.equal(_bits(c.runs[0][2]), _bits(caller.runs[0][2]))
        ), f"{label}: the identity path is not bitwise the caller-list block"
        assert c.blk.get_workspace_size() == caller.blk.get_workspace_size()
    if outputs:
        ix, nb = c.ix, seq_len // BS
        assert torch.equal(ix["ids_out"].view(batch, seq_len, -1), ix["kernel_ids"]) and torch.equal(ix["lens_out"].view(batch, seq_len), ix["kernel_lens"])
        kb = ix["kbar_out"]
        if ix["selects"]:
            assert torch.equal(_bits(kb[:, :nb]), _bits(ix["kbar"])), f"{label}: the caller's compressed keys differ from the slot's"
        assert torch.isnan(kb[:, nb:].float()).all(), f"{label}: a row past the prompt's complete blocks was written"
        if lens_t is not None:
            for bi, ln in enumerate(seq_lens):
                nb_b = ln // BS
                assert (
                    torch.isnan(kb[bi, nb_b:nb].float()).all() and not torch.isnan(kb[bi, :nb_b].float()).any()
                ), f"{label}: entry {bi}: rows past its {nb_b} blocks"
    print(f"{label}: {m1} | {m2}")


@requires_rubin
def test_indexer_outputs_feed_a_caller_list_block_bitwise():
    """The step-0 reuse pin: the list the indexer block hands back (``block_ids_out`` / ``block_lens_out``) fed to a caller-list
    block on the same inputs reproduces ``out`` / the gated O / the LSE BITWISE (one core, one list, one set of operands); the
    list equals the one a second indexer block leaves in its workspace (as sets per row), and ``index_k_compressed``'s rows past
    the prompt's complete blocks stay the caller's."""
    geom_kw, B, S = GEOM_SMALL, 1, 2560
    c = _run(geom_kw, B, S, indexer=True, indexer_outputs=True, seed=23)
    _indexer_two_stage(c, "small8-2 S=2560 indexer (outputs)")
    _check(c, "small8-2 S=2560 indexer (outputs)")
    caller = _run(geom_kw, B, S, index_band=True, inp=c.inp, ref=c.ref, lists=(c.ix["ids_out"].clone(), c.ix["lens_out"].clone()))
    _check(caller, "small8-2 S=2560 caller list = the indexer's output")
    for a, b_ in zip(c.runs[0], caller.runs[0]):
        assert torch.equal(_bits(a) if a.dtype != torch.float32 else a, _bits(b_) if b_.dtype != torch.float32 else b_)
    again = _run(geom_kw, B, S, indexer=True, indexer_outputs=False, inp=c.inp, ref=c.ref)
    assert torch.equal(torch.sort(again.ix["kernel_ids"], dim=-1).values, torch.sort(c.ix["kernel_ids"], dim=-1).values)
    assert torch.equal(_bits(again.ix["kbar"]), _bits(c.ix["kbar_out"][:, : S // BS]))


@requires_rubin
def test_indexer_identity_below_the_bound_launches_only_a_requested_cache_compress():
    """At ``S <= identity_bound`` the indexer block launches nothing of its own (the list is a plan-time constant, the
    workspace the caller-list block's); with ``index_k_compressed`` given exactly ONE launch runs -- the key compress into the
    caller's cache, within one bf16 ulp of the oracle's compressed keys, rows past the prompt untouched -- and the identity
    list / counts are copied into ``block_ids_out`` / ``block_lens_out``."""
    geom_kw, B, S = GEOM_SMALL, 2, 2051
    lens_t = _i32([2051, 1500])
    plain = _run(geom_kw, B, S, indexer=True, seq_lens=lens_t, seed=5)
    m = _indexer_two_stage(plain, "small8-2 S=2051 B=2 indexer identity (no outputs)")
    assert m["identity"] and m["launches"] == 0
    _check(plain, "small8-2 S=2051 B=2 indexer identity")
    caller = _run(
        geom_kw,
        B,
        S,
        index_band=True,
        seq_lens=lens_t,
        inp=plain.inp,
        ref=plain.ref,
        lists=(plain.ix["kernel_ids"].contiguous(), plain.ix["kernel_lens"].contiguous()),
    )
    assert torch.equal(_bits(plain.runs[0][0]), _bits(caller.runs[0][0])) and plain.blk.get_workspace_size() == caller.blk.get_workspace_size()
    with_cache = _run(geom_kw, B, S, indexer=True, indexer_outputs=True, seq_lens=lens_t, inp=plain.inp, ref=plain.ref)
    assert with_cache.ix["launches"] == len(with_cache.runs) * with_cache.blk._indexer.launches_per_execute(True)  # the cache compress only
    ref_geom = RefQsaGeometry(**geom_kw, qsa=RefQsaSpec(index_band=True))
    _, _, _, _, k_c = qsa_indexer_reference(
        plain.inp["h"],
        plain.inp["w_i"],
        plain.inp["w_iq_norm"],
        plain.inp["w_ik_norm"],
        None,
        None,
        plain.inp["cos"],
        plain.inp["sin"],
        ref_geom.qsa,
        ref_geom,
        seq_lens=lens_t,
    )
    _, kbar_ref = _indexer_refs_from_slab(with_cache)
    kb = with_cache.ix["kbar_out"]
    for bi, ln in enumerate((2051, 1500)):
        nb = ln // BS
        mk = _assert_one_rounding(kb[bi, :nb], kbar_ref[bi, :nb], f"entry {bi}: the cached compressed keys")
        d_o = float((kb[bi, :nb].float() - k_c[bi, :nb].float()).abs().max())
        print(
            f"entry {bi}: cache vs the chain on the kernel's band max {mk['ulps']:.2f} x the budget ({100 * mk['bitwise']:.2f} % bitwise); vs the oracle's GEMM + chain max |d| {d_o:.2e}"
        )
        assert torch.isnan(kb[bi, nb:].float()).all()
    assert torch.equal(with_cache.ix["ids_out"].view(B, S, -1), plain.ix["kernel_ids"]) and torch.equal(
        with_cache.ix["lens_out"].view(B, S), plain.ix["kernel_lens"]
    )
    assert torch.equal(_bits(with_cache.runs[0][0]), _bits(plain.runs[0][0]))


@requires_rubin
def test_indexer_block_with_write_through_is_bitwise_the_one_without():
    """The indexer composes with the paged write-through: the same selection, ``out`` / LSE / gated O bitwise, the raw indexer
    key's pool written at the slots (the stage order: indexer, cache write, sparse SDPA)."""
    geom_kw, B, S, ps = GEOM_SMALL, 1, 2560, 16
    plain = _run(geom_kw, B, S, indexer=True, indexer_outputs=True, seed=9)
    _indexer_two_stage(plain, "small8-2 S=2560 indexer plain")
    _check(plain, "small8-2 S=2560 indexer plain")
    t, geom = B * S, plain.geom
    n_pages = t // ps + 2
    gen = torch.Generator(device="cuda").manual_seed(1)
    slot = torch.randperm(n_pages * ps, generator=gen, device="cuda")[:t].to(torch.int32)
    slot[::11] = -1
    k_cache, v_cache = _nan_pool(n_pages, geom.h_kv, ps, geom.d_head), _nan_pool(n_pages, geom.h_kv, ps, geom.d_head)
    index_cache = torch.full((n_pages, ps, geom.qsa.index_head_dim), float("nan"), device="cuda", dtype=BF16)
    wt = _run(
        geom_kw,
        B,
        S,
        indexer=True,
        indexer_outputs=True,
        seed=9,
        inp=plain.inp,
        ref=plain.ref,
        paged_kv_page_size=ps,
        pools=dict(k_cache=k_cache, v_cache=v_cache, slot_mapping=slot, index_k_raw=index_cache),
    )
    _check(wt, "small8-2 S=2560 indexer write-through")
    assert (
        torch.equal(wt.ix["ids_out"], plain.ix["ids_out"])
        and torch.equal(_bits(wt.runs[0][0]), _bits(plain.runs[0][0]))
        and torch.equal(wt.runs[0][1], plain.runs[0][1])
    )
    st = wt.blk._stages
    assert st.index(wt.blk._indexer) == st.index(wt.blk._cache_write) - 1 == st.index(wt.blk._sdpa) - 2
    lay = wt.blk._layout()
    raw = index_k_raw_view(_view(wt.ws, lay.proj, (t, geom.n_qkvg), BF16), geom, B, S)
    live = slot >= 0
    s_ = slot[live].long()
    assert torch.equal(_bits(index_cache[s_ // ps, s_ % ps, :]), _bits(raw[0, live, 0]))


# ============================================================================ the fp8 indexer cache (QsaSpec.index_cache_dtype)
E4M3 = torch.float8_e4m3fn
_E4M3_TOP_FRACTION = 0.08  # every flipped block within 8 % of the row's top bf16 score of the k-th (the standalone selector suite's bound)
# per S: the floor of the e4m3-vs-bf16 set agreement over the sparse rows, set 0.5 % UNDER the lowest measured number (Rubin, 2026-10-14: 99.54 % at
# S = 2560, 99.71 % at 2300 (B = 2, padded), 99.04 % at 4096 on the TP-1 geometry; the standalone selector's 12-seed spread at 4096 is 0.015 %) -- a
# measured bound, never tuned upward; the agreement FALLS with S (the boundary density grows), so a longer S needs its own measured entry
_E4M3_AGREEMENT_FLOOR = {2560: 0.985, 2300: 0.985, 4096: 0.985}


def _bits8(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.uint8)


def _e4m3_cache_rows(kb8: torch.Tensor, kb16: torch.Tensor, nb: int, seq_lens, label: str) -> None:
    """The caller's e4m3 ``index_k_compressed`` against the bf16 block's: the written rows are ``sat_e4m3`` of the bf16 cache rows
    (the same compress kernel on the same inputs, then the cast), a padded sequence's tail rows ``[floor(S_b / 4), floor(S / 4))``
    are ZEROS (the e4m3 form's contract), the rows past ``floor(S / 4)`` the caller's NaN sentinel in both forms."""
    b = kb8.shape[0]
    assert torch.isnan(kb8[:, nb:].float()).all() and torch.isnan(kb16[:, nb:].float()).all(), f"{label}: a row past the prompt's complete blocks was written"
    for bi in range(b):
        nb_b = nb if seq_lens is None else min(nb, int(seq_lens[bi]) // BS)
        assert torch.equal(_bits8(kb8[bi, :nb_b]), _bits8(kb16[bi, :nb_b].to(E4M3))), f"{label}: entry {bi}: the e4m3 cache is not the cast of the bf16 one"
        assert (kb8[bi, nb_b:nb].float() == 0).all(), f"{label}: entry {bi}: a padded sequence's tail rows are not zeros"
        if nb_b < nb:
            assert torch.isnan(kb16[bi, nb_b:nb].float()).all(), f"{label}: entry {bi}: the bf16 form wrote a padded sequence's tail rows"


def _indexer_e4m3_two_stage(cell8: _Cell, cell16: _Cell, label: str) -> dict:
    """Stage 1 of the two-stage oracle on an fp8-indexer-cache cell, beside the bf16 indexer block on the SAME inputs: the count rule,
    the prefix form and the canonical order on every row; the launch census; the e4m3 operands BITWISE the e4m3 rounding of the
    kernel's own bf16 slots; past the bound (a) the selection as a set == the top-k of the fp64 scores of the kernel's OWN e4m3
    operands up to the fp32 tie budget (the arm computes its function), (b) against the fp32-operand oracle under the margin rule
    with ``tol`` = the MEASURED e4m3 perturbation of the scores (+ the tie budget), (c) against the bf16 block's selection: the
    agreement over the SPARSE rows (count == top_k) at or above the measured floor, every flipped block within 8 % of the row's top
    bf16 score of the bf16 k-th score (a rank-boundary flip, never a dropped dominant block); the caller's cache rows pinned."""
    geom, inp, ix8, ix16 = cell8.geom, cell8.inp, cell8.ix, cell16.ix
    q = geom.qsa
    b, s = inp["h"].shape[:2]
    bs, top_k, d_i, h_i = q.block_size, q.top_k, q.index_head_dim, q.index_heads
    dev = inp["h"].device
    seq_lens = inp["seq_lens"]
    lengths = torch.full((b,), s, dtype=torch.long, device=dev) if seq_lens is None else seq_lens.to(dev).long()
    pos = torch.arange(s, device=dev)
    counts = torch.minimum(torch.div(pos + 1, bs, rounding_mode="floor"), torch.tensor(top_k, device=dev)).expand(b, s)
    k_ids, k_lens = ix8["kernel_ids"], ix8["kernel_lens"]
    valid = k_ids >= 0
    assert torch.equal(k_lens.long(), counts), f"{label}: the counts are not min(top_k, floor((pos + 1) / 4))"
    assert torch.equal(valid, torch.arange(top_k, device=dev)[None, None, :] < counts[..., None]), f"{label}: the valid ids are not a prefix of the count"
    n_exec = len(cell8.runs)
    cache = "kbar_out" in ix8
    assert ix8["launches"] == n_exec * cell8.blk._indexer.launches_per_execute(cache), f"{label}: {ix8['launches']} launches over {n_exec} executes"
    assert cell8.blk._indexer.launches_per_execute(cache) == cell16.blk._indexer.launches_per_execute(cache) + (2 if ix8["selects"] else (1 if cache else 0))
    nb = s // bs
    m = dict(identity=not ix8["selects"], launches=ix8["launches"], executes=n_exec)
    if cache:
        _e4m3_cache_rows(ix8["kbar_out"], ix16["kbar_out"], nb, None if seq_lens is None else seq_lens.tolist(), label)
    if not ix8["selects"]:
        j = (
            counts[..., None].to(torch.int32) - 1 - torch.arange(top_k, device=dev, dtype=torch.int32)
        )  # the identity DESCENDING: the block's one canonical order
        assert torch.equal(torch.where(valid, k_ids, torch.full_like(k_ids, -1)), torch.where(valid, j, torch.full_like(j, -1))), f"{label}: not the identity"
        assert torch.equal(k_ids, ix16["kernel_ids"])
        print(f"\n{label}: the identity list (S <= {q.identity_bound}), {ix8['launches']} indexer launch(es) over {n_exec} executes (cache={cache})")
        return m
    diffs = (k_ids[..., :-1] - k_ids[..., 1:]).masked_fill(~valid[..., 1:], 1)
    assert bool((diffs > 0).all()), f"{label}: a row's list is not in descending order"
    # the operands: the kernel's e4m3 queries / compressed keys are the e4m3 rounding of ITS OWN bf16 slots (nothing near the e4m3 range)
    for nm in ("qi", "kbar"):
        x16, x8 = ix8[nm], ix8[nm + "8"]
        assert float(x16.float().abs().max()) < 448.0, f"{label}: a {nm} value at or past the e4m3 range"
        assert torch.equal(_bits8(x8), _bits8(x16.to(E4M3))), f"{label}: {nm}8 is not the e4m3 rounding of the bf16 slot"
    # the bf16 block on the same inputs: its slots are bitwise this block's bf16 slots (one compress kernel, one set of inputs)
    assert torch.equal(_bits(ix8["qi"]), _bits(ix16["qi"])) and torch.equal(
        _bits(ix8["kbar"]), _bits(ix16["kbar"])
    ), f"{label}: the bf16 slots differ between the two blocks"
    ref_geom = RefQsaGeometry(**{k: getattr(geom, k) for k in ("d_model", "h_q", "h_kv", "d_head", "rope_dim")}, qsa=RefQsaSpec(top_k=top_k, index_band=True))
    scores_o = qsa_indexer_reference(
        inp["h"], inp["w_i"], inp["w_iq_norm"], inp["w_ik_norm"], None, None, inp["cos"], inp["sin"], ref_geom.qsa, ref_geom, seq_lens=seq_lens
    )[0]
    live = pos[None, :] < lengths[:, None]
    limit = torch.arange(nb, device=dev)[None, None, :] < torch.div(pos + 1, bs, rounding_mode="floor")[None, :, None]

    def own_scores(qi, kbar):
        sc = torch.relu(torch.einsum("bshd,bjd->bshj", qi.view(b, s, h_i, d_i).double(), kbar.double())).sum(2) / math.sqrt(d_i)
        return sc.masked_fill(~limit, float("-inf"))

    s8, s16 = own_scores(ix8["qi8"].float(), ix8["kbar8"].float()), own_scores(ix16["qi"], ix16["kbar"])
    in8 = _membership(k_ids.reshape(b * s, top_k), nb).view(b, s, nb)
    in16 = _membership(ix16["kernel_ids"].reshape(b * s, top_k), nb).view(b, s, nb)
    # (a) the set == the top-k of the kernel's own e4m3 operand scores up to ties
    fl_a, rows_a, gap_a = _set_vs_topk(in8[live], s8[live], top_k, _TIE_TOL)
    # (b) the margin rule vs the fp32-operand oracle: tol = the measured e4m3 perturbation of the scores
    vis_live = torch.isfinite(scores_o) & live[..., None]
    tol8 = float((scores_o.double() - s8).abs()[vis_live].max()) + _TIE_TOL
    fl_b, rows_b, gap_b = _set_vs_topk(in8[live], scores_o[live].double(), top_k, tol8)
    # (c) vs the bf16 block's selection on the SPARSE live rows: the agreement and the flipped blocks' distance from the bf16 k-th score
    sparse = live & (counts == top_k)
    n_sparse = int(sparse.sum())
    agree = float(((in8 & in16).sum(-1).double() / top_k)[sparse].mean()) if n_sparse else 1.0
    flipped = (in8 ^ in16) & sparse[..., None]
    n_flip = int(flipped.sum())
    max_rel_gap = 0.0
    if n_flip:
        kth16 = torch.where(in16, s16, torch.full_like(s16, float("inf"))).amin(-1)  # the bf16 k-th selected score per row
        top16 = s16.amax(-1)
        rel = ((s16 - kth16[..., None]).abs() / top16[..., None].clamp_min(1e-30))[flipped]
        assert torch.isfinite(rel).all(), f"{label}: a flipped block outside the row's visible range"
        max_rel_gap = float(rel.max())
        assert (
            max_rel_gap <= _E4M3_TOP_FRACTION
        ), f"{label}: a flipped block {max_rel_gap:.3f} x the row's top score away from the bf16 k-th (bound {_E4M3_TOP_FRACTION})"
    floor = _E4M3_AGREEMENT_FLOOR.get(s, 0.0)
    assert agree >= floor, f"{label}: e4m3-vs-bf16 set agreement {100 * agree:.3f} % below the floor {100 * floor:.1f} % over {n_sparse} sparse rows"
    m.update(
        rows_live=int(live.sum()),
        flips_vs_own=fl_a,
        gap_vs_own=gap_a,
        tol_oracle=tol8,
        flips_vs_oracle=fl_b,
        rows_vs_oracle=rows_b,
        gap_vs_oracle=gap_b,
        sparse_rows=n_sparse,
        agreement_vs_bf16=agree,
        flips_vs_bf16=n_flip,
        max_rel_gap_vs_bf16=max_rel_gap,
    )
    print(
        f"\n{label}: {int(live.sum())} live rows, {n_sparse} sparse; vs the kernel's own e4m3 operands {fl_a} flips ({rows_a} rows, gap {gap_a:.2e}); "
        f"vs the fp32-operand oracle {fl_b} flips ({rows_b} rows, gap {gap_b:.3e} <= tol {tol8:.3e}); vs the bf16 block {100 * agree:.3f} % set agreement "
        f"({n_flip} flipped blocks, max gap {100 * max_rel_gap:.2f} % of the row's top score; floor {100 * floor:.1f} %)"
    )
    return m


# (id, geometry, B, S, seq_lens, caller outputs)
_INDEXER_E4M3_MATRIX = [
    pytest.param(GEOM_SMALL, 1, 2560, None, True, id="small8-2-s2560-e4m3-indexer-outputs"),
    pytest.param(GEOM_SMALL, 2, 2300, (2300, 2100), True, id="small8-2-s2300-b2-e4m3-indexer-lens-2100-outputs"),
    pytest.param(GEOM_FLASH_NEXT, 1, 4096, None, False, id="fn24-2-s4096-e4m3-indexer"),
    pytest.param(GEOM_SMALL, 2, 2051, (2051, 1500), True, id="small8-2-s2051-b2-e4m3-identity-cache"),
]


@requires_rubin
@pytest.mark.parametrize("geom_kw, batch, seq_len, seq_lens, outputs", _INDEXER_E4M3_MATRIX)
def test_indexer_e4m3_cache_block_matches_the_two_stage_oracle_and_the_bf16_block(request, geom_kw, batch, seq_len, seq_lens, outputs):
    """One fp8-indexer-cache cell beside the bf16 indexer block on the same inputs: stage 1 (``_indexer_e4m3_two_stage``), stage 2
    (attention exact given the kernel's list, ``_check``), the list contract on every live row; below the identity bound the block
    is bitwise the bf16 indexer block (the same identity list, the same core) and only the cache launches run."""
    lens_t = None if seq_lens is None else _i32(list(seq_lens))
    label = request.node.callspec.id
    cell8 = _run(geom_kw, batch, seq_len, indexer=True, indexer_outputs=outputs, seq_lens=lens_t, seed=seq_len + 7 * batch + 1, index_cache_dtype=E4M3)
    cell16 = _run(geom_kw, batch, seq_len, indexer=True, indexer_outputs=outputs, seq_lens=lens_t, inp=cell8.inp)
    m1 = _indexer_e4m3_two_stage(cell8, cell16, label)
    pos = torch.arange(seq_len, device="cuda").expand(batch, seq_len).reshape(-1)
    kv = torch.full((batch * seq_len,), seq_len, dtype=torch.long, device="cuda") if lens_t is None else lens_t.long().repeat_interleave(seq_len)
    live = pos < kv
    bad = block_ids_contract_violations(
        cell8.ids.reshape(batch * seq_len, -1)[live], pos[live], kv[live], block_size=BS, block_lens=cell8.lens.reshape(-1)[live], top_k=cell8.geom.qsa.top_k
    )
    assert bad == [], bad
    m2 = _check(cell8, label)
    if m1["identity"]:
        assert torch.equal(_bits(cell8.runs[0][0]), _bits(cell16.runs[0][0])) and torch.equal(
            cell8.runs[0][1], cell16.runs[0][1]
        ), f"{label}: the identity path differs from the bf16 indexer block"
    print(f"{label}: {m1} | {m2}")


@requires_rubin
@pytest.mark.parametrize("cache_dtype", [None, E4M3], ids=["bf16", "e4m3"])
@pytest.mark.parametrize("rounding", INDEX_COMPRESS_ROUNDINGS)
def test_indexer_prefill_form_writes_the_declared_rounding_bitwise(rounding, cache_dtype):
    """The PREFILL form (``kv_source="self"``, ``index_k_compressed`` out) under ``QsaSpec.index_compress_rounding``: the block's compressed keys
    -- the caller's cache AND the workspace slot -- and its indexer queries (the ``ix_q`` slot) are BITWISE the compress kernel compiled with the
    SAME attribute over the block's OWN raw band (the slab's index band read back from the workspace: the FROST GEMM's rows, so the chain alone is
    under test), under both conventions and both cache dtypes (e4m3: the cast kernel's rows = the compress kernel's e4m3 store arm, the e4m3 cast of
    the once-rounded bf16 row) -- red for ``"hf"`` when the form compiles the default chain whatever the declaration says.  The attention over the
    list is the sibling cells'; this cell pins the WRITERS."""
    from cudnn.gated_attention_block.kernels.qsa_compress import compile_qsa_compress, run_qsa_compress

    geom_kw, B, S = GEOM_SMALL, 1, 2560  # past the identity bound (2051): the queries, the compressed keys and the scorer run
    c = _run(geom_kw, B, S, indexer=True, indexer_outputs=True, seed=29, launches=1, ref=(), index_cache_dtype=cache_dtype, index_compress_rounding=rounding)
    geom, inp, ix = c.geom, c.inp, c.ix
    q = geom.qsa
    assert ix["selects"] and c.blk._indexer.rounding == rounding
    cd = BF16 if cache_dtype is None else cache_dtype
    t, h_i, d_i, nb = B * S, q.index_heads, q.index_head_dim, S // BS
    lay = c.blk._layout()
    proj = _view(c.ws, lay.proj, (t, geom.n_qkvg), BF16)
    k_raw = index_k_raw_view(proj, geom, B, S)[:, :, 0]  # [B, S, d_i]: the kernel's own raw indexer key band
    q_raw = torch.as_strided(proj, (t, h_i, d_i), (geom.n_qkvg, d_i, 1), proj.storage_offset() + geom.qkvg_offsets[4])  # [T, H_i, d_i]
    cos_q = inp["cos"].reshape(t, 1, geom.rope_dim).expand(t, h_i, geom.rope_dim)  # pool 1: batch = token, block = head (the stage's own launch form)
    sin_q = inp["sin"].reshape(t, 1, geom.rope_dim).expand(t, h_i, geom.rope_dim)
    eps, stream = float(q.index_norm_eps), torch.cuda.current_stream().cuda_stream

    def compress(x, w, cos, sin, pool, out_dtype):  # the from-scratch compress under THIS cell's attribute
        rec = compile_qsa_compress(
            dtype=BF16, d=d_i, rope_dim=geom.rope_dim, pool=pool, has_seq_lens=False, table_dtype=BF16, out_dtype=out_dtype, rounding=rounding
        )
        out = torch.full((int(x.shape[0]), int(x.shape[1]) // pool, d_i), float("nan"), device="cuda", dtype=out_dtype)
        run_qsa_compress(rec, x.contiguous(), out, w, cos, sin, None, eps=eps, norm_weight_offset=0.0, stream=stream)
        torch.cuda.synchronize()
        return out

    bits = (lambda x: x.contiguous().view(torch.uint8)) if cd == E4M3 else _bits  # noqa: E731
    kb16, qi16 = compress(k_raw, inp["w_ik_norm"], inp["cos"], inp["sin"], BS, BF16), compress(q_raw, inp["w_iq_norm"], cos_q, sin_q, 1, BF16)
    assert torch.equal(_bits(ix["kbar"]), _bits(kb16)), f"{rounding}: the compressed-key slot is not the from-scratch compress under the attribute"
    assert torch.equal(_bits(ix["qi"]), _bits(qi16)), f"{rounding}: the query slot is not the from-scratch pool-1 compress under the attribute"
    kb, qi = (
        (kb16, qi16)
        if cd == BF16
        else (compress(k_raw, inp["w_ik_norm"], inp["cos"], inp["sin"], BS, cd), compress(q_raw, inp["w_iq_norm"], cos_q, sin_q, 1, cd))
    )
    assert torch.equal(bits(ix["kbar_out"][:, :nb]), bits(kb)), f"{rounding} / {cd}: the caller's cache is not the from-scratch compress under the attribute"
    assert torch.isnan(ix["kbar_out"][:, nb:].float()).all(), "a row past the prompt's complete blocks was written"
    if cd == E4M3:
        assert torch.equal(bits(ix["kbar8"]), bits(kb)) and torch.equal(bits(ix["qi8"]), bits(qi)), f"{rounding}: the e4m3 scorer operands"
    # the two conventions are different functions on these rows (the attribute is not a no-op): the other chain's compress differs somewhere
    other = [r for r in INDEX_COMPRESS_ROUNDINGS if r != rounding][0]
    rec_o = compile_qsa_compress(dtype=BF16, d=d_i, rope_dim=geom.rope_dim, pool=BS, has_seq_lens=False, table_dtype=BF16, out_dtype=BF16, rounding=other)
    kb_o = torch.full_like(kb16, float("nan"))
    run_qsa_compress(rec_o, k_raw.contiguous(), kb_o, inp["w_ik_norm"], inp["cos"], inp["sin"], None, eps=eps, norm_weight_offset=0.0, stream=stream)
    torch.cuda.synchronize()
    frac = float((_bits(kb_o) == _bits(kb16)).float().mean())
    assert frac < 1.0, f"the {other!r} chain is bitwise the {rounding!r} chain on {nb} blocks -- the attribute would be untestable here"
    print(
        f"\n{rounding} / {cd}: {nb} compressed keys + {t * h_i} query rows bitwise the from-scratch compress; {100 * frac:.2f} % of the keys' elements equal the {other!r} chain's"
    )


@requires_rubin
def test_indexer_e4m3_cache_execute_captured_replays_bitwise_eager():
    """The replay pin of the fp8 indexer cache: a FRESH e4m3-indexer declaration runs ONE eager execute on a new stream (the FROST GEMM's
    graph route creates its handle there -- the route's own capture rule), then its next execute is captured in ``torch.cuda.graph`` on
    that stream under ``set_sync_debug_mode("error")`` (no host sync inside: the casts' scale word, the unit scale blobs and the padded
    prefixes are plan constants or sync-free device math); the capture launches nothing, the replay is bitwise the eager run of the
    same declaration on ``out`` / the LSE, the list, the counts and the e4m3 cache, and a second replay is bitwise the first."""
    geom_kw, B, S, top_k = GEOM_SMALL, 1, 2560, 512
    eager = _run(geom_kw, B, S, indexer=True, indexer_outputs=True, seed=31, index_cache_dtype=E4M3, launches=1)
    _indexer_e4m3_two_stage(eager, _run(geom_kw, B, S, indexer=True, indexer_outputs=True, inp=eager.inp), "small8-2 S=2560 e4m3 (eager twin)")
    inp, geom = eager.inp, eager.geom
    t = B * S
    out = torch.empty(B, S, geom.d_model, device="cuda", dtype=BF16)
    blk = GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, geom, return_lse=True)
    assert blk.check_support()
    blk.compile()
    ws = torch.full((blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device="cuda")
    lse = torch.empty(B, geom.h_q, S, device="cuda", dtype=torch.float32)
    ids_out = torch.full((t, top_k), -7, dtype=torch.int32, device="cuda")
    lens_out = torch.full((t,), -7, dtype=torch.int32, device="cuda")
    kbar_out = torch.full((B, S // BS + 3, geom.qsa.index_head_dim), float("nan"), dtype=E4M3, device="cuda")
    kw = dict(lse=lse, w_iq_norm=inp["w_iq_norm"], w_ik_norm=inp["w_ik_norm"], block_ids_out=ids_out, block_lens_out=lens_out, index_k_compressed=kbar_out)
    args = (inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        blk.execute(*args, **kw)  # the one eager execute on the capture stream (the GEMM route's handle), its results checked below too
    torch.cuda.synchronize()
    warm = (out.clone(), lse.clone(), ids_out.clone(), lens_out.clone(), kbar_out.clone())
    e_out, e_lse, _ = eager.runs[0]
    assert (
        torch.equal(_bits(warm[0]), _bits(e_out)) and torch.equal(warm[1], e_lse) and torch.equal(warm[2], eager.ix["ids_out"])
    ), "the stream run differs from the eager twin"
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    ids_out.fill_(-7)
    kbar_out.fill_(float("nan"))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                blk.execute(*args, **kw)  # the second execute of this declaration, captured
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(out).all() and (ids_out == -7).all(), "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        rep = (out.clone(), lse.clone(), ids_out.clone(), lens_out.clone(), kbar_out.clone())
        assert torch.isfinite(rep[0].float()).all()
        assert torch.equal(_bits(rep[0]), _bits(e_out)) and torch.equal(rep[1], e_lse), "the replay differs from the eager e4m3 block on out / LSE"
        assert torch.equal(rep[2], eager.ix["ids_out"]) and torch.equal(rep[3], eager.ix["lens_out"]), "the replay's list differs from the eager one"
        assert torch.equal(_bits8(rep[4]), _bits8(eager.ix["kbar_out"])), "the replay's e4m3 cache differs from the eager one"
        # a second replay over the same pointers is bitwise the first (no state leaks between replays)
        out.fill_(float("nan"))
        ids_out.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(out), _bits(rep[0])) and torch.equal(ids_out, rep[2])
    finally:
        graph.reset()


@requires_rubin
def test_indexer_block_from_torchs_default_stream_wraps_no_external_stream(monkeypatch):
    """Recipe R1 through the indexer block: executed with no ``current_stream`` from torch's default stream (the raw handle ``0`` every eager
    caller hands in), the stage's row sort runs under ``as_torch_stream`` -- torch's default stream itself -- and ``torch.cuda.ExternalStream``
    is never constructed anywhere on the path (torch before 2.13 made ``ExternalStream(0)`` a fresh non-blocking pool stream, unordered
    against the top-k that writes the list and the sparse core that reads the sorted one: a stale or half-sorted list attended, silently).
    The detector: ``ExternalStream`` monkeypatched to raise (the previous tree's hand-rolled wrapper trips it), the two-stage oracle on the
    result."""

    def boom(*a, **k):
        raise AssertionError(f"torch.cuda.ExternalStream{a} constructed on the execute path: a raw handle goes through as_torch_stream (R1)")

    monkeypatch.setattr(torch.cuda, "ExternalStream", boom)
    assert torch.cuda.current_stream().cuda_stream == 0, "the probe needs torch on its default stream (the handle 0)"
    c = _run(GEOM_SMALL, 1, 2560, indexer=True, indexer_outputs=True, seed=41)
    _indexer_two_stage(c, "small8-2 S=2560 indexer from torch's default stream, no ExternalStream")
    _check(c, "small8-2 S=2560 indexer from torch's default stream, no ExternalStream")


@requires_rubin
def test_indexer_identity_block_first_use_on_an_explicit_stream_with_the_ambient_stream_parked_past_compile():
    """The delayed-producer first-use pin on the IDENTITY branch (``S <= identity_bound``: the list and the counts are plan constants the
    stage copies out and the sparse core reads): the ambient (default) stream is busy with a long spin from BEFORE ``compile()`` until PAST
    the first execute on a side stream.  Plan constants FILLED at ``compile()`` on the ambient stream would sit behind the spin while the
    side stream's DtoD copies and the sparse core read them -- garbage block ids attended, no error (the previous tree); constants whose
    values the first execute uploads on ITS launch stream behind compile's event cannot: ``block_ids_out`` / ``block_lens_out`` equal the
    host-derived identity and ``out`` / the LSE are bitwise a synchronised run of the same declaration.  The premise -- the ambient stream
    still busy when ``compile()`` returned and when the indexer stage (the constants' consumer) was issued -- is recorded by a wrap of the
    stage's execute and asserted, never assumed; a lost premise retries once with a park sized from the measured compile time."""
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is the park; the matmul fallback is too short to outlast compile()")
    import time

    import cuda.bindings.driver as cuda_drv
    from gated_block_stream_probe import park_the_default_stream

    geom_kw, B, S, top_k = GEOM_SMALL, 2, 2051, 512
    sync = _run(geom_kw, B, S, indexer=True, indexer_outputs=True, seed=47, launches=1)  # the synchronised run: compile + execute on one stream
    assert not sync.ix["selects"]
    inp, geom = sync.inp, sync.geom
    t = B * S
    pos = torch.arange(S, device="cuda").repeat(B)
    counts = torch.minimum((pos + 1) // BS, torch.full_like(pos, top_k)).to(torch.int32)
    j = torch.arange(top_k, device="cuda", dtype=torch.int32)
    identity = torch.where(
        j[None, :] < counts[:, None], counts[:, None] - 1 - j[None, :], torch.full((t, top_k), -1, dtype=torch.int32, device="cuda")
    )  # DESCENDING
    ambient, side = torch.cuda.default_stream(), torch.cuda.Stream()
    assert torch.cuda.current_stream() == ambient and side != ambient
    ws_bytes = sync.blk.get_workspace_size()  # the same declaration's size: every buffer is allocated and filled BEFORE the park
    park_s, premise = 30.0, {}
    for _attempt in range(2):
        out = torch.empty(B, S, geom.d_model, device="cuda", dtype=BF16)
        lse = torch.empty(B, geom.h_q, S, device="cuda", dtype=torch.float32)
        blk = GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, geom, return_lse=True)
        assert blk.check_support()
        ws = torch.full((ws_bytes,), 0x7F, dtype=torch.uint8, device="cuda")
        ids_out = torch.full((t, top_k), -7, dtype=torch.int32, device="cuda")
        lens_out = torch.full((t,), -7, dtype=torch.int32, device="cuda")
        stage, orig_exec = blk._indexer, blk._indexer.execute

        def at_the_consumer(*a, _orig=orig_exec, _st=stage, **kw):
            premise["parked_at_consumer"] = not ambient.query()  # recorded when the constants' consumer is issued, then the wrap is gone
            del _st.execute
            return _orig(*a, **kw)

        stage.execute = at_the_consumer
        torch.cuda.synchronize()
        park_the_default_stream(park_s)  # the AMBIENT stream is busy from here, through compile() and past the first execute
        t0 = time.time()
        blk.compile()
        premise["compile_s"] = time.time() - t0
        premise["parked_after_compile"] = not ambient.query()
        assert blk.get_workspace_size() == ws_bytes
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
            lse=lse,
            w_iq_norm=inp["w_iq_norm"],
            w_ik_norm=inp["w_ik_norm"],
            block_ids_out=ids_out,
            block_lens_out=lens_out,
            current_stream=cuda_drv.CUstream(side.cuda_stream),
        )
        side.synchronize()  # the EXECUTION stream only
        print(f"first-use premise: {premise} (park {park_s:.0f} s)")
        if premise["parked_after_compile"] and premise.get("parked_at_consumer"):
            break
        torch.cuda.synchronize()
        park_s = max(2 * park_s, 4.0 * premise["compile_s"] + 20.0)
    else:
        pytest.fail(f"the ambient stream was not busy through compile() and the indexer stage's issue even with a {park_s:.0f} s park: {premise}")
    torch.cuda.synchronize()
    assert torch.equal(ids_out, identity), "the identity list read on the side stream is not the plan's (a constant written on the parked ambient stream)"
    assert torch.equal(lens_out, counts), "the counts read on the side stream are not the plan's"
    s_out, s_lse, _ = sync.runs[0]
    assert torch.equal(_bits(out), _bits(s_out)) and torch.equal(lse, s_lse), "the first use on the explicit stream differs from the synchronised run"


@requires_rubin
@pytest.mark.parametrize("cache_dtype", [None, E4M3], ids=["bf16", "e4m3"])
def test_indexer_execute_allocates_nothing_of_its_own_and_never_synchronizes(cache_dtype):
    """Recipe R9 on the indexer block past the bound, both indexer-cache dtypes: after one warm execute (the first execute's constant upload,
    the GEMM route's handle), three executes under ``torch.cuda.set_sync_debug_mode("error")`` make NO CUDA allocation at all (Rule 1) -- the
    FROST canonical-order kernel replaced the ``torch.sort`` windows and their 4 KB int64 index template (2026-10-29; ``torch.sort`` is never called
    on the path, pinned by wrapping it) -- and no host sync (Rule 3).  The e4m3 arm pays nothing the bf16 arm does not: its padded scale prefix is a
    plan constant handed to ``qsa_select_forward``; the standalone selector without one derives it per call, and the same census sees THAT
    allocate -- the detector is live -- and with the plan's prefix handed in it is quiet."""
    geom_kw, B, S, top_k = GEOM_SMALL, 1, 2560, 512
    c = _run(geom_kw, B, S, indexer=True, indexer_outputs=True, seed=53, index_cache_dtype=cache_dtype, launches=1)
    blk, inp, ws, geom = c.blk, c.inp, c.ws, c.geom
    out = torch.empty(B, S, geom.d_model, device="cuda", dtype=BF16)
    lse = torch.empty(B, geom.h_q, S, device="cuda", dtype=torch.float32)
    kw = dict(
        lse=lse,
        w_iq_norm=inp["w_iq_norm"],
        w_ik_norm=inp["w_ik_norm"],
        block_ids_out=c.ix["ids_out"],
        block_lens_out=c.ix["lens_out"],
        index_k_compressed=c.ix["kbar_out"],
    )
    args = (inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws)
    blk.execute(*args, **kw)  # warm on this stream (the cell's execute above ran on it too)
    torch.cuda.synchronize()
    stat = lambda: torch.cuda.memory_stats()["allocation.all.allocated"]  # noqa: E731
    sort_calls, orig_sort = [0], torch.sort

    def counted_sort(*a, **k):
        sort_calls[0] += 1
        return orig_sort(*a, **k)

    n0 = stat()
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(torch, "sort", counted_sort)
            for _ in range(3):
                blk.execute(*args, **kw)
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    torch.cuda.synchronize()
    n1 = stat()
    assert sort_calls[0] == 0, f"torch.sort ran {sort_calls[0]} times on the execute path: the canonical order is the FROST kernel's (2026-10-29)"
    assert n1 - n0 == 0, f"the indexer block made {n1 - n0} CUDA allocation(s) over three executes (allocation.all.allocated {n0} -> {n1}; cache {cache_dtype})"
    # the stage's launch counter over EVERY execute of this block so far: the cell's own runs, the warm execute and the three executes above,
    # each with the caller's cache handed in (launches_per_execute(True): the cache compress, the two compresses, the scorer pair, the canonical
    # kernel; + the two casts on the e4m3 arm) -- a wrong launch count on the execute path fails here
    want_launches = (len(c.runs) + 4) * blk._indexer.launches_per_execute(True)
    assert blk._indexer.launches == want_launches, f"{blk._indexer.launches} indexer launches over {len(c.runs) + 4} executes, expected {want_launches}"
    assert torch.equal(_bits(out), _bits(c.runs[0][0])) and torch.equal(c.ix["ids_out"].view(B, S, top_k), c.ix["kernel_ids"]), "the re-executes differ"
    print(f"R9 census ({cache_dtype}): block 3 executes {n1 - n0} allocation(s), torch.sort calls {sort_calls[0]}, the block's own 0")
    if cache_dtype is E4M3:
        from cudnn.QSA.selection import qsa_select_forward

        st = blk._indexer
        v = st.views(ws, blk._layout())
        sel_kw = dict(
            top_k=top_k,
            block_size=BS,
            max_seqlen_q=S,
            deterministic=True,
            block_ids_out=v["ids"],
            scores_out=v["scores"],
            cand_buffer=v["cand"],
            cand_batch_offsets=st._const["cand_batch_offsets"],
        )
        m0 = stat()
        qsa_select_forward(v["qi8"], v["kbar8"], st._const["cu_seqlens_q"], **sel_kw)  # the standalone form: the prefix derived per call
        m1 = stat()
        assert m1 > m0, "the detector is dead: the standalone e4m3 selector without a precomputed prefix made no allocation"
        qsa_select_forward(v["qi8"], v["kbar8"], st._const["cu_seqlens_q"], cu_seqlens_q_scale_padded=st._const["cu_seqlens_q_scale_padded"], **sel_kw)
        m2 = stat()
        assert m2 == m1, f"the e4m3 selector allocated {m2 - m1} block(s) although the scale prefix was handed in"
        torch.cuda.synchronize()
        print(f"R9 control: standalone e4m3 select per-call prefix {m1 - m0} allocation(s), with the plan's prefix {m2 - m1}")


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
    within the output bound (``assert_output_close`` against ITS sequence's oracle), the LSE within ``atol 2e-2``, no row dead
    (self-attention: every token sees itself).  The worst magnitudes over the sequences are printed and returned."""
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
    mags = dict(cos=1.0, d_out=0.0, d_o=0.0, d_lse=0.0, seqs=0, o_bound_ratio=0.0)

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
        ob = assert_output_close(o[lo:hi], ref.o_gated[0], f"sequence {i} (rows {lo}:{hi}): gated O vs its oracle")
        mags["o_bound_ratio"] = max(mags["o_bound_ratio"], ob["bound_ratio"])
        assert d_lse <= ATOL, f"LSE off the oracle by {d_lse} (budget {ATOL})"

    failures = compare_packed(c.refs, c.meta["lens"], check)
    assert not failures, f"{label}:\n" + "\n".join(failures)
    c.mags = mags
    print(
        f"\n{label}: lens {c.meta['lens']} ({mags['seqs']} live sequences): min cos(out) {mags['cos']:.6f} max|d out| {mags['d_out']:.4e} "
        f"max|d O_gated| {mags['d_o']:.4e} (worst {mags['o_bound_ratio']:.3f} of the output bound) max|d LSE| {mags['d_lse']:.2e}"
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
    every standard assertion of the matrix (``isfinite``, no sentinel, dead rows exact, ``cos > 0.999`` on ``out``, the gated O
    within the output bound, ``atol 2e-2`` on the LSE, two launches bitwise) -- and (b) the UNFUSED block on the SAME inputs and list:
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
    assert_output_close(o_f, o_u, f"{label}: gated O, fused vs unfused-then-gated")
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
    GATE slice) against (a) every sequence's own oracle (``_check_thd``: ``cos(out) > 0.999``, the gated O within the output bound,
    the LSE within ``atol 2e-2``, two launches bitwise, no sentinel) and (b) the UNFUSED packed block on the SAME inputs and lists:
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
    assert_output_close(o_f, o_u, f"{label}: gated O, fused vs unfused-then-gated")
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


# ---------------------------------------------------------------------------- the in-block indexer's DECODE step over the pools (2026-10-21)


def _extend_indexer_block(geom, inp, s_q, *, ps, k_cache, v_cache, table, icache, ring, ring_table, form, dtype=BF16):
    """One EXTEND-step declaration of the in-block indexer over the pools (``index_select_form=form``: ``"auto"`` past the decode scorer's envelope,
    the ``"extend"`` pin below it; ``index_state_slot_mapping_present=True`` -- the ring's tail write takes the stack's slots; ``fuse_prepare`` None
    reads False) at ``[1, s_q]`` on EMPTY chunk inputs -- the prompt's weights, the K / V pools + the three side caches as samples, the gate fused,
    the LSE out -- checked, compiled, with its sentinel-filled workspace."""
    mk = lambda *shape, dt=dtype: torch.empty(*shape, dtype=dt, device="cuda")  # noqa: E731
    blk = GatedAttentionBlockFwd(
        mk(1, s_q, geom.d_model),
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        mk(1, s_q, geom.rope_dim),
        mk(1, s_q, geom.rope_dim),
        inp["w_o"],
        mk(1, s_q, geom.d_model),
        geom,
        paged_kv_page_size=ps,
        kv_source=KV_SOURCE_PAGED,
        sample_k_cache=k_cache,
        sample_v_cache=v_cache,
        sample_block_table=table,
        sample_index_kv_cache=icache,
        sample_index_state_cache=ring,
        sample_index_state_block_table=ring_table,
        return_lse=True,
        fuse_gate=True,
        index_state_slot_mapping_present=True,
        index_select_form=form,
    )
    assert blk.indexer_extend and blk.index_select_form_resolved == "extend" and blk.fuse_prepare is False and blk.index_cache_dtype == icache.dtype
    assert blk._cache_write.index_ring and blk._cache_write.index_ring_cap == int(ring.shape[1]) and blk.check_support()
    blk.compile()
    ws = torch.full((blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device="cuda")
    return blk, ws


def _decode_indexer_block(geom, inp, s_q, *, ps, k_cache, v_cache, table, icache, ring, ring_table, fuse_prepare, dtype=BF16, sparse_tile_form=None):
    """One decode-indexer declaration (``QsaSpec(index_source="indexer")`` over the pools; the geometry's ``index_cache_dtype`` names the
    cache's dtype -- ``icache`` is in it) at ``[1, s_q]`` on EMPTY step inputs -- the prefill's weights, the K / V pools + the three side
    caches as samples, the gate fused, the LSE out -- checked, compiled, with its sentinel-filled workspace."""
    mk = lambda *shape, dt=dtype: torch.empty(*shape, dtype=dt, device="cuda")  # noqa: E731
    blk = GatedAttentionBlockFwd(
        mk(1, s_q, geom.d_model),
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        mk(1, s_q, geom.rope_dim),
        mk(1, s_q, geom.rope_dim),
        inp["w_o"],
        mk(1, s_q, geom.d_model),
        geom,
        paged_kv_page_size=ps,
        kv_source=KV_SOURCE_PAGED,
        sample_k_cache=k_cache,
        sample_v_cache=v_cache,
        sample_block_table=table,
        sample_index_kv_cache=icache,
        sample_index_state_cache=ring,
        sample_index_state_block_table=ring_table,
        return_lse=True,
        fuse_gate=True,
        fuse_prepare=fuse_prepare,
        **({} if sparse_tile_form is None else dict(sparse_tile_form=sparse_tile_form)),  # the sparse core's body (appended; None = the block's "auto")
    )
    assert blk.indexer_decode and blk.fuse_prepare is fuse_prepare and blk.index_cache_dtype == icache.dtype and blk.check_support()
    blk.compile()
    ws = torch.full((blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device="cuda")
    return blk, ws


@requires_rubin
@pytest.mark.parametrize("rounding", INDEX_COMPRESS_ROUNDINGS)
@pytest.mark.parametrize("cache_dtype", [None, E4M3], ids=["bf16", "e4m3"])
@pytest.mark.parametrize("fuse_prepare", [False, True], ids=["unfused", "fused"])
def test_decode_indexer_steps_after_a_prefill_carry_the_ring_and_the_paged_cache(fuse_prepare, cache_dtype, rounding):
    """The multi-step serving chain on ONE token stream, every writer the block's own (2026-10-29): a 301-token prompt as FOUR CHUNKS
    ``[125, 2, 1, 173]`` through the in-block indexer's EXTEND step over the pools (the first chunk IS the ``P = 0`` prefill through the serving
    caches; the chunk starts sit at residues 0 / 1 / 3 / 0 of the block grid; the two chunks shorter than ``cap = 8`` carry the ring's tail through
    the cache-write RING arm; chunks 2 and 1 pin ``index_select_form="extend"`` where ``"auto"`` would take the decode step), the paged
    compressed-key cache and the ring primed by the chunks' OWN writes -- after EVERY chunk the counts are the visible counts, the list EXACTLY the
    visible set descending (``top_k`` above every count), the chunk-start block listed by a row iff it completes at or before that row, every chunk
    row's attention within the budget of the fp32 oracle fed the kernel's list, the pools' chunk rows the slab's K / V, the ring's last ``cap``
    positions the stream's raw keys, the cache's rows for EVERY complete block of the stream BITWISE a from-scratch prefill compress of the
    stream's raw keys (the compress kernel's dense form) -- then EIGHT decode steps over the pools with the ring / cache / pools CARRIED between
    them: four at ``S_q`` 1 (the step at 303 completes block 75 from three ring rows + its own key), then ``S_q`` 4 with a rejected draft (two of
    four accepted; the draft's completing row had written block 76 -- outside the accepted reach) RE-STEPPED over the same positions (block 76
    rewritten from the accepted keys), a three-of-four acceptance, a full one.  After EVERY decode step: the counts are the visible counts (a
    completing row sees its OWN block), the list EXACTLY the visible set in descending order, the attention of every row within the budget of
    the fp32 oracle fed the kernel's list, the ring rows of the step the slab's raw keys, the pools' step rows the slab's K / V, and the cache's
    rows for every complete block of the ACCEPTED stream BITWISE the from-scratch compress -- so the chunks' rows, the rows a step completes and
    a re-stepped block are one chain.  Both knob readings of the decode steps (the chunks are always unfused).  Under the e4m3 indexer cache
    (``cache_dtype`` e4m3): the chunks' and the steps' rows STORE e4m3 directly (the compress kernels' e4m3 store arm) and after every chunk /
    step the cache's rows are BITWISE a from-scratch e4m3 compress of the stream (``compile_qsa_compress(out_dtype=e4m3)``) -- the chunk form,
    the step form and the fused prepare share ONE rounding convention; the ring stays bf16.  Under BOTH values of
    ``QsaSpec.index_compress_rounding`` (``rounding``): the chunks (the extend step), the decode steps (unfused and fused) and the from-scratch
    compress are compiled with the SAME attribute, so a writer that compiled the default chain whatever the declaration says leaves rows the
    from-scratch ``"hf"`` compress does not reproduce -- the cell is red for ``"hf"`` the moment one writer drops the attribute."""
    import test_block_qsa_decode as _qsa
    from cudnn.gated_attention_block.kernels.qsa_compress import compile_qsa_compress, run_qsa_compress

    geom_kw, S0, ps, cap, top_k = GEOM_SMALL, 301, 16, 8, 512
    chunks = [125, 2, 1, 173]  # the prompt as chunks: starts at 0 / 125 / 127 / 128 (residues 0, 1, 3, 0); two chunks shorter than cap
    assert sum(chunks) == S0
    cd = BF16 if cache_dtype is None else cache_dtype
    geom = GatedAttentionBlockGeometry(
        **geom_kw,
        causal_bottom_right=True,
        qsa=QsaSpec(top_k=top_k, index_source="indexer", index_band=True, index_cache_dtype=cache_dtype, index_compress_rounding=rounding),
    )
    ref_geom = RefGeometry(**geom_kw, causal_bottom_right=True)
    d_i, h_q, h_kv, d_model, D_ = geom.qsa.index_head_dim, geom.h_q, geom.h_kv, geom.d_model, geom.d_head
    G, rpp, n_dense = h_q // h_kv, ps // BS, sum(geom.qkvg_block_widths[:4])
    max_pages = 24  # 384 positions: the table's reach = the RoPE-table ceiling = n_blocks_max x 4
    n_pos = max_pages * ps
    gen = torch.Generator(device="cuda").manual_seed(62)
    pages = (torch.randperm(max_pages, generator=gen, device="cuda") + 1).tolist()  # block 0 the NULL page
    pages_t = torch.tensor(pages, device="cuda")
    table = torch.tensor([pages], dtype=torch.int32, device="cuda")
    n_pages = max_pages + 1
    slot_of = lambda p: pages[p // ps] * ps + p % ps  # noqa: E731
    k_cache, v_cache = _nan_pool(n_pages, h_kv, ps, D_), _nan_pool(n_pages, h_kv, ps, D_)
    # the prompt's inputs: the SAME draw the former whole-prompt prefill cell made (seed 61: the band's weights, the indexer norm weights)
    inp = make_qsa_inputs(RefQsaGeometry(**geom_kw, qsa=RefQsaSpec(top_k=top_k, index_band=True)), 1, S0, dtype=BF16, seed=61, index_source="indexer")
    cos_all, sin_all = build_rope_tables(n_pos, geom.rope_dim, base=ref_geom.rope_base, batch=1, device="cuda", dtype=BF16)
    assert torch.equal(cos_all[:, :S0], inp["cos"]) and torch.equal(sin_all[:, :S0], inp["sin"]), "the prompt's tables are the stream's first rows"
    k_raw_all = torch.full((n_pos, d_i), float("nan"), device="cuda", dtype=BF16)  # the raw band of every COMMITTED position, as the block computed it
    # the side caches START EMPTY: the chunks' own writes prime them (no scatter by the test)
    icache = torch.full((n_pages, rpp, 1, d_i), float("nan"), device="cuda", dtype=cd)
    ring = torch.full((2, cap, 1, d_i), float("nan"), device="cuda", dtype=BF16)  # block 0 the null block
    ring_table = torch.tensor([[1]], dtype=torch.int32, device="cuda")
    recipe = compile_qsa_compress(dtype=BF16, d=d_i, rope_dim=geom.rope_dim, pool=BS, has_seq_lens=False, table_dtype=BF16, out_dtype=cd, rounding=rounding)

    def compress(k_raw):  # the from-scratch prefill compress of a raw-key stream [1, 4 nb, d_i] -> [1, nb, d_i] in the CACHE dtype
        out = torch.full((1, int(k_raw.shape[1]) // BS, d_i), float("nan"), device="cuda", dtype=cd)
        run_qsa_compress(
            recipe,
            k_raw.contiguous(),
            out,
            inp["w_ik_norm"],
            cos_all,
            sin_all,
            None,
            eps=float(geom.qsa.index_norm_eps),
            norm_weight_offset=0.0,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()
        return out

    bits_c = (lambda x: x.view(torch.uint8)) if cd == E4M3 else _bits  # noqa: E731  (8-bit patterns under the e4m3 cache)
    # (1) the prompt as CHUNKS through the EXTEND step: the compressed-key cache and the ring primed by the block's own writes
    L = 0
    for ci, T in enumerate(chunks):
        pos0, Lf = L, L + T
        form = "auto" if T * 4 > 64 else "extend"  # "auto" takes the extend step past the decode scorer's envelope; the short chunks pin it
        blk, ws = _extend_indexer_block(
            geom, inp, T, ps=ps, k_cache=k_cache, v_cache=v_cache, table=table, icache=icache, ring=ring, ring_table=ring_table, form=form
        )
        assert blk._indexer_extend.n_blocks_max == max_pages * rpp and blk._indexer_extend.n_windows == 1 and blk._indexer_extend.rounding == rounding
        pos = torch.arange(T, device="cuda") + pos0
        ref = gated_attention_block_reference(
            inp["h"][:, :Lf], inp["w_qkvg"][:n_dense], inp["w_q_norm"], inp["w_k_norm"], cos_all[:, :Lf], sin_all[:, :Lf], inp["w_o"], ref_geom
        )
        out = torch.full((1, T, d_model), float("nan"), device="cuda", dtype=BF16)
        lse = torch.full((1, h_q, T), float("nan"), device="cuda", dtype=torch.float32)
        ids_out = torch.full((1, T, top_k), -7, dtype=torch.int32, device="cuda")
        lens_out = torch.full((1, T), -7, dtype=torch.int32, device="cuda")
        slots = torch.tensor([slot_of(p) for p in range(pos0, Lf)], dtype=torch.int32, device="cuda")
        # the stack's ring slots: block 1 x cap + pos % cap for the sequence's LAST cap tokens, -1 elsewhere
        ring_slots = torch.tensor([1 * cap + p % cap if p + cap >= Lf else -1 for p in range(pos0, Lf)], dtype=torch.int32, device="cuda")
        blk.execute(
            inp["h"][:, pos0:Lf].contiguous(),
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            cos_all[:, pos0:Lf].contiguous(),
            sin_all[:, pos0:Lf].contiguous(),
            inp["w_o"],
            out,
            ws,
            lse=lse,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=table,
            kv_lens=torch.tensor([Lf], dtype=torch.int32, device="cuda"),
            slot_mapping=slots,
            w_iq_norm=inp["w_iq_norm"],
            w_ik_norm=inp["w_ik_norm"],
            block_ids_out=ids_out,
            block_lens_out=lens_out,
            index_kv_cache=icache,
            index_state_cache=ring,
            index_state_block_table=ring_table,
            index_state_slot_mapping=ring_slots,
            cos_pos=cos_all,
            sin_pos=sin_all,
        )
        torch.cuda.synchronize()
        label = f"chunk {ci} T={T} pos {pos0}..{Lf - 1} form={form} rounding={rounding}"
        v = blk._indexer_extend.views(ws, blk._ws)
        assert v["seqrep"].tolist() == [Lf // BS], f"{label}: the gather's report {v['seqrep'].tolist()}"
        counts = v["counts"].view(T)
        want = ((pos + 1) // BS).clamp(max=blk._indexer_extend.n_blocks_max).to(torch.int32)
        assert torch.equal(counts, want) and torch.equal(lens_out.view(T), torch.minimum(want, torch.full_like(want, top_k))), f"{label}: counts"
        ids = ids_out.view(T, top_k)
        valid = ids >= 0
        assert torch.equal(valid, torch.arange(top_k, device="cuda")[None, :] < lens_out.view(T, 1)), f"{label}: the valid prefix"
        want_ids = torch.where(valid, counts[:, None] - 1 - torch.arange(top_k, device="cuda", dtype=torch.int32)[None, :], torch.full_like(ids, -1))
        assert torch.equal(torch.where(valid, ids, torch.full_like(ids, -1)), want_ids), f"{label}: the list is not the complete visible set, descending"
        b0 = pos0 // BS  # the chunk-start pin: block b0 (the straddling block, or the chunk's first) is listed by a row iff it completed at or before it
        for j in range(T):
            listed = bool((ids[j] == b0).any())
            assert listed == (pos0 + j >= BS * b0 + BS - 1), f"{label}: row {j} (position {pos0 + j}) {'lists' if listed else 'does not list'} block {b0}"
        lay = blk._ws
        proj = _view(ws, lay.proj, (T, geom.n_qkvg), BF16)
        raw = index_k_raw_view(proj, geom, 1, T)[0, :, 0]  # [T, d_i]: the chunk's raw keys (the FROST GEMM's band)
        k_raw_all[pos0:Lf] = raw
        # the ring after the chunk: the stream's last cap positions at their circular slots (the cache-write ring arm); the null block untouched
        for p in range(max(0, Lf - cap), Lf):
            assert torch.equal(_bits(ring[1, p % cap, 0]), _bits(k_raw_all[p])), f"{label}: ring row of position {p}"
        assert torch.isnan(ring[0].float()).all() and torch.isnan(icache[0].float()).all(), f"{label}: the null ring block / page 0 moved"
        # the cache: EVERY complete block of the stream so far bitwise the from-scratch compress of the stream's raw keys; the tail block's row unwritten
        nb = Lf // BS
        if nb:
            kb = compress(k_raw_all[None, : nb * BS])[0]
            j_all = torch.arange(nb, device="cuda")
            rows = icache[pages_t[j_all // rpp], j_all % rpp, 0]
            assert torch.equal(bits_c(rows), bits_c(kb)), f"{label}: the cache's {nb} complete blocks are not the from-scratch compress of the stream"
        if Lf % BS:
            assert torch.isnan(icache[pages[nb // rpp], nb % rpp, 0].float()).all(), f"{label}: the incomplete tail block's row was written"
        # the pools' chunk rows: the slab's K / V, bitwise
        kb_, vb_ = _cols(proj, geom.qkvg_offsets[2], h_kv, D_), _cols(proj, geom.qkvg_offsets[3], h_kv, D_)
        for j in range(T):
            pg, off = slot_of(pos0 + j) // ps, slot_of(pos0 + j) % ps
            assert torch.equal(_bits(k_cache[pg, :, off]), _bits(kb_[j])) and torch.equal(
                _bits(v_cache[pg, :, off]), _bits(vb_[j])
            ), f"{label}: pool row {pos0 + j}"
        # the two-stage oracle's second stage: every chunk row's attention over the kernel's own list, then the gate and the out projection
        allowed = qsa_visible_mask(ids, None, pos, Lf, Lf, BS, top_k=top_k, pos0=None)
        og, ref_lse = _qsa._attend(ref, allowed, pos, geom.scale, BF16, h_q, G)
        o = _view(ws, lay.o, (T, h_q, D_), BF16)
        ref_out = (og.reshape(T, h_q * D_).float() @ inp["w_o"].float().t()).to(BF16)
        assert torch.isfinite(out.float()).all() and torch.isfinite(lse).all(), f"{label}: non-finite"
        d_o, d_lse, c = float((o.float() - og.float()).abs().max()), float((lse[0] - ref_lse).abs().max()), _cos(out[0], ref_out)
        assert d_lse <= ATOL and c > COS_OUT, f"{label}: O {d_o:.3e} LSE {d_lse:.3e} cos {c:.6f}"
        assert_output_close(o, og, f"{label}: gated O vs the two-stage oracle")
        print(
            f"\n{label} cache={cd}: counts {counts[:3].tolist()}..{counts[-2:].tolist()} blocks completed {max(0, nb - b0)} O {d_o:.3e} LSE {d_lse:.3e} cos {c:.6f}"
        )
        L = Lf
    assert L == S0 and torch.isfinite(k_raw_all[:S0].float()).all()
    # (2) the EIGHT decode steps over the pools, the side caches carried from the chunks (the ring's last cap positions, the cache's 75 complete blocks)
    common = dict(ps=ps, k_cache=k_cache, v_cache=v_cache, table=table, icache=icache, ring=ring, ring_table=ring_table, fuse_prepare=fuse_prepare)
    blk1, ws1 = _decode_indexer_block(geom, inp, 1, **common)
    blk4, ws4 = _decode_indexer_block(geom, inp, 4, **common)
    assert blk1._indexer_decode.n_blocks_max == blk4._indexer_decode.n_blocks_max == max_pages * rpp
    # the decode writers carry the declared rounding: the unfused stage's recipes, or the fused prepare's arms record
    for blk in (blk1, blk4):
        assert blk._indexer_decode.rounding == rounding and (blk._prepare is None or blk._prepare._index_arms().rounding == rounding)
    # (3) the steps: (S_q, rows accepted)
    plan = [(1, 1), (1, 1), (1, 1), (1, 1), (4, 2), (4, 4), (4, 3), (4, 4)]
    L, h_acc, completed = S0, inp["h"].clone(), []
    for step, (s_q, n_acc) in enumerate(plan, 1):
        blk, ws = (blk1, ws1) if s_q == 1 else (blk4, ws4)
        pos0, Lf = L, L + s_q
        pos = torch.arange(s_q, device="cuda") + pos0
        h_step = torch.randn(1, s_q, d_model, generator=gen, device="cuda", dtype=torch.float32).to(BF16)
        h_full = torch.cat([h_acc, h_step], 1)
        ref = gated_attention_block_reference(
            h_full, inp["w_qkvg"][:n_dense], inp["w_q_norm"], inp["w_k_norm"], cos_all[:, :Lf], sin_all[:, :Lf], inp["w_o"], ref_geom
        )
        out = torch.full((1, s_q, d_model), float("nan"), device="cuda", dtype=BF16)
        lse = torch.full((1, h_q, s_q), float("nan"), device="cuda", dtype=torch.float32)
        ids_out = torch.full((1, s_q, top_k), -7, dtype=torch.int32, device="cuda")
        lens_out = torch.full((1, s_q), -7, dtype=torch.int32, device="cuda")
        slots = torch.tensor([slot_of(p) for p in range(pos0, Lf)], dtype=torch.int32, device="cuda")
        blk.execute(
            h_step,
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            cos_all[:, pos0:Lf].contiguous(),
            sin_all[:, pos0:Lf].contiguous(),
            inp["w_o"],
            out,
            ws,
            lse=lse,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=table,
            kv_lens=torch.tensor([Lf], dtype=torch.int32, device="cuda"),
            slot_mapping=slots,
            w_iq_norm=inp["w_iq_norm"],
            w_ik_norm=inp["w_ik_norm"],
            block_ids_out=ids_out,
            block_lens_out=lens_out,
            index_kv_cache=icache,
            index_state_cache=ring,
            index_state_block_table=ring_table,
            cos_pos=cos_all,
            sin_pos=sin_all,
        )
        torch.cuda.synchronize()
        label = f"step {step} S_q={s_q} pos {pos0}..{Lf - 1} accept {n_acc} fuse_prepare={fuse_prepare} rounding={rounding}"
        # the counts rule (a completing row sees its own block), the list form
        counts = blk._indexer_decode.views(ws, blk._ws)["counts"].view(s_q)
        want = ((pos + 1) // BS).clamp(max=blk._indexer_decode.n_blocks_max).to(torch.int32)
        assert torch.equal(counts, want) and torch.equal(lens_out.view(s_q), torch.minimum(want, torch.full_like(want, top_k))), f"{label}: counts"
        ids = ids_out.view(s_q, top_k)
        valid = ids >= 0
        assert torch.equal(valid, torch.arange(top_k, device="cuda")[None, :] < lens_out.view(s_q, 1)), f"{label}: the valid prefix"
        assert bool(((ids[:, :-1] - ids[:, 1:]).masked_fill(~valid[:, 1:], 1) > 0).all()), f"{label}: a list is not strictly descending"
        assert bool((torch.where(valid, ids, torch.full_like(ids, -1)).amax(-1) < counts).all()), f"{label}: an id at or past the visible count"
        # the expected list in closed form: with top_k (512) above every visible count (<= 79 blocks here) the scorer's top-k IS the visible set,
        # so the complete returned list must be exactly `count - 1 .. 0` then padding -- a wrong yet well-formed selection cannot hide in the
        # shape checks above; the BINDING top-k (fewer slots than visible blocks) is compared against the fp64 top-k of the kernel's own
        # operands in the decode module's ladder (`check_selection`) and, for the prefill form, by `_indexer_two_stage` here
        want_ids = torch.where(valid, counts[:, None] - 1 - torch.arange(top_k, device="cuda", dtype=torch.int32)[None, :], torch.full_like(ids, -1))
        assert torch.equal(torch.where(valid, ids, torch.full_like(ids, -1)), want_ids), f"{label}: the list is not the complete visible set, descending"
        for j in range(s_q):
            if (pos0 + j + 1) % BS == 0:
                k = (pos0 + j) // BS
                assert int(counts[j]) == k + 1 and bool((ids[j] == k).any()), f"{label}: row {j} completes block {k} and must see / list it (every block fits)"
                completed.append(k)
        # the ring rows of the step: the slab's raw keys, bitwise; the pools' step rows: the slab's K / V, bitwise
        lay = blk._ws
        proj = _view(ws, lay.proj, (s_q, geom.n_qkvg), BF16)
        raw = index_k_raw_view(proj, geom, 1, s_q)[0, :, 0]  # [s_q, d_i]
        for j in range(s_q):
            assert torch.equal(_bits(ring[1, (pos0 + j) % cap, 0]), _bits(raw[j])), f"{label}: ring row of position {pos0 + j}"
        kb_, vb_ = _cols(proj, geom.qkvg_offsets[2], h_kv, D_), _cols(proj, geom.qkvg_offsets[3], h_kv, D_)
        for j in range(s_q):
            pg, off = slot_of(pos0 + j) // ps, slot_of(pos0 + j) % ps
            assert torch.equal(_bits(k_cache[pg, :, off]), _bits(kb_[j])) and torch.equal(
                _bits(v_cache[pg, :, off]), _bits(vb_[j])
            ), f"{label}: pool row {pos0 + j}"
        # the two-stage oracle's second stage: every row's attention over the kernel's own list, then the gate and the out projection
        allowed = qsa_visible_mask(ids, None, pos, Lf, Lf, BS, top_k=top_k, pos0=None)
        og, ref_lse = _qsa._attend(ref, allowed, pos, geom.scale, BF16, h_q, G)
        o = _view(ws, lay.o, (s_q, h_q, D_), BF16)
        ref_out = (og.reshape(s_q, h_q * D_).float() @ inp["w_o"].float().t()).to(BF16)
        assert torch.isfinite(out.float()).all() and torch.isfinite(lse).all(), f"{label}: non-finite"
        d_o, d_lse, c = float((o.float() - og.float()).abs().max()), float((lse[0] - ref_lse).abs().max()), _cos(out[0], ref_out)
        assert d_lse <= ATOL and c > COS_OUT, f"{label}: O {d_o:.3e} LSE {d_lse:.3e} cos {c:.6f}"
        assert_output_close(o, og, f"{label}: gated O vs the two-stage oracle")
        # accept `n_acc` rows: the stream, its raw keys; then the carried state against the from-scratch compress of the ACCEPTED stream
        L = pos0 + n_acc
        h_acc = torch.cat([h_acc, h_step[:, :n_acc]], 1)
        k_raw_all[pos0:L] = raw[:n_acc]
        nb = L // BS
        kb = compress(k_raw_all[None, : nb * BS])[0]
        j_all = torch.arange(nb, device="cuda")
        rows = icache[pages_t[j_all // rpp], j_all % rpp, 0]
        assert torch.equal(bits_c(rows), bits_c(kb)), f"{label}: the cache's {nb} complete blocks are not the from-scratch compress of the accepted stream"
        for p in range(max(L - BS + 1, 0), L):  # the ring rows the NEXT step may pool: the accepted keys
            assert torch.equal(_bits(ring[1, p % cap, 0]), _bits(k_raw_all[p])), f"{label}: ring row {p} after acceptance"
        assert torch.isnan(ring[0].float()).all() and torch.isnan(icache[0].float()).all(), f"{label}: the null ring block / page 0 moved"
        print(f"\n{label} cache={cd}: counts {counts.tolist()} completed {[k for k in completed if k >= (pos0 // BS)]} O {d_o:.3e} LSE {d_lse:.3e} cos {c:.6f}")
    assert L == S0 + 17 and completed == [75, 76, 76, 77, 78], (L, completed)  # 303 | 307 (draft) | 307 (re-step) | 311 | 315


@requires_rubin
@pytest.mark.parametrize("fuse_prepare", [False, True], ids=["unfused", "fused"])
def test_decode_indexer_execute_allocates_nothing_and_runs_no_host_sort(fuse_prepare):
    """Recipe R9 on the decode indexer over the pools, both knob readings: after the warm execute, three executes under
    ``torch.cuda.set_sync_debug_mode("error")`` make NO CUDA allocation at all and call NO ``torch.sort`` -- the selection is the top-k
    KERNEL over the scorer's logits (the prefill form's one admitted transient, torch.sort's index template, has no counterpart here);
    no host sync; the re-executes bitwise (``out``, the list)."""
    import test_block_qsa_decode as _qsa

    case = _qsa._IndexerCase(GEOM_SMALL, 2, (2052, 300, 0, 4097), page=16, seed=71, split=4, fuse_gate=True, return_lse=True, fuse_prepare=fuse_prepare)
    case.compile()
    case.run()
    case.check_selection("R9 warm")
    case.check(what="R9 warm")
    out0, ids0 = case.out.clone(), case.kernel_ids().clone()
    torch.cuda.synchronize()
    stat = lambda: torch.cuda.memory_stats()["allocation.all.allocated"]  # noqa: E731
    sort_calls, orig_sort = [0], torch.sort

    def counted_sort(*a, **k):
        sort_calls[0] += 1
        return orig_sort(*a, **k)

    n0 = stat()
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(torch, "sort", counted_sort)
            for _ in range(3):
                case.execute()
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    torch.cuda.synchronize()
    assert stat() == n0, f"fuse_prepare={fuse_prepare}: the decode indexer block made {stat() - n0} CUDA allocation(s) over three executes"
    assert sort_calls[0] == 0, f"fuse_prepare={fuse_prepare}: torch.sort ran {sort_calls[0]} times on the decode path"
    assert torch.equal(_bits(case.out), _bits(out0)) and torch.equal(case.kernel_ids(), ids0), "the re-executes differ"
    print(f"\nR9 census (decode indexer, fuse_prepare={fuse_prepare}): 3 executes, 0 allocations, 0 host sorts")


@requires_rubin
def test_decode_indexer_steps_on_the_lean_tile_track_the_tcgen05_steps():
    """The end-to-end DECODE step over the lean body: the 301-token prompt through the PREFILL form of the in-block indexer (the pools
    written through, the compressed rows and the ring primed as the multi-step test does), then FOUR ``S_q`` 1 decode steps on TWO blocks
    over two copies of that state -- ``sparse_tile_form="lean"`` (the lean decode tile; its ``sparse_tile`` the lean kernel, the split 65 at
    k 1) and ``"tcgen05"`` (the core) -- fed the same tokens.  After EVERY step: the kernel-produced list and the counts BITWISE between the
    two bodies (the same prepare / scorer / top-k kernels), the ring rows, the compressed-key cache and the pools BITWISE (the same
    write-through), the attention of the step's row within the module's bars of the fp32 oracle fed the kernel's list on BOTH bodies, and
    the two bodies within the same bars of each other (``cos(out) > 0.999``, the gated O within the output bound, the LSE within 2e-2) -- never bitwise across
    the bodies (a different summation order), recorded.  The S_q 4 steps of the multi-step test select PER ROW (the indexer's lists), which
    the lean body declines by name past one row (its CTA gathers ONE list per sequence); the S_q 4 lean cells are the shared-list block's
    (the decode module)."""
    import test_block_qsa_decode as _qsa

    geom_kw, S0, ps, cap, top_k = GEOM_SMALL, 301, 16, 8, 512
    geom = GatedAttentionBlockGeometry(**geom_kw, causal_bottom_right=True, qsa=QsaSpec(top_k=top_k, index_source="indexer", index_band=True))
    ref_geom = RefGeometry(**geom_kw, causal_bottom_right=True)
    d_i, h_q, h_kv, d_model, D_ = geom.qsa.index_head_dim, geom.h_q, geom.h_kv, geom.d_model, geom.d_head
    G, rpp, n_dense = h_q // h_kv, ps // BS, sum(geom.qkvg_block_widths[:4])
    max_pages = 24
    n_pos = max_pages * ps
    gen = torch.Generator(device="cuda").manual_seed(63)
    pages = (torch.randperm(max_pages, generator=gen, device="cuda") + 1).tolist()
    table = torch.tensor([pages], dtype=torch.int32, device="cuda")
    n_pages = max_pages + 1
    slot_of = lambda p: pages[p // ps] * ps + p % ps  # noqa: E731
    k_cache, v_cache = _nan_pool(n_pages, h_kv, ps, D_), _nan_pool(n_pages, h_kv, ps, D_)
    raw_pool = torch.full((n_pages, ps, d_i), float("nan"), device="cuda", dtype=BF16)
    slots0 = torch.tensor([slot_of(p) for p in range(S0)], dtype=torch.int32, device="cuda")
    pre = _run(
        geom_kw,
        1,
        S0,
        indexer=True,
        indexer_outputs=True,
        seed=64,
        paged_kv_page_size=ps,
        pools=dict(k_cache=k_cache, v_cache=v_cache, slot_mapping=slots0, index_k_raw=raw_pool),
    )
    _check(pre, "small8-2 S=301 prefill (the stream's prompt)")
    inp = pre.inp
    cos_all, sin_all = build_rope_tables(n_pos, geom.rope_dim, base=ref_geom.rope_base, batch=1, device="cuda", dtype=BF16)
    lay0 = pre.blk._layout()
    k_raw_all = torch.full((n_pos, d_i), float("nan"), device="cuda", dtype=BF16)
    k_raw_all[:S0] = index_k_raw_view(_view(pre.ws, lay0.proj, (S0, pre.geom.n_qkvg), BF16), pre.geom, 1, S0)[0, :, 0]
    icache = torch.full((n_pages, rpp, 1, d_i), float("nan"), device="cuda", dtype=BF16)
    nb0 = S0 // BS
    kbar0 = pre.ix["kbar_out"][0, :nb0]
    for j in range(nb0):
        icache[pages[j // rpp], j % rpp, 0] = kbar0[j]
    ring = torch.full((2, cap, 1, d_i), float("nan"), device="cuda", dtype=BF16)
    for p in range(S0 - cap, S0):
        ring[1, p % cap, 0] = k_raw_all[p]
    ring_table = torch.tensor([[1]], dtype=torch.int32, device="cuda")
    # two copies of the carried state, one per body
    state = {}
    for body in ("tcgen05", "lean"):
        st = dict(k=k_cache.clone(), v=v_cache.clone(), icache=icache.clone(), ring=ring.clone())
        blk, ws = _decode_indexer_block(
            geom,
            inp,
            1,
            ps=ps,
            k_cache=st["k"],
            v_cache=st["v"],
            table=table,
            icache=st["icache"],
            ring=st["ring"],
            ring_table=ring_table,
            fuse_prepare=True,
            sparse_tile_form=body,
        )
        st.update(blk=blk, ws=ws)
        state[body] = st
    assert (
        state["lean"]["blk"].sparse_tile == "sparse_lean_d256_f16"
        and state["lean"]["blk"].decode_split_kv == 65
        and state["lean"]["blk"].sparse_sub_tiles_per_cta == 1
    )
    assert state["tcgen05"]["blk"].sparse_tile == "sparse_d256_f16" and state["tcgen05"]["blk"].sparse_sub_tiles_per_cta is None
    assert state["lean"]["blk"]._sdpa.kernel_template == "sparse_lean_d256_f16" and state["tcgen05"]["blk"]._sdpa.kernel_template == "sparse_d256_f16"
    L, h_acc = S0, inp["h"].clone()
    for step in range(1, 5):
        pos0, Lf = L, L + 1
        pos = torch.arange(1, device="cuda") + pos0
        h_step = torch.randn(1, 1, d_model, generator=gen, device="cuda", dtype=torch.float32).to(BF16)
        h_full = torch.cat([h_acc, h_step], 1)
        ref = gated_attention_block_reference(
            h_full, inp["w_qkvg"][:n_dense], inp["w_q_norm"], inp["w_k_norm"], cos_all[:, :Lf], sin_all[:, :Lf], inp["w_o"], ref_geom
        )
        res = {}
        for body, st in state.items():
            blk, ws = st["blk"], st["ws"]
            out = torch.full((1, 1, d_model), float("nan"), device="cuda", dtype=BF16)
            lse = torch.full((1, h_q, 1), float("nan"), device="cuda", dtype=torch.float32)
            ids_out = torch.full((1, 1, top_k), -7, dtype=torch.int32, device="cuda")
            lens_out = torch.full((1, 1), -7, dtype=torch.int32, device="cuda")
            slots = torch.tensor([slot_of(pos0)], dtype=torch.int32, device="cuda")
            blk.execute(
                h_step,
                inp["w_qkvg"],
                inp["w_q_norm"],
                inp["w_k_norm"],
                cos_all[:, pos0:Lf].contiguous(),
                sin_all[:, pos0:Lf].contiguous(),
                inp["w_o"],
                out,
                ws,
                lse=lse,
                k_cache=st["k"],
                v_cache=st["v"],
                block_table=table,
                kv_lens=torch.tensor([Lf], dtype=torch.int32, device="cuda"),
                slot_mapping=slots,
                w_iq_norm=inp["w_iq_norm"],
                w_ik_norm=inp["w_ik_norm"],
                block_ids_out=ids_out,
                block_lens_out=lens_out,
                index_kv_cache=st["icache"],
                index_state_cache=st["ring"],
                index_state_block_table=ring_table,
                cos_pos=cos_all,
                sin_pos=sin_all,
            )
            torch.cuda.synchronize()
            label = f"step {step} pos {pos0} {body}"
            counts = blk._indexer_decode.views(ws, blk._ws)["counts"].view(1)
            want = ((pos + 1) // BS).clamp(max=blk._indexer_decode.n_blocks_max).to(torch.int32)
            assert torch.equal(counts, want) and torch.equal(lens_out.view(1), torch.minimum(want, torch.full_like(want, top_k))), f"{label}: counts"
            ids = ids_out.view(1, top_k)
            allowed = qsa_visible_mask(ids, None, pos, Lf, Lf, BS, top_k=top_k, pos0=None)
            og, ref_lse = _qsa._attend(ref, allowed, pos, geom.scale, BF16, h_q, G)
            o = _view(ws, blk._ws.o, (1, h_q, D_), BF16)
            ref_out = (og.reshape(1, h_q * D_).float() @ inp["w_o"].float().t()).to(BF16)
            assert torch.isfinite(out.float()).all() and torch.isfinite(lse).all(), f"{label}: non-finite"
            d_o, d_lse, c = float((o.float() - og.float()).abs().max()), float((lse[0] - ref_lse).abs().max()), _cos(out[0], ref_out)
            assert d_lse <= ATOL and c > COS_OUT, f"{label}: O {d_o:.3e} LSE {d_lse:.3e} cos {c:.6f}"
            assert_output_close(o, og, f"{label}: gated O vs the two-stage oracle")
            res[body] = dict(
                out=out.clone(), lse=lse.clone(), o=o.clone(), ids=ids.clone(), lens=lens_out.clone(), counts=counts.clone(), d_o=d_o, d_lse=d_lse, cos=c
            )
        a, b = res["lean"], res["tcgen05"]
        assert (
            torch.equal(a["ids"], b["ids"]) and torch.equal(a["lens"], b["lens"]) and torch.equal(a["counts"], b["counts"])
        ), f"step {step}: the list differs between the bodies"
        sl, st_ = state["lean"], state["tcgen05"]
        assert torch.equal(_bits(sl["ring"]), _bits(st_["ring"])) and torch.equal(
            _bits(sl["icache"]), _bits(st_["icache"])
        ), f"step {step}: the side caches differ"
        assert torch.equal(_bits(sl["k"]), _bits(st_["k"])) and torch.equal(_bits(sl["v"]), _bits(st_["v"])), f"step {step}: the pools differ"
        c_ab = _cos(a["out"], b["out"])
        d_o_ab, d_lse_ab = float((a["o"].float() - b["o"].float()).abs().max()), float((a["lse"] - b["lse"]).abs().max())
        assert c_ab > COS_OUT and d_lse_ab <= ATOL, f"step {step}: lean vs tcgen05 cos {c_ab:.6f} O {d_o_ab:.3e} LSE {d_lse_ab:.3e}"
        assert_output_close(a["o"], b["o"], f"step {step}: gated O, lean vs tcgen05")
        bitwise = bool(torch.equal(_bits(a["out"]), _bits(b["out"])))
        print(
            f"\nstep {step} pos {pos0}: lean vs oracle O {a['d_o']:.3e} LSE {a['d_lse']:.3e} cos {a['cos']:.6f}; tcgen05 vs oracle O {b['d_o']:.3e} LSE {b['d_lse']:.3e} cos {b['cos']:.6f}; "
            f"lean vs tcgen05 cos {c_ab:.6f} O {d_o_ab:.3e} LSE {d_lse_ab:.3e} (bitwise {bitwise}: not required); list / counts / caches / pools bitwise"
        )
        L = Lf
        h_acc = torch.cat([h_acc, h_step], 1)
    assert L == S0 + 4
