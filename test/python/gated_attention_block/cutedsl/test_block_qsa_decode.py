# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block's block-sparse DECODE mode: a dense ``[B, S_q]`` declaration on a ``QsaSpec`` geometry with
``kv_source="paged"`` over a paged KV cache -- the step's tokens written through, then stage (4) is the index-list sparse core
READING THE POOLS: per sequence the keys of the listed 4-token blocks and the open tail over the first ``kv_lens[b]`` tokens of
its ``block_table`` row, on the bottom-right diagonal (row ``j`` at ``pos_j = kv_lens[b] - S_q + j``).  The list form is a
DECLARATION fact derived from ``S_q``: at ``S_q <= 4`` (the sparse core's shared-list cap) ONE list per sequence,
``block_ids [B, top_k]`` (+ ``block_lens [B]``), selected at the STEP-0 position ``pos_0 = kv_lens[b] - S_q`` and shared by the
step's rows -- the speculative (MTP) rows see the listed blocks plus EVERY token from the step-0 tail start to their own position
(the contract page's definition; the oracle's ``pos0``); past four rows per sequence the same paged read with one list per row
(``[B, S_q, top_k]``): the extend-with-prefix form over the pools (a chunked prefill, a prefix-cache hit) at any ``S_q``.  The
split (``decode_split_kv``: fp32 partials in the workspace + the split combine, the gate riding the combine) is the one
performance knob; the LSE is served over the pools (the core writes it).

Host cells (any device; the declaration cells even on the CPU): the declaration FACTS (the sparse stage in the paged form, the
list form per ``S_q``, no dense route, where the gate runs, the split's default rule, ``seq_lens_present`` forced, the stage
order); the typed declines (the packed form under ``QsaSpec`` over the pools -- the serving table's ``sparse`` row --, a block
table sample that is not contiguous, a split past the item's tile count, ``max_kv_len`` past the table's reach, the bottom-right
diagonal on the block's own K / V, the in-block indexer over the pools); the static ``serving_envelope`` (the DECODE form served
on a Rubin reading with ``sparse_core``, the EXTEND form declined with the ``sparse`` row); the execute-time form checks on an
UNCOMPILED block under ``set_sync_debug_mode("error")`` (the list form per declaration, ``seq_lens`` refused, the whole table
required, ``kv_lens`` required); the split rule.

Rubin cells (``requires_rubin``; one compiled block per cell): the decode ladder against the fp32 oracle -- the dense block
reference over the WHOLE sequence gives the post-norm / post-RoPE Q, K, V and the pre-sigmoid gate; the prefix rows fill the
pools (NaN everywhere else: a gathered unwritten row would poison O), the step's rows are the block's own write-through; the
step rows' attention is the oracle's over ``qsa_visible_mask(..., pos0=)`` (shared) or per row (per-token), then the gate and
the out projection -- ``cos(out) > 0.999``, the gated O (the workspace slot) and the LSE within ``atol 2e-2`` (the sparse
end-to-end module's bars, never widened), a dead row (``kv_lens[b] = 0``; a row whose position is negative because the
sequence is shorter than the step) EXACTLY ``out = 0`` / ``O = 0`` / ``LSE = -inf``, the pools' step rows BITWISE the slab's
post-RoPE K / V bands, the prefix rows and the NULL block untouched, two launches bitwise.  The ladder: ``S_q`` 1 .. 4 (the
shared list; B 1 / 2 / 4; ``kv_lens`` up to 20000 incl. 2052, a dead entry, a one-token sequence and a sequence shorter than
the step), ``S_q`` 8 and 64 (per-token lists), bf16 and f16, pages 16 HND / 32 unified / 64 NHD, splits 1 / 2 / 4 / 17 and
the default rule, ``fuse_gate`` (unsplit: the core's epilogue; split: the combine) and the separate gate launch, ``return_lse``
on / off, ``block_lens`` given / absent, ``fuse_norm_rope``, the indexer band's raw-key write-through, the Flash-Next 24 / 2
geometry at ``d_model`` 2560; the MTP matrix at every residue of the step-0 position on a short sequence (the ``4k + 3`` row
sees the block completed after step 0: the step-0-anchored oracle within budget, the per-row-tail reading farther); the
block's form vs the standalone adapter on the SAME Q / gate views, pools, table and list: BITWISE; ``get_workspace_size()``
honest under a split (a sentinel past the carve untouched); CUDA-graph capture of the FIRST execute on a NEW stream with no
warm-up under ``set_sync_debug_mode("error")``, the replay bitwise the eager run, and the replay over NEW lengths / tables /
lists / slots / inputs bitwise the eager run over them.
"""

import contextlib
import math
import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block import (
    KV_SOURCE_PAGED,
    KV_SOURCE_SELF,
    DecodeEnvelope,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    QsaSpec,
    index_k_raw_view,
)
from cudnn.gated_attention_block.api import _EXTEND_DECLINES, _cols, _SparseSdpa, _view

requirement_error = cutedsl_requirement_error("the gated attention block's sparse decode tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_qsa_reference import qsa_visible_mask, random_block_ids  # noqa: E402
from gated_block_reference import RefGeometry, build_rope_tables, gated_attention_block_reference  # noqa: E402

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_SM107 = (10, 7)
BF16, F16 = torch.bfloat16, torch.float16
D = 256
ROPE = 64
BS = 4  # tokens per selectable block
COS_OUT = 0.999  # the end-to-end suite's bar on `out`
ATOL = 2e-2  # the SDPA stage test's bar on O / LSE -- never widened here
SHARED_MAX_S_Q = 4  # the sparse core's shared-list cap (SparseCapabilities.index_shared_list_max_tokens)
GEOM_SMALL = dict(d_model=512, h_q=8, h_kv=2, d_head=D, rope_dim=ROPE)  # group 4, cheap projections
GEOM_FLASH_NEXT = dict(d_model=2560, h_q=24, h_kv=2, d_head=D, rope_dim=ROPE)  # Qwen3.8-Flash-Next TP 1 (group 12)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


@contextlib.contextmanager
def _no_host_sync():
    """Every declaration-time and form check stays on the host without a device sync (Rule 3) -- enforced, not read."""
    if not torch.cuda.is_available():
        yield
        return
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(prev)


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def _bits(x: torch.Tensor) -> torch.Tensor:
    """The 16-bit patterns of a bf16 / f16 tensor: a NaN-safe bitwise comparison."""
    return x.contiguous().view(torch.int16)


def _sentinel(dtype):
    return float(torch.finfo(dtype).max) / 4


def _pools(n_pages, h, page, layout, dtype, device=_DEV, fill=float("nan")):
    """Two pools as ``[P, H, page, D]`` views: ``hnd`` (compact), ``nhd`` (by strides), ``unified`` (one ``[P, H, page, 2D]`` content
    row per token, K at ``+0``, V at ``+D``).  NaN-filled: the sparse core gathers listed blocks and the open tail only, so a NaN
    reaching O names a row the kernel should never have read (the sparse module's own detector)."""
    if layout == "hnd":
        return tuple(torch.full((n_pages, h, page, D), fill, device=device, dtype=dtype) for _ in range(2))
    if layout == "nhd":
        return tuple(torch.full((n_pages, page, h, D), fill, device=device, dtype=dtype).permute(0, 2, 1, 3) for _ in range(2))
    assert layout == "unified"
    row = torch.full((n_pages, h, page, 2 * D), fill, device=device, dtype=dtype)
    return row[..., :D], row[..., D:]


def _geom(geom_kw, top_k=512, index_band=False, **over):
    kw = dict(geom_kw, causal_bottom_right=True)
    kw.update(over)
    return GatedAttentionBlockGeometry(**kw, qsa=QsaSpec(top_k=top_k, index_band=index_band)), RefGeometry(**kw)


def _declare(geom_kw, batch, s_q, *, page=16, layout="hnd", n_pages=8, dtype=BF16, top_k=512, index_band=False, table=None, geom_over=None, **blk_kw):
    """A declared (not compiled) sparse DECODE block on EMPTY tensors: the declaration cells (any device)."""
    geom, _ = _geom(geom_kw, top_k=top_k, index_band=index_band, **(geom_over or {}))
    mk = lambda *shape: torch.empty(*shape, dtype=dtype, device=_DEV)  # noqa: E731
    kc, vc = _pools(n_pages, geom.h_kv, page, layout, dtype)
    bt = torch.zeros(batch, n_pages, dtype=torch.int32, device=_DEV) if table is None else table
    kw = dict(paged_kv_page_size=page, kv_source=KV_SOURCE_PAGED, sample_k_cache=kc, sample_v_cache=vc, sample_block_table=bt)
    kw.update(blk_kw)
    blk = GatedAttentionBlockFwd(
        mk(batch, s_q, geom.d_model),
        mk(geom.n_qkvg, geom.d_model),
        torch.ones(D, dtype=dtype, device=_DEV),
        torch.ones(D, dtype=dtype, device=_DEV),
        mk(batch, s_q, ROPE),
        mk(batch, s_q, ROPE),
        mk(geom.d_model, geom.h_q * D),
        mk(batch, s_q, geom.d_model),
        geom,
        **kw,
    )
    return blk, dict(k_cache=kc, v_cache=vc, block_table=bt, geom=geom)


def _row(name):
    return next(r for r in _EXTEND_DECLINES if r[0] == name)


# ---------------------------------------------------------------------------
# Host: the declaration facts, the typed declines, the envelope, the execute form checks, the split rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s_q", [1, 4, 5, 64])
@pytest.mark.parametrize("fuse_gate", [False, True], ids=["gate_launch", "fuse_gate"])
def test_declaration_facts_of_the_sparse_decode_mode(s_q, fuse_gate):
    """A dense ``[B, S_q]`` QsaSpec declaration over the pools CONSTRUCTS (no host sync) with the SPARSE stage in its paged form:
    the shared list at ``S_q <= 4`` and per-token lists past it, NO dense decode route (the sparse core has no row cap: 64 rows
    construct), the gate where it was declared (the sparse core fuses it unsplit and routes it into the combine under a split --
    stage (5) is not built either way), the per-sequence KV length forced on, the write-through right before the sparse stage, the
    split the rule picks (or the pinned one), ``max_kv_len`` the table's reach by default."""
    from cudnn.gated_attention_block.api import sparse_decode_split_choice

    with _no_host_sync():
        blk, aux = _declare(GEOM_SMALL, 4, s_q, n_pages=6, fuse_gate=fuse_gate)
    st = blk._sdpa
    assert isinstance(st, _SparseSdpa) and st.paged and st.kv_source == KV_SOURCE_PAGED
    assert st.list_per_sequence is (s_q <= SHARED_MAX_S_Q) and blk.qsa_shared_list is (s_q <= SHARED_MAX_S_Q) and blk.sparse_decode
    assert blk.decode_route is None and blk.gate_fused is fuse_gate and (blk._gate is None) is fuse_gate and st.fuse_gate is fuse_gate
    assert blk.seq_lens_present and st.seq_lens_present
    assert blk._stages.index(blk._cache_write) == blk._stages.index(blk._sdpa) - 1
    assert blk.max_kv_len == 6 * 16
    sm = torch.cuda.get_device_properties(torch.device(_DEV)).multi_processor_count if _DEV == "cuda" else 0
    assert blk.decode_split_kv == sparse_decode_split_choice(batch=4, h_kv=2, s_q=s_q, top_k=512, sm_count=sm) == st.split_kv
    assert 1 <= st.split_kv <= 17
    with _no_host_sync():
        pinned, _ = _declare(GEOM_SMALL, 4, s_q, n_pages=6, fuse_gate=fuse_gate, decode_split_kv=3, max_kv_len=70, return_lse=True)
    assert pinned.decode_split_kv == 3 == pinned._sdpa.split_kv and pinned.max_kv_len == 70 and pinned.return_lse and pinned._sdpa.want_lse
    # the serving declaration's capture-safe projections (the forced-tile JIT route) exist where 256 divides N: the indexer band (+640
    # columns) leaves it, so a band geometry over the pools keeps the driver's route (eager steps; a capture is the driver's typed refusal)
    assert blk._proj.capture_safe and blk._out_proj.capture_safe
    with _no_host_sync():
        band, _ = _declare(GEOM_SMALL, 4, s_q, n_pages=6, index_band=True)
    assert band.geom.n_qkvg % 256 != 0 and not getattr(band._proj, "capture_safe", False) and not getattr(band._out_proj, "capture_safe", False)


def test_the_sparse_split_rule_fills_the_machine_and_stays_inside_the_item():
    """The split the sparse decode mode lowers with when ``decode_split_kv=None``: enough chunks to put one work item per SM
    (``ceil(SMs / (B x H_kv x S_q))``), never more than the item's tile count ``ceil((top_k + 1) / 32)`` nor than the measured cap of 8
    (the 4 x 32K x 1 split sweep on a 212-SM part: the combine's cost grows with the partial count), 1 for a full launch and
    for a non-CUDA reading (0 SMs) -- a performance choice only: the same function under every value."""
    from cudnn.gated_attention_block.api import sparse_decode_split_choice as pick

    assert pick(batch=4, h_kv=2, s_q=1, top_k=512, sm_count=212) == 8  # 212 / 8 = 27 -> the measured cap of 8 (below the 17 tiles of a 512-list item)
    assert pick(batch=1, h_kv=2, s_q=1, top_k=512, sm_count=212) == 8  # 2 items: the cap again
    assert pick(batch=4, h_kv=2, s_q=4, top_k=512, sm_count=212) == 7  # ceil(212 / 32)
    assert pick(batch=32, h_kv=2, s_q=4, top_k=512, sm_count=212) == 1  # 256 items: a full launch never splits
    assert pick(batch=2, h_kv=2, s_q=1, top_k=64, sm_count=212) == 3  # ceil(65 / 32) = 3 tiles at top_k 64
    assert pick(batch=1, h_kv=2, s_q=1, top_k=512, sm_count=0) == 1  # no CUDA device: declined at check_support anyway
    assert pick(batch=1, h_kv=1, s_q=1, top_k=4, sm_count=212) == 1  # a 4-list item is one tile


def test_typed_declines_of_the_sparse_decode_mode():
    """Every request the mode cannot serve, declined at DECLARATION with the feature named: the PACKED form over the pools (the
    serving table's ``sparse`` row -- the core serves THD and paged pools one at a time), a block-table sample that is not
    contiguous (the sparse core binds the whole ``(B, max_pages)`` table), a split past the item's tile count, ``max_kv_len``
    past the table's reach, the bottom-right diagonal on the block's OWN K / V (the decode mode is the paged one), the in-block
    indexer over the pools, the prepare fusion over the pools."""
    name, exc, _pred, msg = _row("sparse")
    mk = lambda *shape: torch.empty(*shape, dtype=BF16, device=_DEV)  # noqa: E731
    geom, _ = _geom(GEOM_SMALL)
    kc, vc = _pools(4, 2, 16, "hnd", BF16)
    with pytest.raises(exc, match="thd=True") as e:
        GatedAttentionBlockFwd(
            mk(32, geom.d_model),
            mk(geom.n_qkvg, geom.d_model),
            mk(D),
            mk(D),
            mk(32, ROPE),
            mk(32, ROPE),
            mk(geom.d_model, geom.h_q * D),
            mk(32, geom.d_model),
            geom,
            thd=True,
            num_sequences=2,
            max_seq_len=32,
            token_capacity=32,
            paged_kv_page_size=16,
            kv_source=KV_SOURCE_PAGED,
            sample_k_cache=kc,
            sample_v_cache=vc,
            sample_block_table=torch.zeros(2, 4, dtype=torch.int32, device=_DEV),
        )
    assert str(e.value) == msg and "one at a time" in msg
    wide = torch.zeros(2, 9, dtype=torch.int32, device=_DEV)[:, :4]  # a row slice of a wider table: not contiguous
    with pytest.raises(ValueError, match="CONTIGUOUS under QsaSpec"):
        _declare(GEOM_SMALL, 2, 1, n_pages=4, table=wide)
    with pytest.raises(NotImplementedError, match=r"ceil\(\(top_k \+ 1\) / 32\)"):
        _declare(GEOM_SMALL, 2, 1, n_pages=4, decode_split_kv=18)
    with pytest.raises(NotImplementedError, match=r"ceil\(\(top_k \+ 1\) / 32\)"):
        _declare(GEOM_SMALL, 2, 1, n_pages=4, top_k=64, decode_split_kv=4)
    with pytest.raises(ValueError, match="exceeds the block table's reach"):
        _declare(GEOM_SMALL, 2, 1, n_pages=4, max_kv_len=65)
    with pytest.raises(ValueError, match="decode_split_kv must be"):
        _declare(GEOM_SMALL, 2, 1, n_pages=4, decode_split_kv=0)
    # the block's own K / V with the bottom-right diagonal: the decode mode is the paged one -- the decline names it
    with pytest.raises(NotImplementedError, match="kv_source='paged'") as e:
        GatedAttentionBlockFwd(
            mk(1, 8, geom.d_model),
            mk(geom.n_qkvg, geom.d_model),
            mk(D),
            mk(D),
            mk(1, 8, ROPE),
            mk(1, 8, ROPE),
            mk(geom.d_model, geom.h_q * D),
            mk(1, 8, geom.d_model),
            geom,
        )
    assert "causal_bottom_right=True" in str(e.value) and "sparse adapter's record declines" not in str(e.value)
    # the in-block indexer over the pools: the step form of the indexer is a follow-up (on a tree without the indexer at all, its own earlier decline speaks)
    with pytest.raises(NotImplementedError, match="indexer"):
        _declare_indexer()
    # the prepare-fusion knob exists on trees that carry the fused prepare kernel: there its own row declines the paged forms (the
    # fusion is validated with the write-through alone); a tree without the knob refuses the keyword itself
    import inspect

    if "fuse_prepare" in inspect.signature(GatedAttentionBlockFwd.__init__).parameters:
        with pytest.raises(NotImplementedError, match="fuse_prepare=True with kv_source='paged'"):
            _declare(GEOM_SMALL, 2, 1, n_pages=4, fuse_prepare=True)
    else:
        with pytest.raises(TypeError, match="fuse_prepare"):
            _declare(GEOM_SMALL, 2, 1, n_pages=4, fuse_prepare=True)


def _declare_indexer():
    geom = GatedAttentionBlockGeometry(**GEOM_FLASH_NEXT, causal_bottom_right=True, qsa=QsaSpec(index_source="indexer", index_band=True))
    mk = lambda *shape: torch.empty(*shape, dtype=BF16, device=_DEV)  # noqa: E731
    kc, vc = _pools(4, 2, 16, "hnd", BF16)
    return GatedAttentionBlockFwd(
        mk(1, 1, geom.d_model),
        mk(geom.n_qkvg, geom.d_model),
        mk(D),
        mk(D),
        mk(1, 1, ROPE),
        mk(1, 1, ROPE),
        mk(geom.d_model, geom.h_q * D),
        mk(1, 1, geom.d_model),
        geom,
        paged_kv_page_size=16,
        kv_source=KV_SOURCE_PAGED,
        sample_k_cache=kc,
        sample_v_cache=vc,
        sample_block_table=torch.zeros(1, 4, dtype=torch.int32, device=_DEV),
    )


def test_the_envelope_serves_the_sparse_decode_mode_and_declines_the_packed_form(monkeypatch):
    """The static record a serving stack reads: on a Rubin reading a QsaSpec geometry with the bottom-right diagonal SERVES the
    DECODE form (``sparse_core`` names stage (4); the requires spell the shared list and the per-token form) and declines the
    EXTEND form with the table's ``sparse`` row (the very message the packed declaration raises); the top-left QsaSpec geometry
    meets the ``diagonal`` row like a dense one; on this device the verdict reads the part."""
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    geom, _ = _geom(GEOM_SMALL)
    env = GatedAttentionBlockFwd.serving_envelope(geom, BF16, _DEV, device_cc=_SM107)
    assert isinstance(env.decode, DecodeEnvelope) and env.decode.served and env.decode.sparse_core and env.decode.reason is None
    assert env.forms == ("decode",) and not env.extend.served and env.extend.reason == _row("sparse")[3]
    assert any("[B, top_k]" in r for r in env.decode.requires) and any("[B, S_q, top_k]" in r for r in env.decode.requires)
    assert env.decode.routes == () and "sparse" not in env.decode.declines and env.extend.declines["sparse"] == _row("sparse")[3]
    top_left = GatedAttentionBlockFwd.serving_envelope(GatedAttentionBlockGeometry(**GEOM_SMALL, qsa=QsaSpec()), BF16, _DEV, device_cc=_SM107)
    assert not top_left.decode.served and top_left.decode.reason == _row("diagonal")[3] and top_left.decode.sparse_core  # a geometry fact, served or not
    dense = GatedAttentionBlockFwd.serving_envelope(GatedAttentionBlockGeometry(**GEOM_SMALL, causal_bottom_right=True), BF16, _DEV, device_cc=_SM107)
    assert dense.decode.served and not dense.decode.sparse_core and dense.decode.routes
    monkeypatch.undo()
    here = GatedAttentionBlockFwd.serving_envelope(geom, BF16, _DEV)
    assert here.decode.served is (_cc() == _SM107 and here.cutedsl_ok)


def test_execute_form_checks_of_the_sparse_decode_mode():
    """On an UNCOMPILED block (the index tensors and the paged inputs are a declaration-vs-argument contract: checked first, no
    plan, no sync): the shared-list form takes ``block_ids [B, top_k]`` / ``block_lens [B]`` and refuses the per-token spelling
    by name, the per-token form the reverse; ``seq_lens`` is refused over the pools; the table is the WHOLE declared
    ``(B, max_pages)`` table; ``kv_lens`` is required; an int64 list is refused, never converted."""
    with _no_host_sync():
        blk, aux = _declare(GEOM_SMALL, 2, 2, n_pages=4)
        per_tok, aux8 = _declare(GEOM_SMALL, 2, 8, n_pages=4)
    geom = aux["geom"]
    i32 = lambda *shape: torch.zeros(*shape, dtype=torch.int32, device=_DEV)  # noqa: E731
    z = lambda *shape: torch.zeros(*shape, dtype=BF16, device=_DEV)  # noqa: E731

    def run(b, *, s_q, ids, lens=None, kv_lens=None, table=None, seq_lens=None, slot=None, pools=None):
        pools = aux if pools is None else pools
        return b.execute(
            z(2, s_q, geom.d_model),
            z(geom.n_qkvg, geom.d_model),
            torch.ones(D, dtype=BF16, device=_DEV),
            torch.ones(D, dtype=BF16, device=_DEV),
            z(2, s_q, ROPE),
            z(2, s_q, ROPE),
            z(geom.d_model, geom.h_q * D),
            z(2, s_q, geom.d_model),
            torch.empty(1 << 20, dtype=torch.uint8, device=_DEV),
            seq_lens=seq_lens,
            block_ids=ids,
            block_lens=lens,
            k_cache=pools["k_cache"],
            v_cache=pools["v_cache"],
            block_table=pools["block_table"] if table is None else table,
            kv_lens=i32(2) if kv_lens is None else kv_lens,
            slot_mapping=i32(2 * s_q) if slot is None else slot,
        )

    with _no_host_sync():
        with pytest.raises(ValueError, match=r"block_ids must be \[2, 512\]"):
            run(blk, s_q=2, ids=i32(4, 512))  # the per-token spelling on the shared-list form
        with pytest.raises(ValueError, match=r"block_ids must be \[2, 512\]"):
            run(blk, s_q=2, ids=i32(2, 2, 512))
        with pytest.raises(ValueError, match=r"block_lens must be \[2\]"):
            run(blk, s_q=2, ids=i32(2, 512), lens=i32(4))
        with pytest.raises(ValueError, match="int64"):
            run(blk, s_q=2, ids=torch.zeros(2, 512, dtype=torch.int64, device=_DEV))
        with pytest.raises(ValueError, match=r"block_ids must be \[16, 512\] or \[2, 8, 512\]"):
            run(per_tok, s_q=8, ids=i32(2, 512), pools=aux8)  # the shared spelling on the per-token form
        with pytest.raises(ValueError, match="seq_lens"):
            run(blk, s_q=2, ids=i32(2, 512), seq_lens=i32(2))
        with pytest.raises(ValueError, match="kv_lens"):
            run(blk, s_q=2, ids=i32(2, 512), kv_lens=i32(3))
        with pytest.raises(ValueError, match="block_table"):
            run(blk, s_q=2, ids=i32(2, 512), table=i32(2, 6))  # wider than the declared table: the sparse core binds exactly (B, max_pages)
        with pytest.raises(ValueError, match="block_table"):
            run(blk, s_q=2, ids=i32(2, 512), table=i32(2, 8)[:, :4])
        with pytest.raises(RuntimeError, match="compile"):
            run(blk, s_q=2, ids=i32(2, 512))  # every form check passed: the uncompiled block stops at its own gate


# ---------------------------------------------------------------------------
# Rubin: the oracle, the cells
# ---------------------------------------------------------------------------


def _rope_tables(n_pos, dtype, ref_geom):
    cos, sin = build_rope_tables(n_pos, ROPE, base=ref_geom.rope_base, batch=1, device=_DEV, dtype=dtype)
    return cos, sin  # [1, n_pos, ROPE]


class _Case:
    """One sparse DECODE step over a batch of sequences with lengths ``kv_lens`` (0 = a dead entry; a length below ``s_q`` leaves
    the rows with a negative position dead): the whole-sequence inputs per entry, the step's inputs (its last ``s_q`` positions),
    the NaN-filled pools holding the ORACLE's prefix K / V, the block table (page 0 = the NULL block of a dead entry, ``-1`` past a
    sequence's live pages), the slot mapping of the step's tokens (-1 on a dead row), the lists (one per sequence at the step-0
    position, or one per row), and the per-sequence fp32 oracle of the step's ``out`` rows, gated O and LSE."""

    def __init__(
        self,
        geom_kw,
        s_q,
        kv_lens,
        *,
        page=16,
        layout="hnd",
        dtype=BF16,
        seed=0,
        top_k=512,
        block_lens=True,
        return_lse=False,
        fuse_gate=False,
        fuse_norm_rope=False,
        split=None,
        index_band=False,
        n_pages=None,
        pos0_anchor=True,
        shuffle=True,
    ):
        self.s_q, self.kv_lens, self.page, self.layout, self.dtype, self.top_k = int(s_q), [int(x) for x in kv_lens], page, layout, dtype, top_k
        self.block_lens_given, self.return_lse, self.fuse_gate, self.index_band = bool(block_lens), bool(return_lse), bool(fuse_gate), bool(index_band)
        self.shared = self.s_q <= SHARED_MAX_S_Q
        self.geom, self.ref_geom = _geom(geom_kw, top_k=top_k, index_band=index_band)
        geom, ref_geom = self.geom, self.ref_geom
        h_q, h_kv, d_model = geom.h_q, geom.h_kv, geom.d_model
        G = h_q // h_kv
        B = len(self.kv_lens)
        g = torch.Generator(device=_DEV).manual_seed(seed)
        self.gen = g

        def randn(*shape, std=0.02):
            return (torch.randn(*shape, generator=g, device=_DEV, dtype=torch.float32) * std).to(dtype)

        n_dense = sum(geom.qkvg_block_widths[:4])
        w_qkvg = randn(geom.n_qkvg, d_model)
        self.w = dict(
            w_qkvg=w_qkvg, w_q_norm=torch.ones(D, dtype=dtype, device=_DEV), w_k_norm=torch.ones(D, dtype=dtype, device=_DEV), w_o=randn(d_model, h_q * D)
        )
        L_max = max(max(self.kv_lens), s_q)
        cos_t, sin_t = _rope_tables(L_max, dtype, ref_geom)
        scale = geom.scale
        # the step's inputs: row j of entry b is at position p_j = L - s_q + j; a negative position is a DEAD row (finite inputs, RoPE row 0)
        self.h = torch.zeros(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.cos = cos_t[0, :1].expand(s_q, ROPE).clone()[None].expand(B, s_q, ROPE).clone()
        self.sin = sin_t[0, :1].expand(s_q, ROPE).clone()[None].expand(B, s_q, ROPE).clone()
        self.ref_out = torch.zeros(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.ref_o_gated = torch.zeros(B, s_q, h_q, D, dtype=dtype, device=_DEV)
        self.ref_lse = torch.full((B, h_q, s_q), float("-inf"), dtype=torch.float32, device=_DEV)
        self.ref_lse_rowtail = self.ref_lse.clone()
        self.ref_o_gated_rowtail = self.ref_o_gated.clone()
        self.ref_rows = {}
        self.live_rows = torch.zeros(B, s_q, dtype=torch.bool, device=_DEV)
        ids_rows, lens_rows = [], []
        for b, L in enumerate(self.kv_lens):
            pos = torch.arange(s_q, device=_DEV) + (L - s_q)
            live = pos >= 0
            self.live_rows[b] = live
            pos0 = torch.tensor([L - s_q], device=_DEV)
            if self.shared:
                ids_b, lens_b = random_block_ids(pos0, top_k, BS, generator=g, kv_lens=torch.tensor([L], device=_DEV), shuffle=shuffle)
            else:
                ids_b, lens_b = random_block_ids(pos, top_k, BS, generator=g, kv_lens=torch.tensor([L], device=_DEV).expand(s_q), shuffle=shuffle)
            ids_rows.append(ids_b)
            lens_rows.append(lens_b)
            if L == 0:
                continue  # a dead entry: no sequence, no reference, block 0, slots -1
            h_full = randn(1, L, d_model, std=1.0)
            ref = gated_attention_block_reference(
                h_full, w_qkvg[:n_dense], self.w["w_q_norm"], self.w["w_k_norm"], cos_t[:, :L], sin_t[:, :L], self.w["w_o"], ref_geom
            )
            self.ref_rows[b] = (ref.k[0], ref.v[0])  # [L, h_kv, D]: the prefix rows fill the pools
            p_live = pos[live]
            self.h[b, live] = h_full[0, p_live]
            self.cos[b, live] = cos_t[0, p_live]
            self.sin[b, live] = sin_t[0, p_live]
            # the step rows' attention over the sequence's keys through the visible mask (the shared list anchored at pos_0, or per row)
            rows = ids_b.expand(s_q, top_k) if self.shared else ids_b
            lens_arg = (lens_b.expand(s_q) if self.shared else lens_b) if block_lens else None
            q_rows = ref.q[0][p_live].float()  # [n_live, h_q, D]
            gate_rows = ref.gate[0][p_live].float()
            for anchored, (o_dst, lse_dst) in ((True, (self.ref_o_gated, self.ref_lse)), (False, (self.ref_o_gated_rowtail, self.ref_lse_rowtail))):
                if not anchored and not self.shared:
                    continue
                allowed = qsa_visible_mask(rows, lens_arg, pos, L, L, BS, top_k=top_k, pos0=(L - s_q) if (anchored and self.shared) else None)[live]
                o = torch.zeros(int(live.sum()), h_q, D, dtype=torch.float32, device=_DEV)
                lse = torch.full((h_q, int(live.sum())), float("-inf"), dtype=torch.float32, device=_DEV)
                for hh in range(h_q):
                    kk, vv = ref.k[0][:, hh // G].float(), ref.v[0][:, hh // G].float()
                    sc = (q_rows[:, hh] @ kk.t()) * scale
                    sc = sc.masked_fill(~allowed, float("-inf"))
                    rmax = sc.amax(dim=-1)
                    dead = torch.isinf(rmax) & (rmax < 0)
                    safe = torch.where(dead, torch.zeros_like(rmax), rmax)
                    p = torch.where(allowed, torch.exp(sc - safe[:, None]), torch.zeros_like(sc))
                    den = p.sum(dim=-1)
                    den_safe = torch.where(dead, torch.ones_like(den), den)
                    oc = (p @ vv) / den_safe[:, None]
                    o[:, hh] = torch.where(dead[:, None], torch.zeros_like(oc), oc)
                    lse[hh] = torch.where(dead, torch.full_like(den, float("-inf")), safe + torch.log(den))
                o16 = o.to(dtype)  # the oracle's single rounding of O, then the gate in fp32, then one rounding
                og = (o16.float() * torch.sigmoid(gate_rows)).to(dtype)
                o_dst[b, live] = og
                lse_dst[b, :, live] = lse
                if anchored:
                    self.ref_out[b, live] = (og.reshape(-1, h_q * D).float() @ self.w["w_o"].float().t()).to(dtype)
        self.block_ids = torch.stack([x.reshape(top_k) if self.shared else x.reshape(s_q, top_k) for x in ids_rows]).contiguous()
        self.block_lens = torch.stack([x.reshape(()) if self.shared else x.reshape(s_q) for x in lens_rows]).contiguous()
        assert self.block_ids.dtype == torch.int32 and self.block_lens.dtype == torch.int32
        # pages: block 0 reserved (NULL); each live entry gets ceil(L / P) pages from a permutation of 1 .. n_pages - 1; -1 past them
        need = sum(-(-L // page) for L in self.kv_lens)
        self.n_pages = n_pages if n_pages is not None else need + 1 + 2
        assert self.n_pages >= need + 1, (self.n_pages, need)
        perm = (torch.randperm(self.n_pages - 1, generator=g, device=_DEV) + 1).tolist()
        self.max_pages = max(-(-L_max // page), 1)
        self.block_table = torch.full((B, self.max_pages), -1, dtype=torch.int32, device=_DEV)
        it = iter(perm)
        slots = torch.full((B * s_q,), -1, dtype=torch.int32, device=_DEV)
        self.k_cache, self.v_cache = _pools(self.n_pages, h_kv, page, layout, dtype)
        self.index_cache = torch.full((self.n_pages, page, geom.qsa.index_head_dim), float("nan"), dtype=dtype, device=_DEV) if index_band else None
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                self.block_table[b] = 0  # the dead entry: the NULL block, slots -1
                continue
            pages = [next(it) for _ in range(-(-L // page))]
            self.block_table[b, : len(pages)] = torch.tensor(pages, dtype=torch.int32, device=_DEV)
            k_rows, v_rows = self.ref_rows[b]
            for p in range(max(L - s_q, 0)):  # the PREFIX: the oracle's rows into the pools
                pg, off = pages[p // page], p % page
                self.k_cache[pg, :, off] = k_rows[p]
                self.v_cache[pg, :, off] = v_rows[p]
            for j in range(s_q):  # the step's live tokens: written through by the block
                p = L - s_q + j
                if p >= 0:
                    slots[b * s_q + j] = pages[p // page] * page + p % page
        self.slot_mapping = slots
        self.kv_lens_t = torch.tensor(self.kv_lens, dtype=torch.int32, device=_DEV)
        self.prefix_bits = (_bits(self.k_cache).clone(), _bits(self.v_cache).clone())
        kw = dict(
            paged_kv_page_size=page,
            kv_source=KV_SOURCE_PAGED,
            sample_k_cache=self.k_cache,
            sample_v_cache=self.v_cache,
            sample_block_table=self.block_table,
            fuse_gate=fuse_gate,
            fuse_norm_rope=fuse_norm_rope,
            return_lse=return_lse,
            decode_split_kv=split,
        )
        self.blk_kw = kw
        self.out = torch.empty(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.lse = torch.empty(B, h_q, s_q, dtype=torch.float32, device=_DEV) if return_lse else None
        self.blk = GatedAttentionBlockFwd(
            self.h, self.w["w_qkvg"], self.w["w_q_norm"], self.w["w_k_norm"], self.cos, self.sin, self.w["w_o"], self.out, self.geom, **kw
        )
        assert isinstance(self.blk._sdpa, _SparseSdpa) and self.blk.sparse_decode and self.blk.qsa_shared_list is self.shared
        self.ws = None

    def compile(self):
        assert self.blk.check_support()
        self.blk.compile()
        self.ws = torch.full((self.blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device=_DEV)
        return self

    def execute(
        self, *, out=None, ws=None, lse=None, h=None, cos=None, sin=None, kv_lens=None, block_table=None, slot_mapping=None, ids=None, lens=None, stream=None
    ):
        out = self.out if out is None else out
        self.blk.execute(
            self.h if h is None else h,
            self.w["w_qkvg"],
            self.w["w_q_norm"],
            self.w["w_k_norm"],
            self.cos if cos is None else cos,
            self.sin if sin is None else sin,
            self.w["w_o"],
            out,
            self.ws if ws is None else ws,
            lse=(self.lse if lse is None else lse) if self.return_lse else None,
            block_ids=self.block_ids if ids is None else ids,
            block_lens=(self.block_lens if lens is None else lens) if self.block_lens_given else None,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            block_table=self.block_table if block_table is None else block_table,
            kv_lens=self.kv_lens_t if kv_lens is None else kv_lens,
            slot_mapping=self.slot_mapping if slot_mapping is None else slot_mapping,
            index_k_raw=self.index_cache,
            current_stream=stream,
        )
        return out

    def run(self):
        if self.ws is None:
            self.compile()
        self.out.fill_(_sentinel(self.dtype))
        if self.lse is not None:
            self.lse.fill_(_sentinel(self.dtype))
        self.execute()
        torch.cuda.synchronize()
        return self.out

    def slab_views(self, ws=None):
        """The post-norm-RoPE Q band, the GATE band, the K / V bands of the slab the last execute left in the workspace, as the
        sparse stage's ``[B, S_q, H, D]`` views at the slab's token stride, and the O slot."""
        ws = self.ws if ws is None else ws
        g, lay = self.geom, self.blk._ws
        B, S = self.blk.batch, self.blk.seq_len
        t = B * S
        proj = _view(ws, lay.proj, (t, g.n_qkvg), self.dtype)
        o_q, o_g, o_k, o_v = g.qkvg_offsets[:4]
        q = _cols(proj, o_q, g.h_q, g.d_head).view(B, S, g.h_q, g.d_head)
        gate = _cols(proj, o_g, g.h_q, g.d_head).view(B, S, g.h_q, g.d_head)
        k = _cols(proj, o_k, g.h_kv, g.d_head).view(B, S, g.h_kv, g.d_head)
        v = _cols(proj, o_v, g.h_kv, g.d_head).view(B, S, g.h_kv, g.d_head)
        o = _view(ws, lay.o, (t, g.h_q, g.d_head), self.dtype).view(B, S, g.h_q, g.d_head)
        return q, gate, k, v, o, proj

    def label(self):
        return f"[{self.geom.h_q}/{self.geom.h_kv} S_q={self.s_q} {'shared' if self.shared else 'per-token'} p{self.page} {self.layout} {self.dtype} split={self.blk.decode_split_kv} gate={'fused' if self.fuse_gate else 'launch'} lse={self.return_lse} lens={self.kv_lens}]"

    def check(self, out=None, ws=None, lse=None, *, what="out"):
        """The standard assertions: finite, no sentinel, dead rows exact, within the budget vs the oracle (out, the gated O slot, the
        LSE), the pools' step rows bitwise the slab's K / V bands, the prefix rows and the NULL block untouched."""
        out = self.out if out is None else out
        ws = self.ws if ws is None else ws
        lse = self.lse if lse is None else lse
        sent = _sentinel(self.dtype)
        live = self.live_rows
        dead = ~live
        q, gate, k, v, o, proj = self.slab_views(ws)
        assert torch.isfinite(out.float()).all(), f"{what}: non-finite out"
        assert not (out.float() == sent).any(), f"{what}: a sentinel survived in out"
        assert torch.isfinite(o.float()).all(), f"{what}: non-finite gated O (a row the sparse core never wrote keeps the workspace fill)"
        if dead.any():
            assert (out[dead] == 0).all() and (o[dead] == 0).all(), f"{what}: a dead row's out and O must be exactly 0"
        c = _cos(out[live], self.ref_out[live])
        d_out = float((out[live].float() - self.ref_out[live].float()).abs().max())
        d_o = float((o[live].float() - self.ref_o_gated[live].float()).abs().max())
        d_lse = 0.0
        if self.return_lse:
            assert lse is not None and not (lse == sent).any(), f"{what}: a sentinel survived in LSE"
            dead_bhs = dead[:, None, :].expand_as(lse)
            if dead.any():
                assert torch.isinf(lse[dead_bhs]).all() and (lse[dead_bhs] < 0).all(), f"{what}: a dead row's LSE must be -inf exactly"
            d_lse = float((lse[~dead_bhs] - self.ref_lse[~dead_bhs]).abs().max()) if live.any() else 0.0
            assert d_lse <= ATOL, f"{what}: LSE off the oracle by {d_lse} (budget {ATOL})"
        assert c > COS_OUT, f"{what}: cos(out) {c} <= {COS_OUT}"
        assert d_o <= ATOL, f"{what}: gated O off the oracle by {d_o} (budget {ATOL})"
        # the write-through: the step's live rows are the slab's post-RoPE K / V rows, bitwise; nothing else moved
        slot = self.slot_mapping
        lv = slot >= 0
        s_ = slot[lv].long()
        kf, vf = k.reshape(-1, self.geom.h_kv, D), v.reshape(-1, self.geom.h_kv, D)
        assert torch.equal(_bits(self.k_cache[s_ // self.page, :, s_ % self.page, :]), _bits(kf[lv])), f"{what}: the K pool rows are not the slab's rows"
        assert torch.equal(_bits(self.v_cache[s_ // self.page, :, s_ % self.page, :]), _bits(vf[lv])), f"{what}: the V pool rows are not the slab's rows"
        for pool, before in zip((self.k_cache, self.v_cache), self.prefix_bits):
            now = _bits(pool)
            touched = (now != before).any(dim=-1).any(dim=1)  # [n_pages, page] rows whose bits moved
            rows_moved = touched.nonzero()
            written = set((int(p) for p in (s_ // self.page * self.page + s_ % self.page).tolist()))
            assert all(int(pg) * self.page + int(off) in written for pg, off in rows_moved), f"{what}: a pool row outside the step's slots moved"
            assert not touched[0].any(), f"{what}: the NULL block moved"
        if self.index_cache is not None:
            raw = index_k_raw_view(proj, self.geom, self.blk.batch, self.blk.seq_len)  # [B, S, 1, index_head_dim]
            assert torch.equal(_bits(self.index_cache[s_ // self.page, s_ % self.page, :]), _bits(raw.reshape(-1, raw.shape[-1])[lv]))
            assert (~torch.isnan(self.index_cache.float())).all(-1).sum().item() == int(lv.sum())
        print(f"\n{self.label()} {what}: cos(out) {c:.6f} max|d out| {d_out:.4e} max|d O_gated| {d_o:.4e} max|d LSE| {d_lse:.2e} dead rows {int(dead.sum())}")
        return dict(cos=c, d_out=d_out, d_o=d_o, d_lse=d_lse, dead=int(dead.sum()))


def _two_launches(case: _Case, what: str):
    a = case.run().clone()
    o_a = case.slab_views()[4].clone()
    lse_a = None if case.lse is None else case.lse.clone()
    b = case.run().clone()
    o_b = case.slab_views()[4].clone()
    assert torch.equal(_bits(a), _bits(b)) and torch.equal(_bits(o_a), _bits(o_b)), f"{what}: two launches are not bitwise"
    if lse_a is not None:
        assert torch.equal(lse_a, case.lse), f"{what}: two launches are not bitwise on LSE"
    return case.check(what=what)


# (id, geometry, S_q, kv_lens, dtype, page, layout, split, fuse_gate, return_lse, block_lens, fuse_norm_rope, index_band, seed)
_LADDER = [
    pytest.param(GEOM_SMALL, 1, (2052, 8192, 0, 4097), BF16, 16, "hnd", None, False, True, True, False, False, 1, id="Sq1-B4-shared-p16-hnd-default-split-lse"),
    pytest.param(
        GEOM_SMALL, 2, (300, 0, 1000, 77), BF16, 64, "nhd", 1, True, False, True, True, False, 2, id="Sq2-B4-shared-p64-nhd-unsplit-fused-gate-fused-norm-rope"
    ),
    pytest.param(
        GEOM_SMALL, 3, (8192, 2052), F16, 32, "unified", 4, True, True, False, False, False, 3, id="Sq3-B2-shared-p32-unified-f16-split4-no-block-lens"
    ),
    pytest.param(
        GEOM_SMALL, 4, (8191, 8190, 8189, 8188), BF16, 16, "hnd", 17, True, True, True, False, False, 4, id="Sq4-B4-shared-every-residue-split17-fused-gate-lse"
    ),
    pytest.param(GEOM_FLASH_NEXT, 4, (4096,), BF16, 16, "hnd", None, True, False, True, True, False, 5, id="Sq4-B1-flash-next-24-2-shared-default-split-fused"),
    pytest.param(
        GEOM_SMALL, 4, (2, 4, 0, 20000), BF16, 16, "hnd", 2, False, True, True, False, False, 6, id="Sq4-B4-shared-shorter-than-the-step-dead-20000-split2"
    ),
    pytest.param(GEOM_SMALL, 1, (1024, 60), BF16, 16, "hnd", 1, False, True, True, False, True, 7, id="Sq1-B2-shared-index-band-raw-key-write-through"),
    pytest.param(GEOM_SMALL, 8, (300, 100), BF16, 16, "hnd", 1, False, True, True, False, False, 8, id="Sq8-B2-per-token-p16-unsplit-lse"),
    pytest.param(GEOM_SMALL, 64, (2500, 64), BF16, 64, "nhd", 2, True, False, True, False, False, 9, id="Sq64-B2-per-token-p64-nhd-split2-fused-gate"),
]


@requires_rubin
@pytest.mark.parametrize("geom_kw, s_q, kv_lens, dtype, page, layout, split, fuse_gate, return_lse, block_lens, fuse_norm_rope, index_band, seed", _LADDER)
def test_sparse_decode_matches_the_oracle(
    geom_kw, s_q, kv_lens, dtype, page, layout, split, fuse_gate, return_lse, block_lens, fuse_norm_rope, index_band, seed
):
    """The decode ladder (module docstring): every cell within the budget vs the step-0-anchored oracle (or the per-row oracle on
    the per-token form), dead rows exact, the write-through rows bitwise the slab, two launches bitwise."""
    case = _Case(
        geom_kw,
        s_q,
        kv_lens,
        page=page,
        layout=layout,
        dtype=dtype,
        seed=seed,
        split=split,
        fuse_gate=fuse_gate,
        return_lse=return_lse,
        block_lens=block_lens,
        fuse_norm_rope=fuse_norm_rope,
        index_band=index_band,
    )
    case.compile()
    if split is not None:
        assert case.blk.decode_split_kv == split
    _two_launches(case, "ladder")


@requires_rubin
@pytest.mark.parametrize("r", [0, 1, 2, 3], ids=lambda r: f"pos0-mod4-{r}")
@pytest.mark.parametrize("s_q", [2, 4], ids=lambda s: f"Sq{s}")
def test_mtp_rows_share_the_step0_list_and_see_the_block_completed_after_step0(s_q, r):
    """The MTP (speculative) rows of a SHORT sequence (every one of its ~60 keys matters) at every residue of the step-0 position:
    the block's rows are within the budget of the STEP-0-ANCHORED oracle (the contract page's definition: the listed blocks plus
    every token from the step-0 tail start to the row's own position), and the per-row-tail reading (vLLM's expand form, the
    oracle at ``pos0=None``), which hides the block completed between step 0 and a row at ``4k + 3``, is farther from the block
    wherever such a row exists (``r < 3`` at ``S_q = 4``; ``r = 2`` at ``S_q = 2``) and coincides otherwise."""
    L = 56 + r + s_q  # pos_0 = L - s_q = 56 + r -> pos_0 mod 4 = r
    case = _Case(GEOM_SMALL, s_q, (L, L + 4), page=16, seed=10 + r, split=1, fuse_gate=True, return_lse=True)
    case.compile()
    m = _two_launches(case, f"mtp r={r}")
    o = case.slab_views()[4]
    live = case.live_rows
    d_pos0 = float((o[live].float() - case.ref_o_gated[live].float()).abs().max())
    d_rowtail = float((o[live].float() - case.ref_o_gated_rowtail[live].float()).abs().max())
    lse_rowtail = float((case.lse - case.ref_lse_rowtail).abs().max())
    hidden_row_exists = any(((L - s_q + j) % 4 == 3) and j > 0 and ((L - s_q + 1) // 4) < ((L - s_q + j + 1) // 4) for j in range(s_q))
    print(
        f"\nmtp S_q={s_q} pos0 mod 4 = {r}: vs the pos0 oracle max|dO| {d_pos0:.3e}; vs the per-row-tail reading max|dO| {d_rowtail:.3e} max|dLSE| {lse_rowtail:.3e}; hidden row {hidden_row_exists}"
    )
    assert d_pos0 <= ATOL
    if hidden_row_exists:
        assert d_rowtail > d_pos0 and lse_rowtail > m["d_lse"], "the per-row-tail reading should be the farther one where a block completed after step 0"
    else:
        assert torch.equal(case.ref_o_gated_rowtail, case.ref_o_gated) and torch.equal(case.ref_lse_rowtail, case.ref_lse)


@requires_rubin
def test_the_block_form_equals_the_standalone_adapter_bitwise_on_the_same_views():
    """The block adds nothing of its own to the sparse core: the standalone adapter on the block's OWN post-norm / post-RoPE Q
    view and GATE view (the slab bands at the slab's token stride), the same pools, table, lengths, list and split writes the
    gated O (and the LSE) BITWISE the block's O slot -- unsplit (the core's epilogue gate) and split (the gate in the combine)."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    for split in (1, 4):
        case = _Case(GEOM_SMALL, 2, (700, 0, 2052, 130), page=16, seed=20 + split, split=split, fuse_gate=True, return_lse=True)
        case.compile()
        case.run()
        case.check(what=f"adapter twin split {split}")
        q, gate, _k, _v, o, _proj = case.slab_views()
        B, S = case.blk.batch, case.blk.seq_len
        o2 = torch.full_like(o, _sentinel(case.dtype))
        lse2 = torch.full_like(case.lse, _sentinel(case.dtype))
        a = SparseGqaFwdDslSm107(
            q=q,
            k=case.k_cache,
            v=case.v_cache,
            o=o2,
            lse=lse2,
            block_ids=case.block_ids,
            block_lens=case.block_lens,
            seq_kv_lens=case.kv_lens_t,
            top_k=case.top_k,
            scale=case.geom.scale,
            paged_kv=True,
            page_size=case.page,
            block_table=case.block_table,
            list_per_sequence=True,
            bottom_right=True,
            split_kv=split,
            epilogue_gate=gate,
        )
        assert a.check_support()
        a.compile()
        ws = torch.empty(max(a.scratch_workspace_bytes(), 16), dtype=torch.uint8, device=_DEV)
        a.execute(workspace=ws)
        torch.cuda.synchronize()
        assert torch.equal(_bits(o2), _bits(o)) and torch.equal(
            lse2, case.lse
        ), f"split {split}: the block's sparse stage differs from the standalone adapter on the same views"


@requires_rubin
def test_get_workspace_size_is_honest_and_holds_the_split_partials():
    """The split partials (the sparse adapter's scratch) sit inside ``get_workspace_size()`` (the engine region behind the carve);
    nothing past the size is written: a sentinel tail survives an execute; the dense sparse block's workspace is unchanged by the
    paged read (no scratch when unsplit)."""
    case = _Case(GEOM_SMALL, 1, (4096, 512, 0), page=16, seed=31, split=8, fuse_gate=True)
    case.compile()
    blk = case.blk
    lay = blk._layout()
    scratch = blk._sdpa.scratch_workspace_bytes()
    assert scratch > 0 and blk.get_workspace_size() >= lay.total_bytes + scratch
    need = blk.get_workspace_size()
    ws = torch.empty(need + 65536, dtype=torch.uint8, device=_DEV)
    ws[need:].fill_(0xA5)
    ws[:need].fill_(0x7F)
    out = torch.empty_like(case.out)
    case.execute(out=out, ws=ws)
    torch.cuda.synchronize()
    assert (ws[need:] == 0xA5).all(), "bytes past get_workspace_size() were written"
    case.check(out, ws, what="sentinel-tailed workspace")
    unsplit = _Case(GEOM_SMALL, 1, (4096, 512, 0), page=16, seed=31, split=1, fuse_gate=True).compile()
    assert unsplit.blk._sdpa.scratch_workspace_bytes() == 0 and unsplit.blk.get_workspace_size() < need
    unsplit.run()
    unsplit.check(what="unsplit twin")


def _written(pool):
    return int((~torch.isnan(pool.float())).all(-1).sum())


@requires_rubin
def test_capture_of_the_first_execute_replays_bitwise_and_takes_new_lengths_tables_lists_slots_and_inputs():
    """The serving stack's defaults: the FIRST execute of a fresh declaration captured on a NEW stream with no eager warm-up under
    ``set_sync_debug_mode("error")`` (no host sync in the path: lengths, tables, lists and slots are device data); the capture
    launches nothing, the replay is BITWISE a fresh eager run; new ``kv_lens`` / ``block_table`` / ``block_ids`` / ``block_lens`` /
    ``slot_mapping`` / ``h`` / RoPE rows through the captured tensors replay bitwise the eager run over them (a split plan: the
    partials in the captured workspace)."""
    case = _Case(GEOM_SMALL, 2, (300, 0, 1000, 77), page=16, seed=41, split=4, fuse_gate=True, return_lse=True)  # fresh: never executed
    case.blk.check_support()
    case.blk.compile()
    case.ws = torch.full((case.blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device=_DEV)
    case.out.fill_(float("nan"))
    case.lse.fill_(float("nan"))
    k_before = _written(case.k_cache)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                case.execute()  # the FIRST execute of this declaration, captured
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(case.out).all() and _written(case.k_cache) == k_before, "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        replayed, replayed_lse = case.out.clone(), case.lse.clone()
        case.check(replayed, lse=replayed_lse, what="replay")
        case.out.fill_(float("nan"))
        case.lse.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(_bits(case.out), _bits(replayed)) and torch.equal(
            case.lse, replayed_lse
        ), "the replay of the captured first execute differs from the eager run"
        # a NEW step through the captured tensors: new lengths (one entry a token longer: its slot and its list move), a re-laid-out
        # table, new lists / counts / slots / h / RoPE rows -- the oracle of the new step through a sibling case over the same shapes
        sib = _Case(GEOM_SMALL, 2, (301, 0, 1000, 77), page=16, seed=43, split=4, fuse_gate=True, return_lse=True, n_pages=case.n_pages)
        assert sib.block_table.shape == case.block_table.shape and sib.k_cache.shape == case.k_cache.shape
        case.k_cache.copy_(sib.k_cache)
        case.v_cache.copy_(sib.v_cache)
        case.block_table.copy_(sib.block_table)
        case.kv_lens_t.copy_(sib.kv_lens_t)
        case.slot_mapping.copy_(sib.slot_mapping)
        case.block_ids.copy_(sib.block_ids)
        case.block_lens.copy_(sib.block_lens)
        case.h.copy_(sib.h)
        case.cos.copy_(sib.cos)
        case.sin.copy_(sib.sin)
        case.w["w_qkvg"].copy_(sib.w["w_qkvg"])
        case.w["w_o"].copy_(sib.w["w_o"])
        case.prefix_bits = (_bits(case.k_cache).clone(), _bits(case.v_cache).clone())
        case.ref_out, case.ref_o_gated, case.ref_lse, case.live_rows, case.kv_lens = sib.ref_out, sib.ref_o_gated, sib.ref_lse, sib.live_rows, sib.kv_lens
        case.out.fill_(float("nan"))
        case.lse.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed2, replayed2_lse = case.out.clone(), case.lse.clone()
        case.check(replayed2, lse=replayed2_lse, what="replay over new lengths / table / lists / slots / inputs")
        case.out.fill_(float("nan"))
        case.lse.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(_bits(case.out), _bits(replayed2)) and torch.equal(
            case.lse, replayed2_lse
        ), "the replay over new lengths / tables / lists / slots differs from the eager run over them"
    finally:
        graph.reset()
