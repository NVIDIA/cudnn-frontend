# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The in-block indexer's EXTEND step over the pools (``QsaSpec(index_source="indexer")`` with ``kv_source="paged"`` on a dense ``[B, S_q = T]``
declaration whose ``index_select_form`` resolves to ``"extend"``; 2026-10-29): a T-row CHUNK of every sequence behind a cached prefix of
``P = kv_lens[b] - T`` tokens selects for itself over the prefix's compressed keys (the paged compressed-key cache) plus the chunk's own
complete blocks, writes the chunk's complete blocks into that cache (the chunk compress, the ring READ-ONLY) and the chunk's last ``cap``
raw keys into the ring (the cache-write RING arm at the stack's ``index_state_slot_mapping``) -- so a FROST prefill primes the caches a FROST
decode step reads, with ONE rounding convention (``QsaSpec.index_compress_rounding``).

Host cells (any device, no host sync): the declaration facts (the route ``"auto"`` -> the decode step inside the decode scorer's query-row
envelope and the extend step past it, the pins, the stage in the indexer's slot, the cache write's ring arm, ``fuse_prepare`` reading False,
the window plan), the carve (the nine slots at the END after the caller-list band carve, every slot 16-byte aligned, every layout without
the form byte-identical), the gather's piece-table precondition, the typed declines (every new row raised with its text EQUAL to the
envelope's -- the chunk-rows row the envelope's text + the two numbers --, the ``indexer_query_rows`` row kept under the explicit ``"decode"``
pin, the vocabulary / off-form / rounding refusals, the execute refusals incl. a ring whose cap differs from the declared sample's, a simulated
cc 10.0 / 10.3 reading declined by the sparse part statement), the envelope's new facts and the append-only tail of ``DecodeEnvelope``, and the
rounding attribute's reach: EVERY writer of an indexer row (the prefill form, the decode step unfused and fused, the extend step) carries the
declared value, the prefill form's ``compile()`` handing it to the compress recipe.

Rubin cells (``requires_rubin``): the EXTEND ladder against the TWO-STAGE oracle of the decode module (``_IndexerCaseBase``, the methods shared with ``_IndexerCase``, subclassed here: the
prefix built from the oracle's own raw band -- the prefix blocks compressed into the paged cache, the ring primed with the last ``cap``
positions below the chunk, exactly the form's ring INPUT contract --, ``n_vis`` derived on the HOST from ``kv_lens`` and the positions, never
from the kernel's ``ix_counts``): chunk starts at every residue (incl. residue 2), a true shorter-than-the-remainder chunk (``T < 4 - P % 4``),
``T < cap`` chunks (the tail carry through the ring arm), the chunk-start MTP pin (a row at position ``p`` lists the straddling block iff
``p >= 4 floor(P / 4) + 3`` and no earlier row lists it), the identity rows + the first sparse row, two select windows at a reduced budget,
the 128K-token reach, the e4m3 cache twins, the 24 / 2 geometry; the SET == the fp64 top-k of the block's OWN ``(ix_q, kbar)`` up to ties;
EVERY complete block of the stream bitwise a from-scratch compress after the chunk (the primed prefix untouched); the ring = the last ``cap``
positions bitwise the stream's raw band, every other slot untouched; ``out`` / the gated O / the LSE within the module's bars; two executes
bitwise with the side caches restored; the extend pin at decode widths vs the decode form on the same inputs (the same function within
the budget, never bitwise); cache parity under both roundings (``"hf"`` additionally against the HF-rounding oracle with the kernel module's
statistic gate); the device-value contract (the dead sequence exact; ``-2`` on ``0 < kv_lens < S_q``; ``-3`` past the table's reach with the
sentinel past every piece's compact slab and past ``get_workspace_size()`` untouched); the workspace honesty + no allocation / no host sync
over three executes; the launch census by kernel NAME (control + 6 at one window, + 2 per further window); capture after one warm-up.
"""

import math
import os
import sys
import types

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block import (
    INDEX_COMPRESS_ROUNDING_DEFAULT,
    INDEX_COMPRESS_ROUNDINGS,
    KV_SOURCE_PAGED,
    QSA_EXTEND_SELECT_BUDGET_BYTES,
    QSA_INDEX_SELECT_FORM_AUTO,
    QSA_INDEX_SELECT_FORM_DECODE,
    QSA_INDEX_SELECT_FORM_EXTEND,
    QSA_INDEX_SELECT_FORMS,
    DecodeEnvelope,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    QsaSpec,
    index_k_raw_view,
)
from cudnn.gated_attention_block import api as _api
from cudnn.gated_attention_block.api import (
    _align_up,
    _CacheWrite,
    _check_extend_piece_table,
    _extend_window_plan,
    _IndexerDecode,
    _IndexerExtend,
    _itemsize,
    _qsa_indexer_extend_slots,
    _SparseSdpa,
    _view,
    ProjBlock,
)

requirement_error = cutedsl_requirement_error("the gated attention block's indexer extend tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_block_qsa_decode as _qsa  # noqa: E402  (the decode module's oracle, helpers and cases)
from gated_block_bounds import assert_output_close  # noqa: E402
from gated_block_census import cupti_kernels as _cupti_kernels  # noqa: E402
from gated_block_reference import gated_attention_block_reference, qk_norm_rope_reference, split_qkvg  # noqa: E402

_DEV, BF16, F16, D, ROPE, BS = _qsa._DEV, _qsa.BF16, _qsa.F16, _qsa.D, _qsa.ROPE, _qsa.BS
GEOM_SMALL, GEOM_FLASH_NEXT = _qsa.GEOM_SMALL, _qsa.GEOM_FLASH_NEXT
_SM107 = _qsa._SM107
_E4M3 = torch.float8_e4m3fn
_SENT_I32 = _qsa._SENT_I32
_bits, _sentinel, _no_host_sync, _attend, _set_vs_topk, _TIE_TOL = (
    _qsa._bits,
    _qsa._sentinel,
    _qsa._no_host_sync,
    _qsa._attend,
    _qsa._set_vs_topk,
    _qsa._TIE_TOL,
)
_SCORE_ROWS = 256  # rows per chunk of the fp64 SET oracle (bounds the [rows, index_heads, n_blocks] scores at the 128K reach)

# the five new decline rows and their ONE shared token (the decode module pins the older rows)
_EXTEND_DECLINE_TOKENS = {
    "indexer_extend_fuse_prepare": "the T-row chunk's ring read and its tail write live in two launches",
    "indexer_extend_ring_slots": "the extend step takes the stack's ring slots",
    "indexer_extend_rope_tables_only": "the extend step is always unfused and its query norm reads the per-token",
    "indexer_extend_ring_rows": "at least block_size - 1 = 3 ring rows",
    "indexer_extend_chunk_rows": "a chunk longer than the table's reach",
}


def _prefix_only_reference(h_full, w_dense, w_q_norm, w_k_norm, cos, sin, ref_geom, *, chunk=8192):
    """The dense reference's POST-norm / post-RoPE Q, K, V and pre-sigmoid gate for a long prefix WITHOUT its O(L^2) attention (the chunk's
    attention over its list is the oracle's second stage anyway): the same projection, ``split_qkvg`` and ``qk_norm_rope_reference`` the
    reference runs, over row chunks.  Returns a namespace with ``q / k / v / gate`` ``[1, L, ...]`` in ``h_full``'s dtype."""
    dtype = h_full.dtype
    L = int(h_full.shape[1])
    q = torch.empty(1, L, ref_geom.h_q, ref_geom.d_head, dtype=dtype, device=h_full.device)
    gate = torch.empty_like(q)
    k = torch.empty(1, L, ref_geom.h_kv, ref_geom.d_head, dtype=dtype, device=h_full.device)
    v = torch.empty_like(k)
    for lo in range(0, L, chunk):
        hi = min(L, lo + chunk)
        proj = (h_full[:, lo:hi].float() @ w_dense.float().t()).to(dtype)
        q_pre, g_c, k_pre, v_c = split_qkvg(proj, ref_geom)
        q[:, lo:hi], _ = qk_norm_rope_reference(
            q_pre, w_q_norm, cos[:, lo:hi], sin[:, lo:hi], ref_geom.rope_dim, ref_geom.qk_norm_eps, qk_norm=ref_geom.qk_norm
        )
        k[:, lo:hi], _ = qk_norm_rope_reference(
            k_pre, w_k_norm, cos[:, lo:hi], sin[:, lo:hi], ref_geom.rope_dim, ref_geom.qk_norm_eps, qk_norm=ref_geom.qk_norm
        )
        gate[:, lo:hi] = g_c
        v[:, lo:hi] = v_c
    return types.SimpleNamespace(q=q, k=k, v=v, gate=gate)


class _IndexerExtendCase(_qsa._IndexerCaseBase):
    """One EXTEND step (a T-row chunk per sequence behind its prefix) of the IN-BLOCK INDEXER over the pools: the state of the decode module's
    ``_IndexerCase``, built here on the shared ``_IndexerCaseBase`` methods for ``S_q = T`` rows -- the whole-sequence inputs per entry (the dense reference, or its prefix-only twin for a very long
    prefix), the NaN pools holding the oracle's prefix K / V, the paged compressed-key cache primed with the PREFIX blocks ``j < floor(P / 4)``
    through the compress kernel over the oracle's raw band, the ring primed with the last ``cap`` positions below the chunk (the form's ring
    INPUT contract), the stack's ring slots for the chunk's LAST ``cap`` positions and ``-1`` elsewhere -- and the block declared with
    ``index_select_form`` (``"extend"``; ``"auto"`` past the decode envelope), ``index_state_slot_mapping_present=True``, ``fuse_prepare`` None /
    False, ``QsaSpec.index_compress_rounding``.  ``check_selection`` is the two-stage oracle's first stage for the chunk form; the inherited
    ``check`` its second (the attention over the KERNEL's list)."""

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
        return_lse=False,
        fuse_gate=False,
        split=None,
        n_pages=None,
        cap=4,
        form=QSA_INDEX_SELECT_FORM_EXTEND,
        rounding=INDEX_COMPRESS_ROUNDING_DEFAULT,
        fuse_prepare=None,
        outputs=True,
        lens_out=True,
        index_cache_dtype=None,
        prefix_only=False,
        max_pages=None,
    ):
        # the chunk form's state, built from scratch on ``_IndexerCaseBase`` (which has no ``__init__``): the decode step's ``_IndexerCase``
        # builds the DECODE declaration over the same oracle state -- this form's differs (``index_select_form``, the stack's ring slots, the
        # rounding attribute, ``cap`` free of ``S_q``) and so does its prefix / ring priming
        self.s_q, self.kv_lens, self.page, self.layout, self.dtype, self.top_k = int(s_q), [int(x) for x in kv_lens], page, layout, dtype, top_k
        self.block_lens_given, self.return_lse, self.fuse_gate, self.index_band = False, bool(return_lse), bool(fuse_gate), True
        self.index_list_form, self.shared, self.fuse_prepare = None, False, fuse_prepare
        self.sparse_tile_form = self.sparse_sub_tiles_per_cta = None
        self.slot_form, self.rope_tables_only, self.outputs, self.lens_out = True, False, bool(outputs), bool(lens_out)
        self.form, self.rounding, self.prefix_only = form, rounding, bool(prefix_only)
        self.index_cache_dtype = index_cache_dtype
        self.cache_dtype = dtype if index_cache_dtype is None else index_cache_dtype
        kw_g = dict(geom_kw, causal_bottom_right=True)
        self.geom = GatedAttentionBlockGeometry(
            **kw_g, qsa=QsaSpec(top_k=top_k, index_source="indexer", index_band=True, index_cache_dtype=index_cache_dtype, index_compress_rounding=rounding)
        )
        self.ref_geom = _qsa.RefGeometry(**kw_g)
        geom, ref_geom = self.geom, self.ref_geom
        h_q, h_kv, d_model, d_i, h_i = geom.h_q, geom.h_kv, geom.d_model, geom.qsa.index_head_dim, geom.qsa.index_heads
        G = h_q // h_kv
        B = len(self.kv_lens)
        g = torch.Generator(device=_DEV).manual_seed(seed)
        self.gen = g

        def randn(*shape, std=0.02):
            return (torch.randn(*shape, generator=g, device=_DEV, dtype=torch.float32) * std).to(dtype)

        # the same draw order as the decode module's _IndexerCase (one seed -> the same inputs for the two step forms)
        n_dense = sum(geom.qkvg_block_widths[:4])
        w_qkvg = randn(geom.n_qkvg, d_model)
        self.w = dict(
            w_qkvg=w_qkvg,
            w_q_norm=torch.ones(D, dtype=dtype, device=_DEV),
            w_k_norm=torch.ones(D, dtype=dtype, device=_DEV),
            w_o=randn(d_model, h_q * D),
            w_iq_norm=(1.0 + 0.03 * randn(d_i, std=1.0).float()).to(dtype),
            w_ik_norm=(1.0 + 0.02 * randn(d_i, std=1.0).float()).to(dtype),
        )
        self.w_band_k = w_qkvg[n_dense + h_i * d_i :]  # [d_i, d_model]: the band's raw-key head
        L_max = max(max(self.kv_lens), s_q)
        self.max_pages = max(-(-L_max // page), 1) if max_pages is None else int(max_pages)
        assert self.max_pages * page >= L_max
        self.n_pos = self.max_pages * page  # the position tables over the table's whole reach (the RoPE-table ceiling met with equality)
        cos_t, sin_t = _qsa._rope_tables(self.n_pos, dtype, ref_geom)
        self.cos_pos, self.sin_pos = cos_t, sin_t
        self.scale, self.G = geom.scale, G
        self.h = torch.zeros(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.cos = cos_t[0, :1].expand(s_q, ROPE).clone()[None].expand(B, s_q, ROPE).clone()
        self.sin = sin_t[0, :1].expand(s_q, ROPE).clone()[None].expand(B, s_q, ROPE).clone()
        self.ref_out = torch.zeros(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.ref_o_gated = torch.zeros(B, s_q, h_q, D, dtype=dtype, device=_DEV)
        self.ref_lse = torch.full((B, h_q, s_q), float("-inf"), dtype=torch.float32, device=_DEV)
        self.ref_rows, self.refs, self.k_raw_full = {}, {}, {}
        self.live_rows = torch.zeros(B, s_q, dtype=torch.bool, device=_DEV)
        for b, L in enumerate(self.kv_lens):
            pos = torch.arange(s_q, device=_DEV) + (L - s_q)
            live = pos >= 0
            self.live_rows[b] = live
            if L == 0:
                continue
            h_full = randn(1, L, d_model, std=1.0)
            if self.prefix_only:
                ref = _prefix_only_reference(h_full, w_qkvg[:n_dense], self.w["w_q_norm"], self.w["w_k_norm"], cos_t[:, :L], sin_t[:, :L], ref_geom)
            else:
                ref = gated_attention_block_reference(
                    h_full, w_qkvg[:n_dense], self.w["w_q_norm"], self.w["w_k_norm"], cos_t[:, :L], sin_t[:, :L], self.w["w_o"], ref_geom
                )
            self.refs[b] = ref
            self.ref_rows[b] = (ref.k[0], ref.v[0])
            # the band's raw key of every position: the oracle's own projection, ONE rounding (the prefix blocks and the ring are built from it)
            self.k_raw_full[b] = (h_full[0].float() @ self.w_band_k.float().t()).to(dtype)  # [L, d_i]
            p_live = pos[live]
            self.h[b, live] = h_full[0, p_live]
            self.cos[b, live] = cos_t[0, p_live]
            self.sin[b, live] = sin_t[0, p_live]
        need = sum(-(-L // page) for L in self.kv_lens)
        self.n_pages = n_pages if n_pages is not None else need + 1 + 2
        assert self.n_pages >= need + 1, (self.n_pages, need)
        perm = (torch.randperm(self.n_pages - 1, generator=g, device=_DEV) + 1).tolist()
        self.block_table = torch.full((B, self.max_pages), -1, dtype=torch.int32, device=_DEV)
        it = iter(perm)
        slots = torch.full((B * s_q,), -1, dtype=torch.int32, device=_DEV)
        self.k_cache, self.v_cache = _qsa._pools(self.n_pages, h_kv, page, layout, dtype)
        self.index_cache = None  # no raw-key pool: the raw key's tail enters the RING through the cache write's ring arm
        self.rpp = page // BS
        self.cap = int(cap)
        self.index_kv_cache = torch.full((self.n_pages, self.rpp, 1, d_i), float("nan"), dtype=self.cache_dtype, device=_DEV)
        self.index_state_cache = torch.full((B + 1, self.cap, 1, d_i), float("nan"), dtype=dtype, device=_DEV)
        self.index_state_block_table = torch.zeros(B, 1, dtype=torch.int32, device=_DEV)
        self.pages = {}
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                self.block_table[b] = 0  # the dead entry: the NULL block / the null ring block, slots -1
                continue
            pages = [next(it) for _ in range(-(-L // page))]
            self.pages[b] = pages
            pages_t = torch.tensor(pages, dtype=torch.int64, device=_DEV)
            self.block_table[b, : len(pages)] = pages_t.to(torch.int32)
            self.index_state_block_table[b, 0] = b + 1
            k_rows, v_rows = self.ref_rows[b]
            pos0 = L - s_q
            if pos0 > 0:  # the PREFIX: the oracle's rows into the pools (vectorised: a 128K prefix is 123k rows)
                p = torch.arange(pos0, device=_DEV)
                self.k_cache[pages_t[p // page], :, p % page] = k_rows[:pos0]
                self.v_cache[pages_t[p // page], :, p % page] = v_rows[:pos0]
            j = torch.arange(s_q, device=_DEV)
            p_step = pos0 + j
            live = p_step >= 0
            slots[b * s_q : (b + 1) * s_q] = torch.where(
                live, pages_t[p_step.clamp_min(0) // page] * page + p_step.clamp_min(0) % page, torch.full_like(p_step, -1)
            ).to(torch.int32)
            # the ring: the last `cap` positions below the chunk (the form's ring INPUT contract: positions 4 floor(P / 4) .. P - 1 among them)
            for p in range(max(0, pos0 - self.cap), max(pos0, 0)):
                self.index_state_cache[b + 1, p % self.cap, 0] = self.k_raw_full[b][p]
        self.slot_mapping = slots
        self.kv_lens_t = torch.tensor(self.kv_lens, dtype=torch.int32, device=_DEV)
        self.prefix_bits = (_bits(self.k_cache).clone(), _bits(self.v_cache).clone())
        # the ring's WRITE rows as the stack spells them for the EXTEND step: block x cap + pos % cap for the sequence's LAST cap tokens, -1 elsewhere
        pos_all = torch.tensor([[L - s_q + j for j in range(s_q)] for L in self.kv_lens], device=_DEV, dtype=torch.long)
        L_t = torch.tensor(self.kv_lens, device=_DEV, dtype=torch.long)[:, None]
        blk_of = self.index_state_block_table.long()
        tail = (pos_all >= 0) & (blk_of > 0) & (pos_all + self.cap >= L_t)
        self.ring_slots = torch.where(tail, blk_of * self.cap + pos_all % self.cap, torch.full_like(pos_all, -1)).reshape(-1).to(torch.int32)
        self.block_ids_out = torch.full((B, s_q, top_k), _SENT_I32, dtype=torch.int32, device=_DEV) if outputs else None
        self.block_lens_out = torch.full((B, s_q), _SENT_I32, dtype=torch.int32, device=_DEV) if lens_out else None
        kw = dict(
            paged_kv_page_size=page,
            kv_source=KV_SOURCE_PAGED,
            sample_k_cache=self.k_cache,
            sample_v_cache=self.v_cache,
            sample_block_table=self.block_table,
            sample_index_kv_cache=self.index_kv_cache,
            sample_index_state_cache=self.index_state_cache,
            sample_index_state_block_table=self.index_state_block_table,
            fuse_gate=fuse_gate,
            return_lse=return_lse,
            decode_split_kv=split,
            fuse_prepare=fuse_prepare,
            index_state_slot_mapping_present=True,
            index_select_form=form,
        )
        self.blk_kw = kw
        self.out = torch.empty(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.lse = torch.empty(B, h_q, s_q, dtype=torch.float32, device=_DEV) if return_lse else None
        self.blk = GatedAttentionBlockFwd(
            self.h, self.w["w_qkvg"], self.w["w_q_norm"], self.w["w_k_norm"], self.cos, self.sin, self.w["w_o"], self.out, self.geom, **kw
        )
        blk = self.blk
        assert isinstance(blk._sdpa, _SparseSdpa) and blk.indexer_decode and blk.indexer_extend and not blk.qsa_shared_list
        assert isinstance(blk._indexer_extend, _IndexerExtend) and blk._indexer_decode is None and blk._indexer is None
        assert blk.index_select_form == form and blk.index_select_form_resolved == QSA_INDEX_SELECT_FORM_EXTEND and blk.fuse_prepare is False
        assert (
            isinstance(blk._cache_write, _CacheWrite)
            and blk._cache_write.index_ring
            and not blk._cache_write.index_band
            and blk._cache_write.index_ring_cap == self.cap
        )
        assert blk.index_cache_dtype == self.cache_dtype and blk._indexer_extend.cache_dtype == self.cache_dtype and blk._indexer_extend.rounding == rounding
        self.ws = None
        self._compress = {}
        self._snap = None

    # -- the side caches between executes: the chunk's tail write overwrites ring slots the straddling block reads, so two executes
    # are bitwise only with the side caches RESTORED in between
    def snapshot(self):
        self._snap = (self.index_kv_cache.clone(), self.index_state_cache.clone())

    def restore(self):
        self.index_kv_cache.copy_(self._snap[0])
        self.index_state_cache.copy_(self._snap[1])

    def prime_prefix(self):
        """The PREFIX blocks of every live sequence (``j < floor(P / 4)``) into the paged compressed-key cache through the compress kernel under
        this case's rounding attribute (what a FROST prefill would have written), once per case; vectorised over the blocks (30720 at the 128K reach)."""
        if getattr(self, "_primed", False):
            return
        for b, L in enumerate(self.kv_lens):
            nb = max(L - self.s_q, 0) // BS
            if L == 0 or nb == 0:
                continue
            kb = self.compress(self.k_raw_full[b][None, : nb * BS], self.cos_pos, self.sin_pos)  # [1, nb, d_i]
            j = torch.arange(nb, device=_DEV)
            self.index_kv_cache[self.block_table[b, j // self.rpp].long(), j % self.rpp, 0] = kb[0]
        torch.cuda.synchronize()
        self._primed = True

    def compress(self, k_raw, cos, sin, pool=BS, out_dtype=None):
        """The compress kernel's prefill form under THIS case's rounding attribute (the from-scratch oracle of every cache row)."""
        from cudnn.gated_attention_block.kernels.qsa_compress import compile_qsa_compress, run_qsa_compress

        d_i = self.geom.qsa.index_head_dim
        od = self.cache_dtype if out_dtype is None else out_dtype
        if (pool, od) not in self._compress:
            self._compress[(pool, od)] = compile_qsa_compress(
                dtype=self.dtype, d=d_i, rope_dim=ROPE, pool=pool, has_seq_lens=False, table_dtype=self.dtype, out_dtype=od, rounding=self.rounding
            )
        k_raw = k_raw.contiguous()
        out = torch.full((int(k_raw.shape[0]), int(k_raw.shape[1]) // pool, d_i), float("nan"), dtype=od, device=_DEV)
        w = self.w["w_ik_norm"] if pool == BS else self.w["w_iq_norm"]
        run_qsa_compress(
            self._compress[(pool, od)],
            k_raw,
            out,
            w,
            cos,
            sin,
            None,
            eps=float(self.geom.qsa.index_norm_eps),
            norm_weight_offset=0.0,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()
        return out

    def compile(self):
        super().compile()  # the shared compile: this form's prefix priming, then the block's check_support / compile / workspace
        self.snapshot()
        return self

    def views(self):
        return self.blk._indexer_extend.views(self.ws, self.blk._ws)

    def n_vis(self) -> torch.Tensor:
        """The visible complete-block count per row ``clamp(floor((pos + 1) / 4), 0, n_blocks_max)`` ``[B, S_q]`` -- HOST arithmetic from ``kv_lens``,
        never the kernel's ``ix_counts``."""
        n_max = self.blk._indexer_extend.n_blocks_max
        return torch.tensor([[max(0, min((L - self.s_q + j + 1) // BS, n_max)) for j in range(self.s_q)] for L in self.kv_lens], device=_DEV)

    def stream_raw(self, b: int) -> torch.Tensor:
        """The sequence's raw-key STREAM after the chunk: the prefix from the oracle's band (what primed the cache and the ring), the chunk's
        rows from the slab's band (the FROST GEMM's) -- ``[L, d_i]``."""
        pos0 = self.kv_lens[b] - self.s_q
        g, lay = self.geom, self.blk._ws
        proj = _view(self.ws, lay.proj, (len(self.kv_lens) * self.s_q, g.n_qkvg), self.dtype)
        raw = index_k_raw_view(proj, g, len(self.kv_lens), self.s_q)[:, :, 0]  # [B, S_q, d_i]
        return torch.cat([self.k_raw_full[b][: max(pos0, 0)], raw[b, max(-pos0, 0) :]], 0)

    def check_selection(self, what: str = "selection") -> dict:
        """The two-stage oracle's FIRST stage for the chunk form, on the kernel's own outputs: the counts rule (``ix_counts`` and ``block_lens_out``
        against the HOST's ``n_vis``), the list canonical (a strictly descending valid prefix then ``-1``, every id below the row's count), dead
        sequences (``counts`` 0, every slot ``-1``, the report ``-1``, no cache / ring row touched), the per-sequence report = ``n_keys``; the indexer
        queries BITWISE the pool-1 compress of the slab's band; EVERY complete block of the stream BITWISE a from-scratch compress of the stream's
        raw keys (the primed prefix untouched, the chunk's blocks -- the straddling one included -- written); the ring = the sequence's last ``cap``
        positions bitwise the stream's raw band, every other slot untouched; the gathered dense kbar = the cache rows through the table (+ the
        zero fill); the chunk-start MTP pin; the SET == the fp64 top-k of the kernel's OWN ``(ix_q, kbar)`` up to ties on every live row."""
        B, s_q, top_k = len(self.kv_lens), self.s_q, self.top_k
        t = B * s_q
        ids, counts, n_vis = self.kernel_ids(), self.kernel_counts(), self.n_vis()
        assert torch.equal(counts.cpu(), n_vis.cpu().to(torch.int32)), f"{what}: the gather's counts are not the visible counts"
        if self.block_lens_out is not None:
            assert torch.equal(
                self.block_lens_out.cpu(), torch.minimum(n_vis, torch.full_like(n_vis, top_k)).cpu().to(torch.int32)
            ), f"{what}: block_lens_out != min(top_k, n_vis)"
        valid = ids >= 0
        n_sel = torch.minimum(n_vis, torch.full_like(n_vis, top_k))
        assert torch.equal(
            valid, torch.arange(top_k, device=_DEV)[None, None, :] < n_sel.to(_DEV)[..., None]
        ), f"{what}: the valid ids are not the prefix min(top_k, n_vis)"
        assert not (ids == _SENT_I32).any(), f"{what}: a sentinel survived in the list"
        diffs = (ids[..., :-1] - ids[..., 1:]).masked_fill(~valid[..., 1:], 1)
        assert bool((diffs > 0).all()), f"{what}: a row's list is not strictly descending"
        assert bool((torch.where(valid, ids, torch.full_like(ids, -1)).amax(-1) < n_vis.to(_DEV)).all()), f"{what}: an id at or past the row's visible count"
        v = self.views()
        g = self.geom
        lay = self.blk._ws
        n_max = self.blk._indexer_extend.n_blocks_max
        proj = _view(self.ws, lay.proj, (t, g.n_qkvg), self.dtype)
        h_i, d_i = g.qsa.index_heads, g.qsa.index_head_dim
        q_raw = torch.as_strided(proj, (t, h_i, d_i), (g.n_qkvg, d_i, 1), proj.storage_offset() + g.qkvg_offsets[ProjBlock.INDEX])
        cos_q = self.cos.view(t, 1, ROPE).expand(t, h_i, ROPE)
        sin_q = self.sin.view(t, 1, ROPE).expand(t, h_i, ROPE)
        qi_ref = self.compress(q_raw.reshape(t, h_i, d_i), cos_q, sin_q, pool=1)
        assert torch.equal(_bits(v["qi"]), _bits(qi_ref)), f"{what}: the indexer queries are not the pool-1 compress of the slab's band"
        seqrep = v["seqrep"].tolist()
        flips = rows = 0
        gap = 0.0
        n_completed = 0
        for b, L in enumerate(self.kv_lens):
            pos0 = L - s_q
            if L == 0:
                assert (
                    torch.isnan(self.index_state_cache[0].float()).all() and torch.isnan(self.index_kv_cache[0].float()).all()
                ), f"{what}: the null ring block / page 0 moved"
                assert bool((counts[b] == 0).all()) and bool((ids[b] == -1).all()), f"{what}: a dead sequence scored"
                assert seqrep[b] == -1, f"{what}: the dead sequence's report is {seqrep[b]}, not -1"
                assert bool((v["kbar"][b, : min(n_max, -(-s_q // BS))].float() == 0).all()), f"{what}: the dead sequence's kbar rows are not the zero fill"
                continue
            n_keys = min(L // BS, n_max)
            assert seqrep[b] == n_keys, f"{what}: sequence {b}: the report is {seqrep[b]}, expected n_keys {n_keys}"
            stream = self.stream_raw(b)  # [L, d_i]
            nb = L // BS
            if nb:
                kb = self.compress(stream[None, : nb * BS], self.cos_pos, self.sin_pos)[0]  # the from-scratch compress of the WHOLE stream
                j_all = torch.arange(nb, device=_DEV)
                rows_c = self.index_kv_cache[self.block_table[b, j_all // self.rpp].long(), j_all % self.rpp, 0]
                assert torch.equal(
                    _bits(rows_c), _bits(kb)
                ), f"{what}: sequence {b}: the cache's {nb} complete blocks are not the from-scratch compress of the stream"
                # the gathered dense kbar: the cache rows through the table (a bit copy), the zero fill past n_keys up to ceil(T / 4)
                assert torch.equal(
                    _bits(v["kbar"][b, :n_keys]), _bits(rows_c[:n_keys])
                ), f"{what}: sequence {b}: ix_kbar is not the cache rows through the table"
            fill_end = min(n_max, -(-s_q // BS))
            if n_keys < fill_end:
                assert bool((v["kbar"][b, n_keys:fill_end].float() == 0).all()), f"{what}: sequence {b}: the kbar zero fill past n_keys is missing"
            # the incomplete tail block's row (if the stream has one) was never written
            if L % BS and nb < self.max_pages * self.rpp:
                assert torch.isnan(
                    self.index_kv_cache[int(self.block_table[b, nb // self.rpp]), nb % self.rpp, 0].float()
                ).all(), f"{what}: sequence {b}: the incomplete tail block's row was written"
            n_completed += max(0, nb - max(pos0, 0) // BS)
            # the ring: slot s holds the LAST position p of the sequence with p % cap == s (the chunk's tail arm, or the primed row); an unwritten slot NaN
            ring_b = self.index_state_cache[b + 1, :, 0]
            for s_ in range(self.cap):
                p_last = (
                    max(p for p in range(max(0, L - self.cap), L) if p % self.cap == s_)
                    if any(p % self.cap == s_ for p in range(max(0, L - self.cap), L))
                    else None
                )
                if p_last is None:
                    assert torch.isnan(ring_b[s_].float()).all(), f"{what}: sequence {b}: ring slot {s_} written without a position"
                    continue
                assert torch.equal(_bits(ring_b[s_]), _bits(stream[p_last])), f"{what}: sequence {b}: ring slot {s_} is not position {p_last}'s raw key"
            # the chunk-start MTP pin: the straddling / first block B0 = floor(max(P, 0) / 4) is listed by a row at position p iff p >= 4 B0 + 3 (and
            # every visible block fits the list); no earlier row of the chunk lists it
            b0 = max(pos0, 0) // BS
            for j in range(s_q):
                p = pos0 + j
                if p < 0:
                    assert int(counts[b, j]) == 0 and bool((ids[b, j] == -1).all()), f"{what}: a row at a negative position scored"
                    continue
                listed = bool((ids[b, j] == b0).any())
                if p < BS * b0 + BS - 1:
                    assert not listed and int(n_vis[b, j]) <= b0, f"{what}: sequence {b} row {j} (position {p}) lists block {b0} before it completes"
                elif int(n_vis[b, j]) <= top_k:
                    assert listed, f"{what}: sequence {b} row {j} (position {p}) does not list its completed block {b0}"
            # the SET: the fp64 top-k of the kernel's own operands over the row's visible blocks, in row chunks
            n = int(n_vis[b].max())
            if n == 0:
                continue
            kbar = v["kbar"][b, :n].double()  # [n, d_i]: the gathered operand (bitwise the cache rows above)
            qi_all = v["qi"].view(B, s_q, h_i, d_i)[b]
            for r0 in range(0, s_q, _SCORE_ROWS):
                r1 = min(s_q, r0 + _SCORE_ROWS)
                live = torch.tensor([pos0 + j >= 0 for j in range(r0, r1)], device=_DEV)
                if not bool(live.any()):
                    continue
                sc = torch.relu(torch.einsum("shd,jd->shj", qi_all[r0:r1].double(), kbar)).sum(1) / math.sqrt(d_i)  # [rows, n]
                limit = torch.arange(n, device=_DEV)[None, :] < n_vis[b, r0:r1].to(_DEV)[:, None]
                sc = sc.masked_fill(~limit, float("-inf"))
                ids_r = ids[b, r0:r1]
                m = torch.zeros(r1 - r0, n + 1, dtype=torch.bool, device=_DEV)
                m.scatter_(1, torch.where(ids_r >= 0, ids_r.long(), torch.full_like(ids_r, n).long()), torch.ones_like(ids_r, dtype=torch.bool))
                f, r, gp = _set_vs_topk(m[:, :n][live], sc[live], top_k, _TIE_TOL)
                flips, rows, gap = flips + f, rows + r, max(gap, gp)
        print(
            f"\n{self.label()} {what}: {n_completed} block(s) completed by the chunk, {flips} tie flips vs the fp64 top-k of the kernel's operands ({rows} rows, gap {gap:.2e})"
        )
        return dict(flips=flips, rows=rows, gap=gap, completed=n_completed)

    def label(self):
        ix = self.blk._indexer_extend
        return (
            f"[extend {self.geom.h_q}/{self.geom.h_kv} T={self.s_q} p{self.page} {self.layout} split={self.blk.decode_split_kv} gate={'fused' if self.fuse_gate else 'launch'} "
            f"lse={self.return_lse} cap={self.cap} form={self.form} rounding={self.rounding} cache={str(self.cache_dtype).replace('torch.', '')} "
            f"windows={ix.n_windows}x{ix.window_rows} NB={ix.n_blocks_max} lens={self.kv_lens}]"
        )


def _two_launches_extend(case: _IndexerExtendCase, what: str) -> dict:
    """Two executes with the side caches restored in between are bitwise on ``out``, the gated O, the LSE, the list, the cache and the ring;
    then the two-stage oracle."""
    case.restore()
    a = case.run().clone()
    ids_a, o_a = case.kernel_ids().clone(), case.slab_views()[4].clone()
    lse_a = None if case.lse is None else case.lse.clone()
    cache_a, ring_a = _bits(case.index_kv_cache).clone(), _bits(case.index_state_cache).clone()
    lens_a = None if case.block_lens_out is None else case.block_lens_out.clone()
    case.restore()
    b = case.run().clone()
    assert (
        torch.equal(_bits(a), _bits(b)) and torch.equal(_bits(o_a), _bits(case.slab_views()[4])) and torch.equal(ids_a, case.kernel_ids())
    ), f"{what}: two launches are not bitwise"
    assert torch.equal(cache_a, _bits(case.index_kv_cache)) and torch.equal(
        ring_a, _bits(case.index_state_cache)
    ), f"{what}: the side caches differ between two launches"
    if lse_a is not None:
        assert torch.equal(lse_a, case.lse), f"{what}: two launches are not bitwise on the LSE"
    if lens_a is not None:
        assert torch.equal(lens_a, case.block_lens_out)
    m1 = case.check_selection(what)
    m2 = case.check(what=what)
    return dict(m1, **m2)


def _declare_extend(geom_kw=GEOM_SMALL, batch=1, s_q=128, *, page=16, n_pages=16, cap=4, form=QSA_INDEX_SELECT_FORM_EXTEND, slot_present=True, **kw):
    """A declared (not compiled) extend-step block on EMPTY tensors through the decode module's ``_declare_indexer`` (any device)."""
    return _qsa._declare_indexer(
        geom_kw, batch, s_q, page=page, n_pages=n_pages, cap=cap, index_select_form=form, index_state_slot_mapping_present=slot_present, **kw
    )


# ---------------------------------------------------------------------------
# Host: the declaration facts, the carve, the window plan, the typed declines, the envelope
# ---------------------------------------------------------------------------


def test_extend_declaration_facts_route_carve_and_window_plan():
    """The route and the form's facts, device-free: ``"auto"`` at ``S_q x next_pow2(index_heads) <= 64`` builds the DECODE step (today's cells untouched),
    past it the EXTEND step -- ``_IndexerExtend`` in the indexer's slot before the sparse core, ``fuse_prepare`` reading False, the cache write's RING
    arm (no raw-key pool arm), the per-row list, no PDL names; the ``"extend"`` pin at decode widths (``S_q`` 1 .. 4, cap 3) builds the same stage, the
    ``"decode"`` pin keeps the decode step; the carve = the caller-list band block's + the nine slots at the END in launch order (``ix_cand`` last),
    every slot 16-byte aligned, every bf16 / e4m3 layout without the form byte-identical to the band block's; the window plan's arithmetic (one
    window at the default budget, the pieces one per sequence, ``R`` floored at 16 and capped at ``T``, two windows at a lowered budget) and the gather's
    piece-table precondition (a malformed piece table is a ``RuntimeError`` naming the piece)."""
    with _no_host_sync():
        for s_q, batch in ((17, 1), (128, 2), (300, 1)):
            blk = _declare_extend(GEOM_SMALL, batch, s_q, form=QSA_INDEX_SELECT_FORM_AUTO, n_pages=32)
            assert blk.indexer_decode and blk.indexer_extend and blk.index_select_form == "auto" and blk.index_select_form_resolved == "extend"
            assert isinstance(blk._indexer_extend, _IndexerExtend) and blk._indexer_decode is None and blk._indexer is None and blk._prepare is None
            assert blk.fuse_prepare is False and blk.fuse_prepare_requested is None and blk.qsa_list_form == "per_row" and not blk.qsa_shared_list
            names = [st.name for st in blk._stages]
            assert names.index("qsa_indexer") == names.index("sdpa_sparse") - 2 and names.index("cache_write") == names.index("sdpa_sparse") - 1, names
            assert blk._cache_write.index_ring and not blk._cache_write.index_band and blk._cache_write.index_ring_cap == 4
            ix = blk._indexer_extend
            assert ix.n_blocks_max == 32 * (16 // BS) and ix.max_kv_len == 32 * 16 and ix.pdl_names == () and blk.pdl_stages == ()
            assert ix.n_windows == 1 and ix.window_rows == batch * s_q and ix.pieces == [(b, 0, s_q) for b in range(batch)] and ix.launches_per_execute() == 6
            assert ix.n_pieces == -(-batch // 4) * 4 and ix.piece_table[:batch] == ix.pieces  # the table padded to the 16-byte boundary
            # the carve: the caller-list band block's layout + the nine slots at the END, in launch order, ix_cand last
            caller = _qsa._declare(GEOM_SMALL, batch, s_q, n_pages=32, index_band=True, index_list_form="per_row")[0]
            lay, lay_c = blk._layout(), caller._layout()
            slots = _qsa_indexer_extend_slots(blk.geom, batch, s_q, _itemsize(BF16), ix.n_blocks_max, ix.window_rows, ix.n_pieces)
            assert [nm for nm, _ in slots] == ["ix_q", "ix_cslot", "ix_kbar", "ix_qpos", "ix_counts", "ix_seqrep", "ix_ids", "ix_scores", "ix_cand"]
            t = batch * s_q
            sizes = dict(slots)
            assert (
                sizes["ix_q"] == t * 4 * 128 * 2
                and sizes["ix_kbar"] == batch * ix.n_blocks_max * 128 * 2
                and sizes["ix_cand"] == ix.window_rows * ix.n_blocks_max * 4
            )
            assert sizes["ix_qpos"] == ix.n_pieces * 4 and sizes["ix_counts"] == t * 4 and sizes["ix_seqrep"] == batch * 4 and sizes["ix_ids"] == t * 512 * 4
            acc = lay_c.total_bytes
            for nm, nbytes in slots:
                assert getattr(lay, nm) == acc and acc % 16 == 0, nm
                acc += _align_up(nbytes)
            assert lay.total_bytes == acc and lay.ix_logits == -1 and lay.ix_ids_sorted == -1 and lay.ix_sort_idx == -1
            assert lay_c.ix_qpos == -1 and lay_c.ix_seqrep == -1 and lay_c.ix_q == -1
            # the e4m3 indexer cache: ix_q / ix_kbar at 1 B per element, the other seven slots the same sizes
            e4 = _declare_extend(GEOM_SMALL, batch, s_q, form=QSA_INDEX_SELECT_FORM_AUTO, n_pages=32, qsa_over=dict(index_cache_dtype=_E4M3))
            lay8 = e4._layout()
            slots8 = _qsa_indexer_extend_slots(e4.geom, batch, s_q, 1, ix.n_blocks_max, ix.window_rows, ix.n_pieces)
            assert (
                dict(slots8)["ix_q"] == sizes["ix_q"] // 2
                and dict(slots8)["ix_kbar"] == sizes["ix_kbar"] // 2
                and [s for s in slots8 if s[0] not in ("ix_q", "ix_kbar")] == [s for s in slots if s[0] not in ("ix_q", "ix_kbar")]
            )
            acc8 = lay_c.total_bytes
            for nm, nbytes in slots8:
                assert getattr(lay8, nm) == acc8, nm
                acc8 += _align_up(nbytes)
            assert lay8.total_bytes == acc8 < lay.total_bytes and e4._indexer_extend.fp8 and e4.index_cache_dtype == _E4M3
        # "auto" inside the envelope: the decode step, untouched; the pins
        dec = _declare_extend(GEOM_SMALL, 2, 4, form=QSA_INDEX_SELECT_FORM_AUTO, cap=7)
        assert dec.indexer_decode and not dec.indexer_extend and dec.index_select_form_resolved == "decode" and isinstance(dec._indexer_decode, _IndexerDecode)
        # the decode form's cache write may be fused into the prepare launch (the cc 10.7 fuse_prepare default): no ring arm either way
        assert dec._indexer_extend is None and (dec._cache_write is None or not dec._cache_write.index_ring) and dec._layout().ix_qpos == -1
        for s_q in (1, 2, 4):
            pin = _declare_extend(GEOM_SMALL, 2, s_q, form=QSA_INDEX_SELECT_FORM_EXTEND, cap=3)
            assert (
                pin.indexer_extend
                and isinstance(pin._indexer_extend, _IndexerExtend)
                and pin._indexer_extend.n_windows == 1
                and pin._cache_write.index_ring_cap == 3
            )
        dpin = _declare_extend(GEOM_SMALL, 1, 4, form=QSA_INDEX_SELECT_FORM_DECODE, cap=7, slot_present=False)
        assert not dpin.indexer_extend and isinstance(dpin._indexer_decode, _IndexerDecode) and dpin.index_select_form == "decode"
    # the window plan's arithmetic (host ints)
    # (the piece TABLE lays the pieces out per window on a 4-int boundary -- the scorer binds a window's q_pos0 slice 16-byte aligned -- with
    # harmless (sequence, 0, 1) padding pieces between the windows; ``pieces`` stays the real pieces in order)
    r, windows, pieces, table = _extend_window_plan(2, 128, 32, QSA_EXTEND_SELECT_BUDGET_BYTES)
    assert (r, windows, pieces) == (256, [(0, 256, 0, 1, 0, 1, 128)], [(0, 0, 128), (1, 0, 128)]) and table == pieces + [(1, 0, 1), (1, 0, 1)]
    r, windows, pieces, table = _extend_window_plan(1, 1, 4, QSA_EXTEND_SELECT_BUDGET_BYTES)
    assert (r, windows, pieces) == (1, [(0, 1, 0, 0, 0, 0, 1)], [(0, 0, 1)]) and table == [(0, 0, 1)] * 4  # R capped at T
    r, windows, pieces, table = _extend_window_plan(1, 4096, 32768, 1 << 20)
    assert r == 16 and len(windows) == 256 and len(pieces) == 256 and len(table) == 4 * 256  # R floored at 16 rows
    r, windows, pieces, table = _extend_window_plan(3, 100, 1024, 3 * 100 * 1024 * 4 // 2 + 1)  # the budget halves the rows: two windows, a sequence cut in two
    assert r == 150 and windows == [(0, 150, 0, 1, 0, 1, 100), (150, 300, 4, 5, 1, 2, 100)] and pieces == [(0, 0, 100), (1, 0, 50), (1, 50, 50), (2, 0, 100)]
    assert table == [(0, 0, 100), (1, 0, 50), (1, 0, 1), (1, 0, 1), (1, 50, 50), (2, 0, 100), (2, 0, 1), (2, 0, 1)]
    for ws_ in windows:
        assert ws_[2] % 4 == 0  # every window's first piece on a 16-byte boundary of ix_qpos
        for p in range(ws_[2], ws_[3] + 1):
            assert ws_[4] <= table[p][0] <= ws_[5]
    assert [table[p] for ws_ in windows for p in range(ws_[2], ws_[3] + 1)] == pieces and len(table) % 4 == 0
    _check_extend_piece_table(table, 100, "qsa_indexer")
    import re

    with pytest.raises(RuntimeError, match=re.escape("select piece 1 of sequence 1 covers rows [50, 101) outside [0, 100)")):
        _check_extend_piece_table([(0, 0, 100), (1, 50, 51)], 100, "qsa_indexer")
    with pytest.raises(RuntimeError, match="piece 0 of sequence 0"):
        _check_extend_piece_table([(0, -1, 4)], 8, "qsa_indexer")
    with pytest.raises(RuntimeError, match="piece 0 of sequence 2"):
        _check_extend_piece_table([(2, 0, 0)], 8, "qsa_indexer")
    # a lowered budget at declaration: two windows, the slot sized by the window rows
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_api, "QSA_EXTEND_SELECT_BUDGET_BYTES", 128 * 64 * 4)  # 128 rows x NB 64 fp32 (16 pages x 4 rows)
        with _no_host_sync():
            two = _declare_extend(GEOM_SMALL, 1, 256, n_pages=16)
        ix = two._indexer_extend
        assert ix.n_windows == 2 and ix.window_rows == 128 and ix.pieces == [(0, 0, 128), (0, 128, 128)] and ix.launches_per_execute() == 8
        assert (
            ix.n_blocks_max == 64
            and dict(_qsa_indexer_extend_slots(two.geom, 1, 256, 2, ix.n_blocks_max, ix.window_rows, ix.n_pieces))["ix_cand"] == 128 * 64 * 4
        )


def test_extend_typed_declines_equal_the_envelope_row_by_row_and_the_accept_cells_construct():
    """Accept AND reject per rule: every new decline row raised on a REAL declaration with its text EQUAL to the envelope's row and its shared token
    (``indexer_extend_chunk_rows``: the envelope's row + the two numbers appended, the raised text STARTS WITH the row); the ``indexer_query_rows`` row
    under the explicit ``"decode"`` pin past the envelope (and ``"auto"`` past it SERVES: the extend stage built); ``cap 2`` refused and ``cap 3`` served
    on the extend form; ``S_q > max_kv_len`` refused naming both numbers; ``fuse_prepare=True`` / ``rope_position_tables_only=True`` /
    ``index_state_slot_mapping_present=False`` with their OWN texts; the vocabulary and a pin off the form (``ValueError``); the rounding attribute's
    rows; the execute refusals on an uncompiled block (incl. a ring whose cap differs from the declared sample's: legal by the ring floor, refused
    before any stage because the cache-write ring arm compiled the declared cap in); a simulated cc 10.0 / 10.3 reading declined by the sparse part
    statement at ``check_support``."""
    import re

    env = GatedAttentionBlockFwd.serving_envelope(_declare_extend().geom, BF16, _DEV, device_cc=_SM107).decode
    assert set(_EXTEND_DECLINE_TOKENS) <= set(env.declines) and all(tok in env.declines[nm] for nm, tok in _EXTEND_DECLINE_TOKENS.items())
    rows = {
        "indexer_extend_fuse_prepare": (NotImplementedError, lambda: _declare_extend(fuse_prepare=True)),
        "indexer_extend_rope_tables_only": (NotImplementedError, lambda: _declare_extend(rope_position_tables_only=True)),
        "indexer_extend_ring_slots": (NotImplementedError, lambda: _declare_extend(slot_present=False)),
        "indexer_extend_ring_rows": (ValueError, lambda: _declare_extend(cap=2)),
        "indexer_extend_chunk_rows": (ValueError, lambda: _declare_extend(GEOM_SMALL, 1, 129, n_pages=8)),
    }
    with _no_host_sync():
        for name, (exc, trip) in rows.items():
            with pytest.raises(exc, match=re.escape(_EXTEND_DECLINE_TOKENS[name])) as e:
                trip()
            if name == "indexer_extend_chunk_rows":
                assert str(e.value).startswith(env.declines[name]) and "S_q=129" in str(e.value) and "max_kv_len=128" in str(e.value), str(e.value)
            else:
                assert str(e.value) == env.declines[name], (name, str(e.value))
        # the explicit "decode" pin past the decode scorer's envelope keeps the indexer_query_rows row (its text the envelope's; the clause names the extend form)
        with pytest.raises(NotImplementedError, match=re.escape("query rows per sequence (at most 64")) as e:
            _declare_extend(GEOM_SMALL, 1, 17, form=QSA_INDEX_SELECT_FORM_DECODE, cap=20, slot_present=False)
        assert str(e.value) == env.declines["indexer_query_rows"] and "index_select_form='extend'" in str(e.value)
        # "auto" past it serves: the extend stage, and "auto" at 16 rows the decode step
        assert _declare_extend(GEOM_SMALL, 1, 17, form=QSA_INDEX_SELECT_FORM_AUTO, cap=4).indexer_extend
        assert not _declare_extend(GEOM_SMALL, 1, 16, form=QSA_INDEX_SELECT_FORM_AUTO, cap=19, slot_present=False).indexer_extend
        # the ring floor: cap 3 served (one window), cap 2 refused above; cap 6 at S_q 4 stays the DECODE form's refusal under "auto"
        assert _declare_extend(cap=3)._indexer_extend is not None
        with pytest.raises(ValueError, match=r"needs at least S_q \+ 4 - 1 = 7 rows"):
            _declare_extend(GEOM_SMALL, 1, 4, form=QSA_INDEX_SELECT_FORM_AUTO, cap=6, slot_present=False)
        # the vocabulary and a pin off the form
        with pytest.raises(ValueError, match=r"index_select_form must be one of \('auto', 'decode', 'extend'\)"):
            _declare_extend(form="chunk")
        with pytest.raises(ValueError, match="index_select_form='extend' selects the step form of the in-block indexer over the pools.*index_source='caller'"):
            _qsa._declare(GEOM_SMALL, 1, 8, n_pages=8, index_band=True, index_select_form="extend")
        with pytest.raises(ValueError, match="index_select_form='decode' selects the step form.*kv_source='self'"):
            geom = GatedAttentionBlockGeometry(**GEOM_SMALL, qsa=QsaSpec(index_source="indexer", index_band=True))
            mk = lambda *shape: torch.empty(*shape, dtype=BF16, device=_DEV)  # noqa: E731
            GatedAttentionBlockFwd(
                mk(1, 8, 512),
                mk(5760, 512),
                torch.ones(D, dtype=BF16, device=_DEV),
                torch.ones(D, dtype=BF16, device=_DEV),
                mk(1, 8, ROPE),
                mk(1, 8, ROPE),
                mk(512, 2048),
                mk(1, 8, 512),
                geom,
                index_select_form="decode",
            )
        # the rounding attribute (a QsaSpec row): the vocabulary, "hf" under caller lists
        with pytest.raises(ValueError, match=r"index_compress_rounding must be one of \('single', 'hf'\)"):
            QsaSpec(index_source="indexer", index_band=True, index_compress_rounding="bf16").validate()
        with pytest.raises(ValueError, match="index_compress_rounding='hf' requires index_source='indexer'"):
            QsaSpec(index_band=True, index_compress_rounding="hf").validate()
        assert _declare_extend(qsa_over=dict(index_compress_rounding="hf"))._indexer_extend.rounding == "hf"
        assert _declare_extend(qsa_over=dict(index_compress_rounding="single"))._indexer_extend.rounding == "single"
        # the execute refusals on an uncompiled block (form only): the prefill form's dense output, a caller list, the commit count, the missing ring slots
        blk = _declare_extend(GEOM_SMALL, 1, 32, n_pages=8)
        s = _qsa._indexer_samples(1, page=16, n_pages=8, cap=4)
        mk = lambda *shape, dt=BF16: torch.empty(*shape, dtype=dt, device=_DEV)  # noqa: E731
        i32 = lambda *shape: torch.zeros(*shape, dtype=torch.int32, device=_DEV)  # noqa: E731
        kc, vc = _qsa._pools(8, 2, 16, "hnd", BF16)
        base = dict(
            k_cache=kc,
            v_cache=vc,
            block_table=i32(1, 8),
            kv_lens=i32(1),
            slot_mapping=i32(32),
            w_iq_norm=torch.ones(128, dtype=BF16, device=_DEV),
            w_ik_norm=torch.ones(128, dtype=BF16, device=_DEV),
            index_kv_cache=s["sample_index_kv_cache"],
            index_state_cache=s["sample_index_state_cache"],
            index_state_block_table=s["sample_index_state_block_table"],
            index_state_slot_mapping=i32(32),
            cos_pos=mk(1, 128, ROPE),
            sin_pos=mk(1, 128, ROPE),
        )
        args = (
            mk(1, 32, 512),
            mk(5760, 512),
            torch.ones(D, dtype=BF16, device=_DEV),
            torch.ones(D, dtype=BF16, device=_DEV),
            mk(1, 32, ROPE),
            mk(1, 32, ROPE),
            mk(512, 2048),
            mk(1, 32, 512),
            torch.empty(16, dtype=torch.uint8, device=_DEV),
        )
        for over, match in (
            (dict(index_k_compressed=mk(1, 8, 128)), "index_k_compressed is the prefill form's dense compressed-key output"),
            (dict(block_ids=i32(32, 512)), "derives the selection in the block"),
            (dict(n_commit=i32(1)), "n_commit .* is refused"),
            (dict(index_state_slot_mapping=None), "execute needs index_state_slot_mapping"),
            (dict(index_k_raw=mk(8, 16, 128)), "index_k_raw is the raw indexer key's pool of a caller-list band declaration"),
            # a ring of cap 8 where the sample had cap 4: past the floor (>= 3) and every stride rule, refused by the declared cap before any stage
            (
                dict(index_state_cache=mk(2, 8, 1, 128)),
                "index_state_cache has cap=8 rows per ring block, but the extend step was declared on a ring sample of cap=4",
            ),
        ):
            kw = dict(base)
            kw.update(over)
            with pytest.raises(ValueError, match=match):
                blk.execute(*args, **kw)
        with pytest.raises(RuntimeError, match="call compile"):
            blk.execute(*args, **base)
    # the part rows: a simulated cc 10.0 / 10.3 reading is the sparse part statement at check_support (before any stage), the Rubin reading passes it
    from cudnn.gated_attention_block import api as api_mod

    for cc in ((10, 0), (10, 3), (9, 0)):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(api_mod, "_device_cc_of", lambda device, _cc=cc: _cc)
            with pytest.raises(NotImplementedError, match=rf"sparse forms .* served on cc 10\.7 \(Rubin\) only.*found SM{cc[0]}{cc[1]}"):
                _declare_extend().check_support()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the prefill form's compile() allocates its plan constants on a CUDA device")
def test_every_indexer_writer_carries_the_declared_rounding(monkeypatch):
    """``QsaSpec.index_compress_rounding`` reaches EVERY writer of an indexer row, not the extend step alone (the attribute is a declaration fact the
    docs and the tracker state for the prefill form, the decode step and the fused prepare's arms -- a stack that declares ``"hf"`` for its
    chunked prefill and its decode steps must get ONE chain): on a host declaration under both conventions the prefill form's stage
    (``_Indexer``), the decode step's unfused stage (``_IndexerDecode``), the fused prepare's ``QsaPrepareIndexArms`` record and the extend step's
    stage each carry the declared value; and the prefill form's ``compile()`` below the identity bound (its one compress recipe stubbed to a
    sentinel, the plan constants it builds allocated on this device) hands ``rounding=`` to ``compile_qsa_compress`` -- the cell that is red when a
    stage compiles the default chain whatever the declaration says.  The device cells (the e2e module's chunked prefill + decode steps and its
    prefill-form cell, both parametrised over the roundings) pin the rows those recipes write."""
    from cudnn.gated_attention_block.api import _Indexer
    from cudnn.gated_attention_block.kernels import qsa_compress as compress_mod  # the module object: the recipe compile is stubbed on it

    geom_kw = GEOM_SMALL
    mk = lambda *shape: torch.empty(*shape, dtype=BF16, device=_DEV)  # noqa: E731
    ones = lambda: torch.ones(D, dtype=BF16, device=_DEV)  # noqa: E731

    def prefill_block(rounding, s_q=8):
        geom = GatedAttentionBlockGeometry(**geom_kw, qsa=QsaSpec(index_source="indexer", index_band=True, index_compress_rounding=rounding))
        return GatedAttentionBlockFwd(mk(1, s_q, 512), mk(5760, 512), ones(), ones(), mk(1, s_q, ROPE), mk(1, s_q, ROPE), mk(512, 2048), mk(1, s_q, 512), geom)

    seen = []
    monkeypatch.setattr(compress_mod, "compile_qsa_compress", lambda **kw: seen.append(dict(kw)) or object())
    with _no_host_sync():
        for rounding in INDEX_COMPRESS_ROUNDINGS:
            over = dict(index_compress_rounding=rounding)
            # the extend step, the decode step (unfused: the stage's two recipes; fused: the prepare record the launch is traced from)
            assert _declare_extend(qsa_over=over)._indexer_extend.rounding == rounding
            off = _qsa._declare_indexer(geom_kw, 1, 1, cap=4, qsa_over=over, fuse_prepare=False)
            on = _qsa._declare_indexer(geom_kw, 1, 1, cap=4, qsa_over=over, fuse_prepare=True)
            assert isinstance(off._indexer_decode, _IndexerDecode) and off._indexer_decode.rounding == rounding and off._prepare is None
            assert on._indexer_decode.fused and on._prepare is not None and on._prepare._index_arms().rounding == rounding
            # the prefill form: the stage's attribute, then its compile() below the identity bound -- the one compress recipe is the key compress
            blk = prefill_block(rounding)
            assert isinstance(blk._indexer, _Indexer) and blk._indexer.rounding == rounding and not blk._indexer.selects
            n0 = len(seen)
            blk._indexer.compile()
            assert len(seen) == n0 + 1 and seen[-1]["pool"] == BS and seen[-1]["rounding"] == rounding, seen[n0:]
    assert [kw["rounding"] for kw in seen] == list(INDEX_COMPRESS_ROUNDINGS)


def test_extend_envelope_facts_are_published_and_the_tail_is_append_only(monkeypatch):
    """The envelope's EXTEND facts on a Rubin reading (the DSL state MODELED as the served one, the sibling modules' convention): ``index_select_forms
    == ("auto", "decode", "extend")``, ``index_extend_ring_rows_min == 3 == index_state_rows_extra`` (the kernel's ``min_ring_rows(..., ring_write=False)``,
    one spelling), ``index_extend_select_budget_bytes == QSA_EXTEND_SELECT_BUDGET_BYTES``, ``index_compress_roundings == ("single", "hf")``, the five
    decline rows verbatim; empty / 0 under f16 and on a dense geometry; the record's tail append-only (the four fields after ``route_cc``, then
    ``quant_modes``)."""
    import dataclasses

    from cudnn.frost import buffers as _buffers
    from cudnn.gated_attention_block.kernels.qsa_compress import min_ring_rows

    monkeypatch.setattr(_buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(_buffers, "_cutedsl_has_sm107", lambda: True)
    geom = GatedAttentionBlockGeometry(**GEOM_SMALL, causal_bottom_right=True, qsa=QsaSpec(index_source="indexer", index_band=True))
    d = GatedAttentionBlockFwd.serving_envelope(geom, BF16, _DEV, device_cc=_SM107).decode
    assert d.served and d.sparse_core
    assert d.index_select_forms == QSA_INDEX_SELECT_FORMS == ("auto", "decode", "extend")
    assert d.index_extend_ring_rows_min == 3 == d.index_state_rows_extra == min_ring_rows(2048, BS, ring_write=False) == min_ring_rows(1, BS, ring_write=False)
    assert d.index_extend_select_budget_bytes == QSA_EXTEND_SELECT_BUDGET_BYTES == 512 << 20
    assert d.index_compress_roundings == INDEX_COMPRESS_ROUNDINGS == ("single", "hf") and INDEX_COMPRESS_ROUNDING_DEFAULT == "single"
    assert d.index_query_rows_max == 64  # the DECODE step's envelope (the explicit-pin row)
    assert all(tok in d.declines[nm] for nm, tok in _EXTEND_DECLINE_TOKENS.items())
    f16 = GatedAttentionBlockFwd.serving_envelope(geom, F16, _DEV, device_cc=_SM107).decode
    assert (
        f16.index_select_forms == ()
        and f16.index_extend_ring_rows_min == 0
        and f16.index_extend_select_budget_bytes == 0
        and f16.index_compress_roundings == ()
    )
    dense = GatedAttentionBlockFwd.serving_envelope(GatedAttentionBlockGeometry(**GEOM_SMALL, causal_bottom_right=True), BF16, _DEV, device_cc=_SM107).decode
    assert dense.index_select_forms == () and dense.index_extend_ring_rows_min == 0 and not (set(_EXTEND_DECLINE_TOKENS) & set(dense.declines))
    caller = GatedAttentionBlockFwd.serving_envelope(
        GatedAttentionBlockGeometry(**GEOM_SMALL, causal_bottom_right=True, qsa=QsaSpec(index_band=True)), BF16, _DEV, device_cc=_SM107
    ).decode
    assert (
        caller.index_select_forms == ("auto", "decode", "extend") and caller.index_extend_ring_rows_min == 3
    )  # the geometry's facts, read before a declaration
    # the append-only tail: the EXTEND step's four fields after `route_cc`, then `quant_modes` (appended 2026-10-30: the quant spellings the DENSE
    # paged forms serve on the part -- the FP8-projection + bf16-attention mode; `()` on every QsaSpec geometry)
    assert [f.name for f in dataclasses.fields(DecodeEnvelope)][-6:] == [
        "route_cc",
        "index_select_forms",
        "index_extend_ring_rows_min",
        "index_extend_select_budget_bytes",
        "index_compress_roundings",
        "quant_modes",
    ]
    assert d.quant_modes == () == caller.quant_modes, "a QsaSpec geometry serves no quant spelling over the pools"


# ---------------------------------------------------------------------------
# Rubin: the EXTEND ladder, the roundings, the device-value contract, the workspace, the census, the capture
# ---------------------------------------------------------------------------

# (geometry, T, kv_lens AFTER the chunk, page, layout, cap, split, fuse_gate, return_lse, seed, index_cache_dtype, prefix_only)
_EXTEND_LADDER = [
    pytest.param(GEOM_SMALL, 20, (321,), 16, "hnd", 4, None, False, True, 201, None, False, id="T20-P301-res1-p16-hnd-cap4"),
    pytest.param(GEOM_SMALL, 128, (6272, 0), 32, "nhd", 8, None, True, True, 202, None, False, id="T128-P6144-res0-dead-p32-nhd-cap8"),
    pytest.param(GEOM_SMALL, 300, (8495,), 16, "hnd", 4, 2, True, False, 203, None, False, id="T300-P8195-res3-p16-split2"),
    pytest.param(GEOM_SMALL, 512, (6658,), 16, "hnd", 4, None, True, True, 204, None, False, id="T512-P6146-res2-p16"),
    pytest.param(GEOM_SMALL, 2, (8196,), 16, "hnd", 4, 1, False, True, 205, None, False, id="T2-P8194-res2-one-block-completes-cap4"),
    pytest.param(GEOM_SMALL, 2, (8195,), 16, "hnd", 4, 1, False, True, 206, None, False, id="T2-P8193-res1-no-block-completes-tail-alone"),
    pytest.param(GEOM_SMALL, 1, (8195,), 16, "hnd", 4, 1, True, True, 207, None, False, id="T1-P8194-res2-no-block-completes"),
    pytest.param(GEOM_SMALL, 2048, (8192,), 16, "unified", 4, 1, True, True, 208, None, False, id="T2048-P6144-res0-p16-unified-split1"),
    pytest.param(GEOM_SMALL, 512, (31232, 31232, 31232, 31232), 16, "hnd", 4, None, True, True, 209, None, False, id="B4-T512-P30720-p16"),
    pytest.param(GEOM_SMALL, 2052, (2052,), 64, "hnd", 8, None, False, True, 210, None, False, id="T2052-P0-identity-rows-plus-first-sparse-row-p64-cap8"),
    pytest.param(GEOM_SMALL, 128, (123008,), 128, "hnd", 4, None, True, True, 211, None, True, id="T128-P122880-128K-reach-p128-prefix-only-oracle"),
    pytest.param(GEOM_SMALL, 20, (321,), 16, "hnd", 4, None, False, True, 201, _E4M3, False, id="T20-P301-res1-e4m3"),
    pytest.param(GEOM_SMALL, 300, (8495,), 16, "hnd", 4, 2, True, False, 203, _E4M3, False, id="T300-P8195-res3-split2-e4m3"),
    pytest.param(GEOM_SMALL, 2048, (8192,), 16, "unified", 4, 1, True, True, 208, _E4M3, False, id="T2048-P6144-unified-e4m3"),
    pytest.param(GEOM_FLASH_NEXT, 20, (321,), 16, "hnd", 4, None, True, True, 212, None, False, id="fn24-2-T20-P301"),
    pytest.param(GEOM_FLASH_NEXT, 128, (6274,), 16, "hnd", 4, None, True, True, 213, None, False, id="fn24-2-T128-P6146-res2"),
]


@requires_rubin
@pytest.mark.parametrize("geom_kw, t, kv_lens, page, layout, cap, split, fuse_gate, return_lse, seed, cache_dtype, prefix_only", _EXTEND_LADDER)
def test_extend_indexer_matches_the_two_stage_oracle(geom_kw, t, kv_lens, page, layout, cap, split, fuse_gate, return_lse, seed, cache_dtype, prefix_only):
    """The EXTEND ladder (module docstring): per cell the selection against the kernel's own operands and the caches' parity (stage 1), the
    attention within the module's bars of the fp32 oracle fed the kernel's list (stage 2), dead sequences exact, the pools' chunk rows bitwise
    the slab, two executes bitwise with the side caches restored.  The e4m3 rows add the bf16 twin on the same inputs: the cache rows and ``ix_q``
    bitwise its cast, the set agreement a statistic with every flip at the rank boundary."""
    kw = dict(page=page, layout=layout, seed=seed, split=split, fuse_gate=fuse_gate, return_lse=return_lse, cap=cap, prefix_only=prefix_only)
    case = _IndexerExtendCase(geom_kw, t, kv_lens, index_cache_dtype=cache_dtype, **kw)
    case.compile()
    if split is not None:
        assert case.blk.decode_split_kv == split
    m = _two_launches_extend(case, "extend ladder")
    print(f"\n{case.label()}: {m}")
    if cache_dtype is not None:
        assert case.index_kv_cache.dtype == case.views()["qi"].dtype == case.views()["kbar"].dtype == _E4M3 and case.index_state_cache.dtype == BF16
        twin = _IndexerExtendCase(geom_kw, t, kv_lens, **kw)
        twin.compile()
        _two_launches_extend(twin, "extend ladder (bf16 twin)")
        m2 = case.check_e4m3_twin(twin, "e4m3 vs the bf16 twin")
        print(f"\n{case.label()}: {m2}")


@requires_rubin
@pytest.mark.parametrize("s_q, kv_lens", [(1, (8195, 301, 0)), (2, (8194, 2052)), (4, (8191, 8190, 8189, 8188))], ids=["Sq1", "Sq2", "Sq4"])
def test_extend_pin_at_decode_widths_computes_the_decode_forms_function(s_q, kv_lens):
    """``index_select_form="extend"`` at ``S_q`` 1 .. 4 against the DECODE form on the same inputs (the same seed: the same weights, prefix, pools and
    caches): the same selection SET up to ties (the decode form's set against the fp64 top-k of the extend form's operands), the same ``out`` / gated O
    within the module's bars (never bitwise: two scorers' fp32 orders), the ring and the cache BITWISE after both (the extend form's tail write and the
    decode form's row writes land the same raw keys at the same slots at ``S_q <= cap``; the completed blocks are one compress chain)."""
    kw = dict(page=16, seed=300 + s_q, cap=8, fuse_gate=True, return_lse=True)
    ext = _IndexerExtendCase(GEOM_SMALL, s_q, kv_lens, **kw)
    dec = _qsa._IndexerCase(GEOM_SMALL, s_q, kv_lens, fuse_prepare=False, **kw)
    assert torch.equal(ext.h, dec.h) and torch.equal(_bits(ext.w["w_qkvg"]), _bits(dec.w["w_qkvg"])) and torch.equal(ext.block_table, dec.block_table)
    ext.compile()
    dec.compile()
    _two_launches_extend(ext, "extend @ decode width")
    _qsa._two_launches_indexer(dec, "decode @ decode width")
    assert torch.equal(_bits(ext.index_state_cache), _bits(dec.index_state_cache)), "the ring differs between the two forms"
    assert torch.equal(_bits(ext.index_kv_cache), _bits(dec.index_kv_cache)), "the compressed-key cache differs between the two forms"
    assert torch.equal(ext.kernel_counts(), dec.kernel_counts()) and torch.equal(ext.block_lens_out, dec.block_lens_out)
    assert torch.equal(_bits(ext.views()["qi"]), _bits(dec.views()["qi"]))
    # the decode form's SET against the fp64 top-k of the extend form's operands (the same operands bitwise), up to ties
    B, top_k = len(kv_lens), ext.top_k
    n_vis = ext.n_vis()
    d_i, h_i = ext.geom.qsa.index_head_dim, ext.geom.qsa.index_heads
    for b, L in enumerate(kv_lens):
        n = int(n_vis[b].max())
        if L == 0 or n == 0:
            continue
        kbar = ext.views()["kbar"][b, :n].double()
        qi = ext.views()["qi"].view(B, s_q, h_i, d_i)[b].double()
        sc = torch.relu(torch.einsum("shd,jd->shj", qi, kbar)).sum(1) / math.sqrt(d_i)
        sc = sc.masked_fill(~(torch.arange(n, device=_DEV)[None, :] < n_vis[b].to(_DEV)[:, None]), float("-inf"))
        live = torch.tensor([L - s_q + j >= 0 for j in range(s_q)], device=_DEV)
        ids_d = dec.kernel_ids()[b]
        m = torch.zeros(s_q, n + 1, dtype=torch.bool, device=_DEV)
        m.scatter_(1, torch.where(ids_d >= 0, ids_d.long(), torch.full_like(ids_d, n).long()), torch.ones_like(ids_d, dtype=torch.bool))
        _set_vs_topk(m[:, :n][live], sc[live], top_k, _TIE_TOL)
    live = ext.live_rows
    c = _qsa._cos(ext.out[live], dec.out[live])
    assert c > _qsa.COS_OUT, f"extend vs decode: cos(out) {c}"
    assert_output_close(ext.slab_views()[4][live], dec.slab_views()[4][live], f"extend vs decode @ S_q {s_q}: the gated O")
    assert float((ext.lse - dec.lse).abs()[~torch.isinf(dec.lse)].max()) <= _qsa.ATOL
    print(f"\nextend vs decode @ S_q {s_q}: cos(out) {c:.6f}")


@requires_rubin
@pytest.mark.parametrize("rounding", INDEX_COMPRESS_ROUNDINGS)
@pytest.mark.parametrize("t, kv_lens, page, cap, seed", [(300, (8495, 0, 321), 16, 4, 221), (2, (8196, 2052), 16, 4, 222)], ids=["T300-res3", "T2-res2"])
def test_extend_cache_parity_under_both_roundings(rounding, t, kv_lens, page, cap, seed):
    """Cache parity under ``QsaSpec.index_compress_rounding``: the ladder's gates under the attribute (every complete block of the stream BITWISE the
    from-scratch compress under the SAME attribute, the queries bitwise the pool-1 compress under it -- the prefill form, the chunk form and the
    oracle's helper are one chain per convention); ``"hf"`` additionally against the HF-rounding ORACLE (``compress_reference(hf_rounding=True)``) with
    the kernel module's statistic gate on the chunk's completed rows and the queries (>= 99.9 % bitwise, every element within the operand-magnitude
    cap); ``"single"`` within one io ulp of the fp32 oracle at the same yardstick."""
    import test_qsa_compress as _c

    case = _IndexerExtendCase(GEOM_SMALL, t, kv_lens, page=page, cap=cap, seed=seed, rounding=rounding, return_lse=True)
    case.compile()
    _two_launches_extend(case, f"cache parity ({rounding})")
    hf = rounding == "hf"
    w_k, w_q = case.w["w_ik_norm"], case.w["w_iq_norm"]
    eps = float(case.geom.qsa.index_norm_eps)
    B, s_q = len(kv_lens), t
    v = case.views()
    for b, L in enumerate(kv_lens):
        if L == 0:
            continue
        pos0 = L - s_q
        nb = L // BS
        b0 = max(pos0, 0) // BS
        if nb <= b0:
            continue  # no block completed by the chunk
        stream = case.stream_raw(b)[None, : nb * BS]  # [1, 4 nb, d_i]
        want32, _ = _c.compress_reference(
            stream, w_k, case.cos_pos, case.sin_pos, None, pool=BS, eps=eps, norm_weight_offset=0.0, rope_dim=ROPE, hf_rounding=hf
        )
        mag = _c.hf_rotation_operand_magnitude(stream, w_k, pool=BS, eps=eps, norm_weight_offset=0.0, rope_dim=ROPE, hf_rounding=hf)
        j_all = torch.arange(b0, nb, device=_DEV)
        got = case.index_kv_cache[case.block_table[b, j_all // case.rpp].long(), j_all % case.rpp, 0][None]  # the chunk's completed rows
        mask = torch.ones_like(got, dtype=torch.bool)
        if hf:
            frac = _c._assert_hf_rounding_points(got, want32[:, b0:nb], mask, f"seq {b}: the chunk's completed rows vs the HF-rounding oracle", mag[:, b0:nb])
        else:
            frac = _c._assert_one_rounding(got, want32[:, b0:nb], mask, f"seq {b}: the chunk's completed rows vs the fp32 oracle", operand_mag=mag[:, b0:nb])
        print(f"\nseq {b} rounding={rounding}: {nb - b0} completed rows, {100 * frac:.3f} % bitwise the {'HF-rounding' if hf else 'fp32'} oracle")
    # the queries: the pool-1 chain under the attribute vs the same oracle (one row per (token, head))
    g, lay = case.geom, case.blk._ws
    proj = _view(case.ws, lay.proj, (B * s_q, g.n_qkvg), BF16)
    h_i, d_i = g.qsa.index_heads, g.qsa.index_head_dim
    q_raw = torch.as_strided(proj, (B * s_q, h_i, d_i), (g.n_qkvg, d_i, 1), proj.storage_offset() + g.qkvg_offsets[ProjBlock.INDEX]).reshape(
        B * s_q * h_i, 1, d_i
    )
    cos_q = case.cos.view(B * s_q, 1, ROPE).expand(B * s_q, h_i, ROPE).reshape(B * s_q * h_i, 1, ROPE)
    sin_q = case.sin.view(B * s_q, 1, ROPE).expand(B * s_q, h_i, ROPE).reshape(B * s_q * h_i, 1, ROPE)
    want32, _ = _c.compress_reference(q_raw, w_q, cos_q, sin_q, None, pool=1, eps=eps, norm_weight_offset=0.0, rope_dim=ROPE, hf_rounding=hf)
    mag = _c.hf_rotation_operand_magnitude(q_raw, w_q, pool=1, eps=eps, norm_weight_offset=0.0, rope_dim=ROPE, hf_rounding=hf)
    got = v["qi"].reshape(B * s_q * h_i, 1, d_i)
    live = case.live_rows.view(B * s_q, 1, 1).expand(B * s_q, h_i, 1).reshape(B * s_q * h_i, 1, 1).expand(-1, 1, d_i)
    if hf:
        frac = _c._assert_hf_rounding_points(got, want32, live, "the indexer queries vs the HF-rounding oracle", mag)
    else:
        frac = _c._assert_one_rounding(got, want32, live, "the indexer queries vs the fp32 oracle", operand_mag=mag)
    print(f"\nqueries rounding={rounding}: {100 * frac:.3f} % bitwise the oracle")


def _sentinel_past(ws: torch.Tensor, n: int) -> torch.Tensor:
    """A workspace of ``n`` bytes + a 4 KiB sentinel tail (0x7F), returned as the whole buffer; the caller hands ``ws[:n]`` to the block."""
    buf = torch.full((n + 4096,), 0x7F, dtype=torch.uint8, device=_DEV)
    return buf


@requires_rubin
def test_extend_device_value_contract_reports_short_and_over_sequences_and_keeps_memory_safe():
    """The device-value contract ``kv_lens[b] in {0} u [S_q, max_pages x P]``: a SHORT sequence (``0 < kv_lens[b] < S_q``) is reported ``-2`` in ``ix_seqrep``
    (its negative-position rows dead, its live rows' selection position-misaligned -- documented, not an oracle), the whole block runs and stays
    finite; a sequence PAST the table's reach (``kv_lens[b] > max_pages x P``) is reported ``-3`` by the indexer stage run on its own (the sparse core's
    own contract excludes it): its clamped ``q_pos0`` keeps every scorer write inside its piece's compact slab -- the sentinel past every piece's used
    slab of ``ix_cand`` and past ``get_workspace_size()`` untouched, the compress's blocks past the table reported ``-2`` and unwritten; the dead
    sequence beside them exact (the ladder's contract, re-asserted here)."""
    from cudnn.gated_attention_block.kernels.qsa_kbar_gather import SEQREP_DEAD, SEQREP_OVER, SEQREP_SHORT

    t = 48
    case = _IndexerExtendCase(GEOM_SMALL, t, (300, 0, 1000), page=16, cap=4, seed=231, return_lse=True)
    case.compile()
    _two_launches_extend(case, "contract: the served rows")
    blk, ws = case.blk, case.ws
    ix = blk._indexer_extend
    # (a) the SHORT sequence: kv_lens 1000 -> 20 (0 < 20 < 48) on sequence 2; the block runs (the core's rows at negative positions are dead)
    case.restore()
    short = torch.tensor([300, 0, 20], dtype=torch.int32, device=_DEV)
    case.execute(kv_lens=short)
    torch.cuda.synchronize()
    rep = case.views()["seqrep"].tolist()
    assert rep == [300 // BS, SEQREP_DEAD, SEQREP_SHORT], rep
    counts = case.kernel_counts()
    assert bool((counts[1] == 0).all()) and bool((counts[2, : t - 20] == 0).all()) and int(counts[2, -1]) == 20 // BS, counts[2].tolist()
    assert torch.isfinite(case.out.float()).all(), "the short sequence's misaligned rows must stay finite (finite inputs through finite kernels)"
    assert bool((case.out[1] == 0).all()), "the dead sequence is not exact beside a short one"
    # (b) the OVER sequence: kv_lens past max_pages x P on sequence 0, through the INDEXER STAGE alone on a sentinel-tailed workspace
    case.restore()
    n = blk.get_workspace_size()
    buf = _sentinel_past(ws, n)
    buf[:n].copy_(ws)
    over = torch.tensor([case.max_pages * 16 + 5, 0, 1000], dtype=torch.int32, device=_DEV)
    lay = blk._ws
    v = ix.views(buf[:n], lay)
    cand_sent = torch.full_like(v["cand"], -7.0)
    v["cand"].copy_(cand_sent)
    proj = _view(buf[:n], lay.proj, (3 * t, blk.geom.n_qkvg), BF16)
    ix.execute(
        proj,
        workspace=buf[:n],
        lay=lay,
        w_iq_norm=case.w["w_iq_norm"],
        w_ik_norm=case.w["w_ik_norm"],
        cos=case.cos,
        sin=case.sin,
        cos_pos=case.cos_pos,
        sin_pos=case.sin_pos,
        index_kv_cache=case.index_kv_cache,
        index_state_cache=case.index_state_cache,
        index_state_block_table=case.index_state_block_table,
        kv_lens=over,
        block_table=case.block_table,
        block_ids_out=case.block_ids_out,
        block_lens_out=case.block_lens_out,
        current_stream=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    rep = v["seqrep"].tolist()
    assert rep == [SEQREP_OVER, SEQREP_DEAD, 1000 // BS], rep
    nb_max = ix.n_blocks_max
    assert int(v["qpos"][0]) == BS * nb_max + BS - 1 - t, "the over sequence's q_pos0 is not the clamp"  # one piece per sequence (one window)
    assert int(v["counts"].view(3, t)[0, -1]) == nb_max  # the TRUE positions clamped at n_blocks_max
    assert bool((buf[n:] == 0x7F).all()), "a write past get_workspace_size()"
    # the sentinel past every piece's used slab: piece p of the window owns [offs[p], offs[p] + rows_p x NB) floats and uses G(q0 + rows) - G(q0) of them
    offs = [0] + [(k + 1) * t * nb_max for k in range(3)]
    assert ix.n_windows == 1 and ix.piece_table[: len(ix.pieces)] == ix.pieces  # one window: the real pieces lead the table, the padding after
    for p, (b, r0, rows) in enumerate(ix.pieces):
        q0 = int(v["qpos"][p])
        used = sum(((q0 + i + 1) // BS) for i in range(rows))
        assert used <= rows * nb_max
        assert bool((v["cand"][offs[p] + used : offs[p + 1]] == -7.0).all()), f"piece {p}: a scorer write past its used compact slab ({used} floats)"
    cs = v["cslot"][0].tolist()
    assert any(x == -2 for x in cs) and all(x in (-1, -2) or 0 <= x < nb_max for x in cs), cs  # a block past the table reports -2 and writes nothing
    case.restore()


@requires_rubin
def test_extend_workspace_is_honest_and_execute_allocates_nothing_and_never_synchronizes():
    """``get_workspace_size()`` honest with the window scratch inside (a 4 KiB sentinel past it untouched by three executes), every slot 16-byte aligned;
    after the first execute (the plan constants' upload) three executes under ``set_sync_debug_mode("error")`` make NO CUDA allocation and no host sync
    (Rules 1 / 3), on the bf16 and the e4m3 cache; the stage's launch counter = ``4 + 2 x n_windows`` per execute."""
    for cache_dtype in (None, _E4M3):
        case = _IndexerExtendCase(GEOM_SMALL, 128, (4096 + 128, 300 + 128, 0), page=16, cap=4, seed=241, return_lse=True, index_cache_dtype=cache_dtype)
        case.compile()
        n = case.blk.get_workspace_size()
        buf = _sentinel_past(case.ws, n)
        case.ws = buf[:n]
        lay = case.blk._ws
        for nm in ("ix_q", "ix_cslot", "ix_kbar", "ix_qpos", "ix_counts", "ix_seqrep", "ix_ids", "ix_scores", "ix_cand"):
            assert getattr(lay, nm) % 16 == 0 and 0 <= getattr(lay, nm) < n, nm
        assert lay.ix_cand + case.blk._indexer_extend.window_rows * case.blk._indexer_extend.n_blocks_max * 4 <= n
        case.restore()
        case.run()  # the first execute: the constants' upload
        launches0 = case.blk._indexer_extend.launches
        torch.cuda.synchronize()
        stat = lambda: torch.cuda.memory_stats()["allocation.all.allocated"]  # noqa: E731
        n0 = stat()
        prev = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(3):
                case.restore()
                case.execute()
        finally:
            torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert stat() - n0 == 0, f"the extend block made {stat() - n0} CUDA allocation(s) over three executes (cache {cache_dtype})"
        assert case.blk._indexer_extend.launches - launches0 == 3 * case.blk._indexer_extend.launches_per_execute() == 3 * 6
        assert bool((buf[n:] == 0x7F).all()), "a write past get_workspace_size()"
        case.check_selection("workspace")
        case.check(what="workspace")


@requires_rubin
def test_extend_launch_census_by_kernel_name_and_the_window_scaling():
    """The CUPTI census by kernel NAME on the 24 / 2 geometry against the caller-list band block on the same shapes (the CONTROL): the extend block =
    the control + 6 launches at one select window -- the pool-1 ``frost_qsa_compress``, the chunk form ``frost_qsa_compress_step``, the gather
    ``cudnn_block_qsa_kbar_gather``, the scorer + top-k pair, the canonical list ``cudnn_block_qsa_canonical_list``; the control's raw-key
    ``frost_cache_write`` replaced by the RING write (two cache writes each) -- and + 2 per further window at a lowered budget (two windows); no
    memset / memcpy on either path."""
    from collections import Counter

    t, kv_lens = 20, (4116, 320, 0, 2071)
    ctl = _qsa._Case(GEOM_FLASH_NEXT, t, kv_lens, page=16, seed=251, split=4, fuse_gate=True, return_lse=True, index_band=True, index_list_form="per_row")
    ctl.compile()
    ctl.run()
    ext = _IndexerExtendCase(GEOM_FLASH_NEXT, t, kv_lens, page=16, cap=4, seed=251, split=4, fuse_gate=True, return_lse=True)
    ext.compile()
    _two_launches_extend(ext, "census (one window)")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_api, "QSA_EXTEND_SELECT_BUDGET_BYTES", 40 * ext.blk._indexer_extend.n_blocks_max * 4)  # 40 rows per window: 80 rows -> two windows
        two = _IndexerExtendCase(GEOM_FLASH_NEXT, t, kv_lens, page=16, cap=4, seed=251, split=4, fuse_gate=True, return_lse=True)
    assert two.blk._indexer_extend.n_windows == 2 and two.blk._indexer_extend.window_rows == 40 and len(two.blk._indexer_extend.pieces) == 4
    assert two.blk._indexer_extend.n_pieces == 8 and two.blk._indexer_extend.windows[1][2] == 4  # the second window's q_pos0 slice starts 16-byte aligned
    two.compile()
    _two_launches_extend(two, "census (two windows)")
    assert torch.equal(two.kernel_ids(), ext.kernel_ids()) and torch.equal(_bits(two.out), _bits(ext.out)), "the window count changed the function"

    # the census of the block's OWN launches: the side caches restored OUTSIDE the profiled call (the restore is two DtoD copies of the test's,
    # not the block's); the helper's warm + measured executes then run on the post-run ring -- the launch list is data-independent
    ext.restore()
    two.restore()
    torch.cuda.synchronize()
    k_ctl, m_ctl = _cupti_kernels(lambda: ctl.execute())
    k_ext, m_ext = _cupti_kernels(lambda: ext.execute())
    k_two, m_two = _cupti_kernels(lambda: two.execute())
    if k_ctl is None or k_ext is None or k_two is None:
        pytest.skip("torch.profiler recorded no CUDA events (CUPTI unavailable on this node); the bitwise cells stand")
    for nm, k, m in (("control (caller-list band)", k_ctl, m_ctl), ("extend (one window)", k_ext, m_ext), ("extend (two windows)", k_two, m_two)):
        print(f"\n{nm}: {len(k)} kernels + {len(m)} memset/memcpy:\n  " + "\n  ".join(x[:140] for x in k))
    assert not (m_ctl or m_ext or m_two), "a memset / memcpy on the execute path"
    has = lambda names, tok: sum(tok in n for n in names)  # noqa: E731
    for names in (k_ext, k_two):
        assert has(names, "frost_qsa_compress_step") == 1 and has(names, "frost_qsa_compress") == 2, names
        assert has(names, "qsa_kbar_gather") == 1 and has(names, "qsa_canonical_list") == 1 and has(names, "frost_cache_write") == 2, names
        assert has(names, "qsa_score_paged") == 0 and has(names, "qsa_topk_decode") == 0 and has(names, "frost_qsa_prepare") == 0, names
        assert not any("nvjet" in n.lower() or "cublas" in n.lower() for n in names), "the slab left the FROST route"
    assert has(k_ctl, "frost_cache_write") == 2 and has(k_ctl, "frost_qsa_compress") == 0 and has(k_ctl, "qsa_kbar_gather") == 0
    assert len(k_ext) - len(k_ctl) == 6, (len(k_ext), len(k_ctl))
    assert len(k_two) - len(k_ext) == 2, (len(k_two), len(k_ext))
    extra = Counter(k_two) - Counter(k_ext)
    assert sum(extra.values()) == 2 and set(extra) <= set(
        Counter(k_ext) - Counter(k_ctl)
    ), extra  # the second window's scorer + top-k pair: kernels the one-window block runs once


@requires_rubin
def test_extend_capture_after_one_warm_up_replays_bitwise_and_takes_a_new_chunk():
    """CUDA-graph capture of the extend block AFTER one eager warm-up ON THE CAPTURE STREAM (the first execute uploads the window plan's constants
    and creates the GEMM route's per-stream handle, which a capture refuses to create; the capture takes the SECOND execute under
    ``set_sync_debug_mode("error")``): the capture launches nothing, the replay is BITWISE the eager run (``out``, the LSE, the
    list, the cache, the ring); a NEW chunk's tensors (another seed and prefix lengths over the same shapes) through the captured buffers replay
    bitwise the eager execute over them and pass the sibling's oracle."""
    t, kv_lens = 24, (321, 0, 2075)
    case = _IndexerExtendCase(GEOM_SMALL, t, kv_lens, page=16, cap=4, seed=261, fuse_gate=True, return_lse=True)
    case.compile()
    case.restore()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        case.run()  # the eager warm-up ON THE CAPTURE STREAM: the plan constants and the GEMM route's per-stream handle land there
    eager_out, eager_lse, eager_ids = case.out.clone(), case.lse.clone(), case.kernel_ids().clone()
    eager_cache, eager_ring = _bits(case.index_kv_cache).clone(), _bits(case.index_state_cache).clone()
    case.restore()
    case.out.fill_(float("nan"))
    case.lse.fill_(float("nan"))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                case.execute()
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(case.out).all(), "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        assert (
            torch.equal(_bits(case.out), _bits(eager_out)) and torch.equal(case.lse, eager_lse) and torch.equal(case.kernel_ids(), eager_ids)
        ), "the replay differs from the eager run"
        assert torch.equal(_bits(case.index_kv_cache), eager_cache) and torch.equal(
            _bits(case.index_state_cache), eager_ring
        ), "the replay's caches differ from the eager run's"
        case.check_selection("replay")
        case.check(what="replay")
        # a NEW chunk through the captured tensors: another seed, other prefix lengths over the same shapes -- the oracle of the new chunk through a sibling case
        sib = _IndexerExtendCase(
            GEOM_SMALL, t, (322, 0, 2074), page=16, cap=4, seed=263, fuse_gate=True, return_lse=True, n_pages=case.n_pages, max_pages=case.max_pages
        )
        sib.prime_prefix()
        for nm in (
            "k_cache",
            "v_cache",
            "block_table",
            "kv_lens_t",
            "slot_mapping",
            "ring_slots",
            "h",
            "cos",
            "sin",
            "index_kv_cache",
            "index_state_cache",
            "index_state_block_table",
        ):
            getattr(case, nm).copy_(getattr(sib, nm))
        for nm in ("w_qkvg", "w_o", "w_iq_norm", "w_ik_norm"):
            case.w[nm].copy_(sib.w[nm])
        case.kv_lens, case.pages, case.refs, case.ref_rows, case.k_raw_full, case.live_rows = (
            sib.kv_lens,
            sib.pages,
            sib.refs,
            sib.ref_rows,
            sib.k_raw_full,
            sib.live_rows,
        )
        case.prefix_bits = (_bits(case.k_cache).clone(), _bits(case.v_cache).clone())
        case.snapshot()
        case.out.fill_(float("nan"))
        case.lse.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        case.oracle_from_list()
        rep_out, rep_lse, rep_ids = case.out.clone(), case.lse.clone(), case.kernel_ids().clone()
        rep_cache, rep_ring = _bits(case.index_kv_cache).clone(), _bits(case.index_state_cache).clone()
        case.check_selection("replay over a new chunk")
        case.check(what="replay over a new chunk")
        case.restore()
        case.out.fill_(float("nan"))
        case.lse.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert (
            torch.equal(_bits(case.out), _bits(rep_out)) and torch.equal(case.lse, rep_lse) and torch.equal(case.kernel_ids(), rep_ids)
        ), "the replay over a new chunk differs from the eager run over it"
        assert torch.equal(_bits(case.index_kv_cache), rep_cache) and torch.equal(_bits(case.index_state_cache), rep_ring)
    finally:
        graph.reset()
