# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fused PREPARE launch of a block-sparse attention step (``kernels/qsa_prepare.py``) and the block knob ``fuse_prepare``.

The kernel replaces, as ONE launch, the chain of pointwise passes between the projection GEMM and attention: norm + RoPE of
Q / K (``qk_norm_rope``), the paged write-through of the post-RoPE K / V and of the raw indexer key (``cache_write``, two
launches), the indexer queries' norm + RoPE (``qsa_compress`` at pool 1) and the raw-key ring + incremental compress
(``qsa_compress_step``).  The acceptance is BITWISE, output by output, against those kernels run on the same inputs -- the
slab's Q / K bands, the V / GATE / INDEX bands untouched, the three pools (padded and out-of-range slots unwritten), the
indexer queries, the ring, the compressed-key cache and the per-row slot report -- over bf16 / f16, int32 / int64 slots,
HND / NHD pools, d256 and d128 head dims (one and two attention rows per warp), decode (1 / 4 / 5 rows per sequence, partial
commit prefixes, a finished sequence, a capacity overflow) and prefill shapes, every arm subset the block or the serving
step binds, and the RoPE-only form.  The launch census (``torch.profiler``, CUPTI) measures the step's launches: five
unfused, one fused.  Plain vectorized LDG / STG plus warp shuffles -- no tcgen05, no TMA -- so these cells run on whatever
CUDA device is at hand.

The block knob ``fuse_prepare`` (a performance knob: the same function): its declaration rows on the host, the stage list
(one ``qsa_prepare`` stage in place of ``qk_norm_rope`` + ``cache_write``, the workspace carve unchanged) and, on Rubin,
the block with the knob on BITWISE the block with it off -- ``out``, the LSE, the pools, the raw-key pool -- on the dense
block and on the block-sparse band block at a decode shape and a prefill shape, with the block's launch census one (two
with the band) shorter.

The PAGED compress arm (the serving stack's ring and compressed-key cache through their block tables, positions from
``kv_lens``), the ring's slot-mapping form and the RoPE-table arm: the fused launch BITWISE the step form's paged arms fed
the same tokens (two call sites of ONE addressing helper per table), the slot form bitwise the derived form, the table arm
bitwise the per-token form, a padding sequence (``kv_lens 0``) writing nothing, a live sequence whose state entry is the null
block or lies past the ring owning no ring in BOTH kernels (no ring row anywhere, a ring-fed block ``-2``), the table arm's host
bound (a table of exactly the addressable positions served, one row short refused by form), the table arm's IN-KERNEL ceiling on
the live positions (a row at or past the table: ``-2`` in both kernels, its Q / K / K-pool / indexer-query rows, its ring row and
its block unwritten, the neighbour below the ceiling bitwise the unfused chain), the refusals in both directions, and the decode
form's census -- four unfused launches, one fused.

The e4m3 STORE arm (``compile_qsa_prepare(index_cache_dtype=torch.float8_e4m3fn)``; sm_89+ -- the cells skip below it and are UNCOVERED
there): the fused launch's indexer queries and paged compressed-key rows BITWISE the unfused chain's (the pool-1 compress and the step
form under the same arm: two call sites of one store helper), and BITWISE the bf16 chain's rows ``.to(float8_e4m3fn)`` on the same inputs
(the e4m3 cast of the bf16 row); the ring in bf16; one launch; the arm composes with the ring-slot and the RoPE-table arms (orthogonal
compile keys); the refusals both ways naming the recipe's dtype; f16 io has no e4m3 arm.

The rounding attribute (``QsaPrepareIndexArms.rounding``; 2026-10-29): the fused launch under ``rounding="hf"`` BITWISE its unfused
twins under the same value (the pool-1 compress and the step form with ``rounding="hf"``: the query arm and the compress arm are two of
the chain's four sites), dense and paged; a distinct compile key; an unknown value refused naming the tuple.  The chunk form's ring TAIL
write (the extend step's second cache-write launch) is the cache-write kernel over the ring's circular slots -- pinned here, where the
cache-write kernel's cells live, with no kernel change.
"""

import os
import re
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry, QsaSpec, index_k_raw_view  # noqa: E402
from cudnn.gated_attention_block.api import _cols, _QsaPrepare, _view  # noqa: E402
from cudnn.gated_attention_block.kernels.cache_write import compile_cache_write, run_cache_write  # noqa: E402
from cudnn.gated_attention_block.kernels.qk_norm_rope import compile_qk_norm_rope, run_qk_norm_rope  # noqa: E402
from cudnn.gated_attention_block.kernels.qsa_compress import (
    compile_qsa_compress,
    compile_qsa_compress_step,
    min_ring_rows,
    run_qsa_compress,
    run_qsa_compress_step,
)  # noqa: E402
from cudnn.gated_attention_block.kernels.qsa_prepare import (  # noqa: E402
    QsaPrepareIndexArms,
    compile_qsa_prepare,
    moved_bytes,
    prepare_warp_counts,
    run_qsa_prepare,
    validate_prepare_shape,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_qsa_reference import full_block_ids  # noqa: E402
from gated_block_reference import RefGeometry, make_inputs  # noqa: E402

pytestmark = pytest.mark.L0

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107
_CC = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
requires_sm89 = pytest.mark.skipif(_CC is None or _CC < (8, 9), reason=f"the e4m3 store arm needs sm_89+ (found {_CC})")
_E4M3 = torch.float8_e4m3fn

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
BF16, F16 = torch.bfloat16, torch.float16
_EPS = 1e-6
_POOL = 4
_PAGE = 16
# The block-sparse model's QSA layer at TP 1 (with its indexer band), a shrunk d256 geometry, and a d128 geometry (two
# attention rows per warp -- the sub-row path of the attention arm).
_GEOM_FN = dict(h_q=24, h_kv=2, d=256, rope=64)
_GEOM_SMALL = dict(h_q=8, h_kv=2, d=256, rope=64)
_GEOM_D128 = dict(h_q=4, h_kv=2, d=128, rope=64)
_INDEX = dict(heads=4, head_dim=128)
_SMALL_D256 = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)
_FLASH_NEXT = dict(d_model=2560, h_q=24, h_kv=2, d_head=256, rope_dim=64)


def _bits(x: torch.Tensor) -> torch.Tensor:
    """The bit patterns of a bf16 / f16 (16-bit) or an e4m3 (8-bit) tensor (a NaN-safe bitwise comparison: the pool sentinel is NaN)."""
    return x.contiguous().view(torch.int8 if x.element_size() == 1 else torch.int16)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype in (BF16, F16, _E4M3):
        return a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))
    return torch.equal(a, b)


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _launch_names(*runs):
    """The CUDA kernel launches per callable under ``torch.profiler`` (memset / memcpy excluded), or ``None`` when the
    profiler records no CUDA activity here (CUPTI absent -- the ONLY failure swallowed).  The caller skips EXPLICITLY on
    ``None`` and asserts on the names OUTSIDE any handler, so a wrong census fails the test instead of printing 'unverified'."""
    from torch.profiler import ProfilerActivity, profile

    names = []
    for run in runs:
        prof = profile(activities=[ProfilerActivity.CUDA])
        try:
            prof.start()
        except Exception:  # noqa: BLE001 -- CUPTI unavailable on this box: the bitwise pins stand on their own
            return None
        try:
            run()
            torch.cuda.synchronize()
        finally:
            prof.stop()
        names.append(
            [
                e.name
                for e in prof.events()
                if e.device_type == torch.autograd.DeviceType.CUDA and "memset" not in e.name.lower() and "memcpy" not in e.name.lower()
            ]
        )
    return names if names and names[0] else None


# ---------------------------------------------------------------------------
# The chain under test: one slab, the unfused kernels on copy A, the fused kernel on copy B
# ---------------------------------------------------------------------------


class _Chain:
    """A decode (or prefill) step's operands at a geometry: the ``[T, N]`` slab with its five bands, the pools, the
    slot mapping (every 5th token padded, one slot past the pool), the per-token and the position-indexed RoPE tables,
    the ring state and the compressed-key cache; ``run_unfused`` / ``run_fused`` write into independent copies.  ``n_pos`` fixes
    the positions the cache and the table hold (default: every position of the step + 8)."""

    def __init__(
        self,
        geom,
        *,
        batch,
        rows,
        dtype=BF16,
        slot_dtype=torch.int32,
        layout="hnd",
        pos0=None,
        n_commit=None,
        arms=None,
        apply_norm=True,
        band=True,
        seed=1234,
        pad_every=5,
        paged=False,
        page=_PAGE,
        cap=None,
        ring_slots=False,
        rope_table=False,
        padding=(),
        n_pos=None,
        index_cache_dtype=None,
        rounding="single",
    ):
        dev = "cuda"
        self.dtype, self.slot_dtype, self.layout, self.apply_norm, self.band = dtype, slot_dtype, layout, apply_norm, band
        self.rounding = arms.rounding if arms is not None else rounding  # the chain's rounding points: the arms record's, the unfused twins'
        self.cache_dtype = (
            dtype if index_cache_dtype is None else index_cache_dtype
        )  # the indexer queries' / compressed-key cache's dtype (e4m3 = the e4m3 store arm)
        self.paged, self.page, self.ring_slots, self.rope_table, self.padding = paged, page, ring_slots, rope_table, tuple(padding)
        self.h_q, self.h_kv, self.d, self.rope = geom["h_q"], geom["h_kv"], geom["d"], geom["rope"]
        self.h_i, self.d_i = _INDEX["heads"], _INDEX["head_dim"]
        self.B, self.S_q = batch, rows
        self.T = batch * rows
        h_q, h_kv, d, h_i, d_i, T = self.h_q, self.h_kv, self.d, self.h_i, self.d_i, self.T
        self.n = (2 * h_q + 2 * h_kv) * d + ((h_i + 1) * d_i if band else 0)
        self.o_q, self.o_g, self.o_k, self.o_v = 0, h_q * d, 2 * h_q * d, 2 * h_q * d + h_kv * d
        self.o_iq, self.o_ik = (2 * h_q + 2 * h_kv) * d, (2 * h_q + 2 * h_kv) * d + h_i * d_i
        self.arms = (
            arms
            if arms is not None
            else (
                QsaPrepareIndexArms(
                    heads=h_i,
                    head_dim=d_i,
                    raw_pool=True,
                    query=True,
                    compress=True,
                    n_commit=n_commit is not None,
                    paged=paged,
                    ring_slots=ring_slots,
                    rope_table=rope_table,
                    rounding=rounding,
                )
                if band
                else None
            )
        )
        gen = torch.Generator(device=dev).manual_seed(seed)

        def rnd(*shape, std=1.0):
            return (torch.randn(*shape, generator=gen, device=dev, dtype=torch.float32) * std).to(dtype)

        self.slab0 = rnd(T, self.n)
        self.w_q, self.w_k = (1.0 + 0.05 * rnd(d).float()).to(dtype), (1.0 - 0.05 * rnd(d).float()).to(dtype)
        self.w_iq, self.w_ik = (1.0 + 0.03 * rnd(d_i).float()).to(dtype), (1.0 + 0.02 * rnd(d_i).float()).to(dtype)
        pos0_list = list(pos0) if pos0 is not None else [0] * batch
        self.pos0 = torch.tensor(pos0_list, dtype=torch.int32, device=dev)
        # the paged form derives pos0 = kv_lens - S_q in-kernel: kv_lens = the KV length AFTER the step; a padding sequence has 0
        self.kv_lens = torch.tensor([0 if b in self.padding else p + rows for b, p in enumerate(pos0_list)], dtype=torch.int32, device=dev)
        self.n_commit = None if n_commit is None else torch.tensor(list(n_commit), dtype=torch.int32, device=dev)
        # the position table holds EXACTLY the positions the compress arm's cache can address -- NB x pool on the dense form,
        # max_pages x P on the paged one -- which is the RoPE-table arm's host bound, met with equality (the accept cell at the bound)
        s_need = int(self.pos0.max().item()) + rows + 8 if n_pos is None else int(n_pos)
        self.nb_cap = (s_need + _POOL - 1) // _POOL
        self.rpp = page // _POOL
        self.n_lpages = (self.nb_cap + self.rpp - 1) // self.rpp
        s_pos = self.n_lpages * page if paged else self.nb_cap * _POOL
        self.s_pos = s_pos
        inv = 1.0 / (1e7 ** (torch.arange(0, self.rope, 2, device=dev, dtype=torch.float32) / self.rope))
        emb = torch.cat([torch.arange(s_pos, device=dev, dtype=torch.float32)[:, None] * inv] * 2, -1)
        self.cos_pos, self.sin_pos = emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]  # [1, S_pos, rope]: every sequence
        # the per-token rows at the positions the kernels derive (paged: kv_lens - S_q + j, a padding sequence's rows clamp to 0 --
        # the RoPE-table arm's in-bounds clamp, so the table arm and the per-token form rotate every row identically)
        eff_pos0 = (self.kv_lens.long() - rows) if paged else self.pos0.long()
        tok_pos = (eff_pos0[:, None] + torch.arange(rows, device=dev)[None, :]).reshape(T).clamp_(0, s_pos - 1)
        self.cos_t, self.sin_t = self.cos_pos[0, tok_pos].contiguous(), self.sin_pos[0, tok_pos].contiguous()  # [T, rope] per token
        self.n_pages = T // _PAGE + 3
        self.slot = torch.randperm(self.n_pages * _PAGE, generator=gen, device=dev)[:T].to(slot_dtype)
        if pad_every:
            self.slot[::pad_every] = -1
        if T > 2:
            self.slot[2] = self.n_pages * _PAGE + 7  # past the pool: writes nothing
        self.live = (self.slot >= 0) & (self.slot < self.n_pages * _PAGE)
        self.ring_n = max(8, min_ring_rows(rows, _POOL)) if cap is None else cap
        self.sent = (
            torch.full((), -448.0 if self.cache_dtype == _E4M3 else -7777.0, dtype=self.cache_dtype).float().item()
        )  # the CACHE's sentinel AS ITS DTYPE HOLDS IT (e4m3: -448, its max magnitude -- torch.full refuses -7777 there)
        if paged:
            # the stack's two side caches: ring blocks 1..B (block 0 null) of `cap` rows; a compressed-key cache of P / 4 rows per page
            # behind a PERMUTED block table (page 0 unused by a live sequence); the padding sequences sit on block 0 / page 0
            n_lpages = self.n_lpages
            self.n_kbar_pages = 1 + batch * n_lpages  # the compressed-key cache's pages (self.n_pages is the K / V pools')
            perm = torch.randperm(self.n_kbar_pages - 1, generator=gen, device=dev) + 1
            self.block_table = perm.view(batch, n_lpages).to(torch.int32).contiguous()
            self.state_table = torch.arange(1, batch + 1, device=dev, dtype=torch.int32).view(batch, 1)
            for b in self.padding:
                self.block_table[b] = 0
                self.state_table[b] = 0
            self.ring0 = rnd(batch + 1, self.ring_n, 1, d_i)
            self.comp0 = torch.full((self.n_kbar_pages, self.rpp, 1, d_i), self.sent, device=dev, dtype=self.cache_dtype)
            pos = eff_pos0[:, None] + torch.arange(rows, device=dev)[None, :]
            live = (pos >= 0) & (self.state_table.long() > 0)
            self.ring_slots_t = torch.where(live, self.state_table.long() * self.ring_n + pos % self.ring_n, torch.full_like(pos, -1)).reshape(T).to(slot_dtype)
        else:
            self.ring0 = rnd(batch, self.ring_n, d_i)
            self.comp0 = torch.full((batch, self.nb_cap, d_i), self.sent, device=dev, dtype=self.cache_dtype)
        self.A = self._fresh()
        self.Bf = self._fresh()

    def kbar_row(self, X, b, blk):
        """Block ``blk`` of sequence ``b`` in copy ``X``'s compressed-key cache, dense or through the block table."""
        if self.paged:
            return X["comp"][int(self.block_table[b, blk // self.rpp]), blk % self.rpp, 0]
        return X["comp"][b, blk]

    def _pool(self, h, d):
        if self.layout == "hnd":
            return torch.full((self.n_pages, h, _PAGE, d), float("nan"), device="cuda", dtype=self.dtype)
        raw = torch.full((self.n_pages, _PAGE, h, d), float("nan"), device="cuda", dtype=self.dtype)
        return raw.permute(0, 2, 1, 3)

    def _fresh(self) -> dict:
        return dict(
            slab=self.slab0.clone(),
            k_pool=self._pool(self.h_kv, self.d),
            v_pool=self._pool(self.h_kv, self.d),
            i_pool=torch.full((self.n_pages, _PAGE, self.d_i), float("nan"), device="cuda", dtype=self.dtype),
            qi=torch.full((self.T, self.h_i, self.d_i), float("nan"), device="cuda", dtype=self.cache_dtype),
            ring=self.ring0.clone(),
            comp=self.comp0.clone(),
            cslot=torch.full((self.B, self.S_q), -9, dtype=torch.int32, device="cuda"),
        )

    def cols(self, slab, off, h, d):
        return torch.as_strided(slab, (self.T, h, d), (self.n, d, 1), slab.storage_offset() + off)

    # -- the unfused chain: the kernels the fused launch replaces, in the block's / the serving step's order ---------------
    def compile_unfused(self):
        dt, sd = self.dtype, self.slot_dtype
        self.r_norm = compile_qk_norm_rope(
            dtype=dt, h_q=self.h_q, h_kv=self.h_kv, d=self.d, rope_dim=self.rope, eps=_EPS, want_rstd=False, apply_norm=self.apply_norm
        )
        self.r_cw_kv = compile_cache_write(dtype=dt, h=self.h_kv, d=self.d, page_size=_PAGE, slot_dtype=sd, two_operands=True)
        a = self.arms
        self.r_cw_ix = (
            compile_cache_write(dtype=dt, h=1, d=self.d_i, page_size=_PAGE, slot_dtype=sd, two_operands=False) if (a is not None and a.raw_pool) else None
        )
        self.r_q1 = (
            compile_qsa_compress(
                dtype=dt, d=self.d_i, rope_dim=self.rope, pool=1, has_seq_lens=False, table_dtype=dt, out_dtype=self.cache_dtype, rounding=self.rounding
            )
            if (a is not None and a.query)
            else None
        )
        self.r_step = (
            compile_qsa_compress_step(
                dtype=dt,
                d=self.d_i,
                rope_dim=self.rope,
                pool=_POOL,
                has_n_commit=a.n_commit,
                table_dtype=dt,
                paged=self.paged,
                ring_slots=self.ring_slots,
                ring_slot_dtype=sd,
                index_cache_dtype=self.cache_dtype,
                rounding=self.rounding,
            )
            if (a is not None and a.compress)
            else None
        )

    def run_unfused(self):
        s, X, st = self.A["slab"], self.A, _stream()
        q, k, v = self.cols(s, self.o_q, self.h_q, self.d), self.cols(s, self.o_k, self.h_kv, self.d), self.cols(s, self.o_v, self.h_kv, self.d)
        wq, wk = (self.w_q, self.w_k) if self.apply_norm else (None, None)
        run_qk_norm_rope(self.r_norm, q, k, q, k, wq, wk, self.cos_t, self.sin_t, stream=st)  # (2)+(3) in place
        run_cache_write(self.r_cw_kv, k, v, X["k_pool"], X["v_pool"], self.slot, stream=st)  # (4w) K / V
        if self.r_cw_ix is not None:  # (4w) the raw indexer key
            run_cache_write(self.r_cw_ix, self.cols(s, self.o_ik, 1, self.d_i), None, X["i_pool"].unsqueeze(1), None, self.slot, stream=st)
        if self.r_q1 is not None:  # the indexer queries: pool 1, batch = token, block = head, the tables broadcast over the heads
            cq, sq = self.cos_t.view(self.T, 1, self.rope).expand(self.T, self.h_i, self.rope), self.sin_t.view(self.T, 1, self.rope).expand(
                self.T, self.h_i, self.rope
            )
            run_qsa_compress(
                self.r_q1, self.cols(s, self.o_iq, self.h_i, self.d_i), X["qi"], self.w_iq, cq, sq, None, eps=_EPS, norm_weight_offset=0.0, stream=st
            )
        if self.r_step is not None:  # the ring + the block each committed row completes (the paged tables when the chain is paged)
            k_new = torch.as_strided(s, (self.B, self.S_q, self.d_i), (self.S_q * self.n, self.n, 1), s.storage_offset() + self.o_ik)
            run_qsa_compress_step(
                self.r_step,
                k_new,
                X["ring"],
                X["comp"],
                self.w_ik,
                self.cos_pos,
                self.sin_pos,
                None if self.paged else self.pos0,
                X["cslot"],
                self.n_commit,
                eps=_EPS,
                norm_weight_offset=0.0,
                stream=st,
                **self.paged_kwargs(),
            )

    def paged_kwargs(self) -> dict:
        """The paged form's position source and tables (the step form and the fused launch share the keyword names)."""
        if not self.paged:
            return {}
        kw = dict(kv_lens=self.kv_lens, block_table=self.block_table, index_state_block_table=self.state_table)
        if self.ring_slots:
            kw["index_state_slot_mapping"] = self.ring_slots_t
        return kw

    # -- the fused launch -------------------------------------------------------------------------------------------
    def compile_fused(self):
        self.r_fused = compile_qsa_prepare(
            dtype=self.dtype,
            h_q=self.h_q,
            h_kv=self.h_kv,
            d=self.d,
            rope_dim=self.rope,
            eps=_EPS,
            page_size=_PAGE,
            slot_dtype=self.slot_dtype,
            apply_norm=self.apply_norm,
            index=self.arms,
            index_cache_dtype=self.cache_dtype,
        )

    def fused_kwargs(self, X) -> dict:
        s, a = X["slab"], self.arms
        kw = {}
        if a is not None:
            if a.raw:
                kw["index_k_raw_src"] = self.cols(s, self.o_ik, 1, self.d_i)
            if a.raw_pool:
                kw["index_k_raw_pool"] = X["i_pool"]
            if a.query:
                kw.update(index_q_src=self.cols(s, self.o_iq, self.h_i, self.d_i), index_q_out=X["qi"], w_iq_norm=self.w_iq)
            if a.compress:
                kw.update(
                    ring=X["ring"],
                    index_k_compressed=X["comp"],
                    w_ik_norm=self.w_ik,
                    cos_pos=self.cos_pos,
                    sin_pos=self.sin_pos,
                    pos0=None if self.paged else self.pos0,
                    compressed_slot=X["cslot"],
                    seq_q=self.S_q,
                    n_commit=self.n_commit,
                    **self.paged_kwargs(),
                )
        return kw

    def run_fused(self, X=None):
        X = self.Bf if X is None else X
        s = X["slab"]
        wq, wk = (self.w_q, self.w_k) if self.apply_norm else (None, None)
        cos_t, sin_t = (None, None) if self.rope_table else (self.cos_t, self.sin_t)  # the RoPE-table arm reads cos_pos / sin_pos at pos0 + j
        run_qsa_prepare(
            self.r_fused,
            self.cols(s, self.o_q, self.h_q, self.d),
            self.cols(s, self.o_k, self.h_kv, self.d),
            self.cols(s, self.o_v, self.h_kv, self.d),
            None,
            None,
            wq,
            wk,
            cos_t,
            sin_t,
            X["k_pool"],
            X["v_pool"],
            self.slot,
            index_eps=_EPS,
            index_norm_weight_offset=0.0,
            stream=_stream(),
            **self.fused_kwargs(X),
        )

    # -- the comparison, output by output -------------------------------------------------------------------------
    def assert_bitwise(self, label: str):
        A, Bf, a = self.A, self.Bf, self.arms
        torch.cuda.synchronize()
        checks = [
            ("Q band", self.cols(A["slab"], self.o_q, self.h_q, self.d), self.cols(Bf["slab"], self.o_q, self.h_q, self.d)),
            ("K band", self.cols(A["slab"], self.o_k, self.h_kv, self.d), self.cols(Bf["slab"], self.o_k, self.h_kv, self.d)),
            ("the whole slab", A["slab"], Bf["slab"]),
            ("V band untouched", self.cols(Bf["slab"], self.o_v, self.h_kv, self.d), self.cols(self.slab0, self.o_v, self.h_kv, self.d)),
            ("GATE band untouched", self.cols(Bf["slab"], self.o_g, self.h_q, self.d), self.cols(self.slab0, self.o_g, self.h_q, self.d)),
            ("K pool", A["k_pool"].contiguous(), Bf["k_pool"].contiguous()),
            ("V pool", A["v_pool"].contiguous(), Bf["v_pool"].contiguous()),
        ]
        if self.band:
            checks.append(
                ("INDEX band untouched", self.cols(Bf["slab"], self.o_iq, self.h_i + 1, self.d_i), self.cols(self.slab0, self.o_iq, self.h_i + 1, self.d_i))
            )
        if a is not None and a.raw_pool:
            checks.append(("raw-key pool", A["i_pool"], Bf["i_pool"]))
        if a is not None and a.query:
            checks.append(("indexer queries", A["qi"], Bf["qi"]))
        if a is not None and a.compress:
            checks += [("ring", A["ring"], Bf["ring"]), ("compressed keys", A["comp"], Bf["comp"]), ("compressed_slot", A["cslot"], Bf["cslot"])]
        bad = [nm for nm, x, y in checks if not _same(x, y)]
        assert not bad, f"{label}: fused differs from the unfused chain on {bad}"
        # The pools: exactly the live slots' rows written (padded / out-of-range slots untouched), the K pool = the K band.
        n_live = int(self.live.sum())
        for pool in (Bf["k_pool"], Bf["v_pool"]):
            assert int((~torch.isnan(pool.float())).all(-1).sum()) == n_live * self.h_kv, f"{label}: pool rows written != live slots x heads"
        s_ = self.slot[self.live].long()
        k_rows = self.cols(Bf["slab"], self.o_k, self.h_kv, self.d)[self.live]
        assert _same(Bf["k_pool"][s_ // _PAGE, :, s_ % _PAGE, :], k_rows), f"{label}: the K pool is not the post-RoPE K band"
        if a is not None and a.raw_pool:
            assert int((~torch.isnan(Bf["i_pool"].float())).all(-1).sum()) == n_live
            assert _same(Bf["i_pool"][s_ // _PAGE, s_ % _PAGE, :], self.cols(self.slab0, self.o_ik, 1, self.d_i)[self.live, 0])
        if a is not None and a.compress:
            n_written = int((Bf["comp"].float() != self.sent).any(-1).sum())
            assert n_written == int((Bf["cslot"] >= 0).sum()), f"{label}: compressed rows written {n_written} != the rows the slot report names"
        print(f"{label}: {len(checks)} outputs bitwise; {n_live} / {self.T} live slots")


def _chain(geom, **kw) -> _Chain:
    ch = _Chain(geom, **kw)
    ch.compile_unfused()
    ch.compile_fused()
    ch.run_unfused()
    ch.run_fused()
    return ch


# ---------------------------------------------------------------------------
# Host rows: the warp arithmetic, the validators, the compile keys, the byte count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [1, 3, 16, 97, 2048])
@pytest.mark.parametrize("geom", [_GEOM_FN, _GEOM_SMALL, _GEOM_D128], ids=["fn", "small", "d128"])
def test_prepare_warp_counts_cover_every_row_of_every_arm(t, geom):
    """Each arm's warp count covers its rows at ``32 // lanes_per_row`` rows per warp with less than one warp of slack."""
    from cudnn.gated_attention_block.kernels.qk_norm_rope import lanes_per_row

    attn, qi, raw = prepare_warp_counts(t, h_q=geom["h_q"], h_kv=geom["h_kv"], d=geom["d"], index_heads=4, index_head_dim=128, index_q=True, index_raw=True)
    rpw_a, rpw_i = 32 // lanes_per_row(geom["d"]), 32 // lanes_per_row(128)
    rows_a, rows_q = t * (geom["h_q"] + 2 * geom["h_kv"]), t * 4
    assert rows_a <= attn * rpw_a < rows_a + rpw_a and rows_q <= qi * rpw_i < rows_q + rpw_i and t <= raw * rpw_i < t + rpw_i
    assert prepare_warp_counts(t, h_q=geom["h_q"], h_kv=geom["h_kv"], d=geom["d"]) == (attn, 0, 0)


@pytest.mark.parametrize(
    "d, rope, page, threads, d_i, match",
    [
        (250, 64, 16, 128, None, "multiple of 8"),
        (256, 48, 16, 128, None, "power of two"),
        (256, 64, 0, 128, None, "page_size must be >= 1"),
        (256, 64, 16, 80, None, "multiple of the 32 lanes"),
        (256, 64, 16, 128, 120, "divide a warp"),
    ],
)
def test_validate_prepare_shape_rejects(d, rope, page, threads, d_i, match):
    with pytest.raises(ValueError, match=match):
        validate_prepare_shape(d, rope, page, threads, d_i)


def test_validate_prepare_shape_accepts_the_served_geometries():
    for g in (_GEOM_FN, _GEOM_SMALL, _GEOM_D128):
        validate_prepare_shape(g["d"], g["rope"], 16, 128, 128)
        validate_prepare_shape(g["d"], g["rope"], 64, 128, None)


@requires_cuda
def test_compile_rejects_unsupported_declarations():
    with pytest.raises(ValueError, match="bf16 / f16"):
        compile_qsa_prepare(dtype=torch.float32, h_q=8, h_kv=2, d=256, rope_dim=64, eps=_EPS, page_size=16)
    with pytest.raises(ValueError, match="int32 or int64"):
        compile_qsa_prepare(dtype=BF16, h_q=8, h_kv=2, d=256, rope_dim=64, eps=_EPS, page_size=16, slot_dtype=torch.int16)
    with pytest.raises(ValueError, match="identity copy"):
        compile_qsa_prepare(dtype=BF16, h_q=8, h_kv=2, d=256, rope_dim=0, eps=_EPS, page_size=16, apply_norm=False)
    with pytest.raises(ValueError, match="n_commit"):
        compile_qsa_prepare(
            dtype=BF16,
            h_q=8,
            h_kv=2,
            d=256,
            rope_dim=64,
            eps=_EPS,
            page_size=16,
            index=QsaPrepareIndexArms(heads=4, head_dim=128, compress=False, n_commit=True),
        )


def test_moved_bytes_counts_the_rows_of_every_arm():
    t, h_q, h_kv, d = 16, 24, 2, 256
    base = (2 * t * h_q + 5 * t * h_kv) * d * 2 + t * 4
    assert moved_bytes(t, h_q, h_kv, d) == base
    arms = QsaPrepareIndexArms(heads=4, head_dim=128, raw_pool=True, query=True, compress=True)
    assert moved_bytes(t, h_q, h_kv, d, index=arms) == base + (2 + 3 + 2 * 4) * t * 128 * 2
    assert moved_bytes(t, h_q, h_kv, d, slot_bytes=8, index=QsaPrepareIndexArms(heads=4, head_dim=128)) == base + t * 4 + 2 * t * 128 * 2


# ---------------------------------------------------------------------------
# The kernel, every output bitwise the unfused chain (any CUDA device)
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize(
    "geom, batch, rows, dtype, slot_dtype, layout, pos0, n_commit",
    [
        pytest.param(_GEOM_FN, 4, 4, BF16, torch.int32, "hnd", (7, 0, 13, 2048), None, id="fn_b4_r4_bf16_i32_hnd"),
        pytest.param(_GEOM_FN, 4, 1, F16, torch.int64, "nhd", (3, 0, 11, 2050), None, id="fn_b4_r1_f16_i64_nhd"),
        pytest.param(_GEOM_D128, 3, 5, BF16, torch.int64, "hnd", (2, 7, 0), (1, 3, 0), id="d128_b3_r5_commit_prefixes"),
        pytest.param(_GEOM_SMALL, 2, 3, BF16, torch.int32, "nhd", (1, 5), (3, 2), id="small_b2_r3_boundaries"),
        pytest.param(_GEOM_SMALL, 2, 64, BF16, torch.int32, "hnd", (0, 0), None, id="small_b2_r64_prefill"),
    ],
)
def test_fused_prepare_is_bitwise_the_unfused_chain(geom, batch, rows, dtype, slot_dtype, layout, pos0, n_commit):
    """Every arm bound: Q / K normed + rotated in place, the three pools, the indexer queries, the ring, the compressed
    keys and the slot report -- bitwise the five unfused kernels; V / GATE / INDEX bands untouched; padded and
    out-of-range slots unwritten; the compressed rows written are exactly the rows the slot report names."""
    ch = _chain(geom, batch=batch, rows=rows, dtype=dtype, slot_dtype=slot_dtype, layout=layout, pos0=pos0, n_commit=n_commit)
    ch.assert_bitwise(f"{batch}x{rows} {dtype} {slot_dtype} {layout}")
    if n_commit is not None:
        # The decode contract on the slot report: a draft (a row past the accepted prefix) and a finished sequence report -1.
        cs = ch.Bf["cslot"]
        for b, nc in enumerate(n_commit):
            assert (cs[b, nc:] == -1).all(), f"sequence {b}: rows past n_commit={nc} must report -1"


@requires_cuda
@pytest.mark.parametrize(
    "arms",
    [
        pytest.param(None, id="attention_only"),
        pytest.param(QsaPrepareIndexArms(heads=4, head_dim=128, raw_pool=True, query=False, compress=False), id="raw_pool_only_the_blocks_form"),
        pytest.param(QsaPrepareIndexArms(heads=4, head_dim=128, raw_pool=False, query=True, compress=False), id="query_only"),
        pytest.param(QsaPrepareIndexArms(heads=4, head_dim=128, raw_pool=False, query=False, compress=True, n_commit=False), id="compress_only"),
        pytest.param(QsaPrepareIndexArms(heads=4, head_dim=128, raw_pool=True, query=False, compress=True, n_commit=False), id="raw_pool_and_compress"),
        pytest.param(
            QsaPrepareIndexArms(heads=4, head_dim=128, raw_pool=False, query=True, compress=True, n_commit=False, rounding="hf"), id="query_and_compress_hf"
        ),
    ],
)
def test_every_arm_subset_is_bitwise_its_unfused_twins(arms):
    """The arms are traced by presence: each subset folds the others out and stays bitwise the kernels it keeps (the
    block binds the raw-pool-only form today; the serving step binds the compress and query arms); under ``rounding="hf"`` the query arm
    and the compress arm stay bitwise their unfused twins compiled with the same value."""
    ch = _chain(_GEOM_SMALL, batch=2, rows=4, pos0=(3, 10), arms=arms, band=True)
    ch.assert_bitwise(f"arms {arms}")


@requires_cuda
@pytest.mark.parametrize("ring_slots, rope_table", [(False, False), (True, True)], ids=["derived_rows_per_token", "ring_slots_rope_table"])
def test_paged_fused_prepare_under_hf_rounding_is_bitwise_the_paged_step_form(ring_slots, rope_table):
    """The PAGED compress arm and the query arm under ``rounding="hf"``: the fused launch bitwise the step form and the pool-1 compress
    compiled with ``rounding="hf"`` (three sequences, one padding), composed with the ring-slot and the RoPE-table arms."""
    ch = _chain(
        _GEOM_SMALL,
        batch=4,
        rows=4,
        pos0=(3, 0, 11, 2050),
        paged=True,
        page=16,
        cap=8,
        ring_slots=ring_slots,
        rope_table=rope_table,
        padding=(1,),
        rounding="hf",
    )
    ch.assert_bitwise(f"hf rounding paged slots={ring_slots} table={rope_table}")
    assert ch.r_fused.index.rounding == "hf" and ch.r_step.rounding == "hf" and ch.r_q1.rounding == "hf"
    plain = _chain(_GEOM_SMALL, batch=4, rows=4, pos0=(3, 0, 11, 2050), paged=True, page=16, cap=8, ring_slots=ring_slots, rope_table=rope_table, padding=(1,))
    assert not _same(plain.Bf["comp"], ch.Bf["comp"]), "the attribute changed nothing on the compressed rows"
    assert torch.equal(plain.Bf["cslot"], ch.Bf["cslot"]) and _same(plain.Bf["ring"], ch.Bf["ring"])


@requires_cuda
def test_rope_only_form_is_bitwise_qk_norm_rope_without_weights():
    """``apply_norm=False`` (the block's ``geometry.qk_norm=False``): the RoPE-only Q / K and the pools, bitwise; the
    passthrough dims ``[rope_dim, D)`` of Q / K are bit-exact copies of the input."""
    ch = _chain(_GEOM_SMALL, batch=2, rows=4, pos0=(0, 9), apply_norm=False, arms=QsaPrepareIndexArms(heads=4, head_dim=128), band=True)
    ch.assert_bitwise("rope-only")
    q_in, q_out = ch.cols(ch.slab0, ch.o_q, ch.h_q, ch.d), ch.cols(ch.Bf["slab"], ch.o_q, ch.h_q, ch.d)
    assert _same(q_in[..., ch.rope :], q_out[..., ch.rope :]) and not _same(q_in[..., : ch.rope], q_out[..., : ch.rope])


@requires_cuda
def test_capacity_overflow_reports_minus_two_and_writes_nothing():
    """A row completing a block past the compressed-key cache reports -2 and writes no row -- the step kernel's contract,
    reproduced bitwise (cache capacity 2 blocks, positions from 4: the row at position 11 completes block 2, past the cache)."""
    ch = _Chain(_GEOM_SMALL, batch=1, rows=8, pos0=(4,))
    ch.comp0 = torch.full((1, 2, ch.d_i), ch.sent, device="cuda", dtype=BF16)
    ch.A, ch.Bf = ch._fresh(), ch._fresh()
    ch.compile_unfused()
    ch.compile_fused()
    ch.run_unfused()
    ch.run_fused()
    ch.assert_bitwise("capacity")
    cs = ch.Bf["cslot"][0].tolist()
    assert cs == [-1, -1, -1, 1, -1, -1, -1, -2], cs


@requires_cuda
def test_two_launches_are_bitwise_and_a_second_mapping_replaces_every_row():
    """Determinism: a second fused launch on fresh copies reproduces the first bit for bit; the same artifact with another
    slot mapping places every live row where the new mapping says (the plan is reused, Rule 4)."""
    ch = _chain(_GEOM_SMALL, batch=2, rows=4, pos0=(6, 1))
    X2 = ch._fresh()
    ch.run_fused(X2)
    torch.cuda.synchronize()
    for key in ("slab", "qi", "ring", "comp", "cslot"):
        assert _same(X2[key], ch.Bf[key]), key
    assert _same(X2["k_pool"].contiguous(), ch.Bf["k_pool"].contiguous()) and _same(X2["i_pool"], ch.Bf["i_pool"])
    gen = torch.Generator(device="cuda").manual_seed(11)
    ch.slot = torch.randperm(ch.n_pages * _PAGE, generator=gen, device="cuda")[: ch.T].to(ch.slot_dtype)
    ch.live = ch.slot >= 0
    X3 = ch._fresh()
    ch.run_fused(X3)
    torch.cuda.synchronize()
    s_ = ch.slot.long()
    assert _same(X3["k_pool"][s_ // _PAGE, :, s_ % _PAGE, :], ch.cols(ch.Bf["slab"], ch.o_k, ch.h_kv, ch.d))
    assert _same(X3["v_pool"][s_ // _PAGE, :, s_ % _PAGE, :], ch.cols(ch.slab0, ch.o_v, ch.h_kv, ch.d))


@requires_cuda
def test_run_rejects_contract_violations():
    """Typed refusals, both directions: an arm's operands missing or extra against the traced arms, a ring too shallow for
    the step, the wrong slot dtype, a missing ``seq_q`` with the compress arm, a Q band of another head count."""
    ch = _Chain(_GEOM_SMALL, batch=2, rows=4, pos0=(0, 0))
    ch.compile_fused()
    X = ch.Bf
    base = ch.fused_kwargs(X)
    q, k, v = ch.cols(X["slab"], ch.o_q, ch.h_q, ch.d), ch.cols(X["slab"], ch.o_k, ch.h_kv, ch.d), ch.cols(X["slab"], ch.o_v, ch.h_kv, ch.d)

    def run(**over):
        kw = dict(base)
        kw.update(over)
        run_qsa_prepare(ch.r_fused, q, k, v, None, None, ch.w_q, ch.w_k, ch.cos_t, ch.sin_t, X["k_pool"], X["v_pool"], ch.slot, stream=_stream(), **kw)

    with pytest.raises(ValueError, match="must be bound"):
        run(ring=None)
    with pytest.raises(ValueError, match="needs at least"):
        run(ring=ch.ring0[:, :4].contiguous())
    with pytest.raises(ValueError, match="seq_q"):
        run(seq_q=None)
    with pytest.raises(ValueError, match="slot dtype is fixed"):
        run_qsa_prepare(
            ch.r_fused, q, k, v, None, None, ch.w_q, ch.w_k, ch.cos_t, ch.sin_t, X["k_pool"], X["v_pool"], ch.slot.to(torch.int64), stream=_stream(), **base
        )
    with pytest.raises(ValueError, match=r"q must be a \[T=8, H=8, D=256\]"):
        run_qsa_prepare(ch.r_fused, q[:, :4], k, v, None, None, ch.w_q, ch.w_k, ch.cos_t, ch.sin_t, X["k_pool"], X["v_pool"], ch.slot, stream=_stream(), **base)
    with pytest.raises(ValueError, match="compiled WITH the RMSNorm"):
        run_qsa_prepare(ch.r_fused, q, k, v, None, None, None, None, ch.cos_t, ch.sin_t, X["k_pool"], X["v_pool"], ch.slot, stream=_stream(), **base)
    r_attn = compile_qsa_prepare(dtype=BF16, h_q=ch.h_q, h_kv=ch.h_kv, d=ch.d, rope_dim=ch.rope, eps=_EPS, page_size=_PAGE)
    with pytest.raises(ValueError, match="traced no indexer arm"):
        run_qsa_prepare(
            r_attn,
            q,
            k,
            v,
            None,
            None,
            ch.w_q,
            ch.w_k,
            ch.cos_t,
            ch.sin_t,
            X["k_pool"],
            X["v_pool"],
            ch.slot,
            stream=_stream(),
            index_k_raw_src=base["index_k_raw_src"],
        )


@requires_cuda
@pytest.mark.parametrize("rows", [1, 4], ids=["one_token", "mtp_four_rows"])
def test_decode_step_launch_census_five_unfused_one_fused(rows):
    """MEASURED, not asserted from the code: the decode step's prepare chain at the block-sparse model's TP-1 geometry
    (B = 4 sequences, one or four rows each) is FIVE kernel launches unfused -- norm + RoPE, the K / V cache write, the raw
    key's cache write, the indexer queries' norm, the ring + compress -- and ONE fused, by the CUDA profiler's census; the
    outputs bitwise (the cell above)."""
    ch = _Chain(_GEOM_FN, batch=4, rows=rows, pos0=(7, 0, 13, 2048))
    ch.compile_unfused()
    ch.compile_fused()
    names = _launch_names(ch.run_unfused, ch.run_fused)
    if names is None:
        pytest.skip("the CUDA profiler recorded no kernel activity on this box (CUPTI unavailable); the bitwise cells stand")
    unfused, fused = names
    print(
        f"census B=4 rows={rows}: unfused {len(unfused)} launches {[n.split('_tensorptr')[0] for n in unfused]}; fused {len(fused)} {[n.split('_tensorptr')[0] for n in fused]}"
    )
    assert len(fused) == 1 and "frost_qsa_prepare" in fused[0]
    assert len(unfused) == 5
    ch.assert_bitwise(f"census B=4 rows={rows}")


# ---------------------------------------------------------------------------
# The block knob fuse_prepare: declaration rows and the stage list (host), bitwise vs the unfused block (Rubin)
# ---------------------------------------------------------------------------


def _make_block(geom_kw, batch, seq_len, dtype=BF16, qsa=None, **blk_kw):
    geom = GatedAttentionBlockGeometry(**geom_kw, qsa=qsa)
    inp = make_inputs(RefGeometry(**geom_kw), batch=batch, seq_len=seq_len, dtype=dtype, device=_DEV)
    if qsa is not None and qsa.index_band:
        # The band's columns of W_qkvg: the reference inputs carry the dense four bands; append the indexer band's rows.
        gen = torch.Generator(device=_DEV).manual_seed(99)
        extra = (torch.randn(qsa.index_band_cols, geom_kw["d_model"], generator=gen, device=_DEV, dtype=torch.float32) * 0.02).to(dtype)
        inp["w_qkvg"] = torch.cat([inp["w_qkvg"], extra], 0).contiguous()
    out = torch.empty(batch, seq_len, geom.d_model, device=_DEV, dtype=dtype)
    blk = GatedAttentionBlockFwd(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, geom, **blk_kw)
    return blk, inp, out


def test_fuse_prepare_declaration_rows():
    """The knob's rows, typed and naming the knob: a non-bool, no write-through to fold, together with ``fuse_norm_rope``,
    the out-of-place layout, the packed and the paged forms, the e4m3 KV cache; and the knob accepted on the in-place bf16 block
    with the attribute."""
    with pytest.raises(TypeError, match="fuse_prepare must be a bool"):
        _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16, fuse_prepare=1)
    with pytest.raises(ValueError, match="needs paged_kv_page_size > 0"):
        _make_block(_SMALL_D256, 1, 64, fuse_prepare=True)
    with pytest.raises(NotImplementedError, match="fuse_norm_rope=True"):
        _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16, fuse_prepare=True, fuse_norm_rope=True)
    with pytest.raises(ValueError, match="inplace_qkv=False"):
        _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16, fuse_prepare=True, inplace_qkv=False)
    with pytest.raises(NotImplementedError, match="paged_kv_page_size with save_for_backward"):
        _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16, fuse_prepare=True, save_for_backward=True)
    # the knob's own arms (the write-through serves thd=True and the paged forms since 2026-10-14; the prepare kernel does not yet)
    with pytest.raises(NotImplementedError, match="fuse_prepare=True with thd=True"):
        _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16, fuse_prepare=True, thd=True, num_sequences=1, max_seq_len=64)
    mk = lambda *shape, dt=BF16: torch.empty(*shape, dtype=dt, device=_DEV)  # noqa: E731
    paged = dict(
        paged_kv_page_size=16,
        kv_source="paged",
        sample_k_cache=mk(4, 2, 16, 256),
        sample_v_cache=mk(4, 2, 16, 256),
        sample_block_table=mk(1, 4, dt=torch.int32),
    )
    with pytest.raises(NotImplementedError, match="fuse_prepare=True with kv_source='paged' on a dense geometry \\(the DECODE form\\)"):
        _make_block(dict(_SMALL_D256, causal_bottom_right=True), 1, 1, fuse_prepare=True, **paged)
    # the block-sparse decode mode over the pools takes the fusion (2026-10-21): the caller-list band block's prepare launch carries the
    # raw-key pool arm in place of the K / V-only write-through + the raw key's own write, the measured default is the INDEXER form's -- the caller-list band form keeps False under AUTO on every part
    from cudnn.gated_attention_block.api import SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS

    band, *_ = _make_block(dict(_SMALL_D256, causal_bottom_right=True), 1, 1, qsa=QsaSpec(index_band=True), fuse_prepare=True, **paged)
    assert band.fuse_prepare is True and band.fuse_prepare_requested is True and isinstance(band._prepare, _QsaPrepare) and band._cache_write is None
    assert band.sparse_decode and not band.indexer_decode and not band._prepare.decode_arms and band._prepare._index_arms().raw_pool
    auto, *_ = _make_block(dict(_SMALL_D256, causal_bottom_right=True), 1, 1, qsa=QsaSpec(index_band=True), **paged)
    assert (
        auto.fuse_prepare is False and auto.fuse_prepare_requested is None and auto._prepare is None and SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS == ((10, 7),)
    )
    assert _make_block(dict(_SMALL_D256, causal_bottom_right=True), 1, 1, qsa=QsaSpec(index_band=True), fuse_prepare=False, **paged)[0]._prepare is None
    with pytest.raises(NotImplementedError, match="fuse_prepare=True with kv_cache_dtype=torch.float8_e4m3fn"):
        _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16, fuse_prepare=True, kv_cache_dtype=torch.float8_e4m3fn)
    blk, *_ = _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16, fuse_prepare=True)
    assert blk.fuse_prepare and blk.paged_kv_page_size == 16 and isinstance(blk._prepare, _QsaPrepare)
    plain, *_ = _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16)
    assert not plain.fuse_prepare and plain._prepare is None


# The fused stage's rope_dim == 0 decline, verbatim (the cells below pin the EXACT message: it names the zero-width table operands).
_ROPE_DIM_ZERO_DECLINE = (
    "qsa_prepare: fuse_prepare=True with geometry.rope_dim=0: the fused launch would bind the RoPE tables cos / sin as zero-width "
    "[T, 0] operands, which its artifact (traced with one-column tables) does not take -- the knob needs rope_dim > 0 (with or without qk_norm)"
)


@pytest.mark.parametrize("qk_norm", [True, False], ids=["rmsnorm_only", "no_norm_no_rope"])
def test_fuse_prepare_declines_every_rope_dim_zero_geometry(qk_norm):
    """``rope_dim=0`` under the knob is ONE typed ``ValueError`` naming the zero-width RoPE table operands, with and without ``qk_norm``
    (the exact message): the fused launch binds ``cos`` / ``sin`` as ``[T, rope_dim]`` operands and its artifact is traced with one-column
    tables, so a ``[T, 0]`` table is not an operand it takes -- the stage declines at ``check_support`` (device-free) instead of letting a
    shape mismatch surface at the launch.  Pinned on the stage alone and on the stage the block built.  Through the block's own
    ``check_support``, the RMSNorm-only geometry (``qk_norm=True``) is a legal declaration that reaches the stage (the Rubin cell below),
    while ``qk_norm=False`` with ``rope_dim=0`` is the geometry's own decline first (``validate()``: an identity copy of Q / K), before any
    stage is asked."""
    geom_kw = dict(_SMALL_D256, rope_dim=0, qk_norm=qk_norm)
    stage = _QsaPrepare(GatedAttentionBlockGeometry(**geom_kw), batch=1, seq_len=64, dtype=BF16, page_size=16)  # validate() runs at check_support
    with pytest.raises(ValueError, match=re.escape(_ROPE_DIM_ZERO_DECLINE)):
        stage.check_support()
    blk, *_ = _make_block(geom_kw, 1, 64, paged_kv_page_size=16, fuse_prepare=True)  # the dense constructor declares; the geometry's rows run at check_support
    assert isinstance(blk._prepare, _QsaPrepare)
    with pytest.raises(ValueError, match=re.escape(_ROPE_DIM_ZERO_DECLINE)):
        blk._prepare.check_support()
    if not qk_norm:
        # through the block, the geometry's own row (an identity copy of Q / K) is declined before any stage is asked -- device-agnostic
        with pytest.raises(ValueError, match="qk_norm=False with rope_dim=0 leaves stage"):
            blk.check_support()


@requires_rubin
def test_rmsnorm_only_block_with_fuse_prepare_is_declined_at_check_support():
    """Through the block's own ``check_support`` (its stage loop, which the projection and SDPA engines admit on Rubin): the RMSNorm-only
    block declared with the knob is the fused stage's exact decline, and the same declaration without the knob passes the plan check --
    the decline is the knob's, not the geometry's."""
    blk, *_ = _make_block(dict(_SMALL_D256, rope_dim=0), 1, 64, paged_kv_page_size=16, fuse_prepare=True)
    with pytest.raises(ValueError, match=re.escape(_ROPE_DIM_ZERO_DECLINE)):
        blk.check_support()
    plain, *_ = _make_block(dict(_SMALL_D256, rope_dim=0), 1, 64, paged_kv_page_size=16)
    assert plain.check_support() is True


def _assert_prepare_compiles_with_pdl(stage, expected, monkeypatch):
    """``_QsaPrepare.compile`` hands ``use_pdl`` to ``compile_qsa_prepare`` for BOTH slot recipes (the flag is a compile key of the
    kernel: the launch attribute and the griddepcontrol pair are traced in), recorded through the kernel module's entry (patched by its
    dotted path: the module is imported once, at the top of this file, in one form) -- no device."""
    seen = []
    monkeypatch.setattr("cudnn.gated_attention_block.kernels.qsa_prepare.compile_qsa_prepare", lambda **kw: seen.append(kw) or ("recipe", kw["slot_dtype"]))
    stage.compile()
    assert sorted((kw["slot_dtype"] for kw in seen), key=str) == [torch.int32, torch.int64]
    assert [kw["use_pdl"] for kw in seen] == [expected, expected], seen


@pytest.mark.parametrize("band", [False, True], ids=["dense", "qsa_band"])
def test_fuse_prepare_builds_one_stage_in_place_of_two_and_keeps_the_carve(band, monkeypatch):
    """With the knob the stage list carries ``qsa_prepare`` in the norm + RoPE slot and neither ``qk_norm_rope`` nor
    ``cache_write``; without it both; the workspace carve is identical (the stage reserves nothing); the fused stage's
    byte count is the two stages' minus nothing but the second read of K.  The fused stage takes the block's resolved ``use_pdl``
    and hands it to the compile (both slot recipes), as the two stages it replaces do: False under the AUTO reading off the decode
    form, True when the block declares it -- the dense bf16 block and the caller-list band block alike (the band's switched chain is
    served; the in-block indexer is the remaining typed decline, pinned elsewhere)."""
    qsa = QsaSpec(index_band=True) if band else None
    kw = dict(qsa=qsa, paged_kv_page_size=16)
    off, *_ = _make_block(_SMALL_D256, 2, 128, **kw)
    on, *_ = _make_block(_SMALL_D256, 2, 128, fuse_prepare=True, **kw)
    names_off, names_on = [st.name for st in off._stages], [st.name for st in on._stages]
    assert "qk_norm_rope" in names_off and "cache_write" in names_off and "qsa_prepare" not in names_off
    assert "qsa_prepare" in names_on and "qk_norm_rope" not in names_on and "cache_write" not in names_on
    assert names_on.index("qsa_prepare") == names_off.index("qk_norm_rope") and len(names_on) == len(names_off) - 1
    assert off._layout() == on._layout()
    assert on._prepare.index_band == band
    assert not hasattr(on._prepare, "workspace_bytes") and not hasattr(on._prepare, "scratch_workspace_bytes")
    t = 2 * 128
    k_read_once = t * 2 * 256 * 2  # the unfused chain reads the K band twice (norm + RoPE, then the cache write); the fused launch once
    slot_read_once = t * 4 if band else 0  # and the band's second cache-write launch reads the slot mapping again
    assert on._prepare.moved_bytes() == off._norm_rope.moved_bytes() + off._cache_write.moved_bytes() - k_read_once - slot_read_once
    # the knob's PDL arm: the stage carries the block's switch and compiles both recipes with it
    assert on.use_pdl is False and on._prepare.use_pdl is False
    _assert_prepare_compiles_with_pdl(on._prepare, False, monkeypatch)
    pdl, *_ = _make_block(_SMALL_D256, 2, 128, fuse_prepare=True, use_pdl=True, **kw)
    assert pdl.use_pdl is True and pdl.use_pdl_requested is True and pdl._prepare.use_pdl is True and pdl._prepare.index_band == band
    _assert_prepare_compiles_with_pdl(pdl._prepare, True, monkeypatch)


def _run_block_pair(geom_kw, batch, seq_len, *, qsa=None, slot_dtype=torch.int32, pad_every=0, seed=7, **blk_kw):
    """The block with ``fuse_prepare`` and the same declaration without, on the same inputs and the same pools' geometry;
    returns ``(on, off)`` dicts of ``blk / out / lse / pools / slot / ws`` (``blk_kw``: declaration keywords both blocks share, e.g.
    ``use_pdl=True``)."""
    runs = {}
    for knob in (True, False):
        blk, inp, out = _make_block(geom_kw, batch, seq_len, qsa=qsa, paged_kv_page_size=_PAGE, return_lse=True, fuse_prepare=knob, **blk_kw)
        blk.check_support()
        blk.compile()
        g, t = blk.geom, batch * seq_len
        ws = torch.full((blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device="cuda")
        gen = torch.Generator(device="cuda").manual_seed(seed)
        n_pages = t // _PAGE + 3
        slot = torch.randperm(n_pages * _PAGE, generator=gen, device="cuda")[:t].to(slot_dtype)
        if pad_every:
            slot[::pad_every] = -1
        k_cache = torch.full((n_pages, g.h_kv, _PAGE, g.d_head), float("nan"), device="cuda", dtype=BF16)
        v_cache = torch.full((n_pages, _PAGE, g.h_kv, g.d_head), float("nan"), device="cuda", dtype=BF16).permute(0, 2, 1, 3)  # NHD by strides
        index_cache = torch.full((n_pages, _PAGE, g.qsa.index_head_dim), float("nan"), device="cuda", dtype=BF16) if g.index_band else None
        lse = torch.full((batch, g.h_q, seq_len), float("nan"), device="cuda", dtype=torch.float32)
        out.fill_(float("nan"))
        kw = dict(k_cache=k_cache, v_cache=v_cache, slot_mapping=slot, lse=lse)
        if g.index_band:
            kw["index_k_raw"] = index_cache
        if qsa is not None:
            pos = torch.arange(seq_len, device="cuda").repeat(batch)
            ids, lens = full_block_ids(pos, qsa.top_k, qsa.block_size)
            kw.update(block_ids=ids.contiguous(), block_lens=lens.contiguous())

        def run(blk=blk, inp=inp, out=out, ws=ws, kw=kw):
            blk.execute(inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], out, ws, **kw)

        run()
        torch.cuda.synchronize()
        runs[knob] = dict(blk=blk, out=out, lse=lse, k_cache=k_cache, v_cache=v_cache, index_cache=index_cache, slot=slot, ws=ws, run=run, inp=inp)
    return runs[True], runs[False]


def _assert_block_pair_bitwise(on, off, label):
    assert torch.isfinite(on["out"].float()).all(), f"{label}: non-finite out"
    assert _same(on["out"], off["out"]), f"{label}: fuse_prepare changed out"
    assert torch.equal(on["lse"], off["lse"]), f"{label}: fuse_prepare changed the LSE"
    assert _same(on["k_cache"].contiguous(), off["k_cache"].contiguous()) and _same(
        on["v_cache"].contiguous(), off["v_cache"].contiguous()
    ), f"{label}: pools differ"
    if on["index_cache"] is not None:
        assert _same(on["index_cache"], off["index_cache"]), f"{label}: the raw-key pool differs"
    assert on["blk"].get_workspace_size() == off["blk"].get_workspace_size()
    g, blk = on["blk"].geom, on["blk"]
    t = blk.batch * blk.seq_len
    proj = _view(on["ws"], blk._layout().proj, (t, g.n_qkvg), BF16)
    live = on["slot"] >= 0
    s_ = on["slot"][live].long()
    k_rows = _cols(proj, g.qkvg_offsets[2], g.h_kv, g.d_head)[live]
    assert _same(on["k_cache"][s_ // _PAGE, :, s_ % _PAGE, :], k_rows), f"{label}: the K pool is not the post-RoPE K band attention consumed"
    if g.index_band:
        raw = index_k_raw_view(proj, g, blk.batch, blk.seq_len).reshape(t, -1)[live]
        assert _same(on["index_cache"][s_ // _PAGE, s_ % _PAGE, :], raw)
    print(f"{label}: out / LSE / pools bitwise; {int(live.sum())} / {t} live slots")


def _assert_block_census(on, off, label, expect_fewer):
    names = _launch_names(on["run"], off["run"])
    if names is None:
        pytest.skip("the CUDA profiler recorded no kernel activity on this box (CUPTI unavailable); the bitwise cells stand")
    n_on, n_off = len(names[0]), len(names[1])
    print(f"{label}: block launches fuse_prepare on {n_on} / off {n_off}: {[n.split('_tensorptr')[0] for n in names[0]]}")
    assert n_off - n_on == expect_fewer, f"{label}: expected {expect_fewer} launches fewer, got {n_off} -> {n_on}"
    assert sum("frost_qsa_prepare" in n for n in names[0]) == 1 and not any("frost_qk_norm_rope" in n or "frost_cache_write" in n for n in names[0])


@requires_rubin
@pytest.mark.parametrize("use_pdl", [None, True], ids=["pdl_auto", "pdl_on"])
def test_dense_block_with_fuse_prepare_is_bitwise_the_two_stage_block(use_pdl):
    """The dense 8/2 d256 block with write-through, B=2 x S=256: the knob on vs off -- ``out``, the LSE and the pools
    bitwise, the K pool the post-RoPE K band; the block's census one launch shorter (the two stages became one).  Under
    ``use_pdl=True`` both blocks launch their chain with the programmatic-dependent-launch attribute (the fused stage traces the
    kernel's griddepcontrol pair, as the two stages it replaces do) and the pair stays bitwise, one launch shorter."""
    label = f"dense 8/2 B=2 S=256 use_pdl={use_pdl}"
    on, off = _run_block_pair(_SMALL_D256, 2, 256, slot_dtype=torch.int64, pad_every=9, use_pdl=use_pdl)
    assert on["blk"].use_pdl is bool(use_pdl) and on["blk"]._prepare.use_pdl is bool(use_pdl)
    _assert_block_pair_bitwise(on, off, label)
    _assert_block_census(on, off, label, expect_fewer=1)


@requires_rubin
@pytest.mark.parametrize("batch, seq_len", [(4, 4), (1, 512)], ids=["decode_shape_b4_s4", "prefill_b1_s512"])
def test_qsa_band_block_with_fuse_prepare_is_bitwise_the_two_stage_block(batch, seq_len):
    """The block-sparse model's TP-1 geometry with the indexer band and write-through (the full list = the identity
    below the bound), at the decode shape (4 sequences x 4 rows: one block completing per sequence) and a prefill shape:
    ``out`` / LSE / the K, V and raw-key pools bitwise with the knob on; the census TWO launches shorter (norm + RoPE, the
    K / V write and the raw-key write became one launch)."""
    qsa = QsaSpec(index_band=True)
    on, off = _run_block_pair(_FLASH_NEXT, batch, seq_len, qsa=qsa, pad_every=13)
    _assert_block_pair_bitwise(on, off, f"fn24-2 band B={batch} S={seq_len}")
    _assert_block_census(on, off, f"fn24-2 band B={batch} S={seq_len}", expect_fewer=2)


# ---------------------------------------------------------------------------
# The PAGED compress arm, the ring's slot form and the RoPE-table arm: bitwise the step form's paged arms (any CUDA device)
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize(
    "geom, batch, rows, dtype, slot_dtype, layout, pos0, n_commit, page, cap, padding",
    [
        pytest.param(_GEOM_FN, 4, 4, BF16, torch.int32, "hnd", (7, 0, 13, 2048), None, 16, 8, (), id="fn_b4_r4_bf16_i32_p16_cap8"),
        pytest.param(_GEOM_FN, 4, 1, F16, torch.int64, "nhd", (3, 0, 11, 2050), None, 32, 4, (), id="fn_b4_r1_f16_i64_p32_cap4"),
        pytest.param(_GEOM_D128, 3, 5, BF16, torch.int64, "hnd", (2, 7, 0), (1, 3, 0), 64, 8, (), id="d128_b3_r5_commit_prefixes_p64"),
        pytest.param(_GEOM_SMALL, 2, 3, BF16, torch.int32, "nhd", (1, 5), (3, 2), 16, 8, (), id="small_b2_r3_boundaries_p16"),
        pytest.param(_GEOM_SMALL, 3, 4, BF16, torch.int32, "hnd", (9, 0, 4), None, 16, 8, (1,), id="small_b3_r4_padding_sequence"),
    ],
)
def test_paged_fused_prepare_is_bitwise_the_paged_step_form(geom, batch, rows, dtype, slot_dtype, layout, pos0, n_commit, page, cap, padding):
    """The decode step's prepare with the PAGED compress arm -- the ring blocks behind the state table, the compressed-key cache
    through a permuted block table at P 16 / 32 / 64, the positions from ``kv_lens`` -- bitwise the unfused chain whose step form
    runs the SAME paged arms (two call sites of one addressing helper per table): Q / K, the pools, the indexer queries, the ring,
    the cache and the slot report; V / GATE / INDEX untouched; the padding sequence (``kv_lens 0`` on the null block / page 0)
    writes nothing and reports -1."""
    ch = _chain(
        geom,
        batch=batch,
        rows=rows,
        dtype=dtype,
        slot_dtype=slot_dtype,
        layout=layout,
        pos0=pos0,
        n_commit=n_commit,
        paged=True,
        page=page,
        cap=cap,
        padding=padding,
    )
    ch.assert_bitwise(f"paged {batch}x{rows} {dtype} {slot_dtype} {layout} P{page} cap{cap}")
    cs = ch.Bf["cslot"]
    if n_commit is not None:
        for b, nc in enumerate(n_commit):
            assert (cs[b, nc:] == -1).all(), f"sequence {b}: rows past n_commit={nc} must report -1"
    for b in padding:
        assert (cs[b] == -1).all(), f"padding sequence {b} must report -1 on every row"
    if padding:
        # the null ring block keeps the chain's prior random state bit for bit (nothing is written there); page 0 keeps its sentinel
        assert _same(ch.Bf["ring"][0], ch.ring0[0]) and bool((ch.Bf["comp"][0].float() == ch.sent).all()), "the null ring block / page 0 were written"
    # the compressed rows the slot report names are the blocks of their sequences, through the table
    for b in range(batch):
        for j, blk in enumerate(cs[b].tolist()):
            if blk >= 0:
                assert bool((ch.kbar_row(ch.Bf, b, blk).float() != ch.sent).any()), f"sequence {b} row {j}: block {blk} reported but its cache row is untouched"


@requires_cuda
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64], ids=["slots_i32", "slots_i64"])
def test_paged_ring_slot_form_is_bitwise_the_derived_form(slot_dtype):
    """The ring's WRITE rows named by the stack's own slot mapping (``block x cap + pos % cap``, ``-1`` on a dead row; the slots share
    the artifact's slot dtype) against the rows the kernel derives from the state table: the fused launch and the step form,
    bitwise -- ring, cache, slot report -- with a padding sequence in the batch."""
    kw = dict(batch=3, rows=4, slot_dtype=slot_dtype, pos0=(5, 0, 14), paged=True, page=16, cap=8, padding=(1,))
    derived = _chain(_GEOM_SMALL, **kw)
    derived.assert_bitwise("paged, derived ring rows")
    explicit = _chain(_GEOM_SMALL, ring_slots=True, **kw)
    explicit.assert_bitwise("paged, the stack's ring slots")
    assert explicit.arms.ring_slots and explicit.r_fused.index.ring_slots and explicit.r_step.ring_slots
    for key in ("slab", "qi", "ring", "comp", "cslot"):
        assert _same(explicit.Bf[key], derived.Bf[key]), key
    assert int((explicit.ring_slots_t < 0).sum()) == 4, "the padding sequence's four rows carry slot -1"


@requires_cuda
def test_paged_state_entry_past_the_ring_or_null_owns_no_ring_in_both_kernels():
    """A LIVE sequence whose state-table entry lies one past the ring's last block (the off-by-one of a ring allocated one block per
    request while the ids start at 1) or is the null block owns no ring in the fused launch AND in the step form (the one shared
    ``ring_entry``): not one ring byte changes for it, a block pooled from this step's rows alone is still written, a block that
    needs ring rows it does not own reports ``-2`` and keeps its page row at the sentinel; the two kernels bitwise each other."""
    ch = _Chain(_GEOM_SMALL, batch=3, rows=4, pos0=(8, 5, 0), paged=True, page=16, cap=8)
    ch.state_table[0, 0] = int(ch.ring0.shape[0])  # one past the ring's blocks 0 .. B
    ch.state_table[1, 0] = 0  # the null block, kv_lens 9: the row at position 7 completes block 1, which needs ring row 4
    ch.compile_unfused()
    ch.compile_fused()
    ch.run_unfused()
    ch.run_fused()
    ch.assert_bitwise("state entries past the ring / null")
    cs = ch.Bf["cslot"].tolist()
    assert cs == [[-1, -1, -1, 2], [-1, -1, -2, -1], [-1, -1, -1, 0]], cs
    # ring blocks 0 .. 2 keep the chain's prior state bit for bit (no ring row for sequences 0 / 1); sequence 2's block 3 holds its raw keys
    assert _same(ch.Bf["ring"][:3], ch.ring0[:3]), "a sequence that owns no ring wrote a ring row"
    raw = ch.cols(ch.slab0, ch.o_ik, 1, ch.d_i)
    for j in range(4):
        assert _same(ch.Bf["ring"][3, j % ch.ring_n, 0], raw[2 * 4 + j, 0]), j
    # the page rows: sequence 0's block 2 written (its own four rows), sequence 1's block 1 at the sentinel
    assert bool((ch.kbar_row(ch.Bf, 0, 2).float() != ch.sent).any()) and bool((ch.kbar_row(ch.Bf, 1, 1).float() == ch.sent).all())


@requires_cuda
@pytest.mark.parametrize("paged", [False, True], ids=["dense_state", "paged_state"])
def test_rope_table_arm_is_bitwise_the_per_token_form(paged):
    """``cos`` / ``sin`` None: the Q / K rows and the indexer-query rows read the position-indexed ``cos_pos`` / ``sin_pos`` at
    ``pos0 + j`` -- bitwise the per-token form fed the rows gathered at the same positions (the padding sequence's rows at the
    table's row 0, the arm's in-bounds clamp), on the dense and the paged compress arm; the unfused chain stays the per-token form."""
    kw = dict(batch=3, rows=4, pos0=(6, 0, 21), paged=paged, page=16, cap=8, padding=(1,) if paged else ())
    per_token = _chain(_GEOM_SMALL, **kw)
    per_token.assert_bitwise(f"per-token rows, paged={paged}")
    table = _chain(_GEOM_SMALL, rope_table=True, **kw)
    table.assert_bitwise(f"RoPE-table arm, paged={paged}")
    assert table.r_fused.index.rope_table
    for key in ("slab", "qi", "ring", "comp", "cslot"):
        assert _same(table.Bf[key], per_token.Bf[key]), key
    assert _same(table.Bf["k_pool"].contiguous(), per_token.Bf["k_pool"].contiguous())
    # the table arm's host bound: the table must cover every position the compress arm's cache can address (paged: the block table's
    # width x P; dense: NB x pool).  The chain's table holds EXACTLY that many rows (the accept cell at the bound, run above); one row
    # short is refused BY FORM, so the in-kernel clamp is never a live row's path.
    bound = table.n_lpages * table.page if paged else table.nb_cap * _POOL
    assert table.cos_pos.shape[1] == bound, (table.cos_pos.shape, bound)
    X = table.Bf
    short = dict(table.fused_kwargs(X))
    short.update(cos_pos=table.cos_pos[:, : bound - 1].contiguous(), sin_pos=table.sin_pos[:, : bound - 1].contiguous())
    with pytest.raises(ValueError, match=f"hand a table of at least {bound} rows"):
        run_qsa_prepare(
            table.r_fused,
            table.cols(X["slab"], table.o_q, table.h_q, table.d),
            table.cols(X["slab"], table.o_k, table.h_kv, table.d),
            table.cols(X["slab"], table.o_v, table.h_kv, table.d),
            None,
            None,
            table.w_q,
            table.w_k,
            None,
            None,
            X["k_pool"],
            X["v_pool"],
            table.slot,
            index_eps=_EPS,
            index_norm_weight_offset=0.0,
            stream=_stream(),
            **short,
        )


@requires_cuda
@pytest.mark.parametrize("paged", [False, True], ids=["dense_state", "paged_state"])
@pytest.mark.parametrize("rows, pos0", [(1, (16, 15)), (4, (14, 3))], ids=["one_row_at_the_ceiling", "mtp_four_rows_straddling_it"])
def test_rope_table_arm_row_at_or_past_the_table_reports_minus_two_and_stores_nothing_rotated(paged, rows, pos0):
    """The RoPE-table arm checks the LIVE (device-derived) position of every row against the table IN-KERNEL -- the host cannot
    (``pos0`` / ``kv_lens`` are device data).  A 16-position compressed-key cache (dense: 4 blocks; paged: one page of P 16 behind a
    1-wide block table) with a table of exactly those 16 rows (the host bound met with equality): the sequence stepping at position 16
    (``pos0 16``; paged ``kv_lens 17``) is reported ``-2`` by the fused launch AND by the step form, its Q / K slab rows keep the
    projection's bits (never rotated with the clamped row 15), its K pool row and its indexer-query row stay at their sentinels, its
    ring row and the cache are untouched; its V pool row and its raw-key pool row (position-free bit copies) are written as in the
    per-token form.  The in-bounds sequence (position 15, ``kv_lens 16``) is bitwise the unfused chain on every output and completes
    block 3 from the ring.  The four-row form straddles the ceiling: rows at 14 / 15 served bitwise (``[-1, 3]``), rows at 16 / 17
    untouched and ``-2``; the second sequence (positions 3 .. 6) completes block 0.  The previous kernel reported ``-1`` for the rows
    past the table and stored their Q / K / K-pool / indexer-query rows rotated at the clamped row."""
    ch = _Chain(_GEOM_SMALL, batch=len(pos0), rows=rows, pos0=pos0, paged=paged, page=16, cap=8, rope_table=True, n_pos=16, pad_every=0)
    assert ch.cos_pos.shape[1] == 16 and ch.nb_cap == 4 and (not paged or ch.n_lpages == 1)
    gen = torch.Generator(device="cuda").manual_seed(5)
    ch.slot = torch.randperm(ch.n_pages * _PAGE, generator=gen, device="cuda")[: ch.T].to(ch.slot_dtype)  # every token on a live slot
    ch.live = torch.ones_like(ch.slot, dtype=torch.bool)
    ch.compile_unfused()
    ch.compile_fused()
    ch.run_unfused()
    ch.run_fused()
    torch.cuda.synchronize()
    A, Bf = ch.A, ch.Bf
    positions = [p + j for p in pos0 for j in range(rows)]  # token t = b * rows + j
    over = [t for t, p in enumerate(positions) if p >= 16]
    served = [t for t, p in enumerate(positions) if p < 16]
    assert over and served, positions
    want = [[-2 if p + j >= 16 else (((p + j) // _POOL) if (p + j) % _POOL == _POOL - 1 else -1) for j in range(rows)] for p in pos0]
    assert Bf["cslot"].tolist() == want, (Bf["cslot"].tolist(), want)
    assert _same(A["cslot"], Bf["cslot"]) and _same(A["ring"], Bf["ring"]) and _same(A["comp"], Bf["comp"]), "the step form and the fused launch disagree"
    q0, k0 = ch.cols(ch.slab0, ch.o_q, ch.h_q, ch.d), ch.cols(ch.slab0, ch.o_k, ch.h_kv, ch.d)
    v0, raw0 = ch.cols(ch.slab0, ch.o_v, ch.h_kv, ch.d), ch.cols(ch.slab0, ch.o_ik, 1, ch.d_i)
    qa, ka = ch.cols(A["slab"], ch.o_q, ch.h_q, ch.d), ch.cols(A["slab"], ch.o_k, ch.h_kv, ch.d)
    qb, kb = ch.cols(Bf["slab"], ch.o_q, ch.h_q, ch.d), ch.cols(Bf["slab"], ch.o_k, ch.h_kv, ch.d)
    s_ = ch.slot.long()
    for t in over:
        assert _same(qb[t], q0[t]) and _same(kb[t], k0[t]), f"token {t} (position {positions[t]}): a Q / K row past the table was rotated and stored"
        assert torch.isnan(Bf["k_pool"][s_[t] // _PAGE, :, s_[t] % _PAGE, :].float()).all(), f"token {t}: the K pool row of a row past the table was written"
        assert _same(Bf["v_pool"][s_[t] // _PAGE, :, s_[t] % _PAGE, :], v0[t]), f"token {t}: the V pool row (a position-free bit copy) must still be written"
        assert _same(Bf["i_pool"][s_[t] // _PAGE, s_[t] % _PAGE, :], raw0[t, 0]), f"token {t}: the raw-key pool row (a bit copy) must still be written"
        assert torch.isnan(Bf["qi"][t].float()).all(), f"token {t}: an indexer-query row past the table was written"
        b, p = t // rows, positions[t]
        ring_row = Bf["ring"][int(ch.state_table[b, 0]), p % ch.ring_n, 0] if paged else Bf["ring"][b, p % ch.ring_n]
        ring_was = ch.ring0[int(ch.state_table[b, 0]), p % ch.ring_n, 0] if paged else ch.ring0[b, p % ch.ring_n]
        assert _same(ring_row, ring_was), f"token {t}: the ring row of a position past the table was written"
    for t in served:
        assert _same(qb[t], qa[t]) and _same(kb[t], ka[t]), f"token {t} (position {positions[t]}): Q / K differ from the unfused chain"
        assert _same(Bf["k_pool"][s_[t] // _PAGE, :, s_[t] % _PAGE, :], A["k_pool"][s_[t] // _PAGE, :, s_[t] % _PAGE, :]), t
        assert _same(Bf["qi"][t], A["qi"][t]), t
    assert _same(Bf["v_pool"].contiguous(), A["v_pool"].contiguous()) and _same(Bf["i_pool"], A["i_pool"])
    # the blocks the report names exist in the cache, and nothing else was written there
    for b, row in enumerate(want):
        for blk in row:
            if blk >= 0:
                assert bool((ch.kbar_row(Bf, b, blk).float() != ch.sent).any()), (b, blk)
    assert int((Bf["comp"].float() != ch.sent).any(-1).sum()) == sum(1 for row in want for blk in row if blk >= 0)


@requires_cuda
@pytest.mark.parametrize("rows", [1, 4], ids=["one_token", "mtp_four_rows"])
def test_paged_decode_form_census_four_unfused_one_fused(rows):
    """MEASURED, not asserted from the code: the decode indexer's prepare chain with the arms the serving step binds
    (``raw_pool=False, query=True, compress=True, paged=True``: no raw-key pool -- the ring replaces it) is FOUR kernel launches
    unfused -- norm + RoPE, the K / V cache write, the indexer queries' norm, the paged ring + compress -- and ONE fused, by the
    CUDA profiler's census; the outputs bitwise."""
    arms = QsaPrepareIndexArms(heads=4, head_dim=128, raw_pool=False, query=True, compress=True, paged=True)
    ch = _Chain(_GEOM_FN, batch=4, rows=rows, pos0=(7, 0, 13, 2048), arms=arms, paged=True, page=16, cap=8)
    ch.compile_unfused()
    ch.compile_fused()
    names = _launch_names(ch.run_unfused, ch.run_fused)
    if names is None:
        pytest.skip("the CUDA profiler recorded no kernel activity on this box (CUPTI unavailable); the bitwise cells stand")
    unfused, fused = names
    print(
        f"paged census B=4 rows={rows}: unfused {len(unfused)} {[n.split('_tensorptr')[0] for n in unfused]}; fused {len(fused)} {[n.split('_tensorptr')[0] for n in fused]}"
    )
    assert len(fused) == 1 and "frost_qsa_prepare" in fused[0]
    assert len(unfused) == 4 and sum("frost_qsa_compress_step" in n for n in unfused) == 1
    ch.assert_bitwise(f"paged census B=4 rows={rows}")


@requires_cuda
def test_paged_run_rejects_contract_violations():
    """Typed refusals, both directions: ``pos0`` on the paged arm, the paged operands missing or handed to a dense recipe, the
    stack's slots on a recipe that derives the rows (and the other way round), a dense 3-D ring on the paged arm, the e4m3
    cache, ``cos`` / ``sin`` handed to the RoPE-table arm or missing without it."""
    ch = _Chain(_GEOM_SMALL, batch=2, rows=4, pos0=(0, 8), paged=True, page=16, cap=8)
    ch.compile_fused()
    X = ch.Bf
    base = ch.fused_kwargs(X)
    q, k, v = ch.cols(X["slab"], ch.o_q, ch.h_q, ch.d), ch.cols(X["slab"], ch.o_k, ch.h_kv, ch.d), ch.cols(X["slab"], ch.o_v, ch.h_kv, ch.d)

    def run(r=None, cos_t=ch.cos_t, sin_t=ch.sin_t, **over):
        kw = dict(base)
        kw.update(over)
        run_qsa_prepare(
            ch.r_fused if r is None else r, q, k, v, None, None, ch.w_q, ch.w_k, cos_t, sin_t, X["k_pool"], X["v_pool"], ch.slot, stream=_stream(), **kw
        )

    with pytest.raises(ValueError, match="pos0 is refused on the paged form"):
        run(pos0=ch.pos0)
    with pytest.raises(ValueError, match="kv_lens .* block_table .* index_state_block_table .* must be bound"):
        run(block_table=None)
    with pytest.raises(ValueError, match="index_state_slot_mapping would be silently ignored"):
        run(index_state_slot_mapping=torch.full((ch.T,), -1, dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match=r"ring must be a 4-D \[blocks >= 1, rows, 1, D=128\]"):
        run(ring=torch.empty(2, 8, 128, device="cuda", dtype=BF16))
    with pytest.raises(ValueError, match="ring holds 4 rows per block; this launch needs at least 7"):
        run(ring=torch.empty(3, 4, 1, 128, device="cuda", dtype=BF16))
    # INVERTED 2026-10-21 (the e4m3 store arm is served): an e4m3 cache under the bf16 recipe names the recipe's dtype; an e4m3 RING stays the ring's rule
    with pytest.raises(ValueError, match=r"index_k_compressed is float8_e4m3fn but this artifact stores the compressed-key cache in torch.bfloat16"):
        run(index_k_compressed=torch.empty(ch.n_kbar_pages, ch.rpp, 1, 128, device="cuda", dtype=torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="raw-key ring is a bit copy .* never quantized"):  # an e4m3 RING is a form error, not the cache's dtype
        run(ring=torch.empty(3, 8, 1, 128, device="cuda", dtype=torch.float8_e4m3fn))
    if _CC >= (8, 9):
        e4 = compile_qsa_prepare(
            dtype=BF16, h_q=ch.h_q, h_kv=ch.h_kv, d=ch.d, rope_dim=ch.rope, eps=_EPS, page_size=_PAGE, index=ch.arms, index_cache_dtype=_E4M3
        )
        assert e4.index_cache_dtype == _E4M3 and e4.compiled is not ch.r_fused.compiled
        with pytest.raises(ValueError, match=r"index_q_out must have dtype torch.float8_e4m3fn \(the recipe's\), got torch.bfloat16"):
            run(e4)
        with pytest.raises(ValueError, match=r"index_k_compressed must have dtype torch.float8_e4m3fn \(the recipe's\), got torch.bfloat16"):
            run(e4, index_q_out=torch.empty(ch.T, ch.h_i, 128, device="cuda", dtype=_E4M3))
    with pytest.raises(ValueError, match="kv_lens must be a contiguous int32"):
        run(kv_lens=ch.kv_lens.to(torch.int64))
    # the dense recipe refuses every paged keyword; the paged recipe's compress arm cannot be bound a dense ring
    dense = compile_qsa_prepare(dtype=BF16, h_q=ch.h_q, h_kv=ch.h_kv, d=ch.d, rope_dim=ch.rope, eps=_EPS, page_size=_PAGE, index=ch.arms._replace(paged=False))
    with pytest.raises(ValueError, match="compiled WITHOUT the paged compress arms"):
        run(dense, ring=torch.empty(2, 8, 128, device="cuda", dtype=BF16), index_k_compressed=torch.empty(2, 16, 128, device="cuda", dtype=BF16), pos0=ch.pos0)
    # the ring's slot arm, both directions
    slots = compile_qsa_prepare(
        dtype=BF16, h_q=ch.h_q, h_kv=ch.h_kv, d=ch.d, rope_dim=ch.rope, eps=_EPS, page_size=_PAGE, index=ch.arms._replace(ring_slots=True)
    )
    with pytest.raises(ValueError, match="bind index_state_slot_mapping"):
        run(slots)
    with pytest.raises(ValueError, match="index_state_slot_mapping must be a contiguous torch.int32"):
        run(slots, index_state_slot_mapping=torch.full((ch.T,), -1, dtype=torch.int64, device="cuda"))
    # the RoPE-table arm, both directions
    table = compile_qsa_prepare(
        dtype=BF16, h_q=ch.h_q, h_kv=ch.h_kv, d=ch.d, rope_dim=ch.rope, eps=_EPS, page_size=_PAGE, index=ch.arms._replace(rope_table=True)
    )
    with pytest.raises(ValueError, match="cos / sin are refused on a recipe that traced the RoPE-table arm"):
        run(table)
    with pytest.raises(ValueError, match="per-token tables are required"):
        run(cos_t=None, sin_t=None)


@requires_cuda
def test_compile_rejects_unsupported_paged_declarations_and_keys_the_arms():
    """``paged`` needs ``compress``; ``ring_slots`` needs ``paged``; ``rope_table`` needs ``compress`` and a rotation; a non-io, non-e4m3
    cache dtype and f16 io with an e4m3 cache are ``ValueError``s; the e4m3 store arm (INVERTED 2026-10-21) compiles on sm_89+ as its own
    artifact -- orthogonal to the slot and the table arms -- and is a typed decline naming the instruction below it.  Every arm is a
    compile key: the paged, the slot and the table artifacts are distinct from the dense one and from each other, and the same key
    returns the same artifact."""
    base = dict(dtype=BF16, h_q=8, h_kv=2, d=256, rope_dim=64, eps=_EPS, page_size=16)
    with pytest.raises(ValueError, match="index.paged .* needs index.compress=True"):
        compile_qsa_prepare(index=QsaPrepareIndexArms(heads=4, head_dim=128, compress=False, paged=True), **base)
    with pytest.raises(ValueError, match="index.ring_slots .* needs index.paged=True"):
        compile_qsa_prepare(index=QsaPrepareIndexArms(heads=4, head_dim=128, compress=True, ring_slots=True), **base)
    with pytest.raises(ValueError, match="index.rope_table .* needs index.compress=True"):
        compile_qsa_prepare(index=QsaPrepareIndexArms(heads=4, head_dim=128, compress=False, rope_table=True), **base)
    with pytest.raises(ValueError, match="rope_dim=0"):
        compile_qsa_prepare(index=QsaPrepareIndexArms(heads=4, head_dim=128, compress=True, rope_table=True), **dict(base, rope_dim=0))
    with pytest.raises(ValueError, match="must hold the io dtype .* or torch.float8_e4m3fn"):
        compile_qsa_prepare(index=QsaPrepareIndexArms(heads=4, head_dim=128, compress=True, paged=True), index_cache_dtype=torch.float32, **base)
    with pytest.raises(ValueError, match="needs bf16 io"):
        compile_qsa_prepare(index=QsaPrepareIndexArms(heads=4, head_dim=128, compress=True, paged=True), index_cache_dtype=_E4M3, **dict(base, dtype=F16))
    arms = QsaPrepareIndexArms(heads=4, head_dim=128, raw_pool=False, query=True, compress=True)
    recipes = [
        compile_qsa_prepare(index=a, **base)
        for a in (arms, arms._replace(paged=True), arms._replace(paged=True, ring_slots=True), arms._replace(paged=True, rope_table=True))
    ]
    assert len({id(r.compiled) for r in recipes}) == 4
    assert compile_qsa_prepare(index=arms._replace(paged=True), **base).compiled is recipes[1].compiled
    assert compile_qsa_prepare(index=arms._replace(paged=True), index_cache_dtype=BF16, **base).compiled is recipes[1].compiled  # the io dtype spelled
    assert all(r.index_cache_dtype == BF16 for r in recipes)
    # the rounding attribute: a distinct artifact per value (the record's appended field rides into the key), the default unchanged, an
    # unknown value refused naming the tuple
    hf = compile_qsa_prepare(index=arms._replace(paged=True, rounding="hf"), **base)
    assert hf.compiled is not recipes[1].compiled and hf.index.rounding == "hf"
    assert compile_qsa_prepare(index=arms._replace(paged=True, rounding="single"), **base).compiled is recipes[1].compiled
    with pytest.raises(ValueError, match=r"rounding must be one of \('single', 'hf'\)"):
        compile_qsa_prepare(index=arms._replace(rounding="vllm"), **base)
    if _CC is not None and _CC < (8, 9):
        with pytest.raises(NotImplementedError, match=r"cvt.rn.satfinite.e4m3x2.f32\) needs sm_89 or later"):
            compile_qsa_prepare(index=arms._replace(paged=True), index_cache_dtype=_E4M3, **base)
    elif _CC is not None:
        e4 = [
            compile_qsa_prepare(index=a, index_cache_dtype=_E4M3, **base)
            for a in (arms._replace(paged=True), arms._replace(paged=True, ring_slots=True), arms._replace(paged=True, rope_table=True))
        ]
        assert len({id(r.compiled) for r in e4} | {id(r.compiled) for r in recipes}) == 7 and all(r.index_cache_dtype == _E4M3 for r in e4)
        assert compile_qsa_prepare(index=arms._replace(paged=True), index_cache_dtype=_E4M3, **base).compiled is e4[0].compiled


# ---------------------------------------------------------------------------
# The e4m3 store arm (sm_89+): the fused launch's e4m3 queries and compressed rows
# ---------------------------------------------------------------------------


@requires_sm89
@pytest.mark.parametrize(
    "rows, pos0, n_commit, page, cap, ring_slots, rope_table, padding",
    [
        pytest.param(1, (7, 0, 13, 2048), None, 16, 8, False, False, (), id="r1_p16"),
        pytest.param(4, (3, 0, 11, 2050), None, 32, 8, True, False, (), id="r4_p32_ring_slots"),
        pytest.param(4, (9, 0, 4), None, 16, 8, False, True, (1,), id="r4_p16_rope_table_padding"),
        pytest.param(5, (2, 7, 0), (1, 3, 0), 64, 8, False, False, (), id="r5_p64_commit_prefixes"),
    ],
)
def test_paged_fused_prepare_e4m3_store_arm_is_bitwise_the_step_form_and_the_bf16_cast(rows, pos0, n_commit, page, cap, ring_slots, rope_table, padding):
    """The fused launch with ``index_cache_dtype=torch.float8_e4m3fn`` (the paged compress arm + the query arm) BITWISE the unfused
    chain under the same arm (the pool-1 compress with ``out_dtype`` e4m3, the step form with the e4m3 cache: two call sites of ONE store
    helper), output by output -- and, against the bf16 chain on the SAME inputs, every e4m3 query / compressed row is the bf16 row
    ``.to(float8_e4m3fn)`` (``|bf16 row| < 448`` first), the slot reports and the bf16 ring identical; the arm composes with the ring-slot
    and the RoPE-table arms; ONE launch by the profiler's census."""
    kw = dict(
        batch=len(pos0), rows=rows, pos0=pos0, n_commit=n_commit, paged=True, page=page, cap=cap, ring_slots=ring_slots, rope_table=rope_table, padding=padding
    )
    ch8 = _chain(_GEOM_SMALL, index_cache_dtype=_E4M3, **kw)
    ch16 = _chain(_GEOM_SMALL, **kw)
    label = f"e4m3 store arm r{rows} P{page} slots={ring_slots} table={rope_table}"
    ch8.assert_bitwise(label)
    ch16.assert_bitwise(label + " (bf16 twin)")
    assert ch8.Bf["qi"].dtype == ch8.Bf["comp"].dtype == _E4M3 and ch8.Bf["ring"].dtype == BF16 and ch8.r_fused.index_cache_dtype == _E4M3
    assert _same(ch8.Bf["ring"], ch16.Bf["ring"]) and torch.equal(
        ch8.Bf["cslot"], ch16.Bf["cslot"]
    ), f"{label}: the ring / the slot report differ from the bf16 chain"
    assert _same(ch8.A["slab"], ch16.A["slab"]) and _same(ch8.Bf["k_pool"].contiguous(), ch16.Bf["k_pool"].contiguous()), f"{label}: the attention arm moved"
    qi16, qi8 = ch16.Bf["qi"], ch8.Bf["qi"]
    live_q = ~torch.isnan(qi16.float()).any(-1)  # the stored query rows (every token under the per-token form; the rows below the table under the table arm)
    assert bool(live_q.any()) and float(qi16[live_q].float().abs().max()) < 448.0
    assert torch.equal(_bits(qi8[live_q]), _bits(qi16[live_q].to(_E4M3))), f"{label}: the e4m3 queries are not the e4m3 cast of the bf16 queries"
    assert torch.equal(torch.isnan(qi8.float()).any(-1), ~live_q), f"{label}: a query row written / unwritten differently from the bf16 chain"
    comp16, comp8 = ch16.Bf["comp"], ch8.Bf["comp"]
    written = (comp16.float() != ch16.sent).any(-1)
    assert torch.equal(written, comp8.float().ne(ch8.sent).any(-1)), f"{label}: a compressed row written / unwritten differently from the bf16 chain"
    assert int(written.sum()) == int((ch8.Bf["cslot"] >= 0).sum()) > 0
    assert float(comp16[written].float().abs().max()) < 448.0
    assert torch.equal(_bits(comp8[written]), _bits(comp16[written].to(_E4M3))), f"{label}: the e4m3 compressed rows are not the e4m3 cast of the bf16 rows"
    assert ch8.sent == -448.0 and bool((comp8[0].float() == ch8.sent).all()), f"{label}: page 0 was written"
    names = _launch_names(ch8.run_fused)
    if names is not None:
        (fused,) = names
        assert len(fused) == 1 and "frost_qsa_prepare" in fused[0], fused


# ---------------------------------------------------------------------------
# The chunk form's ring TAIL write: the cache-write kernel over the ring's circular slots (no kernel change -- the pin the extend step's stage relies on)
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64], ids=["slots_i32", "slots_i64"])
@pytest.mark.parametrize("cap", [3, 4, 8], ids=["cap3", "cap4", "cap8"])
def test_cache_write_ring_arm_writes_the_last_cap_tokens_at_the_circular_slots(cap, slot_dtype):
    """The stack's ring ``[blocks, cap, 1, 128]`` permuted to a ``[blocks, 1, cap, 128]`` pool (``page_size = cap``) IS a cache-write pool, and
    the stack's circular slots ``block x cap + pos % cap`` for the LAST ``cap`` tokens of every sequence (``-1`` elsewhere, ``-1`` on the
    padding sequence on the null block) are its slot mapping: after one ``compile_cache_write(h=1, d=128, page_size=cap, two_operands=False)``
    launch over a ``[B x T, 1, 128]`` raw-key band the ring rows hold the raw rows bit for bit, every other row and the null block are
    untouched -- the chunk form's tail write with no kernel change (the chunk compress reads the ring; this launch follows it in stream order)."""
    batch, t, d = 3, 11, 128
    g = torch.Generator(device="cuda").manual_seed(cap)
    raw = torch.randn(batch * t, d, generator=g, device="cuda").to(BF16)
    kv_lens, state = [37, 5, 0], [1, 2, 0]  # a long sequence, one SHORTER than the chunk (its first rows below position 0), a padding one
    ring = torch.full((batch + 1, cap, 1, d), float("nan"), device="cuda", dtype=BF16)
    want = ring.clone()
    slots = []
    for b in range(batch):
        for j in range(t):
            pos = kv_lens[b] - t + j
            live = state[b] > 0 and pos >= 0 and pos + cap >= kv_lens[b]
            slots.append(state[b] * cap + pos % cap if live else -1)
            if live:
                want[state[b], pos % cap, 0] = raw[b * t + j]
    slot = torch.tensor(slots, dtype=slot_dtype, device="cuda")
    r = compile_cache_write(dtype=BF16, h=1, d=d, page_size=cap, slot_dtype=slot_dtype, two_operands=False)
    run_cache_write(r, raw.view(batch * t, 1, d), None, ring.permute(0, 2, 1, 3), None, slot, stream=_stream())
    torch.cuda.synchronize()
    assert _same(ring, want), f"cap {cap}: the ring rows are not the last {cap} raw keys at their circular slots"
    assert bool(torch.isnan(ring[0].float()).all()), "the null block was written"
    n_live = sum(1 for x in slots if x >= 0)
    assert n_live == min(cap, kv_lens[0]) + min(cap, kv_lens[1]) and int((~torch.isnan(ring.float()).any(-1)).sum()) == n_live


@requires_cuda
def test_paged_kv_lens_at_the_int32_extremes_are_dead_in_both_kernels():
    """The fused prepare launch's compress arm and the step form on ``kv_lens`` {INT_MIN, INT_MIN + 1, -1, 0} beside a ``4`` control (S_q 4, every
    sequence on live pages and a live ring block): the ONE shared ``step_position`` derives the row-0 position from ``max(L, 0) - S_q``, so every
    ``L <= 0`` sequence is DEAD in both kernels -- every slot ``-1``, not one ring or cache byte of it moved -- instead of the int32 wrap at ``INT_MIN``
    landing its rows at a live positive position; the two kernels bitwise each other on every output, the control's row 3 completes block 0."""
    ch = _Chain(_GEOM_SMALL, batch=5, rows=4, pos0=(0, 0, 0, 0, 0), paged=True, page=16, cap=8)
    int_min = -(2**31)
    ch.kv_lens = torch.tensor([int_min, int_min + 1, -1, 0, 4], dtype=torch.int32, device="cuda")
    ch.compile_unfused()
    ch.compile_fused()
    ch.run_unfused()
    ch.run_fused()
    ch.assert_bitwise("kv_lens at the int32 extremes")
    cs = ch.Bf["cslot"].tolist()
    assert cs == [[-1] * 4] * 4 + [[-1, -1, -1, 0]], cs
    assert _same(ch.Bf["ring"][:5], ch.ring0[:5]), "a dead sequence (kv_lens <= 0) wrote a ring row"
    for b in range(4):
        assert bool((ch.kbar_row(ch.Bf, b, 0).float() == ch.sent).all()), f"sequence {b} wrote its block 0"
    assert bool((ch.kbar_row(ch.Bf, 4, 0).float() != ch.sent).any()), "the control did not write block 0"
    raw = ch.cols(ch.slab0, ch.o_ik, 1, ch.d_i)
    for j in range(4):
        assert _same(ch.Bf["ring"][5, j % ch.ring_n, 0], raw[4 * 4 + j, 0]), j
