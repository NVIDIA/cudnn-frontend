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
"""

import os
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
    """The 16-bit patterns of a bf16 / f16 tensor (a NaN-safe bitwise comparison: the pool sentinel is NaN)."""
    return x.contiguous().view(torch.int16)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype in (BF16, F16):
        return torch.equal(_bits(a), _bits(b))
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
    the ring state and the compressed-key cache; ``run_unfused`` / ``run_fused`` write into independent copies."""

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
    ):
        dev = "cuda"
        self.dtype, self.slot_dtype, self.layout, self.apply_norm, self.band = dtype, slot_dtype, layout, apply_norm, band
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
            else (QsaPrepareIndexArms(heads=h_i, head_dim=d_i, raw_pool=True, query=True, compress=True, n_commit=n_commit is not None) if band else None)
        )
        gen = torch.Generator(device=dev).manual_seed(seed)

        def rnd(*shape, std=1.0):
            return (torch.randn(*shape, generator=gen, device=dev, dtype=torch.float32) * std).to(dtype)

        self.slab0 = rnd(T, self.n)
        self.w_q, self.w_k = (1.0 + 0.05 * rnd(d).float()).to(dtype), (1.0 - 0.05 * rnd(d).float()).to(dtype)
        self.w_iq, self.w_ik = (1.0 + 0.03 * rnd(d_i).float()).to(dtype), (1.0 + 0.02 * rnd(d_i).float()).to(dtype)
        self.pos0 = torch.tensor(list(pos0) if pos0 is not None else [0] * batch, dtype=torch.int32, device=dev)
        self.n_commit = None if n_commit is None else torch.tensor(list(n_commit), dtype=torch.int32, device=dev)
        s_pos = int(self.pos0.max().item()) + rows + 8
        inv = 1.0 / (1e7 ** (torch.arange(0, self.rope, 2, device=dev, dtype=torch.float32) / self.rope))
        emb = torch.cat([torch.arange(s_pos, device=dev, dtype=torch.float32)[:, None] * inv] * 2, -1)
        self.cos_pos, self.sin_pos = emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]  # [1, S_pos, rope]: every sequence
        tok_pos = (self.pos0[:, None].long() + torch.arange(rows, device=dev)[None, :]).reshape(T)
        self.cos_t, self.sin_t = self.cos_pos[0, tok_pos].contiguous(), self.sin_pos[0, tok_pos].contiguous()  # [T, rope] per token
        self.n_pages = T // _PAGE + 3
        self.slot = torch.randperm(self.n_pages * _PAGE, generator=gen, device=dev)[:T].to(slot_dtype)
        if pad_every:
            self.slot[::pad_every] = -1
        if T > 2:
            self.slot[2] = self.n_pages * _PAGE + 7  # past the pool: writes nothing
        self.live = (self.slot >= 0) & (self.slot < self.n_pages * _PAGE)
        self.ring_n = max(8, min_ring_rows(rows, _POOL))
        self.ring0 = rnd(batch, self.ring_n, d_i)
        self.nb_cap = (s_pos + _POOL - 1) // _POOL
        self.sent = torch.full((), -7777.0, dtype=dtype).float().item()  # the sentinel AS THE DTYPE HOLDS IT
        self.comp0 = torch.full((batch, self.nb_cap, d_i), self.sent, device=dev, dtype=dtype)
        self.A = self._fresh()
        self.Bf = self._fresh()

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
            qi=torch.full((self.T, self.h_i, self.d_i), float("nan"), device="cuda", dtype=self.dtype),
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
            compile_qsa_compress(dtype=dt, d=self.d_i, rope_dim=self.rope, pool=1, has_seq_lens=False, table_dtype=dt) if (a is not None and a.query) else None
        )
        self.r_step = (
            compile_qsa_compress_step(dtype=dt, d=self.d_i, rope_dim=self.rope, pool=_POOL, has_n_commit=a.n_commit, table_dtype=dt)
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
        if self.r_step is not None:  # the ring + the block each committed row completes
            k_new = torch.as_strided(s, (self.B, self.S_q, self.d_i), (self.S_q * self.n, self.n, 1), s.storage_offset() + self.o_ik)
            run_qsa_compress_step(
                self.r_step,
                k_new,
                X["ring"],
                X["comp"],
                self.w_ik,
                self.cos_pos,
                self.sin_pos,
                self.pos0,
                X["cslot"],
                self.n_commit,
                eps=_EPS,
                norm_weight_offset=0.0,
                stream=st,
            )

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
                    pos0=self.pos0,
                    compressed_slot=X["cslot"],
                    seq_q=self.S_q,
                    n_commit=self.n_commit,
                )
        return kw

    def run_fused(self, X=None):
        X = self.Bf if X is None else X
        s = X["slab"]
        wq, wk = (self.w_q, self.w_k) if self.apply_norm else (None, None)
        run_qsa_prepare(
            self.r_fused,
            self.cols(s, self.o_q, self.h_q, self.d),
            self.cols(s, self.o_k, self.h_kv, self.d),
            self.cols(s, self.o_v, self.h_kv, self.d),
            None,
            None,
            wq,
            wk,
            self.cos_t,
            self.sin_t,
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
    ],
)
def test_every_arm_subset_is_bitwise_its_unfused_twins(arms):
    """The arms are traced by presence: each subset folds the others out and stays bitwise the kernels it keeps (the
    block binds the raw-pool-only form today; the serving step binds the compress and query arms)."""
    ch = _chain(_GEOM_SMALL, batch=2, rows=4, pos0=(3, 10), arms=arms, band=True)
    ch.assert_bitwise(f"arms {arms}")


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
    with pytest.raises(NotImplementedError, match="fuse_prepare=True with kv_source='paged'"):
        _make_block(
            dict(_SMALL_D256, causal_bottom_right=True),
            1,
            1,
            paged_kv_page_size=16,
            fuse_prepare=True,
            kv_source="paged",
            sample_k_cache=mk(4, 2, 16, 256),
            sample_v_cache=mk(4, 2, 16, 256),
            sample_block_table=mk(1, 4, dt=torch.int32),
        )
    with pytest.raises(NotImplementedError, match="fuse_prepare=True with kv_cache_dtype=torch.float8_e4m3fn"):
        _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16, fuse_prepare=True, kv_cache_dtype=torch.float8_e4m3fn)
    blk, *_ = _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16, fuse_prepare=True)
    assert blk.fuse_prepare and blk.paged_kv_page_size == 16 and isinstance(blk._prepare, _QsaPrepare)
    plain, *_ = _make_block(_SMALL_D256, 1, 64, paged_kv_page_size=16)
    assert not plain.fuse_prepare and plain._prepare is None


def _assert_prepare_compiles_with_pdl(stage, expected, monkeypatch):
    """``_QsaPrepare.compile`` hands ``use_pdl`` to ``compile_qsa_prepare`` for BOTH slot recipes (the flag is a compile key of the
    kernel: the launch attribute and the griddepcontrol pair are traced in), recorded through the kernel module's entry -- no device."""
    import cudnn.gated_attention_block.kernels.qsa_prepare as kernel_module

    seen = []
    monkeypatch.setattr(kernel_module, "compile_qsa_prepare", lambda **kw: seen.append(kw) or ("recipe", kw["slot_dtype"]))
    stage.compile()
    assert sorted((kw["slot_dtype"] for kw in seen), key=str) == [torch.int32, torch.int64]
    assert [kw["use_pdl"] for kw in seen] == [expected, expected], seen


@pytest.mark.parametrize("band", [False, True], ids=["dense", "qsa_band"])
def test_fuse_prepare_builds_one_stage_in_place_of_two_and_keeps_the_carve(band, monkeypatch):
    """With the knob the stage list carries ``qsa_prepare`` in the norm + RoPE slot and neither ``qk_norm_rope`` nor
    ``cache_write``; without it both; the workspace carve is identical (the stage reserves nothing); the fused stage's
    byte count is the two stages' minus nothing but the second read of K.  The fused stage takes the block's resolved ``use_pdl``
    and hands it to the compile (both slot recipes), as the two stages it replaces do: False under the AUTO reading off the decode
    form, True when the dense bf16 block declares it; on the band block an explicit True is the block's own typed decline."""
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
    if band:
        with pytest.raises(NotImplementedError, match="use_pdl=True is served for the bf16 / f16 block only"):
            _make_block(_SMALL_D256, 2, 128, fuse_prepare=True, use_pdl=True, **kw)
    else:
        pdl, *_ = _make_block(_SMALL_D256, 2, 128, fuse_prepare=True, use_pdl=True, **kw)
        assert pdl.use_pdl is True and pdl._prepare.use_pdl is True
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
