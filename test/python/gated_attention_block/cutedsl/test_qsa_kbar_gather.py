# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dense ``kbar`` gather + the extend step's device metadata (``kernels/qsa_kbar_gather.py``): the paged compressed-key cache
through the main block table into the contiguous scorer operand, the per-row visible counts, the per-piece clamped first position
(the memory guard of the compact-logits scratch) and the per-sequence report, all from ``kv_lens`` on device.

The acceptance is BITWISE against a host re-derivation of the module's own contract: every gathered row IS the cache row the table
names (a bit copy, bf16 and e4m3), a ``-1`` / out-of-range page and the rows a dead / short sequence's chunk can read are ZERO rows,
every row past the fill keeps its sentinel; the metadata for every ``kv_lens`` class (dead / shorter than the chunk / served / past the
table's reach) with one to three pieces per sequence.  The reach-guard cell runs the REAL scorer over a loose, plan-constant slab layout
with a sequence past the table's reach and asserts the guard past the slab untouched (SM100-family GPU: the scorer is a tcgen05
kernel; it skips elsewhere and is UNCOVERED there).  Plain vectorized LDG / STG, no shuffle -- these cells run on whatever CUDA
device is at hand.
"""

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.gated_attention_block.kernels.qsa_kbar_gather import (  # noqa: E402
    DEFAULT_THREADS_PER_CTA,
    SEQREP_DEAD,
    SEQREP_OVER,
    SEQREP_SHORT,
    build_qsa_kbar_gather,
    compile_qsa_kbar_gather,
    moved_bytes,
    run_qsa_kbar_gather,
    validate_gather_shape,
)

pytestmark = pytest.mark.L0

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
_CC = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
needs_sm100_family = pytest.mark.skipif(_CC is None or _CC[0] != 10, reason=f"the scorer is a tcgen05 kernel (cc 10.x); found {_CC}")
_E4M3 = torch.float8_e4m3fn
_D = 128
_POOL = 4


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _i32(values):
    return torch.tensor(list(values), dtype=torch.int32, device="cuda")


def _sent_of(dtype) -> float:
    """The sentinel AS THE DTYPE HOLDS IT (bf16 rounds -7777.0; e4m3 holds -448.0, its max magnitude)."""
    return torch.full((), -448.0 if dtype == _E4M3 else -7777.0, dtype=dtype).float().item()


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int8 if x.element_size() == 1 else torch.int16)


class _Cache:
    """A paged compressed-key cache of ``P / 4`` rows per page for ``batch`` sequences behind a PERMUTED block table (page 0 unused by a
    live sequence; a sequence in ``dead`` sits on page 0 throughout), filled with random rows in ``dtype`` (e4m3: the bf16 draw cast)."""

    def __init__(self, batch, page, max_pages, dtype, *, dead=(), seed=0):
        self.rpp, self.page, self.batch, self.dtype, self.max_pages = page // _POOL, page, batch, dtype, max_pages
        self.n_pages = 1 + batch * max_pages
        g = torch.Generator(device="cuda").manual_seed(seed + 7)
        perm = torch.randperm(self.n_pages - 1, generator=g, device="cuda") + 1
        self.table = perm.view(batch, max_pages).to(torch.int32).contiguous()
        for b in dead:
            self.table[b] = 0
        rows = torch.randn(self.n_pages, self.rpp, 1, _D, generator=g, device="cuda", dtype=torch.float32)
        self.cache = rows.to(torch.bfloat16).to(dtype) if dtype == _E4M3 else rows.to(dtype)

    @property
    def reach(self) -> int:
        return self.max_pages * self.rpp

    def row(self, b, j):
        """The cache row block ``j`` of sequence ``b`` names, or a zero row for a ``-1`` / out-of-range page (the kernel's convention)."""
        pg = int(self.table[b, j // self.rpp])
        if 0 <= pg < self.n_pages:
            return self.cache[pg, j % self.rpp, 0]
        return torch.zeros(_D, device="cuda", dtype=self.dtype)


def _expected(kv_lens, t, nb, pieces):
    """The module docstring's contract on the host: per sequence ``(n_keys, fill_end, counts[T], rep)`` and per piece ``qpos``."""
    seq, qpos = [], []
    for L in kv_lens:
        n_keys = min(max(L, 0) // _POOL, nb)
        fill_end = max(n_keys, min(nb, -(-t // _POOL)))
        counts = [min(max(((L - t + i + 1) // _POOL) if (L - t + i + 1) > 0 else 0, 0), nb) for i in range(t)]
        rep = n_keys
        if L > _POOL * nb:
            rep = SEQREP_OVER
        if 0 < L < t:
            rep = SEQREP_SHORT
        if L <= 0:
            rep = SEQREP_DEAD
        seq.append((n_keys, fill_end, counts, rep))
    for b, r0, rows in pieces:
        L = kv_lens[b]
        qpos.append(min(max(max(L - t, 0) + r0, 0), _POOL * nb + _POOL - 1 - rows))
    return seq, qpos


def _run(cx: _Cache, kv_lens, pieces, t, *, nb=None, threads=DEFAULT_THREADS_PER_CTA):
    """One gather launch over sentinel-filled outputs; returns ``(kbar, qpos, counts, rep, recipe)``."""
    nb = cx.reach if nb is None else nb
    kbar = torch.full((cx.batch, nb, _D), _sent_of(cx.dtype), device="cuda", dtype=cx.dtype)
    qpos = torch.full((len(pieces),), -9, dtype=torch.int32, device="cuda")
    counts = torch.full((cx.batch * t,), -9, dtype=torch.int32, device="cuda")
    rep = torch.full((cx.batch,), -9, dtype=torch.int32, device="cuda")
    r = build_qsa_kbar_gather(
        cx.cache,
        cx.table,
        _i32(kv_lens),
        _i32([p[0] for p in pieces]),
        _i32([p[1] for p in pieces]),
        _i32([p[2] for p in pieces]),
        kbar,
        qpos,
        counts,
        rep,
        t_rows=t,
        threads_per_cta=threads,
        stream=_stream(),
    )
    torch.cuda.synchronize()
    return kbar, qpos, counts, rep, r


def _assert_gather(cx: _Cache, kv_lens, t, kbar, label):
    """Every row of ``kbar`` against the contract: the cache row through the table below ``n_keys``, zeros up to the fill, the sentinel past it."""
    nb = int(kbar.shape[1])
    sent = torch.full((_D,), _sent_of(cx.dtype), device="cuda", dtype=cx.dtype)
    zero = torch.zeros(_D, device="cuda", dtype=cx.dtype)
    seq, _ = _expected(kv_lens, t, nb, [])
    for b, (n_keys, fill_end, _, _) in enumerate(seq):
        for j in range(nb):
            want = cx.row(b, j) if j < n_keys else (zero if j < fill_end else sent)
            assert torch.equal(_bits(kbar[b, j]), _bits(want)), f"{label}: sequence {b} row {j} (n_keys {n_keys}, fill {fill_end})"


# ---------------------------------------------------------------------------
# Host rows -- no GPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "d, elem_bytes, pool, threads, match",
    [
        (128, 4, 4, 128, "1 or 2 bytes"),
        (124, 2, 4, 128, "multiple of 16"),
        (128, 2, 4, 100, "multiple of the 16 lanes"),
        (128, 1, 4, 100, "multiple of the 8 lanes"),
        (128, 2, 0, 128, "pool"),
    ],
)
def test_validate_gather_shape_rejects(d, elem_bytes, pool, threads, match):
    with pytest.raises(ValueError, match=match):
        validate_gather_shape(d, elem_bytes, pool, threads)


def test_validate_gather_shape_accepts_the_indexer_geometry_and_the_report_values_are_distinct():
    validate_gather_shape(_D, 2, _POOL, 128)
    validate_gather_shape(_D, 1, _POOL, 128)
    assert (SEQREP_DEAD, SEQREP_SHORT, SEQREP_OVER) == (-1, -2, -3)
    assert moved_bytes(32768, _D) == 2 * 32768 * _D * 2 and moved_bytes(10, _D, elem_bytes=1) == 2 * 10 * _D


def test_compile_rejects_unsupported_cache_dtypes():
    with pytest.raises(ValueError, match="bf16 / f16 or torch.float8_e4m3fn"):
        compile_qsa_kbar_gather(dtype=torch.float32)


# ---------------------------------------------------------------------------
# The gather: a bit copy through the table, zeros where the contract says, the sentinel elsewhere
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, _E4M3], ids=["bf16", "e4m3"])
@pytest.mark.parametrize("page", [16, 32, 64, 128])
def test_gather_rows_are_the_cache_rows_through_the_table_bitwise(page, dtype):
    """B = 3 behind a permuted table at page 16 / 32 / 64 / 128 (4 .. 32 compressed rows per page): a LONG live sequence whose table row
    carries a ``-1`` page and an out-of-range page (zero rows there, the cache rows elsewhere, bit for bit), a SHORT live one (``n_keys <
    ceil(T / 4)``: zero rows up to ``ceil(T / 4)``), a DEAD one (``kv_lens 0``, page 0 throughout: zeros up to ``ceil(T / 4)``); every row
    past the fill keeps its sentinel on every sequence."""
    t, max_pages = 37, 5
    cx = _Cache(3, page, max_pages, dtype, dead=(2,), seed=page)
    rpp = cx.rpp
    cx.table[0, 2] = -1  # an unassigned page inside the long sequence's keys
    cx.table[0, 3] = cx.n_pages + 5  # a page past the cache
    kv_lens = [max_pages * page - 3, t, 0]  # long (reach - 3 tokens), short (= T: one complete block, 10 rows of the chunk), dead
    kbar, qpos, counts, rep, r = _run(cx, kv_lens, [(0, 0, t), (1, 0, t), (2, 0, t)], t)
    assert r.dtype == dtype and r.elem_bytes == (1 if dtype == _E4M3 else 2) and kbar.dtype == dtype
    _assert_gather(cx, kv_lens, t, kbar, f"page {page} {dtype}")
    seq, want_q = _expected(kv_lens, t, cx.reach, [(0, 0, t), (1, 0, t), (2, 0, t)])
    assert rep.tolist() == [s[3] for s in seq] == [(max_pages * page - 3) // _POOL, t // _POOL, SEQREP_DEAD]
    assert qpos.tolist() == want_q
    # the -1 and the out-of-range page of sequence 0 are ZERO rows (the paged scorer's convention), the neighbouring pages real rows
    zero = torch.zeros(_D, device="cuda", dtype=dtype)
    for lp in (2, 3):
        for row in range(rpp):
            assert torch.equal(_bits(kbar[0, lp * rpp + row]), _bits(zero))
    assert not torch.equal(_bits(kbar[0, 1 * rpp]), _bits(zero))


@requires_cuda
def test_gather_metadata_for_every_kv_lens_class():
    """Four sequences -- DEAD (0), SHORT (0 < L < T), SERVED (T <= L <= 4 NB) and OVER the reach (L > 4 NB, the stack-bug case) -- with
    one to three select pieces each (the plan-constant piece table): ``ix_qpos`` = ``max(L - T, 0) + r0`` clamped at ``4 NB + 3 - rows``
    (and at 0), ``ix_counts`` from the TRUE position clamped to ``[0, NB]`` (0 on a negative position), ``ix_seqrep`` ``-1 / -2 / n_keys /
    -3``, the kbar rows of every class as the contract says (the OVER sequence filled to NB from its table)."""
    t, page, max_pages = 48, 16, 6
    cx = _Cache(4, page, max_pages, torch.bfloat16, dead=(0,), seed=3)
    nb = cx.reach  # 24 rows = 96 tokens of reach
    kv_lens = [0, 30, 70, _POOL * nb + 5]
    pieces = [(0, 0, 48), (1, 0, 20), (1, 20, 28), (2, 0, 16), (2, 16, 16), (2, 32, 16), (3, 0, 48)]
    kbar, qpos, counts, rep, _ = _run(cx, kv_lens, pieces, t)
    seq, want_q = _expected(kv_lens, t, nb, pieces)
    assert rep.tolist() == [SEQREP_DEAD, SEQREP_SHORT, 70 // _POOL, SEQREP_OVER]
    assert qpos.tolist() == want_q
    # the OVER sequence's pieces are CLAMPED: 4 NB + 3 - rows = 4 x 24 + 3 - 48 = 51 < L - T = 53; the served one is not (70 - 48 + r0)
    assert want_q[-1] == _POOL * nb + _POOL - 1 - 48 == 51 and want_q[3:6] == [22, 38, 54]
    # the short sequence's pieces start at r0 (L - T < 0 -> 0); the dead one's at 0
    assert want_q[:3] == [0, 0, 20]
    got_counts = counts.view(4, t).tolist()
    for b, (_, _, want_c, _) in enumerate(seq):
        assert got_counts[b] == want_c, f"sequence {b}: counts differ from the position rule"
    assert got_counts[0] == [0] * t  # dead: 0 everywhere
    assert got_counts[3][0] == (kv_lens[3] - t + 1) // _POOL == 13 and got_counts[3][-1] == nb  # over: the TRUE positions, clamped at NB at the end
    assert got_counts[2][0] == (70 - 48 + 1) // _POOL and got_counts[2][-1] == 70 // _POOL
    assert got_counts[1][:21] == [0] * 21 and got_counts[1][21] == 1  # L - T = -18: rows 0 .. 20 sit at positions -18 .. 2 (no complete block)
    _assert_gather(cx, kv_lens, t, kbar, "kv_lens classes")


@requires_cuda
def test_gather_dead_sequence_positions_never_wrap_at_the_int32_extremes():
    """``kv_lens`` in {INT_MIN, INT_MIN + 1, -1, 0, 4} with T 4, NB 4 and one piece ``(b, 0, 4)`` per sequence: every ``L <= 0`` is DEAD with
    zero counts, ``q_pos0 = r0 = 0`` and the report ``-1`` whatever its magnitude -- the positions are derived from ``max(L, 0)``, so the int32
    ``L - T`` that wrapped positive at ``INT_MIN`` (counts [4, 4, 4, 0], ``q_pos0`` 15 under a correct ``-1`` report) never enters; the ``L = 4``
    control is served (counts [0, 0, 0, 1], ``q_pos0`` 0, report 1).  The kbar rows follow the contract (one zero row for the dead ones)."""
    t, int_min = 4, -(2**31)
    cx = _Cache(5, 16, 1, torch.bfloat16, dead=(0, 1, 2, 3), seed=11)
    nb = cx.reach
    assert nb == 4
    kv_lens = [int_min, int_min + 1, -1, 0, 4]
    pieces = [(b, 0, t) for b in range(5)]
    kbar, qpos, counts, rep, _ = _run(cx, kv_lens, pieces, t)
    seq, want_q = _expected(kv_lens, t, nb, pieces)  # host ints: no wrap in the oracle
    assert rep.tolist() == [SEQREP_DEAD] * 4 + [1]
    assert qpos.tolist() == want_q == [0] * 5
    got = counts.view(5, t).tolist()
    assert got[:4] == [[0] * t] * 4, f"a dead sequence's counts are not zero: {got[:4]}"
    assert got[4] == [0, 0, 0, 1] == seq[4][2]
    _assert_gather(cx, kv_lens, t, kbar, "int32 extremes")


@requires_cuda
def test_gather_is_one_launch_and_allocates_nothing():
    """MEASURED: ONE kernel launch (``torch.profiler``'s census), no allocation and no host sync on the execute path."""
    from torch.profiler import ProfilerActivity, profile

    t = 20
    cx = _Cache(2, 16, 4, torch.bfloat16, seed=9)
    pieces = [(0, 0, t), (1, 0, t)]
    r = compile_qsa_kbar_gather(dtype=torch.bfloat16)
    kbar = torch.empty(2, cx.reach, _D, device="cuda", dtype=torch.bfloat16)
    qpos, counts, rep = (torch.empty(n, dtype=torch.int32, device="cuda") for n in (2, 2 * t, 2))
    kv = _i32([40, 0])
    ps, pr0, prw = (_i32([p[i] for p in pieces]) for i in range(3))  # the piece table from `pieces`: seq, r0, rows

    def launch():
        run_qsa_kbar_gather(r, cx.cache, cx.table, kv, ps, pr0, prw, kbar, qpos, counts, rep, t_rows=t, stream=_stream())

    launch()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            launch()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before, "the gather allocated on the execute path"
    prof = profile(activities=[ProfilerActivity.CUDA])
    try:
        prof.start()
    except Exception:  # noqa: BLE001 -- CUPTI unavailable on this box: the bitwise cells stand
        pytest.skip("the CUDA profiler recorded no kernel activity on this box (CUPTI unavailable)")
    try:
        launch()
        torch.cuda.synchronize()
    finally:
        prof.stop()
    names = [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and "memset" not in e.name.lower() and "memcpy" not in e.name.lower()
    ]
    if not names:
        pytest.skip("the CUDA profiler recorded no kernel activity on this box (CUPTI unavailable)")
    assert len(names) == 1 and "frost_qsa_kbar_gather" in names[0], names


@requires_cuda
def test_gather_two_launches_are_bitwise_and_one_artifact_serves_every_page_size_batch_and_reach():
    """The key is ``(dtype, d, pool, threads, use_pdl, device)``: page 16 / 64 / 128, B 1 / 3, reach 8 .. 128 rows, T 1 .. 300 through
    ONE compiled object; two launches on the same inputs bitwise."""
    r0 = compile_qsa_kbar_gather(dtype=torch.bfloat16)
    assert compile_qsa_kbar_gather(dtype=torch.bfloat16).compiled is r0.compiled
    assert compile_qsa_kbar_gather(dtype=_E4M3).compiled is not r0.compiled
    for page, batch, max_pages, t in ((16, 1, 2, 1), (64, 3, 5, 300), (128, 2, 4, 7), (16, 3, 32, 128)):
        cx = _Cache(batch, page, max_pages, torch.bfloat16, seed=page + batch)
        kv_lens = [min(cx.reach * _POOL, t + 9 * (b + 1)) for b in range(batch)]
        pieces = [(b, 0, t) for b in range(batch)]
        a = _run(cx, kv_lens, pieces, t)
        b_ = _run(cx, kv_lens, pieces, t)
        assert a[4].compiled is r0.compiled
        assert torch.equal(_bits(a[0]), _bits(b_[0])) and torch.equal(a[1], b_[1]) and torch.equal(a[2], b_[2]) and torch.equal(a[3], b_[3])
        _assert_gather(cx, kv_lens, t, a[0], f"page {page} B {batch} reach {cx.reach} T {t}")


@requires_cuda
def test_gather_host_refuses_a_chunk_past_the_table_reach_and_malformed_operands():
    """``T > 4 x n_blocks_max`` is refused naming both numbers (the clamp's precondition); a kbar reach past the table's; the dtype
    refusals both ways (an e4m3 cache under a bf16 recipe and the converse, naming the recipe's dtype); shape / stride / dtype / device
    refusals of every operand, typed and naming it."""
    t = 8
    cx = _Cache(2, 16, 3, torch.bfloat16, seed=5)
    r = compile_qsa_kbar_gather(dtype=torch.bfloat16)
    r8 = compile_qsa_kbar_gather(dtype=_E4M3)
    nb = cx.reach
    kbar = torch.empty(2, nb, _D, device="cuda", dtype=torch.bfloat16)
    kv, ps, pr0, prw = _i32([20, 0]), _i32([0, 1]), _i32([0, 0]), _i32([t, t])
    qpos, counts, rep = (torch.empty(n, dtype=torch.int32, device="cuda") for n in (2, 2 * t, 2))

    def run(rr=r, cache=cx.cache, table=cx.table, kv=kv, ps=ps, pr0=pr0, prw=prw, kb=kbar, qp=qpos, ct=counts, rp=rep, t_rows=t):
        run_qsa_kbar_gather(rr, cache, table, kv, ps, pr0, prw, kb, qp, ct, rp, t_rows=t_rows, stream=_stream())

    with pytest.raises(ValueError, match=rf"T={_POOL * nb + 1} rows exceeds the table's reach of {_POOL} x {nb} = {_POOL * nb} tokens"):
        run(t_rows=_POOL * nb + 1, ct=torch.empty(2 * (_POOL * nb + 1), dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="t_rows .* must be >= 1"):
        run(t_rows=0, ct=torch.empty(0, dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="rows past the table's reach could only be zeros"):
        run(kb=torch.empty(2, nb + 1, _D, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(ValueError, match=r"index_kv_cache is torch.float8_e4m3fn but this artifact gathers a torch.bfloat16 cache"):
        run(cache=cx.cache.to(_E4M3))
    with pytest.raises(ValueError, match=r"index_kv_cache is torch.bfloat16 but this artifact gathers a torch.float8_e4m3fn cache"):
        run(rr=r8)
    with pytest.raises(ValueError, match=r"ix_kbar is torch.float8_e4m3fn but this artifact gathers a torch.bfloat16 cache"):
        run(kb=kbar.to(_E4M3))
    with pytest.raises(ValueError, match="ix_kbar must be contiguous"):
        run(kb=torch.empty(2, nb, 2 * _D, device="cuda", dtype=torch.bfloat16)[:, :, :_D])
    with pytest.raises(ValueError, match=r"ix_kbar must be \[B=2, n_blocks_max >= 1, D=128\]"):
        run(kb=torch.empty(3, nb, _D, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="block_table must have contiguous columns"):
        run(table=cx.table.t().contiguous().t())
    with pytest.raises(ValueError, match="kv_lens must be a contiguous int32 CUDA tensor of shape"):
        run(kv=kv.to(torch.int64))
    with pytest.raises(ValueError, match="compiled for CUDA device"):  # the artifact's device is part of its key; operands elsewhere are refused
        run(rr=r._replace(device=r.device + 1))
    with pytest.raises(ValueError, match="piece_r0 must be a contiguous int32 CUDA tensor of shape"):
        run(pr0=_i32([0, 0, 0]))
    with pytest.raises(ValueError, match="piece_seq must be a"):
        run(ps=_i32([]))
    with pytest.raises(ValueError, match=r"ix_counts must be a contiguous int32 tensor of B x T = 2 x 8 entries"):
        run(ct=torch.empty(2 * t - 1, dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="ix_seqrep must be a contiguous int32 CUDA tensor of shape"):
        run(rp=torch.empty(3, dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match=r"index_kv_cache must be \[num_pages >= 1, P / pool >= 1, 1, D=128\]"):
        run(cache=cx.cache[:, :, :, :64].contiguous().view(cx.n_pages, cx.rpp, 1, 64))
    with pytest.raises(ValueError, match="must be a CUDA tensor"):
        run(cache=cx.cache.cpu())


# ---------------------------------------------------------------------------
# The reach-guard cell: the clamp keeps the REAL scorer's compact slab inside the loose, plan-constant regions
# ---------------------------------------------------------------------------


@requires_cuda
@needs_sm100_family
def test_gather_clamp_keeps_the_dsa_compact_slab_inside_the_loose_regions():
    """A sequence with ``kv_lens = max_pages x P + 5`` (past the table's reach: the stack-bug case) next to a served one.  The gather clamps its
    ``q_pos0`` at ``4 NB + 3 - rows`` and reports ``-3``; the compact-logits scorer (``qsa_select_forward``, the DSA's THD path) then runs
    over LOOSE plan-constant bases ``base_{p+1} = base_p + rows_p x NB`` on a sentinel-filled scratch with a guard past the window's dense
    bound: the guard is untouched, nothing faults, and the served sequence's selection equals its tight-prefix twin as a set."""
    from cudnn import DSA
    from cudnn.QSA.selection import QSA_BLOCK_SIZE, QSA_TOP_K, qsa_select_forward

    t, page, max_pages = 64, 16, 8
    cx = _Cache(2, page, max_pages, torch.bfloat16, seed=11)
    nb = cx.reach  # 32 rows
    kv_lens = [t + 40, max_pages * page + 5]  # served (26 keys), OVER the reach
    pieces = [(0, 0, t), (1, 0, t)]
    kbar, qpos, counts, rep, _ = _run(cx, kv_lens, pieces, t)
    assert rep.tolist() == [(t + 40) // _POOL, SEQREP_OVER]
    assert qpos.tolist() == [40, _POOL * nb + _POOL - 1 - t]
    dev = kbar.device
    qi = torch.randn(2 * t, 4, _D, device=dev, dtype=torch.bfloat16)
    cu = _i32([0, t, 2 * t])
    guard = 4096
    loose = torch.tensor([0, t * nb, 2 * t * nb], dtype=torch.int64, device=dev)
    cand = torch.full((2 * t * nb + guard,), 1.5e30, dtype=torch.float32, device=dev)
    ids = torch.full((2 * t, QSA_TOP_K), -7, dtype=torch.int32, device=dev)
    scores = torch.full((2 * t, QSA_TOP_K), 3.0, dtype=torch.float32, device=dev)
    qsa_select_forward(qi, kbar, cu, q_pos0=qpos, max_seqlen_q=t, cand_buffer=cand, cand_batch_offsets=loose, block_ids_out=ids, scores_out=scores)
    torch.cuda.synchronize()
    assert bool((cand[2 * t * nb :] == 1.5e30).all()), "the scorer wrote past the loose slab: the clamp did not hold"
    # the served sequence: the SAME set as a tight-prefix call on its own
    cu0 = _i32([0, t])
    cu_k = torch.arange(2, dtype=torch.int32, device=dev) * nb
    offs, n_floats = DSA.compress_topk_cand_buffer_size_thd(cu0, cu_k, QSA_BLOCK_SIZE, qpos[:1].contiguous())
    tight = qsa_select_forward(
        qi[:t],
        kbar[:1],
        cu0,
        q_pos0=qpos[:1].contiguous(),
        max_seqlen_q=t,
        cand_buffer=torch.empty(n_floats, dtype=torch.float32, device=dev),
        cand_batch_offsets=offs,
    )
    torch.cuda.synchronize()
    assert torch.equal(torch.sort(ids[:t]).values, torch.sort(tight["block_ids"]).values)
    # the OVER sequence is position-MISALIGNED, as the -3 report says: the scorer's list lengths follow the CLAMPED first position
    # (floor((q_pos0 + i + 1) / 4), every one <= NB by the clamp), while ix_counts carry the TRUE position's rule -- the two differ here
    rows = torch.arange(t, device=dev, dtype=torch.int32)
    lens_over = (ids[t:] >= 0).sum(-1).to(torch.int32)
    assert torch.equal(lens_over, ((qpos[1] + rows + 1) // QSA_BLOCK_SIZE).clamp(max=QSA_TOP_K)), "the OVER sequence's lists do not follow the clamped position"
    assert int(lens_over.max()) <= nb and not torch.equal(
        lens_over, counts.view(2, t)[1]
    ), "the clamped sequence should read misaligned against the true-position counts"
