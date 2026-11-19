# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SM100 D256 swap-AB split decode tile reducing its partials inside the launch (``CLUSTER_K_REDUCE``).

Forward only, over a paged cache with per-batch lengths. Every cell runs the two-launch form and the in-launch form of the
same split on the same inputs and compares raw bytes, beside an independent float64 reference. The module has no ``test_``
prefix: ``test_sdpa_fwd_decode_d256_sm100.py`` exposes it through a guarded class.
"""

import math

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot

D = 256
H, KH = 24, 4
G = H // KH
# A partial the reduction reads but no chunk wrote would be a finite, dominant term: 3e30 as Stats, 3e30 as O.
WS_POISON = 3.0e30
O_ATOL = {torch.float16: 2e-2, torch.bfloat16: 5e-2}
LSE_ATOL = 5e-3


def _device():
    from frost_test_utils import _dsl_installed

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("the in-launch reduction is verified on compute capability 10.0")
    if not _dsl_installed():
        pytest.skip("requires a supported CuTe DSL")


class Runner:
    """One declaration (dtype, tokens, declared cache, page, split, packing, mask) over a library of sequences.

    A batch is an ordered list of sequence ids; a sequence keeps its Q rows, K/V rows and lengths in every batch, while its
    pages land on other pool pages. ``run(order, reduce)`` executes the standalone adapter and returns the whole O and Stats
    storage (the declared rows and a sentinel margin) plus the workspace.
    """

    def __init__(
        self,
        *,
        dtype=torch.bfloat16,
        s_q=1,
        declared=2048,
        page=64,
        split=8,
        pack=True,
        causal=False,
        stats=True,
        lens=(2048,),
        q_lens=None,
        seed=0,
        margin=1,
        o_offset=0,
    ):
        self.dtype, self.s_q, self.declared, self.page, self.split, self.pack, self.causal, self.stats = dtype, s_q, declared, page, split, pack, causal, stats
        # margin: token rows of NaN storage behind the declared ones; o_offset: elements the O storage starts past an aligned base.
        self.margin, self.o_offset = margin, o_offset
        self.lens = list(lens)
        self.q_lens = list(q_lens) if q_lens is not None else [s_q] * len(self.lens)
        self.pages = -(-declared // page)
        rng = torch.Generator().manual_seed(seed)
        n = len(self.lens)
        self.q = torch.randn(n, s_q, H, D, generator=rng).to(dtype)
        self.k = torch.randn(n, self.pages * page, KH, D, generator=rng).to(dtype)
        self.v = torch.randn(n, self.pages * page, KH, D, generator=rng).to(dtype)
        self.seed = seed
        self._plans = {}

    def batch(self, order):
        b, dev = len(order), "cuda"
        index = torch.tensor(order)
        rng = torch.Generator().manual_seed(self.seed * 7919 + b * 31 + sum((i + 1) * s for i, s in enumerate(order)) % 104729)
        pool = b * self.pages + 3
        table = torch.randperm(pool, generator=rng)[: b * self.pages].view(b, self.pages).int()
        k_pool = torch.zeros(pool, KH, self.page, D, dtype=self.dtype)
        v_pool = torch.zeros(pool, KH, self.page, D, dtype=self.dtype)
        k_seq = self.k.index_select(0, index).view(b * self.pages, self.page, KH, D).permute(0, 2, 1, 3)
        v_seq = self.v.index_select(0, index).view(b * self.pages, self.page, KH, D).permute(0, 2, 1, 3)
        k_pool[table.view(-1).long()] = k_seq
        v_pool[table.view(-1).long()] = v_seq
        # O and Stats sit inside larger NaN-filled storage: the token rows past the declared ones must stay untouched.
        rows = self.s_q + self.margin
        o_flat = torch.full((b * rows * H * D + 8,), float("nan"), dtype=self.dtype, device=dev)
        o_store = o_flat[self.o_offset : self.o_offset + b * rows * H * D].view(b, rows, H, D)
        lse_store = torch.full((b, H, rows), float("nan"), device=dev)
        return dict(
            order=list(order),
            q=self.q.index_select(0, index).to(dev).transpose(1, 2),  # BHSD strides over BSHD storage
            k=k_pool.to(dev),
            v=v_pool.to(dev),
            table=table.to(dev),
            kl=torch.tensor([self.lens[s] for s in order], dtype=torch.int32, device=dev),
            ql=torch.tensor([self.q_lens[s] for s in order], dtype=torch.int32, device=dev),
            o_store=o_store,
            lse_store=lse_store,
            o=o_store[:, : self.s_q].transpose(1, 2),
            lse=lse_store[:, :, : self.s_q],
        )

    def plan(self, t, reduce):
        from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

        key = (len(t["order"]), bool(reduce))
        if key not in self._plans:
            api = SdpaFwdDslSm100(
                t["q"],
                t["k"],
                t["v"],
                t["o"],
                t["lse"] if self.stats else None,
                is_causal=self.causal,
                causal_bottom_right=self.causal,
                seq_q_lens_present=True,
                seq_kv_lens_present=True,
                paged_page_size=self.page,
                paged_max_seq_len_kv=self.declared,
                paged_table_stride=t["table"].stride(),
                split_kv=self.split,
                pack_gqa=self.pack,
                **({"cluster_reduce": True} if reduce else {}),
            )
            api.check_support()
            api.compile()
            assert api._k_mod.__file__.endswith("sm100/d256_f16_swap_ab/kernel.py") and api._k_mod.N_Q == 16
            assert api._dense_spec.fused_split == bool(reduce)
            assert (api._dense_spec.combine.fn is None) == bool(reduce)
            self._plans[key] = api
        return self._plans[key]

    def workspace(self, api):
        words = -(-max(api.scratch_workspace_bytes(), 16) // 4)
        return torch.full((words,), WS_POISON, device="cuda")

    def launch(self, t, api, ws):
        api.execute(
            t["q"],
            t["k"],
            t["v"],
            t["o"],
            lse_tensor=t["lse"] if self.stats else None,
            seq_kv_lens=t["kl"],
            seq_q_lens=t["ql"],
            block_table=t["table"],
            workspace=ws,
        )

    def run(self, order, reduce, ws=None, t=None):
        t = self.batch(order) if t is None else t
        api = self.plan(t, reduce)
        ws = self.workspace(api) if ws is None else ws
        t["o_store"].fill_(float("nan"))
        t["lse_store"].fill_(float("nan"))
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")  # lengths and tables are read on the device
        try:
            self.launch(t, api, ws)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        torch.cuda.synchronize()
        return t, api, ws

    def reference(self, t):
        """float64 O [b, s_q, H, D] and natural-log Stats [b, H, s_q] under the declaration's own mask and lengths."""
        b = len(t["order"])
        o = torch.zeros(b, self.s_q, H, D, dtype=torch.float64)
        lse = torch.full((b, H, self.s_q), -math.inf, dtype=torch.float64)
        for row, s in enumerate(t["order"]):
            nk, nq = self.lens[s], min(self.q_lens[s], self.s_q)
            if nk == 0 or nq == 0:
                continue
            q = self.q[s].double()  # [s_q, H, D]
            k = self.k[s, :nk].double().repeat_interleave(G, dim=1)  # [nk, H, D]
            v = self.v[s, :nk].double().repeat_interleave(G, dim=1)
            scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(D)
            if self.causal:
                key = torch.arange(nk).view(1, 1, nk)
                tok = torch.arange(self.s_q).view(1, self.s_q, 1)
                scores = scores.masked_fill(key > tok + nk - nq, -math.inf)
            row_lse = torch.logsumexp(scores, -1)  # [H, s_q]
            dead = torch.isinf(row_lse)
            p = torch.where(dead.unsqueeze(-1), torch.zeros_like(scores), torch.softmax(scores, -1))
            out = torch.einsum("hqk,khd->qhd", torch.nan_to_num(p), v)
            out[nq:] = 0
            row_lse[:, nq:] = -math.inf
            o[row], lse[row] = out, row_lse
        return o, lse

    def check(self, t, label=""):
        """The declared rows against the reference; dead rows exactly 0 / -inf; the storage outside the graph untouched."""
        ref_o, ref_lse = self.reference(t)
        got = t["o_store"][:, : self.s_q].cpu().double()
        assert torch.isfinite(got).all(), f"{label}: non-finite O"
        torch.testing.assert_close(got, ref_o, atol=O_ATOL[self.dtype], rtol=0, msg=lambda m: f"{label}: {m}")
        dead = torch.isinf(ref_lse)  # [b, H, s_q]
        if dead.any():
            assert got.transpose(1, 2)[dead].abs().max().item() == 0.0, f"{label}: a row without keys must store O = 0"
        assert torch.isnan(t["o_store"][:, self.s_q :]).all(), f"{label}: O storage past the declared tokens was written"
        if not self.stats:
            assert torch.isnan(t["lse_store"]).all(), f"{label}: Stats storage written without a Stats output"
            return dict(o=t["o_store"][:, : self.s_q])
        got_lse = t["lse_store"][:, :, : self.s_q].cpu().double()
        torch.testing.assert_close(got_lse[~dead], ref_lse[~dead], atol=LSE_ATOL, rtol=0, msg=lambda m: f"{label}: {m}")
        if dead.any():
            assert (got_lse[dead] == -math.inf).all(), f"{label}: a row without keys must store Stats = -inf"
        assert torch.isnan(t["lse_store"][:, :, self.s_q :]).all(), f"{label}: Stats storage past the declared tokens was written"
        return dict(o=t["o_store"][:, : self.s_q], lse=t["lse_store"][:, :, : self.s_q])

    def slabs(self, api, ws, b):
        """The partial O and Stats slabs of a workspace, as (splits, b, ...) views."""
        combine = api._dense_spec.combine
        n_o, n_lse = self.split * b * self.s_q * H * D, self.split * b * H * self.s_q
        assert combine.lse_offset % 4 == 0 and combine.lse_offset >= 4 * n_o
        o = ws[:n_o].view(self.split, b, self.s_q, H, D)
        lse = ws[combine.lse_offset // 4 : combine.lse_offset // 4 + n_lse].view(self.split, b, H, self.s_q)
        return o, lse

    def separate_combine(self, two_launch_api, o_partial, lse_partial, b):
        """The two-launch plan's own reduction entry over given partial slabs, into fresh final buffers."""
        o = torch.full((b, self.s_q, H, D), float("nan"), dtype=self.dtype, device="cuda")
        lse = torch.full((b, H, self.s_q), float("nan"), device="cuda")
        stream = torch.cuda.current_stream().cuda_stream
        final_lse, lse_strides = (lse.data_ptr(), lse.stride()) if self.stats else (None, (0, 0, 0))
        two_launch_api._dense_spec.combine.fn(
            o_partial.data_ptr(), lse_partial.data_ptr(), o.data_ptr(), final_lse, (b, H, self.s_q, D), self.split, o.stride(), lse_strides, stream
        )
        torch.cuda.synchronize()
        return o, lse


def _same_bytes(a, b):
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def _compare_forms(runner, order, *, reordered_control=False):
    """Both forms on one batch: each against the reference, then raw bytes of O, Stats and the partial slabs, and the
    separate reduction over the slab the in-launch form left. Returns the number of compared O elements."""
    b = len(order)
    t2, api2, ws2 = runner.run(order, reduce=False)
    two = snapshot(runner.check(t2, "two-launch"))
    slab_o2, slab_lse2 = (x.clone() for x in runner.slabs(api2, ws2, b))
    t1, api1, ws1 = runner.run(order, reduce=True)
    one = snapshot(runner.check(t1, "in-launch"))
    assert_bitwise(one, two, "in-launch reduction against the two-launch form")
    slab_o1, slab_lse1 = runner.slabs(api1, ws1, b)
    assert _same_bytes(slab_o1, slab_o2) and _same_bytes(slab_lse1, slab_lse2), "the partial slabs of the two forms differ"
    o_sep, lse_sep = runner.separate_combine(api2, slab_o1.contiguous(), slab_lse1.contiguous(), b)
    assert _same_bytes(o_sep, t1["o_store"][:, : runner.s_q]), "the separate reduction over the in-launch slab differs from the in-launch output"
    if runner.stats:
        assert _same_bytes(lse_sep, t1["lse_store"][:, :, : runner.s_q]), "the separate reduction's Stats differ from the in-launch Stats"
    if reordered_control:
        # The control: the same partials in another split order must NOT reproduce the bytes (three or more comparable
        # chunks), or the byte cells above could not see a reduction in another order.
        o_rev, _ = runner.separate_combine(api2, slab_o1.flip(0).contiguous(), slab_lse1.flip(0).contiguous(), b)
        assert not _same_bytes(o_rev, o_sep), "a reduction in the reverse split order reproduced every byte: the byte cells have no power"
    return b * runner.s_q * H * D


# name -> Runner keyword arguments. Lengths are per sequence; "declared" is the cache length the plan is compiled for.
CELLS = {
    # packed 1 and 2 tokens (the first plan of the class)
    "bf16-q1-p-s8-full": dict(s_q=1, split=8, lens=(2048, 1920)),
    "bf16-q1-p-s4-declared1536": dict(s_q=1, split=4, declared=1536, lens=(1536, 1000)),
    "bf16-q1-p-s2-ragged": dict(s_q=1, split=2, lens=(2048, 129, 0, 1)),
    "bf16-q2-p-s8-causal": dict(s_q=2, split=8, causal=True, lens=(2048, 1027), q_lens=(2, 1)),
    "f16-q2-p-s4-causal": dict(dtype=torch.float16, s_q=2, split=4, causal=True, lens=(2048, 513)),
    # fewer live tiles than chunks, one key, an empty sequence, tile and page edges under one long declaration
    "bf16-q1-p-s8-short-live": dict(s_q=1, split=8, lens=(0, 1, 64, 127, 128, 129, 1000, 2048)),
    "bf16-q1-p-s8-declared512": dict(s_q=1, split=8, declared=512, lens=(512, 300, 1)),  # more chunks than declared tiles
    # ring depth and beyond: 1 .. 5 tiles per chunk
    "bf16-q1-p-s4-ring": dict(s_q=1, split=4, declared=2560, lens=(512, 1024, 1536, 2048, 2560)),
    # pages 16 and 32, lengths that end inside a page and inside a tile
    "bf16-q1-p-s8-page16": dict(s_q=1, split=8, page=16, lens=(2048, 1010)),
    "f16-q1-p-s8-page32": dict(dtype=torch.float16, s_q=1, split=8, page=32, lens=(2048, 1000, 100)),
    # unpacked 3 .. 8 tokens (the decode tile's plan of a multi-token step), bottom-right causal
    "bf16-q3-u-s4-causal": dict(s_q=3, split=4, pack=False, causal=True, lens=(2048, 700)),
    "bf16-q6-u-s4-causal": dict(s_q=6, split=4, pack=False, causal=True, lens=(2048, 1536, 5), q_lens=(6, 6, 6)),
    "f16-q4-u-s2-causal": dict(dtype=torch.float16, s_q=4, split=2, pack=False, causal=True, lens=(2048, 3)),
    "bf16-q8-u-s8-trim": dict(s_q=8, split=8, pack=False, causal=True, lens=(2048, 2048), q_lens=(8, 5)),
    # O storage that is not 8-byte addressable takes the element store in both forms
    "bf16-q1-p-s8-odd-output": dict(s_q=1, split=8, lens=(2048, 77), o_offset=1),
    "f16-q6-u-s4-odd-output": dict(dtype=torch.float16, s_q=6, split=4, pack=False, causal=True, lens=(1536,), o_offset=3),
    # without a Stats output the reduction still runs on the partial Stats
    "bf16-q1-p-s8-nostats": dict(s_q=1, split=8, stats=False, lens=(2048, 600)),
    "bf16-q6-u-s4-nostats": dict(s_q=6, split=4, pack=False, causal=True, stats=False, lens=(1536,)),
}


@pytest.mark.parametrize("name", sorted(CELLS))
def test_reduction_matches_reference_and_two_launch_bytes(name):
    _device()
    runner = Runner(seed=sorted(CELLS).index(name) + 1, **CELLS[name])
    _compare_forms(runner, list(range(len(runner.lens))))


@pytest.mark.parametrize(
    "dtype,users,splits,needed",
    [(torch.bfloat16, 64, (4, 8, 8), 2_000_000), (torch.float16, 24, (8,), 250_000)],
    ids=["bf16", "f16"],
)
def test_byte_identity_element_budget(dtype, users, splits, needed):
    """A one-ulp fp32 difference reaches a stored half element with probability about 2^-16 (bf16) or 2^-13 (f16), so byte
    identity is asserted over enough elements to see one: full caches, random scores (every chunk carries comparable weight),
    4 and 8 chunks, with the reverse-order control."""
    _device()
    compared = 0
    for i, split in enumerate(splits):
        runner = Runner(dtype=dtype, s_q=2, split=split, causal=True, lens=[2048] * users, seed=100 + i)
        compared += _compare_forms(runner, list(range(users)), reordered_control=True)
    print(f"byte identity of the in-launch reduction: {compared} {str(dtype).split('.')[-1]} O elements compared, 0 differing")
    assert compared >= needed


def test_two_chunks_commute():
    """With two chunks every order gives the same sum: the reverse-order control must reproduce the bytes there, which is why
    the two-chunk cells count for nothing in the element budget."""
    _device()
    runner = Runner(s_q=2, split=2, causal=True, lens=[2048] * 4, seed=7)
    t, api, ws = runner.run([0, 1, 2, 3], reduce=True)
    two = runner.plan(t, False)
    slab_o, slab_lse = runner.slabs(api, ws, 4)
    o_rev, _ = runner.separate_combine(two, slab_o.flip(0).contiguous(), slab_lse.flip(0).contiguous(), 4)
    assert _same_bytes(o_rev, t["o_store"][:, :2])


@pytest.mark.parametrize("reduce", [False, True], ids=["two-launch", "in-launch"])
def test_stale_partials_are_never_read(reduce):
    """A long launch followed by a short one on the SAME workspace, not re-poisoned: the short launch's chunks past the live
    tiles must rewrite their partials, or the reduction would read the previous launch's finite ones."""
    _device()
    long_lens, short_lens = (2048, 2048, 2048), (1, 100, 0)
    full = Runner(s_q=1, split=8, lens=long_lens, seed=31)
    short = Runner(s_q=1, split=8, lens=short_lens, seed=31)  # the same sequences, shorter live caches
    t, api, ws = full.run([0, 1, 2], reduce)
    full.check(t, "long")
    short._plans = full._plans  # one compiled plan, one workspace
    t, _, _ = short.run([0, 1, 2], reduce, ws=ws)
    reused = snapshot(short.check(t, "short on the used workspace"))
    t, _, _ = short.run([0, 1, 2], reduce)
    assert_bitwise(snapshot(short.check(t, "short on a fresh workspace")), reused, "a reused workspace changed the result")


@pytest.mark.parametrize("reduce", [False, True], ids=["two-launch", "in-launch"])
@pytest.mark.parametrize("pack,s_q,split", [(True, 2, 8), (False, 6, 4)], ids=["packed-q2-split8", "unpacked-q6-split4"])
def test_batch_invariance(reduce, pack, s_q, split):
    """A sequence alone, among 8 and among 64, reordered, duplicated and with other neighbours: its O and Stats bytes do not
    move at a fixed split."""
    _device()
    lens = [2048, 1000, 0, 129, 1536, 64, 2047, 1]
    runner = Runner(s_q=s_q, split=split, pack=pack, causal=True, lens=lens, seed=53)
    t, _, _ = runner.run([0], reduce)
    target = snapshot(runner.check(t, "alone"))
    orders = ([0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0], [3, 0, 0, 5], [1, 4] * 16 + [0] + [6, 2] * 15 + [0], [0] + [7] * 63)
    for order in orders:
        t, _, _ = runner.run(order, reduce)
        actual = snapshot(runner.check(t, f"batch of {len(order)}"))
        for row, sequence in enumerate(order):
            if sequence == 0:
                assert_bitwise({name: actual[name][row : row + 1] for name in target}, target, f"sequence 0 at row {row} of {len(order)}")


@pytest.mark.parametrize(
    "name,kwargs",
    [("bf16-packed-split8", dict(s_q=2, split=8, causal=True)), ("f16-unpacked-split4", dict(dtype=torch.float16, s_q=6, split=4, pack=False, causal=True))],
)
def test_replay_capture_and_golden(name, kwargs):
    """Fresh-buffer replay, a retained CUDA graph against eager execution, changed inputs under the retained graph, and one
    golden per path that BOTH forms must reproduce (compared last: a runtime without a recorded golden skips there)."""
    _device()
    runner = Runner(lens=(2048, 1027, 0), seed=11, **kwargs)
    order = [0, 1, 2]
    results = {}
    for reduce in (False, True):
        results[reduce] = assert_replays(lambda: runner.check(runner.run(order, reduce)[0]), repeats=8)
    assert_bitwise(results[True], results[False], "in-launch reduction against the two-launch form")
    inputs = dict(q=runner.q.clone(), k=runner.k, v=runner.v)
    # a retained graph of the in-launch form: replay equals eager, and follows changed lengths and data
    t = runner.batch(order)
    api = runner.plan(t, True)
    ws = runner.workspace(api)
    runner.launch(t, api, ws)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.launch(t, api, ws)
    try:
        t["o_store"].fill_(float("nan"))
        t["lse_store"].fill_(float("nan"))
        ws.fill_(WS_POISON)
        graph.replay()
        torch.cuda.synchronize()
        assert_bitwise(snapshot(runner.check(t, "retained replay")), results[True], "retained replay against eager")
        runner.lens = [129, 2048, 700]
        t["kl"].copy_(torch.tensor(runner.lens, dtype=torch.int32))
        t["q"].mul_(0.5)
        runner.q = runner.q * 0.5
        graph.replay()
        torch.cuda.synchronize()
        runner.check(t, "retained replay with changed lengths and queries")
    finally:
        torch.cuda.synchronize()
        graph.reset()
    # One golden per path: the two forms are byte-equal above, so both reproduce it.
    assert_golden(f"sdpa-decode-d256-swap-ab-{name}", inputs, results[True])


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group("gpu_exclusive")
def test_warm_refill_stress():
    """The tile's ring, P slots and max slots are refilled inside one launch (ten key tiles per chunk); the in-launch form is
    a new rendering of that mainloop, so it is stressed warm behind the control of its cluster size."""
    from warm_stress import StressCase, assert_warm_stress, stress_validation

    _device()
    split, users = 2, 9
    runner = Runner(s_q=2, split=split, causal=True, declared=2560, lens=[2560] * users, seed=17, margin=0)
    order = list(range(users))
    t, api, ws = runner.run(order, True)
    reference = snapshot(runner.check(t, "stress reference"))
    two, _, _ = runner.run(order, False)
    assert_bitwise(snapshot(runner.check(two, "stress shape, two-launch")), reference, "stress shape: in-launch against two-launch")

    def launch():
        runner.launch(t, api, ws)

    assert_warm_stress(
        StressCase(
            "D256 swap-AB decode, in-launch reduction",
            launch,
            {"o": t["o_store"], "lse": t["lse_store"]},
            thread_blocks=users * KH * split,
            refills_per_launch=9,
            reference=reference,
        ),
        validation=stress_validation(cluster=split),
        warm=256,
        poisoned=256,
    )


def _graph(runner, order, *, knobs=None):
    """A paged cuDNN graph of the runner's declaration. ``knobs`` = a public record for the swap-AB engine; None = the
    default ranked list with the FROST engines enabled. Returns (graph, variant pack, batch tensors, workspace)."""
    import cudnn
    import cudnn.sdpa  # noqa: F401
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.fwd.kernels.sm100.d256_f16_contract import SWAP_AB

    t = runner.batch(order)
    b = len(order)
    io = cudnn.data_type.HALF if runner.dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    table = t["table"].view(b, 1, runner.pages, 1).contiguous()
    kl, ql = t["kl"].view(b, 1, 1, 1), t["ql"].view(b, 1, 1, 1)
    stats = t["lse"].unsqueeze(-1)
    q, k, v, tk, tv, sq, sk = (g.tensor_like(x) for x in (t["q"], t["k"], t["v"], table, table, ql, kl))
    kw = dict(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=True,
        attn_scale=1.0 / math.sqrt(D),
        use_padding_mask=True,
        seq_len_q=sq,
        seq_len_kv=sk,
        paged_attention_k_table=tk,
        paged_attention_v_table=tv,
        paged_attention_max_seq_len_kv=runner.declared,
    )
    if runner.causal:
        kw["use_causal_mask_bottom_right"] = True
    o, st = g.sdpa(**kw)
    o.set_output(True).set_dim(t["o"].shape).set_stride(t["o"].stride())
    st.set_output(True).set_dim(stats.shape).set_stride(stats.stride()).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    if knobs is None:
        g.create_execution_plans([cudnn.heur_mode.A])
    else:
        g.create_execution_plan(next(f for f in MANIFEST if f.name == "frost_sdpa_fwd").offered_ids()[SWAP_AB], knobs)
    pack = {q: t["q"], k: t["k"], v: t["v"], tk: table, tv: table, sq: ql, sk: kl, o: t["o"], st: stats}
    return g, pack, t


def _execute_graph(g, pack, t, index):
    g.select_plan(index)
    g.check_support()
    g.build_plans()
    ws = torch.full((-(-max(g.get_workspace_size(), 16) // 4),), WS_POISON, device="cuda")
    t["o_store"].fill_(float("nan"))
    t["lse_store"].fill_(float("nan"))
    g.execute(pack, ws.view(torch.uint8))
    torch.cuda.synchronize()


def _record(split, pack, reduce=None):
    import cudnn

    kt = cudnn.knob_type
    record = {kt.TILE_M: 16, kt.TILE_N: 128, kt.TILE_CGA_M: 1, kt.SCHED_POLICY: 0, kt.PACK_GQA: int(pack), kt.SPLIT_KV: split}
    if reduce is not None:
        record[kt.CLUSTER_K_REDUCE] = int(reduce)
    return record


@pytest.mark.parametrize("users,declared,expect_split", [(1, 2048, 8), (2, 2048, 8), (4, 2048, 4), (1, 1536, 4)])
def test_default_plan_reduces_in_the_launch_with_the_bytes_of_its_split(users, declared, expect_split):
    """The heuristics' own lead plan for a two-token step: today's split with the knob on top, one launch, the public record
    carrying the knob; a record without the knob rebuilds the two-launch plan of that split, byte for byte."""
    import cudnn
    from cudnn.sdpa.fwd.kernels.sm100.d256_f16_contract import SWAP_AB
    from frost_test_utils import _is_plan_for

    _device()
    if torch.cuda.get_device_properties(0).multi_processor_count != 148:
        pytest.skip("the expected splits are those of a 148-SM part")
    runner = Runner(s_q=2, split=expect_split, causal=True, declared=declared, lens=[declared] * users, seed=61)
    order = list(range(users))
    g, pack, t = _graph(runner, order)
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    lead = next(i for i, n in enumerate(names) if _is_plan_for(n, SWAP_AB))
    assert g.plans[lead].knobs.split_kv == expect_split and g.plans[lead].knobs.cluster_reduce is True
    engine_id, public = g.get_engine_and_knobs_at_index(lead)
    assert public[cudnn.knob_type.CLUSTER_K_REDUCE] == 1 and public[cudnn.knob_type.SPLIT_KV] == expect_split
    _execute_graph(g, pack, t, lead)
    assert g._compiled_plans[lead]._compiled.kernel_template == "decode_d256_f16"
    fused = snapshot(runner.check(t, "default plan"))
    for record in ({key: value for key, value in public.items() if key != cudnn.knob_type.CLUSTER_K_REDUCE}, dict(public)):
        g2, pack2, t2 = _graph(runner, order, knobs=record)
        _execute_graph(g2, pack2, t2, 0)
        assert_bitwise(snapshot(runner.check(t2, "rebuilt record")), fused, "a public record against the default plan")


@pytest.mark.parametrize(
    "change,match",
    [
        (dict(split=16), "2 to 8 KV chunks"),
        (dict(split=1), "2 to 8 KV chunks"),
    ],
    ids=["split16", "unsplit"],
)
def test_explicit_record_outside_the_form_is_declined(change, match):
    """A public record asking for the knob where the form does not exist is refused when the plan is built; the same record
    without the knob builds."""
    import cudnn

    _device()
    runner = Runner(s_q=1, split=change["split"], lens=(2048,), seed=3)
    with pytest.raises((cudnn.cudnnGraphNotSupportedError, NotImplementedError, ValueError), match=match):
        g, _, _ = _graph(runner, [0], knobs=_record(change["split"], True, True))
        g.select_plan(0)
        g.check_support()
        g.build_plans()
    g, pack, t = _graph(runner, [0], knobs=_record(change["split"], True))
    _execute_graph(g, pack, t, 0)
    runner.check(t, "the record without the knob")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(split_kv=16), "2 to 8 KV chunks"),
        (dict(split_kv=1), "2 to 8 KV chunks"),
        (dict(split_kv=4, use_pdl=True), "programmatic-dependent-launch"),
        (dict(split_kv=4, has_sink=True), "does not carry a sink"),
        (dict(split_kv=4, stats_log2=True), "does not carry a sink"),
    ],
    ids=["split16", "unsplit", "pdl", "sink", "log2-stats"],
)
def test_standalone_request_outside_the_form_is_declined(kwargs, match):
    """The standalone adapter's twin of the row decline: typed, before any kernel is built."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    _device()
    runner = Runner(s_q=1, lens=(2048,), seed=5)
    t = runner.batch([0])
    api = SdpaFwdDslSm100(
        t["q"],
        t["k"],
        t["v"],
        t["o"],
        t["lse"],
        seq_q_lens_present=True,
        seq_kv_lens_present=True,
        paged_page_size=runner.page,
        paged_max_seq_len_kv=runner.declared,
        paged_table_stride=t["table"].stride(),
        pack_gqa=True,
        cluster_reduce=True,
        **kwargs,
    )
    with pytest.raises(NotImplementedError, match=match):
        api.check_support()


def test_python_frame_refuses_an_unaligned_workspace():
    """The reduction reads the partial rows as 16-byte groups: the Python frame, like the native binder, refuses a split
    workspace that is not 16-byte aligned, and serves the reduced launch on an aligned one."""
    _device()
    runner = Runner(s_q=1, split=4, lens=(2048,), seed=9)
    t = runner.batch([0])
    api = runner.plan(t, True)
    api._dense_spec.native = None
    store = torch.full((runner.workspace(api).numel() + 4,), WS_POISON, device="cuda")
    with pytest.raises(ValueError, match="16-byte aligned"):
        runner.launch(t, api, store[1:])
    runner.launch(t, api, store[4:])  # 16-byte aligned again
    torch.cuda.synchronize()
    runner.check(t, "python frame")
