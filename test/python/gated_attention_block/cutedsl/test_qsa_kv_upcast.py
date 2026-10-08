# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``kernels/qsa_kv_upcast.py`` -- the e4m3 paged KV pool gathered by a block list and cast to a dense bf16 scratch, then the
bf16 d256 DECODE tile over the scratch as a plain padded dense SDPA.

Host cells (any machine): the geometry couplings, the typed declines (e5m2 by name, the form checks, the Rubin-line / DSL gate
as a ``NotImplementedError``), the view helpers, and the SASS pin of the hot path (an sm_107a trace-compile in a subprocess,
decoded by an ``nvdisasm`` that knows the target: every row load is 128-bit, no spills) -- skipped where no such nvdisasm is
on the PATH.

Rubin cells: (a) the cast kernel alone BITWISE a torch gather + ``(pool.float() * scale).bfloat16()`` through the compaction
contract -- the rows in list order, a ``-1`` inside the count and an id past the visible range skipped, the open tail's 1-3
tokens (two blocks under a shared list), rows past ``kv_len`` never read (the pool's unwritten rows are NaN), a ``-1`` table
page read as zero rows that still count, ``S_kv = 0`` and a negative position as a DEAD item (``kv_len_t = 0``), every row
past ``kv_len_t`` zero (the sentinel-filled scratch proves every row was written), two launches bitwise; HND and NHD pools,
pages 16 / 48 / 64, shared and per-token lists, with and without ``block_lens``, both zero-fill modes; (b) cast + the dense
decode tile (the Rubin f16 engine pinned, the serving template asserted) against the fp32 oracle run on the DEQUANTIZED pool
-- the SAME function, within the sparse suite's budget -- and, REPORTED without a gate, against the oracle on the
pre-quantization bf16 K / V (that distance belongs to the serving recipe, not to these kernels).
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile

import pytest
import torch

from cudnn.gated_attention_block.kernels import qsa_kv_upcast as upcast

pytestmark = pytest.mark.L0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_qsa_reference import qsa_visible_mask, random_block_ids  # noqa: E402

D, BS, ROWS = upcast.D, upcast.BLOCK_SIZE, upcast.ROWS
H_Q, H_KV = 24, 2  # the block-sparse model's geometry at TP 1 (G = 12 packed query rows per item)
ATOL = 2e-2  # O and LSE vs the fp32 oracle on the dequantized pool: the sparse suite's budget (tighter than the dense suite's)
_DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _gpu_reason():
    if not torch.cuda.is_available():
        return "the cast kernel needs a Rubin-line GPU"
    return upcast.dsl_decline_reason()


_WHY = _gpu_reason()
_GPU = pytest.mark.skipif(_WHY is not None, reason=_WHY or "")
requires_rubin = pytest.mark.requires_rubin


# ---------------------------------------------------------------------------
# host cells
# ---------------------------------------------------------------------------


def test_geometry_couplings_and_views():
    upcast._validate_geometry()
    assert upcast.ROWS == upcast.UNITS_PER_ITEM * upcast.ROW_TILE == 2176
    assert upcast.TOPK_MAX + upcast.TAIL_SLOTS <= upcast.UNITS_PER_ITEM * upcast.SLOTS_PER_UNIT
    assert BS * upcast.TOPK_MAX + 6 <= ROWS  # a shared list's longest compaction (512 blocks + a two-block tail of 6 rows)
    assert upcast.scratch_shape(5) == (5, 2, ROWS, D) and upcast.scratch_bytes(5) == 5 * 2 * ROWS * D * 2
    assert upcast.n_units_of(3) == 3 * 17
    assert upcast.n_ctas_of(10, 212) == 10 and upcast.n_ctas_of(10**6, 212) == 212 * upcast.CTAS_PER_SM and upcast.n_ctas_of(0, 212) == 1
    scratch = torch.empty(upcast.scratch_shape(6), dtype=torch.bfloat16)
    kv_len_t = torch.zeros(6, dtype=torch.int32)
    k, v, sk = upcast.dense_core_views(scratch, kv_len_t)
    assert tuple(k.shape) == (6, 1, ROWS, D) and tuple(v.shape) == (6, 1, ROWS, D) and tuple(sk.shape) == (6, 1, 1, 1)
    assert k.stride(0) == 2 * ROWS * D and k.stride(2) == D and k.stride(3) == 1 and v.data_ptr() == scratch[:, 1].data_ptr()
    q = torch.randn(2, 3, H_Q, D, dtype=torch.bfloat16)
    qi = upcast.q_items_view(q, H_KV)
    assert tuple(qi.shape) == (2 * 3 * H_KV, H_Q // H_KV, 1, D) and qi.data_ptr() == q.data_ptr()
    assert torch.equal(upcast.items_to_rows(qi, 2, 3, H_Q), q)
    with pytest.raises(ValueError, match="scratch must be"):
        upcast.dense_core_views(scratch[:, :1], kv_len_t)
    with pytest.raises(ValueError, match="q must be"):
        upcast.q_items_view(torch.randn(1, 1, 5, D, dtype=torch.bfloat16), H_KV)


def test_declines_are_typed(monkeypatch):
    with pytest.raises(ValueError, match="top_k must be a multiple of 4"):
        upcast.validate_params(h_kv=2, top_k=502, page_size=16, zero_fill="full")
    with pytest.raises(ValueError, match="top_k must be a multiple"):
        upcast.validate_params(h_kv=2, top_k=1024, page_size=16, zero_fill="full")
    with pytest.raises(ValueError, match="page_size must be a positive multiple of 4"):
        upcast.validate_params(h_kv=2, top_k=512, page_size=6, zero_fill="full")
    with pytest.raises(ValueError, match="zero_fill must be one of"):
        upcast.validate_params(h_kv=2, top_k=512, page_size=16, zero_fill="none")
    with pytest.raises(ValueError, match="h_kv must be >= 1"):
        upcast.validate_params(h_kv=0, top_k=512, page_size=16, zero_fill="full")
    # e5m2 is declined BY NAME; a bf16 pool is a form error.
    with pytest.raises(NotImplementedError, match="e5m2"):
        upcast.check_pool(torch.zeros(1, 2, 16, D, dtype=torch.float8_e5m2), "k_pool", h_kv=2, page_size=16, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="must be torch.float8_e4m3fn"):
        upcast.check_pool(torch.zeros(1, 2, 16, D, dtype=torch.bfloat16), "k_pool", h_kv=2, page_size=16, device=torch.device("cpu"))
    with pytest.raises(ValueError, match=r"must be \[num_pages >= 1"):
        upcast.check_pool(torch.zeros(1, 2, 32, D, dtype=torch.float8_e4m3fn), "k_pool", h_kv=2, page_size=16, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="contiguous along D"):
        upcast.check_pool(torch.zeros(1, 2, 16, D, 2, dtype=torch.float8_e4m3fn)[..., 0], "k_pool", h_kv=2, page_size=16, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="16-byte-aligned item / row strides"):
        upcast.check_out(torch.zeros(1, ROWS, D + 4, dtype=torch.bfloat16)[:, :, :D], "out_k", n_items=1, device=torch.device("cpu"))
    # The Rubin-line / DSL gate runs BEFORE any compile and is a NotImplementedError naming the device (Rule 7 shape).
    monkeypatch.setattr(upcast, "compute_capability", lambda dev: (8, 0))
    monkeypatch.setattr(upcast, "current_device", lambda: 0)
    with pytest.raises(NotImplementedError, match="targets the Rubin line .* cc 8.0"):
        upcast.compile_qsa_kv_upcast(h_kv=2, top_k=512, page_size=16, list_per_sequence=True, has_block_lens=True)
    monkeypatch.setattr(upcast, "compute_capability", lambda dev: (10, 7))
    monkeypatch.setattr("cudnn.frost.buffers.cutedsl_arch_requirement_error", lambda cc: "SM107 requires a CuTe DSL build supporting sm_107a (probe)")
    with pytest.raises(NotImplementedError, match="sm_107a"):
        upcast.compile_qsa_kv_upcast(h_kv=2, top_k=512, page_size=16, list_per_sequence=True, has_block_lens=True)


def _nvdisasm_for_sm107():
    """An ``nvdisasm`` that decodes sm_107a: ``$CUDA_PATH/bin`` first, then the PATH; None when absent or too old."""
    cands = []
    cp = os.environ.get("CUDA_PATH") or os.environ.get("CUDAToolkit_ROOT")
    if cp:
        cands.append(os.path.join(cp, "bin", "nvdisasm"))
    w = shutil.which("nvdisasm")
    if w:
        cands.append(w)
    for c in cands:
        if os.path.exists(c):
            return c
    return None


_SASS_PROBE = r"""
import os, sys
os.environ["CUTE_DSL_DUMP_DIR"] = sys.argv[1]
os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
os.environ["CUTE_DSL_ARCH"] = "sm_107a"
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
from cudnn.gated_attention_block.kernels import qsa_kv_upcast as m
m.compute_capability = lambda dev: (10, 7)
m.current_device = lambda: 0
m.compile_qsa_kv_upcast(h_kv=2, top_k=512, page_size=64, list_per_sequence=True, has_block_lens=True)
print("COMPILED")
"""


def test_sass_pins_128_bit_row_traffic_and_no_spills():
    """The row-49 detector as a pin: every pool-row load of the hot path is ``LDG.E.128`` (the only bare ``LDG.E`` are the
    32-bit index loads: ids, table entries, lengths, scales), every slab store ``STG.E.128`` (plus the one ``kv_len_t`` word),
    no ``STL`` / ``LDL``.  An sm_107a trace-compile in a fresh process (no launch), decoded by an nvdisasm that knows the
    target; skipped where none is available."""
    import glob
    import re

    nvd = _nvdisasm_for_sm107()
    if nvd is None:
        pytest.skip("no nvdisasm on CUDA_PATH / PATH")
    try:
        from cudnn.frost.buffers import _cutedsl_has_sm107

        if not _cutedsl_has_sm107():
            pytest.skip("the installed CuTe DSL has no sm_107a target")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cannot probe the DSL: {e}")
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ)
        env.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
        res = subprocess.run([sys.executable, "-c", _SASS_PROBE, td], env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0 and "COMPILED" in res.stdout, res.stderr[-3000:]
        cubins = glob.glob(os.path.join(td, "**", "*.cubin"), recursive=True)
        assert cubins, "the trace-compile dumped no cubin"
        dis = subprocess.run([nvd, "-c", cubins[-1]], capture_output=True, text=True, timeout=300)
        if dis.returncode != 0:
            pytest.skip(f"this nvdisasm cannot decode the sm_107a cubin: {dis.stderr[-200:]}")
        ops = []
        for line in dis.stdout.splitlines():
            m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\d\s+)?([A-Z][A-Z0-9_.]*)", line)
            if m:
                ops.append(m.group(1))
    ldg128 = sum(1 for o in ops if o.startswith("LDG") and o.endswith(".128"))
    ldg_bare = sum(1 for o in ops if o.startswith("LDG") and not o.endswith(".128"))
    stg128 = sum(1 for o in ops if o.startswith("STG") and o.endswith(".128"))
    stg_bare = sum(1 for o in ops if o.startswith("STG") and not o.endswith(".128"))
    spills = sum(1 for o in ops if o.startswith("STL") or o.startswith("LDL"))
    # 2 operands x 2 row pairs x (SLOTS_PER_WARP) slots = the row loads of one unit, every one 128-bit.
    assert ldg128 == 2 * 2 * upcast.SLOTS_PER_WARP, (ldg128, ldg_bare)
    # The bare loads are the per-lane 32-bit INDEX loads (one id per scan step, the table word per slot, the lengths, the two
    # scales): bounded, and none of them a row of the pool.
    assert ldg_bare <= 32, ldg_bare
    assert stg128 >= 2 * ldg128 and stg_bare == 1, (stg128, stg_bare)  # two 16-B halves per row load + the zero fill; kv_len_t is the scalar
    assert spills == 0, spills


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _quantize(x: torch.Tensor):
    """Per-tensor e4m3 quantization of a bf16 ``[B, S, H_kv, D]`` tensor: ``(pool, dequantized bf16, scale fp32 [1])`` with the
    dequantized values EXACTLY the cast kernel's ``bf16(fp32(e4m3) * scale)``."""
    amax = x.float().abs().amax().clamp_min(1e-12)
    scale = (amax / 448.0).reshape(1).to(torch.float32)
    pool = (x.float() / scale).to(torch.float8_e4m3fn)
    deq = (pool.float() * scale).to(torch.bfloat16)
    return pool, deq, scale.contiguous()


def _paginate(pools, kv_lens, P, hnd, gen, *, dead_pages=()):
    """Scatter the first ``kv_lens[b]`` tokens of every ``[B, SKV, H_kv, D]`` e4m3 tensor in ``pools`` into ``[num_pages, H_kv, P, D]``
    pools (HND, or NHD storage permuted into that view) through ONE shuffled int32 table (the kernel takes one table for K and V);
    every unwritten row holds NaN (a gathered unwritten row poisons the output); table entries past a sequence's live pages are
    -1; ``dead_pages`` = ``(b, p)`` LIVE pages whose table entry is set to -1 (the kernel reads them as ZERO rows that still count).
    Returns ``(views, table)``."""
    B, SKV, KH, _ = pools[0].shape
    dev = pools[0].device
    max_pages = -(-SKV // P)
    num_pages = B * max_pages + 3
    shape = (num_pages, KH, P, D) if hnd else (num_pages, P, KH, D)
    perm = torch.randperm(num_pages, device=dev, generator=gen)[: B * max_pages].view(B, max_pages)
    table = perm.to(torch.int32).clone()
    views = []
    for pool_dense in pools:
        pool = torch.full(shape, float("nan"), device=dev, dtype=torch.float8_e4m3fn)
        for b in range(B):
            L = int(kv_lens[b])
            n_live = -(-L // P)
            for p in range(max_pages):
                if p >= n_live:
                    continue
                t0, t1 = p * P, min((p + 1) * P, L)
                rows = pool_dense[b, t0:t1]  # [t1 - t0, KH, D]
                page = int(perm[b, p])
                if hnd:
                    pool[page, :, : t1 - t0] = rows.permute(1, 0, 2)
                else:
                    pool[page, : t1 - t0] = rows
        views.append(pool if hnd else pool.permute(0, 2, 1, 3))
    for b in range(B):
        n_live = -(-int(kv_lens[b]) // P)
        table[b, n_live:] = -1
    for b, p in dead_pages:
        table[b, p] = -1
    return views, table.contiguous()


def _lists(B, S_q, top_k, kv_lens, gen, *, per_token, degenerate=True):
    """Random lists (the oracle's builder): shared ``[B, top_k]`` anchored at the step-0 position, or per-token ``[B, S_q,
    top_k]`` at every row's own position; with ``degenerate``, entry 2 of every row with >= 8 entries is set to -1 and entry 5
    to an id far past any visible range -- both must contribute nothing."""
    dev = torch.device("cuda")
    kvl = torch.tensor(kv_lens, device=dev, dtype=torch.long)
    if per_token:
        pos = (kvl[:, None] - S_q) + torch.arange(S_q, device=dev)[None, :]
        ids, lens = random_block_ids(pos, top_k, BS, generator=gen, kv_lens=kvl[:, None].expand(B, S_q), shuffle=True)
    else:
        ids, lens = random_block_ids(kvl - S_q, top_k, BS, generator=gen, kv_lens=kvl, shuffle=True)
    ids, lens = ids.contiguous(), lens.contiguous()
    if degenerate:
        flat_ids, flat_lens = ids.view(-1, top_k), lens.view(-1)
        for r in range(flat_ids.shape[0]):
            if int(flat_lens[r]) >= 8:
                flat_ids[r, 2] = -1
                flat_ids[r, 5] = 1 << 20
    return ids, lens


def _expected_rows(ids_row, count, n_vis, n_anchor):
    """The compaction contract on the host: the listed blocks' visible rows in list order, then the open tail."""
    rows = []
    n_blk_vis = (n_vis + BS - 1) // BS
    for i in range(count):
        blk = int(ids_row[i])
        if 0 <= blk < n_blk_vis:
            rows.extend(range(BS * blk, min(BS * blk + BS, n_vis)))
    rows.extend(range(BS * (n_anchor // BS), n_vis))
    return rows


def _item_geometry(b, s, S_q, L, lens_row, top_k, per_token):
    pos = L - S_q + s
    anchor = pos if per_token else L - S_q
    n_vis = max(min(pos + 1, L), 0)
    n_anchor = max(min(anchor + 1, L), 0)
    n_sel = min(top_k, n_anchor // BS)
    count = n_sel if lens_row is None else min(max(int(lens_row), 0), n_sel)
    return n_vis, n_anchor, count


def _run_cast(k_pool, v_pool, table, ids, lens, kv_lens, k_scale, v_scale, *, B, S_q, top_k, P, per_token, zero_fill="full", launches=2):
    """Compile (plan time) and launch the cast kernel ``launches`` times into sentinel (NaN) filled scratches; returns the list of
    ``(scratch, kv_len_t)``."""
    dev = k_pool.device
    r = upcast.compile_qsa_kv_upcast(h_kv=H_KV, top_k=top_k, page_size=P, list_per_sequence=not per_token, has_block_lens=lens is not None, zero_fill=zero_fill)
    seq_kv = torch.tensor(kv_lens, device=dev, dtype=torch.int32)
    n_items = B * S_q * H_KV
    outs = []
    for _ in range(launches):
        scratch = torch.full(upcast.scratch_shape(n_items), float("nan"), device=dev, dtype=torch.bfloat16)
        kv_len_t = torch.full((n_items,), -7, device=dev, dtype=torch.int32)
        upcast.run_qsa_kv_upcast(
            r,
            k_pool=k_pool,
            v_pool=v_pool,
            block_table=table,
            block_ids=ids,
            block_lens=lens,
            seq_kv_lens=seq_kv,
            k_scale=k_scale,
            v_scale=v_scale,
            out_k=scratch[:, 0],
            out_v=scratch[:, 1],
            kv_len_t=kv_len_t,
            s_q=S_q,
            stream=_stream(),
        )
        torch.cuda.synchronize()
        outs.append((scratch, kv_len_t))
    return outs


def _check_cast(outs, kd, vd, ids, lens, kv_lens, *, B, S_q, top_k, per_token, dead_tokens=None, zero_fill="full"):
    """Every item against the compaction contract: ``kv_len_t``, the rows BITWISE the dequantized dense rows in list order
    (zeros where the page is dead), zeros past ``kv_len_t`` (to the slab's end, or to the next 128-row tile), no sentinel left,
    two launches bitwise.  Returns the total rows written."""
    (scratch, kv_len_t), *rest = outs
    for s2, k2 in rest:
        assert torch.equal(s2.view(torch.int16), scratch.view(torch.int16)) and torch.equal(k2, kv_len_t), "two launches must be bitwise"
    kv_len_h = kv_len_t.tolist()
    ids_h = ids.view(-1, top_k).cpu()
    lens_h = None if lens is None else lens.view(-1).cpu()
    total = 0
    for b in range(B):
        L = int(kv_lens[b])
        for s in range(S_q):
            row = b if not per_token else b * S_q + s
            n_vis, n_anchor, count = _item_geometry(b, s, S_q, L, None if lens_h is None else lens_h[row], top_k, per_token)
            rows = _expected_rows(ids_h[row], count, n_vis, n_anchor)
            for h in range(H_KV):
                item = (b * S_q + s) * H_KV + h
                n = len(rows)
                total += n
                assert kv_len_h[item] == n, f"item {item} (b {b}, s {s}, h {h}): kv_len_t {kv_len_h[item]} != {n} expected rows"
                idx = torch.tensor(rows, device=kd.device, dtype=torch.long)
                exp_k = kd[b, idx, h] if n else kd[b, :0, h]
                exp_v = vd[b, idx, h] if n else vd[b, :0, h]
                if dead_tokens is not None and n:
                    dead = dead_tokens[b][idx]
                    exp_k = torch.where(dead[:, None], torch.zeros_like(exp_k), exp_k)
                    exp_v = torch.where(dead[:, None], torch.zeros_like(exp_v), exp_v)
                got_k, got_v = scratch[item, 0, :n], scratch[item, 1, :n]
                assert torch.equal(
                    got_k.view(torch.int16), exp_k.view(torch.int16)
                ), f"item {item}: K rows differ from the torch cast (first {int((got_k != exp_k).any(-1).nonzero()[:1])})"
                assert torch.equal(got_v.view(torch.int16), exp_v.view(torch.int16)), f"item {item}: V rows differ from the torch cast"
                end = ROWS if zero_fill == "full" else -(-n // upcast.ROW_TILE) * upcast.ROW_TILE
                tail = scratch[item, :, n:end]
                assert (tail.view(torch.int16) == 0).all(), f"item {item}: the rows [{n}, {end}) must be zero"
                if end < ROWS:
                    assert torch.isnan(scratch[item, :, end:]).all(), f"item {item}: tile zero-fill must not touch rows past {end}"
    if zero_fill == "full":
        assert not torch.isnan(scratch).any(), "a sentinel survived: a slab row was never written"
    return total


def _oracle(q, kd, vd, ids, lens, kv_lens, S_q, scale, top_k, *, per_token):
    """fp32 softmax over the dequantized dense K / V restricted to the visible set (shared list anchored at the step-0
    position, or per-token at the row's own position).  Returns O ``[B, S_q, H_q, D]`` fp32 and the natural-log LSE
    ``[B, H_q, S_q]`` (-inf on dead rows)."""
    B = q.shape[0]
    SKV = kd.shape[1]
    G = H_Q // H_KV
    dev = q.device
    ref_o = torch.zeros(B, S_q, H_Q, D, device=dev, dtype=torch.float32)
    ref_lse = torch.full((B, H_Q, S_q), float("-inf"), device=dev, dtype=torch.float32)
    for b in range(B):
        L = int(kv_lens[b])
        pos = torch.arange(S_q, device=dev) + (L - S_q)
        if per_token:
            ids_rows, lens_rows, pos0 = ids[b], (None if lens is None else lens[b]), None
        else:
            ids_rows, lens_rows, pos0 = ids[b][None].expand(S_q, -1), (None if lens is None else lens[b][None].expand(S_q)), L - S_q
        allowed = qsa_visible_mask(ids_rows, lens_rows, pos, L, SKV, BS, top_k=top_k, pos0=pos0)
        for h in range(H_Q):
            kk, vv = kd[b, :, h // G].float(), vd[b, :, h // G].float()
            sc = (q[b, :, h].float() @ kk.t()) * scale
            sc = sc.masked_fill(~allowed, float("-inf"))
            rmax = sc.amax(dim=-1)
            dead = torch.isinf(rmax) & (rmax < 0)
            safe = torch.where(dead, torch.zeros_like(rmax), rmax)
            p = torch.where(allowed, torch.exp(sc - safe[:, None]), torch.zeros_like(sc))
            den = p.sum(dim=-1)
            den_safe = torch.where(dead, torch.ones_like(den), den)
            oc = (p @ vv) / den_safe[:, None]
            ref_o[b, :, h] = torch.where(dead[:, None], torch.zeros_like(oc), oc)
            ref_lse[b, h] = torch.where(dead, torch.full_like(den, float("-inf")), safe + torch.log(den))
    return ref_o, ref_lse


def _rubin_engine():
    from cudnn.sdpa.fwd.engines import engine_name

    return engine_name(arch="sm107")


def _dense_core(q, scratch, kv_len_t, *, B, S_q, scale):
    """The bf16 d256 DECODE tile over the scratch as a plain PADDED dense SDPA: one 'batch' per item, the item's packed
    query-head group as the Q rows (``S_q' = 1``), ``seq_len_kv = kv_len_t``; the Rubin FROST engine pinned and the serving
    template asserted.  Returns ``(O [B, S_q, H_q, D] bf16, LSE [B, H_q, S_q] fp32)``."""
    import cudnn

    dev = q.device
    n_items = B * S_q * H_KV
    G = H_Q // H_KV
    qi = upcast.q_items_view(q, H_KV)
    kview, vview, seq_kv = upcast.dense_core_views(scratch, kv_len_t)
    seq_q = torch.ones(n_items, 1, 1, 1, device=dev, dtype=torch.int32)
    o_gpu = torch.empty(n_items, G, 1, D, device=dev, dtype=torch.bfloat16)
    stats_gpu = torch.empty(n_items, G, 1, 1, device=dev, dtype=torch.float32)
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq, tk, tv = g.tensor_like(qi), g.tensor_like(kview), g.tensor_like(vview)
    tsq, tsk = g.tensor_like(seq_q), g.tensor_like(seq_kv)
    o, st = g.sdpa(name="sdpa", q=tq, k=tk, v=tv, generate_stats=True, attn_scale=scale, use_padding_mask=True, seq_len_q=tsq, seq_len_kv=tsk)
    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    st.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    engine = _rubin_engine()
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    index = next((i for i, n in enumerate(names) if n == engine or n.startswith(engine + "[")), None)
    assert index is not None, f"no plan for the Rubin engine {engine!r}; plans={names}"
    g.select_plan(index)
    g.check_support()
    g.build_plans()
    served = g._compiled_plans[g._plan_index]._compiled.kernel_template
    assert served == "decode_d256_f16", f"the dense core over the scratch must be the d256 decode tile, got {served} ({names[index]})"
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    torch.cuda.set_sync_debug_mode("error")
    try:
        g.execute({tq: qi, tk: kview, tv: vview, tsq: seq_q, tsk: seq_kv, o: o_gpu, st: stats_gpu}, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    o_rows = upcast.items_to_rows(o_gpu, B, S_q, H_Q)
    lse = stats_gpu.view(B, S_q, H_KV, G).reshape(B, S_q, H_Q).permute(0, 2, 1).contiguous()
    return o_rows, lse


def _inputs(B, S_q, kv_lens, seed):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    SKV = max(max(kv_lens), BS)
    q = torch.randn(B, S_q, H_Q, D, device=dev, dtype=torch.float32, generator=gen).to(torch.bfloat16)
    k = torch.randn(B, SKV, H_KV, D, device=dev, dtype=torch.float32, generator=gen).to(torch.bfloat16)
    v = torch.randn(B, SKV, H_KV, D, device=dev, dtype=torch.float32, generator=gen).to(torch.bfloat16)
    return q, k, v, gen


# ---------------------------------------------------------------------------
# Rubin cells
# ---------------------------------------------------------------------------


@_GPU
@requires_rubin
@pytest.mark.parametrize(
    "B, S_q, kv_lens, P, hnd, top_k, per_token, with_lens, zero_fill, dead",
    [
        pytest.param(4, 1, [4097, 0, 5, 2052], 16, True, 512, False, True, "full", [(0, 3)], id="shared-sq1-page16-HND-deadpage"),
        pytest.param(3, 4, [8192, 7, 2], 64, False, 512, False, True, "full", [], id="shared-sq4-page64-NHD-mtp"),
        pytest.param(2, 2, [3000, 1030], 48, True, 64, True, True, "full", [], id="pertoken-sq2-page48-HND-top64"),
        pytest.param(2, 1, [1000, 2051], 16, True, 512, False, False, "tile", [], id="shared-sq1-page16-nolens-tilefill"),
    ],
)
def test_cast_is_bitwise_the_torch_cast_through_the_compaction_contract(B, S_q, kv_lens, P, hnd, top_k, per_token, with_lens, zero_fill, dead):
    q, k, v, gen = _inputs(B, S_q, kv_lens, seed=11)
    pk, kd, k_scale = _quantize(k)
    pv, vd, v_scale = _quantize(v)
    (k_pool, v_pool), table = _paginate([pk, pv], kv_lens, P, hnd, gen, dead_pages=dead)
    ids, lens = _lists(B, S_q, top_k, kv_lens, gen, per_token=per_token)
    dead_tokens = None
    if dead:
        dead_tokens = torch.zeros(B, k.shape[1], dtype=torch.bool, device=k.device)
        for b, p in dead:
            dead_tokens[b, p * P : (p + 1) * P] = True
    outs = _run_cast(
        k_pool,
        v_pool,
        table,
        ids,
        lens if with_lens else None,
        kv_lens,
        k_scale,
        v_scale,
        B=B,
        S_q=S_q,
        top_k=top_k,
        P=P,
        per_token=per_token,
        zero_fill=zero_fill,
    )
    total = _check_cast(
        outs, kd, vd, ids, lens if with_lens else None, kv_lens, B=B, S_q=S_q, top_k=top_k, per_token=per_token, dead_tokens=dead_tokens, zero_fill=zero_fill
    )
    kv_len_t = outs[0][1].tolist()
    print(
        f"\ncast {B}x{S_q} lens {kv_lens} page {P} {'HND' if hnd else 'NHD'} top_k {top_k} {'per-token' if per_token else 'shared'}: kv_len_t {kv_len_t}, {total} rows written, BITWISE the torch cast"
    )
    # The degenerate rows the contract names, visible in kv_len_t itself:
    if kv_lens[1] == 0:
        assert all(x == 0 for x in kv_len_t[S_q * H_KV : 2 * S_q * H_KV]), "S_kv = 0 must be a dead item (kv_len_t = 0)"
    if not per_token and S_q == 4 and kv_lens[2] == 2:
        # L = 2 < S_q: rows 0 / 1 have a negative position (dead), rows 2 / 3 see tokens [0, 1) / [0, 2) through the tail only
        items = [kv_len_t[(2 * S_q + s) * H_KV] for s in range(S_q)]
        assert items == [0, 0, 1, 2], items


@_GPU
@requires_rubin
def test_run_form_declines_are_typed():
    dev = torch.device("cuda")
    r = upcast.compile_qsa_kv_upcast(h_kv=H_KV, top_k=64, page_size=16, list_per_sequence=True, has_block_lens=True)
    k_pool = torch.zeros(2, H_KV, 16, D, device=dev, dtype=torch.float8_e4m3fn)
    scratch = torch.zeros(upcast.scratch_shape(2 * H_KV), device=dev, dtype=torch.bfloat16)
    good = dict(
        k_pool=k_pool,
        v_pool=k_pool,
        block_table=torch.zeros(2, 1, device=dev, dtype=torch.int32),
        block_ids=torch.full((2, 64), -1, device=dev, dtype=torch.int32),
        block_lens=torch.zeros(2, device=dev, dtype=torch.int32),
        seq_kv_lens=torch.zeros(2, device=dev, dtype=torch.int32),
        k_scale=torch.ones(1, device=dev),
        v_scale=torch.ones(1, device=dev),
        out_k=scratch[:, 0],
        out_v=scratch[:, 1],
        kv_len_t=torch.zeros(2 * H_KV, device=dev, dtype=torch.int32),
        s_q=1,
        stream=_stream(),
    )
    upcast.run_qsa_kv_upcast(r, **good)
    torch.cuda.synchronize()
    assert (scratch == 0).all() and (good["kv_len_t"] == 0).all()
    for change, exc, msg in [
        (dict(block_lens=None), ValueError, "compiled with block_lens"),
        (dict(block_ids=torch.full((2, 128), -1, device=dev, dtype=torch.int32)), ValueError, "block_ids must be int32"),
        (dict(block_ids=torch.full((2, 64), -1, device=dev, dtype=torch.int64)), ValueError, "must be an int32 tensor"),
        (dict(seq_kv_lens=torch.zeros(3, device=dev, dtype=torch.int32)), ValueError, "seq_kv_lens must have shape"),
        (dict(k_scale=torch.ones(1, device=dev, dtype=torch.bfloat16)), ValueError, "k_scale must be a 1-element fp32"),
        (dict(k_scale=1.0), ValueError, "k_scale must be a 1-element fp32"),
        (dict(s_q=5), ValueError, "s_q must be >= 1"),
        (dict(v_pool=torch.zeros(3, H_KV, 16, D, device=dev, dtype=torch.float8_e4m3fn)), ValueError, "same number of pages"),
        (dict(v_pool=k_pool.to(torch.float8_e5m2)), NotImplementedError, "e5m2"),
        (dict(kv_len_t=torch.zeros(3, device=dev, dtype=torch.int32)), ValueError, "kv_len_t must have shape"),
        (dict(out_k=scratch[:1, 0]), ValueError, r"out_k must be \["),
    ]:
        with pytest.raises(exc, match=msg):
            upcast.run_qsa_kv_upcast(r, **{**good, **change})


@_GPU
@requires_rubin
@pytest.mark.parametrize(
    "B, S_q, kv_lens, P, hnd, top_k, per_token",
    [
        pytest.param(2, 1, [8192, 2052], 16, True, 512, False, id="shared-sq1-B2-8K-2052-page16-HND"),
        pytest.param(4, 4, [4097, 0, 7, 2600], 64, False, 512, False, id="shared-sq4-B4-mixed-page64-NHD"),
        pytest.param(2, 2, [3000, 1030], 48, True, 64, True, id="pertoken-sq2-B2-page48-top64"),
    ],
)
def test_cast_then_dense_decode_tile_matches_the_oracle_on_the_dequantized_pool(B, S_q, kv_lens, P, hnd, top_k, per_token):
    q, k, v, gen = _inputs(B, S_q, kv_lens, seed=5)
    scale = 1.0 / math.sqrt(D)
    pk, kd, k_scale = _quantize(k)
    pv, vd, v_scale = _quantize(v)
    (k_pool, v_pool), table = _paginate([pk, pv], kv_lens, P, hnd, gen)
    ids, lens = _lists(B, S_q, top_k, kv_lens, gen, per_token=per_token)
    outs = _run_cast(k_pool, v_pool, table, ids, lens, kv_lens, k_scale, v_scale, B=B, S_q=S_q, top_k=top_k, P=P, per_token=per_token, launches=1)
    scratch, kv_len_t = outs[0]
    _check_cast(outs, kd, vd, ids, lens, kv_lens, B=B, S_q=S_q, top_k=top_k, per_token=per_token)
    o, lse = _dense_core(q, scratch, kv_len_t, B=B, S_q=S_q, scale=scale)
    ref_o, ref_lse = _oracle(q, kd, vd, ids, lens, kv_lens, S_q, scale, top_k, per_token=per_token)
    of = o.float()
    dead = torch.isinf(ref_lse)  # [B, H, S]
    assert torch.isfinite(of).all(), "non-finite O"
    if dead.any():
        assert torch.isinf(lse[dead]).all() and (lse[dead] < 0).all(), "a dead row's LSE must be -inf exactly"
        assert (of.permute(0, 2, 1, 3)[dead] == 0).all(), "a dead row's O must be 0 exactly"
    live = ~dead
    max_o = float((of - ref_o).abs().max())
    max_lse = float((lse[live] - ref_lse[live]).abs().max()) if live.any() else 0.0
    assert max_o <= ATOL, f"O off the oracle on the dequantized pool by {max_o} (budget {ATOL})"
    assert max_lse <= ATOL, f"LSE off the oracle on the dequantized pool by {max_lse} (budget {ATOL})"
    # REPORTED, no gate: the distance to the oracle on the pre-quantization bf16 K / V (the serving recipe's error).
    pre_o, pre_lse = _oracle(q, k, v, ids, lens, kv_lens, S_q, scale, top_k, per_token=per_token)
    pre_do = float((of - pre_o).abs().max())
    pre_dlse = float((lse[live] - pre_lse[live]).abs().max()) if live.any() else 0.0
    print(
        f"\ncast + decode tile {B}x{S_q} lens {kv_lens} page {P} {'HND' if hnd else 'NHD'} top_k {top_k} {'per-token' if per_token else 'shared'}: "
        f"vs the DEQUANTIZED oracle max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f} (budget {ATOL}); dead rows {int(dead.sum())}; "
        f"vs the PRE-QUANTIZATION bf16 oracle max|dO| {pre_do:.5f} max|dLSE| {pre_dlse:.5f} (reported, no gate); kv_len_t {kv_len_t.tolist()}"
    )
