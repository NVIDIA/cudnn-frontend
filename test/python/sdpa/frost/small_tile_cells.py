# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""The degenerate-input matrix of the f16/bf16 d128-FILE small tiles, the fp64 oracle of the whole forward contract and the pinned
graph run -- ONE helper for every arch's small-tile module (the Rubin head-dim module carries the original copy; the SM100 module
imports this one).  ``run_graph`` takes the engine ROW to pin, the only arch-specific input.

Every cell asserts on the OUTPUT directly where the contract is degenerate (keyless rows EXACTLY 0 / -inf, never a residue), launches
TWICE on sentinel-filled buffers (the two-launch delta) and compares the live rows against the fp64 oracle at the row's tolerances.
"""

import math
from types import SimpleNamespace

import torch

from frost_test_utils import make_dense_stats

# the existing f16 row tolerances: O against the fp64 oracle (an fp32 accumulation, one 16-bit rounding at the store), the LSE pair
TOL_O = dict(atol=5e-2, rtol=3e-2)
TOL_LSE = dict(atol=1e-4, rtol=1e-4)

CASES = {
    # the degenerate-input matrix (sdpa-invariants.md s9): the KV ring sweep n_kv = 1 (the P14 phase-drift shape), 2, 3, 4 (the ring
    # depth), 5 (depth + 1), 8 -- every one with B*H > 1; tail tiles (S not a multiple of 128; the row's synthesized KV padding on the
    # dense one, whose Q tail also exercises the OOB-row store guard); GQA; a zero-length KV batch; keyless bottom-right rows with a
    # sink and padded-Q trim; keyless rows past a left window; a right band; ONE CTA / one tile in the whole launch (B*H = 1, n_kv = 1)
    "dense_one_tile_bh1": dict(b=1, h=1, h_kv=1, s_q=128, s_kv=128),
    "dense_nkv1": dict(b=2, h=2, h_kv=2, s_q=128, s_kv=128),
    "dense_nkv2": dict(b=2, h=4, h_kv=4, s_q=256, s_kv=256),
    "dense_nkv3_tail_300": dict(b=2, h=2, h_kv=2, s_q=300, s_kv=300),
    "dense_nkv4_512": dict(b=2, h=2, h_kv=2, s_q=512, s_kv=512),
    "dense_nkv5_640": dict(b=2, h=2, h_kv=2, s_q=640, s_kv=640),
    "dense_nkv8_1024": dict(b=2, h=2, h_kv=2, s_q=1024, s_kv=1024),
    "causal_tail_385": dict(b=1, h=4, h_kv=4, s_q=385, s_kv=385, causal=True),
    "right_band_64_tail_300": dict(b=2, h=2, h_kv=2, s_q=300, s_kv=300, right_bound=64),
    "causal_swa128_padded_385": dict(b=2, h=2, h_kv=2, s_q=385, s_kv=385, causal=True, window_left=128, kv_lens=[385, 200]),
    "causal_gqa_8_2": dict(b=2, h=8, h_kv=2, s_q=256, s_kv=256, causal=True),
    "padded_kv_zero_batch": dict(b=3, h=4, h_kv=4, s_q=256, s_kv=256, kv_lens=[256, 0, 100]),
    "bottom_right_keyless_sink": dict(b=3, h=8, h_kv=4, s_q=256, s_kv=256, bottom_right=True, q_lens=[256, 129, 0], kv_lens=[200, 256, 128], sink=True),
}


def bhsd(b, h, s, d, dtype):
    return torch.randn(b, s, h, d, device="cuda", dtype=dtype).transpose(1, 2)


def ref(q, k, v, *, scale, causal=False, bottom_right=False, seq_q_lens=None, seq_kv_lens=None, sinks=None, window_left=None, right_bound=None):
    """fp64 oracle of the whole contract: O and the natural-log LSE per row class (keyless rows -inf / 0)."""
    b, h_q, s_q, _ = q.shape
    _, h_kv, s_kv, _ = v.shape
    dev = q.device
    g = h_q // h_kv
    k_ref = k.repeat_interleave(g, dim=1).double()
    v_ref = v.repeat_interleave(g, dim=1).double()
    scores = torch.matmul(q.double(), k_ref.transpose(-1, -2)) * scale
    q_lens = seq_q_lens.flatten().to(device=dev, dtype=torch.int64) if seq_q_lens is not None else torch.full((b,), s_q, dtype=torch.int64, device=dev)
    kv_lens = seq_kv_lens.flatten().to(device=dev, dtype=torch.int64) if seq_kv_lens is not None else torch.full((b,), s_kv, dtype=torch.int64, device=dev)
    i = torch.arange(s_q, device=dev).view(1, 1, s_q, 1)
    j = torch.arange(s_kv, device=dev).view(1, 1, 1, s_kv)
    diag = i + (kv_lens - q_lens).view(b, 1, 1, 1) if bottom_right else i
    dead_q = i >= q_lens.view(b, 1, 1, 1)
    masked = dead_q | (j >= kv_lens.view(b, 1, 1, 1))
    if causal:
        masked = masked | (j > diag)
    if right_bound is not None:
        masked = masked | (j > diag + right_bound)
    if window_left is not None:
        masked = masked | (j < diag - window_left)
    scores = scores.masked_fill(masked, float("-inf"))
    full = scores if sinks is None else torch.cat([scores, sinks.view(1, h_q, 1, 1).double().expand(b, h_q, s_q, 1)], dim=-1)
    smax = full.max(dim=-1).values
    ssum = torch.exp(full - smax.unsqueeze(-1)).sum(dim=-1).nan_to_num(0.0)
    lse = smax + torch.log(ssum)
    o = torch.matmul(torch.softmax(full, dim=-1).nan_to_num(0.0)[..., :s_kv], v_ref)
    dead = dead_q.squeeze(-1)
    return dict(o=o, lse=lse.masked_fill(dead, float("-inf")), smax=smax.masked_fill(dead, float("-inf")), empty=torch.isneginf(lse) | dead)


def case_inputs(case, d, dtype, *, seed=0):
    torch.manual_seed(seed)
    c = CASES[case]
    q = bhsd(c["b"], c["h"], c["s_q"], d, dtype)
    k = bhsd(c["b"], c["h_kv"], c["s_kv"], d, dtype)
    v = bhsd(c["b"], c["h_kv"], c["s_kv"], d, dtype)
    kv_lens = torch.tensor(c["kv_lens"], dtype=torch.int32, device="cuda").view(c["b"], 1, 1, 1) if "kv_lens" in c else None
    q_lens = torch.tensor(c["q_lens"], dtype=torch.int32, device="cuda").view(c["b"], 1, 1, 1) if "q_lens" in c else None
    sink = torch.randn(1, c["h"], 1, 1, dtype=torch.float32, device="cuda") if c.get("sink") else None
    causal = bool(c.get("causal") or c.get("bottom_right"))
    sdpa_kwargs = dict(use_causal_mask_bottom_right=True) if c.get("bottom_right") else (dict(use_causal_mask=True) if causal else {})
    window_left = c.get("window_left")
    if window_left is not None:
        sdpa_kwargs["sliding_window_length"] = window_left + 1  # the graph's length L keeps L keys up to the diagonal: W = L - 1
    right_bound = c.get("right_bound")
    if right_bound is not None:
        sdpa_kwargs["diagonal_band_right_bound"] = right_bound  # alone: with use_causal_mask the bound is forced back to 0
    scale = 1.0 / math.sqrt(d) if not c.get("bottom_right") else 0.75
    oracle = ref(
        q, k, v, scale=scale, causal=causal, bottom_right=bool(c.get("bottom_right")), seq_q_lens=q_lens, seq_kv_lens=kv_lens,
        sinks=sink.flatten() if sink is not None else None, window_left=window_left, right_bound=right_bound,
    )  # fmt: skip
    return q, k, v, dict(scale=scale, sdpa_kwargs=sdpa_kwargs, seq_len_kv=kv_lens, seq_len_q=q_lens, sink=sink), oracle


def run_graph(
    q,
    k,
    v,
    *,
    row,
    scale,
    sdpa_kwargs=None,
    seq_len_kv=None,
    seq_len_q=None,
    sink=None,
    stats=True,
    stats_log2=False,
    cga=None,
    split_kv=None,
    score_outputs=False,
):
    """Build the graph, pin ``row`` (optionally a cga / split_kv knob through the public autotune record, optionally the two score
    side outputs), launch TWICE on sentinel-filled outputs (the two-launch delta) and return O, Stats, the side outputs and the
    plan's knobs."""
    import cudnn

    b, h_q, s_q, _ = q.shape
    d_v = v.shape[-1]
    dtype = q.dtype
    o_gpu = torch.empty(b, s_q, h_q, d_v, device="cuda", dtype=dtype).transpose(1, 2)
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq, tk, tv = g.tensor_like(q), g.tensor_like(k), g.tensor_like(v)
    kw = dict(name="sdpa", q=tq, k=tk, v=tv, generate_stats=stats, attn_scale=scale, stats_use_log2=stats_log2)
    vp = {tq: q, tk: k, tv: v}
    side = {}
    if score_outputs:  # the graph's two optional ports, declared like Stats (b, h, s, 1) fp32
        for name in ("score_max", "score_sum_exp"):
            buf = make_dense_stats(b, h_q, s_q, "contiguous")
            t = g.tensor(dim=list(buf.shape), stride=list(buf.stride()), data_type=cudnn.data_type.FLOAT, name=name)
            kw[name] = t
            vp[t] = buf
            side[name] = buf
    if seq_len_kv is not None:
        slk = g.tensor_like(seq_len_kv)
        kw.update(seq_len_kv=slk, use_padding_mask=True)
        vp[slk] = seq_len_kv
        slq_t = seq_len_q if seq_len_q is not None else torch.full((b, 1, 1, 1), s_q, dtype=torch.int32, device="cuda")
        slq = g.tensor_like(slq_t)
        kw["seq_len_q"] = slq
        vp[slq] = slq_t
    if sink is not None:
        st = g.tensor_like(sink)
        kw["sink_token"] = st
        vp[st] = sink
    kw.update(sdpa_kwargs or {})
    o, stats_t = g.sdpa(**kw)
    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    stats_gpu = None
    if stats:
        stats_gpu = make_dense_stats(b, h_q, s_q, "contiguous")
        stats_t.set_output(True).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)
        vp[stats_t] = stats_gpu
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    ours = [i for i, n in enumerate(names) if n == row or n.startswith(row + "[")]
    idx = next((i for i in ours if g.plans[i].knobs.split_kv in (None, 1)), None)
    assert idx is not None, f"no unsplit {row} plan in {names}"
    picked_cga, picked_split = g.plans[idx].knobs.cga, g.plans[idx].knobs.split_kv or 1
    if (cga is not None and picked_cga != cga) or (split_kv is not None and picked_split != split_kv):
        # The heuristics rank ONE knob point per row (the auto pick); another member of the row's cga / split domain is requested
        # the way an autotuner replays it -- the public record (engine_id, {knob_type: int}) through create_execution_plan.
        engine_id, record = g.get_engine_and_knobs_at_index(idx)
        record = dict(record)
        if cga is not None:
            record[cudnn.knob_type.TILE_CGA_M] = cga
            picked_cga = cga
        if split_kv is not None:
            record[cudnn.knob_type.SPLIT_KV] = split_kv
            picked_split = split_kv
        g.create_execution_plan(engine_id, record)
        idx = g.get_execution_plan_count() - 1
    g.select_plan(idx)
    knobs = SimpleNamespace(cga=picked_cga, split_kv=picked_split)
    g.check_support()
    g.build_plans()
    vp[o] = o_gpu
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)

    def launch():
        o_gpu.fill_(float("nan"))  # sentinel: an unwritten live row stays NaN
        if stats_gpu is not None:
            stats_gpu.fill_(float("nan"))
        for buf in side.values():
            buf.fill_(float("nan"))
        g.execute(vp, ws)
        torch.cuda.synchronize()
        return o_gpu.clone(), (stats_gpu.clone() if stats_gpu is not None else None), {n: buf.clone() for n, buf in side.items()}

    o1, st1, side1 = launch()
    o2, st2, side2 = launch()
    assert torch.equal(o1, o2) and (st1 is None or torch.equal(st1, st2)), "two-launch delta on O / Stats"
    assert all(torch.equal(side1[n], side2[n]) for n in side), "two-launch delta on the score side outputs"
    return dict(o=o_gpu, stats=stats_gpu, knobs=knobs, **side)


def check(out, oracle, *, has_stats=True, stats_log2=False):
    """Keyless rows EXACTLY 0 / -inf (never NaN, never a residue); live rows against the fp64 oracle."""
    empty = oracle["empty"]
    o = out["o"]
    assert torch.isfinite(o).all(), "O has a non-finite (an unwritten sentinel or a NaN residue)"
    assert (o[empty] == 0).all(), "a keyless row's O must be exactly 0 (a SELECT, never a multiply by zero)"
    torch.testing.assert_close(o[~empty].double(), oracle["o"][~empty], **TOL_O)
    if has_stats:
        lse = out["stats"].squeeze(-1)
        assert torch.isneginf(lse[empty]).all(), "a keyless row's Stats must be exactly -inf"
        ref_lse = oracle["lse"][~empty] * (math.log2(math.e) if stats_log2 else 1.0)
        torch.testing.assert_close(lse[~empty].double(), ref_lse, **TOL_LSE)


def check_degenerate_rows(out, case):
    """The per-case output assertions beyond the oracle: a zero-length KV batch reads O = 0 / Stats = -inf; rows past a left window are keyless."""
    c = CASES[case]
    if "kv_lens" in c and 0 in c["kv_lens"]:
        empty_b = c["kv_lens"].index(0)
        assert (out["o"][empty_b] == 0).all() and torch.isneginf(out["stats"][empty_b]).all(), "a zero-length KV batch: O = 0, Stats = -inf"
    if "window_left" in c:
        first_dead = c["kv_lens"][1] + c["window_left"]
        lse = out["stats"].squeeze(-1)
        assert torch.isneginf(lse[1, :, first_dead:]).all() and (out["o"][1, :, first_dead:] == 0).all(), "rows past the window are keyless"
        assert torch.isfinite(lse[1, :, :first_dead]).all() and torch.isfinite(lse[0]).all()
