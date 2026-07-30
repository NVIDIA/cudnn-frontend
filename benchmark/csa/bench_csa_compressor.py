# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSA fused-compressor benchmark with CUDA-graph variants (PR #427 fairness follow-up).

This harness measures the wall-clock cost of the gated-softmax pooling region that the
fused ``cudnn.csa`` kernels replace, over the PR's production shapes (ratio=4, coff=2),
in a **single run** that emits two tables:

  * per-call    -- each variant timed with a pair of CUDA events and a synchronize,
                   launch/host overhead included (reproduces the methodology behind the
                   per-call wall-clock table in docs/fe-oss-apis/csa.md).
  * graph       -- the SAME region captured into a CUDA graph and timed as a replay.

Graph variants are captured **symmetrically** for eager and fused, as two directly
measured columns (no subtraction):

  * fwd-graph   -- forward only. Eager forward is captured under ``torch.no_grad`` (no
                   autograd graph); fused forward is the wrapper (no autograd by design).
  * total-graph -- forward + backward captured together. Eager backward is the autograd
                   backward of the captured forward; it is captured with **stable,
                   pre-allocated zero ``.grad`` buffers** (the supported torch graph
                   pattern): the captured region zeros those buffers in place, then runs
                   forward + backward, so every replay accumulates into a zeroed buffer --
                   numerically identical to a single fresh backward (verified once per
                   shape via a graph-vs-fresh-backward cross-check printed at the end).
                   Fused backward is the explicit backward wrapper (kernel launches, no
                   autograd), captured right after the fused forward.

The backward replay alone is NOT a separately captured quantity (the eager total graph
captures fwd+bwd as one unit and cannot be split into independent replays); it is
approximately ``total - fwd`` and is reported only as that reference, never as a column.

Motivation: the eager region is ~39 forward and ~51 backward kernel launches per call, so
its per-call wall clock is dominated by launch/host overhead the fused path (1 + 1
kernels) does not pay. Capturing each side into a graph collapses its per-operation
launches into a single replay, which is the fairest wall-clock basis for launch-bound
shapes. Numbers are reported as-measured (median of ``--iters`` after ``--warmup``); no
variant is favored.

Not collected by pytest. Run, e.g.::

    CUDA_VISIBLE_DEVICES=0 python benchmark/csa/bench_csa_compressor.py --iters 50 --warmup 20
"""

import argparse
import contextlib
import json
import math
import statistics

import torch

import cudnn.csa

# ---------------------------------------------------------------------------
# Eager reference (verbatim upstream numerics; mirrors test_CSA_compressor.py)
# ---------------------------------------------------------------------------


def _batch_of_row(cu_seqlens, total):
    """Segment index owning each packed row (mirror of Megatron-LM ``batch_of_row``)."""
    n_seg = cu_seqlens.shape[0] - 1
    row_idx = torch.arange(total, device=cu_seqlens.device, dtype=torch.int64)
    return torch.bucketize(row_idx, cu_seqlens[1:], right=True).clamp(max=max(n_seg - 1, 0))


def _overlap_transform_thd(tensor, is_first_in_seg, head_dim, fill_value=0):
    """Mirror of Megatron-LM Compressor._overlap_transform_thd (coff == 2)."""
    n, ratio, b_dim, _ = tensor.size()
    d = head_dim
    new_tensor = tensor.new_full((n, 2 * ratio, b_dim, d), fill_value)
    new_tensor[:, ratio:] = tensor[:, :, :, d:]
    prev_data = torch.roll(tensor[:, :, :, :d], shifts=1, dims=0)
    prev_data[is_first_in_seg] = fill_value
    new_tensor[:, :ratio] = prev_data
    return new_tensor


def eager_pool(kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, d, coff):
    """Verbatim upstream eager pooling region (softmax weights rounded to bf16)."""
    device = kv.device
    row_idx = torch.arange(total_comp, device=device, dtype=cu_seqlens_comp.dtype)
    batch_ids = _batch_of_row(cu_seqlens_comp, total_comp)
    valid_comp = row_idx < cu_seqlens_comp[-1]
    local_pos = row_idx - cu_seqlens_comp[batch_ids]
    local_pos = torch.where(valid_comp, local_pos, torch.zeros_like(local_pos))
    base = cu_seqlens[batch_ids].unsqueeze(1) + local_pos.unsqueeze(1) * ratio
    base = torch.where(valid_comp.unsqueeze(1), base, torch.zeros_like(base))
    offsets = torch.arange(ratio, device=device, dtype=base.dtype).unsqueeze(0)
    gather_idx = base + offsets  # (total_comp, ratio)

    kv_grouped = kv[gather_idx]
    score_grouped = score[gather_idx]
    score_grouped = score_grouped + ape.view(1, ratio, 1, -1)

    if coff == 2:
        is_first = local_pos == 0
        kv_grouped = _overlap_transform_thd(kv_grouped, is_first, d, fill_value=0)
        score_grouped = _overlap_transform_thd(score_grouped, is_first, d, fill_value=float("-inf"))

    weights = torch.softmax(score_grouped, dim=1, dtype=torch.float32).to(kv_grouped.dtype)
    out = (kv_grouped * weights).sum(dim=1)
    return out  # (total_comp, 1, d)


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def make_inputs(lens, d, ratio, coff, seed=1234, device="cuda"):
    """Build a random THD pack (kv, score, ape, cu, cuc, total_comp, grad_out) for ``lens``."""
    total = sum(lens)
    w = coff * d
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kv = torch.randn(total, 1, w, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    score = (torch.randn(total, 1, w, generator=gen, dtype=torch.float32).mul_(1.5)).to(torch.bfloat16)
    ape = torch.randn(ratio, w, generator=gen, dtype=torch.float32).mul_(0.25)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0)], dtype=torch.int32, device=device)
    seg_comp = torch.tensor([seg_len // ratio for seg_len in lens])
    cuc = torch.tensor([0, *seg_comp.cumsum(0)], dtype=torch.int32, device=device)
    total_comp = int(cuc[-1].item())
    go = torch.randn(total_comp, 1, d, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    return kv.to(device), score.to(device), ape.to(device), cu, cuc, total_comp, go.to(device)


# ---------------------------------------------------------------------------
# Variant callables (static-buffer friendly)
# ---------------------------------------------------------------------------


def fused_forward(kv, score, ape, cu, cuc, total_comp, ratio, d, coff):
    """One fused forward wrapper call; returns the pooled output."""
    total = kv.shape[0]
    return cudnn.csa.csa_compressor_forward_wrapper(
        kv.view(total, -1),
        score.view(total, -1),
        ape,
        cu,
        cuc,
        ratio=ratio,
        head_dim=d,
        coff=coff,
        total_comp=total_comp,
    )["out"]


def fused_backward(kv, score, ape, cu, cuc, go, ratio, d, coff):
    """One fused backward wrapper call; returns the gradient TupleDict."""
    total = kv.shape[0]
    return cudnn.csa.csa_compressor_backward_wrapper(
        kv.view(total, -1),
        score.view(total, -1),
        ape,
        cu,
        cuc,
        go.view(go.shape[0], d),
        ratio=ratio,
        head_dim=d,
        coff=coff,
    )


def fused_forward_backward(kv, score, ape, cu, cuc, go, total_comp, ratio, d, coff):
    """Fused forward immediately followed by the fused backward (the total region)."""
    fused_forward(kv, score, ape, cu, cuc, total_comp, ratio, d, coff)
    fused_backward(kv, score, ape, cu, cuc, go, ratio, d, coff)


def eager_forward(kv, score, ape, cu, cuc, total_comp, ratio, d, coff):
    """One eager forward over the replaced region (verbatim upstream numerics)."""
    return eager_pool(kv, score, ape, cu, cuc, total_comp, ratio, d, coff)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _noop():
    """Do nothing (placeholder ``pre`` hook for graph replay timing)."""
    pass


def _median_event_ms(fn, warmup, iters):
    """Median CUDA-event wall clock (ms) of per-call fn() (launch overhead included)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def _median_eager_bwd_event_ms(make_fwd, go, warmup, iters):
    """Median CUDA-event wall clock (ms) of ONLY the autograd backward.

    The forward that builds the grad graph runs OUTSIDE the timed region (matching the PR
    harness and the docs' "eager backward goes through torch autograd" basis). ``make_fwd``
    returns a fresh leaf set plus the forward output, so backward always starts from an
    unallocated ``.grad`` (the fresh-gradient path).
    """
    for _ in range(warmup):
        _kvl, _scl, _apl, o = make_fwd()
        o.backward(go)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        _kvl, _scl, _apl, o = make_fwd()  # forward outside timing
        torch.cuda.synchronize()
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        o.backward(go)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def _median_graph_ms(graph, pre, warmup, iters):
    """Median CUDA-event wall clock (ms) of graph replays. ``pre`` runs before each replay."""
    for _ in range(warmup):
        pre()
        graph.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        pre()
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        graph.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def _capture(fn, warmup, no_grad=False):
    """Warmup fn() on a side stream, then capture fn() into a CUDAGraph (capture-safe)."""
    cm = torch.no_grad() if no_grad else contextlib.nullcontext()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        with cm:
            for _ in range(warmup):
                fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with cm:
            fn()
    return g


def _capture_eager_total(kv, score, ape, cu, cuc, total_comp, go, ratio, d, coff, warmup):
    """Capture eager fwd+bwd with STABLE pre-allocated zero ``.grad`` buffers.

    The leaf ``.grad`` tensors are created as zeros BEFORE warmup and kept across
    warmup/capture/replay (never released). The captured region zeros them IN PLACE, then
    runs forward + autograd backward, so every replay accumulates into a zeroed buffer --
    numerically identical to a single fresh backward. The in-place zero cost is inside the
    graph (honestly counted in the total-graph time).

    Returns ``(graph, check)`` where ``check`` compares one graph replay's grads against a
    fresh non-graph backward (graph-vs-fresh numerical-consistency evidence).
    """
    kvl, scl, apl = _make_leaves(kv, score, ape)
    # Stable zero grad buffers: fixed addresses the captured graph can replay against.
    kvl.grad = torch.zeros_like(kvl)
    scl.grad = torch.zeros_like(scl)
    apl.grad = torch.zeros_like(apl)

    def _fwd_bwd():
        """The captured region: zero the kept grad buffers, then eager fwd + bwd."""
        # In-place zero of the SAME buffers each replay (kept, not released).
        kvl.grad.zero_()
        scl.grad.zero_()
        apl.grad.zero_()
        o = eager_forward(kvl, scl, apl, cu, cuc, total_comp, ratio, d, coff)
        o.backward(go)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            _fwd_bwd()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _fwd_bwd()

    # ---- numerics cross-check: one graph replay vs a fresh non-graph backward ----
    g.replay()
    torch.cuda.synchronize()
    gk, gs, ga = kvl.grad.clone(), scl.grad.clone(), apl.grad.clone()

    ref_k, ref_s, ref_a = _make_leaves(kv, score, ape)
    o_ref = eager_forward(ref_k, ref_s, ref_a, cu, cuc, total_comp, ratio, d, coff)
    o_ref.backward(go)
    torch.cuda.synchronize()

    check = {
        "equal_dKV": torch.equal(gk, ref_k.grad),
        "equal_dScore": torch.equal(gs, ref_s.grad),
        "max_abs_dKV": (gk.float() - ref_k.grad.float()).abs().max().item(),
        "max_abs_dScore": (gs.float() - ref_s.grad.float()).abs().max().item(),
        "max_abs_dAPE": (ga - ref_a.grad).abs().max().item(),
        "allclose": (
            torch.allclose(gk, ref_k.grad, rtol=2e-3, atol=2e-3)
            and torch.allclose(gs, ref_s.grad, rtol=2e-3, atol=2e-3)
            and torch.allclose(ga, ref_a.grad, rtol=1e-4, atol=1e-4)
        ),
    }
    return g, check


# ---------------------------------------------------------------------------
# Per-shape measurement
# ---------------------------------------------------------------------------


def measure(lens, d, ratio, coff, warmup, iters, seed=1234):
    """Measure per-call and graph-replay wall clock for one shape (eager vs fused).

    Returns a dict of median CUDA-event timings (ms) plus a numerics ``check`` comparing
    one eager total-graph replay against a fresh non-graph backward.
    """
    kv, score, ape, cu, cuc, total_comp, go = make_inputs(lens, d, ratio, coff, seed)
    out = {}

    # ---- per-call (non-graph): reproduces the PR's event methodology ----
    out["eager_fwd"] = _median_event_ms(lambda: eager_forward(kv, score, ape, cu, cuc, total_comp, ratio, d, coff), warmup, iters)
    out["fused_fwd"] = _median_event_ms(lambda: fused_forward(kv, score, ape, cu, cuc, total_comp, ratio, d, coff), warmup, iters)

    # eager backward per-call: forward (builds grad graph) outside timing, backward timed.
    def _fwd_for_bwd():
        """Fresh leaves + eager forward (outside timing) feeding one timed backward."""
        kvl, scl, apl = _make_leaves(kv, score, ape)
        o = eager_forward(kvl, scl, apl, cu, cuc, total_comp, ratio, d, coff)
        return kvl, scl, apl, o

    out["eager_bwd"] = _median_eager_bwd_event_ms(_fwd_for_bwd, go, warmup, iters)

    out["fused_bwd"] = _median_event_ms(lambda: fused_backward(kv, score, ape, cu, cuc, go, ratio, d, coff), warmup, iters)

    # ---- graph variants ----
    # forward-only graphs (eager under no_grad so no autograd graph is built).
    g_eager_fwd = _capture(
        lambda: eager_forward(kv, score, ape, cu, cuc, total_comp, ratio, d, coff),
        warmup,
        no_grad=True,
    )
    out["eager_fwd_graph"] = _median_graph_ms(g_eager_fwd, pre=_noop, warmup=warmup, iters=iters)
    del g_eager_fwd

    g_fused_fwd = _capture(lambda: fused_forward(kv, score, ape, cu, cuc, total_comp, ratio, d, coff), warmup)
    out["fused_fwd_graph"] = _median_graph_ms(g_fused_fwd, pre=_noop, warmup=warmup, iters=iters)
    del g_fused_fwd

    # total graphs: forward + backward captured together (static grad buffers for eager).
    g_eager_total, out["check"] = _capture_eager_total(kv, score, ape, cu, cuc, total_comp, go, ratio, d, coff, warmup)
    out["eager_total_graph"] = _median_graph_ms(g_eager_total, pre=_noop, warmup=warmup, iters=iters)
    del g_eager_total

    g_fused_total = _capture(
        lambda: fused_forward_backward(kv, score, ape, cu, cuc, go, total_comp, ratio, d, coff),
        warmup,
    )
    out["fused_total_graph"] = _median_graph_ms(g_fused_total, pre=_noop, warmup=warmup, iters=iters)
    del g_fused_total

    return out


def _fmt(ms):
    """Format a millisecond value as a fixed-width microsecond column."""
    return f"{ms * 1000:7.1f}"


def _make_leaves(kv, score, ape):
    """Fresh grad-enabled leaf clones of the three inputs (one construction site)."""
    return kv.clone().requires_grad_(True), score.clone().requires_grad_(True), ape.clone().requires_grad_(True)


def _x(num, den):
    """Speedup helper (callers pass displayed 1-decimal values)."""
    return float("inf") if den == 0 else num / den


def _xtrunc1(num, den):
    """Speedup from displayed 1-decimal values, truncated to 1 decimal (never rounded up)."""
    return float("inf") if den == 0 else math.floor((num / den) * 10) / 10


def main():
    """Run all shapes and print the per-call/graph tables, checks, and JSON payload."""
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    shapes = [
        ([8192], 128),
        ([8192, 8192, 8192], 128),
        ([8192], 512),
        ([8192, 8192, 8192], 512),
    ]
    ratio, coff = 4, 2

    print(f"# CSA compressor benchmark -- ratio={ratio} coff={coff} " f"warmup={args.warmup} iters={args.iters} (median, CUDA events, us)")
    print(f"# GPU: {torch.cuda.get_device_name()} CC={torch.cuda.get_device_capability()}")

    rows = []
    for lens, d in shapes:
        r = measure(lens, d, ratio, coff, args.warmup, args.iters, args.seed)
        tag = f"{len(lens)}x{lens[0] // 1000}k/d{d}"
        rows.append((tag, d, r))
        torch.cuda.empty_cache()

    # ---- per-call table ----
    print("\n# per-call wall clock (CUDA events, launch overhead included):")
    h1 = f"{'shape':>14} | {'eager_fwd':>9} {'fused_fwd':>9} | {'eager_bwd':>9} {'fused_bwd':>9}"
    print(h1)
    print("-" * len(h1))
    for tag, _, r in rows:
        print(f"{tag:>14} | {_fmt(r['eager_fwd'])} {_fmt(r['fused_fwd'])} | " f"{_fmt(r['eager_bwd'])} {_fmt(r['fused_bwd'])}")

    # ---- graph table (fwd-graph and total-graph, directly measured; no bwd-graph) ----
    print("\n# graph replay (fwd-only graph; fwd+bwd total graph):")
    h2 = f"{'shape':>14} | {'e_fgraph':>9} {'f_fgraph':>9} {'fwd':>6} | " f"{'e_tgraph':>9} {'f_tgraph':>9} {'total':>6}"
    print(h2)
    print("-" * len(h2))
    for tag, _, r in rows:
        efg = round(r["eager_fwd_graph"] * 1000, 1)
        ffg = round(r["fused_fwd_graph"] * 1000, 1)
        etg = round(r["eager_total_graph"] * 1000, 1)
        ftg = round(r["fused_total_graph"] * 1000, 1)
        print(
            f"{tag:>14} | {_fmt(r['eager_fwd_graph'])} {_fmt(r['fused_fwd_graph'])} "
            f"{_xtrunc1(efg, ffg):5.1f}x | {_fmt(r['eager_total_graph'])} {_fmt(r['fused_total_graph'])} "
            f"{_xtrunc1(etg, ftg):5.1f}x"
        )

    # ---- speedup summary (eager / fused), per-call and graph, from displayed values ----
    print("\n# speedup (eager / fused), from displayed values:")
    for tag, _, r in rows:
        ef = round(r["eager_fwd"] * 1000, 1)
        ff = round(r["fused_fwd"] * 1000, 1)
        eb = round(r["eager_bwd"] * 1000, 1)
        fb = round(r["fused_bwd"] * 1000, 1)
        efg = round(r["eager_fwd_graph"] * 1000, 1)
        ffg = round(r["fused_fwd_graph"] * 1000, 1)
        etg = round(r["eager_total_graph"] * 1000, 1)
        ftg = round(r["fused_total_graph"] * 1000, 1)
        print(f"  {tag:>14}: fwd {_x(ef, ff):5.2f}x (fwd-graph {_x(efg, ffg):5.2f}x) | " f"bwd {_x(eb, fb):5.2f}x | total-graph {_x(etg, ftg):5.2f}x")

    # ---- eager graph-vs-fresh-backward numerics cross-check ----
    print("\n# eager total-graph vs fresh non-graph backward (numerical consistency):")
    for tag, _, r in rows:
        c = r["check"]
        flag = "PASS" if c["allclose"] else "FAIL"
        print(
            f"  {tag:>14}: dKV eq={c['equal_dKV']} max={c['max_abs_dKV']:.3e} | "
            f"dScore eq={c['equal_dScore']} max={c['max_abs_dScore']:.3e} | "
            f"dAPE max={c['max_abs_dAPE']:.3e} -> {flag}"
        )

    # ---- machine-readable JSON ----
    payload = {
        "gpu": torch.cuda.get_device_name(),
        "cc": list(torch.cuda.get_device_capability()),
        "ratio": ratio,
        "coff": coff,
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": [
            {
                "shape": t,
                "head_dim": d,
                "check": r["check"],
                **{k: round(v * 1000.0, 1) for k, v in r.items() if k != "check"},
            }
            for t, d, r in rows
        ],
    }
    print("\n# JSON\n" + json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
