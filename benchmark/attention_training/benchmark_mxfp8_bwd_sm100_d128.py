# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare SM100 / SM103 D128 MXFP8 backward engines on the same quantized inputs.

    CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 python benchmark/attention_training/benchmark_mxfp8_bwd_sm100_d128.py \\
        --s 2048 4096 8192 16384 --b 1 --hq 88 --hkv 8 --check --check-tol 0.01

Engines are pinned by plan name. The ``sdpa_bwd_sm100_mxfp8_pair`` engine uses a BSHD copy; other
engines use BHSD. Timing reports the median of interleaved single-call CUDA
graph replays, each after a 512 MiB L2 flush and a 0.2 s idle gap. CUDA events
measure the entire chain, including PDL overlap.

``--check`` reports gradient relative-L2 differences from the first engine.
``--check-tol`` also fails on missing engines or nonfinite/excessive differences.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch

import cudnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "test" / "python"))  # sdpa.mxfp8_quant (the repo's MXFP8 test quantizer)

D = 128
BLOCK = 32


def _cdiv(a, b):
    return -(-a // b)


def _stride(shape, layout):
    b, h, s, d = shape
    return [h * s * d, s * d, d, 1] if layout == "bhsd" else [s * h * d, d, h * d, 1]


def _sf_dims(b, h, s):
    return (b, h, _cdiv(s, 128) * 128, _cdiv(_cdiv(D, BLOCK), 4) * 4), (b, h, _cdiv(_cdiv(s, BLOCK), 4) * 4, _cdiv(D, 128) * 128)


def make_inputs(b, hq, hkv, s, scale, seed=0, causal=False):
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    torch.manual_seed(seed)
    q, k, v, do = (torch.randn(b, h, s, D, device="cuda") for h in (hq, hkv, hkv, hq))
    quant = {}
    for name, x, h in (("q", q, hq), ("k", k, hkv), ("v", v, hkv), ("dO", do, hq)):
        d8, _, sfd, s8, _, sfs = quantize_to_mxfp8(x, b, h, s, D, BLOCK, torch.float8_e4m3fn, with_ref=False)
        quant[name] = (d8.contiguous(), s8.contiguous(), sfd.contiguous(), sfs.contiguous())
    # fp32 forward per (batch, head): O and the natural-log LSE the backward consumes
    o = torch.empty(b, hq, s, D, device="cuda")
    stats = torch.empty(b, hq, s, 1, device="cuda")
    grp = hq // hkv
    for bb in range(b):
        for h in range(hq):
            sc = (q[bb, h] @ k[bb, h // grp].T) * scale
            if causal:
                sc = sc.masked_fill(torch.ones(s, s, dtype=torch.bool, device="cuda").triu(1), float("-inf"))
            stats[bb, h, :, 0] = torch.logsumexp(sc, dim=-1)
            o[bb, h] = torch.softmax(sc, dim=-1) @ v[bb, h // grp]
            del sc
    return dict(quant=quant, o=o.to(torch.bfloat16), dO_f16=do.to(torch.bfloat16), stats=stats)


def _to_layout(x, layout):
    """A BHSD-contiguous tensor's values in ``layout`` storage, logical dims still (B, H, S, D)."""
    return x if layout == "bhsd" else x.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


def build(engine, b, hq, hkv, s, scale, inputs, layout, amax=False, deterministic=False, causal=False):
    fp8 = cudnn.data_type.FP8_E4M3
    g = cudnn.pygraph(io_data_type=fp8, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    shq, shk = (b, hq, s, D), (b, hkv, s, D)
    t = {}
    for name, sh in (("q", shq), ("q_T", shq), ("k", shk), ("k_T", shk), ("v", shk), ("dO", shq), ("dO_T", shq)):
        t[name] = g.tensor(name=name, dim=list(sh), stride=_stride(sh, layout), data_type=fp8)
    for name in ("o_f16", "dO_f16"):
        t[name] = g.tensor(name=name, dim=list(shq), stride=_stride(shq, layout), data_type=cudnn.data_type.BFLOAT16)
    t["stats"] = g.tensor(name="stats", dim=[b, hq, s, 1], stride=[hq * s, s, 1, 1], data_type=cudnn.data_type.FLOAT)
    q_row, q_col = _sf_dims(b, hq, s)
    k_row, k_col = _sf_dims(b, hkv, s)
    for name, dims in (("sf_q", q_row), ("sf_q_T", q_col), ("sf_k", k_row), ("sf_k_T", k_col), ("sf_v", k_row), ("sf_dO", q_row), ("sf_dO_T", q_col)):
        t[name] = g.tensor(
            name=name,
            dim=list(dims),
            stride=[dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1],
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )
    outs = g.sdpa_mxfp8_backward(
        name="bwd", q=t["q"], q_T=t["q_T"], k=t["k"], k_T=t["k_T"], v=t["v"], o_f16=t["o_f16"], dO_f16=t["dO_f16"], dO=t["dO"],
        dO_T=t["dO_T"], stats=t["stats"], descale_q=t["sf_q"], descale_q_T=t["sf_q_T"], descale_k=t["sf_k"], descale_k_T=t["sf_k_T"],
        descale_v=t["sf_v"], descale_dO=t["sf_dO"], descale_dO_T=t["sf_dO_T"], attn_scale=scale,
        **({"use_deterministic_algorithm": True} if deterministic else {}),
        **({"use_causal_mask": True} if causal else {}),
    )  # fmt: skip
    dq_t, dk_t, dv_t, *amax_t = outs
    for o_t, sh in ((dq_t, shq), (dk_t, shk), (dv_t, shk)):
        o_t.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim(list(sh)).set_stride(_stride(sh, layout))
    for a in amax_t:
        a.set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
        if amax:
            a.set_output(True)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]
    if engine == "backend":
        idx = next((i for i, n in enumerate(names) if not n.startswith("sdpa_")), None)
    else:
        idx = next((i for i, n in enumerate(names) if n == engine or n.startswith(engine + "[")), None)
    if idx is None:
        raise RuntimeError(f"{engine} is not offered for this graph; plans = {names}")
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    pack = {}
    for name, (row, col, sfr, sfc) in inputs["quant"].items():
        if name == "v":
            pack[t["v"]] = _to_layout(row, layout)
            pack[t["sf_v"]] = sfr
            continue
        pack[t[name]] = _to_layout(row, layout)
        pack[t[name + "_T"]] = _to_layout(col, layout)
        pack[t["sf_" + name]] = sfr
        pack[t["sf_" + name + "_T"]] = sfc
    pack[t["o_f16"]] = _to_layout(inputs["o"], layout)
    pack[t["dO_f16"]] = _to_layout(inputs["dO_f16"], layout)
    pack[t["stats"]] = inputs["stats"]
    grads = tuple(_to_layout(torch.empty(sh, device="cuda", dtype=torch.bfloat16), layout) for sh in (shq, shk, shk))
    pack.update(dict(zip((dq_t, dk_t, dv_t), grads)))
    if amax:
        pack.update({a: torch.zeros(1, 1, 1, 1, device="cuda") for a in amax_t})
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    # every engine (backend and python rows alike) launches on the handle's stream: one handle per run, re-streamed for
    # CUDA-graph capture
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)
    return dict(engine=engine, plan=names[idx], layout=layout, graph=g, pack=pack, ws=ws, grads=grads, handle=handle,
                call=lambda: g.execute(pack, ws, handle=handle))  # fmt: skip


def _capture(run):
    """A CUDA graph of one call (the handle re-streamed onto the capture stream, then back)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    cudnn.set_stream(handle=run["handle"], stream=side.cuda_stream)
    try:
        with torch.cuda.stream(side):
            run["call"]()  # warm-up on the capture stream
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            run["call"]()
    finally:
        torch.cuda.synchronize()
        cudnn.set_stream(handle=run["handle"], stream=torch.cuda.current_stream().cuda_stream)
    return graph


def _elapsed_ms(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


def time_cold(runs, reps):
    """Median ms per engine of ``reps`` cold calls, each after an L2 flush and an idle gap, the engines interleaved."""
    flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    graphs = {r["engine"]: _capture(r) for r in runs}
    samples = {r["engine"]: [] for r in runs}
    for _ in range(reps):
        for r in runs:
            flush.zero_()
            torch.cuda.synchronize()
            time.sleep(0.2)
            samples[r["engine"]].append(_elapsed_ms(graphs[r["engine"]].replay))
    out = {n: statistics.median(v) for n, v in samples.items()}
    bad = {n: v for n, v in out.items() if not (math.isfinite(v) and v > 0)}
    if bad:  # a broken measurement, not a result
        raise RuntimeError(f"timing produced non-finite or non-positive medians: {bad}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--hq", type=int, default=88)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--scale", type=float, default=None, help="attn_scale (default 1/sqrt(128))")
    ap.add_argument("--engines", nargs="+", default=["backend", "sdpa_bwd_sm100_d128_mxfp8", "sdpa_bwd_sm100_mxfp8_pair"])
    ap.add_argument("--reps", type=int, default=7, help="timed cold calls per engine (median)")
    ap.add_argument("--amax", action="store_true", help="request amax_dQ / dK / dV (engines that cannot are reported as skipped)")
    ap.add_argument("--deterministic", action="store_true", help="request use_deterministic_algorithm (batch-invariant plans)")
    ap.add_argument("--causal", action="store_true", help="top-left causal mask")
    ap.add_argument("--check", action="store_true", help="relative L2 difference of each engine's outputs from the first engine's")
    ap.add_argument("--check-tol", type=float, default=None,
                    help="with --check: fail on missing comparisons, nonfinite gradients, or relative-L2 differences above this; "
                         "without it --check only reports")  # fmt: skip
    ap.add_argument("--json", default=None, help="append one JSON record per S to this file")
    args = ap.parse_args()
    if args.check_tol is not None and not args.check:
        ap.error("--check-tol needs --check")
    failed = []
    scale = args.scale if args.scale is not None else 1.0 / math.sqrt(D)
    if os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "0") not in ("1", "true", "yes", "on"):
        print("note: CUDNN_FRONTEND_ENABLE_FROST_ENGINES is not set; opt-in FROST rows will not be offered", file=sys.stderr)
    dev = torch.cuda.get_device_properties(0)
    print(f"{dev.name}, {dev.multi_processor_count} SMs, cuDNN backend {cudnn.backend_version()}, cudnn-frontend {cudnn.__version__}", flush=True)
    for s in args.s:
        inputs = make_inputs(args.b, args.hq, args.hkv, s, scale, causal=args.causal)
        runs, skipped = [], {}
        for e in args.engines:
            layout = "bshd" if e == "sdpa_bwd_sm100_mxfp8_pair" else "bhsd"
            try:
                runs.append(build(e, args.b, args.hq, args.hkv, s, scale, inputs, layout, amax=args.amax, deterministic=args.deterministic, causal=args.causal))
            except Exception as ex:  # noqa: BLE001 -- report and keep benchmarking the rest
                skipped[e] = str(ex).splitlines()[0][:160]
        if args.check_tol is not None:
            failed.extend(f"S={s} {name}: {reason}" for name, reason in skipped.items())
            if len(runs) < 2:
                failed.append(f"S={s}: --check-tol requires at least two available engines")
        for r in runs:
            r["call"]()
        torch.cuda.synchronize()
        diffs = {}
        if args.check and runs:
            base = [x.float() for x in runs[0]["grads"]]
            for r in runs[1:]:
                rel = [((x.float() - y).norm() / y.norm()).item() for x, y in zip(r["grads"], base)]
                diffs[r["engine"]] = [round(v, 6) if math.isfinite(v) else None for v in rel]  # JSON has no NaN
                if args.check_tol is not None and not all(math.isfinite(v) and v <= args.check_tol for v in rel):
                    failed.append(f"S={s} {r['engine']}: rel-L2 dQ / dK / dV {diffs[r['engine']]} > {args.check_tol}")
        cold = time_cold(runs, args.reps) if runs else {}
        base = "backend" if "backend" in cold else (runs[0]["engine"] if runs else None)
        flags = (("causal", args.causal), ("deterministic", args.deterministic), ("amax", args.amax))
        mode = " ".join(name for name, on in flags if on)
        print(f"S={s} B={args.b} Hq={args.hq} Hkv={args.hkv} {mode}".rstrip(), flush=True)
        for r in runs:
            n = r["engine"]
            sc = cold[base] / cold[n] if base else float("nan")
            extra = f"  rel-L2 vs {runs[0]['engine']} {diffs[n]}" if n in diffs else ""
            print(f"  {n:28s} cold {cold[n]:9.4f} ms ({sc:5.3f}x)  {r['layout'].upper()}  plan {r['plan']}{extra}", flush=True)
        for n, why in skipped.items():
            print(f"  {n:28s} skipped: {why}", flush=True)
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(dict(s=s, b=args.b, hq=args.hq, hkv=args.hkv, amax=args.amax, deterministic=args.deterministic,
                                        causal=args.causal,
                                        cold_ms=cold, diffs=diffs,
                                        plans={r["engine"]: r["plan"] for r in runs}, layouts={r["engine"]: r["layout"] for r in runs},
                                        skipped=skipped, gpu=dev.name), allow_nan=False) + "\n")  # fmt: skip
        del runs, inputs
        torch.cuda.empty_cache()
    if failed:
        print("check failed:\n  " + "\n  ".join(failed), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
