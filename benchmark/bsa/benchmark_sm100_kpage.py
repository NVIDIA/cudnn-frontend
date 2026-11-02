# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100/SM103 blk128 BSA with and without kpage on layouts from open-source
video sparse-attention systems (see kpage_layouts.py).

    python benchmark_sm100_kpage.py                      # the eight layouts of kpage_layouts.BENCHMARK_LAYOUTS
    python benchmark_sm100_kpage.py --layouts ssta:1280x720x121:16 vsa:832x480x81:64:0.875:12 --check

Timings are CUDA-event medians of the attention call only; the kpage layout
is built once outside the timed region (its build time is reported
separately).  The video layouts pad every 3-D tile on both the
query and the key side, so the layout is built with q_block_sizes (padding
query rows are skipped and written as zero); the base path has no query-side
padding and computes those rows.
"""

import argparse
import importlib.metadata
import json
import math
import statistics
import time

import torch
from cudnn import BSA

import kpage_layouts

BLOCK = 128


def event_ms(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def warm_up_gpu(seconds=1.0):
    """Bring the clocks up before the first timed layout (a cold GPU makes the first rows a few percent slow)."""
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        a = a @ a
        torch.cuda.synchronize()


def _valid_rows(sizes):
    return (torch.arange(BLOCK, device="cuda").view(1, BLOCK) < sizes.view(-1, 1)).reshape(-1)


def make_inputs(layout, gen):
    """q/do over the query partition, k/v over the KV partition; padding rows take no part (any finite value would do)."""
    q_valid, kv_valid = _valid_rows(layout.q_sizes), _valid_rows(layout.block_sizes)
    tensors = []
    for seqlen, valid in ((layout.T, q_valid), (layout.seqlen_k, kv_valid), (layout.seqlen_k, kv_valid), (layout.T, q_valid)):
        t = torch.randn((1, layout.H, seqlen, 128), generator=gen).to(torch.bfloat16).cuda()
        t[0, :, ~valid] = 0
        tensors.append(t)
    return tensors, q_valid


def sampled_reference(layout, q, k, v, do, blocks, head):
    """fp32 O and dQ for a few query blocks of one head (full rows of K/V they attend)."""
    outs, dqs, rows = [], [], []
    bs, qs = layout.block_sizes.tolist(), layout.q_sizes.tolist()
    for qb in blocks:
        cnt = int(layout.q2k_cnt[0, head, qb])
        kv_rows = torch.cat([torch.arange(j * BLOCK, j * BLOCK + bs[j], device="cuda") for j in layout.q2k_idx[0, head, qb, :cnt].tolist()])
        q_rows = torch.arange(qb * BLOCK, qb * BLOCK + qs[qb], device="cuda")
        q_blk = q[0, head, q_rows].float().detach().requires_grad_()
        s = q_blk @ k[0, head, kv_rows].float().T / math.sqrt(128)
        o = torch.softmax(s, -1) @ v[0, head, kv_rows].float()
        o.backward(do[0, head, q_rows].float())
        outs.append(o.detach())
        dqs.append(q_blk.grad)
        rows.append(q_rows)
    return torch.cat(outs), torch.cat(dqs), torch.cat(rows)


def run(layout, spec, args, gen):
    layout = layout.to("cuda")
    stats = layout.stats()
    (q, k, v, do), valid = make_inputs(layout, gen)
    idx, cnt, bs = layout.q2k_idx, layout.q2k_cnt, layout.block_sizes
    with_bwd = not args.no_bwd
    rec = dict(
        layout=layout.name,
        spec=spec,
        desc=layout.desc,
        heads=layout.H,
        tokens=layout.T,  # padded length
        **{key: stats[key] for key in ("H", "T", "seqlen_k", "nQ", "nB", "pad_frac_blk128", "q_pad_frac_blk128", "partial_block_frac", "useful_tflop_fwd")},
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    page_layout = BSA.kpage_prepare_layout(idx, None, bs, cnt, q_block_sizes=layout.q_sizes)
    torch.cuda.synchronize()
    rec["kpage_layout_build_ms"] = (time.perf_counter() - t0) * 1e3

    base_fwd = lambda: BSA.block_sparse_attention_forward(q, k, v, idx, None, bs, cnt)
    kpage_fwd = lambda: BSA.block_sparse_attention_forward(q, k, v, idx, None, bs, cnt, kpage=page_layout)
    rec["bsa_fwd_ms"] = event_ms(base_fwd, args.iters, args.warmup)
    rec["kpage_fwd_ms"] = event_ms(kpage_fwd, args.iters, args.warmup)
    rec["fwd_speedup"] = rec["bsa_fwd_ms"] / rec["kpage_fwd_ms"]

    if with_bwd:
        o, lse = kpage_fwd()
        kpage_bwd = lambda: BSA.block_sparse_attention_backward(do, q, k, v, o, lse, idx, None, bs, cnt, kpage=page_layout)
        rec["kpage_bwd_ms"] = event_ms(kpage_bwd, args.iters, args.warmup)
        ob, lseb = base_fwd()
        base_bwd = lambda: BSA.block_sparse_attention_backward(do, q, k, v, ob, lseb, idx, None, bs, cnt)
        rec["bsa_bwd_ms"] = event_ms(base_bwd, args.iters, args.warmup)
        rec["fwd_bwd_speedup"] = (rec["bsa_fwd_ms"] + rec["bsa_bwd_ms"]) / (rec["kpage_fwd_ms"] + rec["kpage_bwd_ms"])

    if args.check:
        o_base = base_fwd()["o_tensor"]
        o_kp, lse_kp = kpage_fwd()
        blocks = torch.linspace(0, layout.nQ - 1, steps=min(4, layout.nQ)).round().long().tolist()
        o_ref, dq_ref, rows = sampled_reference(layout, q, k, v, do, blocks, head=0)
        rec["check"] = dict(
            kpage_o_max_abs_err=(o_kp[0, 0, rows].float() - o_ref).abs().max().item(),
            bsa_o_max_abs_err=(o_base[0, 0, rows].float() - o_ref).abs().max().item(),
            kpage_vs_bsa_o_max_abs_diff=(o_kp[0, :, valid].float() - o_base[0, :, valid].float()).abs().max().item(),
            kpage_pad_rows_zero=bool((o_kp[0, :, ~valid] == 0).all()),
        )
        if with_bwd:
            dq = BSA.block_sparse_attention_backward(do, q, k, v, o_kp, lse_kp, idx, None, bs, cnt, kpage=page_layout)["dq_tensor"]
            rec["check"]["kpage_dq_max_abs_err"] = (dq[0, 0, rows].float() - dq_ref).abs().max().item()
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layouts", nargs="*", default=kpage_layouts.BENCHMARK_LAYOUTS)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--no-bwd", action="store_true")
    parser.add_argument("--check", action="store_true", help="compare O and dQ of sampled query blocks against an fp32 reference")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="/tmp/bsa-sm100-kpage.json")
    args = parser.parse_args()
    major, _ = torch.cuda.get_device_capability()
    assert major == 10, "kpage requires an SM100/SM103 GPU"

    report = dict(
        gpu=torch.cuda.get_device_name(),
        sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
        torch=torch.__version__,
        cutlass_dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
        cuda=torch.version.cuda,
        iters=args.iters,
        warmup=args.warmup,
        results=[],
    )
    gen = torch.Generator().manual_seed(args.seed)
    warm_up_gpu()
    header = f"{'layout':32s} {'tokens':>7s} {'H':>3s} {'pad% q/k':>11s} {'part%':>6s} {'BSA fwd':>9s} {'KPage fwd':>10s} {'x':>6s}"
    if not args.no_bwd:
        header += f" {'BSA f+b':>10s} {'KPage f+b':>10s} {'x':>6s}"
    print(header)

    def show(rec, indent=""):
        pad = f"{100 * rec['q_pad_frac_blk128']:4.1f}/{100 * rec['pad_frac_blk128']:4.1f}"
        line = (
            f"{indent + rec['layout']:32s} {rec['tokens']:7.0f} {rec['heads']:3d} {pad:>11s} {100 * rec['partial_block_frac']:6.1f} "
            f"{rec['bsa_fwd_ms']:9.2f} {rec['kpage_fwd_ms']:10.2f} {rec['fwd_speedup']:6.2f}"
        )
        if not args.no_bwd:
            if "kpage_bwd_ms" in rec:
                line += f" {rec['bsa_fwd_ms'] + rec['bsa_bwd_ms']:10.2f} {rec['kpage_fwd_ms'] + rec['kpage_bwd_ms']:10.2f} {rec['fwd_bwd_speedup']:6.2f}"
            else:
                line += f" {'-':>10s} {'-':>10s} {'-':>6s}"
        print(line, flush=True)
        if args.check:
            print(indent + "    check:", json.dumps(rec["check"]), flush=True)

    for spec in args.layouts:
        rec = run(kpage_layouts.build(spec), spec, args, gen)
        report["results"].append(rec)
        show(rec)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
