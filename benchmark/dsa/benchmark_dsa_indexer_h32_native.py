#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark native-H32 indexer backward against the padded default backend.

For aligned top-k, both backends receive the same tensors.  For CSA's logical
2049--2051 extents, the default control receives a pre-padded 2176-slot view;
the native backend receives the compact logical tensors.  The pre-padded
control excludes adapter cost and is therefore the conservative kernel-path
baseline.  ``eager_pad_default_us`` additionally measures caller-side padding.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch
import torch.nn.functional as F

from benchmark_dsa_indexer_h32 import _make_case, _outputs, _time_once
from cudnn import DSA


def _call(backend, q, w, k, attn, predict, indices, grad_loss, outputs):
    return DSA.indexer_backward_wrapper(
        q,
        w,
        k,
        attn,
        predict,
        indices,
        grad_loss,
        sm_scale=1.0,
        loss_coeff=float(q.shape[0] * q.shape[1]),
        block_I=128,
        topk_indices_global=False,
        d_index_q=outputs[0],
        d_weights=outputs[1],
        d_index_k=outputs[2],
        backend=backend,
        deterministic=False,
        workspace=None,
    )


def _diff_stats(lhs, rhs):
    stats = []
    for actual, expected in zip(lhs, rhs):
        actual_f = actual.float()
        expected_f = expected.float()
        delta = (actual_f - expected_f).abs()
        stats.append(
            {
                "max_abs": delta.max().item(),
                "mean_abs": delta.mean().item(),
                "mismatch_rtol_0.03_atol_0.03_pct": ((delta > 0.03 + 0.03 * expected_f.abs()).float().mean() * 100).item(),
                "finite": bool(torch.isfinite(actual_f).all() and torch.isfinite(expected_f).all()),
            }
        )
    return stats


def _summary(values):
    ordered = sorted(values)
    result = {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[len(ordered) // 10],
        "p90_us": ordered[(9 * len(ordered)) // 10],
        "min_us": ordered[0],
    }
    if not all(math.isfinite(value) and value > 0.0 for value in result.values()):
        raise RuntimeError(f"invalid benchmark timings: {result}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=120)
    parser.add_argument("--include-eager-pad", action="store_true")
    args = parser.parse_args()
    if not 128 <= args.topk <= 2176:
        parser.error("--topk must be in the native-H32 envelope [128, 2176]")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats must be positive")

    padded_topk = ((args.topk + 127) // 128) * 128
    q, w, k, attn_logical_base, predict_logical, indices_logical = _make_case(1, 512, 4096, args.topk, 0, False)
    pad = padded_topk - args.topk
    attn_padded_base = F.pad(attn_logical_base, (0, pad))
    predict_padded = F.pad(predict_logical, (0, pad))
    indices_padded = F.pad(indices_logical, (0, pad), value=-1)

    grad_loss = torch.ones((), device="cuda", dtype=torch.float32)
    l2_flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    attn_native = attn_logical_base.clone()
    attn_default = attn_padded_base.clone()
    attn_eager = attn_logical_base.clone()
    out_native = _outputs(q, w, k)
    out_default = _outputs(q, w, k)
    out_eager = _outputs(q, w, k)

    native = lambda: _call("sm100_v2", q, w, k, attn_native, predict_logical, indices_logical, grad_loss, out_native)
    padded_default = lambda: _call("default", q, w, k, attn_default, predict_padded, indices_padded, grad_loss, out_default)

    def eager_pad_default():
        return _call(
            "default",
            q,
            w,
            k,
            F.pad(attn_eager, (0, pad)),
            F.pad(predict_logical, (0, pad)),
            F.pad(indices_logical, (0, pad), value=-1),
            grad_loss,
            out_eager,
        )

    padded_default()
    native()
    torch.cuda.synchronize()
    correctness = _diff_stats(out_native, out_default)
    if not all(item["finite"] and item["mismatch_rtol_0.03_atol_0.03_pct"] == 0.0 for item in correctness):
        raise RuntimeError(f"native-H32 correctness check failed: {correctness}")

    cases = [
        ("padded_default_us", padded_default, attn_default, attn_padded_base),
        ("native_h32_us", native, attn_native, attn_logical_base),
    ]
    if args.include_eager_pad and pad:
        cases.append(("eager_pad_default_us", eager_pad_default, attn_eager, attn_logical_base))

    for _ in range(args.warmup):
        for _, fn, attn, base in cases:
            attn.copy_(base)
            fn()
    torch.cuda.synchronize()

    timings = {name: [] for name, _, _, _ in cases}
    for repeat in range(args.repeats):
        shift = repeat % len(cases)
        for name, fn, attn, base in cases[shift:] + cases[:shift]:
            timings[name].append(_time_once(name, fn, attn, base, l2_flush))

    summaries = {name: _summary(values) for name, values in timings.items()}
    baseline = summaries["padded_default_us"]["median_us"]
    native_us = summaries["native_h32_us"]["median_us"]
    print(
        json.dumps(
            {
                "sm": torch.cuda.get_device_capability(),
                "logical_shape": [1, 512, 4096, 32, 128, args.topk],
                "default_physical_topk": padded_topk,
                "timings": summaries,
                "native_speedup_vs_padded_default": baseline / native_us,
                "native_latency_change_pct": (native_us / baseline - 1.0) * 100.0,
                "correctness_native_vs_padded_default": correctness,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
