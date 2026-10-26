# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Heuristics regret: is the frontend's default pick as fast as the best cuDNN path for the same case?

Reads attention_training / attention_inference result CSVs and pairs, per identical case (every configuration column,
data type included, except the backend and the measurement), the ``cudnn_auto`` row (default dispatch) with the
``cudnn`` (native backend only) and ``cudnn_oss`` (FROST only) rows of the same CSV. A runner writes one timestamped CSV per invocation, so legs are
only paired within a file: repeated runs in one directory stay separate cases rather than overwriting each other.
``regret = t(cudnn_auto) / min(t(cudnn),
t(cudnn_oss))``: 1.0 means the heuristics picked the faster path; above ``--threshold`` is reported as a misrank.
A case where ``cudnn_auto`` fails while another leg succeeds is reported as a coverage loss.

    python -m benchmark.heuristics_regret <results dir or CSV> [...] [--threshold 1.05] [--top 30] [--fail]
"""

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict

LEGS = ("cudnn", "cudnn_oss", "cudnn_auto")
# Measurement / provenance columns; every other column identifies the case.
NON_KEY = {
    "backend",
    "time_ms",
    "tflops",
    "gbps",
    "sol_pct",
    "max_diff",
    "num_iterations",
    "success",
    "skipped",
    "error_message",
    "gpu_name",
    "cudnn_version",
    "cudnn_backend_version",
    "peak_mma_tflops",
    "backend_detail",
}
SHOW = ("config_name", "model_name", "phase", "profile_pass", "data_type", "kv_cache_dtype", "attn_mask", "batch_size")
SHAPE = ("q_seqlen", "kv_seqlen", "q_tokens", "kv_len", "num_q_heads", "num_kv_heads", "head_dim_qk", "sliding_window_size")


def _csvs(paths):
    for p in paths:
        if os.path.isdir(p):
            yield from sorted(glob.glob(os.path.join(p, "**", "*.csv"), recursive=True))
        else:
            yield p


def _load(paths):
    cases = defaultdict(dict)
    for f in _csvs(paths):
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                leg = row.get("backend")
                if leg not in LEGS:
                    continue
                key = (os.path.relpath(f),) + tuple(sorted((k, v) for k, v in row.items() if k and k not in NON_KEY))
                ok = row.get("success") == "True" and row.get("skipped") != "True"
                try:
                    t = float(row["time_ms"]) if ok else None
                except (TypeError, ValueError):
                    t = None
                if leg in cases[key]:
                    raise SystemExit(f"{f}: two {leg} rows for one case; cannot tell which measurement to compare")
                cases[key][leg] = (t if t and t > 0 else None, row.get("backend_detail") or "")
    return cases


def _label(key):
    d = dict(key[1:])
    head = " ".join(d[k] for k in SHOW if d.get(k))
    shape = " ".join(f"{k}={d[k]}" for k in SHAPE if d.get(k))
    return f"{key[0]}: {head} {shape}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--threshold", type=float, default=1.05)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--fail", action="store_true", help="exit 1 when any misrank or coverage loss is found")
    a = ap.parse_args()

    cases = _load(a.paths)
    regrets, misranks, losses = [], [], []
    for key, legs in cases.items():
        auto = legs.get("cudnn_auto")
        ref = {leg: legs[leg][0] for leg in ("cudnn", "cudnn_oss") if leg in legs and legs[leg][0]}
        if auto is None or not ref:
            continue
        best_leg = min(ref, key=ref.get)
        if auto[0] is None:
            losses.append((key, best_leg, ref[best_leg]))
            continue
        r = auto[0] / ref[best_leg]
        regrets.append(r)
        if r > a.threshold:
            misranks.append((r, key, auto, best_leg, ref[best_leg]))

    if not regrets and not losses:
        print("no case has a cudnn_auto row next to a cudnn / cudnn_oss row")
        return 0
    gm = math.exp(sum(math.log(r) for r in regrets) / len(regrets)) if regrets else float("nan")
    print(f"## Heuristics regret (cudnn_auto vs the faster of cudnn / cudnn_oss, same data type)\n")
    print(f"- compared cases: {len(regrets)}; geomean regret {gm:.3f}; worst {max(regrets, default=float('nan')):.3f}")
    print(f"- misranks (regret > {a.threshold:.2f}): {len(misranks)}; coverage losses (cudnn_auto failed, another leg ran): {len(losses)}\n")
    if misranks:
        print(f"| regret | case | default pick | faster leg (ms) | default (ms) |\n|---|---|---|---|---|")
        for r, key, auto, best_leg, tb in sorted(misranks, key=lambda x: -x[0])[: a.top]:
            print(f"| {r:.2f} | {_label(key)} | {auto[1] or 'n/a'} | {best_leg} {tb:.4f} | {auto[0]:.4f} |")
        print()
    if losses:
        print("| coverage loss | faster leg (ms) |\n|---|---|")
        for key, best_leg, tb in losses[: a.top]:
            print(f"| {_label(key)} | {best_leg} {tb:.4f} |")
    return 1 if a.fail and (misranks or losses) else 0


if __name__ == "__main__":
    sys.exit(main())
