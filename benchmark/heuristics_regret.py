# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Heuristics regret: is the frontend's default pick as fast as the best cuDNN path for the same case?

Reads attention_training / attention_inference result CSVs and pairs, per identical case (every configuration column,
data type included, except the backend and the measurement), the ``cudnn_auto`` row (default dispatch) with the
``cudnn`` (native backend only) and ``cudnn_oss`` (FROST only) rows of the same CSV. A runner writes one timestamped CSV per invocation, so legs are
only paired within a file: repeated runs in one directory stay separate cases rather than overwriting each other.
``regret = t(cudnn_auto) / min(t(cudnn),
t(cudnn_oss))``: 1.0 means the heuristics picked the faster path; above ``--threshold`` is reported as a misrank.
A case where ``cudnn_auto`` fails while another leg succeeds is reported as a coverage loss. When ``cudnn_auto`` ran
the same plan as the faster leg (``backend_detail``), the case is not a misrank whatever the timings say: their
spread is run-to-run noise, reported as the noise floor the threshold has to clear.

    python -m benchmark.heuristics_regret <results dir or CSV> [...] [--threshold 1.05] [--top 30] [--fail]

``--baseline <an earlier run's dir or CSV> [...]`` adds the run-to-run view: each case's ``cudnn_auto`` time against the
same case there (matched on the configuration columns both runs record). A case slower by more than ``--regress`` is
listed under "plan changed" when its default plan differs between the runs (a dispatch change) and under "same plan"
otherwise (a kernel, backend or measurement change); cases whose default stopped running are listed apart.
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
SHOW = ("config_name", "model_name", "phase", "profile_pass", "data_type", "kv_cache_dtype", "attn_mask", "batch_size", "layout", "kv_layout")
SHAPE = ("q_seqlen", "kv_seqlen", "q_tokens", "kv_len", "num_q_heads", "num_kv_heads", "head_dim_qk", "sliding_window_size", "thd_min_fill", "kv_min_fill")


def _csvs(paths):
    for p in paths:
        if os.path.isdir(p):
            yield from sorted(glob.glob(os.path.join(p, "**", "*.csv"), recursive=True))
        else:
            yield p


def _load(paths):
    cases = defaultdict(dict)
    for f in _csvs(paths):
        found = defaultdict(dict)
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
                if leg in found[key]:
                    print(f"skipping {f}: two {leg} rows for one case; cannot tell which measurement to compare", file=sys.stderr)
                    break
                found[key][leg] = (t if t and t > 0 else None, row.get("backend_detail") or "")
            else:
                cases.update(found)
    return cases


def _plan(detail):
    """The plan name in a backend_detail ("cudnn_auto plan=X" or a bare name); None when unrecorded or unknown."""
    detail = (detail or "").strip()
    plan = detail.split("plan=", 1)[1].strip() if "plan=" in detail else detail
    return None if plan in ("", "unknown") else plan


def _label(key):
    d = dict(key[1:])
    head = " ".join(d[k] for k in SHOW if d.get(k))
    shape = " ".join(f"{k}={d[k]}" for k in SHAPE if d.get(k))
    return f"{key[0]}: {head} {shape}"


def _defaults(paths):
    """{configuration columns: (time, detail)} of every cudnn_auto row; a case two files both record is dropped."""
    out, seen = {}, set()
    for key, legs in _load(paths).items():
        if "cudnn_auto" not in legs:
            continue
        case = key[1:]
        if case in seen:
            out.pop(case, None)
            continue
        seen.add(case)
        out[case] = legs["cudnn_auto"]
    return out


def _report_baseline(paths, baseline, regress, top):
    now, then = _defaults(paths), _defaults(baseline)
    cols = {k for case in now for k, _ in case} & {k for case in then for k, _ in case}
    project = lambda case: tuple((k, v) for k, v in case if k in cols)
    then = {project(c): v for c, v in then.items()}
    ratios, changed, same, lost = [], [], [], []
    for case, (t, detail) in now.items():
        prev = then.get(project(case))
        if prev is None:
            continue
        if t is None:
            if prev[0] is not None:
                lost.append((case, prev))
            continue
        if prev[0] is None:
            continue
        r = t / prev[0]
        ratios.append(r)
        if r > regress:
            (changed if _plan(detail) != _plan(prev[1]) else same).append((r, case, prev, (t, detail)))
    if not ratios and not lost:
        print("## Run to run: no cudnn_auto case matched the baseline\n")
        return False
    gm = math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else float("nan")
    print("## Run to run (cudnn_auto time now / baseline, same case)\n")
    print(f"- matched cases: {len(ratios)}; geomean {gm:.3f}; slower than {regress:.2f}x: {len(changed)} with a new default plan, {len(same)} on the same plan")
    print(f"- default stopped running: {len(lost)}\n")
    for title, rows in (("new default plan", changed), ("same default plan", same)):
        if rows:
            print(f"| now/baseline ({title}) | case | baseline plan (ms) | plan now (ms) |\n|---|---|---|---|")
            for r, case, prev, cur in sorted(rows, key=lambda x: -x[0])[:top]:
                print(f"| {r:.2f} | {_label(('',) + case)} | {_plan(prev[1])} {prev[0]:.4f} | {_plan(cur[1])} {cur[0]:.4f} |")
            print()
    if lost:
        print("| default stopped running | baseline plan (ms) |\n|---|---|")
        for case, prev in lost[:top]:
            print(f"| {_label(('',) + case)} | {_plan(prev[1])} {prev[0]:.4f} |")
        print()
    return bool(changed or lost)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--threshold", type=float, default=1.05)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--fail", action="store_true", help="exit 1 when any misrank or coverage loss is found (or, with --baseline, a slower new default plan)")
    ap.add_argument("--baseline", nargs="+", default=None, help="an earlier run's results dir or CSV: report cudnn_auto cases slower than there")
    ap.add_argument("--regress", type=float, default=1.2, help="--baseline: now/baseline ratio reported as slower")
    a = ap.parse_args()

    regressed = _report_baseline(a.paths, a.baseline, a.regress, a.top) if a.baseline else False

    cases = _load(a.paths)
    regrets, misranks, losses, same_plan = [], [], [], []
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
        if _plan(auto[1]) is not None and _plan(auto[1]) == _plan(legs[best_leg][1]):
            same_plan.append(max(r, 1 / r))
            regrets.append(1.0)
            continue
        regrets.append(r)
        if r > a.threshold:
            misranks.append((r, key, auto, best_leg, ref[best_leg]))

    if not regrets and not losses:
        print("no case has a cudnn_auto row next to a cudnn / cudnn_oss row")
        return 1 if a.fail and regressed else 0
    gm = math.exp(sum(math.log(r) for r in regrets) / len(regrets)) if regrets else float("nan")
    print(f"## Heuristics regret (cudnn_auto vs the faster of cudnn / cudnn_oss, same data type)\n")
    print(f"- compared cases: {len(regrets)}; geomean regret {gm:.3f}; worst {max(regrets, default=float('nan')):.3f}")
    print(f"- misranks (regret > {a.threshold:.2f}): {len(misranks)}; coverage losses (cudnn_auto failed, another leg ran): {len(losses)}")
    if same_plan:
        print(f"- same plan as the faster leg (regret 1.0): {len(same_plan)}; their timing spread, the noise floor: max {max(same_plan):.3f}")
    print()
    if misranks:
        print(f"| regret | case | default pick | faster leg (ms) | default (ms) |\n|---|---|---|---|---|")
        for r, key, auto, best_leg, tb in sorted(misranks, key=lambda x: -x[0])[: a.top]:
            print(f"| {r:.2f} | {_label(key)} | {auto[1] or 'n/a'} | {best_leg} {tb:.4f} | {auto[0]:.4f} |")
        print()
    if losses:
        print("| coverage loss | faster leg (ms) |\n|---|---|")
        for key, best_leg, tb in losses[: a.top]:
            print(f"| {_label(key)} | {best_leg} {tb:.4f} |")
    return 1 if a.fail and (misranks or losses or regressed) else 0


if __name__ == "__main__":
    sys.exit(main())
