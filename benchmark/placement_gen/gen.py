# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit placement-table entries from a frozen study and score them on its holdout.

    python benchmark/placement_gen/gen.py <study.json> [--emit]

A study JSON names the row, device, dtype, harness adapter, head pair and the train / holdout JSONL files written by the
study drivers (one record per (case, leg) with ``ms``, ``plan``, ``case``). Each case is turned into graph facts and
classified with ``cudnn.sdpa.placement_table.class_key`` -- the same code the runtime uses -- so a fitted class is
exactly the class a graph looks up.

Per class the fit is monotone in ``log2_work``: ``always`` (FROST never loses beyond the margin), ``never`` (it never
wins), or a threshold halfway between the last training loss and the first win past it, in the direction that keeps more
wins. A class whose
wins and losses interleave so that neither direction keeps a win is reported as NON-MONOTONE and gets no entry: split
the class (add a key field) rather than fitting around it.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from types import SimpleNamespace

import cudnn
from cudnn.sdpa import placement_table as pt

_DTYPES = {"bf16": cudnn.data_type.BFLOAT16, "f16": cudnn.data_type.HALF}


def _facts_inference(c, s):
    """Inference harness (generation phase): q tokens per request, dense or paged KV (case field, else the study default)."""
    return SimpleNamespace(
        b=c["b"],
        h_q=c["hq"],
        h_kv=c.get("hkv", 1),
        s_q=c["q"],
        s_kv=c["kv"],
        d_qk=s["d_qk"],
        d_v=s["d_v"],
        dtype=_DTYPES[s["dtype"]],
        causal=True,
        bottom_right=True,
        window_left=c.get("window"),
        has_sink=bool(c.get("sink")),
        has_paged_kv=c.get("layout", s.get("layout", "dense")) == "paged",
        thd=False,
        max_total_seq_len_q=None,
        device_cc=tuple(s["device_cc"]),
        has_bias=False,
        has_dropout=False,
        right_band_widening=False,
    )


def _facts_training(c, s):
    """Training harness: s_q == s_kv == s; THD at fill < 1 declares its live total (ragged)."""
    thd = c["layout"] == "thd"
    total = c["b"] * c["s"] if c.get("fill", 1.0) >= 1.0 else int(c["b"] * c["s"] * (1 + c["fill"]) / 2)
    return SimpleNamespace(
        b=c["b"],
        h_q=c["hq"],
        h_kv=c["hkv"],
        s_q=c["s"],
        s_kv=c["s"],
        d_qk=c.get("d_qk", s.get("d_qk")),
        d_v=c.get("d_v", s.get("d_v")),
        dtype=_DTYPES[s["dtype"]],
        causal=c["mask"] != "no_mask",
        bottom_right=False,
        window_left=c.get("window"),
        has_sink=False,
        has_paged_kv=False,
        thd=thd,
        max_total_seq_len_q=total if thd else None,
        device_cc=tuple(s["device_cc"]),
        has_bias=False,
        has_dropout=False,
        right_band_widening=False,
    )


ADAPTERS = {"inference": _facts_inference, "training": _facts_training}


def load(path):
    rows = collections.defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if r.get("ms") is not None:
            rows[r["idx"]][r["leg"]] = r
    return [d for d in rows.values() if "cudnn" in d and "cudnn_oss" in d]


def fit(points, margin):
    """points: [(log2_work, frost/backend ratio)] of one class -> (direction, threshold) or ("non-monotone", None)."""
    wins = [w for w, r in points if r < 1 - margin]
    losses = [w for w, r in points if r > 1 + margin]
    if not wins:
        return "never", None
    if not losses:
        return "always", None
    top, bottom = max(losses), min(losses)
    above, below = [w for w in wins if w > top], [w for w in wins if w < bottom]
    if not above and not below:
        return "non-monotone", None
    # The threshold sits halfway (in log2 work) between the last loss and the first win past it, rounded away from the
    # loss so the emitted value never re-admits it.
    if len(above) >= len(below):
        return "ge", math.ceil((top + min(above)) / 2 * 1000) / 1000
    return "le", math.floor((bottom + max(below)) / 2 * 1000) / 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--margin", type=float, default=0.03, help="|ratio - 1| within this is neutral")
    ap.add_argument("--emit", action="store_true", help="print the TABLE block for cudnn/sdpa/placement_table.py")
    args = ap.parse_args()
    s = json.load(open(args.study))
    to_facts = ADAPTERS[s["adapter"]]
    train, hold = load(s["train"]), load(s["holdout"])

    by = collections.defaultdict(list)
    for d in train:
        f = to_facts(d["cudnn"]["case"], s)
        key = pt.class_key(s["row"], f)
        if key is not None:
            by[key].append((pt.log2_work(f), d["cudnn_oss"]["ms"] / d["cudnn"]["ms"]))
    table = {k: fit(v, args.margin) for k, v in by.items()}

    def pick(case):
        """True / False for a class with an entry; None where the runtime falls back to the hand-written rules."""
        f = to_facts(case, s)
        key = pt.class_key(s["row"], f)
        direction, t = table.get(key, (None, None))
        if direction in (None, "non-monotone"):
            return None
        if direction in ("never", "always"):
            return direction == "always"
        w = pt.log2_work(f)
        return w >= t if direction == "ge" else w <= t

    g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
    print(f"study {s['id']}: {len(train)} train / {len(hold)} holdout cases, {len(table)} classes")
    for k, (direction, t) in sorted(table.items(), key=lambda kv: str(kv[0])):
        n = len(by[k])
        print(f"  {k[3:]}  n={n:3d}  {direction}" + ("" if t is None else f" {t:.2f}"))
    # A case whose class has no entry is scored by what the runtime does there -- the hand-written fallback, i.e. the
    # measured default leg (``cudnn_auto``); without that leg it is left unscored and counted.
    for label, rows in (("train", train), ("holdout", hold)):
        for rule, chooser in (("generated", lambda d: pick(d["cudnn"]["case"])), ("measured default", None), ("backend only", lambda d: False)):
            regs, unscored = [], 0
            for d in rows:
                b, f = d["cudnn"]["ms"], d["cudnn_oss"]["ms"]
                choice = None if chooser is None else chooser(d)
                if choice is None:
                    if "cudnn_auto" not in d:
                        unscored += 1
                        continue
                    regs.append(d["cudnn_auto"]["ms"] / min(b, f))
                else:
                    regs.append((f if choice else b) / min(b, f))
            why = "no measured default leg" if chooser is None else "no entry and no measured default leg"
            extra = f"  ({unscored} unscored: {why})" if unscored else ""
            if regs:
                print(f"  {label:8s} {rule:16s} regret geo {g(regs):.3f} max {max(regs):.3f} >1.05 {sum(r > 1.05 for r in regs)}/{len(regs)}{extra}")
            elif rows:
                print(f"  {label:8s} {rule:16s} NOT SCORED: 0/{len(rows)} cases{extra}")
    bad = [k for k, (direction, _) in table.items() if direction == "non-monotone"]
    if bad:
        print("NON-MONOTONE classes (split the key; no entry emitted):", *bad, sep="\n  ")
    if args.emit:
        print(f"\n    # study {s['id']}")
        for k, (direction, t) in sorted(table.items(), key=lambda kv: str(kv[0])):
            if direction != "non-monotone":
                print(f"    {k!r}: Entry({direction!r}, {t!r}, {s['id']!r}),")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
