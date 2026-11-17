# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit placement-table entries from a frozen study and score them on its holdout.

    python benchmark/placement_gen/gen.py <study.json> [--emit]
    python benchmark/placement_gen/gen.py <study.json> --constrained [--two-feature] [--rules-out RULES.json] [--emit]
    python benchmark/placement_gen/gen.py <study.json> --score RULES.json <holdout.jsonl ...>

A study JSON names the row, device, dtype, harness adapter, head pair and the train / holdout JSONL files written by the
study drivers (one record per (case, leg) with ``ms``, ``plan``, ``case``). Each case is turned into graph facts and
classified with ``cudnn.sdpa.placement_table.class_key`` -- the same code the runtime uses -- so a fitted class is
exactly the class a graph looks up.

Per class, every scalar in ``placement_table.FEATURES`` is fit monotonically: ``always`` (FROST never loses beyond the
margin), ``never`` (it never wins), or a threshold halfway between the last training loss and the first win past it, in
the direction that keeps more wins. The feature with the lowest training regret is kept and recorded in the entry; the
frozen holdout, run once at the frozen head, remains the independent qualification. A class whose
wins and losses interleave so that neither direction keeps a win is reported as NON-MONOTONE and gets no entry: split
the class (add a key field) rather than fitting around it.

``--constrained`` fits under the holdout criterion instead (AGENTS.md S12). It needs the measured default leg (``cudnn_auto``).
A rule's changed cells are those whose verdict differs from the default; it is feasible when they run at geomean <= 1.0 and
worst <= 1.05 against that leg. The feasible rule with the lowest training regret wins (ties: fewer changed cells, then
``cost``). With ``--two-feature``, a single-feature rule may also carry a second condition, ``ge`` on KV or query tokens
(``Entry.also``); a single-feature rule wins a tie. No feasible rule means no entry. ``--score`` scores frozen rules on a
holdout once: per class, changed cells at geomean <= 1.0 and worst <= 1.10; a twin (on / off the KV tile) passes when both
sides pass with at least one changed cell.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import math
import sys

import cudnn
from cudnn.sdpa import placement_table as pt
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

_DTYPES = {"bf16": cudnn.data_type.BFLOAT16, "f16": cudnn.data_type.HALF}


def _facts_inference(c, s):
    """Inference harness: q tokens per request, dense or paged KV, head pair and mask (case fields, else the study default).
    ``mask`` "none" is a non-causal context; any other value is the harness's causal band, bottom-right (a square reads the
    same from either corner)."""
    paged = c.get("layout", s.get("layout", "dense")) == "paged"
    causal = c.get("mask", "causal") != "none"
    return SdpaGraphFacts(
        b=c["b"],
        h_q=c["hq"],
        h_kv=c.get("hkv", 1),
        s_q=c["q"],
        s_kv=c["kv"],
        d_qk=c.get("d_qk", s.get("d_qk")),
        d_v=c.get("d_v", s.get("d_v")),
        dtype=_DTYPES[s["dtype"]],
        causal=causal,
        bottom_right=causal,
        window_left=c.get("window"),
        has_sink=bool(c.get("sink")),
        has_paged_kv=paged,
        page_size=c.get("page", 64) if paged else 0,
        padded=paged,
        thd=bool(c.get("thd", False)),
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
    return SdpaGraphFacts(
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
        deterministic=bool(c.get("det", False)),
    )


def _facts_recorded(c, s):
    """The graph facts the study driver read from the FE graph itself (``case["facts"]``, e.g. FlashInfer's own
    declaration): nothing is reconstructed from case fields. dtype names and tuples were stored as their repr."""
    kw = {}
    for name, v in c["facts"].items():
        if isinstance(v, str) and v.startswith("data_type."):
            v = getattr(cudnn.data_type, v.split(".", 1)[1])
        elif isinstance(v, str) and v.startswith("("):
            v = ast.literal_eval(v)
        kw[name] = v
    return SdpaGraphFacts(**kw)


ADAPTERS = {"inference": _facts_inference, "training": _facts_training, "recorded": _facts_recorded}


def load(paths):
    """Cases with both the backend and the FROST leg timed. ``paths`` is one JSONL or a list; records join on (file, idx), so
    splits from separate driver runs (each counting idx from 0) never overwrite each other."""
    rows = collections.defaultdict(dict)
    for path in [paths] if isinstance(paths, str) else paths:
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("ms") is not None:
                    rows[(path, r["idx"])][r["leg"]] = r
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
        split = "ge", math.ceil((top + min(above)) / 2 * 1000) / 1000
    else:
        split = "le", math.floor((bottom + max(below)) / 2 * 1000) / 1000

    # A threshold also gives up the wins on its far side. "always" / "never" replace it when every loss they accept is
    # marginal (within twice the neutral margin) and their worst cell, then their geomean, is lower.
    def regret(rule):
        direction, t = rule
        regs = [
            max(r, 1.0) if direction == "always" or (direction == "ge" and w >= t) or (direction == "le" and w <= t) else max(1 / r, 1.0) for w, r in points
        ]
        return max(regs), sum(map(math.log, regs))

    flat = [rule for rule in (("always", None), ("never", None)) if regret(rule)[0] <= 1 + 2 * margin]
    return min([split, *flat], key=regret)


def _cells(paths, s):
    """Per class key, the cells with all three legs timed: facts, backend / FROST / measured-default ms, and whether the default
    ran FROST."""
    to_facts = ADAPTERS[s["adapter"]]
    by = collections.defaultdict(list)
    for d in load(paths):
        if "cudnn_auto" not in d:
            continue
        f = to_facts(d["cudnn"]["case"], s)
        key = pt.class_key(s["row"], f)
        if key is not None:
            x = dict(be=d["cudnn"]["ms"], fr=d["cudnn_oss"]["ms"], au=d["cudnn_auto"]["ms"], dflt="sdpa_fwd" in (d["cudnn_auto"].get("plan") or ""))
            x.update({k: round(fn(f), 6) for k, fn in pt.FEATURES.items()})
            by[key].append(x)
    return by


def _geo(v):
    return math.exp(sum(map(math.log, v)) / len(v)) if v else 1.0


def _leads(x, rule):
    """rule: (direction, threshold, feature[, direction, threshold, feature]); the second triple must also hold."""
    if rule[0] in ("always", "never"):
        return rule[0] == "always"
    hold = lambda d, t, feat: x[feat] >= t if d == "ge" else x[feat] <= t  # noqa: E731
    return hold(*rule[:3]) and (len(rule) < 6 or hold(*rule[3:]))


def _changed(xs, rule):
    regret = _geo([(x["fr"] if _leads(x, rule) else x["be"]) / min(x["be"], x["fr"]) for x in xs])
    return regret, [(x["fr"] if _leads(x, rule) else x["be"]) / x["au"] for x in xs if _leads(x, rule) != x["dflt"]]


def constrained_fit(xs, two_feature=False):
    """The lowest-train-regret feasible rule of one class (module docstring), or None."""

    def cuts(feat):
        v = sorted({x[feat] for x in xs})
        return [round((a + b) / 2, 3) for a, b in zip(v, v[1:])]

    singles = [("always", None, None), ("never", None, None)]
    singles += [(d, t, feat) for feat in pt.FEATURES for t in cuts(feat) for d in ("le", "ge")]
    pairs = []
    if two_feature:
        pairs = [r + ("ge", t2, f2) for r in singles[2:] for f2 in ("kv", "q") if f2 != r[2] for t2 in cuts(f2)]
    best = None
    for i, rule in enumerate(singles + pairs):
        regret, ch = _changed(xs, rule)
        if ch and (_geo(ch) > 1.0 or max(ch) > 1.05):
            continue
        rank = (round(regret, 6), i >= len(singles), len(ch), rule[2] != "cost")
        if best is None or rank < best[0]:
            best = (rank, rule)
    return None if best is None else best[1]


def _entry(key, rule, study):
    direction, t, feat = rule[:3]
    args = [repr(direction), repr(t), repr(study)]
    if feat not in (None, "cost") or len(rule) == 6:
        args.append(repr(feat or "cost"))
    if len(rule) == 6:
        args.append(repr(tuple(rule[3:])))
    return f"    {key!r}: Entry({', '.join(args)}),"


def constrained_main(args, s):
    if args.score:
        rules = {tuple(tuple(v) if isinstance(v, list) else v for v in json.loads(k)): tuple(r) for k, r in json.load(open(args.score)).items()}
        ok = {}
        for key, xs in sorted(_cells(args.paths, s).items(), key=lambda kv: str(kv[0])):
            rule = rules.get(key[3:])
            if rule is None:
                continue
            _, ch = _changed(xs, rule)
            ok[key[3:]] = bool(ch) and _geo(ch) <= 1.0 and max(ch) <= 1.10
            verdict = "PASS" if ok[key[3:]] else ("VACUOUS" if not ch else "FAIL")
            print(f"  {key[3:]}  n={len(xs)}  {rule}  changed {len(ch)} geo {_geo(ch):.3f} worst {max(ch, default=1.0):.3f}  {verdict}")
        twins = collections.defaultdict(list)
        for tail, v in ok.items():
            twins[tail[:-1]].append(v)
        for tail, v in sorted(twins.items(), key=str):
            print(f"  twin {tail}  {'PASS' if all(v) and len(v) == 2 else 'FAIL'}")
        return 0
    rules = {}
    for key, xs in sorted(_cells(s["train"], s).items(), key=lambda kv: str(kv[0])):
        rule = constrained_fit(xs, args.two_feature)
        default = _geo([x["au"] / min(x["be"], x["fr"]) for x in xs])
        if rule is None:
            print(f"  {key[3:]}  n={len(xs)}  NO FEASIBLE RULE (default regret {default:.3f}): no entry")
            continue
        regret, ch = _changed(xs, rule)
        print(
            f"  {key[3:]}  n={len(xs)}  {rule}  regret {regret:.3f} (default {default:.3f})  changed {len(ch)} geo {_geo(ch):.3f} worst {max(ch, default=1.0):.3f}"
        )
        rules[key] = rule
    if args.rules_out:
        json.dump({json.dumps(list(k[3:])): list(r) for k, r in rules.items()}, open(args.rules_out, "w"), indent=1)
    if args.emit:
        print(f"\n    # study {s['id']}")
        for key, rule in rules.items():
            print(_entry(key, rule, s["id"]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("paths", nargs="*", help="--score: the holdout JSONL files")
    ap.add_argument("--margin", type=float, default=0.03, help="|ratio - 1| within this is neutral")
    ap.add_argument("--emit", action="store_true", help="print the TABLE block for cudnn/sdpa/placement_table.py")
    ap.add_argument("--constrained", action="store_true", help="fit under the holdout criterion against the measured default leg")
    ap.add_argument("--two-feature", action="store_true", help="--constrained: also try a second condition on KV or query tokens")
    ap.add_argument("--rules-out", help="--constrained: write the fitted rules (JSON) for --score")
    ap.add_argument("--score", help="score these frozen rules (JSON) on the holdout files given as paths")
    args = ap.parse_args()
    s = json.load(open(args.study))
    if args.constrained or args.score:
        return constrained_main(args, s)
    to_facts = ADAPTERS[s["adapter"]]
    train, hold = load(s["train"]), load(s["holdout"])

    by = collections.defaultdict(list)
    for d in train:
        f = to_facts(d["cudnn"]["case"], s)
        key = pt.class_key(s["row"], f)
        if key is not None:
            by[key].append((f, d["cudnn_oss"]["ms"] / d["cudnn"]["ms"], d["cudnn"]["ms"], d["cudnn_oss"]["ms"]))

    def class_regret(pts, feat, rule):
        direction, t = rule
        out = []
        for f, _, b, fr in pts:
            lead = direction == "always" or (direction == "ge" and pt.FEATURES[feat](f) >= t) or (direction == "le" and pt.FEATURES[feat](f) <= t)
            out.append((fr if lead else b) / min(b, fr))
        return sum(map(math.log, out)) / len(out), max(out)

    # Per class, fit every feature and keep the one with the lowest training regret (geo, then max); "cost" wins ties.
    table, feature_of = {}, {}
    for k, pts in by.items():
        best = None
        for feat in pt.FEATURES:
            rule = fit([(pt.FEATURES[feat](f), r) for f, r, _, _ in pts], args.margin)
            if rule[0] == "non-monotone":
                continue
            score = class_regret(pts, feat, rule)
            if best is None or score < best[0]:
                best = (score, feat, rule)
        table[k] = best[2] if best else ("non-monotone", None)
        feature_of[k] = best[1] if best else "cost"

    def pick(case):
        """True / False for a class with an entry; None where the runtime falls back to the hand-written rules."""
        f = to_facts(case, s)
        key = pt.class_key(s["row"], f)
        direction, t = table.get(key, (None, None))
        if direction in (None, "non-monotone"):
            return None
        if direction in ("never", "always"):
            return direction == "always"
        w = pt.FEATURES[feature_of[key]](f)
        return w >= t if direction == "ge" else w <= t

    g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
    print(f"study {s['id']}: {len(train)} train / {len(hold)} holdout cases, {len(table)} classes")
    for k, (direction, t) in sorted(table.items(), key=lambda kv: str(kv[0])):
        n = len(by[k])
        print(f"  {k[3:]}  n={n:3d}  {direction}" + ("" if t is None else f" {t:.2f} on {feature_of[k]}"))
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
                feat = "" if t is None or feature_of[k] == "cost" else f", {feature_of[k]!r}"
                print(f"    {k!r}: Entry({direction!r}, {t!r}, {s['id']!r}{feat}),")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
