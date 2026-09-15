# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chart generation: two figures per config.

- context:    TFLOPS, one subplot per kind — full prefill (s_q == s_kv) and
              chunked prefill (small s_q, long KV) — stacked vertically.
- generation: GB/s, one subplot per MTP width (q_tokens = 1 + MTP), stacked
              vertically. Only the config's current batch/kv axes are plotted,
              so CSVs from older, wider sweeps replot cleanly.

The x axis reads as a tensor-parallel progression: shapes form clusters and
inside each cluster the groups run tp1 -> tp2 -> tp4 -> tp8 (per-shard head
counts under each tick), separated by light vertical rules.

Every expanded case owns an x slot whether or not it ran: an unsupported
(model, shape, backend) combination is a blank slot, not a dropped tick, so
coverage gaps are visible at a glance. Series are cudnn and cudnn_oss only
(hatched = fp8 KV cache).
"""

import re
from pathlib import Path
from typing import List

BACKEND_ORDER = ["cudnn", "cudnn_oss"]
# Two-hue categorical palette, fixed order, CVD-validated (ΔE 30.3 worst
# adjacent pair); fp8-kv variants reuse the base hue with a hatch texture.
BACKEND_COLORS = {
    "cudnn": "#76B900",
    "cudnn_oss": "#5B8DEF",
}

CLUSTER_GAP = 0.7  # x-units of air between shape clusters


def _fmt_len(s: int) -> str:
    return f"{s // 1024}k" if s >= 1024 and s % 1024 == 0 else str(s)


def _mtp_title(q_tokens: int) -> str:
    if q_tokens <= 8:
        return f"MTP={q_tokens - 1} (q_tokens={q_tokens})"
    return f"q_tokens={q_tokens}"


def _tp_of(name: str) -> int:
    m = re.search(r"-tp(\d+)$", name)
    return int(m.group(1)) if m else 1


def _base_of(name: str) -> str:
    return re.sub(r"-tp\d+$", "", name)


def _clustered_bars(ax, pdf, series, metric, sol_labels):
    """Grouped bars where groups are (cluster, tp): inside a cluster the ticks
    run tp1 -> tp8, clusters are separated by a gap + a light rule, and the
    cluster's shape label sits centered beneath its span."""
    import numpy as np

    groups = list(dict.fromkeys(zip(pdf.cluster, pdf.tick)))
    width = 0.8 / len(series)
    xs, cluster_spans = [], {}
    x = 0.0
    prev_cluster = None
    for cluster, _tick in groups:
        if prev_cluster is not None and cluster != prev_cluster:
            x += CLUSTER_GAP
        xs.append(x)
        cluster_spans.setdefault(cluster, [x, x])[1] = x
        prev_cluster = cluster
        x += 1.0
    xs = np.asarray(xs)

    for j, s_name in enumerate(series):
        vals, sols = [], []
        for cluster, tick in groups:
            row = pdf[(pdf.series == s_name) & (pdf.cluster == cluster) & (pdf.tick == tick) & pdf.success]
            vals.append(row[metric].median() if len(row) else np.nan)
            sols.append(row.sol_pct.median() if len(row) and row.sol_pct.notna().any() else None)
        base = s_name.replace(" (fp8 kv)", "")
        bars = ax.bar(
            xs + j * width,
            vals,
            width,
            label=s_name,
            color=BACKEND_COLORS.get(base, "#999999"),
            hatch="//" if s_name.endswith("(fp8 kv)") else None,
            edgecolor="white",
            linewidth=0.4,
        )
        if sol_labels:
            for bar, sol in zip(bars, sols):
                if sol and bar.get_height() > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{sol:.0f}%", ha="center", va="bottom", fontsize=6, rotation=90)

    center = width * (len(series) - 1) / 2
    ax.set_xticks(xs + center)
    ax.set_xticklabels([t for _c, t in groups], fontsize=6.5)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 1.0)
    spans = list(cluster_spans.items())
    for cluster, (lo, hi) in spans:
        ax.text((lo + hi) / 2 + center, -0.18, cluster, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    for (_c1, (_lo1, hi1)), (_c2, (lo2, _hi2)) in zip(spans, spans[1:]):
        # midpoint of the air between cluster 1's last bar and cluster 2's first
        ax.axvline((hi1 + 0.8 + lo2) / 2 - width / 2, color="gray", alpha=0.25, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def _plot_phase(df, config, phase: str, outdir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = "tflops" if phase == "context" else "gbps"
    metric_label = "TFLOPS" if phase == "context" else "GB/s  (labels: % of memory SOL)"

    # Keep every expanded case — failures included — so unsupported combos
    # hold their slot as a blank bar.
    pdf = df[df.phase == phase].copy()
    pdf = pdf[pdf.backend.isin(BACKEND_ORDER)]
    # older CSVs may carry model shards the config has since dropped
    pdf = pdf[pdf.model_name.isin({m.name for m in config.models})]
    if phase == "generation":
        # honor the config's current axes: older CSVs may carry batches or
        # cache lengths the config has since dropped as redundant
        pdf = pdf[pdf.batch_size.isin(config.generation_batch_sizes)]
        pdf = pdf[pdf.kv_len.isin({kv for _q, kv in config.generation_shapes})]
    if pdf.empty:
        return []

    pdf["tp"] = pdf.model_name.map(_tp_of)
    pdf["base_model"] = pdf.model_name.map(_base_of)
    pdf["heads"] = pdf.apply(lambda r: f"{int(r.num_q_heads)}/{int(r.num_kv_heads)}h", axis=1)
    pdf["tick"] = pdf.apply(lambda r: f"tp{r.tp}\n{r.heads}", axis=1)

    if phase == "context":
        pdf["row_key"] = pdf.apply(lambda r: "full" if r.q_tokens == r.kv_len else "chunked", axis=1)
        pdf["cluster"] = pdf.apply(
            lambda r: f"{r.base_model}  s{_fmt_len(r.kv_len)}" if r.q_tokens == r.kv_len else f"{r.base_model}  q{_fmt_len(r.q_tokens)}/kv{_fmt_len(r.kv_len)}",
            axis=1,
        )
        row_keys = [k for k in ("full", "chunked") if (pdf.row_key == k).any()]
        row_titles = {"full": "full prefill (s_q = s_kv)", "chunked": "chunked prefill (small s_q, long KV)"}
    else:
        pdf["row_key"] = pdf.q_tokens
        pdf["cluster"] = pdf.apply(lambda r: f"{r.base_model}  b{r.batch_size} kv{_fmt_len(r.kv_len)}", axis=1)
        row_keys = sorted(pdf.q_tokens.unique())
        row_titles = {q: _mtp_title(int(q)) for q in row_keys}

    pdf["series"] = pdf.apply(lambda r: r.backend + (" (fp8 kv)" if r.get("kv_cache_dtype") == "fp8_e4m3" else ""), axis=1)

    base_order = {}
    for m in config.models:
        base_order.setdefault(_base_of(m.name), len(base_order))
    pdf = pdf.sort_values(
        by=["base_model", "batch_size", "q_tokens", "kv_len", "tp"],
        key=lambda col: col.map(base_order) if col.name == "base_model" else col,
    )
    series_order = [b + suf for b in BACKEND_ORDER for suf in ("", " (fp8 kv)")]
    series = [s for s in series_order if s in set(pdf.series)]
    if not series:
        return []

    max_groups = max(len(dict.fromkeys(zip(sub.cluster, sub.tick))) for sub in (pdf[pdf.row_key == k] for k in row_keys))
    fig_w = max(10, 0.55 * max_groups * len(series))
    fig, axes = plt.subplots(len(row_keys), 1, figsize=(fig_w, 3.9 * len(row_keys)), squeeze=False)
    for ax, key in zip(axes[:, 0], row_keys):
        sub = pdf[pdf.row_key == key]
        _clustered_bars(ax, sub, series, metric, sol_labels=phase == "generation")
        ax.set_title(row_titles[key], fontsize=10, loc="left")
        ax.set_ylabel(metric_label, fontsize=8)

    gpu = next(iter(pdf.gpu_name.dropna()), "unknown GPU")
    fig.suptitle(f"{config.name} — {phase} — {gpu}", fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=min(4, len(series)))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.subplots_adjust(hspace=0.55)
    path = outdir / f"{config.name}_{phase}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return [path]


def generate_charts(df, config) -> List[Path]:
    outdir = Path(config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for phase in ("context", "generation"):
        paths.extend(_plot_phase(df, config, phase, outdir))
    return paths
