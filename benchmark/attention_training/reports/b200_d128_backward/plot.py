# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the recorded backward summaries; this script does not run benchmarks."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

HERE = Path(__file__).resolve().parent
COLORS = {"cuDNN BE": "#739d00", "FA4": "#167ab0", "FROST": "#b4477e"}
GOLD = "#d39b00"
INK = "#23313b"
SEQUENCES = (2048, 4096, 8192, 16384)


def load_measurements(path):
    """Reject incomplete/invalid chart inputs and inconsistent summary arithmetic."""
    data = json.loads(path.read_text())
    rows = data["measurements"]
    assert data["schema"] == "sdpa_backward_summary_v1"
    assert len(rows) == 23
    keys = {(r["suite"], r["baseline"], r["mask"], r["sequence_length"]) for r in rows}
    expected = {("dense", base, mask, s) for base in ("cuDNN BE", "FA4") for mask in ("causal", "none") for s in SEQUENCES}
    expected |= {("thd_cp8", base, "causal", s) for base in ("cuDNN BE", "FA4") for s in SEQUENCES[:3]}
    expected.add(("thd_cp2", "FA4", "causal", 8192))
    assert keys == expected and len(keys) == len(rows)
    for row in rows:
        assert row["baseline_us"] > 0 and row["frost_us"] > 0
        ratio = estimated_ratio(row)
        assert math.isclose(100 * (ratio - 1), row["speedup_pct"], abs_tol=0.0051)
        if row["target_ratio"] is not None:
            assert row["target_met"] == (row["target_ratio"] <= 1.03)
        if row["suite"].startswith("thd"):
            assert row["batch_size"] * row["sequence_length"] == 524288
    return rows


def estimated_ratio(row):
    if row["ratio_estimator"] == "median_of_six_paired_baseline_over_frost_ratios":
        return row["paired_baseline_over_frost"]
    assert row["ratio_estimator"] == "ratio_of_summary_latencies"
    return row["baseline_us"] / row["frost_us"]


def select(rows, suite, baseline, mask="causal"):
    return sorted((r for r in rows if (r["suite"], r["baseline"], r["mask"]) == (suite, baseline, mask)), key=lambda r: r["sequence_length"])


def theme():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#aab4ba",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "grid.color": "#dbe1e5",
            "svg.fonttype": "none",
            "svg.hashsalt": "b200-d128-backward-v1",
        }
    )


def save(fig, output, name):
    svg = output / f"{name}.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(output / f"{name}.png", dpi=170, facecolor="white")
    plt.close(fig)


def latency_panel(ax, records, title, limits, y_ticks, labels=True):
    baseline = records[0]["baseline"]
    x = [r["sequence_length"] for r in records]
    a = [r["baseline_us"] / 1000 for r in records]
    b = [r["frost_us"] / 1000 for r in records]
    ax.plot(x, a, color=COLORS[baseline], marker="s", linewidth=2.4, markersize=7, label=baseline)
    ax.plot(x, b, color=COLORS["FROST"], marker="o", linestyle="--", linewidth=2.2, markersize=6, label="FROST (paired)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(x[0] / 1.22, x[-1] * 1.26)
    ax.set_ylim(*limits)
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1024:g}K"))
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", pad=13)
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Complete backward latency (ms)")
    if labels:
        for r, xx, aa, bb in zip(records, x, a, b):
            ax.annotate(
                f"{r['speedup_pct']:+.2f}%",
                (xx, max(aa, bb)),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                color=INK,
                fontsize=10,
                weight="semibold",
            )
    ax.legend(loc="upper left", frameon=False, fontsize=10)


def thd_latency(rows, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), gridspec_kw={"width_ratios": [1, 1, 0.95]})
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.32, top=0.72, wspace=0.32)
    fig.suptitle("Report-mapped THD/GQA: complete backward only", x=0.06, y=0.975, ha="left", fontsize=20, weight="bold")
    fig.text(0.06, 0.90, "GB200 · BF16 · d = 128 · causal · B × S = 524,288 per rank", fontsize=12)
    fig.text(0.06, 0.855, "CP8 improves over cuDNN; the sustained CP2 8K gap to FA4 remains.", fontsize=13, weight="semibold")
    for ax, baseline, title in zip(axes[:2], ("cuDNN BE", "FA4"), ("CP8 · before / after", "CP8 · FA4 comparison")):
        latency_panel(ax, select(rows, "thd_cp8", baseline), title, (2.5, 14), [3, 5, 10])
    r = select(rows, "thd_cp2", "FA4")[0]
    ax = axes[2]
    values = [r["baseline_us"] / 1000, r["frost_us"] / 1000]
    bars = ax.bar([0, 1], values, width=0.57, color=[COLORS["FA4"], COLORS["FROST"]])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.3f}", ha="center", weight="semibold")
    ax.set_xticks([0, 1], ["FA4", "FROST"])
    ax.set_ylim(0, 49)
    ax.set_ylabel("Complete backward latency (ms)")
    ax.grid(axis="y", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title("CP2 · 8K · sustained 1200 W", loc="left", pad=13)
    ax.text(0.5, 0.94, "FROST / FA4 = 1.042×\n3% target missed", transform=ax.transAxes, ha="center", va="top", fontsize=11, color="#a33a32")
    fig.text(
        0.06,
        0.14,
        "CP8: Hq/Hkv = 4/1 · paced duty 0.2 · observed 1830 MHz\nLabels: FROST speed-up = 100 × (baseline / FROST − 1). Each baseline has its own paired FROST series.",
        fontsize=10,
        linespacing=1.5,
    )
    fig.text(
        0.71,
        0.14,
        "CP2: Hq/Hkv = 16/4 · separate power class\nRatio: median of six paired brackets;\nbar heights: median provider latencies.",
        fontsize=9.5,
        linespacing=1.5,
    )
    fig.text(
        0.06,
        0.045,
        "All helpers included (FROST: five launches). Forward and CP communication excluded. Lower latency is better.",
        fontsize=10,
        color="#52616a",
    )
    save(fig, output, "thd_backward_latency")


def dense_latency(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(12.7, 10))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.80, hspace=0.36, wspace=0.24)
    fig.suptitle("Dense MHA: cuDNN parity, causal wins versus FA4", x=0.08, y=0.965, ha="left", fontsize=19, weight="bold")
    fig.text(0.08, 0.92, "GB200 · BF16 · B = 2 · Hq = Hkv = 8 · d = 128 · complete backward", fontsize=12)
    fig.text(0.08, 0.87, "Long unmasked cases miss the FA4 target; cuDNN parity holds across the measured grid.", fontsize=12)
    for row, baseline in enumerate(("cuDNN BE", "FA4")):
        for col, mask in enumerate(("causal", "none")):
            latency_panel(
                axes[row, col],
                select(rows, "dense", baseline, mask),
                f"{'Causal' if mask == 'causal' else 'Unmasked'} · {'before / after cuDNN' if baseline == 'cuDNN BE' else 'FA4 comparison'}",
                (0.035, 8),
                [0.05, 0.1, 0.5, 1, 5],
            )
    fig.text(
        0.08,
        0.065,
        "Paced duty 0.5 · observed 1830 MHz · all three FROST launches included · both axes logarithmic\nLabels: FROST speed-up = 100 × (baseline / FROST − 1). Each baseline has its own paired FROST series.",
        fontsize=10,
        linespacing=1.5,
    )
    save(fig, output, "dense_backward_latency")


def winner_panel(ax, rows, specs, title):
    ax.set_title(title, loc="left", pad=17)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(len(specs) - 0.5, -0.5)
    ax.set_xticks(range(4), ["2K", "4K", "8K", "16K"])
    ax.set_yticks(range(len(specs)), [x[0] for x in specs])
    ax.tick_params(length=0, axis="both", pad=9)
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    for y, (_, suite, base, mask) in enumerate(specs):
        cells = {r["sequence_length"]: r for r in select(rows, suite, base, mask)}
        for x, sequence in enumerate(SEQUENCES):
            r = cells.get(sequence)
            face, edge, width = "#edf0f2", "white", 3
            label, color = "No qualified\nresult", "#707b83"
            if r:
                ratio = estimated_ratio(r)
                winner = "FROST" if ratio >= 1 else base
                face, color = ("#d9ebd0" if ratio >= 1 else "#d9e7f2"), INK
                label = f"{winner}\n{r['speedup_pct']:+.2f}%"
                if base == "FA4" and not r["target_met"]:
                    label += "  ×"
                if max(ratio, 1 / ratio) - 1 < 0.03:
                    edge, width = GOLD, 2.5
            ax.add_patch(Rectangle((x - 0.475, y - 0.46), 0.95, 0.92, facecolor=face, edgecolor=edge, linewidth=width))
            ax.text(x, y, label, ha="center", va="center", fontsize=11, color=color, linespacing=1.5)


def winners(rows, output):
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 9.8), gridspec_kw={"height_ratios": [1, 1]})
    fig.subplots_adjust(left=0.30, right=0.96, bottom=0.18, top=0.80, hspace=0.47)
    fig.suptitle("Backward winner and remaining coverage", x=0.05, y=0.96, ha="left", fontsize=21, weight="bold")
    fig.text(0.05, 0.91, "Cell text: observed winner and signed FROST speed-up versus that row’s baseline", fontsize=12)
    winner_panel(
        axes[0],
        rows,
        [
            ("cuDNN · causal", "dense", "cuDNN BE", "causal"),
            ("cuDNN · unmasked", "dense", "cuDNN BE", "none"),
            ("FA4 · causal", "dense", "FA4", "causal"),
            ("FA4 · unmasked", "dense", "FA4", "none"),
        ],
        "Dense MHA · paced duty 0.5 · B=2 / H=8",
    )
    winner_panel(
        axes[1],
        rows,
        [
            ("CP8 · cuDNN · paced", "thd_cp8", "cuDNN BE", "causal"),
            ("CP8 · FA4 · paced", "thd_cp8", "FA4", "causal"),
            ("CP4 · FA4", "thd_cp4", "FA4", "causal"),
            ("CP2 · FA4 · sustained", "thd_cp2", "FA4", "causal"),
        ],
        "THD/GQA · causal · 524,288 tokens per rank",
    )
    handles = [
        Patch(facecolor="#d9ebd0", label="FROST lower latency"),
        Patch(facecolor="#d9e7f2", label="Baseline lower latency"),
        Patch(facecolor="white", edgecolor=GOLD, linewidth=2, label="Near tie (<3% winner advantage)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.105), ncol=3, frameon=False, fontsize=10)
    fig.text(
        0.05, 0.07, "×  FROST misses the FA4 1.03× ceiling under the prespecified estimator. Gold is a convention, not a confidence interval.", fontsize=10
    )
    fig.text(
        0.05,
        0.035,
        "CP8 is paced / uncapped; CP2 is sustained / 1200 W. Compare only within a cell. Unqualified or absent results are not plotted.",
        fontsize=10,
        color="#52616a",
    )
    save(fig, output, "backward_winner_map")


def write_table(rows, output):
    lines = [
        "# Recorded backward measurements",
        "",
        "Generated from [measurements.json](measurements.json) by [plot.py](plot.py). All values include the complete backward chain.",
        "",
    ]
    for suite, title in (("dense", "Dense MHA, paced duty 0.5"), ("thd_cp8", "CP8 THD/GQA, paced duty 0.2"), ("thd_cp2", "CP2 THD/GQA, sustained 1200 W")):
        lines += [
            f"## {title}",
            "",
            "| B | S | Mask | Baseline | Baseline ms | Paired FROST ms | FROST speed-up | SOL % (MMA) |",
            "|---:|---:|---|---|---:|---:|---:|---|",
        ]
        for r in rows:
            if r["suite"] == suite:
                lines.append(
                    f"| {r['batch_size']} | {r['sequence_length']} | {r['mask']} | {r['baseline']} | {r['baseline_us'] / 1000:.6f} | {r['frost_us'] / 1000:.6f} | {r['speedup_pct']:+.2f}% | unavailable |"
                )
        lines += [""]
    lines += [
        "Dense/CP8 speed-up is the ratio of summarized latencies. CP2 uses the median of six paired baseline/FROST ratios; dividing its displayed latency medians is a different estimator. See the [methodology](README.md#measurement-method-and-limits).",
        "",
    ]
    (output / "measurements.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "measurements.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    rows = load_measurements(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    theme()
    thd_latency(rows, args.output_dir)
    dense_latency(rows, args.output_dir)
    winners(rows, args.output_dir)
    write_table(rows, args.output_dir)
    print(f"Validated {len(rows)} summaries; generated three SVG/PNG pairs and measurements.md.")


if __name__ == "__main__":
    main()
