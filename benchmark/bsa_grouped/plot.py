# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the summary table and figure from bench.py's raw measurements."""

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    config = data["metadata"]["args"]
    topks = config["topk"]
    rows = []
    for phase in ("fwd", "bwd"):
        for topk in topks:
            cases = [case for case in data["cases"] if case["topk"] == topk]
            if len(cases) != len(config["seeds"]) or {c["seed"] for c in cases} != set(config["seeds"]):
                raise ValueError(f"incomplete seed set for topK {topk}")
            baseline = statistics.median(c[phase]["native_blk64"]["tflops"] for c in cases)
            for variant in cases[0][phase]:
                values = [c[phase][variant]["tflops"] for c in cases]
                rows.append(
                    dict(
                        phase=phase,
                        topk=topk,
                        sparsity=1 - topk * 64 / config["length"],
                        variant=variant,
                        median_ms=statistics.median(c[phase][variant]["median_ms"] for c in cases),
                        tflops=statistics.median(values),
                        seed_min_tflops=min(values),
                        seed_max_tflops=max(values),
                        speedup_vs_native64=statistics.median(values) / baseline,
                    )
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    labels = {
        "native_blk64": "cuDNN FE native blk64",
        "grouped_blk128": "cuDNN FE grouped 128",
        "grouped_default": "cuDNN FE grouped 128 (default buckets)",
        "grouped_one_bucket": "cuDNN FE grouped 128 (one bucket)",
        "cake_blk64": "CAKE CUDA blk64",
    }
    colors = {"native_blk64": "#3976b3", "grouped_blk128": "#80b800", "grouped_default": "#80b800", "grouped_one_bucket": "#9467bd", "cake_blk64": "#ef9842"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, axes = plt.subplots(2, 1, figsize=(19, 12))
    fig.suptitle("Fully Paired Block Sparse Attention · blk64 → blk128", fontsize=25, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.943,
        f'{data["metadata"]["gpu"]} / SM100  |  B=1  |  Hq=Hkv={config["heads"]}  |  Nq=Nkv={config["length"]}  |  D=128  |  BF16',
        ha="center",
        fontsize=14,
        color="#444444",
    )
    for ax, phase in zip(axes, ("fwd", "bwd")):
        variants = [name for name in labels if any(r["phase"] == phase and r["variant"] == name for r in rows)]
        width = 0.8 / len(variants)
        x = np.arange(len(topks))
        largest = 0
        for i, variant in enumerate(variants):
            values = [next(r for r in rows if r["phase"] == phase and r["topk"] == k and r["variant"] == variant) for k in topks]
            y = np.array([r["tflops"] for r in values])
            error = np.array([[r["tflops"] - r["seed_min_tflops"] for r in values], [r["seed_max_tflops"] - r["tflops"] for r in values]])
            positions = x + (i - (len(variants) - 1) / 2) * width
            ax.bar(
                positions,
                y,
                width * 0.94,
                label=labels[variant],
                color=colors[variant],
                edgecolor="#4c5860",
                linewidth=0.6,
                yerr=error,
                capsize=3,
                error_kw={"linewidth": 1},
            )
            largest = max(largest, max(r["seed_max_tflops"] for r in values))
            for pos, r in zip(positions, values):
                text = f'{r["tflops"]:.0f}'
                if variant.startswith("grouped"):
                    text = f'{r["speedup_vs_native64"]:.2f}×\n' + text
                ax.annotate(
                    text, (pos, r["seed_max_tflops"]), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold"
                )
        ax.set_title("Forward · O + LSE" if phase == "fwd" else "Backward · dQ + dK + dV", loc="left", fontweight="bold", fontsize=17)
        ax.set_xticks(x, [f'topK={k}\nSparsity={100*(1-k*64/config["length"]):.1f}%' for k in topks])
        ax.set_ylabel("Active-work throughput (TFLOPS)", fontweight="bold")
        ax.set_ylim(0, largest * 1.23)
        ax.grid(axis="y", linestyle=(0, (4, 4)), alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", fontsize=11, ncol=len(variants), frameon=False)
    fig.text(
        0.5,
        0.037,
        f'{len(config["seeds"])} seeds × {config["rounds"]} randomized CUDA Graph rounds. Bars: median; whiskers: seed min–max (not confidence intervals). × = speedup vs native blk64.',
        ha="center",
        color="#555555",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.018,
        "Includes all grouped GPU permutations and backward reset/pre/post kernels. Excludes CPU preparation, routing, compilation and allocation. Synthetic 100% paired graph.",
        ha="center",
        color="#555555",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.065, right=0.99, hspace=0.40)
    for suffix in ("png", "svg", "pdf"):
        destination = args.output_dir / f"performance.{suffix}"
        fig.savefig(destination, dpi=160)
        if suffix == "svg":
            destination.write_text("\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n")
    plt.close(fig)

    lines = [
        "| topK | Sparsity | FWD native64 | FWD grouped | FWD CAKE | BWD native64 | BWD grouped default | BWD grouped one bucket | Prepare default (ms) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for topk in topks:

        def cell(phase, variant):
            r = next((r for r in rows if r["phase"] == phase and r["topk"] == topk and r["variant"] == variant), None)
            if r is None:
                return "—"
            return f'{r["tflops"]:.1f} ({r["median_ms"]:.3f} ms)'

        prep = statistics.median(c["prepare_ms"]["grouped_default"] for c in data["cases"] if c["topk"] == topk)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(topk),
                    f'{100*(1-topk*64/config["length"]):.1f}%',
                    cell("fwd", "native_blk64"),
                    cell("fwd", "grouped_blk128"),
                    cell("fwd", "cake_blk64"),
                    cell("bwd", "native_blk64"),
                    cell("bwd", "grouped_default"),
                    cell("bwd", "grouped_one_bucket"),
                    f"{prep:.1f}",
                ]
            )
            + " |"
        )
    (args.output_dir / "table.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
