# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chart generation for linear attention benchmark results.

Reads the CSV emitted by the shmoo runner (one --format_output line per
(backend, batch, seqlen) case) and generates one comparison chart per batch
size, with Forward and Backward TFLOPS panels side by side — same style as
the SDPA training benchmark charts.

    python plot_results.py results/gdn/b300/gdn_labench.csv \
        --output-dir results/gdn/b300 --gpu-name B300 --cudnn-version 9.24.0
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Backend display configuration; `order` fixes both bar grouping and legend order.
BACKEND_CONFIG = {
    "fla": {"name": "FLA (Triton)", "color": "#FF8C00", "order": 0},
    "flash_qla": {"name": "FlashQLA (TileLang)", "color": "#6495ED", "order": 1},
    "cudnn": {"name": "cuDNN", "color": "#76b900", "order": 2},
}

LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 8
TITLE_FONT_SIZE = 12
BAR_LABEL_FONT_SIZE = 6

CSV_COLUMNS = [
    "case_tag",
    "backend",
    "variant",
    "batch_size",
    "seqlen",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "fwd_ms",
    "bwd_ms",
    "fwd_tflops",
    "bwd_tflops",
    "max_diff",
    "num_iters",
]


def get_backend_display_name(backend: str, cudnn_version: str = None) -> str:
    base_name = BACKEND_CONFIG.get(backend, {}).get("name", backend)
    if backend == "cudnn" and cudnn_version:
        base_name = f"{base_name} {cudnn_version}"
    return base_name


def generate_charts(df: pd.DataFrame, output_dir: Path, gpu_name: str = "", cudnn_version: str = None, variant: str = "gdn", batch_sizes: list = None) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df[df["variant"] == variant].copy()
    if batch_sizes:
        df = df[df["batch_size"].isin(batch_sizes)].copy()
    if df.empty:
        raise ValueError(f"No rows for variant {variant!r}")

    df["backend_display"] = df["backend"].map(lambda b: get_backend_display_name(b, cudnn_version=cudnn_version))
    df["backend_order"] = df["backend"].map(lambda b: BACKEND_CONFIG.get(b, {}).get("order", 99))

    palette = {}
    for _, row in df[["backend", "backend_display"]].drop_duplicates().iterrows():
        palette[row["backend_display"]] = BACKEND_CONFIG.get(row["backend"], {}).get("color", "gray")

    saved_paths = []
    for batch_size in sorted(df["batch_size"].unique()):
        sub = df[df["batch_size"] == batch_size].copy()
        sub.sort_values(["seqlen", "backend_order"], inplace=True)
        hue_order = list(sub.sort_values("backend_order")["backend_display"].drop_duplicates())

        fwd_df = sub[sub["fwd_tflops"] > 0]
        bwd_df = sub[sub["bwd_tflops"] > 0]
        has_fwd = not fwd_df.empty
        has_bwd = not bwd_df.empty

        if has_fwd and has_bwd:
            fig, (ax_fwd, ax_bwd) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
        elif has_fwd:
            fig, ax_fwd = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
            ax_bwd = None
        else:
            fig, ax_bwd = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
            ax_fwd = None

        heads = sub["num_q_heads"].iloc[0]
        head_dim = sub["head_dim"].iloc[0]
        gpu_info = f" ({gpu_name})" if gpu_name else ""
        fig.suptitle(
            f"{variant.upper()} Linear Attention (BF16) — Batch = {batch_size}, Heads = {heads}, d = {head_dim}{gpu_info}",
            fontsize=TITLE_FONT_SIZE,
        )

        for ax, pass_df, pass_name, y_col in (
            (ax_fwd, fwd_df, "Forward", "fwd_tflops"),
            (ax_bwd, bwd_df, "Backward", "bwd_tflops"),
        ):
            if ax is None or pass_df.empty:
                continue
            sns.barplot(
                data=pass_df,
                x="seqlen",
                y=y_col,
                hue="backend_display",
                hue_order=hue_order,
                ax=ax,
                palette=palette,
                edgecolor="black",
                linewidth=0.5,
                errorbar=None,
            )
            ax.set_xlabel("Sequence Length", fontsize=LABEL_FONT_SIZE)
            ax.set_ylabel("TFLOPS", fontsize=LABEL_FONT_SIZE)
            ax.set_title(pass_name, fontsize=TITLE_FONT_SIZE)
            ax.legend(title="Backend", fontsize=LEGEND_FONT_SIZE)
            ax.tick_params(axis="x", rotation=45)
            for container in ax.containers:
                ax.bar_label(container, fmt="%.0f", fontsize=BAR_LABEL_FONT_SIZE)

        plt.tight_layout()
        output_path = output_dir / f"{variant}_b{batch_size}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths.append(output_path)
        print(f"Chart saved to {output_path}")

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="Results CSV from the shmoo runner")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: alongside the CSV)")
    parser.add_argument("--gpu-name", default="", help="GPU name for the chart title")
    parser.add_argument("--cudnn-version", default=None, help="cuDNN backend version for the legend (e.g. 9.24.0)")
    parser.add_argument("--variant", default="gdn", help="Linear attention variant to plot")
    parser.add_argument("--batch-sizes", default=None, help="Comma-separated batch sizes to plot (default: all in the CSV)")
    args = parser.parse_args()
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")] if args.batch_sizes else None

    df = pd.read_csv(args.csv)
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    output_dir = args.output_dir if args.output_dir is not None else args.csv.parent
    generate_charts(df, output_dir, gpu_name=args.gpu_name, cudnn_version=args.cudnn_version, variant=args.variant, batch_sizes=batch_sizes)


if __name__ == "__main__":
    main()
