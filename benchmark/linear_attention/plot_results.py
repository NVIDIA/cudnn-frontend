# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chart generation for linear attention benchmark results.

Reads the CSV emitted by the shmoo runner (one --format_output line per
(backend, batch, seqlen) case) and generates one comparison chart per batch
size, with Forward and Backward TFLOPS panels side by side — same style as
the SDPA training benchmark charts.

    python plot_results.py results/gdn/gb300/gdn_20260813.csv \
        --output-dir results/gdn/gb300 --gpu-name GB300 --cudnn-version 9.24.0
"""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Backend display configuration; `order` fixes both bar grouping and legend order.
BACKEND_CONFIG = {
    "fla": {"name": "FLA (Triton)", "color": "#FF8C00", "order": 0},
    "flash_qla": {"name": "FlashQLA (TileLang)", "color": "#6495ED", "order": 1},
    "flash_kda": {"name": "FlashKDA", "color": "#9370DB", "order": 2},
    "cudnn": {"name": "cuDNN (default)", "color": "#76b900", "order": 3},
    "cudnn_state_on": {"name": "cuDNN (state on)", "color": "#2f6e00", "order": 4},
}

# Backends dropped from every chart (rows may still exist in older CSVs).
UNAVAILABLE_BACKENDS = ()

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
    "fwd_bw",
    "bwd_bw",
]

# One chart per metric: (fwd column, bwd column, y-axis label, bar label
# format, filename suffix).
METRIC_CONFIG = (
    ("fwd_tflops", "bwd_tflops", "TFLOPS", "%.0f", "_flops"),
    ("fwd_bw", "bwd_bw", "DRAM Bandwidth (TB/s)", "%.2f", "_bw"),
)


def get_backend_display_name(backend: str, cudnn_version: Optional[str] = None) -> str:
    return BACKEND_CONFIG.get(backend, {}).get("name", backend)


def generate_charts(
    df: pd.DataFrame,
    output_dir: Path,
    gpu_name: str = "",
    cudnn_version: Optional[str] = None,
    variant: str = "gdn",
    batch_sizes: Optional[List[int]] = None,
    x_axis: str = "seqlen",
) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df[df["variant"] == variant].copy()
    df = df[~df["backend"].isin(UNAVAILABLE_BACKENDS)].copy()
    if batch_sizes:
        df = df[df["batch_size"].isin(batch_sizes)].copy()
    if df.empty:
        raise ValueError(f"No rows for variant {variant!r}")

    df["backend_display"] = df["backend"].map(lambda b: get_backend_display_name(b, cudnn_version=cudnn_version))
    df["backend_order"] = df["backend"].map(lambda b: BACKEND_CONFIG.get(b, {}).get("order", 99))

    palette = {}
    for _, row in df[["backend", "backend_display"]].drop_duplicates().iterrows():
        palette[row["backend_display"]] = BACKEND_CONFIG.get(row["backend"], {}).get("color", "gray")

    x_col, group_col = ("seqlen", "batch_size") if x_axis == "seqlen" else ("batch_size", "seqlen")
    x_label = "Sequence Length" if x_axis == "seqlen" else "Batch Size"

    saved_paths = []
    for group_val in sorted(df[group_col].unique()):
        sub = df[df[group_col] == group_val].copy()
        sub.sort_values([x_col, "backend_order"], inplace=True)

        for fwd_col, bwd_col, y_label, bar_fmt, file_suffix in METRIC_CONFIG:
            if fwd_col not in sub.columns or bwd_col not in sub.columns:
                continue
            fwd_df = sub[sub[fwd_col] > 0]
            bwd_df = sub[sub[bwd_col] > 0]
            has_fwd = not fwd_df.empty
            has_bwd = not bwd_df.empty
            if not has_fwd and not has_bwd:
                continue

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
            group_label = f"Batch = {group_val}" if x_axis == "seqlen" else f"Sequence Length = {group_val}"
            fig.suptitle(
                f"{variant.upper()} Linear Attention (BF16) — {group_label}, Heads = {heads}, d = {head_dim}{gpu_info}",
                fontsize=TITLE_FONT_SIZE,
            )

            for ax, pass_df, pass_name, y_col in (
                (ax_fwd, fwd_df, "Forward", fwd_col),
                (ax_bwd, bwd_df, "Backward", bwd_col),
            ):
                if ax is None or pass_df.empty:
                    continue
                hue_order = list(pass_df.sort_values("backend_order")["backend_display"].drop_duplicates())
                sns.barplot(
                    data=pass_df,
                    x=x_col,
                    y=y_col,
                    hue="backend_display",
                    hue_order=hue_order,
                    ax=ax,
                    palette=palette,
                    edgecolor="black",
                    linewidth=0.5,
                    errorbar=None,
                )
                ax.set_xlabel(x_label, fontsize=LABEL_FONT_SIZE)
                ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE)
                ax.set_title(pass_name, fontsize=TITLE_FONT_SIZE)
                ax.legend(title="Backend", fontsize=LEGEND_FONT_SIZE, loc="upper left")
                ax.tick_params(axis="x", rotation=45)
                for container in ax.containers:
                    ax.bar_label(container, fmt=bar_fmt, fontsize=BAR_LABEL_FONT_SIZE)

            plt.tight_layout()
            gv = int(group_val)
            if df[group_col].nunique() == 1:
                # the sweep pinned the group dimension: fixed-batch (seqlen
                # sweep) / fixed-seq (batch sweep) result-tree naming
                stem = f"{variant}_fixed_batch" if x_axis == "seqlen" else f"{variant}_fixed_seq"
            else:
                stem = f"{variant}_b{gv}" if x_axis == "seqlen" else f"{variant}_t{gv}_bsweep"
            output_path = output_dir / f"{stem}{file_suffix}.png"
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
    parser.add_argument(
        "--x-axis", default="seqlen", choices=("seqlen", "batch"), help="Bar-group axis: seqlen (one chart per batch) or batch (one chart per seqlen)"
    )
    args = parser.parse_args()
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")] if args.batch_sizes else None

    df = pd.read_csv(args.csv)
    # fwd_bw/bwd_bw are newer columns; older CSVs simply skip the BW charts.
    missing = [c for c in CSV_COLUMNS if c not in df.columns and c not in ("fwd_bw", "bwd_bw")]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    output_dir = args.output_dir if args.output_dir is not None else args.csv.parent
    generate_charts(df, output_dir, gpu_name=args.gpu_name, cudnn_version=args.cudnn_version, variant=args.variant, batch_sizes=batch_sizes, x_axis=args.x_axis)


if __name__ == "__main__":
    main()
