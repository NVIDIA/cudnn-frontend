# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chart generation for DSA sparse attention benchmark results.

One figure per config: forward and backward TFLOPS side by side, sequence
length on the x axis, one bar per (model, dtype, deterministic) series. A
dashed line marks the dense-MMA peak recorded by the benchmark (median of the
per-row ``peak_mma_tflops``) so sparse-gather efficiency reads against the
same SOL the dense suites use.
"""

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from .config_types import DsaBenchmarkConfig

logger = logging.getLogger(__name__)

# Two-hue categorical palette shared with attention_inference (cudnn green,
# blue), then darker tones; assigned per series in legend order.
SERIES_COLORS = ["#76B900", "#5B8DEF", "#2E8B57", "#1F3F8F", "#B8860B", "#8B0000"]

LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 8
TITLE_FONT_SIZE = 12
BAR_LABEL_FONT_SIZE = 6

_DTYPE_TAG = {"bfloat16": "BF16", "float16": "FP16"}


def _fmt_len(s: int) -> str:
    return f"{s // 1024}k" if s >= 1024 and s % 1024 == 0 else str(s)


def _series_label(row) -> str:
    dtype = _DTYPE_TAG.get(row["data_type"], row["data_type"])
    det = " det" if str(row["deterministic_bwd"]).lower() == "true" else ""
    return f"{row['model_name']} H{row['num_q_heads']} K={row['topk']} ({dtype}){det}"


def generate_charts(
    df: "pd.DataFrame",
    config: "DsaBenchmarkConfig",
    output_dir: Optional[Path] = None,
    peak_tflops: Optional[float] = None,
) -> list:
    """Save ``<config>.webp`` with fwd/bwd TFLOPS subplots; returns saved paths."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = df[(df["success"] == True) & (df["tflops"] > 0)].copy()
    if df.empty:
        raise ValueError("No successful results to plot")

    if peak_tflops is None and "peak_mma_tflops" in df.columns:
        ok = df[df["peak_mma_tflops"].notna() & (df["peak_mma_tflops"] > 0)]
        peak_tflops = float(ok["peak_mma_tflops"].median()) if not ok.empty else None

    if output_dir is None:
        output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df["series"] = df.apply(_series_label, axis=1)
    df["seqlen_label"] = df.apply(lambda r: f"{_fmt_len(int(r['q_seqlen']))}x{_fmt_len(int(r['kv_seqlen']))}", axis=1)
    df["det_order"] = df["deterministic_bwd"].astype(str).str.lower().map({"false": 0, "true": 1})
    df.sort_values(["q_seqlen", "num_q_heads", "data_type", "det_order"], inplace=True)

    series_order = list(
        df[["series", "num_q_heads", "data_type", "det_order"]].drop_duplicates().sort_values(["num_q_heads", "data_type", "det_order"])["series"]
    )
    palette = {s: SERIES_COLORS[i % len(SERIES_COLORS)] for i, s in enumerate(series_order)}
    seqlen_order = list(df.sort_values("q_seqlen")["seqlen_label"].drop_duplicates())

    fwd_df = df[df["profile_pass"] == "fwd"]
    bwd_df = df[df["profile_pass"] == "bwd"]
    panels = [(t, d) for t, d in (("Forward", fwd_df), ("Backward", bwd_df)) if not d.empty]

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6), dpi=150, squeeze=False)
    fig.suptitle(f"{config.name} — DSA sparse attention (shared K=V, d={int(df['head_dim_qk'].iloc[0])})", fontsize=TITLE_FONT_SIZE)

    for ax, (title, pdf) in zip(axes[0], panels):
        sns.barplot(
            data=pdf,
            x="seqlen_label",
            y="tflops",
            order=seqlen_order,
            hue="series",
            hue_order=[s for s in series_order if s in set(pdf["series"])],
            ax=ax,
            palette=palette,
            edgecolor="black",
            linewidth=0.5,
            errorbar=None,
        )
        if peak_tflops:
            ax.axhline(peak_tflops, linestyle="--", linewidth=1.3, color="#333333", alpha=0.85, label="dense MMA peak", zorder=1)
        ax.set_xlabel("Sequence Length (s_q x s_kv)", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("TFLOPS (on gathered rows)", fontsize=LABEL_FONT_SIZE)
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.legend(title="Model", fontsize=LEGEND_FONT_SIZE)
        ax.tick_params(axis="x", rotation=45)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", fontsize=BAR_LABEL_FONT_SIZE)

    plt.tight_layout()
    output_path = output_dir / f"{config.name}.webp"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pil_kwargs={"lossless": True, "quality": 100, "method": 6, "exact": True})
    plt.close()
    logger.info(f"Chart saved to {output_path}")
    return [output_path]
