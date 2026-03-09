"""
Chart generation for norm benchmark results.

Generates comparison bar charts showing backend performance side-by-side.
"""

from pathlib import Path
from typing import Optional, TYPE_CHECKING, List
import logging

if TYPE_CHECKING:
    import pandas as pd
    from .config_types import NormBenchmarkConfig

logger = logging.getLogger(__name__)

BACKEND_CONFIG = {
    "cudnn": {"name": "cuDNN", "color": "#76b900", "order": 0},
    "pytorch": {"name": "PyTorch", "color": "#EE4C2C", "order": 1},
    "torch_compile": {"name": "torch.compile", "color": "#6495ED", "order": 2},
}

LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 8
TITLE_FONT_SIZE = 12
BAR_LABEL_FONT_SIZE = 6


def get_backend_display_name(backend: str, data_type: str) -> str:
    base_name = BACKEND_CONFIG.get(backend, {}).get("name", backend)
    if data_type == "float16":
        return f"{base_name} (FP16)"
    elif data_type == "bfloat16":
        return f"{base_name} (BF16)"
    return base_name


def get_backend_color(backend: str) -> str:
    return BACKEND_CONFIG.get(backend, {}).get("color", "gray")


def generate_norm_charts(
    df: "pd.DataFrame",
    config: "NormBenchmarkConfig",
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Generate benchmark charts for norm results.

    Creates two charts:
    - Time (ms) comparison
    - Bandwidth (GB/s) comparison

    Each with forward and backward subplots.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    df = df[df["success"] == True].copy()

    if df.empty:
        raise ValueError("No successful results to plot")

    if output_dir is None:
        output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df["backend_display"] = df.apply(lambda r: get_backend_display_name(r["backend"], r["data_type"]), axis=1)
    df["norm_label"] = df.apply(
        lambda r: f"{r['norm_name']}\n{r['N']}x{r['C']}",
        axis=1,
    )

    df["backend_order"] = df["backend"].map(lambda b: BACKEND_CONFIG.get(b, {}).get("order", 99))
    df.sort_values(["norm_name", "N", "backend_order"], inplace=True)

    unique_combos = df[["backend", "data_type", "backend_display"]].drop_duplicates()
    palette = {}
    for _, row in unique_combos.iterrows():
        palette[row["backend_display"]] = get_backend_color(row["backend"])

    saved_paths = []

    has_fwd = (df["fwd_time_ms"] > 0).any() and (df["fwd_time_ms"] < float("inf")).all()
    has_bwd = (df["bwd_time_ms"] > 0).any() and (df["bwd_time_ms"] < float("inf")).all()

    # Chart 1: Time (ms)
    if has_fwd or has_bwd:
        ncols = (1 if has_fwd else 0) + (1 if has_bwd else 0)
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6), dpi=150)
        if ncols == 1:
            axes = [axes]

        ax_idx = 0
        if has_fwd:
            ax = axes[ax_idx]
            ax_idx += 1
            fwd_df = df[df["fwd_time_ms"] > 0]
            _plot_grouped_bars(ax, fwd_df, "norm_label", "fwd_time_ms", "backend_display", palette)
            ax.set_xlabel("Norm Configuration", fontsize=LABEL_FONT_SIZE)
            ax.set_ylabel("Time (ms)", fontsize=LABEL_FONT_SIZE)
            ax.set_title(f"{config.name} Norm Forward (Time)", fontsize=TITLE_FONT_SIZE)
            ax.legend(title="Backend", fontsize=LEGEND_FONT_SIZE)

        if has_bwd:
            ax = axes[ax_idx]
            bwd_df = df[df["bwd_time_ms"] > 0]
            _plot_grouped_bars(ax, bwd_df, "norm_label", "bwd_time_ms", "backend_display", palette)
            ax.set_xlabel("Norm Configuration", fontsize=LABEL_FONT_SIZE)
            ax.set_ylabel("Time (ms)", fontsize=LABEL_FONT_SIZE)
            ax.set_title(f"{config.name} Norm Backward (Time)", fontsize=TITLE_FONT_SIZE)
            ax.legend(title="Backend", fontsize=LEGEND_FONT_SIZE)

        plt.tight_layout()
        time_path = output_dir / f"{config.name}_time.png"
        plt.savefig(time_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths.append(time_path)
        logger.info(f"Chart saved to {time_path}")

    # Chart 2: Bandwidth (GB/s)
    if has_fwd or has_bwd:
        ncols = (1 if has_fwd else 0) + (1 if has_bwd else 0)
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6), dpi=150)
        if ncols == 1:
            axes = [axes]

        ax_idx = 0
        if has_fwd:
            ax = axes[ax_idx]
            ax_idx += 1
            fwd_df = df[df["fwd_gbps"] > 0]
            _plot_grouped_bars(ax, fwd_df, "norm_label", "fwd_gbps", "backend_display", palette)
            ax.set_xlabel("Norm Configuration", fontsize=LABEL_FONT_SIZE)
            ax.set_ylabel("Bandwidth (GB/s)", fontsize=LABEL_FONT_SIZE)
            ax.set_title(f"{config.name} Norm Forward (Bandwidth)", fontsize=TITLE_FONT_SIZE)
            ax.legend(title="Backend", fontsize=LEGEND_FONT_SIZE)

        if has_bwd:
            ax = axes[ax_idx]
            bwd_df = df[df["bwd_gbps"] > 0]
            _plot_grouped_bars(ax, bwd_df, "norm_label", "bwd_gbps", "backend_display", palette)
            ax.set_xlabel("Norm Configuration", fontsize=LABEL_FONT_SIZE)
            ax.set_ylabel("Bandwidth (GB/s)", fontsize=LABEL_FONT_SIZE)
            ax.set_title(f"{config.name} Norm Backward (Bandwidth)", fontsize=TITLE_FONT_SIZE)
            ax.legend(title="Backend", fontsize=LEGEND_FONT_SIZE)

        plt.tight_layout()
        bw_path = output_dir / f"{config.name}_bandwidth.png"
        plt.savefig(bw_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths.append(bw_path)
        logger.info(f"Chart saved to {bw_path}")

    return saved_paths


def _plot_grouped_bars(ax, df, x_col, y_col, hue_col, palette):
    """Plot grouped bar chart."""
    import numpy as np

    labels = df[x_col].unique()
    hue_vals = df[hue_col].unique()
    n_groups = len(labels)
    n_bars = len(hue_vals)
    bar_width = 0.8 / max(n_bars, 1)

    x = np.arange(n_groups)

    for j, hue in enumerate(hue_vals):
        vals = []
        for label in labels:
            subset = df[(df[x_col] == label) & (df[hue_col] == hue)]
            vals.append(subset[y_col].values[0] if len(subset) > 0 else 0)
        offset = (j - n_bars / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            vals,
            bar_width,
            label=hue,
            color=palette.get(hue, "gray"),
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=BAR_LABEL_FONT_SIZE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
