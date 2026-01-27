"""
Chart generation for SDPA benchmark results.

Generates comparison bar charts showing backend performance side-by-side.
"""

from pathlib import Path
from typing import Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    import pandas as pd
    from .config_types import BenchmarkConfig

logger = logging.getLogger(__name__)

# Backend display configuration
# Each backend has a base color; FP8 variants get a darker/different shade
BACKEND_CONFIG = {
    "cudnn": {"name": "cudnn", "color": "#76b900", "color_fp8": "#4a7500", "order": 0},
    "pyt_cudnn": {"name": "cuDNN (PyTorch)", "color": "#90EE90", "color_fp8": "#228B22", "order": 1},
    "pyt_flash_attention": {"name": "FAv2 (PyTorch)", "color": "#6495ED", "color_fp8": "#0000CD", "order": 2},
    "pyt_efficient_attention": {"name": "xFormers (PyTorch)", "color": "#FF00FF", "color_fp8": "#8B008B", "order": 3},
    "pyt_math": {"name": "Standard Attention", "color": "#FF8C00", "color_fp8": "#D2691E", "order": 4},
    "flash_attention": {"name": "FAv2 (Native)", "color": "#F08080", "color_fp8": "#CD5C5C", "order": 5},
    "flash_attention_3": {"name": "FAv3", "color": "#FFA500", "color_fp8": "#FF6600", "order": 6},
    "flash_attention_4": {"name": "FAv4", "color": "#FFD700", "color_fp8": "#DAA520", "order": 7},
}

# Font sizes for plot elements
LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 8
TITLE_FONT_SIZE = 12
BAR_LABEL_FONT_SIZE = 6


def get_backend_display_name(backend: str, data_type: str) -> str:
    """
    Get display name for backend+dtype combination.

    Args:
        backend: Backend name (e.g., "cudnn")
        data_type: Data type (e.g., "bfloat16", "fp8")

    Returns:
        Display name for legend (e.g., "cuDNN FE (FP8)")
    """
    base_name = BACKEND_CONFIG.get(backend, {}).get("name", backend)
    if data_type == "fp8":
        return f"{base_name} (FP8)"
    elif data_type == "float16":
        return f"{base_name} (FP16)"
    return base_name


def get_backend_color(backend: str, data_type: str) -> str:
    """
    Get color for backend+dtype combination.

    Args:
        backend: Backend name
        data_type: Data type

    Returns:
        Color string for matplotlib
    """
    config = BACKEND_CONFIG.get(backend, {})
    if data_type == "fp8" and "color_fp8" in config:
        return config["color_fp8"]
    return config.get("color", "gray")


def generate_comparison_chart(
    df: "pd.DataFrame",
    config: "BenchmarkConfig",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate comparison bar chart with multiple backends side-by-side.

    Creates a figure with:
    - Left subplot: Forward pass TFLOPS by configuration
    - Right subplot: Backward pass TFLOPS by configuration
    - Each backend+dtype combo as a separate bar group

    Args:
        df: DataFrame with benchmark results (from BenchmarkRunner.results_to_dataframe)
        config: BenchmarkConfig used for the run
        output_path: Optional path for output file. If None, uses config.output_dir

    Returns:
        Path to the saved chart file
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Filter to successful results only
    df = df[df["success"] == True].copy()

    if df.empty:
        raise ValueError("No successful results to plot")

    # Create backend+dtype display name for legend
    df["backend_display"] = df.apply(lambda r: get_backend_display_name(r["backend"], r["data_type"]), axis=1)

    # Create config label for x-axis (model/seqlen/mask)
    df["config_label"] = df.apply(
        lambda r: f"{r['model_name']}\n{r['q_seqlen']}x{r['kv_seqlen']}\n{r['attn_mask']}",
        axis=1,
    )

    # Sort by backend order for consistent legend
    df["backend_order"] = df["backend"].map(lambda b: BACKEND_CONFIG.get(b, {}).get("order", 99))
    df.sort_values(["model_name", "q_seqlen", "attn_mask", "backend_order"], inplace=True)

    # Build color palette based on unique backend+dtype combinations
    # Get unique (backend, data_type, backend_display) tuples to map colors correctly
    unique_combos = df[["backend", "data_type", "backend_display"]].drop_duplicates()
    palette = {}
    for _, row in unique_combos.iterrows():
        palette[row["backend_display"]] = get_backend_color(row["backend"], row["data_type"])

    # Determine if we have fwd/bwd data
    has_fwd = (df["fwd_tflops"] > 0).any()
    has_bwd = (df["bwd_tflops"] > 0).any()

    if has_fwd and has_bwd:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
        ax_fwd, ax_bwd = axes
    elif has_fwd:
        fig, ax_fwd = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
        ax_bwd = None
    elif has_bwd:
        fig, ax_bwd = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
        ax_fwd = None
    else:
        raise ValueError("No forward or backward TFLOPS data to plot")

    # Calculate y-axis limit
    max_tflops = max(
        df["fwd_tflops"].max() if has_fwd else 0,
        df["bwd_tflops"].max() if has_bwd else 0,
    )
    ylim_max = max_tflops * 1.15  # Add 15% headroom for labels

    # Plot forward pass
    if ax_fwd is not None:
        fwd_df = df[df["fwd_tflops"] > 0]
        if not fwd_df.empty:
            sns.barplot(
                data=fwd_df,
                x="config_label",
                y="fwd_tflops",
                hue="backend_display",
                ax=ax_fwd,
                palette=palette,
                edgecolor="black",
                linewidth=0.5,
            )
            ax_fwd.set_xlabel("Configuration", fontsize=LABEL_FONT_SIZE)
            ax_fwd.set_ylabel("TFLOPS", fontsize=LABEL_FONT_SIZE)
            ax_fwd.set_title("SDPA Forward Pass", fontsize=TITLE_FONT_SIZE)
            ax_fwd.legend(title="Backend", fontsize=LEGEND_FONT_SIZE)
            ax_fwd.tick_params(axis="x", rotation=45, labelsize=8)
            ax_fwd.tick_params(axis="y", labelsize=LABEL_FONT_SIZE)
            ax_fwd.set_ylim(0, ylim_max)

            # Add value labels on bars
            for container in ax_fwd.containers:
                ax_fwd.bar_label(container, fmt="%.0f", fontsize=BAR_LABEL_FONT_SIZE)

    # Plot backward pass
    if ax_bwd is not None:
        bwd_df = df[df["bwd_tflops"] > 0]
        if not bwd_df.empty:
            sns.barplot(
                data=bwd_df,
                x="config_label",
                y="bwd_tflops",
                hue="backend_display",
                ax=ax_bwd,
                palette=palette,
                edgecolor="black",
                linewidth=0.5,
            )
            ax_bwd.set_xlabel("Configuration", fontsize=LABEL_FONT_SIZE)
            ax_bwd.set_ylabel("TFLOPS", fontsize=LABEL_FONT_SIZE)
            ax_bwd.set_title("SDPA Backward Pass", fontsize=TITLE_FONT_SIZE)
            ax_bwd.legend(title="Backend", fontsize=LEGEND_FONT_SIZE)
            ax_bwd.tick_params(axis="x", rotation=45, labelsize=8)
            ax_bwd.tick_params(axis="y", labelsize=LABEL_FONT_SIZE)
            ax_bwd.set_ylim(0, ylim_max)

            # Add value labels on bars
            for container in ax_bwd.containers:
                ax_bwd.bar_label(container, fmt="%.0f", fontsize=BAR_LABEL_FONT_SIZE)

    plt.tight_layout()

    # Determine output path
    if output_path is None:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{config.name}_comparison.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Chart saved to {output_path}")
    return output_path


def generate_charts_by_mask(
    df: "pd.DataFrame",
    config: "BenchmarkConfig",
    output_dir: Optional[Path] = None,
) -> list:
    """
    Generate separate charts for each mask type.

    This creates cleaner charts when benchmarking both causal and non-causal masks.
    Each chart shows seqlen on x-axis and backends as grouped bars.

    Args:
        df: DataFrame with benchmark results
        config: BenchmarkConfig used for the run
        output_dir: Directory for output files

    Returns:
        List of paths to saved chart files
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = df[df["success"] == True].copy()

    if df.empty:
        raise ValueError("No successful results to plot")

    if output_dir is None:
        output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    masks = df["attn_mask"].unique()

    for mask in masks:
        mask_df = df[df["attn_mask"] == mask].copy()

        # Create display names
        mask_df["backend_display"] = mask_df.apply(lambda r: get_backend_display_name(r["backend"], r["data_type"]), axis=1)
        mask_df["seqlen_label"] = mask_df.apply(lambda r: f"{r['q_seqlen']}x{r['kv_seqlen']}", axis=1)

        # Build palette
        unique_combos = mask_df[["backend", "data_type", "backend_display"]].drop_duplicates()
        palette = {}
        for _, row in unique_combos.iterrows():
            palette[row["backend_display"]] = get_backend_color(row["backend"], row["data_type"])

        # Sort
        mask_df["backend_order"] = mask_df["backend"].map(lambda b: BACKEND_CONFIG.get(b, {}).get("order", 99))
        mask_df.sort_values(["q_seqlen", "backend_order"], inplace=True)

        has_fwd = (mask_df["fwd_tflops"] > 0).any()
        has_bwd = (mask_df["bwd_tflops"] > 0).any()

        if has_fwd and has_bwd:
            fig, (ax_fwd, ax_bwd) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
        elif has_fwd:
            fig, ax_fwd = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
            ax_bwd = None
        else:
            fig, ax_bwd = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
            ax_fwd = None

        mask_title = "Causal" if mask == "top_left" else "Non-Causal" if mask == "no_mask" else mask

        if ax_fwd is not None:
            fwd_df = mask_df[mask_df["fwd_tflops"] > 0]
            if not fwd_df.empty:
                sns.barplot(
                    data=fwd_df,
                    x="seqlen_label",
                    y="fwd_tflops",
                    hue="backend_display",
                    ax=ax_fwd,
                    palette=palette,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax_fwd.set_xlabel("Sequence Length", fontsize=LABEL_FONT_SIZE)
                ax_fwd.set_ylabel("TFLOPS", fontsize=LABEL_FONT_SIZE)
                ax_fwd.set_title(f"{config.name} Forward ({mask_title})", fontsize=TITLE_FONT_SIZE)
                ax_fwd.legend(title="Backend", fontsize=LEGEND_FONT_SIZE)
                ax_fwd.tick_params(axis="x", rotation=45)
                for container in ax_fwd.containers:
                    ax_fwd.bar_label(container, fmt="%.0f", fontsize=BAR_LABEL_FONT_SIZE)

        if ax_bwd is not None:
            bwd_df = mask_df[mask_df["bwd_tflops"] > 0]
            if not bwd_df.empty:
                sns.barplot(
                    data=bwd_df,
                    x="seqlen_label",
                    y="bwd_tflops",
                    hue="backend_display",
                    ax=ax_bwd,
                    palette=palette,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax_bwd.set_xlabel("Sequence Length", fontsize=LABEL_FONT_SIZE)
                ax_bwd.set_ylabel("TFLOPS", fontsize=LABEL_FONT_SIZE)
                ax_bwd.set_title(f"{config.name} Backward ({mask_title})", fontsize=TITLE_FONT_SIZE)
                ax_bwd.legend(title="Backend", fontsize=LEGEND_FONT_SIZE)
                ax_bwd.tick_params(axis="x", rotation=45)
                for container in ax_bwd.containers:
                    ax_bwd.bar_label(container, fmt="%.0f", fontsize=BAR_LABEL_FONT_SIZE)

        plt.tight_layout()
        output_path = output_dir / f"{config.name}_{mask}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths.append(output_path)
        logger.info(f"Chart saved to {output_path}")

    return saved_paths


def generate_seqlen_scaling_chart(
    df: "pd.DataFrame",
    config: "BenchmarkConfig",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate a chart showing performance scaling with sequence length.

    This chart is useful when benchmarking multiple sequence lengths with
    the same model configuration.

    Args:
        df: DataFrame with benchmark results
        config: BenchmarkConfig used for the run
        output_path: Optional path for output file

    Returns:
        Path to the saved chart file
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Filter to successful results only
    df = df[df["success"] == True].copy()

    if df.empty:
        raise ValueError("No successful results to plot")

    # Create backend+dtype display name
    df["backend_display"] = df.apply(lambda r: get_backend_display_name(r["backend"], r["data_type"]), axis=1)

    # Use q_seqlen for x-axis (assuming symmetric seqlens for this chart)
    df["seqlen"] = df["q_seqlen"]

    # Build color palette based on unique backend+dtype combinations
    unique_combos = df[["backend", "data_type", "backend_display"]].drop_duplicates()
    palette = {}
    for _, row in unique_combos.iterrows():
        palette[row["backend_display"]] = get_backend_color(row["backend"], row["data_type"])

    # Create figure
    has_fwd = (df["fwd_tflops"] > 0).any()
    has_bwd = (df["bwd_tflops"] > 0).any()

    if has_fwd and has_bwd:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
        ax_fwd, ax_bwd = axes
    elif has_fwd:
        fig, ax_fwd = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
        ax_bwd = None
    else:
        fig, ax_bwd = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
        ax_fwd = None

    # Plot forward
    if ax_fwd is not None and has_fwd:
        fwd_df = df[df["fwd_tflops"] > 0]
        sns.barplot(
            data=fwd_df,
            x="seqlen",
            y="fwd_tflops",
            hue="backend_display",
            ax=ax_fwd,
            palette=palette,
            edgecolor="black",
            linewidth=0.5,
        )
        ax_fwd.set_xlabel("Sequence Length", fontsize=LABEL_FONT_SIZE)
        ax_fwd.set_ylabel("TFLOPS", fontsize=LABEL_FONT_SIZE)
        ax_fwd.set_title("SDPA Forward Pass", fontsize=TITLE_FONT_SIZE)
        ax_fwd.legend(title="Backend", fontsize=LEGEND_FONT_SIZE)
        ax_fwd.tick_params(axis="x", rotation=45)

        for container in ax_fwd.containers:
            ax_fwd.bar_label(container, fmt="%.0f", fontsize=BAR_LABEL_FONT_SIZE)

    # Plot backward
    if ax_bwd is not None and has_bwd:
        bwd_df = df[df["bwd_tflops"] > 0]
        sns.barplot(
            data=bwd_df,
            x="seqlen",
            y="bwd_tflops",
            hue="backend_display",
            ax=ax_bwd,
            palette=palette,
            edgecolor="black",
            linewidth=0.5,
        )
        ax_bwd.set_xlabel("Sequence Length", fontsize=LABEL_FONT_SIZE)
        ax_bwd.set_ylabel("TFLOPS", fontsize=LABEL_FONT_SIZE)
        ax_bwd.set_title("SDPA Backward Pass", fontsize=TITLE_FONT_SIZE)
        ax_bwd.legend(title="Backend", fontsize=LEGEND_FONT_SIZE)
        ax_bwd.tick_params(axis="x", rotation=45)

        for container in ax_bwd.containers:
            ax_bwd.bar_label(container, fmt="%.0f", fontsize=BAR_LABEL_FONT_SIZE)

    plt.tight_layout()

    # Determine output path
    if output_path is None:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{config.name}_seqlen_scaling.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Chart saved to {output_path}")
    return output_path
