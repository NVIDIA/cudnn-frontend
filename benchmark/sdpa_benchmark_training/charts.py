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
    "cudnn": {"name": "cudnn", "color": "#76b900", "color_fp8": "#4a7500", "color_mxfp8": "#2d5a00", "order": 0},
    "pyt_cudnn": {"name": "cuDNN (PyTorch)", "color": "#90EE90", "color_fp8": "#228B22", "color_mxfp8": "#006400", "order": 1},
    "pyt_flash_attention": {"name": "FAv2 (PyTorch)", "color": "#6495ED", "color_fp8": "#0000CD", "color_mxfp8": "#00008B", "order": 2},
    "pyt_efficient_attention": {"name": "xFormers (PyTorch)", "color": "#FF00FF", "color_fp8": "#8B008B", "color_mxfp8": "#4B0082", "order": 3},
    "pyt_math": {"name": "Standard Attention", "color": "#FF8C00", "color_fp8": "#D2691E", "color_mxfp8": "#8B4513", "order": 4},
    "flash_attention": {"name": "FAv2 (Native)", "color": "#F08080", "color_fp8": "#CD5C5C", "color_mxfp8": "#8B0000", "order": 5},
    "flash_attention_3": {"name": "FAv3", "color": "#FFA500", "color_fp8": "#FF6600", "color_mxfp8": "#CC5200", "order": 6},
    "flash_attention_4": {"name": "FAv4", "color": "#FFD700", "color_fp8": "#DAA520", "color_mxfp8": "#B8860B", "order": 7},
}

# Font sizes for plot elements
LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 8
TITLE_FONT_SIZE = 12
BAR_LABEL_FONT_SIZE = 6


def _get_model_info(config: "BenchmarkConfig") -> str:
    """Build a concise model info string for chart titles (heads and dims)."""
    if len(config.models) != 1:
        return ""
    m = config.models[0]
    dim = f"{m.head_dim_qk}" if m.head_dim_qk == m.head_dim_vo else f"{m.head_dim_qk}/{m.head_dim_vo}"
    return f", heads={m.num_q_heads}/{m.num_kv_heads}, d={dim}"


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
    elif data_type == "mxfp8":
        return f"{base_name} (MXFP8)"
    elif data_type == "float16":
        return f"{base_name} (FP16)"
    elif data_type == "bfloat16":
        return f"{base_name} (BF16)"
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
    if data_type == "mxfp8" and "color_mxfp8" in config:
        return config["color_mxfp8"]
    return config.get("color", "gray")


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
                model_info = _get_model_info(config)
                ax_fwd.set_title(f"{config.name} Forward ({mask_title}, batch={config.batch_size}{model_info})", fontsize=TITLE_FONT_SIZE)
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
                model_info = _get_model_info(config)
                ax_bwd.set_title(f"{config.name} Backward ({mask_title}, batch={config.batch_size}{model_info})", fontsize=TITLE_FONT_SIZE)
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
