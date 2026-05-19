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

# Per-seqlen bar grouping order. Bars are placed by (dtype_order, backend_order)
# so the display reads: cuDNN BF16, FAv4 BF16, cuDNN MXFP8, cuDNN FP8.
DTYPE_ORDER = {
    "bfloat16": 0,
    "float16": 0,  # same bucket as bf16
    "mxfp8": 1,
    "fp8": 2,
}


def _get_model_info(config: "BenchmarkConfig") -> str:
    """Build a concise model info string for chart titles (heads and dims)."""
    if len(config.models) != 1:
        return ""
    m = config.models[0]
    dim = f"{m.head_dim_qk}" if m.head_dim_qk == m.head_dim_vo else f"{m.head_dim_qk}/{m.head_dim_vo}"
    return f", heads={m.num_q_heads}/{m.num_kv_heads}, d={dim}"


def _format_cudnn_backend_version(v) -> str:
    """92200 -> '9.22.0'. Returns None for missing/invalid values."""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    return f"{n // 10000}.{(n % 10000) // 100}.{n % 100}"


def get_backend_display_name(backend: str, data_type: str, cudnn_backend_version=None) -> str:
    """
    Get display name for backend+dtype combination.

    Args:
        backend: Backend name (e.g., "cudnn")
        data_type: Data type (e.g., "bfloat16", "fp8")
        cudnn_backend_version: Integer version from cudnn.backend_version()
            (e.g. 92200 -> 9.22.0). Appended to the display name for cuDNN rows.

    Returns:
        Display name for legend (e.g., "cudnn 9.22.0 (FP8)")
    """
    base_name = BACKEND_CONFIG.get(backend, {}).get("name", backend)
    if backend == "cudnn":
        v = _format_cudnn_backend_version(cudnn_backend_version)
        if v:
            base_name = f"{base_name} {v}"
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
    df = df[df["backend"] != "flash_attention_4"].copy()

    # Main charts show only non-deterministic mode; det-vs-nondet comparison
    # is in generate_det_overhead_charts. Fwd rows always carry det=False, so
    # this filter only drops the det=True bwd rows.
    df = df[df["deterministic_bwd"].astype(str).str.lower() == "false"].copy()

    # Drop SWA fp8/mxfp8 backward rows: cuDNN's fp8/mxfp8 bwd has no
    # SWA-aware kernel and falls back to a dense path that scales O(N^2)
    # instead of O(N*W). The result is misleading single-digit TFLOPS bars
    # next to bf16 bwd's correct linear scaling. Forward fp8/mxfp8 honors
    # SWA and is kept.
    swa_bad = (df["sliding_window_size"].fillna(0).astype(float) > 0) & (df["profile_pass"] == "bwd") & (df["data_type"].isin(["fp8", "mxfp8"]))
    df = df[~swa_bad].copy()

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
        mask_df["backend_display"] = mask_df.apply(
            lambda r: get_backend_display_name(
                r["backend"],
                r["data_type"],
                r.get("cudnn_backend_version") if r["backend"] == "cudnn" else None,
            ),
            axis=1,
        )
        mask_df["seqlen_label"] = mask_df.apply(lambda r: f"{r['q_seqlen']}x{r['kv_seqlen']}", axis=1)

        # Build palette
        unique_combos = mask_df[["backend", "data_type", "backend_display"]].drop_duplicates()
        palette = {}
        for _, row in unique_combos.iterrows():
            palette[row["backend_display"]] = get_backend_color(row["backend"], row["data_type"])

        # Sort by (seqlen, dtype_order, backend_order). Within each seqlen
        # group, bars are grouped by dtype (BF16 first, then MXFP8, then FP8),
        # and within each dtype bars are ordered by backend_order.
        mask_df["backend_order"] = mask_df["backend"].map(lambda b: BACKEND_CONFIG.get(b, {}).get("order", 99))
        mask_df["dtype_order"] = mask_df["data_type"].map(lambda d: DTYPE_ORDER.get(d, 99))
        mask_df.sort_values(["q_seqlen", "dtype_order", "backend_order"], inplace=True)

        # Build the hue order with the same precedence so seaborn lays out
        # the grouped bars and the legend in this exact order.
        hue_rows = (
            mask_df[["backend", "data_type", "backend_display", "dtype_order", "backend_order"]].drop_duplicates().sort_values(["dtype_order", "backend_order"])
        )
        hue_order = list(hue_rows["backend_display"])

        # Split rows by pass. Under the new schema each row has a single
        # time_ms/tflops tagged by profile_pass.
        fwd_df = mask_df[(mask_df["profile_pass"] == "fwd") & (mask_df["tflops"] > 0)]
        bwd_df = mask_df[(mask_df["profile_pass"] == "bwd") & (mask_df["tflops"] > 0)]
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

        mask_title = "Causal" if mask == "top_left" else "Non-Causal" if mask == "no_mask" else mask

        model_info = _get_model_info(config)
        suptitle = f"{config.name} — {mask_title}, batch={config.batch_size}{model_info} (non-deterministic)"
        fig.suptitle(suptitle, fontsize=TITLE_FONT_SIZE)

        if ax_fwd is not None and has_fwd:
            sns.barplot(
                data=fwd_df,
                x="seqlen_label",
                y="tflops",
                hue="backend_display",
                hue_order=hue_order,
                ax=ax_fwd,
                palette=palette,
                edgecolor="black",
                linewidth=0.5,
                errorbar=None,
            )
            ax_fwd.set_xlabel("Sequence Length", fontsize=LABEL_FONT_SIZE)
            ax_fwd.set_ylabel("TFLOPS", fontsize=LABEL_FONT_SIZE)
            ax_fwd.set_title("Forward", fontsize=TITLE_FONT_SIZE)
            ax_fwd.legend(title="Backend", fontsize=LEGEND_FONT_SIZE)
            ax_fwd.tick_params(axis="x", rotation=45)
            for container in ax_fwd.containers:
                ax_fwd.bar_label(container, fmt="%.0f", fontsize=BAR_LABEL_FONT_SIZE)

        if ax_bwd is not None and has_bwd:
            sns.barplot(
                data=bwd_df,
                x="seqlen_label",
                y="tflops",
                hue="backend_display",
                hue_order=hue_order,
                ax=ax_bwd,
                palette=palette,
                edgecolor="black",
                linewidth=0.5,
                errorbar=None,
            )
            ax_bwd.set_xlabel("Sequence Length", fontsize=LABEL_FONT_SIZE)
            ax_bwd.set_ylabel("TFLOPS", fontsize=LABEL_FONT_SIZE)
            ax_bwd.set_title("Backward", fontsize=TITLE_FONT_SIZE)
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


def generate_det_overhead_charts(
    df: "pd.DataFrame",
    config: "BenchmarkConfig",
    output_dir: Optional[Path] = None,
) -> list:
    """
    For each (config, mask), emit a bwd-only bar chart showing raw TFLOPS for
    both deterministic and non-deterministic modes, side-by-side per seqlen.
    Restricted to bf16 on the two backends that actually compete (cuDNN and
    FAv4); FP8/MXFP8 are cuDNN-only and off-topic for this comparison.

    Only emits a chart if at least one seqlen has both det modes present.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    bwd = df[
        (df["success"] == True) & (df["profile_pass"] == "bwd") & (df["backend"].isin(["cudnn", "flash_attention_4"])) & (df["data_type"] == "bfloat16")
    ].copy()
    if bwd.empty:
        return []

    # deterministic_bwd is stored as string "True"/"False" after CSV roundtrip
    bwd["det_flag"] = bwd["deterministic_bwd"].astype(str).str.lower()
    if not {"true", "false"}.issubset(set(bwd["det_flag"].unique())):
        return []  # need both det modes

    # Build a unified hue label so each bar reads e.g. "cudnn 9.22.0 (BF16) det"
    def _label(row) -> str:
        base = get_backend_display_name(
            row["backend"],
            row["data_type"],
            row.get("cudnn_backend_version") if row["backend"] == "cudnn" else None,
        )
        suffix = "det" if row["det_flag"] == "true" else "nondet"
        return f"{base} {suffix}"

    bwd["backend_display"] = bwd.apply(_label, axis=1)
    bwd["seqlen_label"] = bwd.apply(lambda r: f"{r['q_seqlen']}x{r['kv_seqlen']}", axis=1)
    bwd["backend_order"] = bwd["backend"].map(lambda b: BACKEND_CONFIG.get(b, {}).get("order", 99))
    # Order: within each (backend, dtype) pair, nondet comes first then det.
    bwd["det_order"] = bwd["det_flag"].map({"false": 0, "true": 1})

    if output_dir is None:
        output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Color: keep backend base color; use a darker tone for det to distinguish.
    palette = {}
    for _, row in bwd[["backend", "data_type", "backend_display", "det_flag"]].drop_duplicates().iterrows():
        base = get_backend_color(row["backend"], row["data_type"])
        if row["det_flag"] == "true":
            # darken the base color slightly for det bars
            from matplotlib.colors import to_rgb

            r, g, b = to_rgb(base)
            palette[row["backend_display"]] = (r * 0.65, g * 0.65, b * 0.65)
        else:
            palette[row["backend_display"]] = base

    saved = []
    for mask in sorted(bwd["attn_mask"].unique()):
        sub = bwd[bwd["attn_mask"] == mask].copy()
        if sub.empty:
            continue
        sub.sort_values(["q_seqlen", "backend_order", "det_order"], inplace=True)
        hue_rows = sub[["backend_display", "backend_order", "det_order"]].drop_duplicates().sort_values(["backend_order", "det_order"])
        hue_order = list(hue_rows["backend_display"])

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        sns.barplot(
            data=sub,
            x="seqlen_label",
            y="tflops",
            hue="backend_display",
            hue_order=hue_order,
            ax=ax,
            palette=palette,
            edgecolor="black",
            linewidth=0.5,
            errorbar=None,
        )

        mask_title = "Causal" if mask == "top_left" else "Non-Causal" if mask == "no_mask" else mask
        model_info = _get_model_info(config)
        ax.set_xlabel("Sequence Length", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("TFLOPS", fontsize=LABEL_FONT_SIZE)
        ax.set_title(
            f"{config.name} Backward det vs non-det ({mask_title}, batch={config.batch_size}{model_info})",
            fontsize=TITLE_FONT_SIZE,
        )
        ax.legend(title="Backend", fontsize=LEGEND_FONT_SIZE)
        ax.tick_params(axis="x", rotation=45)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", fontsize=BAR_LABEL_FONT_SIZE)

        plt.tight_layout()
        out = output_dir / f"{config.name}_{mask}_det_overhead.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(out)
        logger.info(f"Det chart saved to {out}")

    return saved
