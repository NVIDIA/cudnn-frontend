# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chart generation for DSA sparse attention benchmark results.

Two figures per config:

- ``<config>.webp`` (legacy, unchanged): forward and backward TFLOPS side by
  side for the core sparse attention over a flat bf16 pool, sequence length
  on the x axis, one bar per (model, dtype, deterministic) series. A dashed
  line marks the dense-MMA peak recorded by the benchmark (median of the
  per-row ``peak_mma_tflops``) so sparse-gather efficiency reads against the
  same SOL the dense suites use.
- ``<config>_extended.webp``: one panel per (component, phase) that has any
  case beyond that legacy set (a storage format, a paged pool, an indexer
  row, a generation shape). x = shape (``{q}x{kv}`` or ``B{B} q{q}
  kv{kv}``), hue = ``{kv_dtype}[ paged][ hierarchical]``, y = TFLOPS for
  core context / bprop and milliseconds otherwise. Every enumerated cell
  owns an x slot: a skipped or failed cell is drawn as a hatched blank
  marker, so coverage gaps are visible instead of silently dropped.

Frames with no success rows never raise: the legacy figure is skipped with
a log line and the extended figure still draws the coverage grid.
"""

import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

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

_DTYPE_TAG = {"bfloat16": "BF16", "float16": "FP16", "mxfp8": "MXFP8", "nvfp4": "NVFP4"}

# Appended BenchmarkResult columns with the value a legacy CSV row implies.
_AXIS_DEFAULTS = {"component": "core", "phase": "context", "kv_dtype": "bfloat16", "paged": False, "batch_size": 1, "indexer_mode": None}

# Height of a hatched blank marker as a fraction of the panel's tallest bar.
_BLANK_FRACTION = 0.04


def _fmt_len(s: int) -> str:
    return f"{s // 1024}k" if s >= 1024 and s % 1024 == 0 else str(s)


def _series_label(row, with_head_dim: bool = False) -> str:
    dtype = _DTYPE_TAG.get(row["data_type"], row["data_type"])
    det = " det" if str(row["deterministic_bwd"]).lower() == "true" else ""
    dim = f" d={row['head_dim_qk']}" if with_head_dim else ""
    return f"{row['model_name']} H{row['num_q_heads']}{dim} K={row['topk']} ({dtype}){det}"


def _truthy(series) -> "pd.Series":
    """Boolean view of a column that may hold bools or CSV strings ('True'/'False')."""
    return series.astype(str).str.lower().isin(("true", "1"))


def _with_axis_defaults(df: "pd.DataFrame") -> "pd.DataFrame":
    """Add the appended axis columns a legacy CSV lacks, with their legacy values."""
    df = df.copy()
    if "phase" not in df.columns:
        # Rows written before the column existed: today's fwd/bwd passes.
        df["phase"] = df["profile_pass"].map({"fwd": "context", "bwd": "bprop"})
    for col, default in _AXIS_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    df["paged"] = _truthy(df["paged"])
    df["success"] = _truthy(df["success"])
    df["indexer_mode"] = df["indexer_mode"].where(df["indexer_mode"].notna(), None)
    return df


def _legacy_mask(df: "pd.DataFrame"):
    return (df["component"] == "core") & (df["kv_dtype"] == "bfloat16") & (~df["paged"]) & df["phase"].isin(("context", "bprop"))


def generate_charts(
    df: "pd.DataFrame",
    config: "DsaBenchmarkConfig",
    output_dir: Optional[Path] = None,
    peak_tflops: Optional[float] = None,
) -> list:
    """Save ``<config>.webp`` (legacy rows) and ``<config>_extended.webp`` (appended axes); returns saved paths.

    Never raises on a frame with no success rows: each figure that has
    nothing to show is logged and left out.
    """
    if output_dir is None:
        output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df is None or df.empty:
        logger.warning("No results to plot")
        return []
    df = _with_axis_defaults(df)
    legacy = _legacy_mask(df)

    paths: List[Path] = []
    legacy_path = _legacy_figure(df[legacy], config, output_dir, peak_tflops)
    if legacy_path is not None:
        paths.append(legacy_path)
    extended_path = _extended_figure(df, df[~legacy], config, output_dir)
    if extended_path is not None:
        paths.append(extended_path)
    return paths


def _legacy_figure(df: "pd.DataFrame", config: "DsaBenchmarkConfig", output_dir: Path, peak_tflops: Optional[float]) -> Optional[Path]:
    """The fwd/bwd TFLOPS figure, byte-for-byte today's chart for legacy rows."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = df[(df["success"] == True) & (df["tflops"] > 0)].copy()
    if df.empty:
        logger.warning(f"No successful core bf16 flat rows to plot for '{config.name}'; legacy chart skipped")
        return None

    if peak_tflops is None and "peak_mma_tflops" in df.columns:
        ok = df[df["peak_mma_tflops"].notna() & (df["peak_mma_tflops"] > 0)]
        peak_tflops = float(ok["peak_mma_tflops"].median()) if not ok.empty else None

    head_dims = sorted(int(d) for d in df["head_dim_qk"].unique())
    df["series"] = df.apply(lambda r: _series_label(r, with_head_dim=len(head_dims) > 1), axis=1)
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
    fig.suptitle(f"{config.name} — DSA sparse attention (shared K=V, d={'/'.join(map(str, head_dims))})", fontsize=TITLE_FONT_SIZE)

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
    return output_path


# ---------------------------------------------------------------------- extended figure

_PHASE_ORDER = {"context": 0, "generation": 1, "bprop": 2}
_COMPONENT_ORDER = {"core": 0, "indexer": 1}
_KV_DTYPE_ORDER = {"bfloat16": 0, "float16": 0, "mxfp8": 1, "nvfp4": 2}


def _shape_label(row) -> str:
    if row["phase"] == "generation":
        return f"B{int(row['batch_size'])} q{int(row['q_seqlen'])} kv{_fmt_len(int(row['kv_seqlen']))}"
    return f"{_fmt_len(int(row['q_seqlen']))}x{_fmt_len(int(row['kv_seqlen']))}"


def _shape_key(row):
    if row["phase"] == "generation":
        return (int(row["kv_seqlen"]), int(row["batch_size"]), int(row["q_seqlen"]))
    return (int(row["q_seqlen"]), int(row["kv_seqlen"]), 0)


def _hue_label(row, with_model: bool) -> str:
    tag = _DTYPE_TAG.get(row["kv_dtype"], str(row["kv_dtype"]))
    if row["paged"]:
        tag += " paged"
    mode = row["indexer_mode"]
    if isinstance(mode, str) and mode not in ("", "dense"):
        tag += f" {mode}"
    if str(row["deterministic_bwd"]).lower() == "true":
        tag += " det"
    return f"{row['model_name']} {tag}" if with_model else tag


def _hue_key(row):
    mode = row["indexer_mode"] if isinstance(row["indexer_mode"], str) else ""
    return (str(row["model_name"]), _KV_DTYPE_ORDER.get(row["kv_dtype"], 9), int(bool(row["paged"])), mode, str(row["deterministic_bwd"]).lower())


def _panel_metric(component: str, phase: str):
    if component == "core" and phase in ("context", "bprop"):
        return "tflops", "TFLOPS (on gathered rows)"
    return "time_ms", "time (ms)"


def _extended_figure(df: "pd.DataFrame", beyond: "pd.DataFrame", config: "DsaBenchmarkConfig", output_dir: Path) -> Optional[Path]:
    """One panel per (component, phase) with any case beyond the legacy set; skipped cells hatched."""
    import matplotlib.pyplot as plt
    import numpy as np

    if beyond.empty:
        return None
    panel_keys = sorted(
        {(c, p) for c, p in zip(beyond["component"], beyond["phase"])}, key=lambda k: (_COMPONENT_ORDER.get(k[0], 9), _PHASE_ORDER.get(k[1], 9))
    )
    if not panel_keys:
        return None

    ncols = min(3, len(panel_keys))
    nrows = -(-len(panel_keys) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 5.5 * nrows), dpi=150, squeeze=False)
    fig.suptitle(f"{config.name} — DSA sparse attention, extended axes (hatched = skipped or failed cell)", fontsize=TITLE_FONT_SIZE)
    successes = 0

    for ax, (component, phase) in zip(axes.flat, panel_keys):
        pdf = df[(df["component"] == component) & (df["phase"] == phase)].copy()
        metric, ylabel = _panel_metric(component, phase)
        with_model = pdf["model_name"].nunique() > 1
        pdf["shape"] = pdf.apply(_shape_label, axis=1)
        pdf["shape_key"] = pdf.apply(_shape_key, axis=1)
        pdf["hue"] = pdf.apply(lambda r: _hue_label(r, with_model), axis=1)
        pdf["hue_key"] = pdf.apply(_hue_key, axis=1)
        shapes = [s for _k, s in sorted({(k, s) for k, s in zip(pdf["shape_key"], pdf["shape"])})]
        hues = [h for _k, h in sorted({(k, h) for k, h in zip(pdf["hue_key"], pdf["hue"])})]
        ok = pdf[pdf["success"] & (pdf[metric] > 0) & np.isfinite(pdf[metric].astype(float))]
        ymax = float(ok[metric].max()) if not ok.empty else 1.0
        successes += len(ok)

        xs = np.arange(len(shapes), dtype=float)
        width = 0.8 / max(1, len(hues))
        for j, hue in enumerate(hues):
            color = SERIES_COLORS[j % len(SERIES_COLORS)]
            vals, blanks = [], []
            for shape in shapes:
                cell_ok = ok[(ok["hue"] == hue) & (ok["shape"] == shape)]
                cell_any = pdf[(pdf["hue"] == hue) & (pdf["shape"] == shape)]
                vals.append(float(cell_ok[metric].median()) if len(cell_ok) else np.nan)
                blanks.append(_BLANK_FRACTION * ymax if (len(cell_any) and not len(cell_ok)) else np.nan)
            bars = ax.bar(xs + j * width, vals, width, label=hue, color=color, edgecolor="black", linewidth=0.5)
            ax.bar(xs + j * width, blanks, width, fill=False, hatch="///", edgecolor="#888888", linewidth=0.5)
            fmt = "%.0f" if metric == "tflops" else "%.2f"
            ax.bar_label(bars, labels=["" if np.isnan(v) else fmt % v for v in vals], fontsize=BAR_LABEL_FONT_SIZE, rotation=90, padding=1)

        ax.set_xticks(xs + width * (len(hues) - 1) / 2)
        ax.set_xticklabels(shapes, fontsize=7)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("Shape (per request)" if phase == "generation" else "Sequence Length (s_q x s_kv)", fontsize=LABEL_FONT_SIZE)
        ax.set_title(f"{component} / {phase}", fontsize=TITLE_FONT_SIZE)
        # Headroom above the tallest bar for the legend and the rotated bar labels.
        ax.set_ylim(0, ymax * 1.6 if ymax > 0 else 1.0)
        ax.legend(title="Variant", fontsize=LEGEND_FONT_SIZE, loc="upper left", ncol=2 if len(hues) > 6 else 1)  # kv_dtype [paged] [mode] [det]
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    for ax in list(axes.flat)[len(panel_keys) :]:
        ax.axis("off")

    if successes == 0:
        logger.warning(f"No successful extended-axis rows for '{config.name}'; the extended chart shows coverage only")
    plt.tight_layout()
    output_path = output_dir / f"{config.name}_extended.webp"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pil_kwargs={"lossless": True, "quality": 100, "method": 6, "exact": True})
    plt.close()
    logger.info(f"Chart saved to {output_path}")
    return output_path
