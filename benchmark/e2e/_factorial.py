# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure statistics and reporting helpers for balanced factorial benchmarks.

This module intentionally has no torch, CUDA, or model dependencies.  Keeping
the design validation and analysis here makes the experiment math inexpensive
to test on CPU and reusable by other end-to-end model benchmarks.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import statistics

AXIS_MASKS = {"gdn": 4, "mlp": 2, "attn": 1}


def percentile(values, q):
    """Return a linearly interpolated quantile for a non-empty sample."""
    values = [float(value) for value in values]
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"percentile q must be in [0, 1], got {q}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("percentile values must be finite")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def distribution(values):
    """Summarize a non-empty numeric sample without external dependencies."""
    values = [float(value) for value in values]
    if not values:
        raise ValueError("distribution requires at least one value")
    return {
        "p10": percentile(values, 0.1),
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "mean": statistics.mean(values),
        "count": len(values),
    }


def paired_stats(numerator, denominator):
    """Compute paired ratios/deltas for aligned treatment batches."""
    numerator = [float(value) for value in numerator]
    denominator = [float(value) for value in denominator]
    if not numerator or len(numerator) != len(denominator):
        raise ValueError("paired samples must be non-empty and have equal lengths")
    if any(value <= 0.0 or not math.isfinite(value) for value in numerator + denominator):
        raise ValueError("paired timing samples must be finite and positive")

    ratios = [new / old for new, old in zip(numerator, denominator)]
    deltas = [new - old for new, old in zip(numerator, denominator)]
    return {
        "paired_ratio_p10": percentile(ratios, 0.1),
        "paired_ratio_p50": percentile(ratios, 0.5),
        "paired_ratio_p90": percentile(ratios, 0.9),
        "paired_delta_p10_ms": percentile(deltas, 0.1),
        "paired_delta_p50_ms": percentile(deltas, 0.5),
        "paired_delta_p90_ms": percentile(deltas, 0.9),
        "wins": sum(new < old for new, old in zip(numerator, denominator)),
        "batches": len(numerator),
    }


def williams_orders(treatments=8):
    """Return a standard even-treatment Williams carryover design."""
    if treatments < 2 or treatments % 2:
        raise ValueError("Williams design requires an even treatment count >= 2")

    base = [0]
    for offset in range(1, treatments // 2 + 1):
        base.append(offset)
        if offset != treatments // 2:
            base.append(treatments - offset)
    orders = tuple(tuple((value + shift) % treatments for value in base) for shift in range(treatments))
    validate_design(orders)
    return orders


def validate_design(orders):
    """Assert positional and first-order carryover balance."""
    orders = tuple(tuple(order) for order in orders)
    treatments = len(orders)
    if treatments < 2 or any(len(order) != treatments for order in orders):
        raise ValueError(f"expected a square Williams design, got {orders}")
    expected = set(range(treatments))
    if any(set(order) != expected for order in orders):
        raise ValueError(f"each Williams row must contain treatments {sorted(expected)} exactly once")

    positions = Counter((position, treatment) for order in orders for position, treatment in enumerate(order))
    adjacency = Counter(pair for order in orders for pair in zip(order, order[1:]))
    if set(positions.values()) != {1} or len(positions) != treatments * treatments:
        raise ValueError(f"unbalanced treatment positions: {positions}")
    expected_adjacencies = treatments * (treatments - 1)
    if set(adjacency.values()) != {1} or len(adjacency) != expected_adjacencies:
        raise ValueError(f"unbalanced ordered carryover: {adjacency}")


def _validated_batch_times(batch_times, axis_masks):
    masks = list(axis_masks.values())
    expected_masks = [1 << bit for bit in range(len(masks))]
    if sorted(masks) != expected_masks:
        raise ValueError(f"axis masks must be unique one-bit masks {expected_masks}, got {masks}")

    expected_keys = set(range(1 << len(masks)))
    if set(batch_times) != expected_keys:
        raise ValueError(f"batch_times keys must be {sorted(expected_keys)}, got {sorted(batch_times)}")

    normalized = {}
    batch_count = None
    for mask in sorted(expected_keys):
        values = [float(value) for value in batch_times[mask]]
        if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
            raise ValueError(f"treatment {mask} timings must be finite and positive")
        if batch_count is None:
            batch_count = len(values)
        elif len(values) != batch_count:
            raise ValueError("every treatment must have the same number of batches")
        normalized[mask] = values
    return normalized, batch_count


def factorial_main_effects(batch_times, axis_masks=AXIS_MASKS):
    """Average each axis's conditional effects within every paired batch.

    Deltas use accelerated-minus-incumbent milliseconds.  Ratios are averaged
    in log space so reciprocal speedups are treated symmetrically.
    """
    batch_times, batch_count = _validated_batch_times(batch_times, axis_masks)
    treatment_count = len(batch_times)
    results = {}
    for axis, mask in axis_masks.items():
        delta_samples = []
        ratio_samples = []
        for batch in range(batch_count):
            conditional_deltas = []
            conditional_logs = []
            for incumbent in range(treatment_count):
                if incumbent & mask:
                    continue
                accelerated = incumbent | mask
                old = batch_times[incumbent][batch]
                new = batch_times[accelerated][batch]
                conditional_deltas.append(new - old)
                conditional_logs.append(math.log(new / old))
            delta_samples.append(statistics.mean(conditional_deltas))
            ratio_samples.append(math.exp(statistics.mean(conditional_logs)))
        results[axis] = {
            "conditional_delta_ms": distribution(delta_samples),
            "conditional_geomean_ratio": distribution(ratio_samples),
        }
    return results


def shapley_savings(batch_times, axis_masks=AXIS_MASKS):
    """Allocate baseline-minus-all savings to every axis, including interactions."""
    batch_times, batch_count = _validated_batch_times(batch_times, axis_masks)
    axis_count = len(axis_masks)
    all_mask = (1 << axis_count) - 1
    samples = {axis: [] for axis in axis_masks}
    totals = []

    for batch in range(batch_count):
        times = {mask: values[batch] for mask, values in batch_times.items()}
        per_axis = {}
        for axis, axis_mask in axis_masks.items():
            contribution = 0.0
            for incumbent in range(all_mask + 1):
                if incumbent & axis_mask:
                    continue
                # ``int.bit_count`` is Python 3.10+, while the package still
                # supports Python 3.9.
                subset_size = bin(incumbent).count("1")
                weight = math.factorial(subset_size) * math.factorial(axis_count - subset_size - 1) / math.factorial(axis_count)
                contribution += weight * (times[incumbent] - times[incumbent | axis_mask])
            per_axis[axis] = contribution
            samples[axis].append(contribution)
        total = times[0] - times[all_mask]
        if not math.isclose(sum(per_axis.values()), total, rel_tol=1e-10, abs_tol=1e-8):
            raise AssertionError(f"Shapley additivity failed at batch {batch}: {per_axis}, total={total}")
        totals.append(total)

    return {
        "saving_ms": {axis: distribution(values) for axis, values in samples.items()},
        "combined_saving_ms": distribution(totals),
        "raw_saving_ms": samples,
    }


def config_fingerprint(config):
    """Hash a JSON-compatible resolved configuration using canonical encoding."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_results(current, previous):
    """Compare two independent formal runs with the same experiment contract.

    Each run's ``111/000`` headline remains internally paired by batch.  Samples
    are not paired *across* runs, so this function compares arm-level p50s and
    the two within-run paired headline estimates without claiming cross-run
    pairing or significance.
    """

    def comparability_fingerprint(metadata, label):
        try:
            value = metadata["config"]["comparability_fingerprint"]["sha256"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{label} artifact lacks a comparability fingerprint") from error
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} artifact has an invalid comparability fingerprint")
        return value

    current_fingerprint = comparability_fingerprint(current, "current")
    previous_fingerprint = comparability_fingerprint(previous, "previous")
    if current_fingerprint != previous_fingerprint:
        raise ValueError(
            "cannot compare artifacts with different comparability fingerprints: " f"current={current_fingerprint}, previous={previous_fingerprint}"
        )

    current_summary = current.get("summary", {})
    previous_summary = previous.get("summary", {})
    if set(current_summary) != set(previous_summary) or not current_summary:
        raise ValueError("current and previous artifacts must contain the same non-empty treatment summaries")

    arms = {}
    for bits in sorted(current_summary):
        try:
            current_p50 = float(current_summary[bits]["p50_ms"])
            previous_p50 = float(previous_summary[bits]["p50_ms"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{bits} treatment summaries must contain numeric p50_ms values") from error
        if not all(math.isfinite(value) and value > 0.0 for value in (current_p50, previous_p50)):
            raise ValueError(f"{bits} p50 values must be finite and positive")
        arms[bits] = {
            "previous_p50_ms": previous_p50,
            "current_p50_ms": current_p50,
            "p50_change_ms": current_p50 - previous_p50,
            "p50_change_percent": (current_p50 / previous_p50 - 1.0) * 100.0,
            "p50_ratio_current_over_previous": current_p50 / previous_p50,
            "paired_across_runs": False,
        }

    try:
        current_headline = float(current["comparisons"]["all_vs_baseline"]["paired_ratio_p50"])
        previous_headline = float(previous["comparisons"]["all_vs_baseline"]["paired_ratio_p50"])
    except (KeyError, TypeError) as error:
        raise ValueError("artifacts lack the within-run paired all_vs_baseline headline") from error
    if not all(math.isfinite(value) and value > 0.0 for value in (current_headline, previous_headline)):
        raise ValueError("within-run paired headline ratios must be finite and positive")

    return {
        "comparability_fingerprint_sha256": current_fingerprint,
        "paired_across_runs": False,
        "note": (
            "Runs are independent: arm p50 changes and the change between the two within-run paired "
            "111/000 estimates are cross-run comparisons, not paired samples."
        ),
        "arms": arms,
        "headline": {
            "metric": "within_run_paired_111_over_000_ratio_p50",
            "previous": previous_headline,
            "current": current_headline,
            "change": current_headline - previous_headline,
            "change_percentage_points": (current_headline - previous_headline) * 100.0,
            "relative_change_percent": (current_headline / previous_headline - 1.0) * 100.0,
            "paired_across_runs": False,
        },
    }


def _format_path(path):
    return str(path).replace("|", "\\|")


def render_markdown(metadata, *, raw_json_link, raw_json_sha256):
    """Render a compact, self-contained Markdown report from a result payload."""
    config = metadata["config"]
    summary = metadata["summary"]
    effects = metadata["main_effects"]
    shapley = metadata["shapley"]
    route = metadata["route"]
    variants = {variant["bits"]: variant for variant in config["variants"]}
    all_vs_baseline = metadata["comparisons"]["all_vs_baseline"]
    ratio = all_vs_baseline["paired_ratio_p50"]
    is_smoke = config["mode"] == "smoke"

    lines = [
        "# Qwen3.8 2^3 factorial benchmark",
        "",
        f"Generated: `{metadata['completed_utc']}`  ",
        f"Mode: `{config['mode']}`  ",
        f"Comparability fingerprint: `{config['comparability_fingerprint']['sha256']}`  ",
        f"Build/provenance fingerprint: `{config['build_fingerprint']['sha256']}`  ",
        f"Raw JSON: [`{raw_json_link}`]({_format_path(raw_json_link)}) (`sha256:{raw_json_sha256}`)",
        "",
    ]
    if is_smoke:
        lines.extend(
            [
                "## Smoke validation",
                "",
                "**Validation only. Do not use smoke timings for performance trends, headlines, or model-speedup claims.** "
                f"Smoke uses M=bs*seq={config['bs'] * config['seq']} instead of formal M=8192, so fixed launch/dispatch overheads and small-M effects dominate.",
                "",
                "The route, correctness, and artifact plumbing gates passed. The p50 values below are diagnostics only.",
                "",
                "| bits (G/M/A) | GDN | MLP | full attention | diagnostic p50 step |",
                "|---|---|---|---|---:|",
            ]
        )
        for bits in sorted(summary):
            result = summary[bits]
            variant = variants[bits]
            lines.append(
                f"| `{bits}` | {'cuDNN' if variant['gdn'] else 'stock FLA'} | "
                f"{'cuDNN' if variant['mlp'] else 'stock FLA'} | "
                f"{'cuDNN backend' if variant['attn'] else 'Torch'} | {result['p50_ms']:.3f} ms |"
            )
    else:
        lines.extend(
            [
                "## Result",
                "",
                f"The all-cuDNN arm is `{ratio:.5f}x` the paired baseline elapsed time "
                f"({(1.0 - ratio) * 100.0:.2f}% lower, `{1.0 / ratio:.3f}x` speedup; "
                f"{all_vs_baseline['wins']}/{all_vs_baseline['batches']} paired wins).",
                "",
                "| bits (G/M/A) | GDN | MLP | full attention | p50 step | paired ratio vs 000 | wins vs 000 |",
                "|---|---|---|---|---:|---:|---:|",
            ]
        )
        for bits in sorted(summary):
            result = summary[bits]
            variant = variants[bits]
            wins = "--" if bits == "000" else f"{result['wins_vs_baseline']}/{result['batches']}"
            lines.append(
                f"| `{bits}` | {'cuDNN' if variant['gdn'] else 'stock FLA'} | "
                f"{'cuDNN' if variant['mlp'] else 'stock FLA'} | "
                f"{'cuDNN backend' if variant['attn'] else 'Torch'} | "
                f"{result['p50_ms']:.3f} ms | {result['paired_ratio_p50']:.5f} | {wins} |"
            )
        lines.extend(
            [
                "",
                "## Factorial attribution",
                "",
                "Conditional ratios average each axis over its four paired contexts in log space. "
                "Shapley savings allocate baseline-minus-all elapsed time, including interactions; "
                "they are not module-time shares.",
                "",
                "| axis | conditional ratio (p50) | conditional speedup | conditional delta (p50) | Shapley saving (p50) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for axis in AXIS_MASKS:
            axis_ratio = effects[axis]["conditional_geomean_ratio"]["p50"]
            axis_delta = effects[axis]["conditional_delta_ms"]["p50"]
            axis_shapley = shapley["saving_ms"][axis]["p50"]
            lines.append(f"| {axis} | {axis_ratio:.5f} | {1.0 / axis_ratio:.3f}x | {axis_delta:+.3f} ms | {axis_shapley:.3f} ms |")

    comparison = metadata.get("comparison")
    if comparison is not None:
        headline = comparison["headline"]
        lines.extend(
            [
                "",
                "## Cross-run comparison",
                "",
                f"**Not paired across runs.** {comparison['note']}",
            ]
        )
        if "previous_artifact" in comparison:
            previous_artifact = comparison["previous_artifact"]
            lines.append(f"Previous artifact: `{_format_path(previous_artifact['path'])}` (`sha256:{previous_artifact['sha256']}`).")
        lines.extend(
            [
                "",
                f"The within-run paired `111/000` ratio changed from `{headline['previous']:.5f}` to "
                f"`{headline['current']:.5f}` ({headline['change_percentage_points']:+.3f} percentage points).",
                "",
                "| bits | previous p50 | current p50 | non-paired change | change |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for bits, arm in sorted(comparison["arms"].items()):
            lines.append(
                f"| `{bits}` | {arm['previous_p50_ms']:.3f} ms | {arm['current_p50_ms']:.3f} ms | "
                f"{arm['p50_change_ms']:+.3f} ms | {arm['p50_change_percent']:+.2f}% |"
            )

    mode_overrides = config.get("mode_overrides", {})
    override_text = json.dumps(mode_overrides, sort_keys=True) if mode_overrides else "none"
    recipe_anchor = config["numerical_recipe"].get("anchor")
    if recipe_anchor is None:
        recipe_anchor_text = "none (local policy)"
    else:
        recipe_anchor_text = f"{recipe_anchor['project']}@{recipe_anchor['commit']}:" f"{recipe_anchor['path']}::{recipe_anchor['symbol']}"
    lines.extend(
        [
            "",
            "## Configuration and gates",
            "",
            f"- Device: `{config['device']}` ({config['sm_count']} SMs, `{config['device_id']}`)",
            f"- Software: Python `{config['python']}`, PyTorch `{config['torch']}` / CUDA `{config['torch_cuda']}`, "
            f"CUDA driver/runtime `{config['cuda_driver']}/{config['cuda_runtime']}`, cuDNN FE `{config['cudnn_frontend']}`, "
            f"cuDNN backend `{config['cudnn_backend']}`, FLA `{config['fla']}`",
            f"- Model: `{config['preset']}`, shape `{json.dumps(config['resolved_shape'], sort_keys=True)}`, attention layers `{config['attn_layers']}`",
            f"- Numerical recipe: `{config['numerical_recipe']['id']}`; parameters/activations "
            f"`{config['numerical_recipe']['parameter_dtype']}`/`{config['numerical_recipe']['activation_dtype']}`; "
            f"scope `{config['numerical_recipe']['scope']}`; alignment `{config['numerical_recipe']['alignment']}`",
            f"- Numerical anchor: `{recipe_anchor_text}`",
            f"- Input/timing: batch `{config['bs']}`, sequence `{config['seq']}`, warmup `{config['warmup']}`, "
            f"batches `{config['rounds']}`, repeats `{config['repeats']}`",
            f"- Mode overrides: `{override_text}`",
            "- Correctness: every arm passed explicit finite-value checks and the recorded aggregate BF16 loss/gradient gate.",
            f"- Native GDN calls: `{route['native_gdn_calls']}/{route['expected_native_gdn_calls']}`; "
            f"FROST dSwiGLU calls: `{route['frost_calls']}/{route['expected_frost_calls']}`; pointwise fallback: `{route['pointwise_calls']}`.",
            f"- Full-attention calls: `{json.dumps(route['full_attention_calls'], sort_keys=True)}`; Torch baseline contract passed.",
        ]
    )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"Repository commit: `{config['provenance']['git']['commit']}`; dirty: `{config['provenance']['git']['dirty']}`.",
            "",
            "| source | path | sha256 |",
            "|---|---|---|",
        ]
    )
    for name, source in sorted(config["provenance"]["sources"].items()):
        lines.append(f"| {name} | `{_format_path(source['path'])}` | `{source['sha256']}` |")
    lines.extend(
        [
            "",
            "The source hashes above are provenance only; the runner does not pin a private checkout or require a particular GDN source hash.",
            "",
        ]
    )
    return "\n".join(lines)
