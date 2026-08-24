# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
from pathlib import Path
import sys
import unittest

E2E_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2E_DIR))

from _factorial import (  # noqa: E402
    compare_results,
    config_fingerprint,
    factorial_main_effects,
    paired_stats,
    render_markdown,
    shapley_savings,
    validate_design,
    williams_orders,
)

RUN_MATRIX_PATH = E2E_DIR / "Qwen3.8" / "run_matrix.py"
RUN_MATRIX_SPEC = importlib.util.spec_from_file_location("qwen38_run_matrix_cpu_test", RUN_MATRIX_PATH)
if RUN_MATRIX_SPEC is None or RUN_MATRIX_SPEC.loader is None:
    raise RuntimeError(f"cannot load {RUN_MATRIX_PATH}")
RUN_MATRIX = importlib.util.module_from_spec(RUN_MATRIX_SPEC)
sys.modules[RUN_MATRIX_SPEC.name] = RUN_MATRIX
RUN_MATRIX_SPEC.loader.exec_module(RUN_MATRIX)


class FactorialStatisticsTest(unittest.TestCase):
    def setUp(self):
        savings = {1: 2.0, 2: 5.0, 4: 10.0}
        self.batch_times = {
            mask: [baseline - sum(value for bit, value in savings.items() if mask & bit) for baseline in (100.0, 110.0, 120.0)] for mask in range(8)
        }

    def _metadata(self, *, mode="formal", fingerprint="comparable", scale=1.0):
        effects = factorial_main_effects(self.batch_times)
        shapley = shapley_savings(self.batch_times)
        variants = []
        summary = {}
        for mask in range(8):
            bits = f"{mask:03b}"
            variants.append(
                {
                    "bits": bits,
                    "name": bits,
                    "gdn": bool(mask & 4),
                    "mlp": bool(mask & 2),
                    "attn": bool(mask & 1),
                }
            )
            paired = paired_stats(self.batch_times[mask], self.batch_times[0])
            summary[bits] = {
                "p50_ms": sorted(self.batch_times[mask])[1] * scale,
                "paired_ratio_p50": paired["paired_ratio_p50"],
                "wins_vs_baseline": paired["wins"],
                "batches": paired["batches"],
            }
        all_comparison = paired_stats(self.batch_times[7], self.batch_times[0])
        return {
            "completed_utc": "2026-08-20T12:00:00Z",
            "config": {
                "mode": mode,
                "comparability_fingerprint": {"sha256": fingerprint, "inputs": {}},
                "build_fingerprint": {"sha256": "build123", "inputs": {}},
                "device": "NVIDIA B200",
                "device_id": "cuda:2",
                "sm_count": 148,
                "python": "3.10",
                "torch": "test",
                "torch_cuda": "13.0",
                "cuda_driver": 13020,
                "cuda_runtime": 13000,
                "cudnn_frontend": "test",
                "cudnn_backend": 92300,
                "fla": "test",
                "preset": "qwen3.8-27b",
                "resolved_shape": {"hidden": 5120},
                "attn_layers": [3],
                "bs": 4,
                "seq": 2048,
                "warmup": 3,
                "rounds": 8,
                "repeats": 1,
                "mode_overrides": {},
                "numerical_recipe": {
                    "id": "conservative-bf16-v1",
                    "parameter_dtype": "bfloat16",
                    "activation_dtype": "bfloat16",
                    "scope": "forward_backward_no_optimizer",
                    "anchor": {
                        "project": "NVIDIA-NeMo/Megatron-Bridge",
                        "commit": "2e77041c194d106beb7462e226d7ca06b33ea63f",
                        "path": "src/megatron/bridge/training/mixed_precision.py",
                        "symbol": "bf16_mixed",
                    },
                    "alignment": "upstream_anchored_subset",
                },
                "variants": variants,
                "provenance": {
                    "git": {"commit": "deadbeef", "dirty": False},
                    "sources": {"runner": {"path": "benchmark/e2e/Qwen3.8/run_matrix.py", "sha256": "face"}},
                },
            },
            "summary": summary,
            "comparisons": {"all_vs_baseline": all_comparison},
            "main_effects": effects,
            "shapley": shapley,
            "route": {
                "native_gdn_calls": 100,
                "expected_native_gdn_calls": 100,
                "frost_calls": 100,
                "expected_frost_calls": 100,
                "pointwise_calls": 0,
                "full_attention_calls": {"torch": 10, "cudnn": 10},
            },
        }

    def test_williams_design_balances_positions_and_carryover(self):
        orders = williams_orders()
        self.assertEqual(orders[0], (0, 1, 7, 2, 6, 3, 5, 4))
        validate_design(orders)
        with self.assertRaises(ValueError):
            validate_design(orders[:-1])
        with self.assertRaises(ValueError):
            williams_orders(7)

    def test_paired_statistics_are_batch_aligned(self):
        result = paired_stats([8.0, 18.0], [10.0, 20.0])
        self.assertAlmostEqual(result["paired_ratio_p50"], 0.85)
        self.assertAlmostEqual(result["paired_delta_p50_ms"], -2.0)
        self.assertEqual(result["wins"], 2)
        with self.assertRaises(ValueError):
            paired_stats([1.0], [1.0, 2.0])

    def test_main_effects_and_shapley_recover_additive_savings(self):
        effects = factorial_main_effects(self.batch_times)
        self.assertAlmostEqual(effects["gdn"]["conditional_delta_ms"]["p50"], -10.0)
        self.assertAlmostEqual(effects["mlp"]["conditional_delta_ms"]["p50"], -5.0)
        self.assertAlmostEqual(effects["attn"]["conditional_delta_ms"]["p50"], -2.0)

        shapley = shapley_savings(self.batch_times)
        self.assertAlmostEqual(shapley["saving_ms"]["gdn"]["p50"], 10.0)
        self.assertAlmostEqual(shapley["saving_ms"]["mlp"]["p50"], 5.0)
        self.assertAlmostEqual(shapley["saving_ms"]["attn"]["p50"], 2.0)
        self.assertAlmostEqual(shapley["combined_saving_ms"]["p50"], 17.0)

    def test_config_fingerprint_is_canonical_and_sensitive(self):
        first = {"shape": {"hidden": 5120, "layers": 4}, "mode": "formal"}
        reordered = {"mode": "formal", "shape": {"layers": 4, "hidden": 5120}}
        self.assertEqual(config_fingerprint(first), config_fingerprint(reordered))
        changed = copy.deepcopy(first)
        changed["shape"]["layers"] = 8
        self.assertNotEqual(config_fingerprint(first), config_fingerprint(changed))
        recipe_changed = copy.deepcopy(first)
        recipe_changed["numerical_recipe"] = {"id": "future-low-precision-v1"}
        self.assertNotEqual(config_fingerprint(first), config_fingerprint(recipe_changed))
        with self.assertRaises(ValueError):
            config_fingerprint({"invalid": float("nan")})

    def test_markdown_contains_result_configuration_and_provenance(self):
        metadata = self._metadata()
        report = render_markdown(metadata, raw_json_link="result.json", raw_json_sha256="cafe")
        self.assertIn("Comparability fingerprint: `comparable`", report)
        self.assertIn("Build/provenance fingerprint: `build123`", report)
        self.assertIn("## Factorial attribution", report)
        self.assertIn("Numerical recipe: `conservative-bf16-v1`", report)
        self.assertIn("NVIDIA-NeMo/Megatron-Bridge@2e77041c", report)
        self.assertIn("benchmark/e2e/Qwen3.8/run_matrix.py", report)
        self.assertIn("provenance only", report)

    def test_smoke_markdown_is_validation_only_without_speedup_headline(self):
        metadata = self._metadata(mode="smoke")
        metadata["config"]["bs"] = 1
        metadata["config"]["seq"] = 128
        report = render_markdown(metadata, raw_json_link="result.json", raw_json_sha256="cafe")
        self.assertIn("Validation only", report)
        self.assertIn("M=bs*seq=128", report)
        self.assertNotIn("The all-cuDNN arm", report)
        self.assertNotIn("## Factorial attribution", report)

    def test_cross_run_comparison_is_nonpaired_and_requires_fingerprint_match(self):
        previous = self._metadata(scale=1.0)
        current = self._metadata(scale=0.9)
        current["config"]["build_fingerprint"]["sha256"] = "new-build"
        current["config"]["provenance"]["sources"]["runner"]["sha256"] = "new-source"
        previous["comparisons"]["all_vs_baseline"]["paired_ratio_p50"] = 0.90
        current["comparisons"]["all_vs_baseline"]["paired_ratio_p50"] = 0.85

        comparison = compare_results(current, previous)
        self.assertFalse(comparison["paired_across_runs"])
        self.assertAlmostEqual(comparison["arms"]["000"]["p50_change_percent"], -10.0)
        self.assertAlmostEqual(comparison["headline"]["change_percentage_points"], -5.0)
        current["comparison"] = comparison
        report = render_markdown(current, raw_json_link="result.json", raw_json_sha256="cafe")
        self.assertIn("Not paired across runs", report)
        self.assertIn("within-run paired `111/000`", report)

        incompatible = self._metadata(fingerprint="different")
        with self.assertRaisesRegex(ValueError, "different comparability fingerprints"):
            compare_results(current, incompatible)

    def test_cross_run_comparison_rejects_nonfinite_p50(self):
        previous = self._metadata()
        current = self._metadata()
        current["summary"]["111"]["p50_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            compare_results(current, previous)

    def test_cross_run_comparison_labels_missing_or_nonnumeric_p50(self):
        for invalid in (None, "not-a-number", 10**10000):
            with self.subTest(invalid=invalid):
                previous = self._metadata()
                current = self._metadata()
                if invalid is None:
                    del previous["summary"]["111"]["p50_ms"]
                else:
                    previous["summary"]["111"]["p50_ms"] = invalid
                with self.assertRaisesRegex(ValueError, "111 treatment summaries must contain numeric p50_ms values"):
                    compare_results(current, previous)

    def test_torch_baseline_route_contract_is_strict(self):
        valid = {
            "can_use_flash": True,
            "can_use_cudnn": False,
            "flash_sdp_enabled": True,
            "grad_fn_chain": ["TransposeBackward0", "ScaledDotProductFlashAttentionBackward0"],
        }
        RUN_MATRIX._validate_torch_baseline_route(valid)
        for name, invalid_value in (
            ("can_use_flash", False),
            ("can_use_cudnn", True),
            ("flash_sdp_enabled", False),
            ("grad_fn_chain", ["SomeOtherBackward0"]),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "route contract failed"):
                invalid = copy.deepcopy(valid)
                invalid[name] = invalid_value
                RUN_MATRIX._validate_torch_baseline_route(invalid)


if __name__ == "__main__":
    unittest.main()
