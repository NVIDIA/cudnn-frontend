# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

E2E_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = E2E_DIR / "Qwen-Image" / "run_model.py"
SPEC = importlib.util.spec_from_file_location("qwen_image_model_cpu_test", MODEL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODEL_PATH}")
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)

RUNNER_PATH = E2E_DIR / "Qwen-Image" / "run_bf16.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("qwen_image_bf16_cpu_test", RUNNER_PATH)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise RuntimeError(f"cannot load {RUNNER_PATH}")
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)


class QwenImageSpecTest(unittest.TestCase):
    def test_formal_shape_preserves_published_kernel_dimensions(self):
        shape = MODEL.resolve_shape("formal")
        self.assertEqual(shape["image_tokens"], 4096)
        self.assertEqual(shape["text_tokens"], 512)
        self.assertEqual(shape["joint_tokens"], 4608)
        self.assertEqual((shape["hidden"], shape["heads"], shape["head_dim"], shape["ffn"]), (3072, 24, 128, 12288))
        self.assertEqual(shape["layers"], 4)
        self.assertEqual(MODEL.PUBLISHED_SHAPE["full_layers"], 60)

    def test_smoke_changes_only_repeated_and_token_work(self):
        shape = MODEL.resolve_shape("smoke")
        self.assertEqual((shape["layers"], shape["image_tokens"], shape["text_tokens"]), (1, 256, 64))
        self.assertEqual((shape["hidden"], shape["heads"], shape["head_dim"], shape["ffn"]), (3072, 24, 128, 12288))

    def test_invalid_image_token_grid_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "perfect square"):
            MODEL.resolve_shape("smoke", image_tokens=255)

    def test_bf16_leaf_has_immutable_upstream_anchors(self):
        self.assertEqual(len(MODEL.OFFICIAL_MODEL["revision"]), 40)
        self.assertEqual(len(MODEL.DIFFUSERS_ANCHOR["commit"]), 40)
        self.assertEqual(len(MODEL.DIFFUSERS_ANCHOR["source_sha256"]), 64)
        self.assertEqual(set(MODEL.DIFFUSERS_ANCHOR["supporting_sources"]), {"attention", "activations"})
        for source in MODEL.DIFFUSERS_ANCHOR["supporting_sources"].values():
            self.assertEqual(len(source["source_sha256"]), 64)
        self.assertEqual(MODEL.NUMERICAL_RECIPE["id"], "qwen-image-conservative-bf16-v1")
        self.assertEqual(MODEL.NUMERICAL_RECIPE["scope"], "inference_transformer_forward")

    def test_bf16_leaf_is_complete_mlp_attention_factorial(self):
        self.assertEqual(
            RUNNER.VARIANTS,
            (
                ("00", "torch", "torch_flash"),
                ("01", "torch", "cudnn"),
                ("10", "cudnn", "torch_flash"),
                ("11", "cudnn", "cudnn"),
            ),
        )
        self.assertEqual(RUNNER.AXIS_MASKS, {"mlp": 2, "attn": 1})
        self.assertEqual(len(RUNNER.williams_orders(4)), 4)

    def test_bf16_protocol_requires_complete_four_arm_cycles(self):
        args = SimpleNamespace(mode="formal", warmup=None, rounds=None, repeats=None)
        self.assertEqual(RUNNER._resolve_protocol(args), {"warmup": 3, "rounds": 40, "repeats": 3})
        args.rounds = 6
        with self.assertRaisesRegex(ValueError, "multiple of 4"):
            RUNNER._resolve_protocol(args)

    def test_bf16_report_names_off_on_scope_and_both_axes(self):
        batches = {
            "00": [10.0, 10.2],
            "01": [8.0, 8.1],
            "10": [9.8, 10.0],
            "11": [7.8, 7.9],
        }
        summary = {}
        for bits, values in batches.items():
            paired = RUNNER.paired_stats(values, batches["00"])
            summary[bits] = {
                "p10_ms": min(values),
                "p50_ms": sum(values) / len(values),
                "p90_ms": max(values),
                **paired,
            }
        integer_batches = {int(bits, 2): values for bits, values in batches.items()}
        comparisons = {
            "all_vs_baseline": RUNNER.paired_stats(batches["11"], batches["00"]),
            "attention_with_torch_mlp": RUNNER.paired_stats(batches["01"], batches["00"]),
            "attention_with_cudnn_mlp": RUNNER.paired_stats(batches["11"], batches["10"]),
            "mlp_with_flash_attention": RUNNER.paired_stats(batches["10"], batches["00"]),
            "mlp_with_cudnn_attention": RUNNER.paired_stats(batches["11"], batches["01"]),
        }
        metadata = {
            "completed_utc": "2026-08-21T00:00:00Z",
            "config": {
                "mode": "formal",
                "comparability_fingerprint": {"sha256": "comparable"},
                "build_fingerprint": {"sha256": "build"},
                "shape": MODEL.resolve_shape("formal"),
                "numerical_recipe": dict(MODEL.NUMERICAL_RECIPE),
                "workload": "single_conditional_transformer_forward_no_checkpoint",
                "model_anchor": dict(MODEL.OFFICIAL_MODEL),
                "diffusers_anchor": dict(MODEL.DIFFUSERS_ANCHOR),
            },
            "summary": summary,
            "comparisons": comparisons,
            "main_effects": RUNNER.factorial_main_effects(integer_batches, axis_masks=RUNNER.AXIS_MASKS),
            "shapley": RUNNER.shapley_savings(integer_batches, axis_masks=RUNNER.AXIS_MASKS),
            "correctness": {
                "model_output_rel_l2": {bits: 0.0 for bits in batches},
                "padding_adapter": {"rel_l2": 0.0},
            },
            "route": {
                "torch_probe": {"natural_choice_name": "CUDNN_ATTENTION", "forced_choice_name": "FLASH_ATTENTION"},
                "attention_calls": {"torch_flash": 8, "cudnn": 8},
                "mlp_calls": {"torch": 16, "cudnn": 16},
            },
            "provenance": {
                "sources": {
                    "cudnn_gelu_mlp": {"path": "_gelu_mlp.py", "sha256": "face"},
                    "diffusers_attention": {"path": "attention.py", "sha256": "feed"},
                    "diffusers_activations": {"path": "activations.py", "sha256": "beef"},
                }
            },
        }
        report = RUNNER._render_markdown(metadata, "result.json", "cafe")
        self.assertIn("bits (M/A)", report)
        self.assertIn("both-cuDNN-treatment `11`", report)
        self.assertIn("turns cuDNN off only for the two measured axes", report)
        self.assertIn("natural `CUDNN_ATTENTION`", report)
        self.assertIn("cudnn_gelu_mlp", report)
        self.assertIn("diffusers_attention", report)
        self.assertIn("diffusers_activations", report)


if __name__ == "__main__":
    unittest.main()
